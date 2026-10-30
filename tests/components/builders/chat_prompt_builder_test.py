import unittest
from pathlib import Path
from unittest.mock import patch, mock_open, call, Mock

from ravelin.components.builders.chat_prompt_builder import ChatPromptBuilder
from ravelin.dataclasses import ChatMessage, ByteStream
from ravelin.dataclasses.chat_message import TextContent


class TestChatPromptBuilder(unittest.TestCase):
    def setUp(self):
        self.system_template = "You are a helpful bot named {{ bot_name }}."
        self.user_template = "Hello, I am {{ name }} from {{ city }}."
        self.required_variables = ["name"]
        self.builder = ChatPromptBuilder(self.system_template, self.user_template, self.required_variables)

    def _assert_chat_messages(self, first, second):
        self.assertEqual(len(first), len(second))
        for i, message in enumerate(second):
            # Update comparison to handle TextContent list
            if isinstance(first[i].content, str):
                self.assertEqual([TextContent(text=first[i].content)], message.content)
            else:
                self.assertEqual(first[i].content, message.content)
            self.assertEqual(first[i].role, message.role)

    def test_init(self):
        self.assertListEqual(self.required_variables, self.builder.required_variables)

    def test_name(self):
        mock_pipeline = Mock()
        mock_pipeline.get_component_name.return_value = "my_prompt_builder"
        self.builder.__haystack_added_to_pipeline__ = mock_pipeline

        self.assertEqual("my_prompt_builder", self.builder._name)
        mock_pipeline.get_component_name.assert_called_once_with(self.builder)

    @patch(
        "ravelin.components.builders.chat_prompt_builder.get_local_asset_storage_path",
        return_value=Path("/path/to/assets"),
    )
    @patch("builtins.open", new_callable=mock_open)
    def test_to_dict(self, mock_file_open, _):
        result = self.builder.to_dict()

        print(mock_file_open.call_args_list)

        # Assert that the templates are written to the correct file paths
        mock_file_open.assert_any_call(
            Path(f"/path/to/assets/prompt_templates/{self.builder._name}.system_prompt.jinja2"),
            "w",
            encoding="utf-8",
        )
        mock_file_open.assert_any_call(
            Path(f"/path/to/assets/prompt_templates/{self.builder._name}.user_prompt.jinja2"),
            "w",
            encoding="utf-8",
        )
        mock_file_open().write.assert_has_calls(
            [
                call("You are a helpful bot named {{ bot_name }}."),
                call("Hello, I am {{ name }} from {{ city }}."),
            ]
        )

        self.assertEqual(
            "ravelin.components.builders.chat_prompt_builder.ChatPromptBuilder",
            result["type"],
        )
        self.assertEqual(
            f"prompt_templates/{self.builder._name}.system_prompt.jinja2", result["init_parameters"]["system_template"]
        )
        self.assertEqual(
            f"prompt_templates/{self.builder._name}.user_prompt.jinja2", result["init_parameters"]["user_template"]
        )
        self.assertListEqual(self.required_variables, result["init_parameters"]["required_variables"])

    @patch(
        "ravelin.components.builders.chat_prompt_builder.get_local_asset_storage_path",
        return_value=Path("/path/to/assets"),
    )
    @patch("builtins.open", new_callable=mock_open)
    def test_from_dict(self, mock_file_open, _):
        data = {
            "type": "ravelin.components.builders.chat_prompt_builder.ChatPromptBuilder",
            "init_parameters": {
                "system_template": "prompt_templates/chat_prompt_builder_123.system_prompt.jinja2",
                "user_template": "prompt_templates/chat_prompt_builder_123.user_prompt.jinja2",
                "required_variables": ["name"],
            },
        }

        mock_file_open().read.side_effect = [
            "You are a helpful bot named {{ bot_name }}.",
            "Hello, I am {{ name }} from {{ city }}.",
        ]

        builder = ChatPromptBuilder.from_dict(data)

        # Assert that the templates are read from the correct file paths
        mock_file_open.assert_any_call(
            Path("/path/to/assets/prompt_templates/chat_prompt_builder_123.system_prompt.jinja2"),
            "r",
            encoding="utf-8",
        )
        mock_file_open.assert_any_call(
            Path("/path/to/assets/prompt_templates/chat_prompt_builder_123.user_prompt.jinja2"),
            "r",
            encoding="utf-8",
        )

        self.assertEqual("You are a helpful bot named {{ bot_name }}.", builder._system_template_string)
        self.assertEqual("Hello, I am {{ name }} from {{ city }}.", builder._user_template_string)
        self.assertListEqual(["name"], builder.required_variables)

    def test_from_dict_with_missing_templates(self):
        data = {
            "type": "ravelin.components.builders.chat_prompt_builder.ChatPromptBuilder",
            "init_parameters": {
                "system_template": None,
                "user_template": None,
                "required_variables": [],
            },
        }

        builder = ChatPromptBuilder.from_dict(data)

        self.assertIsNone(builder._system_template_string)
        self.assertIsNone(builder._user_template_string)
        self.assertListEqual([], builder.required_variables)

    def test_run(self):
        result = self.builder.run(name="John", city="New York", bot_name="Marry")
        self.assertIn("messages", result)
        self._assert_chat_messages(
            [
                ChatMessage.from_system("You are a helpful bot named Marry."),
                ChatMessage.from_user("Hello, I am John from New York."),
            ],
            result["messages"],
        )

    def test_run_with_history(self):
        history = [
            ChatMessage.from_user("Greetings!"),
            ChatMessage.from_assistant("Where are you from?"),
        ]
        result = self.builder.run(name="John", city="New York", bot_name="Marry", _history=history)
        self.assertIn("messages", result)
        self._assert_chat_messages(
            [
                ChatMessage.from_system("You are a helpful bot named Marry."),
                ChatMessage.from_user("Greetings!"),
                ChatMessage.from_assistant("Where are you from?"),
                ChatMessage.from_user("Hello, I am John from New York."),
            ],
            result["messages"],
        )

    def test_run_with_media(self):
        mock_media = Mock(autospec=ByteStream)
        result = self.builder.run(name="John", city="New York", bot_name="Marry", _media=[mock_media])
        self._assert_chat_messages(
            [
                ChatMessage.from_system("You are a helpful bot named Marry."),
                ChatMessage.from_user("Hello, I am John from New York.", media=[mock_media]),
            ],
            result["messages"],
        )

    def test_run_without_input(self):
        builder = ChatPromptBuilder(system_template="This is a template without input")
        result = builder.run()
        self.assertIn("messages", result)
        self._assert_chat_messages(
            [ChatMessage.from_system("This is a template without input")],
            result["messages"],
        )

    @patch("ravelin.components.builders.chat_prompt_builder.logging.warning")
    def test_run_without_template(self, mock_warning):
        builder = ChatPromptBuilder()
        history = [
            ChatMessage.from_user("Greetings!"),
            ChatMessage.from_assistant("Where are you from?"),
        ]
        result = builder.run(_history=history)
        self.assertIn("messages", result)
        self._assert_chat_messages(history, result["messages"])
        mock_warning.assert_called_once_with("No prompt templates provided to ChatPromptBuilder.")

    def test_run_with_missing_optional_input(self):
        result = self.builder.run(name="John")
        self.assertIn("messages", result)
        self._assert_chat_messages(
            [
                ChatMessage.from_system("You are a helpful bot named ."),
                ChatMessage.from_user("Hello, I am John from ."),
            ],
            result["messages"],
        )

    def test_run_with_missing_required_input(self):
        with self.assertRaisesRegex(ValueError, "Missing required input variables in ChatPromptBuilder: name$"):
            self.builder.run(city="New York")

        with self.assertRaisesRegex(ValueError, "Missing required input variables in ChatPromptBuilder: name$"):
            self.builder.run()


if __name__ == "__main__":
    unittest.main()
