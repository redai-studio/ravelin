import unittest
from unittest.mock import AsyncMock, Mock, patch

from haystack.tools import Tool

from ravelin.assets.model_services import ModelService, ChatModelService
from ravelin.components.generators.chat_generator import ChatGenerator
from ravelin.core.restate.context import _restate_ctx
from ravelin.dataclasses import ChatMessage
from ravelin.dataclasses.asset_reference import AssetReference


class TestChatGenerator(unittest.TestCase):
    def setUp(self):
        self.model_service = Mock(spec=ChatModelService)
        self.model_service.reference = AssetReference(name="test-model-service", source="local")
        self.model_service.infer.return_value = [ChatMessage.from_assistant("Hello!")]
        self.tools = [Mock(spec=Tool)]
        self.chat_generator = ChatGenerator(
            model_service=self.model_service,
            model="gpt-3.5-turbo",
            stream=False,
            tools=self.tools,
            timeout=5.0,
            generation_kwargs={"foo": "bar"},
        )
        self.messages = [ChatMessage.from_system("Hello")]

    def test_init(self):
        self.assertEqual(self.model_service, self.chat_generator.model_service)
        self.assertEqual("gpt-3.5-turbo", self.chat_generator.model)
        self.assertEqual(5.0, self.chat_generator.timeout)
        self.assertEqual({"foo": "bar"}, self.chat_generator.generation_kwargs)
        self.assertEqual(3, self.chat_generator.max_retries)

    def test_to_dict(self):
        generator_dict = self.chat_generator.to_dict()

        self.assertIn("init_parameters", generator_dict)
        self.assertIn("type", generator_dict)
        self.assertEqual("ravelin.components.generators.chat_generator.ChatGenerator", generator_dict["type"])
        self.assertEqual(self.model_service.reference.to_dict(), generator_dict["init_parameters"]["model_service"])
        self.assertEqual("gpt-3.5-turbo", generator_dict["init_parameters"]["model"])
        self.assertFalse(generator_dict["init_parameters"]["stream"])
        self.assertEqual(5.0, generator_dict["init_parameters"]["timeout"])
        self.assertEqual({"foo": "bar"}, generator_dict["init_parameters"]["generation_kwargs"])
        self.assertEqual(3, generator_dict["init_parameters"]["max_retries"])

    @patch("ravelin.components.generators.chat_generator.ModelServiceFactory.load_from_reference")
    def test_from_dict_happy_path(self, mock_load_service_from_reference):
        data = {
            "type": "ravelin.components.generators.chat_generator.ChatGenerator",
            "init_parameters": {
                "model_service": {
                    "name": "test-model-service",
                    "source": "local",
                },
                "model": "gpt-3.5-turbo",
                "stream": True,
                "timeout": 5.0,
                "generation_kwargs": {"foo": "bar"},
                "max_retries": 5,
            },
        }
        mock_load_service_from_reference.return_value = self.model_service
        generator = ChatGenerator.from_dict(data)

        mock_load_service_from_reference.assert_called_once_with(
            AssetReference.from_dict(data["init_parameters"]["model_service"])
        )
        self.assertIsInstance(generator, ChatGenerator)
        self.assertEqual(self.model_service, generator.model_service)
        self.assertEqual("gpt-3.5-turbo", generator.model)
        self.assertTrue(generator.stream)
        self.assertEqual(5.0, generator.timeout)
        self.assertEqual({"foo": "bar"}, generator.generation_kwargs)
        self.assertEqual(5, generator.max_retries)

    @patch("ravelin.components.generators.chat_generator.ModelServiceFactory.load_from_reference")
    def test_from_dict_with_invalid_model_service_type(self, mock_load_service_from_reference):
        data = {
            "type": "ravelin.components.generators.chat_generator.ChatGenerator",
            "init_parameters": {
                "model_service": {
                    "name": "test-model-service",
                    "source": "local",
                },
                "model": "gpt-3.5-turbo",
                "stream": False,
                "timeout": 5.0,
                "generation_kwargs": {"foo": "bar"},
                "max_retries": 3,
            },
        }

        mock_load_service_from_reference.return_value = Mock(spec=ModelService)
        with self.assertRaisesRegex(
            TypeError, "The model service must be an instance of ChatModelService, got <class 'unittest.mock.Mock'>"
        ):
            ChatGenerator.from_dict(data)

    def test_run_with_successful_response(self):
        result = self.chat_generator.run(self.messages)
        self.model_service.infer.assert_called_once_with(
            self.messages,
            model="gpt-3.5-turbo",
            timeout=5.0,
            stream=False,
            tools=self.tools,
            inference_kwargs={"foo": "bar"},
            max_retries=3,
        )
        self.assertIn("replies", result)
        self.assertEqual(1, len(result["replies"]))
        self.assertEqual("Hello!", result["replies"][0].text)

    def test_run_with_unsuccessful_response(self):
        self.model_service.infer.side_effect = ValueError("foobar")
        with self.assertRaisesRegex(ValueError, "foobar"):
            self.chat_generator.run(self.messages)

    def test_run_with_no_messages(self):
        result = self.chat_generator.run([])
        self.model_service.infer.assert_called_once_with(
            [],
            model="gpt-3.5-turbo",
            timeout=5.0,
            stream=False,
            tools=self.tools,
            inference_kwargs={"foo": "bar"},
            max_retries=3,
        )
        self.assertIn("replies", result)

    def test_run_with_custom_max_retries(self):
        """Test that custom max_retries value is passed to the model service."""
        custom_generator = ChatGenerator(
            model_service=self.model_service,
            model="gpt-3.5-turbo",
            timeout=5.0,
            max_retries=10,
        )
        result = custom_generator.run(self.messages)
        self.model_service.infer.assert_called_once_with(
            self.messages,
            model="gpt-3.5-turbo",
            timeout=5.0,
            stream=False,
            tools=[],
            inference_kwargs={},
            max_retries=10,
        )
        self.assertIn("replies", result)

    def test_run_passes_configured_max_retries_inside_restate_context(self):
        custom_generator = ChatGenerator(
            model_service=self.model_service,
            model="gpt-3.5-turbo",
            timeout=5.0,
            max_retries=10,
        )

        token = _restate_ctx.set(Mock())
        try:
            result = custom_generator.run(self.messages)
        finally:
            _restate_ctx.reset(token)

        self.model_service.infer.assert_called_once_with(
            self.messages,
            model="gpt-3.5-turbo",
            timeout=5.0,
            stream=False,
            tools=[],
            inference_kwargs={},
            max_retries=10,
        )
        self.assertIn("replies", result)


class TestChatGeneratorAsync(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.model_service = AsyncMock(spec=ChatModelService)
        self.model_service.reference = AssetReference(name="test-model-service", source="local")
        self.model_service.ainfer.return_value = [ChatMessage.from_assistant("Hello!")]
        self.tools = [Mock(spec=Tool)]
        self.chat_generator = ChatGenerator(
            model_service=self.model_service,
            model="gpt-3.5-turbo",
            stream=False,
            tools=self.tools,
            timeout=5.0,
            generation_kwargs={"foo": "bar"},
        )
        self.messages = [ChatMessage.from_system("Hello")]

    async def test_run_with_successful_response(self):
        result = await self.chat_generator.run_async(self.messages)
        self.model_service.ainfer.assert_called_once_with(
            self.messages,
            model="gpt-3.5-turbo",
            timeout=5.0,
            stream=False,
            tools=self.tools,
            inference_kwargs={"foo": "bar"},
            max_retries=3,
        )
        self.assertIn("replies", result)
        self.assertEqual(1, len(result["replies"]))
        self.assertEqual("Hello!", result["replies"][0].text)

    async def test_run_async_passes_configured_max_retries_inside_restate_context(self):
        custom_generator = ChatGenerator(
            model_service=self.model_service,
            model="gpt-3.5-turbo",
            timeout=5.0,
            max_retries=10,
        )

        token = _restate_ctx.set(Mock())
        try:
            result = await custom_generator.run_async(self.messages)
        finally:
            _restate_ctx.reset(token)

        self.model_service.ainfer.assert_called_once_with(
            self.messages,
            model="gpt-3.5-turbo",
            timeout=5.0,
            stream=False,
            tools=[],
            inference_kwargs={},
            max_retries=10,
        )
        self.assertIn("replies", result)


if __name__ == "__main__":
    unittest.main()
