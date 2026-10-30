import unittest
from unittest.mock import MagicMock, patch, AsyncMock

from haystack.tools import Tool
from haystack.components.agents.state import State
from haystack.dataclasses.chat_message import TextContent

from ravelin.components.tools.tool_invoker import ToolInvoker
from ravelin.dataclasses.chat_message import ChatMessage, ChatRole, ToolCall


class TestToolInvoker(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Create mock tools
        self.mock_tool = Tool(
            name="mock_tool",
            description="A mock tool",
            function=lambda x: f"Result: {x}",
            parameters={"type": "object", "properties": {}},
        )
        self.tools = [self.mock_tool]

        # Create sample messages
        self.text_message = ChatMessage.from_user("Hello")
        self.tool_call = ToolCall(tool_name="mock_tool", arguments={"param": "value"}, id="123")
        self.tool_message = ChatMessage.from_assistant(text="Using tool", tool_calls=[self.tool_call])
        self.messages = [self.text_message, self.tool_message]

        # Create a sample state
        self.state = State(schema={})

    @patch("ravelin.components.tools.tool_invoker.HaystackToolInvoker")
    def test_init(self, mock_haystack_invoker):
        invoker = ToolInvoker(tools=self.tools, raise_on_failure=True)
        self.assertEqual(invoker.tools, self.tools)
        self.assertTrue(invoker.raise_on_failure)
        self.assertFalse(invoker.convert_result_to_json_string)
        self.assertFalse(invoker.combine_tool_results_to_single_message)
        mock_haystack_invoker.assert_called_once()

        mock_haystack_invoker.reset_mock()
        invoker = ToolInvoker(tools=self.tools, raise_on_failure=False, convert_result_to_json_string=True)
        self.assertFalse(invoker.raise_on_failure)
        self.assertTrue(invoker.convert_result_to_json_string)
        mock_haystack_invoker.assert_called_once()

    @patch("ravelin.components.tools.tool_invoker.HaystackToolInvoker")
    def test_run(self, mock_haystack_invoker):
        # Mock the haystack invoker's run method
        mock_instance = mock_haystack_invoker.return_value
        haystack_result_message = MagicMock()
        haystack_result_message.role = ChatRole.TOOL
        # Mock the content property
        haystack_result_message.content = [TextContent(text="Tool result")]

        mock_instance.run.return_value = {"tool_messages": [haystack_result_message], "state": self.state}

        # Create our invoker
        invoker = ToolInvoker(tools=self.tools)

        # Run the method
        result = invoker.run(messages=self.messages, state=self.state)

        # Verify the Haystack invoker was called with the right parameters
        mock_instance.run.assert_called_once_with(messages=self.messages, state=self.state, tools=None)

        # Verify the result was converted correctly
        self.assertIn("tool_messages", result)
        self.assertIn("state", result)
        self.assertEqual(1, len(result["tool_messages"]))
        self.assertIsInstance(result["tool_messages"][0], ChatMessage)
        self.assertEqual(result["state"], self.state)

    @patch("ravelin.components.tools.tool_invoker.HaystackToolInvoker")
    async def test_run_async(self, mock_haystack_invoker):
        # Mock the haystack invoker's run_async method
        mock_instance = mock_haystack_invoker.return_value
        haystack_result_message = MagicMock()
        haystack_result_message.role = ChatRole.TOOL
        # Mock the content property
        haystack_result_message.content = [TextContent(text="Tool result")]

        mock_instance.run_async = AsyncMock()
        mock_instance.run_async.return_value = {"tool_messages": [haystack_result_message], "state": self.state}

        # Create our invoker
        invoker = ToolInvoker(tools=self.tools)

        # Run the async method
        result = await invoker.run_async(messages=self.messages, state=self.state)

        # Verify the Haystack invoker was called with the right parameters
        mock_instance.run_async.assert_called_once_with(messages=self.messages, state=self.state, tools=None)

        # Verify the result was converted correctly
        self.assertIn("tool_messages", result)
        self.assertIn("state", result)
        self.assertEqual(1, len(result["tool_messages"]))
        self.assertIsInstance(result["tool_messages"][0], ChatMessage)
        self.assertEqual(result["state"], self.state)

    def test_to_dict(self):
        invoker = ToolInvoker(
            tools=self.tools,
            raise_on_failure=True,
            convert_result_to_json_string=True,
            combine_tool_results_to_single_message=True,
        )

        serialized = invoker.to_dict()

        self.assertIn("tools", serialized["init_parameters"])
        self.assertTrue(serialized["init_parameters"]["raise_on_failure"])
        self.assertTrue(serialized["init_parameters"]["convert_result_to_json_string"])
        self.assertTrue(serialized["init_parameters"]["combine_tool_results_to_single_message"])
        self.assertNotIn("streaming_callbacks", serialized["init_parameters"])

    @patch("ravelin.components.tools.tool_invoker.deserialize_tools_or_toolset_inplace")
    def test_from_dict(self, mock_deserialize_tools):
        data = {
            "init_parameters": {
                "tools": [{"name": "mock_tool"}],
                "raise_on_failure": True,
                "convert_result_to_json_string": False,
            }
        }

        with patch("ravelin.components.tools.tool_invoker.default_from_dict") as mock_default_from_dict:
            mock_default_from_dict.return_value = ToolInvoker(tools=[self.mock_tool])
            ToolInvoker.from_dict(data)
            mock_deserialize_tools.assert_called_once()
            mock_default_from_dict.assert_called_once()

    @patch("ravelin.components.tools.tool_invoker.HaystackToolInvoker")
    def test_combine_single_tool_result(self, mock_haystack_invoker):
        """Test that combining a single tool result works correctly."""
        # Mock the haystack invoker's run method
        mock_instance = mock_haystack_invoker.return_value

        # Create a tool result message
        tool_call = ToolCall(tool_name="mock_tool", arguments={"param": "value"}, id="123")
        tool_result = ChatMessage.from_tool(tool_result="Result: test", origin=tool_call, error=False)

        # Mock return value
        mock_instance.run.return_value = {"tool_messages": [tool_result], "state": self.state}

        # Create invoker with combine flag
        invoker = ToolInvoker(tools=self.tools, combine_tool_results_to_single_message=True)

        # Run the method
        result = invoker.run(messages=self.messages, state=self.state)

        # Verify single message is returned
        self.assertEqual(1, len(result["tool_messages"]))
        combined_message = result["tool_messages"][0]
        self.assertEqual(ChatRole.TOOL, combined_message.role)
        self.assertEqual(1, len(combined_message.tool_call_results))

    @patch("ravelin.components.tools.tool_invoker.HaystackToolInvoker")
    def test_combine_multiple_tool_results(self, mock_haystack_invoker):
        """Test that combining multiple tool results merges them into a single message."""
        # Mock the haystack invoker's run method
        mock_instance = mock_haystack_invoker.return_value

        # Create multiple tool result messages
        tool_call1 = ToolCall(tool_name="tool1", arguments={"arg": "value1"}, id="123")
        tool_result1 = ChatMessage.from_tool(tool_result="Result 1", origin=tool_call1, error=False)

        tool_call2 = ToolCall(tool_name="tool2", arguments={"arg": "value2"}, id="456")
        tool_result2 = ChatMessage.from_tool(tool_result="Result 2", origin=tool_call2, error=False)

        tool_call3 = ToolCall(tool_name="tool3", arguments={"arg": "value3"}, id="789")
        tool_result3 = ChatMessage.from_tool(tool_result="Result 3", origin=tool_call3, error=False)

        # Mock return value with multiple messages
        mock_instance.run.return_value = {
            "tool_messages": [tool_result1, tool_result2, tool_result3],
            "state": self.state,
        }

        # Create invoker with combine flag
        invoker = ToolInvoker(tools=self.tools, combine_tool_results_to_single_message=True)

        # Run the method
        result = invoker.run(messages=self.messages, state=self.state)

        # Verify single message is returned
        self.assertEqual(1, len(result["tool_messages"]))
        combined_message = result["tool_messages"][0]
        self.assertEqual(ChatRole.TOOL, combined_message.role)
        # Verify all three tool call results are in the combined message
        self.assertEqual(3, len(combined_message.tool_call_results))
        self.assertEqual("Result 1", combined_message.tool_call_results[0].result)
        self.assertEqual("Result 2", combined_message.tool_call_results[1].result)
        self.assertEqual("Result 3", combined_message.tool_call_results[2].result)

    @patch("ravelin.components.tools.tool_invoker.HaystackToolInvoker")
    def test_no_combine_keeps_separate(self, mock_haystack_invoker):
        """Test that with combine flag False, tool results remain separate."""
        # Mock the haystack invoker's run method
        mock_instance = mock_haystack_invoker.return_value

        # Create multiple tool result messages
        tool_call1 = ToolCall(tool_name="tool1", arguments={"arg": "value1"}, id="123")
        tool_result1 = ChatMessage.from_tool(tool_result="Result 1", origin=tool_call1, error=False)

        tool_call2 = ToolCall(tool_name="tool2", arguments={"arg": "value2"}, id="456")
        tool_result2 = ChatMessage.from_tool(tool_result="Result 2", origin=tool_call2, error=False)

        # Mock return value with multiple messages
        mock_instance.run.return_value = {"tool_messages": [tool_result1, tool_result2], "state": self.state}

        # Create invoker WITHOUT combine flag (default is False)
        invoker = ToolInvoker(tools=self.tools, combine_tool_results_to_single_message=False)

        # Run the method
        result = invoker.run(messages=self.messages, state=self.state)

        # Verify messages remain separate
        self.assertEqual(2, len(result["tool_messages"]))
        self.assertEqual("Result 1", result["tool_messages"][0].tool_call_result.result)
        self.assertEqual("Result 2", result["tool_messages"][1].tool_call_result.result)

    @patch("ravelin.components.tools.tool_invoker.HaystackToolInvoker")
    async def test_combine_multiple_tool_results_async(self, mock_haystack_invoker):
        """Test that combining multiple tool results works in async mode."""
        # Mock the haystack invoker's run_async method
        mock_instance = mock_haystack_invoker.return_value
        mock_instance.run_async = AsyncMock()

        # Create multiple tool result messages
        tool_call1 = ToolCall(tool_name="tool1", arguments={"arg": "value1"}, id="123")
        tool_result1 = ChatMessage.from_tool(tool_result="Result 1", origin=tool_call1, error=False)

        tool_call2 = ToolCall(tool_name="tool2", arguments={"arg": "value2"}, id="456")
        tool_result2 = ChatMessage.from_tool(tool_result="Result 2", origin=tool_call2, error=False)

        # Mock return value with multiple messages
        mock_instance.run_async.return_value = {"tool_messages": [tool_result1, tool_result2], "state": self.state}

        # Create invoker with combine flag
        invoker = ToolInvoker(tools=self.tools, combine_tool_results_to_single_message=True)

        # Run the async method
        result = await invoker.run_async(messages=self.messages, state=self.state)

        # Verify single message is returned
        self.assertEqual(1, len(result["tool_messages"]))
        combined_message = result["tool_messages"][0]
        self.assertEqual(ChatRole.TOOL, combined_message.role)
        self.assertEqual(2, len(combined_message.tool_call_results))


if __name__ == "__main__":
    unittest.main()
