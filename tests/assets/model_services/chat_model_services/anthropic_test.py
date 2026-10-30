import unittest
from base64 import b64encode
from unittest.mock import Mock, AsyncMock, patch, call

from httpx import Response
from haystack.tools import Tool
from haystack.dataclasses import ToolCallDelta
from haystack.utils import Secret

from ravelin.assets.model_services.chat_model_services.anthropic import AnthropicChatModelService
from ravelin.streaming import EventType
from ravelin.streaming.event import GENERATION_DELTA
from ravelin.dataclasses import ByteStream, ChatMessage, ChatRole, StreamingChunk
from ravelin.dataclasses.chat_message import TextContent, ToolCall
from ravelin.utils.streaming import ServerSentEvent


class TestAnthropicChatModelService(unittest.TestCase):
    def setUp(self):
        self.model_service = AnthropicChatModelService(
            name="test_anthropic_model_service",
            base_url="https://anthropic/url/",
            api_key=Secret.from_token("test_api_key"),
        )

    def test_initialization(self):
        self.assertEqual(self.model_service.name, "test_anthropic_model_service")
        self.assertDictEqual(
            {
                "base_url": "https://anthropic/url/",
                "organization": None,
                "inference_api_format": "anthropic-chat",
            },
            self.model_service.configs,
        )
        self.assertEqual("test_api_key", self.model_service.secrets["api_key"].resolve_value())

    @patch("ravelin.assets.model_services.chat_model_services.anthropic.logger")
    def test_format_chat_message_simple(self, mock_logger):
        message = ChatMessage.from_user("Hello")
        formatted_message = self.model_service.format_chat_message(message)
        self.assertDictEqual({"role": "user", "content": "Hello"}, formatted_message)
        mock_logger.warning.assert_not_called()

    def test_format_chat_message_with_cache_control(self):
        """A cache_control marker in meta forces block form and attaches to the last content block."""
        message = ChatMessage.from_user("Hello", meta={"cache_control": {"type": "ephemeral"}})
        formatted_message = self.model_service.format_chat_message(message)
        self.assertDictEqual(
            {
                "role": "user",
                "content": [{"type": "text", "text": "Hello", "cache_control": {"type": "ephemeral"}}],
            },
            formatted_message,
        )

    @patch("ravelin.assets.model_services.chat_model_services.anthropic.logger")
    def test_format_chat_message_with_media(self, mock_logger):
        message = ChatMessage.from_user(
            "Multimodal example",
            [ByteStream(b"data", mime_type="image/png"), ByteStream(b"data2", mime_type="application/pdf")],
        )
        formatted_message = self.model_service.format_chat_message(message)
        self.assertDictEqual(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": b64encode(b"data").decode("utf-8"),
                        },
                    },
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": b64encode(b"data2").decode("utf-8"),
                        },
                    },
                    {"type": "text", "text": "Multimodal example"},
                ],
            },
            formatted_message,
        )
        mock_logger.warning.assert_not_called()

    @patch("ravelin.assets.model_services.chat_model_services.anthropic.logger")
    def test_format_chat_message_tool_call(self, mock_logger):
        message = ChatMessage.from_assistant(
            text="Example tool call", tool_calls=[ToolCall(tool_name="test_tool", arguments={"foo": "bar"}, id="123")]
        )
        expected = {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "Example tool call",
                },
                {
                    "type": "tool_use",
                    "id": "123",
                    "name": "test_tool",
                    "input": {"foo": "bar"},
                },
            ],
        }
        self.assertEqual(expected, self.model_service.format_chat_message(message))
        mock_logger.warning.assert_not_called()

    def test_convert_message_tool_call_invalid(self):
        with self.assertRaisesRegex(
            ValueError, "`ToolCall` must have a non-null `id` attribute to be used with Anthropic."
        ):
            self.model_service.format_chat_message(
                ChatMessage.from_assistant(
                    text="Example tool call",
                    tool_calls=[ToolCall(tool_name="test_tool", arguments={"foo": "bar"})],
                )
            )

    @patch("ravelin.assets.model_services.chat_model_services.anthropic.logger")
    def test_format_chat_message_tool_call_result(self, mock_logger):
        message = ChatMessage.from_tool(
            "result", origin=ToolCall(tool_name="test_tool", arguments={"foo": "bar"}, id="123"), error=False
        )
        expected = {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "123",
                    "is_error": False,
                    "content": "result",
                },
            ],
        }
        self.assertEqual(expected, self.model_service.format_chat_message(message))
        mock_logger.warning.assert_not_called()

    def test_convert_message_tool_call_result_invalid(self):
        with self.assertRaisesRegex(
            ValueError, "`ToolCall` must have a non-null `id` attribute to be used with Anthropic."
        ):
            self.model_service.format_chat_message(
                ChatMessage.from_tool(
                    "result", origin=ToolCall(tool_name="test_tool", arguments={"foo": "bar"}), error=False
                )
            )

    @patch("ravelin.assets.model_services.chat_model_services.anthropic.logger")
    def test_format_chat_message_warnings(self, mock_logger):
        message = ChatMessage.from_system("You are a helpful bot.")
        _ = self.model_service.format_chat_message(message)
        mock_logger.warning.assert_called_once_with(
            "The role %s is not supported by the Anthropic API.", ChatRole.SYSTEM
        )

    @patch("ravelin.assets.model_services.chat_model_services.anthropic.AnthropicChatModelService._build_tool_call")
    def test_build_message(self, mock_build_tool_call):
        mock_build_tool_call.return_value = (
            ToolCall(tool_name="get_weather", arguments={"location": "San Francisco, CA"}, id="toolu_01A"),
        )

        content = [
            {
                "type": "thinking",
                "thinking": "To answer this question, I will: 1. Use the get_weather tool to get ...",
                "signature": "WaUjzkypQ2m...",
            },
            {
                "type": "redacted_thinking",
                "data": "EmwKAhgBEg...",
            },
            {
                "type": "text",
                "text": "I will need to call the get_weather tool to get the weather in San Francisco, CA.",
            },
            {
                "type": "tool_use",
                "id": "toolu_01A",
                "name": "get_weather",
                "input": {"location": "San Francisco, CA"},
            },
        ]
        chat_message = self.model_service._build_message(
            content,
            model="claude-3-5-sonnet-20241022",
            msg_id="msg_123",
            stop_reason="tool_use",
            stop_sequence=None,
            usage={"input_tokens": 25, "output_tokens": 120},
        )
        mock_build_tool_call.assert_called_once_with(content[-1])
        self.assertEqual(
            [
                TextContent(text="I will need to call the get_weather tool to get the weather in San Francisco, CA."),
                mock_build_tool_call.return_value,
            ],
            chat_message.content,
        )
        self.assertEqual(
            "To answer this question, I will: 1. Use the get_weather tool to get ...\a[REDACTED]",
            chat_message.reasoning_content,
        )
        self.assertEqual("assistant", chat_message.role)
        self.assertEqual("claude-3-5-sonnet-20241022", chat_message.meta["model"])
        self.assertEqual("msg_123", chat_message.meta["id"])
        self.assertEqual("tool_use", chat_message.meta["stop_reason"])
        self.assertListEqual(content[:2], chat_message.meta["thinking_blocks"])
        self.assertIsNone(chat_message.meta["stop_sequence"])
        self.assertEqual({"input_tokens": 25, "output_tokens": 120}, chat_message.meta["usage"])

    def test_build_tool_call(self):
        raw_tc = {
            "type": "tool_use",
            "id": "toolu_01A",
            "name": "get_weather",
            "input": {"location": "San Francisco, CA"},
        }
        tool_call = AnthropicChatModelService._build_tool_call(raw_tc)
        self.assertEqual(
            ToolCall(tool_name="get_weather", arguments={"location": "San Francisco, CA"}, id="toolu_01A"), tool_call
        )

    def test_build_tool_call_stream(self):
        raw_tc_stream = {
            "type": "tool_use",
            "id": "toolu_01A",
            "name": "get_weather",
            "raw_input": '{"location": "San Francisco, CA"}',
        }
        tool_call = AnthropicChatModelService._build_tool_call(raw_tc_stream, stream=True)
        self.assertEqual(
            ToolCall(tool_name="get_weather", arguments={"location": "San Francisco, CA"}, id="toolu_01A"), tool_call
        )

    @patch("ravelin.assets.model_services.chat_model_services.anthropic.logger")
    def test_build_tool_call_invalid(self, mock_logger):
        with self.assertRaisesRegex(ValueError, "Invalid tool call format: "):
            AnthropicChatModelService._build_tool_call({"type": "tool_use"})
        mock_logger.assert_not_called()

        tool_call = AnthropicChatModelService._build_tool_call(
            {"type": "tool_use", "id": "toolu_01A", "name": "get_weather", "raw_input": "{invalid"}, stream=True
        )
        self.assertIsNone(tool_call)
        mock_logger.warning.assert_called_once_with(
            (
                "Anthropic returned a malformed JSON string for tool call arguments as a part of its streaming"
                " response. This tool call will be skipped. Tool call ID: %s, Tool name: %s, Arguments: %s"
            ),
            "toolu_01A",
            "get_weather",
            "{invalid",
        )

    def test_build_chunk_from_data(self):
        test_cases = [
            (
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_123",
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": "claude-3-5-sonnet-20240620",
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 25, "output_tokens": 1},
                    },
                },
                StreamingChunk(
                    content="",
                    meta={
                        "id": "msg_123",
                        "model": "claude-3-5-sonnet-20240620",
                        "usage": {"input_tokens": 25, "output_tokens": 1},
                    },
                    start=True,
                    finish_reason=None,
                ),
                None,
            ),
            (
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                StreamingChunk(content="", meta={"index": 0}, index=0, start=True),
                None,
            ),
            (
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}},
                StreamingChunk(content="Hello", meta={"index": 0}, index=0),
                None,
            ),
            (
                {"type": "content_block_stop", "index": 0},
                StreamingChunk(content="", meta={"index": 0}, index=0),
                None,
            ),
            (
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": 15},
                },
                StreamingChunk(
                    content="",
                    meta={
                        "usage": {"output_tokens": 15},
                        "stop_sequence": None,
                    },
                    finish_reason="end_turn",
                ),
                None,
            ),
            (
                {"type": "ping"},
                None,
                None,
            ),
            (
                {"type": "message_stop"},
                None,
                None,
            ),
            (
                {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}},
                None,
                (
                    RuntimeError,
                    "Error received during model inference streaming: "
                    "{'type': 'overloaded_error', 'message': 'Overloaded'}",
                ),
            ),
            (
                {"foo": "bar"},
                None,
                (
                    ValueError,
                    "Invalid data received during model inference streaming: {'foo': 'bar'}",
                ),
            ),
            (
                {"type": "unknown"},
                None,
                None,
            ),
        ]
        for data, chunk, exception in test_cases:
            with self.subTest(data=data, chunk=chunk, exception=exception):
                if exception is not None:
                    with self.assertRaisesRegex(exception[0], exception[1]):
                        self.model_service._build_chunk_from_data(data)
                else:
                    result = self.model_service._build_chunk_from_data(data)
                    if chunk is not None and result is not None:
                        # Check fields individually to make debugging easier
                        self.assertEqual(chunk.content, result.content)
                        self.assertEqual(chunk.reasoning_content, result.reasoning_content)
                        self.assertEqual(chunk.meta, result.meta)
                        self.assertEqual(chunk.index, result.index)
                        self.assertEqual(chunk.start, result.start)
                        self.assertEqual(chunk.finish_reason, result.finish_reason)
                    else:
                        self.assertEqual(chunk, result)

    def test_build_chunk_from_data_with_tool_use(self):
        data = [
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "tool_use", "id": "toolu_01T", "name": "get_weather", "input": {}},
            },
            {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": ""}},
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": '{"location":'},
            },
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": ' "San Francisco, CA"}'},
            },
            {"type": "content_block_stop", "index": 1},
        ]
        streaming_chunks = [
            StreamingChunk(
                content="",
                tool_calls=[ToolCallDelta(index=1, id="toolu_01T", tool_name="get_weather", arguments="")],
                meta={"index": 1},
                index=1,
                start=True,
            ),
            StreamingChunk(
                content="",
                tool_calls=[ToolCallDelta(index=1, arguments="")],
                meta={"index": 1},
                index=1,
            ),
            StreamingChunk(
                content="",
                tool_calls=[ToolCallDelta(index=1, arguments='{"location":')],
                meta={"index": 1},
                index=1,
            ),
            StreamingChunk(
                content="",
                tool_calls=[ToolCallDelta(index=1, arguments=' "San Francisco, CA"}')],
                meta={"index": 1},
                index=1,
            ),
            StreamingChunk(content="", meta={"index": 1}, index=1),
        ]
        for i, (data_chunk, expected_chunk) in enumerate(zip(data, streaming_chunks)):
            with self.subTest(i=i, data_chunk=data_chunk, expected_chunk=expected_chunk):
                result = self.model_service._build_chunk_from_data(data_chunk)
                self.assertEqual(expected_chunk.content, result.content)
                self.assertEqual(expected_chunk.meta, result.meta)
                self.assertEqual(expected_chunk.index, result.index)
                self.assertEqual(expected_chunk.start, result.start)

                if expected_chunk.tool_calls:
                    self.assertIsNotNone(result.tool_calls)
                    self.assertEqual(len(expected_chunk.tool_calls), len(result.tool_calls))
                    for j, tc in enumerate(expected_chunk.tool_calls):
                        self.assertEqual(tc.index, result.tool_calls[j].index)
                        self.assertEqual(tc.id, result.tool_calls[j].id)
                        self.assertEqual(tc.tool_name, result.tool_calls[j].tool_name)
                        self.assertEqual(tc.arguments, result.tool_calls[j].arguments)

    def test_build_chunk_from_data_with_thinking(self):
        data = [
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "thinking", "thinking": ""},
            },
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "thinking_delta", "thinking": "Let me analyze this step by step..."},
            },
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "signature_delta", "signature": "WaUjzkypQ2m..."},
            },
            {"type": "content_block_stop", "index": 1},
            {
                "type": "content_block_start",
                "index": 2,
                "content_block": {"type": "redacted_thinking", "data": "EmwKAhgBEg..."},
            },
            {"type": "content_block_stop", "index": 2},
        ]
        streaming_chunks = [
            StreamingChunk(
                content="",
                reasoning_content="",
                tool_calls=None,
                meta={"index": 1},
                index=1,
                start=True,
            ),
            StreamingChunk(
                content="",
                reasoning_content="Let me analyze this step by step...",
                tool_calls=None,
                meta={"index": 1},
                index=1,
            ),
            StreamingChunk(
                content="",
                reasoning_content="",
                tool_calls=None,
                meta={"index": 1, "signature": "WaUjzkypQ2m..."},
                index=1,
            ),
            StreamingChunk(content="", tool_calls=None, meta={"index": 1}, index=1),
            StreamingChunk(
                content="",
                reasoning_content="\a[REDACTED]",
                tool_calls=None,
                meta={"index": 2, "redacted_thinking_block": {"type": "redacted_thinking", "data": "EmwKAhgBEg..."}},
                index=2,
                start=True,
            ),
            StreamingChunk(content="", tool_calls=None, meta={"index": 2}, index=2),
        ]
        for i, (data_chunk, expected_chunk) in enumerate(zip(data, streaming_chunks)):
            with self.subTest(i=i, data_chunk=data_chunk, expected_chunk=expected_chunk):
                result = self.model_service._build_chunk_from_data(data_chunk)
                self.assertEqual(expected_chunk.content, result.content)
                self.assertEqual(expected_chunk.reasoning_content, result.reasoning_content)
                self.assertEqual(expected_chunk.meta, result.meta)
                self.assertEqual(expected_chunk.index, result.index)
                self.assertEqual(expected_chunk.start, result.start)

    def test_connect_chunks(self):
        chunks = [
            StreamingChunk(
                content="", tool_calls=None, meta={"id": "msg_123", "usage": {"input_tokens": 25, "output_tokens": 1}}
            ),
            StreamingChunk(
                content="",
                reasoning_content="",
                tool_calls=None,
                meta={"index": 0, "thinking_blocks": [{"type": "thinking", "thinking": ""}]},
                index=0,
                start=True,
            ),
            StreamingChunk(
                content="",
                reasoning_content="Let me think about ",
                tool_calls=None,
                meta={"index": 0},
                index=0,
            ),
            StreamingChunk(
                content="",
                reasoning_content="this...",
                tool_calls=None,
                meta={"index": 0},
                index=0,
            ),
            StreamingChunk(
                content="",
                reasoning_content="",
                tool_calls=None,
                meta={"index": 0, "signature": "WaUjzkypQ2m..."},
                index=0,
            ),
            StreamingChunk(content="", tool_calls=None, meta={"index": 0}, index=0),
            StreamingChunk(
                content="",
                reasoning_content="\a[REDACTED]",
                tool_calls=None,
                meta={"index": 1, "redacted_thinking_block": {"type": "redacted_thinking", "data": "EmwKAhgBEg..."}},
                index=1,
                start=True,
            ),
            StreamingChunk(content="", tool_calls=None, meta={"index": 1}, index=1),
            StreamingChunk(content="Hello ", tool_calls=None, meta={"index": 2}, index=2, start=True),
            StreamingChunk(content="world", tool_calls=None, meta={"index": 2}, index=2),
            StreamingChunk(
                content="",
                tool_calls=[ToolCallDelta(index=3, id="toolu_01T", tool_name="get_weather", arguments="")],
                meta={"index": 3},
                index=3,
                start=True,
            ),
            StreamingChunk(
                content="",
                tool_calls=[ToolCallDelta(index=3, arguments='{"location":')],
                meta={"index": 3},
                index=3,
            ),
            StreamingChunk(
                content="",
                tool_calls=[ToolCallDelta(index=3, arguments=' "San Francisco, CA"}')],
                meta={"index": 3},
                index=3,
            ),
            StreamingChunk(content="", meta={"index": 3}, index=3),
            StreamingChunk(
                content="",
                meta={"stop_sequence": None, "usage": {"output_tokens": 15}},
                finish_reason="tool_use",
            ),
        ]
        connected_chunk = self.model_service._connect_chunks(chunks)
        self.assertEqual(
            [
                TextContent(text="Hello world"),
                ToolCall(tool_name="get_weather", id="toolu_01T", arguments={"location": "San Francisco, CA"}),
            ],
            connected_chunk.content,
        )
        self.assertEqual("Let me think about this...\a[REDACTED]", connected_chunk.reasoning_content)
        # Check expected keys and values individually for better debugging
        expected_meta = {
            "id": "msg_123",
            "stop_reason": "tool_use",
            "stop_sequence": None,
            "usage": {"input_tokens": 25, "output_tokens": 15},
            "thinking_blocks": [
                {"type": "thinking", "thinking": "Let me think about this...", "signature": "WaUjzkypQ2m..."},
                {"type": "redacted_thinking", "data": "EmwKAhgBEg..."},
            ],
        }

        # Check that all expected keys exist with correct values
        for key, value in expected_meta.items():
            self.assertEqual(value, connected_chunk.meta.get(key), f"Key '{key}' mismatch")

    @patch("ravelin.assets.model_services.chat_model_services.anthropic.logger")
    def test_check_stop_reason(self, mock_logger):
        message = ChatMessage.from_assistant(text="foo", meta={"stop_reason": "end_turn", "id": "foobar"})
        self.model_service._check_stop_reason(message)
        mock_logger.warning.assert_not_called()

        message = ChatMessage.from_assistant(text="Hello w", meta={"stop_reason": "max_tokens", "id": "msg_123"})
        self.model_service._check_stop_reason(message)
        mock_logger.warning.assert_called_once_with(
            "The assistant completion %s has been truncated before reaching a natural stopping point. "
            "Increase the max_tokens parameter or pick another model to allow for longer completions.",
            "msg_123",
        )

        message = ChatMessage.from_assistant(
            text="<END>", meta={"stop_reason": "stop_sequence", "stop_sequence": "<END>", "id": "msg_456"}
        )
        self.model_service._check_stop_reason(message)
        mock_logger.warning.assert_called_with(
            "The assistant completion %s stopped due to a custom stop sequence: '%s'",
            "msg_456",
            "<END>",
        )

        message = ChatMessage.from_assistant(text="Hello!", meta={"stop_reason": "tool_use", "id": "msg_789"})
        self.model_service._check_stop_reason(message)
        mock_logger.warning.assert_called_with(
            "The assistant completion %s stopped due to a tool call, but no valid tool call was found in "
            "the response. This could be due to invalid tool call formatting or model errors.",
            "msg_789",
        )

    @patch("httpx.Client")
    def test_build_request_with_system_prompt(self, mock_client):
        messages = [ChatMessage.from_system("You are a helpful bot."), ChatMessage.from_user(text="Hello")]
        self.model_service.build_request(mock_client, messages, model="claude-3-5-sonnet", stream=False)

        expected_json_body = {
            "model": "claude-3-5-sonnet",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 8192,
            "stream": False,
            "system": "You are a helpful bot.",
        }

        expected_headers = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
            "x-api-key": "test_api_key",
        }

        mock_client.build_request.assert_called_once_with(
            "POST",
            url="https://anthropic/url/v1/messages",
            json=expected_json_body,
            headers=expected_headers,
            timeout=None,
        )

    @patch("httpx.Client")
    def test_build_request_with_tools(self, mock_client):
        message = [ChatMessage.from_user("Hello")]
        mock_tool = Mock(autospec=Tool)
        self.model_service.build_request(
            mock_client,
            message,
            model="claude-3-5-sonnet",
            stream=True,
            tools=[mock_tool],
            inference_kwargs={"max_tokens": 4096},
        )

        expected_json_body = {
            "model": "claude-3-5-sonnet",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 4096,
            "stream": True,
            "tools": [
                {"name": mock_tool.name, "description": mock_tool.description, "input_schema": mock_tool.parameters}
            ],
        }
        expected_headers = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
            "x-api-key": "test_api_key",
        }
        mock_client.build_request.assert_called_with(
            "POST",
            url="https://anthropic/url/v1/messages",
            json=expected_json_body,
            headers=expected_headers,
            timeout=None,
        )

    @patch("httpx.Client")
    def test_build_request_with_extra_headers(self, mock_client):
        """Test that extra_headers are properly merged into the request headers."""
        messages = [ChatMessage.from_user("Hello")]
        extra_headers = {"X-Custom-Header": "custom_value", "Authorization": "Bearer custom_token"}

        self.model_service.build_request(
            mock_client,
            messages,
            model="claude-3-5-sonnet",
            stream=False,
            extra_headers=extra_headers,
        )

        expected_json_body = {
            "model": "claude-3-5-sonnet",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 8192,
            "stream": False,
        }

        expected_headers = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
            "x-api-key": "test_api_key",
            "X-Custom-Header": "custom_value",
            "Authorization": "Bearer custom_token",
        }

        mock_client.build_request.assert_called_once_with(
            "POST",
            url="https://anthropic/url/v1/messages",
            json=expected_json_body,
            headers=expected_headers,
            timeout=None,
        )

    @patch("httpx.Client")
    def test_build_request_with_cached_system_prompt(self, mock_client):
        """A cache_control marker on the system message emits a block list instead of a bare string."""
        messages = [
            ChatMessage.from_system("You are a helpful bot.", meta={"cache_control": {"type": "ephemeral"}}),
            ChatMessage.from_user(text="Hello"),
        ]
        self.model_service.build_request(mock_client, messages, model="claude-3-5-sonnet", stream=False)

        expected_json_body = {
            "model": "claude-3-5-sonnet",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 8192,
            "stream": False,
            "system": [{"type": "text", "text": "You are a helpful bot.", "cache_control": {"type": "ephemeral"}}],
        }
        expected_headers = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
            "x-api-key": "test_api_key",
        }
        mock_client.build_request.assert_called_once_with(
            "POST",
            url="https://anthropic/url/v1/messages",
            json=expected_json_body,
            headers=expected_headers,
            timeout=None,
        )

    @patch("httpx.Client")
    def test_build_request_automatic_caching_passthrough(self, mock_client):
        """Automatic (top-level) cache_control flows through inference_kwargs untouched into the body root."""
        messages = [ChatMessage.from_user("Hello")]
        self.model_service.build_request(
            mock_client,
            messages,
            model="claude-3-5-sonnet",
            stream=False,
            inference_kwargs={"cache_control": {"type": "ephemeral"}},
        )

        expected_json_body = {
            "model": "claude-3-5-sonnet",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 8192,
            "stream": False,
            "cache_control": {"type": "ephemeral"},
        }
        expected_headers = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
            "x-api-key": "test_api_key",
        }
        mock_client.build_request.assert_called_once_with(
            "POST",
            url="https://anthropic/url/v1/messages",
            json=expected_json_body,
            headers=expected_headers,
            timeout=None,
        )

    @patch("ravelin.assets.model_services.chat_model_services._base.get_component_context")
    @patch("ravelin.assets.model_services.chat_model_services.anthropic.AnthropicChatModelService._check_stop_reason")
    @patch("ravelin.assets.model_services.chat_model_services.anthropic.AnthropicChatModelService._connect_chunks")
    @patch("ravelin.assets.model_services.chat_model_services.anthropic.AnthropicChatModelService._build_chunk")
    @patch("ravelin.assets.model_services.chat_model_services.anthropic.ServerSentEventStream")
    def test_parse_response_stream(
        self, mock_sse_stream, mock_build_chunk, mock_connect_chunks, mock_check_stop_reason, mock_get_comp_ctx
    ):
        sse_events = [
            ServerSentEvent(
                event="content_block_start",
                data='{"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}',
            ),
            ServerSentEvent(
                event="content_block_delta",
                data='{"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi"}}',
            ),
            ServerSentEvent(
                event="content_block_delta",
                data='{"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "!"}}',
            ),
            ServerSentEvent(event="content_block_stop", data='{"type": "content_block_stop", "index": 0}'),
        ]
        streaming_chunks = [
            StreamingChunk(content="", tool_calls=None, meta={"index": 0}, index=0, start=True),
            StreamingChunk(content="Hi", tool_calls=None, meta={"index": 0}, index=0),
            StreamingChunk(content="!", tool_calls=None, meta={"index": 0}, index=0),
            StreamingChunk(content="", tool_calls=None, meta={"index": 0}, index=0),
        ]
        streaming_chunks_iter = iter(streaming_chunks)

        mock_comp_ctx = Mock()
        mock_get_comp_ctx.return_value = mock_comp_ctx

        # Set up mocks
        mock_sse_stream.return_value.__iter__.return_value = iter(sse_events)
        mock_build_chunk.side_effect = lambda sse: next(streaming_chunks_iter)

        chat_messages = self.model_service.parse_response(Response(200), stream=True)

        # Check mock calls
        self.assertListEqual(
            [call(sse) for sse in sse_events],
            mock_build_chunk.call_args_list,
        )
        # Lifecycle events live on infer, not parse_response — collector emits one DELTA per chunk.
        self.assertEqual(len(streaming_chunks), mock_comp_ctx.emit.call_count)
        mock_comp_ctx.emit.assert_any_call(
            EventType.TOKEN_STREAM, streaming_chunks[1], subtype=GENERATION_DELTA, generation_id=None
        )
        subtypes = [c.kwargs.get("subtype") for c in mock_comp_ctx.emit.call_args_list]
        self.assertEqual([GENERATION_DELTA] * len(streaming_chunks), subtypes)
        mock_connect_chunks.assert_called_once_with(streaming_chunks)
        mock_check_stop_reason.assert_called_once_with(chat_messages[0])
        self.assertEqual([mock_connect_chunks.return_value], chat_messages)

    def test_parse_response_no_stream(self):
        response_json = {
            "content": [{"text": "Hi! My name is Claude.", "type": "text"}],
            "id": "msg_013Zva2CMHLNnXjNJJKqJ2EF",
            "model": "claude-3-5-sonnet-20240620",
            "role": "assistant",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "type": "message",
            "usage": {"input_tokens": 2095, "output_tokens": 503},
        }
        chat_messages = self.model_service.parse_response(Response(200, json=response_json))
        self.assertEqual(1, len(chat_messages))
        self.assertEqual([TextContent(text="Hi! My name is Claude.")], chat_messages[0].content)
        self.assertEqual("assistant", chat_messages[0].role)
        self.assertEqual("claude-3-5-sonnet-20240620", chat_messages[0].meta["model"])
        self.assertEqual("msg_013Zva2CMHLNnXjNJJKqJ2EF", chat_messages[0].meta["id"])
        self.assertEqual("end_turn", chat_messages[0].meta["stop_reason"])
        self.assertEqual(None, chat_messages[0].meta["stop_sequence"])
        self.assertEqual({"input_tokens": 2095, "output_tokens": 503}, chat_messages[0].meta["usage"])


class TestAnthropicChatModelServiceAsync(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.model_service = AnthropicChatModelService(
            name="test_anthropic_model_service",
            base_url="https://anthropic/url/",
            api_key=Secret.from_token("test_api_key"),
        )

    @patch("ravelin.assets.model_services.chat_model_services._base.get_component_context")
    @patch("ravelin.assets.model_services.chat_model_services.anthropic.AnthropicChatModelService._check_stop_reason")
    @patch("ravelin.assets.model_services.chat_model_services.anthropic.AnthropicChatModelService._connect_chunks")
    @patch("ravelin.assets.model_services.chat_model_services.anthropic.AnthropicChatModelService._build_chunk")
    @patch("ravelin.assets.model_services.chat_model_services.anthropic.AsyncServerSentEventStream")
    async def test_aparse_response_stream(
        self, mock_sse_stream, mock_build_chunk, mock_connect_chunks, mock_check_stop_reason, mock_get_comp_ctx
    ):
        sse_events = [
            ServerSentEvent(
                event="content_block_start",
                data='{"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}',
            ),
            ServerSentEvent(
                event="content_block_delta",
                data='{"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi"}}',
            ),
            ServerSentEvent(
                event="content_block_delta",
                data='{"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "!"}}',
            ),
            ServerSentEvent(event="content_block_stop", data='{"type": "content_block_stop", "index": 0}'),
        ]
        streaming_chunks = [
            StreamingChunk(content="", tool_calls=None, meta={"index": 0}, index=0, start=True),
            StreamingChunk(content="Hi", tool_calls=None, meta={"index": 0}, index=0),
            StreamingChunk(content="!", tool_calls=None, meta={"index": 0}, index=0),
            StreamingChunk(content="", tool_calls=None, meta={"index": 0}, index=0),
        ]
        streaming_chunks_iter = iter(streaming_chunks)

        mock_comp_ctx = Mock()
        mock_comp_ctx.emit_async = AsyncMock()
        mock_get_comp_ctx.return_value = mock_comp_ctx

        # Set up mocks — use a real async generator for __aiter__
        async def _async_sse_iter():
            for e in sse_events:
                yield e

        mock_sse_stream.return_value.__aiter__ = lambda self: _async_sse_iter()
        mock_build_chunk.side_effect = lambda sse: next(streaming_chunks_iter)

        chat_messages = await self.model_service.aparse_response(Response(200), stream=True)

        # Check mock calls
        self.assertListEqual(
            [call(sse) for sse in sse_events],
            mock_build_chunk.call_args_list,
        )
        # Lifecycle events live on ainfer — collector emits one DELTA per chunk.
        self.assertEqual(len(streaming_chunks), mock_comp_ctx.emit_async.call_count)
        mock_comp_ctx.emit_async.assert_any_call(
            EventType.TOKEN_STREAM, streaming_chunks[1], subtype=GENERATION_DELTA, generation_id=None
        )
        subtypes = [c.kwargs.get("subtype") for c in mock_comp_ctx.emit_async.call_args_list]
        self.assertEqual([GENERATION_DELTA] * len(streaming_chunks), subtypes)
        mock_connect_chunks.assert_called_once_with(streaming_chunks)
        mock_check_stop_reason.assert_called_once_with(chat_messages[0])
        self.assertEqual([mock_connect_chunks.return_value], chat_messages)

    async def test_aparse_response_no_stream(self):
        response_json = {
            "content": [{"text": "Hi! My name is Claude.", "type": "text"}],
            "id": "msg_013Zva2CMHLNnXjNJJKqJ2EF",
            "model": "claude-3-5-sonnet-20240620",
            "role": "assistant",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "type": "message",
            "usage": {"input_tokens": 2095, "output_tokens": 503},
        }
        chat_messages = await self.model_service.aparse_response(Response(200, json=response_json))
        self.assertEqual(1, len(chat_messages))
        self.assertEqual([TextContent(text="Hi! My name is Claude.")], chat_messages[0].content)
        self.assertEqual("assistant", chat_messages[0].role)
        self.assertEqual("claude-3-5-sonnet-20240620", chat_messages[0].meta["model"])
        self.assertEqual("msg_013Zva2CMHLNnXjNJJKqJ2EF", chat_messages[0].meta["id"])
        self.assertEqual("end_turn", chat_messages[0].meta["stop_reason"])
        self.assertEqual(None, chat_messages[0].meta["stop_sequence"])
        self.assertEqual({"input_tokens": 2095, "output_tokens": 503}, chat_messages[0].meta["usage"])


if __name__ == "__main__":
    unittest.main()
