import json
import unittest
from base64 import b64encode
from unittest.mock import Mock, AsyncMock, patch, call, MagicMock

from haystack.dataclasses import ToolCallDelta
from haystack.tools import Tool
from haystack.utils import Secret
from httpx import Response

from ravelin.streaming import EventType
from ravelin.streaming.event import GENERATION_COMPLETE, GENERATION_DELTA
from ravelin.assets.model_services.chat_model_services.openai import (
    OpenAIChatModelService,
    _convert_message_to_openai_format,
)
from ravelin.dataclasses import ByteStream, ChatMessage, ChatRole, StreamingChunk
from ravelin.dataclasses.chat_message import TextContent, ToolCall, ToolCallResult
from ravelin.utils.streaming import ServerSentEvent


class TestConvertMessageToOpenAIFormat(unittest.TestCase):
    def test_convert_message_simple(self):
        message = ChatMessage.from_assistant(text="Hello")
        expected = {"role": "assistant", "content": "Hello"}
        self.assertEqual(expected, _convert_message_to_openai_format(message))

    def test_convert_message_with_media(self):
        message = ChatMessage.from_user(
            text="Hello", media=[ByteStream(data=b"test data", meta={"detail": "low"}, mime_type="image/png")]
        )
        expected = {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64," + b64encode(b"test data").decode("utf-8"),
                        "detail": "low",
                    },
                },
                {"type": "text", "text": "Hello"},
            ],
        }
        self.assertEqual(expected, _convert_message_to_openai_format(message))

    def test_convert_message_tool_call(self):
        message = ChatMessage.from_assistant(
            text="Example tool call", tool_calls=[ToolCall(tool_name="test_tool", arguments={"foo": "bar"}, id="123")]
        )
        expected = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Example tool call"},
            ],
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "test_tool",
                        "arguments": json.dumps({"foo": "bar"}),
                    },
                    "id": "123",
                },
            ],
        }
        self.assertEqual(expected, _convert_message_to_openai_format(message))

    def test_convert_message_tool_call_invalid(self):
        with self.assertRaisesRegex(ValueError, "Tool calls should only be included for assistant messages."):
            _convert_message_to_openai_format(
                ChatMessage(
                    _role=ChatRole.USER, _content=[ToolCall(tool_name="test_tool", arguments={"foo": "bar"}, id="123")]
                )
            )
        with self.assertRaisesRegex(
            ValueError, "`ToolCall` must have a non-null `id` attribute to be used with OpenAI."
        ):
            _convert_message_to_openai_format(
                ChatMessage.from_assistant(tool_calls=[ToolCall(tool_name="test_tool", arguments={"foo": "bar"})])
            )

    def test_convert_message_tool_call_result(self):
        message = ChatMessage.from_tool(
            "result", origin=ToolCall(tool_name="test_tool", arguments={"foo": "bar"}, id="123"), error=False
        )
        expected = {
            "role": "tool",
            "content": "result",
            "tool_call_id": "123",
        }

        self.assertEqual(expected, _convert_message_to_openai_format(message))

    def test_convert_message_tool_call_result_invalid(self):
        with self.assertRaisesRegex(ValueError, "Tool call results should only be included for tool messages."):
            _convert_message_to_openai_format(
                ChatMessage(
                    _role=ChatRole.USER,
                    _content=[
                        ToolCallResult(
                            result="result", origin=ToolCall(tool_name="tool", arguments={}, id="123"), error=False
                        )
                    ],
                )
            )
        with self.assertRaisesRegex(ValueError, "Tool call results should not be included with other content."):
            _convert_message_to_openai_format(
                ChatMessage(
                    _role=ChatRole.TOOL,
                    _content=[
                        TextContent(text="Hello"),
                        ToolCallResult(
                            result="result", origin=ToolCall(tool_name="tool", arguments={}, id="123"), error=False
                        ),
                    ],
                )
            )
        with self.assertRaisesRegex(
            ValueError, "`ToolCall` must have a non-null `id` attribute to be used with OpenAI."
        ):
            _convert_message_to_openai_format(
                ChatMessage.from_tool("result", origin=ToolCall(tool_name="test_tool", arguments={"foo": "bar"}))
            )


class TestOpenAIChatModelService(unittest.TestCase):
    def setUp(self):
        self.model_service = OpenAIChatModelService(name="test_openai_model_service", base_url="test_uri")
        self.tools = [MagicMock(autospec=Tool)]
        self.sse_contents = [
            ServerSentEvent(
                data=(
                    '{"choices":[{"delta":{"content":null,"reasoning_content":"reasoning","role":"assistant"},"finish_r'
                    'eason":null,"index":0,"logprobs":null}],"model":"deepseek-chat","object":"chat.completion.chunk"}'
                )
            ),
            ServerSentEvent(
                data=(
                    '{"choices":[{"delta":{"content":"Hello ","role":"assistant"},"finish_reason":null,'
                    '"index":0,"logprobs":null}],"model":"deepseek-chat","object":"chat.completion.chunk"}'
                )
            ),
            ServerSentEvent(
                data=(
                    '{"choices":[{"delta":{"content":null,"tool_calls":[{"index":0,"id":"123","type":"function",'
                    '"function":{"name":"test_tool","arguments":"{\\"foo"}}]},"finish_reason":null,'
                    '"index":0,"logprobs":null}],"model":"deepseek-chat","object":"chat.completion.chunk"}'
                )
            ),
            ServerSentEvent(
                data=(
                    '{"choices":[{"delta":{"content":"world!","tool_calls":[{"index":0, "function":{"arguments":'
                    '"\\":\\"bar\\"}"}}]},"finish_reason":null,'
                    '"index":0,"logprobs":null}],"model":"deepseek-chat","object":"chat.completion.chunk"}'
                )
            ),
            ServerSentEvent(
                data=(
                    '{"choices":[{"delta":{"content":"","role":null},"finish_reason":"stop",'
                    '"index":0,"logprobs":null}],"model":"deepseek-chat","object":"chat.completion.chunk"}'
                )
            ),
            ServerSentEvent(
                data=(
                    '{"choices":[],"model":"deepseek-chat","object":"chat.completion.chunk",'
                    '"usage":{"completion_tokens":17,"prompt_tokens":523,"total_tokens":540}}'
                )
            ),
        ]
        self.streaming_chunks = [
            StreamingChunk(
                content="",
                reasoning_content="reasoning",
                meta={"model": "deepseek-chat", "usage": None},
                index=0,
                start=True,  # First chunk in a stream has start=True
            ),
            StreamingChunk(
                content="Hello ",
                meta={"model": "deepseek-chat", "usage": None},
                index=0,
                start=False,
            ),
            StreamingChunk(
                content="",
                tool_calls=[ToolCallDelta(index=0, id="123", tool_name="test_tool", arguments='{"foo')],
                meta={"model": "deepseek-chat", "usage": None},
                index=0,
                start=False,
            ),
            StreamingChunk(
                content="",
                tool_calls=[ToolCallDelta(index=0, arguments='":"bar"}')],
                meta={"model": "deepseek-chat", "usage": None},
                index=0,
                start=False,
            ),
            StreamingChunk(
                content="world!",
                meta={"model": "deepseek-chat", "usage": None},
                index=0,
                start=False,
            ),
            StreamingChunk(
                content="",
                meta={"model": "deepseek-chat", "usage": None},
                index=0,
                finish_reason="stop",  # Explicitly setting finish_reason for the last chunk
                start=False,
            ),
        ]

    def test_initialization(self):
        self.assertEqual("test_openai_model_service", self.model_service.name)
        self.assertEqual(
            {
                "base_url": "test_uri",
                "organization": None,
                "inference_api_format": "openai-chat",
                "support_stream_options": True,
            },
            self.model_service.configs,
        )
        self.assertEqual({}, self.model_service.secrets)

    def test_initialization_all_args(self):
        openai_model_service = OpenAIChatModelService(
            name="test_openai_model_service",
            api_key=Secret.from_token("test_api_key"),
            base_url="test_uri",
            organization="test_org",
            support_stream_options=False,
            client_kwargs={"headers": {"user_agent": "test"}},
        )
        self.assertEqual("test_openai_model_service", openai_model_service.name)
        self.assertEqual(
            {
                "base_url": "test_uri",
                "organization": "test_org",
                "client_kwargs": {"headers": {"user_agent": "test"}},
                "inference_api_format": "openai-chat",
                "support_stream_options": False,
            },
            openai_model_service.configs,
        )
        self.assertEqual("test_api_key", openai_model_service.secrets["api_key"].resolve_value())

    @patch("ravelin.assets.model_services.chat_model_services.openai.OpenAIChatModelService._build_tool_calls")
    def test_build_message(self, mock_build_tool_calls):
        mock_build_tool_calls.return_value = [
            ToolCall(tool_name="get_current_weather", arguments={"location": "Boston, MA"}, id="call_abc123"),
            ToolCall(tool_name="test_tool", arguments={"foo": "bar"}, id="123"),
        ]
        choice = {
            "message": {
                "content": "Hello",
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {"name": "get_current_weather", "arguments": '{\n"location": "Boston, MA"\n}'},
                    },
                    {
                        "id": "123",
                        "type": "function",
                        "function": {"name": "test_tool", "arguments": '{"foo": "bar"}'},
                    },
                ],
            },
            "index": 0,
            "finish_reason": "tool_calls",
        }
        chat_message = self.model_service._build_message(choice, "gpt-3")
        mock_build_tool_calls.assert_called_once_with(choice["message"]["tool_calls"])
        self.assertEqual([TextContent(text="Hello"), *mock_build_tool_calls.return_value], chat_message.content)
        self.assertEqual("assistant", chat_message.role)
        self.assertEqual("gpt-3", chat_message.meta["model"])
        self.assertEqual(0, chat_message.meta["index"])
        self.assertEqual("tool_calls", chat_message.meta["finish_reason"])

    @patch("ravelin.assets.model_services.chat_model_services.openai.logger")
    def test_build_tool_calls(self, mock_logger):
        raw_tcs = [
            {
                "id": "call_abc123",
                "type": "function",
                "function": {"name": "get_current_weather", "arguments": '{\n"location": "Boston, MA"\n}'},
            },
        ]
        tool_calls = OpenAIChatModelService._build_tool_calls(raw_tcs)
        self.assertEqual(
            [ToolCall(tool_name="get_current_weather", arguments={"location": "Boston, MA"}, id="call_abc123")],
            tool_calls,
        )
        mock_logger.assert_not_called()

    @patch("ravelin.assets.model_services.chat_model_services.openai.logger")
    def test_build_tool_calls_invalid(self, mock_logger):
        with self.assertRaisesRegex(ValueError, "Invalid tool call format: "):
            OpenAIChatModelService._build_tool_calls([{"id": "call_abc123"}])
        mock_logger.assert_not_called()

        tool_calls = OpenAIChatModelService._build_tool_calls(
            [{"id": "123", "type": "function", "function": {"name": "tool", "arguments": "{invalid"}}]
        )
        self.assertListEqual([], tool_calls)
        mock_logger.warning.assert_called_once_with(
            (
                "OpenAI returned a malformed JSON string for tool call arguments. This tool call "
                "will be skipped. To always generate a valid JSON, set `tools_strict` to `True`. "
                "Tool call ID: %s, Tool name: %s, Arguments: %s"
            ),
            "123",
            "tool",
            "{invalid",
        )

    def test_build_chunk(self):
        for i in range(4):
            streaming_chunk = self.model_service._build_chunk(self.sse_contents[i])
            self.assertEqual(self.streaming_chunks[i].content, streaming_chunk.content)
            # Only check model and usage in meta since index and finish_reason are now separate fields
            self.assertEqual(self.streaming_chunks[i].meta.get("model"), streaming_chunk.meta.get("model"))
            self.assertEqual(self.streaming_chunks[i].meta.get("usage"), streaming_chunk.meta.get("usage"))
            # Check that index and finish_reason are properly set as separate fields
            self.assertEqual(self.streaming_chunks[i].index, streaming_chunk.index)
            self.assertEqual(self.streaming_chunks[i].finish_reason, streaming_chunk.finish_reason)

    def test_connect_chunks(self):
        chat_message = self.model_service._connect_chunks(self.streaming_chunks)
        self.assertEqual(
            [TextContent(text="Hello world!"), ToolCall(tool_name="test_tool", arguments={"foo": "bar"}, id="123")],
            chat_message.content,
        )
        self.assertEqual("assistant", chat_message.role)
        self.assertEqual("deepseek-chat", chat_message.meta["model"])
        # Verify that finish_reason is correctly added to the chat message meta
        self.assertEqual("stop", chat_message.meta["finish_reason"])

    @patch("ravelin.assets.model_services.chat_model_services.openai.logger.warning")
    def test_check_finish_reason(self, mock_logger_warning):
        chat_message = ChatMessage.from_assistant(text="Hello", meta={"index": 0, "finish_reason": "stop"})
        self.model_service._check_finish_reason(chat_message)
        mock_logger_warning.assert_not_called()

        chat_message = ChatMessage.from_assistant(text="Hello", meta={"index": 1, "finish_reason": "length"})
        self.model_service._check_finish_reason(chat_message)
        mock_logger_warning.assert_called_once_with(
            "The completion for index %s has been truncated before reaching a natural stopping point. "
            "Increase the max_tokens parameter to allow for longer completions.",
            1,
        )

        chat_message = ChatMessage.from_assistant(text="Hello", meta={"index": 2, "finish_reason": "content_filter"})
        self.model_service._check_finish_reason(chat_message)
        mock_logger_warning.assert_called_with(
            "The completion for index %s has been truncated due to the content filter.",
            2,
        )

        chat_message = ChatMessage.from_assistant(text="Hello", meta={"index": 3, "finish_reason": "tool_calls"})
        self.model_service._check_finish_reason(chat_message)
        mock_logger_warning.assert_called_with(
            (
                "The completion for index %s claimed to have tool calls, but no valid tool call was found in "
                "the response. This could be due to invalid tool call formatting or model errors."
            ),
            3,
        )

    def test_build_request(self):
        client = Mock()
        messages = [ChatMessage.from_user(text="Hello")]
        openai_model_service = OpenAIChatModelService(
            name="test_openai_model_service", base_url="https://openai/url/", api_key=Secret.from_token("test_api_key")
        )
        openai_model_service.build_request(client, messages, model="gpt-3.5-turbo", stream=True, tools=self.tools)

        expected_json_body = {
            "model": "gpt-3.5-turbo",
            "messages": [_convert_message_to_openai_format(message) for message in messages],
            "stream": True,
            "stream_options": {"include_usage": True},
            "tools": [{"type": "function", "function": {**t.tool_spec}} for t in self.tools],
        }

        expected_headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer test_api_key",
            "api-key": "test_api_key",
        }

        client.build_request.assert_called_once_with(
            "POST",
            url="https://openai/url/v1/chat/completions",
            json=expected_json_body,
            headers=expected_headers,
            timeout=None,
        )

    def test_build_request_with_extra_headers(self):
        """Test that extra_headers are properly merged into the request headers."""
        client = Mock()
        messages = [ChatMessage.from_user(text="Hello")]
        extra_headers = {"X-Custom-Header": "custom_value", "Authorization": "Bearer custom_token"}
        openai_model_service = OpenAIChatModelService(
            name="test_openai_model_service", base_url="https://openai/url/", api_key=Secret.from_token("test_api_key")
        )

        openai_model_service.build_request(
            client, messages, model="gpt-3.5-turbo", stream=False, extra_headers=extra_headers
        )

        expected_json_body = {
            "model": "gpt-3.5-turbo",
            "messages": [_convert_message_to_openai_format(message) for message in messages],
            "stream": False,
        }

        expected_headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer custom_token",  # extra_headers should override
            "api-key": "test_api_key",
            "X-Custom-Header": "custom_value",
        }

        client.build_request.assert_called_once_with(
            "POST",
            url="https://openai/url/v1/chat/completions",
            json=expected_json_body,
            headers=expected_headers,
            timeout=None,
        )

    @patch("ravelin.assets.model_services.chat_model_services._base.get_component_context")
    @patch("ravelin.assets.model_services.chat_model_services.openai.OpenAIChatModelService._check_finish_reason")
    @patch("ravelin.assets.model_services.chat_model_services.openai.OpenAIChatModelService._connect_chunks")
    @patch("ravelin.assets.model_services.chat_model_services.openai.OpenAIChatModelService._build_chunk")
    @patch("ravelin.assets.model_services.chat_model_services.openai.ServerSentEventStream")
    def test_parse_response_stream(
        self, mock_sse_stream, mock_build_chunk, mock_connect_chunks, mock_check_finish_reason, mock_get_comp_ctx
    ):
        sse_iter = iter(self.sse_contents)
        streaming_chunks_iter = iter(self.streaming_chunks)

        mock_comp_ctx = Mock()
        mock_get_comp_ctx.return_value = mock_comp_ctx

        # Set up mocks
        mock_sse_stream.return_value.__iter__.return_value = sse_iter
        mock_build_chunk.side_effect = lambda sse: next(streaming_chunks_iter)

        chat_messages = self.model_service.parse_response(Response(200), stream=True)

        # Check mock calls
        self.assertListEqual(
            [call(sse) for sse in self.sse_contents],
            mock_build_chunk.call_args_list,
        )
        # Lifecycle events (start/complete/error) live on infer/ainfer, not
        # parse_response — the collector emits one GENERATION_DELTA per chunk.
        self.assertEqual(6, mock_comp_ctx.emit.call_count)
        mock_comp_ctx.emit.assert_any_call(
            EventType.TOKEN_STREAM, self.streaming_chunks[0], subtype=GENERATION_DELTA, generation_id=None
        )
        subtypes = [c.kwargs.get("subtype") for c in mock_comp_ctx.emit.call_args_list]
        self.assertEqual([GENERATION_DELTA] * 6, subtypes)
        mock_connect_chunks.assert_called_once_with(self.streaming_chunks)
        mock_check_finish_reason.assert_called_once_with(chat_messages[0])
        self.assertEqual([mock_connect_chunks.return_value], chat_messages)

    def test_parse_response_invalid_stream(self):
        empty_response_stream = Response(200, content=iter([]))
        with self.assertRaisesRegex(
            ValueError, "The model service 'test_openai_model_service' returned an empty stream."
        ):
            self.model_service.parse_response(empty_response_stream, stream=True)

    def test_parse_response_no_stream(self):
        response_json = {
            "model": "gpt-3",
            "choices": [
                {"message": {"content": "Hello", "reasoning_content": "foo"}, "index": 0, "finish_reason": "stop"}
            ],
        }
        chat_messages = self.model_service.parse_response(Response(200, json=response_json))
        self.assertEqual(1, len(chat_messages))
        self.assertEqual("Hello", chat_messages[0].text)
        self.assertEqual("foo", chat_messages[0].reasoning_content)
        self.assertEqual("assistant", chat_messages[0].role)
        self.assertEqual("gpt-3", chat_messages[0].meta["model"])
        # In the non-streaming case, index and finish_reason are still in meta since they come directly from the API response
        self.assertEqual(0, chat_messages[0].meta["index"])
        self.assertEqual("stop", chat_messages[0].meta["finish_reason"])

    @patch("ravelin.assets.model_services.chat_model_services._base.get_component_context")
    def test_infer_forwards_generation_id_through_override(self, mock_get_ctx):
        """OpenAI overrides infer(); it must forward generation_id to the base emit path."""

        mock_ctx = Mock()
        mock_get_ctx.return_value = mock_ctx

        response_json = {"model": "gpt-3", "choices": [{"message": {"content": "hi"}, "index": 0}]}
        with patch.object(self.model_service, "_client") as mock_client:
            mock_client.build_request.return_value = Mock()
            mock_response = Mock()
            mock_response.json.return_value = response_json
            mock_response.raise_for_status.return_value = None
            mock_client.send.return_value = mock_response

            self.model_service.infer(
                [ChatMessage.from_user("hi")], model="gpt-3", timeout=5.0, stream=False, generation_id="g0"
            )

        complete_calls = [c for c in mock_ctx.emit.call_args_list if c.kwargs.get("subtype") == GENERATION_COMPLETE]
        self.assertEqual(1, len(complete_calls))
        self.assertEqual("g0", complete_calls[0].kwargs.get("generation_id"))

    def test_parse_response_invalid_no_stream(self):
        resp = Response(200, content="")
        with self.assertRaisesRegex(
            ValueError, "The model service 'test_openai_model_service' returned an invalid JSON response: "
        ):
            self.model_service.parse_response(resp)

    def test_infer_streaming_with_multiple_completions_error(self):
        """Test that infer raises ValueError when streaming with multiple completions (n > 1)."""
        messages = [ChatMessage.from_user("Hello")]
        with self.assertRaisesRegex(
            ValueError, "Streaming with multiple completions is not yet supported for OpenAI models."
        ):
            self.model_service.infer(
                messages, model="gpt-3.5-turbo", timeout=10.0, stream=True, inference_kwargs={"n": 2}
            )

    def test_build_message_with_logprobs(self):
        """Test that logprobs are correctly captured in non-streaming responses."""
        choice = {
            "message": {"content": "hello world"},
            "index": 0,
            "finish_reason": "stop",
            "logprobs": {
                "content": [
                    {
                        "token": "hello",
                        "logprob": -0.0001,
                        "bytes": [104, 101, 108, 108, 111],
                        "top_logprobs": [
                            {"token": "hello", "logprob": -0.0001, "bytes": [104, 101, 108, 108, 111]},
                            {"token": "Hello", "logprob": -14.75, "bytes": [72, 101, 108, 108, 111]},
                        ],
                    },
                    {
                        "token": " world",
                        "logprob": 0.0,
                        "bytes": [32, 119, 111, 114, 108, 100],
                        "top_logprobs": [
                            {"token": " world", "logprob": 0.0, "bytes": [32, 119, 111, 114, 108, 100]},
                            {"token": " earth", "logprob": -19.0, "bytes": [32, 101, 97, 114, 116, 104]},
                        ],
                    },
                ],
                "refusal": None,
            },
        }

        chat_message = self.model_service._build_message(choice, "gpt-4o-mini")

        self.assertEqual("hello world", chat_message.text)
        self.assertEqual("assistant", chat_message.role)
        self.assertEqual("gpt-4o-mini", chat_message.meta["model"])
        self.assertEqual(0, chat_message.meta["index"])
        self.assertEqual("stop", chat_message.meta["finish_reason"])

        # Verify logprobs are captured correctly
        self.assertIn("logprobs", chat_message.meta)
        logprobs = chat_message.meta["logprobs"]
        self.assertEqual(2, len(logprobs["content"]))
        self.assertEqual("hello", logprobs["content"][0]["token"])
        self.assertEqual(" world", logprobs["content"][1]["token"])
        self.assertIsNone(logprobs["refusal"])

    def test_connect_chunks_with_logprobs(self):
        """Test that logprobs are correctly accumulated from multiple streaming chunks."""
        # Create SSE events with logprobs (similar to real OpenAI streaming response)
        sse_with_logprobs = [
            ServerSentEvent(
                data=(
                    '{"choices":[{"delta":{"content":"","role":"assistant"},"finish_reason":null,'
                    '"index":0,"logprobs":{"content":[],"refusal":null}}],'
                    '"model":"gpt-4o-mini","object":"chat.completion.chunk"}'
                )
            ),
            ServerSentEvent(
                data=(
                    '{"choices":[{"delta":{"content":"hello"},"finish_reason":null,'
                    '"index":0,"logprobs":{"content":[{"token":"hello","logprob":-0.0001,'
                    '"bytes":[104,101,108,108,111],"top_logprobs":[{"token":"hello","logprob":-0.0001,'
                    '"bytes":[104,101,108,108,111]},{"token":"Hello","logprob":-14.75,'
                    '"bytes":[72,101,108,108,111]}]}],"refusal":null}}],'
                    '"model":"gpt-4o-mini","object":"chat.completion.chunk"}'
                )
            ),
            ServerSentEvent(
                data=(
                    '{"choices":[{"delta":{"content":" world"},"finish_reason":null,'
                    '"index":0,"logprobs":{"content":[{"token":" world","logprob":0.0,'
                    '"bytes":[32,119,111,114,108,100],"top_logprobs":[{"token":" world","logprob":0.0,'
                    '"bytes":[32,119,111,114,108,100]},{"token":" earth","logprob":-19.0,'
                    '"bytes":[32,101,97,114,116,104]}]}],"refusal":null}}],'
                    '"model":"gpt-4o-mini","object":"chat.completion.chunk"}'
                )
            ),
            ServerSentEvent(
                data=(
                    '{"choices":[{"delta":{"content":""},"finish_reason":"stop",'
                    '"index":0,"logprobs":null}],"model":"gpt-4o-mini","object":"chat.completion.chunk"}'
                )
            ),
            ServerSentEvent(
                data=(
                    '{"choices":[],"model":"gpt-4o-mini","object":"chat.completion.chunk",'
                    '"usage":{"completion_tokens":2,"prompt_tokens":10,"total_tokens":12}}'
                )
            ),
        ]

        # Build chunks from SSE events (not mocking, using real _build_chunk)
        chunks = [self.model_service._build_chunk(sse) for sse in sse_with_logprobs]

        # Verify intermediate chunks have logprobs
        self.assertIn("logprobs", chunks[0].meta)
        self.assertEqual([], chunks[0].meta["logprobs"]["content"])  # First chunk has empty content array

        self.assertIn("logprobs", chunks[1].meta)
        self.assertEqual(1, len(chunks[1].meta["logprobs"]["content"]))
        self.assertEqual("hello", chunks[1].meta["logprobs"]["content"][0]["token"])

        self.assertIn("logprobs", chunks[2].meta)
        self.assertEqual(1, len(chunks[2].meta["logprobs"]["content"]))
        self.assertEqual(" world", chunks[2].meta["logprobs"]["content"][0]["token"])

        # Connect chunks and verify logprobs are accumulated correctly
        chat_message = self.model_service._connect_chunks(chunks)

        self.assertEqual("hello world", chat_message.text)
        self.assertEqual("assistant", chat_message.role)
        self.assertEqual("gpt-4o-mini", chat_message.meta["model"])
        self.assertEqual("stop", chat_message.meta["finish_reason"])

        # Verify accumulated logprobs match non-streaming format
        self.assertIn("logprobs", chat_message.meta)
        logprobs = chat_message.meta["logprobs"]
        self.assertEqual(2, len(logprobs["content"]))
        self.assertEqual("hello", logprobs["content"][0]["token"])
        self.assertEqual(-0.0001, logprobs["content"][0]["logprob"])
        self.assertEqual(" world", logprobs["content"][1]["token"])
        self.assertEqual(0.0, logprobs["content"][1]["logprob"])
        self.assertIsNone(logprobs["refusal"])


class TestOpenAIChatModelServiceAsync(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.model_service = OpenAIChatModelService(name="test_openai_model_service", base_url="test_uri")
        self.tools = [MagicMock(autospec=Tool)]
        self.sse_contents = [
            ServerSentEvent(
                data=(
                    '{"choices":[{"delta":{"content":null,"reasoning_content":"reasoning","role":"assistant"},"finish_r'
                    'eason":null,"index":0,"logprobs":null}],"model":"deepseek-chat","object":"chat.completion.chunk"}'
                )
            ),
            ServerSentEvent(
                data=(
                    '{"choices":[{"delta":{"content":"Hello ","role":"assistant"},"finish_reason":null,'
                    '"index":0,"logprobs":null}],"model":"deepseek-chat","object":"chat.completion.chunk"}'
                )
            ),
            ServerSentEvent(
                data=(
                    '{"choices":[{"delta":{"content":null,"tool_calls":[{"index":0,"id":"123","type":"function",'
                    '"function":{"name":"test_tool","arguments":"{\\"foo"}}]},"finish_reason":null,'
                    '"index":0,"logprobs":null}],"model":"deepseek-chat","object":"chat.completion.chunk"}'
                )
            ),
            ServerSentEvent(
                data=(
                    '{"choices":[{"delta":{"content":"world!","tool_calls":[{"index":0, "function":{"arguments":'
                    '"\\":\\"bar\\"}"}}]},"finish_reason":null,'
                    '"index":0,"logprobs":null}],"model":"deepseek-chat","object":"chat.completion.chunk"}'
                )
            ),
            ServerSentEvent(
                data=(
                    '{"choices":[{"delta":{"content":"","role":null},"finish_reason":"stop",'
                    '"index":0,"logprobs":null}],"model":"deepseek-chat","object":"chat.completion.chunk"}'
                )
            ),
            ServerSentEvent(
                data=(
                    '{"choices":[],"model":"deepseek-chat","object":"chat.completion.chunk",'
                    '"usage":{"completion_tokens":17,"prompt_tokens":523,"total_tokens":540}}'
                )
            ),
        ]
        self.streaming_chunks = [
            StreamingChunk(
                content="",
                reasoning_content="reasoning",
                meta={"model": "deepseek-chat", "usage": None},
                index=0,
                start=True,  # First chunk in a stream has start=True
            ),
            StreamingChunk(
                content="Hello ",
                meta={"model": "deepseek-chat", "usage": None},
                index=0,
                start=False,
            ),
            StreamingChunk(
                content="",
                tool_calls=[ToolCallDelta(index=0, id="123", tool_name="test_tool", arguments='{"foo')],
                meta={"model": "deepseek-chat", "usage": None},
                index=0,
                start=False,
            ),
            StreamingChunk(
                content="",
                tool_calls=[ToolCallDelta(index=0, arguments='":"bar"}')],
                meta={"model": "deepseek-chat", "usage": None},
                index=0,
                start=False,
            ),
            StreamingChunk(
                content="world!",
                meta={"model": "deepseek-chat", "usage": None},
                index=0,
                start=False,
            ),
            StreamingChunk(
                content="",
                meta={"model": "deepseek-chat", "usage": None},
                index=0,
                finish_reason="stop",  # Explicitly setting finish_reason for the last chunk
                start=False,
            ),
        ]

    @patch("ravelin.assets.model_services.chat_model_services._base.get_component_context")
    @patch("ravelin.assets.model_services.chat_model_services.openai.OpenAIChatModelService._check_finish_reason")
    @patch("ravelin.assets.model_services.chat_model_services.openai.OpenAIChatModelService._connect_chunks")
    @patch("ravelin.assets.model_services.chat_model_services.openai.OpenAIChatModelService._build_chunk")
    @patch("ravelin.assets.model_services.chat_model_services.openai.AsyncServerSentEventStream")
    async def test_aparse_response_stream(
        self, mock_sse_stream, mock_build_chunk, mock_connect_chunks, mock_check_finish_reason, mock_get_comp_ctx
    ):
        streaming_chunks_iter = iter(self.streaming_chunks)

        mock_comp_ctx = Mock()
        mock_comp_ctx.emit_async = AsyncMock()
        mock_get_comp_ctx.return_value = mock_comp_ctx

        # Set up mocks — use a real async generator for __aiter__
        sse_events = list(self.sse_contents)

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
        # Lifecycle events live on ainfer, not aparse_response — the collector
        # emits one GENERATION_DELTA per chunk.
        self.assertEqual(6, mock_comp_ctx.emit_async.call_count)
        mock_comp_ctx.emit_async.assert_any_call(
            EventType.TOKEN_STREAM, self.streaming_chunks[0], subtype=GENERATION_DELTA, generation_id=None
        )
        subtypes = [c.kwargs.get("subtype") for c in mock_comp_ctx.emit_async.call_args_list]
        self.assertEqual([GENERATION_DELTA] * 6, subtypes)
        mock_connect_chunks.assert_called_once_with(self.streaming_chunks)
        mock_check_finish_reason.assert_called_once_with(chat_messages[0])
        self.assertEqual([mock_connect_chunks.return_value], chat_messages)

    async def test_aparse_response_no_stream(self):
        response_json = {
            "model": "gpt-3",
            "choices": [
                {"message": {"content": "Hello", "reasoning_content": "foo"}, "index": 0, "finish_reason": "stop"}
            ],
        }
        chat_messages = await self.model_service.aparse_response(Response(200, json=response_json))
        self.assertEqual(1, len(chat_messages))
        self.assertEqual("Hello", chat_messages[0].text)
        self.assertEqual("foo", chat_messages[0].reasoning_content)
        self.assertEqual("assistant", chat_messages[0].role)
        self.assertEqual("gpt-3", chat_messages[0].meta["model"])
        # In the non-streaming case, index and finish_reason are still in meta since they come directly from the API response
        self.assertEqual(0, chat_messages[0].meta["index"])
        self.assertEqual("stop", chat_messages[0].meta["finish_reason"])

    async def test_aparse_response_invalid_no_stream(self):
        resp = Response(200, content="")
        with self.assertRaisesRegex(
            ValueError, "The model service 'test_openai_model_service' returned an invalid JSON response: "
        ):
            await self.model_service.aparse_response(resp)

    async def test_ainfer_streaming_with_multiple_completions_error(self):
        """Test that ainfer raises ValueError when streaming with multiple completions (n > 1)."""
        messages = [ChatMessage.from_user("Hello")]
        with self.assertRaisesRegex(
            ValueError, "Streaming with multiple completions is not yet supported for OpenAI models."
        ):
            await self.model_service.ainfer(
                messages, model="gpt-3.5-turbo", timeout=10.0, stream=True, inference_kwargs={"n": 2}
            )


if __name__ == "__main__":
    unittest.main()
