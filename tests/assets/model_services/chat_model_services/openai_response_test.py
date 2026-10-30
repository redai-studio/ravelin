import json
import unittest
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

from haystack.dataclasses import ToolCallDelta
from haystack.tools import Tool
from haystack.utils import Secret
from httpx import Response

from ravelin.assets.model_services.chat_model_services.openai_response import (
    OpenAIResponseModelService,
    _convert_message_to_response_items,
)
from ravelin.dataclasses import ByteStream, ChatMessage, ChatRole, StreamingChunk
from ravelin.dataclasses.chat_message import TextContent, ToolCall
from ravelin.streaming.event import GENERATION_COMPLETE, GENERATION_DELTA
from ravelin.utils.streaming import ServerSentEvent


class TestConvertMessageToResponseItems(unittest.TestCase):
    def test_user_text(self):
        message = ChatMessage.from_user(text="Hello")
        self.assertEqual(
            [{"role": "user", "content": [{"type": "input_text", "text": "Hello"}]}],
            _convert_message_to_response_items(message),
        )

    def test_assistant_text(self):
        message = ChatMessage.from_assistant(text="Hi there")
        self.assertEqual(
            [{"role": "assistant", "content": [{"type": "output_text", "text": "Hi there"}]}],
            _convert_message_to_response_items(message),
        )

    def test_user_image(self):
        message = ChatMessage.from_user(
            text="What's this?",
            media=[ByteStream(data=b"png-bytes", mime_type="image/png", meta={"detail": "low"})],
        )
        items = _convert_message_to_response_items(message)
        self.assertEqual(1, len(items))
        content = items[0]["content"]
        self.assertEqual(2, len(content))
        self.assertEqual("input_image", content[0]["type"])
        self.assertTrue(content[0]["image_url"].startswith("data:image/png;base64,"))
        self.assertEqual("low", content[0]["detail"])
        self.assertEqual({"type": "input_text", "text": "What's this?"}, content[1])

    def test_assistant_tool_call_splits_into_multiple_items(self):
        message = ChatMessage.from_assistant(
            text="Calling tool",
            tool_calls=[ToolCall(tool_name="do_thing", arguments={"foo": "bar"}, id="call_1")],
        )
        items = _convert_message_to_response_items(message)
        self.assertEqual(
            [
                {"role": "assistant", "content": [{"type": "output_text", "text": "Calling tool"}]},
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "do_thing",
                    "arguments": json.dumps({"foo": "bar"}),
                },
            ],
            items,
        )

    def test_assistant_tool_call_missing_id_raises(self):
        message = ChatMessage.from_assistant(
            tool_calls=[ToolCall(tool_name="do_thing", arguments={"foo": "bar"})],
        )
        with self.assertRaisesRegex(ValueError, "`ToolCall` must have a non-null `id` attribute"):
            _convert_message_to_response_items(message)

    def test_tool_result(self):
        message = ChatMessage.from_tool(
            "the result",
            origin=ToolCall(tool_name="do_thing", arguments={}, id="call_1"),
            error=False,
        )
        self.assertEqual(
            [{"type": "function_call_output", "call_id": "call_1", "output": "the result"}],
            _convert_message_to_response_items(message),
        )

    def test_tool_result_without_id_raises(self):
        message = ChatMessage.from_tool(
            "the result",
            origin=ToolCall(tool_name="do_thing", arguments={}),
            error=False,
        )
        with self.assertRaisesRegex(ValueError, "`ToolCall` must have a non-null `id` attribute"):
            _convert_message_to_response_items(message)

    def test_system_message_raises(self):
        with self.assertRaisesRegex(ValueError, "System messages are handled via the top-level `instructions` field"):
            _convert_message_to_response_items(ChatMessage.from_system("system prompt"))

    def test_reasoning_blocks_are_reattached_before_message_content(self):
        message = ChatMessage.from_assistant(
            text="continue",
            meta={
                "reasoning_blocks": [{"type": "reasoning", "id": "rs_1", "summary": []}],
            },
        )
        items = _convert_message_to_response_items(message)
        self.assertEqual(
            [
                {"type": "reasoning", "id": "rs_1", "summary": []},
                {"role": "assistant", "content": [{"type": "output_text", "text": "continue"}]},
            ],
            items,
        )

    def test_tool_call_on_user_message_raises(self):
        message = ChatMessage(
            _role=ChatRole.USER,
            _content=[ToolCall(tool_name="do_thing", arguments={"foo": "bar"}, id="call_1")],
        )
        with self.assertRaisesRegex(ValueError, "Tool calls should only be included for assistant messages."):
            _convert_message_to_response_items(message)


class TestOpenAIResponseModelService(unittest.TestCase):
    def setUp(self):
        self.model_service = OpenAIResponseModelService(name="test_svc", base_url="test_uri")
        self.tools = [MagicMock(autospec=Tool)]
        self.tools[0].name = "tool_a"
        self.tools[0].description = "does thing"
        self.tools[0].parameters = {"type": "object", "properties": {}}

        # A representative sequence of SSE events for a stream that produces reasoning + text + a tool call.
        self.sse_events = [
            ServerSentEvent(
                event="response.created",
                data=json.dumps({"type": "response.created", "response": {"id": "resp_1", "model": "gpt-5"}}),
            ),
            ServerSentEvent(
                event="response.output_item.added",
                data=json.dumps(
                    {
                        "type": "response.output_item.added",
                        "output_index": 0,
                        "item": {"id": "rs_1", "type": "reasoning", "summary": []},
                    }
                ),
            ),
            ServerSentEvent(
                event="response.reasoning_summary_text.delta",
                data=json.dumps(
                    {
                        "type": "response.reasoning_summary_text.delta",
                        "output_index": 0,
                        "item_id": "rs_1",
                        "delta": "thinking...",
                    }
                ),
            ),
            ServerSentEvent(
                event="response.output_text.delta",
                data=json.dumps(
                    {
                        "type": "response.output_text.delta",
                        "output_index": 1,
                        "item_id": "msg_1",
                        "delta": "Hello ",
                    }
                ),
            ),
            ServerSentEvent(
                event="response.output_text.delta",
                data=json.dumps(
                    {
                        "type": "response.output_text.delta",
                        "output_index": 1,
                        "item_id": "msg_1",
                        "delta": "world!",
                    }
                ),
            ),
            ServerSentEvent(
                event="response.output_item.added",
                data=json.dumps(
                    {
                        "type": "response.output_item.added",
                        "output_index": 2,
                        "item": {
                            "id": "fc_1",
                            "type": "function_call",
                            "call_id": "call_1",
                            "name": "tool_a",
                            "arguments": "",
                        },
                    }
                ),
            ),
            ServerSentEvent(
                event="response.function_call_arguments.delta",
                data=json.dumps(
                    {
                        "type": "response.function_call_arguments.delta",
                        "output_index": 2,
                        "item_id": "fc_1",
                        "delta": '{"foo":',
                    }
                ),
            ),
            ServerSentEvent(
                event="response.function_call_arguments.delta",
                data=json.dumps(
                    {
                        "type": "response.function_call_arguments.delta",
                        "output_index": 2,
                        "item_id": "fc_1",
                        "delta": '"bar"}',
                    }
                ),
            ),
            ServerSentEvent(
                event="response.completed",
                data=json.dumps(
                    {
                        "type": "response.completed",
                        "response": {
                            "id": "resp_1",
                            "model": "gpt-5",
                            "status": "completed",
                            "output": [
                                {"id": "fc_1", "type": "function_call", "call_id": "call_1", "name": "tool_a"},
                            ],
                            "usage": {"input_tokens": 12, "output_tokens": 5},
                        },
                    }
                ),
            ),
        ]

    def test_initialization(self):
        self.assertEqual("test_svc", self.model_service.name)
        self.assertEqual(
            {
                "base_url": "test_uri",
                "organization": None,
                "inference_api_format": "openai-response",
            },
            self.model_service.configs,
        )
        self.assertEqual({}, self.model_service.secrets)

    def test_initialization_all_args(self):
        svc = OpenAIResponseModelService(
            name="svc",
            api_key=Secret.from_token("test_api_key"),
            base_url="test_uri",
            organization="test_org",
            client_kwargs={"headers": {"user_agent": "test"}},
        )
        self.assertEqual(
            {
                "base_url": "test_uri",
                "organization": "test_org",
                "client_kwargs": {"headers": {"user_agent": "test"}},
                "inference_api_format": "openai-response",
            },
            svc.configs,
        )
        self.assertEqual("test_api_key", svc.secrets["api_key"].resolve_value())

    def test_build_request_dual_auth_and_flat_tool_schema(self):
        client = Mock()
        messages = [ChatMessage.from_system("be nice"), ChatMessage.from_user(text="Hello")]
        svc = OpenAIResponseModelService(
            name="svc", base_url="https://openai/url/", api_key=Secret.from_token("test_api_key")
        )

        svc.build_request(
            client,
            messages,
            model="gpt-5",
            stream=True,
            tools=self.tools,
            inference_kwargs={"reasoning": {"effort": "medium"}, "max_output_tokens": 256},
        )

        expected_body = {
            "model": "gpt-5",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "Hello"}]}],
            "stream": True,
            "reasoning": {"effort": "medium"},
            "max_output_tokens": 256,
            "instructions": "be nice",
            "tools": [
                {
                    "type": "function",
                    "name": "tool_a",
                    "description": "does thing",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
        }
        expected_headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer test_api_key",
            "api-key": "test_api_key",
        }
        client.build_request.assert_called_once_with(
            "POST",
            url="https://openai/url/v1/responses",
            json=expected_body,
            headers=expected_headers,
            timeout=None,
        )

    def test_build_request_extra_headers_override(self):
        client = Mock()
        messages = [ChatMessage.from_user(text="Hi")]
        svc = OpenAIResponseModelService(
            name="svc", base_url="https://openai/url/", api_key=Secret.from_token("test_api_key")
        )

        svc.build_request(
            client,
            messages,
            model="gpt-5",
            stream=False,
            extra_headers={"X-Custom": "v", "Authorization": "Bearer other"},
        )
        _, kwargs = client.build_request.call_args
        self.assertEqual("Bearer other", kwargs["headers"]["Authorization"])
        self.assertEqual("test_api_key", kwargs["headers"]["api-key"])
        self.assertEqual("v", kwargs["headers"]["X-Custom"])

    def test_build_request_system_message_must_be_single_text(self):
        client = Mock()
        bad_system = ChatMessage(_role=ChatRole.SYSTEM, _content=[TextContent(text="a"), TextContent(text="b")])
        svc = OpenAIResponseModelService(name="svc", base_url="https://openai/url/")
        with self.assertRaisesRegex(ValueError, "The system prompt must be a single text message."):
            svc.build_request(client, [bad_system, ChatMessage.from_user(text="hi")], model="gpt-5")

    def test_build_message_non_stream(self):
        response_json = {
            "id": "resp_1",
            "model": "gpt-5",
            "status": "completed",
            "usage": {"input_tokens": 3, "output_tokens": 5},
            "output": [
                {
                    "id": "rs_1",
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "why"}],
                },
                {
                    "id": "msg_1",
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Hi"}],
                },
                {
                    "id": "fc_1",
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "tool_a",
                    "arguments": '{"foo": "bar"}',
                },
            ],
        }
        message = self.model_service._build_message(response_json)
        self.assertEqual("Hi", message.text)
        self.assertEqual("why", message.reasoning_content)
        self.assertEqual(
            [ToolCall(id="call_1", tool_name="tool_a", arguments={"foo": "bar"})],
            [tc for tc in message.content if isinstance(tc, ToolCall)],
        )
        self.assertEqual("resp_1", message.meta["id"])
        self.assertEqual("tool_calls", message.meta["finish_reason"])
        self.assertEqual({"input_tokens": 3, "output_tokens": 5}, message.meta["usage"])
        self.assertEqual(1, len(message.meta["reasoning_blocks"]))
        self.assertEqual("rs_1", message.meta["reasoning_blocks"][0]["id"])

    def test_build_message_incomplete_max_tokens_maps_to_length(self):
        response_json = {
            "id": "resp_2",
            "model": "gpt-5",
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [{"id": "msg_1", "type": "message", "content": [{"type": "output_text", "text": "partial"}]}],
        }
        message = self.model_service._build_message(response_json)
        self.assertEqual("length", message.meta["finish_reason"])
        self.assertEqual({"reason": "max_output_tokens"}, message.meta["incomplete_details"])

    def test_build_tool_call_malformed_arguments_is_skipped(self):
        with patch("ravelin.assets.model_services.chat_model_services.openai_response.logger") as mock_logger:
            tc = OpenAIResponseModelService._build_tool_call(
                {"call_id": "call_x", "name": "tool_x", "arguments": "{invalid"}
            )
            self.assertIsNone(tc)
            mock_logger.warning.assert_called_once()

    def test_build_tool_call_missing_key_raises(self):
        with self.assertRaisesRegex(ValueError, "Invalid tool call format:"):
            OpenAIResponseModelService._build_tool_call({"call_id": "x"})

    def test_build_chunk_text_delta(self):
        sse = ServerSentEvent(
            event="response.output_text.delta",
            data=json.dumps(
                {
                    "type": "response.output_text.delta",
                    "output_index": 1,
                    "item_id": "msg_1",
                    "delta": "Hello",
                }
            ),
        )
        chunk = self.model_service._build_chunk(sse)
        self.assertEqual("Hello", chunk.content)
        self.assertEqual(1, chunk.index)
        self.assertIsNone(chunk.finish_reason)

    def test_build_chunk_function_call_start_and_delta(self):
        start = self.model_service._build_chunk(
            ServerSentEvent(
                event="response.output_item.added",
                data=json.dumps(
                    {
                        "type": "response.output_item.added",
                        "output_index": 2,
                        "item": {
                            "id": "fc_1",
                            "type": "function_call",
                            "call_id": "call_1",
                            "name": "tool_a",
                            "arguments": "",
                        },
                    }
                ),
            )
        )
        self.assertIsNotNone(start)
        self.assertEqual(1, len(start.tool_calls))
        self.assertEqual("call_1", start.tool_calls[0].id)
        self.assertEqual("tool_a", start.tool_calls[0].tool_name)
        self.assertTrue(start.start)

        delta = self.model_service._build_chunk(
            ServerSentEvent(
                event="response.function_call_arguments.delta",
                data=json.dumps(
                    {
                        "type": "response.function_call_arguments.delta",
                        "output_index": 2,
                        "item_id": "fc_1",
                        "delta": '{"foo":',
                    }
                ),
            )
        )
        self.assertEqual('{"foo":', delta.tool_calls[0].arguments)

    def test_build_chunk_reasoning_delta(self):
        sse = ServerSentEvent(
            event="response.reasoning_summary_text.delta",
            data=json.dumps(
                {
                    "type": "response.reasoning_summary_text.delta",
                    "output_index": 0,
                    "item_id": "rs_1",
                    "delta": "hmm",
                }
            ),
        )
        chunk = self.model_service._build_chunk(sse)
        self.assertEqual("hmm", chunk.reasoning_content)
        self.assertEqual("", chunk.content)

    def test_build_chunk_completed(self):
        sse = ServerSentEvent(
            event="response.completed",
            data=json.dumps(
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_1",
                        "model": "gpt-5",
                        "status": "completed",
                        "output": [],
                        "usage": {"input_tokens": 3, "output_tokens": 2},
                    },
                }
            ),
        )
        chunk = self.model_service._build_chunk(sse)
        self.assertEqual("stop", chunk.finish_reason)
        self.assertEqual({"input_tokens": 3, "output_tokens": 2}, chunk.meta["usage"])

    def test_build_chunk_failed_raises(self):
        sse = ServerSentEvent(
            event="response.failed",
            data=json.dumps({"type": "response.failed", "response": {"error": {"message": "boom"}}}),
        )
        with self.assertRaisesRegex(RuntimeError, "Error received during Responses streaming: boom"):
            self.model_service._build_chunk(sse)

    def test_build_chunk_ignored_events(self):
        sse = ServerSentEvent(event="response.in_progress", data=json.dumps({"type": "response.in_progress"}))
        self.assertIsNone(self.model_service._build_chunk(sse))

    def test_connect_chunks_reassembles_tool_arguments(self):
        chunks = [
            StreamingChunk(content="", meta={"id": "resp_1", "model": "gpt-5"}, start=True),
            StreamingChunk(
                content="",
                meta={"item_id": "rs_1", "reasoning_block": {"type": "reasoning", "id": "rs_1"}},
                index=0,
                start=True,
            ),
            StreamingChunk(content="", reasoning_content="think", meta={"item_id": "rs_1"}, index=0),
            StreamingChunk(content="Hello ", meta={"item_id": "msg_1"}, index=1),
            StreamingChunk(content="world!", meta={"item_id": "msg_1"}, index=1),
            StreamingChunk(
                content="",
                tool_calls=[ToolCallDelta(index=2, id="call_1", tool_name="tool_a", arguments="")],
                meta={"item_id": "fc_1"},
                index=2,
                start=True,
            ),
            StreamingChunk(
                content="",
                tool_calls=[ToolCallDelta(index=2, arguments='{"foo":')],
                meta={"item_id": "fc_1"},
                index=2,
            ),
            StreamingChunk(
                content="",
                tool_calls=[ToolCallDelta(index=2, arguments='"bar"}')],
                meta={"item_id": "fc_1"},
                index=2,
            ),
            StreamingChunk(
                content="",
                meta={"id": "resp_1", "model": "gpt-5", "usage": {"input_tokens": 3, "output_tokens": 5}},
                finish_reason="tool_calls",
            ),
        ]
        message = self.model_service._connect_chunks(chunks)
        self.assertEqual("Hello world!", message.text)
        self.assertEqual("think", message.reasoning_content)
        tool_calls = [tc for tc in message.content if isinstance(tc, ToolCall)]
        self.assertEqual([ToolCall(id="call_1", tool_name="tool_a", arguments={"foo": "bar"})], tool_calls)
        self.assertEqual("tool_calls", message.meta["finish_reason"])
        self.assertEqual({"input_tokens": 3, "output_tokens": 5}, message.meta["usage"])
        self.assertEqual(1, len(message.meta["reasoning_blocks"]))

    def test_reasoning_start_chunk_carries_no_placeholder_reasoning_content(self):
        """The reasoning-item start (response.output_item.added / type=reasoning) must not carry a placeholder
        reasoning_content: only the reasoning_block, so that a stream with a reasoning item but no summary/text
        delta (e.g. reasoning.summary=none) reassembles with reasoning_content=None like the non-streaming path."""
        sse = ServerSentEvent(
            event="response.output_item.added",
            data=json.dumps(
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": {"id": "rs_1", "type": "reasoning", "summary": []},
                }
            ),
        )
        chunk = self.model_service._build_chunk(sse)
        self.assertIsNone(chunk.reasoning_content)
        self.assertIn("reasoning_block", chunk.meta)

        # And end-to-end: the reassembled ChatMessage has reasoning_content=None despite a reasoning block existing.
        message = self.model_service._connect_chunks(
            [
                StreamingChunk(content="", meta={"id": "resp_1", "model": "gpt-5"}, start=True),
                chunk,
                StreamingChunk(content="Hi", meta={"item_id": "msg_1"}, index=1),
                StreamingChunk(
                    content="",
                    meta={"id": "resp_1", "model": "gpt-5", "usage": {"input_tokens": 3, "output_tokens": 1}},
                    finish_reason="stop",
                ),
            ]
        )
        self.assertEqual("Hi", message.text)
        self.assertIsNone(message.reasoning_content)
        self.assertEqual(1, len(message.meta["reasoning_blocks"]))

    def test_connect_chunks_empty_raises(self):
        with self.assertRaisesRegex(ValueError, "Cannot reassemble an empty chunk list."):
            self.model_service._connect_chunks([])

    @patch("ravelin.assets.model_services.chat_model_services._base.get_component_context")
    @patch(
        "ravelin.assets.model_services.chat_model_services."
        "openai_response.OpenAIResponseModelService._check_finish_reason"
    )
    @patch(
        "ravelin.assets.model_services.chat_model_services."
        "openai_response.OpenAIResponseModelService._connect_chunks"
    )
    @patch("ravelin.assets.model_services.chat_model_services.openai_response.ServerSentEventStream")
    def test_parse_response_stream_emits_deltas(
        self, mock_sse_stream, mock_connect_chunks, mock_check_finish_reason, mock_get_comp_ctx
    ):
        mock_comp_ctx = Mock()
        mock_get_comp_ctx.return_value = mock_comp_ctx

        mock_sse_stream.return_value.__iter__.return_value = iter(self.sse_events)

        chat_messages = self.model_service.parse_response(Response(200), stream=True)

        # Every non-None chunk should get emitted as a GENERATION_DELTA.
        subtypes = [c.kwargs.get("subtype") for c in mock_comp_ctx.emit.call_args_list]
        # The created + item-added(reasoning) + reasoning-delta + 2 text deltas + item-added(fc) + 2 arg deltas
        # + completed = 9 non-None chunks.
        self.assertEqual([GENERATION_DELTA] * 9, subtypes)
        mock_connect_chunks.assert_called_once()
        mock_check_finish_reason.assert_called_once_with(chat_messages[0])

    def test_parse_response_no_stream(self):
        response_json = {
            "id": "resp_1",
            "model": "gpt-5",
            "status": "completed",
            "output": [{"id": "msg_1", "type": "message", "content": [{"type": "output_text", "text": "Hi"}]}],
        }
        chat_messages = self.model_service.parse_response(Response(200, json=response_json))
        self.assertEqual(1, len(chat_messages))
        self.assertEqual("Hi", chat_messages[0].text)
        self.assertEqual("stop", chat_messages[0].meta["finish_reason"])

    def test_parse_response_invalid_no_stream(self):
        with self.assertRaisesRegex(ValueError, "returned an invalid JSON response"):
            self.model_service.parse_response(Response(200, content=""))

    @patch("ravelin.assets.model_services.chat_model_services._base.get_component_context")
    def test_infer_forwards_generation_id_through_base(self, mock_get_ctx):
        mock_ctx = Mock()
        mock_get_ctx.return_value = mock_ctx

        response_json = {
            "id": "resp_1",
            "model": "gpt-5",
            "status": "completed",
            "output": [{"id": "msg_1", "type": "message", "content": [{"type": "output_text", "text": "hi"}]}],
        }
        with patch.object(self.model_service, "_client") as mock_client:
            mock_client.build_request.return_value = Mock()
            mock_response = Mock()
            mock_response.json.return_value = response_json
            mock_response.raise_for_status.return_value = None
            mock_client.send.return_value = mock_response

            self.model_service.infer(
                [ChatMessage.from_user("hi")], model="gpt-5", timeout=5.0, stream=False, generation_id="g0"
            )

        complete_calls = [c for c in mock_ctx.emit.call_args_list if c.kwargs.get("subtype") == GENERATION_COMPLETE]
        self.assertEqual(1, len(complete_calls))
        self.assertEqual("g0", complete_calls[0].kwargs.get("generation_id"))


class TestOpenAIResponseModelServiceAsync(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.model_service = OpenAIResponseModelService(name="test_svc", base_url="test_uri")

    @patch("ravelin.assets.model_services.chat_model_services._base.get_component_context")
    @patch(
        "ravelin.assets.model_services.chat_model_services."
        "openai_response.OpenAIResponseModelService._check_finish_reason"
    )
    @patch(
        "ravelin.assets.model_services.chat_model_services."
        "openai_response.OpenAIResponseModelService._connect_chunks"
    )
    @patch("ravelin.assets.model_services.chat_model_services.openai_response.AsyncServerSentEventStream")
    async def test_aparse_response_stream_emits_deltas(
        self, mock_sse_stream, mock_connect_chunks, mock_check_finish_reason, mock_get_comp_ctx
    ):
        mock_comp_ctx = Mock()
        mock_comp_ctx.emit_async = AsyncMock()
        mock_get_comp_ctx.return_value = mock_comp_ctx

        sse_events = [
            ServerSentEvent(
                event="response.output_text.delta",
                data=json.dumps(
                    {"type": "response.output_text.delta", "output_index": 0, "item_id": "m", "delta": "hi"}
                ),
            ),
            ServerSentEvent(
                event="response.completed",
                data=json.dumps(
                    {
                        "type": "response.completed",
                        "response": {
                            "id": "resp_1",
                            "model": "gpt-5",
                            "status": "completed",
                            "output": [],
                            "usage": {"input_tokens": 1, "output_tokens": 1},
                        },
                    }
                ),
            ),
        ]

        async def _async_iter():
            for e in sse_events:
                yield e

        mock_sse_stream.return_value.__aiter__ = lambda self: _async_iter()

        chat_messages = await self.model_service.aparse_response(Response(200), stream=True)
        subtypes = [c.kwargs.get("subtype") for c in mock_comp_ctx.emit_async.call_args_list]
        self.assertEqual([GENERATION_DELTA, GENERATION_DELTA], subtypes)
        mock_connect_chunks.assert_called_once()
        mock_check_finish_reason.assert_called_once_with(chat_messages[0])

    async def test_aparse_response_no_stream(self):
        response_json = {
            "id": "resp_1",
            "model": "gpt-5",
            "status": "completed",
            "output": [{"id": "msg_1", "type": "message", "content": [{"type": "output_text", "text": "Hi"}]}],
        }
        chat_messages = await self.model_service.aparse_response(Response(200, json=response_json))
        self.assertEqual(1, len(chat_messages))
        self.assertEqual("Hi", chat_messages[0].text)

    async def test_aparse_response_invalid_no_stream(self):
        with self.assertRaisesRegex(ValueError, "returned an invalid JSON response"):
            await self.model_service.aparse_response(Response(200, content=""))


class TestCheckFinishReasonWarnings(unittest.TestCase):
    def setUp(self):
        self.svc = OpenAIResponseModelService(name="svc", base_url="test_uri")

    @patch("ravelin.assets.model_services.chat_model_services.openai_response.logger.warning")
    def test_length(self, mock_warn):
        msg = ChatMessage.from_assistant(text="x", meta={"id": "r", "finish_reason": "length"})
        self.svc._check_finish_reason(msg)
        mock_warn.assert_called_once()

    @patch("ravelin.assets.model_services.chat_model_services.openai_response.logger.warning")
    def test_content_filter(self, mock_warn):
        msg = ChatMessage.from_assistant(text="x", meta={"id": "r", "finish_reason": "content_filter"})
        self.svc._check_finish_reason(msg)
        mock_warn.assert_called_once()

    @patch("ravelin.assets.model_services.chat_model_services.openai_response.logger.warning")
    def test_tool_calls_without_call(self, mock_warn):
        msg = ChatMessage.from_assistant(text="x", meta={"id": "r", "finish_reason": "tool_calls"})
        self.svc._check_finish_reason(msg)
        mock_warn.assert_called_once()

    @patch("ravelin.assets.model_services.chat_model_services.openai_response.logger.warning")
    def test_stop_is_silent(self, mock_warn):
        msg = ChatMessage.from_assistant(text="x", meta={"id": "r", "finish_reason": "stop"})
        self.svc._check_finish_reason(msg)
        mock_warn.assert_not_called()


# Silence unused-import warnings for helpers imported for readability.
_ = call
