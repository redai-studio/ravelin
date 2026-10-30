import base64
import json
import unittest
from unittest.mock import patch, Mock, AsyncMock, call

from haystack.dataclasses import StreamingChunk
from haystack.tools import Tool
from haystack.utils import Secret
from httpx import Response

from ravelin.streaming import EventType
from ravelin.streaming.event import GENERATION_DELTA
from ravelin.assets.model_services.chat_model_services.amazon_bedrock import BedrockAnthropicChatModelService
from ravelin.dataclasses.chat_message import ChatMessage, TextContent


class TestBedrockAnthropicChatModelService(unittest.TestCase):
    def setUp(self):
        self.model_service = BedrockAnthropicChatModelService(
            name="test_bedrock_anthropic_model_service",
            model="claude-3-5-sonnet",
            base_url="https://bedrock-anthropic/url/",
            api_key=Secret.from_token("test_api_key"),
        )

    def test_initialization(self):
        self.assertEqual(self.model_service.name, "test_bedrock_anthropic_model_service")
        self.assertDictEqual(
            {
                "base_url": "https://bedrock-anthropic/url/",
                "model": "claude-3-5-sonnet",
                "organization": None,
                "inference_api_format": "bedrock-anthropic-chat",
            },
            self.model_service.configs,
        )
        self.assertEqual("test_api_key", self.model_service.secrets["api_key"].resolve_value())
        self.assertEqual(
            "https://bedrock-anthropic/url/model/claude-3-5-sonnet/invoke",
            self.model_service.build_request_url("inference", model="claude-3-5-sonnet"),
        )
        self.assertEqual(
            "https://bedrock-anthropic/url/model/claude-3-5-sonnet/invoke-with-response-stream",
            self.model_service.stream_url,
        )

    def test_decode_data(self):
        # Test decoding a stream of bytes with an invalid content type
        with self.assertRaises(NotImplementedError):
            self.model_service._decode_data(b"test_bytes", bedrock_content_type="application/xml")

        # Test decoding a stream of bytes with an invalid response format
        with self.assertRaisesRegex(RuntimeError, "Invalid response format"):
            self.model_service._decode_data(b'{"invalid": "response"}')

        # Test decoding a stream of bytes with an encoded data chunk
        decoded_data = self.model_service._decode_data(
            b'{"chunk": {"bytes": "eyJjb250ZW50IjogIkhlbGxvICJ9"}}', bedrock_content_type="application/json"
        )
        self.assertDictEqual(json.loads(base64.b64decode("eyJjb250ZW50IjogIkhlbGxvICJ9", validate=True)), decoded_data)

        # Test decoding a stream of bytes with an internal server exception
        with self.assertRaisesRegex(RuntimeError, "Internal server exception"):
            self.model_service._decode_data(
                b'{"internalServerException": {"message": "Internal server exception"}}',
                bedrock_content_type="application/json",
            )

        with self.assertRaisesRegex(RuntimeError, "model/call-time-model/invoke-with-response-stream"):
            self.model_service._decode_data(
                b'{"internalServerException": {"message": "Internal server exception"}}',
                request_url="https://bedrock/url/model/call-time-model/invoke-with-response-stream",
            )

        # Test decoding a stream of bytes with a model stream error
        with self.assertRaisesRegex(RuntimeError, "Model stream error"):
            self.model_service._decode_data(
                b'{"modelStreamErrorException": "Model stream error"}', bedrock_content_type="application/json"
            )

        # Test decoding a stream of bytes with a validation exception
        with self.assertRaisesRegex(RuntimeError, "Validation exception"):
            self.model_service._decode_data(
                b'{"validationException": {"message": "Validation exception"}}', bedrock_content_type="application/json"
            )

        # Test decoding a stream of bytes with a throttling exception
        with self.assertRaisesRegex(RuntimeError, "Throttling exception"):
            self.model_service._decode_data(
                b'{"throttlingException": {"message": "Throttling exception"}}', bedrock_content_type="application/json"
            )

        # Test decoding a stream of bytes with a model timeout exception
        with self.assertRaisesRegex(RuntimeError, "Model timeout exception"):
            self.model_service._decode_data(
                b'{"modelTimeoutException": {"message": "Model timeout exception"}}',
                bedrock_content_type="application/json",
            )

        # Test decoding a stream of bytes with a service unavailable exception
        with self.assertRaisesRegex(RuntimeError, "Service unavailable exception"):
            self.model_service._decode_data(
                b'{"serviceUnavailableException": {"message": "Service unavailable exception"}}',
                bedrock_content_type="application/json",
            )

    @patch("httpx.Client")
    def test_build_request_with_system_prompt(self, mock_client):
        messages = [ChatMessage.from_system("You are a helpful AI bot."), ChatMessage.from_user("Hello")]

        expected_json_body = {
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 8192,
            "system": "You are a helpful AI bot.",
            "anthropic_version": "bedrock-2023-05-31",
        }

        expected_headers = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
            "Authorization": "Bearer test_api_key",
        }

        self.model_service.build_request(mock_client, messages, model="claude-3-5-sonnet", stream=False)
        mock_client.build_request.assert_called_once_with(
            "POST",
            url="https://bedrock-anthropic/url/model/claude-3-5-sonnet/invoke",
            json=expected_json_body,
            headers=expected_headers,
            timeout=None,
        )

        self.model_service.build_request(mock_client, messages, model="claude-3-5-sonnet", stream=True)
        mock_client.build_request.assert_called_with(
            "POST",
            url="https://bedrock-anthropic/url/model/claude-3-5-sonnet/invoke-with-response-stream",
            json=expected_json_body,
            headers=expected_headers,
            timeout=None,
        )

    @patch("httpx.Client")
    def test_build_request_with_tools(self, mock_client):
        single_message = [ChatMessage.from_user("Hello")]
        mock_tool = Mock(autospec=Tool)
        self.model_service.build_request(
            mock_client,
            single_message,
            model="claude-3-5-sonnet",
            stream=False,
            tools=[mock_tool],
            inference_kwargs={"max_tokens": 4096},
        )

        expected_json_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 4096,
            "tools": [
                {"name": mock_tool.name, "description": mock_tool.description, "input_schema": mock_tool.parameters}
            ],
        }
        expected_headers = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
            "Authorization": "Bearer test_api_key",
        }
        mock_client.build_request.assert_called_with(
            "POST",
            url="https://bedrock-anthropic/url/model/claude-3-5-sonnet/invoke",
            json=expected_json_body,
            headers=expected_headers,
            timeout=None,
        )

    @patch("httpx.Client")
    def test_build_request_uses_call_time_model_for_url_and_default_max_tokens(self, mock_client):
        self.model_service.build_request(
            mock_client,
            [ChatMessage.from_user("Hello")],
            model="claude-opus-4",
        )

        mock_client.build_request.assert_called_once()
        self.assertEqual(
            "https://bedrock-anthropic/url/model/claude-opus-4/invoke",
            mock_client.build_request.call_args.kwargs["url"],
        )
        self.assertEqual(32000, mock_client.build_request.call_args.kwargs["json"]["max_tokens"])

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
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 8192,
            "anthropic_version": "bedrock-2023-05-31",
        }

        expected_headers = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
            "X-Custom-Header": "custom_value",
            "Authorization": "Bearer custom_token",
        }

        mock_client.build_request.assert_called_once_with(
            "POST",
            url="https://bedrock-anthropic/url/model/claude-3-5-sonnet/invoke",
            json=expected_json_body,
            headers=expected_headers,
            timeout=None,
        )

    @patch("ravelin.assets.model_services.chat_model_services._base.get_component_context")
    @patch(
        "ravelin.assets.model_services.chat_model_services.amazon_bedrock."
        "BedrockAnthropicChatModelService._check_stop_reason"
    )
    @patch(
        "ravelin.assets.model_services.chat_model_services.amazon_bedrock."
        "BedrockAnthropicChatModelService._connect_chunks"
    )
    @patch(
        "ravelin.assets.model_services.chat_model_services.amazon_bedrock."
        "BedrockAnthropicChatModelService._build_chunk_from_data"
    )
    @patch(
        "ravelin.assets.model_services.chat_model_services.amazon_bedrock."
        "BedrockAnthropicChatModelService._decode_data"
    )
    def test_parse_response_stream(
        self,
        mock_decode_data,
        mock_build_chunk_from_data,
        mock_connect_chunks,
        mock_check_stop_reason,
        mock_get_comp_ctx,
    ):
        event_streams = [f'{{"chunk": {{"bytes": "base_64_encoded_bytes_{idx}"}}}}' for idx in range(4)]
        streaming_chunks = [
            StreamingChunk(content="", meta={"index": 0}),
            StreamingChunk(content="Hi", meta={"index": 0}),
            StreamingChunk(content="!", meta={"index": 0}),
            StreamingChunk(content="", meta={"index": 0}),
        ]
        streaming_chunks_iter = iter(streaming_chunks)

        # Set up mocks
        response = Mock()
        response.iter_lines.return_value = iter(event_streams)
        response.headers.get.return_value = "application/json"
        mock_decode_data.side_effect = lambda event_stream, bedrock_content_type, request_url: json.loads(event_stream)
        mock_build_chunk_from_data.side_effect = lambda decoded_data: next(streaming_chunks_iter)
        mock_comp_ctx = Mock()
        mock_get_comp_ctx.return_value = mock_comp_ctx

        chat_messages = self.model_service.parse_response(response, stream=True)

        # Check mock calls
        self.assertListEqual(
            [
                call(
                    event_stream,
                    bedrock_content_type="application/json",
                    request_url=self.model_service.stream_url,
                )
                for event_stream in event_streams
            ],
            mock_decode_data.call_args_list,
        )
        self.assertListEqual(
            [call(json.loads(event_stream)) for event_stream in event_streams],
            mock_build_chunk_from_data.call_args_list,
        )
        # Lifecycle events live on infer, not parse_response — collector emits one DELTA per chunk.
        self.assertEqual(len(streaming_chunks), mock_comp_ctx.emit.call_count)
        for chunk in streaming_chunks:
            mock_comp_ctx.emit.assert_any_call(
                EventType.TOKEN_STREAM, chunk, subtype=GENERATION_DELTA, generation_id=None
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
        self.assertEqual([TextContent("Hi! My name is Claude.")], chat_messages[0].content)
        self.assertEqual("assistant", chat_messages[0].role)
        self.assertEqual("claude-3-5-sonnet-20240620", chat_messages[0].meta["model"])
        self.assertEqual("msg_013Zva2CMHLNnXjNJJKqJ2EF", chat_messages[0].meta["id"])
        self.assertEqual("end_turn", chat_messages[0].meta["stop_reason"])
        self.assertEqual(None, chat_messages[0].meta["stop_sequence"])
        self.assertEqual({"input_tokens": 2095, "output_tokens": 503}, chat_messages[0].meta["usage"])


class TestBedrockAnthropicChatModelServiceAsync(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.model_service = BedrockAnthropicChatModelService(
            name="test_bedrock_anthropic_model_service",
            model="claude-3-5-sonnet",
            base_url="https://bedrock-anthropic/url/",
            api_key=Secret.from_token("test_api_key"),
        )

    @patch("ravelin.assets.model_services.chat_model_services._base.get_component_context")
    @patch(
        "ravelin.assets.model_services.chat_model_services.amazon_bedrock."
        "BedrockAnthropicChatModelService._check_stop_reason"
    )
    @patch(
        "ravelin.assets.model_services.chat_model_services.amazon_bedrock."
        "BedrockAnthropicChatModelService._connect_chunks"
    )
    @patch(
        "ravelin.assets.model_services.chat_model_services.amazon_bedrock."
        "BedrockAnthropicChatModelService._build_chunk_from_data"
    )
    @patch(
        "ravelin.assets.model_services.chat_model_services.amazon_bedrock."
        "BedrockAnthropicChatModelService._decode_data"
    )
    async def test_aparse_response_stream(
        self,
        mock_decode_data,
        mock_build_chunk_from_data,
        mock_connect_chunks,
        mock_check_stop_reason,
        mock_get_comp_ctx,
    ):
        event_streams = [f'{{"chunk": {{"bytes": "base_64_encoded_bytes_{idx}"}}}}' for idx in range(4)]
        streaming_chunks = [
            StreamingChunk(content="", meta={"index": 0}),
            StreamingChunk(content="Hi", meta={"index": 0}),
            StreamingChunk(content="!", meta={"index": 0}),
            StreamingChunk(content="", meta={"index": 0}),
        ]
        streaming_chunks_iter = iter(streaming_chunks)

        # Create an async generator function
        async def async_event_stream_generator():
            for item in event_streams:
                yield item

        # Set up mocks
        response = Mock()
        response.aiter_lines.return_value = async_event_stream_generator()
        response.headers.get.return_value = "application/json"
        mock_decode_data.side_effect = lambda event_stream, bedrock_content_type, request_url: json.loads(event_stream)
        mock_build_chunk_from_data.side_effect = lambda decoded_data: next(streaming_chunks_iter)
        mock_comp_ctx = Mock()
        mock_comp_ctx.emit_async = AsyncMock()
        mock_get_comp_ctx.return_value = mock_comp_ctx

        chat_messages = await self.model_service.aparse_response(response, stream=True)

        # Check mock calls
        self.assertListEqual(
            [
                call(
                    event_stream,
                    bedrock_content_type="application/json",
                    request_url=self.model_service.stream_url,
                )
                for event_stream in event_streams
            ],
            mock_decode_data.call_args_list,
        )
        self.assertListEqual(
            [call(json.loads(event_stream)) for event_stream in event_streams],
            mock_build_chunk_from_data.call_args_list,
        )
        # Lifecycle events live on ainfer — collector emits one DELTA per chunk.
        self.assertEqual(len(streaming_chunks), mock_comp_ctx.emit_async.call_count)
        for chunk in streaming_chunks:
            mock_comp_ctx.emit_async.assert_any_call(
                EventType.TOKEN_STREAM, chunk, subtype=GENERATION_DELTA, generation_id=None
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
        self.assertEqual([TextContent("Hi! My name is Claude.")], chat_messages[0].content)
        self.assertEqual("assistant", chat_messages[0].role)
        self.assertEqual("claude-3-5-sonnet-20240620", chat_messages[0].meta["model"])
        self.assertEqual("msg_013Zva2CMHLNnXjNJJKqJ2EF", chat_messages[0].meta["id"])
        self.assertEqual("end_turn", chat_messages[0].meta["stop_reason"])
        self.assertEqual(None, chat_messages[0].meta["stop_sequence"])
        self.assertEqual({"input_tokens": 2095, "output_tokens": 503}, chat_messages[0].meta["usage"])


if __name__ == "__main__":
    unittest.main()
