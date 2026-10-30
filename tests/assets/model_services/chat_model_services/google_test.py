import unittest
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import base64

from haystack.tools import Tool
from haystack.dataclasses import ToolCallDelta
from haystack.utils import Secret
from httpx import Response

from ravelin.streaming import EventType
from ravelin.streaming.event import GENERATION_DELTA
from ravelin.assets.model_services.chat_model_services.google import GeminiChatModelService
from ravelin.dataclasses import ByteStream, ChatMessage, ChatRole, StreamingChunk
from ravelin.dataclasses.chat_message import ToolCall, TextContent, MediaContent
from ravelin.utils.streaming import ServerSentEvent


class TestGeminiChatModelService(unittest.TestCase):
    def setUp(self):
        self.model_service = GeminiChatModelService(name="test_gemini_service", base_url="test_uri")
        self.test_image = ByteStream(data=b"test image data", mime_type="image/jpeg")
        self.test_video = ByteStream(
            data=b"test video data", mime_type="video/mp4", meta={"startOffset": 10, "endOffset": 30}
        )
        self.sse_content_usage = [
            {
                "promptTokenCount": 28,
                "candidatesTokenCount": 7,
                "totalTokenCount": 35,
                "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 28}],
                "thoughtsTokenCount": 6,
            },
            {
                "promptTokenCount": 28,
                "candidatesTokenCount": 9,
                "totalTokenCount": 37,
                "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 28}],
                "thoughtsTokenCount": 6,
            },
            {
                "promptTokenCount": 28,
                "candidatesTokenCount": 14,
                "totalTokenCount": 42,
                "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 28}],
                "thoughtsTokenCount": 6,
            },
            {
                "promptTokenCount": 28,
                "candidatesTokenCount": 16,
                "totalTokenCount": 44,
                "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 28}],
                "thoughtsTokenCount": 6,
            },
            {
                "promptTokenCount": 28,
                "candidatesTokenCount": 18,
                "totalTokenCount": 46,
                "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 28}],
                "thoughtsTokenCount": 6,
            },
        ]
        # Create specific SSE contents for different test cases
        self.sse_contents = [
            # Reasoning content
            ServerSentEvent(
                data=json.dumps(
                    {
                        "candidates": [
                            {
                                "content": {
                                    "parts": [{"text": "Hello"}, {"thought": True, "text": "I should be friendly"}]
                                },
                                "finishReason": None,
                                "index": 0,
                            }
                        ],
                        "modelVersion": "gemini-2.5-pro-preview-03-25",
                        "usageMetadata": self.sse_content_usage[0],
                    }
                )
            ),
            # Regular text content
            ServerSentEvent(
                data=json.dumps(
                    {
                        "candidates": [{"content": {"parts": [{"text": " World!"}]}, "finishReason": None, "index": 0}],
                        "responseId": "resp123",
                        "modelVersion": "gemini-2.5-pro-preview-03-25",
                        "usageMetadata": self.sse_content_usage[1],
                    }
                )
            ),
            # Tool call with signature
            ServerSentEvent(
                data=json.dumps(
                    {
                        "candidates": [
                            {
                                "content": {
                                    "parts": [
                                        {"functionCall": {"name": "test_tool", "args": {"foo": "bar"}}},
                                    ]
                                },
                                "finishReason": None,
                                "index": 0,
                            }
                        ],
                        "modelVersion": "gemini-2.5-pro-preview-03-25",
                        "usageMetadata": self.sse_content_usage[2],
                    }
                )
            ),
            # Text with signature in same part
            ServerSentEvent(
                data=json.dumps(
                    {
                        "candidates": [
                            {
                                "content": {
                                    "parts": [
                                        {"text": "Final answer.", "thoughtSignature": "789012"},
                                    ]
                                },
                                "finishReason": "STOP",
                                "index": 0,
                            }
                        ],
                        "modelVersion": "gemini-2.5-pro-preview-03-25",
                        "usageMetadata": self.sse_content_usage[3],
                    }
                )
            ),
            # Unsupported content type
            ServerSentEvent(
                data=json.dumps(
                    {
                        "candidates": [
                            {
                                "content": {"parts": [{"unsupported_type": "some content"}]},
                                "finishReason": "STOP",
                                "index": 0,
                            }
                        ],
                        "modelVersion": "gemini-2.5-pro-preview-03-25",
                        "usageMetadata": self.sse_content_usage[4],
                    }
                )
            ),
            # Unsupported content type
            ServerSentEvent(
                data=json.dumps(
                    {
                        "candidates": [
                            {
                                "content": {"parts": [{"unsupported_type": "some content"}]},
                                "finishReason": "STOP",
                                "index": 0,
                            }
                        ],
                        "modelVersion": "gemini-2.5-pro-preview-03-25",
                        "usageMetadata": self.sse_content_usage[3],
                    }
                )
            ),
        ]
        self.streaming_chunks = [
            StreamingChunk(
                content="",
                reasoning_content="My reasoning",
                meta={"model": "gemini-pro", "usage": self.sse_content_usage[0]},
                index=0,
                start=True,
            ),
            StreamingChunk(
                content="Hello",
                meta={"model": "gemini-pro", "usage": self.sse_content_usage[1]},
                index=0,
            ),
            StreamingChunk(
                content=" world",
                meta={"model": "gemini-pro", "usage": self.sse_content_usage[2]},
                index=0,
            ),
            StreamingChunk(
                content="",  # Add empty content string for tool-call-only chunk
                tool_calls=[
                    ToolCallDelta(
                        index=0,
                        id=None,  # Gemini doesn't provide IDs
                        tool_name="test_tool",
                        arguments=json.dumps({"foo": "bar"}),
                    )
                ],
                meta={"model": "gemini-pro", "usage": self.sse_content_usage[3]},
                index=0,
                finish_reason="STOP",
            ),
        ]

    def test_format_chat_message_simple(self):
        message = ChatMessage.from_user(text="Hello")
        expected = {"role": "user", "parts": [{"text": "Hello"}]}
        self.assertEqual(expected, self.model_service.format_chat_message(message))

        # Assistant messages should have role "model" in Gemini format
        message = ChatMessage.from_assistant(text="Hello")
        expected = {"role": "model", "parts": [{"text": "Hello"}]}
        self.assertEqual(expected, self.model_service.format_chat_message(message))

    @patch("ravelin.assets.model_services.chat_model_services.google.logger.warning")
    def test_format_chat_message_system(self, mock_warning):
        # System messages should trigger a warning and fall back to user role
        message = ChatMessage.from_system(text="You are a helpful assistant")
        result = self.model_service.format_chat_message(message)

        # Check that it falls back to user role
        self.assertEqual("user", result["role"])
        self.assertEqual([{"text": "You are a helpful assistant"}], result["parts"])

        # Check that warning was logged
        mock_warning.assert_called_once_with(
            "The role %s is not supported by the Gemini API. Falling back to user role.", ChatRole.SYSTEM
        )

    def test_format_chat_message_with_media(self):
        message = ChatMessage.from_user(text="Image:", media=[self.test_image])
        result = self.model_service.format_chat_message(message)

        self.assertEqual("user", result["role"])
        self.assertEqual(2, len(result["parts"]))
        self.assertEqual("image/jpeg", result["parts"][0]["inlineData"]["mimeType"])
        self.assertEqual(base64.b64encode(b"test image data").decode("utf-8"), result["parts"][0]["inlineData"]["data"])
        self.assertEqual({"text": "Image:"}, result["parts"][1])

    def test_format_chat_message_with_video_metadata(self):
        message = ChatMessage.from_user(text="Video:", media=[self.test_video])
        result = self.model_service.format_chat_message(message)

        self.assertEqual("user", result["role"])
        self.assertEqual(2, len(result["parts"]))

        video_part = result["parts"][0]
        self.assertEqual("video/mp4", video_part["inlineData"]["mimeType"])
        self.assertEqual({"startOffset": 10, "endOffset": 30}, video_part["videoMetadata"])

    def test_format_chat_message_unsupported_media(self):
        unsupported_media = ByteStream(data=b"test", mime_type="unsupported/type")
        message = ChatMessage.from_user(text="Test", media=[unsupported_media])

        with self.assertRaisesRegex(ValueError, "Unsupported media type 'unsupported/type'"):
            self.model_service.format_chat_message(message)

    def test_format_chat_message_tool_call(self):
        tool_call = ToolCall(tool_name="weather", arguments={"location": "New York"})
        message = ChatMessage.from_assistant(tool_calls=[tool_call])

        result = self.model_service.format_chat_message(message)
        self.assertEqual(1, len(result["parts"]))
        self.assertEqual({"name": "weather", "args": {"location": "New York"}}, result["parts"][0]["functionCall"])

    def test_format_chat_message_tool_result(self):
        tool_call = ToolCall(tool_name="weather", arguments={"location": "New York"})
        message = ChatMessage.from_tool("Sunny and 75°F", origin=tool_call)

        result = self.model_service.format_chat_message(message)
        self.assertNotIn("role", result)  # Tool messages don't have a role in Gemini
        self.assertEqual(1, len(result["parts"]))
        self.assertEqual("weather", result["parts"][0]["functionResponse"]["name"])
        self.assertEqual({"output": "Sunny and 75°F"}, result["parts"][0]["functionResponse"]["response"])

    def test_format_chat_message_tool_error(self):
        tool_call = ToolCall(tool_name="weather", arguments={"location": "New York"})
        message = ChatMessage.from_tool("API error", origin=tool_call, error=True)

        result = self.model_service.format_chat_message(message)
        self.assertEqual({"error": "API error", "output": "N/A"}, result["parts"][0]["functionResponse"]["response"])

    def test_format_chat_message_simple_with_signature(self):
        message = ChatMessage.from_assistant("Hello! How may I help you today?", meta={"thoughtSignature": "123456"})

        result = self.model_service.format_chat_message(message)
        self.assertEqual(1, len(result["parts"]))
        self.assertEqual(2, len(result["parts"][0]))
        self.assertEqual("Hello! How may I help you today?", result["parts"][0]["text"])
        self.assertEqual("123456", result["parts"][0]["thoughtSignature"])

    def test_format_chat_message_multiple_tool_calls_with_signature(self):
        tool_call_1 = ToolCall(tool_name="weather", arguments={"location": "New York"})
        tool_call_2 = ToolCall(tool_name="weather", arguments={"location": "Berlin"})
        message = ChatMessage.from_assistant(
            "I need to fetch weather data for both New York and Berlin",
            tool_calls=[tool_call_1, tool_call_2],
            meta={"thoughtSignature": "123456"},
        )

        result = self.model_service.format_chat_message(message)
        self.assertEqual(3, len(result["parts"]))  # text and two function calls

        # Text block
        self.assertEqual(1, len(result["parts"][0]))
        self.assertIn("text", result["parts"][0])

        # Tool call #1 with signature
        self.assertEqual(2, len(result["parts"][1]))
        self.assertIn("functionCall", result["parts"][1])
        self.assertEqual("123456", result["parts"][1]["thoughtSignature"])

        # Tool call #2 without signature
        self.assertEqual(1, len(result["parts"][2]))
        self.assertIn("functionCall", result["parts"][2])

    def test_extract_candidate_contents(self):
        candidate = {
            "content": {
                "parts": [
                    {"text": "Hello world"},
                    {"thought": True, "text": "thoughts here"},
                    {"text": "<End of content>", "thoughtSignature": "123456"},
                    {"functionCall": {"name": "test_tool", "args": {"foo": "bar"}}},
                    {"unsupported_part": "something"},
                ]
            }
        }

        contents = list(self.model_service._extract_candidate_contents(candidate))

        expected = [
            {"text": "Hello world"},
            {"thought": "thoughts here"},
            {"text": "<End of content>", "thoughtSignature": "123456"},
            {"functionCall": {"name": "test_tool", "args": {"foo": "bar"}}},
        ]
        self.assertEqual(expected, contents)

    def test_extract_candidate_meta(self):
        candidate = {
            "index": 1,
            "finishReason": "STOP",
            "safetyRatings": [{"category": "HARM", "probability": "LOW"}],
            "irrelevant_field": "something",
        }

        meta = self.model_service._extract_candidate_meta(candidate)

        expected = {"index": 1, "finish_reason": "STOP", "safetyRatings": [{"category": "HARM", "probability": "LOW"}]}
        self.assertEqual(expected, meta)

    @patch("ravelin.assets.model_services.chat_model_services.google.logger.warning")
    def test_build_chunks(self, mock_warning):
        test_cases = [
            {
                "test_name": "reasoning_content",
                "sse_index": 0,
                "expected_content": "Hello",
                "expected_reasoning": "I should be friendly",
                "expected_tool_calls": [],
                "expected_model": "gemini-2.5-pro-preview-03-25",
                "expected_msg_id": None,
                "expect_warning": False,
            },
            {
                "test_name": "text_content",
                "sse_index": 1,
                "expected_content": " World!",
                "expected_reasoning": None,
                "expected_tool_calls": [],
                "expected_model": "gemini-2.5-pro-preview-03-25",
                "expected_msg_id": "resp123",
                "expect_warning": False,
            },
            {
                "test_name": "tool_call",
                "sse_index": 2,
                "expected_content": "",
                "expected_reasoning": None,
                "expected_tool_calls": [{"name": "test_tool", "args": {"foo": "bar"}}],
                "expected_model": "gemini-2.5-pro-preview-03-25",
                "expected_msg_id": None,
                "expect_warning": False,
            },
            {
                "test_name": "text_with_signature",
                "sse_index": 3,
                "expected_content": "Final answer.",
                "expected_reasoning": None,
                "expected_tool_calls": [],
                "expected_model": "gemini-2.5-pro-preview-03-25",
                "expected_msg_id": None,
                "expect_warning": False,
                "expected_finish_reason": "STOP",
                "expected_signature": "789012",
            },
            {
                "test_name": "unsupported_content",
                "sse_index": 4,
                "expected_content": "",
                "expected_reasoning": None,
                "expected_tool_calls": [],
                "expected_model": "gemini-2.5-pro-preview-03-25",
                "expected_msg_id": None,
                "expect_warning": True,
                "expected_finish_reason": "STOP",
            },
        ]

        for tc in test_cases:
            with self.subTest(name=tc["test_name"]):
                # Reset mock between subtests
                mock_warning.reset_mock()

                # Get the SSE for this test case
                sse = self.sse_contents[tc["sse_index"]]

                # Test chunk building
                chunks = list(self.model_service._build_chunks(sse))
                self.assertEqual(1, len(chunks), f"Should have 1 chunk for {tc['test_name']}")

                chunk = chunks[0]
                self.assertEqual(tc["expected_content"], chunk.content, f"Content mismatch for {tc['test_name']}")
                self.assertEqual(
                    tc["expected_reasoning"], chunk.reasoning_content, f"Reasoning mismatch for {tc['test_name']}"
                )
                # For tool calls, we need to handle ToolCallDelta format differently
                if tc["test_name"] == "tool_call":
                    self.assertIsNotNone(chunk.tool_calls, f"Expected tool_calls to be present for {tc['test_name']}")
                    self.assertEqual(1, len(chunk.tool_calls), f"Expected one tool call for {tc['test_name']}")
                    self.assertEqual(
                        "test_tool", chunk.tool_calls[0].tool_name, f"Tool name mismatch for {tc['test_name']}"
                    )
                    self.assertEqual(
                        '{"foo": "bar"}', chunk.tool_calls[0].arguments, f"Tool args mismatch for {tc['test_name']}"
                    )
                else:
                    self.assertIsNone(chunk.tool_calls, f"Expected tool_calls to be None for {tc['test_name']}")
                self.assertEqual(tc["expected_model"], chunk.meta["model"], f"Model mismatch for {tc['test_name']}")

                # Check msg_id if expected
                if tc["expected_msg_id"]:
                    self.assertEqual(
                        tc["expected_msg_id"], chunk.meta["msg_id"], f"msg_id mismatch for {tc['test_name']}"
                    )

                # Check finish reason if expected
                if "expected_finish_reason" in tc:
                    # For test purposes, verify that the finish_reason is in the chunk or the meta
                    actual_finish_reason = chunk.finish_reason or chunk.meta.get("finish_reason")
                    self.assertEqual(
                        tc["expected_finish_reason"],
                        actual_finish_reason,
                        f"finish_reason mismatch for {tc['test_name']}",
                    )

                # Check usage metadata
                self.assertEqual(
                    self.sse_content_usage[tc["sse_index"]],
                    chunk.meta["usage"],
                    f"Usage metadata mismatch for {tc['test_name']}",
                )

                # Check signature if expected
                if "expected_signature" in tc:
                    self.assertEqual(
                        tc["expected_signature"],
                        chunk.meta.get("thoughtSignature"),
                        f"thoughtSignature mismatch for {tc['test_name']}",
                    )

                # Check warning behavior
                if tc["expect_warning"]:
                    mock_warning.assert_called_once()
                    self.assertIn("Unsupported content type", mock_warning.call_args[0][0])
                else:
                    mock_warning.assert_not_called()

    def test_connect_chunks(self):
        messages = self.model_service._connect_chunks(self.streaming_chunks)

        self.assertEqual(1, len(messages))
        message = messages[0]
        self.assertEqual(ChatRole.ASSISTANT, message.role)
        self.assertEqual("Hello world", message.text)
        self.assertEqual("My reasoning", message.reasoning_content)
        self.assertEqual("gemini-pro", message.meta["model"])
        self.assertEqual("STOP", message.meta["finish_reason"])
        self.assertEqual(0, message.meta["index"])
        # The last chunk has finish_reason, so its usage is the final one (index 3)
        self.assertEqual(self.sse_content_usage[3], message.meta["usage"])

        # Check if tool call was added
        tool_calls = [c for c in message.content if isinstance(c, ToolCall)]
        self.assertEqual(1, len(tool_calls))
        self.assertEqual("test_tool", tool_calls[0].tool_name)
        self.assertEqual({"foo": "bar"}, tool_calls[0].arguments)

    @patch("ravelin.assets.model_services.chat_model_services.google.logger.warning")
    def test_check_finish_reason(self, mock_warning):
        # Test normal completion
        message = ChatMessage.from_assistant(text="Hello", meta={"index": 0, "finish_reason": "STOP"})
        self.model_service._check_finish_reason(message)
        mock_warning.assert_not_called()

        # Test truncation
        message = ChatMessage.from_assistant(text="Hello", meta={"index": 1, "finish_reason": "MAX_TOKENS"})
        self.model_service._check_finish_reason(message)
        self.assertIn("truncated before reaching a natural stopping point", mock_warning.call_args[0][0])

        # Test safety filter
        message = ChatMessage.from_assistant(
            text="Hello", meta={"index": 2, "finish_reason": "SAFETY", "safetyRatings": []}
        )
        self.model_service._check_finish_reason(message)
        self.assertIn("stopped due to safety filters", mock_warning.call_args[0][0])

    @patch(
        "ravelin.assets.model_services.chat_model_services.google." "GeminiChatModelService._extract_candidate_contents"
    )
    @patch("ravelin.assets.model_services.chat_model_services.google.GeminiChatModelService._extract_candidate_meta")
    def test_build_message(self, mock_extract_meta, mock_extract_contents):
        # Setup mocked return values
        mock_contents = [
            {"text": "Hello world"},
            {"thought": "I should be helpful"},
            {"functionCall": {"name": "weather", "args": {"location": "Seattle"}}},
            {"inlineData": {"mimeType": "image/jpeg", "data": base64.b64encode(b"test_image_data").decode("utf-8")}},
        ]
        mock_extract_contents.return_value = mock_contents

        mock_meta = {
            "index": 2,
            "finish_reason": "STOP",
            "safetyRatings": [{"category": "HARM", "probability": "NEGLIGIBLE"}],
        }
        mock_extract_meta.return_value = mock_meta

        # Test parameters
        candidate = {"some": "candidate data"}  # Actual content doesn't matter due to mocking
        model = "gemini-pro-1.5"
        msg_id = "response_123456"
        usage = {"promptTokenCount": 15, "candidatesTokenCount": 25, "totalTokenCount": 40}

        # Call the method
        message = self.model_service._build_message(candidate=candidate, model=model, msg_id=msg_id, usage=usage)

        # Verify extraction methods were called
        mock_extract_contents.assert_called_once_with(candidate)
        mock_extract_meta.assert_called_once_with(candidate)

        # Verify basic properties
        self.assertEqual(ChatRole.ASSISTANT, message.role)
        self.assertEqual("Hello world", message.text)
        self.assertEqual("I should be helpful", message.reasoning_content)

        # Verify metadata
        self.assertEqual(model, message.meta["model"])
        self.assertEqual(msg_id, message.meta["msg_id"])
        self.assertEqual(usage, message.meta["usage"])
        self.assertEqual("STOP", message.meta["finish_reason"])
        self.assertEqual(2, message.meta["index"])
        self.assertIn("safetyRatings", message.meta)

        # Verify content types and order
        self.assertEqual(3, len(message.content))

        # Check text content
        self.assertIsInstance(message.content[0], TextContent)
        self.assertEqual("Hello world", message.content[0].text)

        # Check tool call
        self.assertIsInstance(message.content[1], ToolCall)
        self.assertEqual("weather", message.content[1].tool_name)
        self.assertEqual({"location": "Seattle"}, message.content[1].arguments)

        # Check media content
        self.assertIsInstance(message.content[2], MediaContent)
        self.assertEqual("image/jpeg", message.content[2].media.mime_type)
        self.assertEqual(b"test_image_data", message.content[2].media.data)

        # Test with minimal parameters (but still including usage)
        # Reset mocks with simpler values
        mock_extract_contents.reset_mock()
        mock_extract_meta.reset_mock()

        mock_extract_contents.return_value = [{"text": "Simple text only"}]
        mock_extract_meta.return_value = {"index": 0}
        minimal_usage = {"promptTokenCount": 5, "candidatesTokenCount": 3, "totalTokenCount": 8}

        minimal_message = self.model_service._build_message(
            candidate={"different": "candidate"}, model="gemini-pro", usage=minimal_usage
        )

        self.assertEqual("Simple text only", minimal_message.text)
        self.assertEqual("gemini-pro", minimal_message.meta["model"])
        self.assertEqual(minimal_usage, minimal_message.meta["usage"])
        self.assertNotIn("msg_id", minimal_message.meta)

    def test_build_message_with_signature(self):
        """Test that _build_message correctly handles thoughtSignature in extracted content."""
        candidate = {
            "content": {
                "parts": [
                    {"text": "Final answer with signature", "thoughtSignature": "abc123"},
                ]
            },
            "finishReason": "STOP",
            "index": 0,
        }

        message = self.model_service._build_message(
            candidate=candidate,
            model="gemini-pro",
            msg_id="msg_123",
            usage={"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15},
        )

        # Verify the text content was extracted
        self.assertEqual("Final answer with signature", message.text)

        # Verify the thoughtSignature is in meta
        self.assertEqual("abc123", message.meta.get("thoughtSignature"))

        # Verify other metadata
        self.assertEqual("gemini-pro", message.meta["model"])
        self.assertEqual("msg_123", message.meta["msg_id"])
        self.assertEqual("STOP", message.meta["finish_reason"])

    def test_build_request_basic(self):
        client = Mock()
        messages = [ChatMessage.from_user(text="Hello")]

        self.model_service.build_request(client, messages, model="gemini-pro")

        client.build_request.assert_called_once()
        call_args = client.build_request.call_args[1]

        self.assertEqual("POST", call_args["method"])
        self.assertEqual("test_uri/v1beta/models/gemini-pro:generateContent", call_args["url"])
        self.assertEqual([{"role": "user", "parts": [{"text": "Hello"}]}], call_args["json"]["contents"])

    def test_api_key_uses_header_not_query_parameter(self):
        client = Mock()
        service = GeminiChatModelService(
            name="test_gemini_service",
            base_url="https://generativelanguage.googleapis.com",
            api_key=Secret.from_token("test_api_key"),
        )

        service.build_request(client, [ChatMessage.from_user(text="Hello")], model="models/gemini-pro")

        request_kwargs = client.build_request.call_args.kwargs
        self.assertEqual("test_api_key", request_kwargs["headers"]["x-goog-api-key"])
        self.assertNotIn("key", request_kwargs["params"])
        self.assertNotIn("test_api_key", request_kwargs["url"])

    def test_build_request_with_system(self):
        client = Mock()
        messages = [ChatMessage.from_system(text="You are a helpful AI"), ChatMessage.from_user(text="Hello")]

        self.model_service.build_request(client, messages, model="gemini-pro")

        json_body = client.build_request.call_args[1]["json"]
        self.assertEqual([{"text": "You are a helpful AI"}], json_body["systemInstruction"]["parts"])
        self.assertEqual([{"role": "user", "parts": [{"text": "Hello"}]}], json_body["contents"])

    def test_build_request_with_tools(self):
        client = Mock()
        messages = [ChatMessage.from_user(text="Hello")]

        tool = MagicMock(spec=Tool)
        tool.tool_spec = {"name": "weather", "description": "Get weather", "parameters": {}}

        self.model_service.build_request(client, messages, model="gemini-pro", tools=[tool])

        json_body = client.build_request.call_args[1]["json"]
        self.assertEqual([{"functionDeclarations": [tool.tool_spec]}], json_body["tools"])

    def test_build_request_with_extra_headers(self):
        """Test that extra_headers are properly merged into the request headers."""
        client = Mock()
        messages = [ChatMessage.from_user(text="Hello")]
        extra_headers = {"X-Custom-Header": "custom_value", "Authorization": "Bearer custom_token"}

        self.model_service.build_request(
            client, messages, model="gemini-pro", stream=False, extra_headers=extra_headers
        )

        call_args = client.build_request.call_args[1]
        expected_headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer custom_token",  # extra_headers should override
            "X-Custom-Header": "custom_value",
        }

        self.assertEqual(expected_headers, call_args["headers"])

    def test_build_request_with_inference_kwargs(self):
        client = Mock()
        messages = [ChatMessage.from_user(text="Hello")]

        self.model_service.build_request(
            client,
            messages,
            model="gemini-pro",
            inference_kwargs={"temperature": 0.7, "max_tokens": 100, "top_p": 0.95},
        )

        json_body = client.build_request.call_args[1]["json"]
        self.assertEqual({"temperature": 0.7, "maxOutputTokens": 100, "topP": 0.95}, json_body["generationConfig"])

    @patch("ravelin.assets.model_services.chat_model_services._base.get_component_context")
    @patch("ravelin.assets.model_services.chat_model_services.google.ServerSentEventStream")
    @patch("ravelin.assets.model_services.chat_model_services.google.GeminiChatModelService._build_chunks")
    @patch("ravelin.assets.model_services.chat_model_services.google.GeminiChatModelService._connect_chunks")
    def test_parse_response_stream(self, mock_connect, mock_build_chunks, mock_sse_stream, mock_get_comp_ctx):
        # Use only the first 4 SSEs that correspond to the streaming_chunks
        relevant_sse_contents = self.sse_contents[:4]

        # Set up SSE stream mock
        mock_sse_stream.return_value.__iter__.return_value = relevant_sse_contents

        # Create a lookup table for the side_effect
        # The streaming chunks are actually not correlated to the SSE contents, but we are not checking that here
        sse_to_chunk = {
            id(relevant_sse_contents[i]): [self.streaming_chunks[i]] for i in range(len(relevant_sse_contents))
        }

        # Define the side_effect function to use our lookup table
        def side_effect(sse):
            return sse_to_chunk.get(id(sse), [])

        mock_build_chunks.side_effect = side_effect

        mock_message = ChatMessage.from_assistant(text="Complete response")
        mock_connect.return_value = [mock_message]

        mock_comp_ctx = Mock()
        mock_get_comp_ctx.return_value = mock_comp_ctx

        result = self.model_service.parse_response(Response(200), stream=True)

        # Lifecycle events live on infer, not parse_response — collector emits one DELTA per chunk.
        self.assertEqual(len(self.streaming_chunks), mock_comp_ctx.emit.call_count)
        for chunk in self.streaming_chunks:
            mock_comp_ctx.emit.assert_any_call(
                EventType.TOKEN_STREAM, chunk, subtype=GENERATION_DELTA, generation_id=None
            )
        subtypes = [c.kwargs.get("subtype") for c in mock_comp_ctx.emit.call_args_list]
        self.assertEqual([GENERATION_DELTA] * len(self.streaming_chunks), subtypes)
        # Verify chunks were connected
        mock_connect.assert_called_once_with(self.streaming_chunks)
        self.assertEqual([mock_message], result)


class TestGeminiChatModelServiceAsync(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.model_service = GeminiChatModelService(name="test_gemini_service", base_url="test_uri")
        self.response_json = {
            "candidates": [{"content": {"parts": [{"text": "Hello world"}]}, "finishReason": "STOP", "index": 0}],
            "modelVersion": "gemini-pro",
            "responseId": "resp123",
        }

    async def test_aparse_response_non_stream(self):
        result = await self.model_service.aparse_response(Response(200, json=self.response_json))

        self.assertEqual(1, len(result))
        message = result[0]
        self.assertEqual("Hello world", message.text)
        self.assertEqual("gemini-pro", message.meta["model"])
        self.assertEqual("resp123", message.meta["msg_id"])
        self.assertEqual("STOP", message.meta["finish_reason"])

    @patch("ravelin.assets.model_services.chat_model_services._base.get_component_context")
    @patch("ravelin.assets.model_services.chat_model_services.google.AsyncServerSentEventStream")
    @patch("ravelin.assets.model_services.chat_model_services.google.GeminiChatModelService._build_chunks")
    @patch("ravelin.assets.model_services.chat_model_services.google.GeminiChatModelService._connect_chunks")
    async def test_aparse_response_stream(self, mock_connect, mock_build_chunks, mock_sse_stream, mock_get_comp_ctx):
        mock_sse = MagicMock()

        async def _async_sse_iter():
            yield mock_sse

        mock_sse_stream.return_value.__aiter__ = lambda self: _async_sse_iter()

        mock_chunk = MagicMock()
        mock_build_chunks.return_value = [mock_chunk]

        mock_message = MagicMock()
        mock_connect.return_value = [mock_message]

        mock_comp_ctx = Mock()
        mock_comp_ctx.emit_async = AsyncMock()
        mock_get_comp_ctx.return_value = mock_comp_ctx

        result = await self.model_service.aparse_response(Response(200), stream=True)

        mock_build_chunks.assert_called_once_with(mock_sse)
        # Lifecycle lives on ainfer — collector emits one DELTA per chunk.
        self.assertEqual(1, mock_comp_ctx.emit_async.call_count)
        mock_comp_ctx.emit_async.assert_any_call(
            EventType.TOKEN_STREAM, mock_chunk, subtype=GENERATION_DELTA, generation_id=None
        )
        mock_connect.assert_called_once_with([mock_chunk])
        self.assertEqual([mock_message], result)


class TestVertexAnthropicChatModelService(unittest.TestCase):
    def setUp(self):
        from ravelin.assets.model_services.chat_model_services.google import VertexAnthropicChatModelService

        self.model_resource = (
            "projects/test-project/locations/us-central1/publishers/anthropic/" "models/claude-sonnet-4@20250514"
        )
        self.model_service = VertexAnthropicChatModelService(
            name="test_vertex_anthropic_service",
            base_url="https://us-central1-aiplatform.googleapis.com",
            api_key=Secret.from_token("test_api_key"),
        )

    def test_initialization(self):
        """Test that VertexAnthropicChatModelService inherits from AnthropicChatModelService."""
        from ravelin.assets.model_services.chat_model_services.anthropic import AnthropicChatModelService

        self.assertIsInstance(self.model_service, AnthropicChatModelService)
        self.assertEqual(self.model_service.name, "test_vertex_anthropic_service")
        self.assertEqual(self.model_service.api_key.resolve_value(), "test_api_key")

    def test_stream_route(self):
        """Test that the stream route expands the Vertex model resource."""
        expected_url = (
            "https://us-central1-aiplatform.googleapis.com/v1/projects/test-project/locations/us-central1/"
            "publishers/anthropic/models/claude-sonnet-4@20250514:streamRawPredict"
        )
        self.assertEqual(self.model_service.build_request_url("stream", model=self.model_resource), expected_url)

    def test_inference_route(self):
        """Test that the inference route expands the Vertex model resource."""
        expected_url = (
            "https://us-central1-aiplatform.googleapis.com/v1/projects/test-project/locations/us-central1/"
            "publishers/anthropic/models/claude-sonnet-4@20250514:rawPredict"
        )
        self.assertEqual(self.model_service.build_request_url("inference", model=self.model_resource), expected_url)

    @patch("ravelin.assets.model_services.chat_model_services.google.logger")
    @patch("httpx.Client")
    def test_build_request_basic(self, mock_client, mock_logger):
        """Test that build_request uses the correct Vertex AI format and logs default max_tokens usage."""
        messages = [ChatMessage.from_user(text="Hello")]

        self.model_service.build_request(mock_client, messages, model=self.model_resource, stream=False)

        expected_json_body = {
            "anthropic_version": "vertex-2023-10-16",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 64000,
            "stream": False,
        }

        expected_headers = {
            "Content-Type": "application/json",
            "api-key": "test_api_key",
            "Authorization": "Bearer test_api_key",
        }

        mock_client.build_request.assert_called_once_with(
            "POST",
            url=self.model_service.build_request_url("inference", model=self.model_resource),
            json=expected_json_body,
            headers=expected_headers,
            timeout=None,
        )

        # Should log about using default max_tokens
        mock_logger.info.assert_called_once_with(
            "max_tokens is required for Vertex Anthropic Chat API. Using default value inferred from model: %s", 64000
        )

    @patch("httpx.Client")
    def test_build_request_with_stream(self, mock_client):
        """Test that build_request uses the correct streaming endpoint."""
        messages = [ChatMessage.from_user(text="Hello")]

        self.model_service.build_request(mock_client, messages, model=self.model_resource, stream=True)

        expected_json_body = {
            "anthropic_version": "vertex-2023-10-16",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 64000,
            "stream": True,
        }

        expected_headers = {
            "Content-Type": "application/json",
            "api-key": "test_api_key",
            "Authorization": "Bearer test_api_key",
        }

        mock_client.build_request.assert_called_once_with(
            "POST",
            url=self.model_service.build_request_url("stream", model=self.model_resource),
            json=expected_json_body,
            headers=expected_headers,
            timeout=None,
        )

    @patch("httpx.Client")
    def test_build_request_with_system_prompt(self, mock_client):
        """Test that system prompts are handled correctly."""
        messages = [ChatMessage.from_system("You are a helpful assistant."), ChatMessage.from_user(text="Hello")]

        self.model_service.build_request(mock_client, messages, model=self.model_resource, stream=False)

        expected_json_body = {
            "anthropic_version": "vertex-2023-10-16",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 64000,
            "stream": False,
            "system": "You are a helpful assistant.",
        }

        expected_headers = {
            "Content-Type": "application/json",
            "api-key": "test_api_key",
            "Authorization": "Bearer test_api_key",
        }

        mock_client.build_request.assert_called_once_with(
            "POST",
            url=self.model_service.build_request_url("inference", model=self.model_resource),
            json=expected_json_body,
            headers=expected_headers,
            timeout=None,
        )

    @patch("httpx.Client")
    def test_build_request_with_tools(self, mock_client):
        """Test that tools are included in the request."""
        messages = [ChatMessage.from_user(text="Hello")]
        mock_tool = Mock(autospec=Tool)
        mock_tool.name = "test_tool"
        mock_tool.description = "Test tool"
        mock_tool.parameters = {"type": "object", "properties": {}}

        self.model_service.build_request(
            mock_client, messages, model=self.model_resource, stream=False, tools=[mock_tool]
        )

        expected_json_body = {
            "anthropic_version": "vertex-2023-10-16",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 64000,
            "stream": False,
            "tools": [
                {"name": "test_tool", "description": "Test tool", "input_schema": {"type": "object", "properties": {}}}
            ],
        }

        expected_headers = {
            "Content-Type": "application/json",
            "api-key": "test_api_key",
            "Authorization": "Bearer test_api_key",
        }

        mock_client.build_request.assert_called_once_with(
            "POST",
            url=self.model_service.build_request_url("inference", model=self.model_resource),
            json=expected_json_body,
            headers=expected_headers,
            timeout=None,
        )

    @patch("httpx.Client")
    def test_build_request_with_extra_headers(self, mock_client):
        """Test that extra_headers are properly merged into the request headers."""
        messages = [ChatMessage.from_user(text="Hello")]
        extra_headers = {"X-Custom-Header": "custom_value", "Authorization": "Bearer custom_token"}

        self.model_service.build_request(
            mock_client, messages, model=self.model_resource, stream=False, extra_headers=extra_headers
        )

        expected_headers = {
            "Content-Type": "application/json",
            "api-key": "test_api_key",
            "X-Custom-Header": "custom_value",
            "Authorization": "Bearer custom_token",  # overrides api-key
        }

        mock_client.build_request.assert_called_once_with(
            "POST",
            url=self.model_service.build_request_url("inference", model=self.model_resource),
            json={
                "anthropic_version": "vertex-2023-10-16",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 64000,
                "stream": False,
            },
            headers=expected_headers,
            timeout=None,
        )

    @patch("httpx.Client")
    def test_build_request_with_inference_kwargs(self, mock_client):
        """Test that inference_kwargs are included in the request body."""
        messages = [ChatMessage.from_user(text="Hello")]

        self.model_service.build_request(
            mock_client,
            messages,
            model=self.model_resource,
            stream=False,
            inference_kwargs={"temperature": 0.7, "top_p": 0.9, "max_tokens": 4096},
        )

        expected_json_body = {
            "anthropic_version": "vertex-2023-10-16",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 4096,  # Should use the provided value
            "stream": False,
            "temperature": 0.7,
            "top_p": 0.9,
        }

        expected_headers = {
            "Content-Type": "application/json",
            "api-key": "test_api_key",
            "Authorization": "Bearer test_api_key",
        }

        mock_client.build_request.assert_called_once_with(
            "POST",
            url=self.model_service.build_request_url("inference", model=self.model_resource),
            json=expected_json_body,
            headers=expected_headers,
            timeout=None,
        )


if __name__ == "__main__":
    unittest.main()
