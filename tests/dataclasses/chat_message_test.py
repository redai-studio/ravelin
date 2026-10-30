import unittest
from base64 import b64encode
from typing import Dict, Any

from ravelin.dataclasses.byte_stream import ByteStream
from ravelin.dataclasses.chat_message import ChatMessage, MediaContent
from haystack.dataclasses.chat_message import (
    ChatMessage as HaystackChatMessage,
    ChatRole,
    TextContent,
    ToolCall,
    ToolCallResult,
    ReasoningContent,
)
from haystack.dataclasses.image_content import ImageContent


class TestChatMessage(unittest.TestCase):
    def setUp(self):
        self.text = "Hello, World!"
        self.tool_call = ToolCall(tool_name="test_tool", arguments={"arg": "value"}, id="123")
        self.tool_result = ToolCallResult(result="test result", origin=self.tool_call, error=False)
        self.media = ByteStream(data=b"test data", mime_type="image/jpeg")
        self.media_content = MediaContent(media=self.media)

    def test_from_user(self):
        message = ChatMessage.from_user(self.text)
        self.assertEqual(ChatRole.USER, message.role)
        self.assertEqual(self.text, message.text)
        self.assertEqual(1, len(message.texts))

    def test_from_user_with_media(self):
        message = ChatMessage.from_user(self.text, media=[self.media])
        self.assertEqual(ChatRole.USER, message.role)
        self.assertEqual(self.text, message.text)
        self.assertEqual(1, len(message.media))
        self.assertEqual(self.media.data, message.media[0].data)

    def test_from_system(self):
        message = ChatMessage.from_system(self.text)
        self.assertEqual(ChatRole.SYSTEM, message.role)
        self.assertEqual(self.text, message.text)

    def test_from_assistant(self):
        meta = {"temperature": 0.7}
        message = ChatMessage.from_assistant(text=self.text, tool_calls=[self.tool_call], meta=meta)
        self.assertEqual(ChatRole.ASSISTANT, message.role)
        self.assertEqual(self.text, message.text)
        self.assertEqual(1, len(message.tool_calls))
        self.assertEqual(meta, message.meta)

    def test_from_assistant_with_reasoning_content(self):
        message = ChatMessage.from_assistant(text=self.text, reasoning_content="reasoning content")
        self.assertEqual("reasoning content", message.reasoning_content)
        self.assertEqual(self.text, message.text)

    def test_from_tool(self):
        message = ChatMessage.from_tool(tool_result="test result", origin=self.tool_call)
        self.assertEqual(ChatRole.TOOL, message.role)
        self.assertIsNotNone(message.tool_call_result)
        self.assertEqual("test result", message.tool_call_result.result)
        self.assertFalse(message.tool_call_result.error)

    def test_from_tool_raises_for_non_str(self):
        from haystack.dataclasses import TextContent

        with self.assertRaises(NotImplementedError):
            ChatMessage.from_tool(tool_result=[TextContent(text="hi")], origin=self.tool_call)

    def test_is_from(self):
        message = ChatMessage.from_user(self.text)
        self.assertTrue(message.is_from(ChatRole.USER))
        self.assertFalse(message.is_from(ChatRole.SYSTEM))

    def test_properties(self):
        # Test user message with media content
        user_message = ChatMessage(_role=ChatRole.USER, _content=[self.media_content, TextContent(text=self.text)])

        # Test basic properties for user message
        self.assertIsNone(user_message.name)
        self.assertEqual(2, len(user_message))
        self.assertEqual(self.text, user_message.text)
        self.assertEqual([self.text], user_message.texts)
        self.assertEqual(0, len(user_message.tool_calls))
        self.assertIsNone(user_message.tool_call)
        self.assertEqual(0, len(user_message.tool_call_results))
        self.assertIsNone(user_message.tool_call_result)
        self.assertIsNone(user_message.reasoning_content)
        self.assertEqual(0, len(user_message.reasonings))
        self.assertIsNone(user_message.reasoning)

        # Test media property
        self.assertEqual(1, len(user_message.media))
        self.assertEqual(self.media.data, user_message.media[0].data)
        self.assertEqual(self.media.mime_type, user_message.media[0].mime_type)

        # Test images property
        images = user_message.images
        self.assertEqual(1, len(images))
        self.assertIsInstance(images[0], ImageContent)
        self.assertEqual(self.media.mime_type, images[0].mime_type)
        # Verify the base64 encoded data
        expected_base64 = b64encode(self.media.data).decode("utf-8")
        self.assertEqual(expected_base64, images[0].base64_image)

        # Test content property
        self.assertEqual(2, len(user_message.content))
        self.assertIsInstance(user_message.content[0], MediaContent)
        self.assertIsInstance(user_message.content[1], TextContent)

        # Test assistant message (which doesn't have media content)
        assistant_message = ChatMessage.from_assistant(
            text=self.text, tool_calls=[self.tool_call], reasoning_content="test reasoning"
        )

        # Test basic properties for assistant message
        self.assertIsNone(assistant_message.name)
        self.assertEqual(2, len(assistant_message))
        self.assertEqual(self.text, assistant_message.text)
        self.assertEqual([self.text], assistant_message.texts)
        self.assertEqual(1, len(assistant_message.tool_calls))
        self.assertEqual(self.tool_call, assistant_message.tool_call)
        self.assertEqual(0, len(assistant_message.tool_call_results))
        self.assertIsNone(assistant_message.tool_call_result)
        self.assertEqual("test reasoning", assistant_message.reasoning_content)

        # Test reasoning properties
        self.assertEqual(1, len(assistant_message.reasonings))
        self.assertEqual("test reasoning", assistant_message.reasonings[0].reasoning_text)
        self.assertIsNotNone(assistant_message.reasoning)
        self.assertEqual("test reasoning", assistant_message.reasoning.reasoning_text)

        # Test media-related properties for assistant message
        self.assertEqual(0, len(assistant_message.media))
        self.assertEqual(0, len(assistant_message.images))
        self.assertEqual(2, len(assistant_message.content))
        self.assertIsInstance(assistant_message.content[0], TextContent)
        self.assertIsInstance(assistant_message.content[1], ToolCall)

    def test_to_dict(self):
        message = ChatMessage.from_assistant(text=self.text, reasoning_content="reasoning", tool_calls=[self.tool_call])
        data = message.to_dict()
        self.assertIsNone(data["name"])
        self.assertEqual(ChatRole.ASSISTANT.value, data["role"])
        self.assertEqual("reasoning", data["reasoning_content"])
        self.assertEqual(2, len(data["content"]))  # Text and tool call
        self.assertEqual(self.text, data["content"][0]["text"])
        self.assertIn("tool_call", data["content"][1])

    def test_to_dict_with_media(self):
        message = ChatMessage.from_user(text=self.text, media=[self.media])
        data = message.to_dict()
        self.assertEqual(ChatRole.USER.value, data["role"])
        self.assertEqual(2, len(data["content"]))
        self.assertIn("media", data["content"][0])
        self.assertEqual(self.media.to_dict(), data["content"][0]["media"])

    def test_to_trace_dict(self):
        message = ChatMessage.from_user(text=self.text, media=[self.media])
        data = message._to_trace_dict()
        self.assertEqual(ChatRole.USER.value, data["role"])
        self.assertEqual(2, len(data["content"]))
        self.assertIn("media", data["content"][0])
        self.assertEqual(self.media._to_trace_dict(), data["content"][0]["media"])

    def test_from_dict(self):
        data: Dict[str, Any] = {
            "role": "assistant",
            "content": [
                {"text": self.text},
                {"tool_call": {"tool_name": "test_tool", "arguments": {"arg": "value"}, "id": "123"}},
            ],
            "reasoning_content": "reasoning",
            "name": "test",
        }
        message = ChatMessage.from_dict(data)
        self.assertEqual("test", message.name)
        self.assertEqual(ChatRole.ASSISTANT, message.role)
        self.assertEqual("reasoning", message.reasoning_content)
        self.assertEqual(self.text, message.text)
        self.assertEqual(1, len(message.tool_calls))
        self.assertDictEqual({}, message.meta)

    def test_from_dict_with_media(self):
        data = {
            "role": "user",
            "content": [{"media": self.media.to_dict()}, {"text": self.text}],
        }
        message = ChatMessage.from_dict(data)
        self.assertEqual(ChatRole.USER, message.role)
        self.assertEqual(1, len(message.media))
        self.assertEqual(self.text, message.text)

    def test_from_dict_with_invalid_content(self):
        data = {
            "role": "assistant",
            "content": [{"invalid_type": "some content"}],
            "meta": {},
        }
        with self.assertRaises(ValueError):
            ChatMessage.from_dict(data)

    def test_from_dict_with_simple_content(self):
        data = {
            "role": "assistant",
            "content": self.text,
        }
        message = ChatMessage.from_dict(data)
        self.assertEqual(ChatRole.ASSISTANT, message.role)
        self.assertEqual(self.text, message.text)

    def test_to_dict_with_invalid_content(self):
        class InvalidContent:
            pass

        message = ChatMessage(_role=ChatRole.ASSISTANT, _content=[InvalidContent()])  # noqa
        with self.assertRaises(TypeError):
            message.to_dict()

    def test_from_haystack_message(self):
        # Create a haystack message with text and image content
        base64_image = b64encode(b"test image data").decode("utf-8")
        image_content = ImageContent(base64_image=base64_image, mime_type="image/jpeg")
        text_content = TextContent(text=self.text)
        reasoning_content = ReasoningContent(reasoning_text="Test reasoning")
        haystack_message = HaystackChatMessage(
            _role=ChatRole.USER, _content=[reasoning_content, text_content, image_content]
        )

        # Convert to our ChatMessage
        message = ChatMessage.from_haystack_message(haystack_message)

        # Verify conversion
        self.assertEqual(ChatRole.USER, message.role)
        self.assertEqual(self.text, message.text)
        self.assertEqual(1, len(message.media))
        self.assertEqual(b"test image data", message.media[0].data)

        # Verify reasoning content conversion
        self.assertEqual("Test reasoning", message.reasoning_content)
        self.assertEqual(1, len(message.reasonings))
        self.assertEqual("Test reasoning", message.reasoning.reasoning_text)


if __name__ == "__main__":
    unittest.main()
