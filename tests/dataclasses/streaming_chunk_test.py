"""
This module contains tests for the StreamingChunk class.
"""

import unittest
from haystack.dataclasses import ToolCallDelta

from ravelin.dataclasses import StreamingChunk


class TestStreamingChunk(unittest.TestCase):
    """
    Tests for the StreamingChunk class.
    """

    def test_streaming_chunk_creation(self):
        """Test creating a StreamingChunk instance with content."""
        chunk = StreamingChunk(content="test content")
        self.assertEqual("test content", chunk.content)
        self.assertIsNone(chunk.reasoning_content)
        self.assertIsNone(chunk.tool_calls)
        self.assertFalse(chunk.start)  # Default is False
        self.assertIsNone(chunk.finish_reason)

    def test_streaming_chunk_with_reasoning(self):
        """Test creating a StreamingChunk instance with reasoning_content."""
        chunk = StreamingChunk(content="", reasoning_content="reasoning")
        self.assertEqual("", chunk.content)
        self.assertEqual("reasoning", chunk.reasoning_content)
        self.assertIsNone(chunk.tool_calls)
        self.assertFalse(chunk.start)
        self.assertIsNone(chunk.finish_reason)

    def test_streaming_chunk_with_tool_calls(self):
        """Test creating a StreamingChunk instance with tool_calls."""
        tool_calls = [ToolCallDelta(index=0, id="123", tool_name="test_tool", arguments='{"foo":"bar"}')]
        chunk = StreamingChunk(content="", tool_calls=tool_calls, index=0)
        self.assertEqual("", chunk.content)
        self.assertIsNone(chunk.reasoning_content)
        self.assertEqual(tool_calls, chunk.tool_calls)
        self.assertEqual(1, len(chunk.tool_calls))
        self.assertEqual("123", chunk.tool_calls[0].id)
        self.assertEqual("test_tool", chunk.tool_calls[0].tool_name)
        self.assertEqual('{"foo":"bar"}', chunk.tool_calls[0].arguments)
        self.assertFalse(chunk.start)
        self.assertIsNone(chunk.finish_reason)

    def test_streaming_chunk_with_start_and_finish(self):
        """Test creating a StreamingChunk with start flag and finish_reason set."""
        chunk = StreamingChunk(content="hello", start=True, finish_reason="stop")
        self.assertEqual("hello", chunk.content)
        self.assertTrue(chunk.start)
        self.assertEqual("stop", chunk.finish_reason)

    def test_streaming_chunk_tool_calls_with_finish_reason(self):
        """Test creating a StreamingChunk with tool_calls and finish_reason."""
        tool_calls = [ToolCallDelta(index=0, id="123", tool_name="test_tool", arguments='{"foo":"bar"}')]
        chunk = StreamingChunk(content="", tool_calls=tool_calls, index=0, finish_reason="tool_calls")
        self.assertEqual("", chunk.content)
        self.assertEqual(tool_calls, chunk.tool_calls)
        self.assertEqual("tool_calls", chunk.finish_reason)


if __name__ == "__main__":
    unittest.main()
