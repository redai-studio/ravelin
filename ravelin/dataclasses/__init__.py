"""
The 'dataclasses' module is a package that contains all the dataclasses used in the SDK.
"""

from ravelin.dataclasses.asset_reference import AssetReference
from ravelin.dataclasses.byte_stream import ByteStream
from ravelin.dataclasses.chat_message import ChatMessage, ChatRole, ToolCall, ToolCallResult
from ravelin.dataclasses.embedding_fields import EmbeddingField
from ravelin.dataclasses.streaming_chunk import StreamingChunk

__all__ = [
    "AssetReference",
    "ByteStream",
    "ChatMessage",
    "ChatRole",
    "EmbeddingField",
    "StreamingChunk",
    "ToolCall",
    "ToolCallResult",
]
