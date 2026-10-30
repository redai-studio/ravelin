"""
This module holds various streaming buffer backends capable of handling StreamingEvent
emitted during pipeline executions.

Current built-in backends:
    StreamingBackend            abstract base — snapshot management + storage contract
    SyncCapableBackend          subclass for backends that work without an event loop
    InMemoryStreamingBackend    in-process backend for dev and tests
    CallbackStreamingBackend    inline-callback backend for scripts, logging, and tests
    RedisStreamingBackend       Redis-backed backend for production (requires redis extra)
"""

from ravelin.streaming.backends.base import StreamingBackend, SyncCapableBackend
from ravelin.streaming.backends.callback import CallbackStreamingBackend
from ravelin.streaming.backends.in_memory import InMemoryStreamingBackend
from ravelin.streaming.backends.redis import RedisStreamingBackend

__all__ = [
    "StreamingBackend",
    "SyncCapableBackend",
    "CallbackStreamingBackend",
    "InMemoryStreamingBackend",
    "RedisStreamingBackend",
]
