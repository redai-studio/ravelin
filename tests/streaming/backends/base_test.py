"""Tests for StreamingBackend.write_event dispatch and SyncCapableBackend."""

import unittest
from unittest.mock import AsyncMock

from ravelin.dataclasses import StreamingChunk
from ravelin.streaming.backends.base import StreamingBackend, SyncCapableBackend
from ravelin.streaming.event import (
    GENERATION_COMPLETE,
    GENERATION_ERROR,
    GENERATION_START,
    LIFECYCLE_COMPLETE,
    LIFECYCLE_ERROR,
    LIFECYCLE_START,
    EventType,
    StreamingEvent,
)


def _make_event(
    event_type: EventType,
    subtype=None,
    payload=None,
    run_id="run-1",
    component="llm",
    call_index=0,
):
    return StreamingEvent(
        run_id=run_id,
        component=component,
        call_index=call_index,
        type=event_type,
        subtype=subtype,
        payload=payload,
    )


class _ConcreteBackend(StreamingBackend):
    """Minimal concrete subclass for testing the base template logic."""

    def __init__(self):
        super().__init__()
        self.persisted: list[StreamingEvent] = []

    async def _persist_event(self, event):
        self.persisted.append(event)

    def subscribe(self, run_id, from_event_id=None):
        raise NotImplementedError

    async def close_run(self, run_id):
        pass

    async def close(self):
        pass


class TestWriteEventDispatch(unittest.IsolatedAsyncioTestCase):
    """Test the base write_event template method dispatch logic."""

    def setUp(self):
        self.backend = _ConcreteBackend()
        # Spy on the hooks.
        self.backend.clear_snapshot_async = AsyncMock()
        self.backend._accumulate_snapshot_async = AsyncMock()
        self.backend._reset_snapshot_content_async = AsyncMock()
        self.backend._increment_retry_async = AsyncMock()
        self.backend._on_component_complete = AsyncMock()

    async def test_token_stream_regular_chunk_accumulates(self):
        """Regular TOKEN_STREAM (no subtype) accumulates snapshot then persists."""
        chunk = StreamingChunk(content="hi")
        event = _make_event(EventType.TOKEN_STREAM, payload=chunk)

        await self.backend.write_event(event)

        self.backend._accumulate_snapshot_async.assert_awaited_once_with(event)
        self.backend.clear_snapshot_async.assert_not_awaited()
        self.assertEqual(1, len(self.backend.persisted))

    async def test_token_stream_generation_subtypes(self):
        """START resets content, ERROR increments retry, COMPLETE clears the key."""
        for subtype, expected_hook in [
            (GENERATION_START, "_reset_snapshot_content_async"),
            (GENERATION_ERROR, "_increment_retry_async"),
            (GENERATION_COMPLETE, "clear_snapshot_async"),
        ]:
            with self.subTest(subtype=subtype):
                self.backend.clear_snapshot_async.reset_mock()
                self.backend._accumulate_snapshot_async.reset_mock()
                self.backend._reset_snapshot_content_async.reset_mock()
                self.backend._increment_retry_async.reset_mock()
                self.backend.persisted.clear()

                event = _make_event(EventType.TOKEN_STREAM, subtype=subtype)
                await self.backend.write_event(event)

                for hook in ("_reset_snapshot_content_async", "_increment_retry_async", "clear_snapshot_async"):
                    if hook == expected_hook:
                        getattr(self.backend, hook).assert_awaited_once()
                    else:
                        getattr(self.backend, hook).assert_not_awaited()
                # None of the generation subtypes accumulate.
                self.backend._accumulate_snapshot_async.assert_not_awaited()
                self.assertEqual(1, len(self.backend.persisted))

    async def test_lifecycle_complete_and_error_call_on_component_complete(self):
        """LIFECYCLE complete/error call _on_component_complete after persist."""
        for subtype in (LIFECYCLE_COMPLETE, LIFECYCLE_ERROR):
            with self.subTest(subtype=subtype):
                self.backend._on_component_complete.reset_mock()
                self.backend.persisted.clear()

                event = _make_event(EventType.LIFECYCLE, subtype=subtype)
                await self.backend.write_event(event)

                self.backend._on_component_complete.assert_awaited_once_with(event)
                self.assertEqual(1, len(self.backend.persisted))

    async def test_lifecycle_start_does_not_call_on_component_complete(self):
        event = _make_event(EventType.LIFECYCLE, subtype=LIFECYCLE_START)
        await self.backend.write_event(event)

        self.backend._on_component_complete.assert_not_awaited()

    async def test_component_activity_no_snapshot_hooks(self):
        """COMPONENT_ACTIVITY events go straight through — no snapshot interaction."""
        event = _make_event(EventType.COMPONENT_ACTIVITY, subtype="tool_start")
        await self.backend.write_event(event)

        self.backend._accumulate_snapshot_async.assert_not_awaited()
        self.backend.clear_snapshot_async.assert_not_awaited()
        self.backend._on_component_complete.assert_not_awaited()
        self.assertEqual(1, len(self.backend.persisted))


class _ConcreteSyncBackend(SyncCapableBackend):
    """Minimal concrete subclass for testing SyncCapableBackend."""

    def __init__(self):
        super().__init__()
        self.persisted: list[StreamingEvent] = []

    def _persist_event_sync(self, event):
        self.persisted.append(event)


class TestSyncCapableBackend(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.backend = _ConcreteSyncBackend()

    def test_write_event_sync_calls_persist(self):
        event = _make_event(EventType.TOKEN_STREAM, payload=StreamingChunk(content="x"))
        self.backend.write_event_sync(event)
        self.assertEqual(1, len(self.backend.persisted))

    async def test_async_persist_delegates_to_sync(self):
        """The async _persist_event delegates to _persist_event_sync."""
        event = _make_event(EventType.COMPONENT_ACTIVITY)
        await self.backend._persist_event(event)
        self.assertEqual(1, len(self.backend.persisted))

    async def test_subscribe_yields_nothing(self):
        events = [e async for e in self.backend.subscribe("run-1")]
        self.assertEqual([], events)

    async def test_close_run_and_close_are_noop(self):
        await self.backend.close_run("run-1")
        await self.backend.close()


if __name__ == "__main__":
    unittest.main()
