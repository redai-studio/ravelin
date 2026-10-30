"""Tests for InMemoryStreamingBackend."""

import asyncio
import unittest

from ravelin.dataclasses import StreamingChunk
from ravelin.streaming.backends.in_memory import InMemoryStreamingBackend
from ravelin.streaming.event import (
    GENERATION_COMPLETE,
    GENERATION_ERROR,
    GENERATION_START,
    LIFECYCLE_COMPLETE,
    ComponentInvocation,
    EventType,
    StreamingEvent,
    StreamingSnapshot,
)


def _token_event(run_id="r1", component="llm", call_index=0, content="hi", subtype=None, generation_id=None):
    return StreamingEvent(
        run_id=run_id,
        component=component,
        call_index=call_index,
        type=EventType.TOKEN_STREAM,
        subtype=subtype,
        payload=StreamingChunk(content=content),
        generation_id=generation_id,
    )


def _lifecycle_event(run_id="r1", component="llm", call_index=0, subtype=LIFECYCLE_COMPLETE, payload=None):
    return StreamingEvent(
        run_id=run_id,
        component=component,
        call_index=call_index,
        type=EventType.LIFECYCLE,
        subtype=subtype,
        payload=payload,
    )


class TestSnapshotAccumulation(unittest.IsolatedAsyncioTestCase):
    """Snapshot accumulate, clear, and get_snapshot."""

    async def test_accumulate_and_get(self):
        backend = InMemoryStreamingBackend()
        await backend.write_event(_token_event(content="hello "))
        await backend.write_event(_token_event(content="world"))

        snap = await backend.get_snapshot("r1", "llm", 0)
        self.assertIsInstance(snap, StreamingSnapshot)
        self.assertEqual("hello world", snap.payload.content)

    async def test_generation_start_clears_snapshot(self):
        backend = InMemoryStreamingBackend()
        await backend.write_event(_token_event(content="old"))
        await backend.write_event(_token_event(subtype=GENERATION_START))

        snap = await backend.get_snapshot("r1", "llm", 0)
        self.assertEqual("", snap.payload.content)

    async def test_generation_complete_clears_snapshot(self):
        backend = InMemoryStreamingBackend()
        await backend.write_event(_token_event(content="data"))
        await backend.write_event(_token_event(subtype=GENERATION_COMPLETE))

        snap = await backend.get_snapshot("r1", "llm", 0)
        self.assertEqual("", snap.payload.content)

    async def test_lifecycle_complete_removes_snapshot(self):
        backend = InMemoryStreamingBackend()
        await backend.write_event(_token_event(content="data"))
        await backend.write_event(_lifecycle_event(subtype=LIFECYCLE_COMPLETE))

        key = ComponentInvocation("r1", "llm", 0)
        self.assertNotIn(key, backend._snapshots)

    async def test_get_snapshot_returns_copy(self):
        """get_snapshot returns a copy — mutating it doesn't affect the stored snapshot."""
        backend = InMemoryStreamingBackend()
        await backend.write_event(_token_event(content="x"))

        snap = await backend.get_snapshot("r1", "llm", 0)
        snap.payload.content = "mutated"

        original = await backend.get_snapshot("r1", "llm", 0)
        self.assertEqual("x", original.payload.content)

    async def test_get_snapshot_empty_when_not_started(self):
        backend = InMemoryStreamingBackend()
        snap = await backend.get_snapshot("r1", "llm", 0)
        self.assertEqual("", snap.payload.content)

    async def test_reasoning_content_accumulates(self):
        backend = InMemoryStreamingBackend()
        chunk = StreamingChunk(content="", reasoning_content="think", index=0)
        event = StreamingEvent(
            run_id="r1",
            component="llm",
            call_index=0,
            type=EventType.TOKEN_STREAM,
            subtype=None,
            payload=chunk,
        )
        await backend.write_event(event)

        snap = await backend.get_snapshot("r1", "llm", 0)
        self.assertEqual("think", snap.payload.reasoning_content)

    async def test_tool_call_deltas_accumulate(self):
        backend = InMemoryStreamingBackend()
        from haystack.dataclasses import ToolCallDelta

        delta = ToolCallDelta(index=0, tool_name="search", arguments="q")
        chunk = StreamingChunk(content="", tool_calls=[delta], index=0)
        event = StreamingEvent(
            run_id="r1",
            component="llm",
            call_index=0,
            type=EventType.TOKEN_STREAM,
            subtype=None,
            payload=chunk,
        )
        await backend.write_event(event)

        snap = await backend.get_snapshot("r1", "llm", 0)
        self.assertEqual(1, len(snap.payload.tool_call_deltas))


class TestRetryCountAndGenerationKey(unittest.IsolatedAsyncioTestCase):
    """Retry counting and per-generation_id key isolation."""

    async def test_error_increments_retry_count(self):
        backend = InMemoryStreamingBackend()
        await backend.write_event(_token_event(subtype=GENERATION_ERROR, generation_id="g0"))
        await backend.write_event(_token_event(subtype=GENERATION_ERROR, generation_id="g0"))

        snap = await backend.get_snapshot("r1", "llm", 0, "g0")
        self.assertEqual(2, snap.payload.retry_count)

    async def test_error_before_start_creates_key(self):
        """generation_error can arrive before generation_start (initial-send failure)."""
        backend = InMemoryStreamingBackend()
        await backend.write_event(_token_event(subtype=GENERATION_ERROR, generation_id="g0"))
        # A later start (retry) resets content but preserves the count.
        await backend.write_event(_token_event(content="partial", subtype=GENERATION_START, generation_id="g0"))
        await backend.write_event(_token_event(content="new", generation_id="g0"))

        snap = await backend.get_snapshot("r1", "llm", 0, "g0")
        self.assertEqual(1, snap.payload.retry_count)
        self.assertEqual("new", snap.payload.content)

    async def test_start_preserves_retry_count(self):
        backend = InMemoryStreamingBackend()
        await backend.write_event(_token_event(content="attempt1", generation_id="g0"))
        await backend.write_event(_token_event(subtype=GENERATION_ERROR, generation_id="g0"))
        await backend.write_event(_token_event(subtype=GENERATION_START, generation_id="g0"))

        snap = await backend.get_snapshot("r1", "llm", 0, "g0")
        self.assertEqual(1, snap.payload.retry_count)
        self.assertEqual("", snap.payload.content)

    async def test_complete_deletes_key(self):
        backend = InMemoryStreamingBackend()
        await backend.write_event(_token_event(subtype=GENERATION_ERROR, generation_id="g0"))
        await backend.write_event(_token_event(subtype=GENERATION_COMPLETE, generation_id="g0"))

        snap = await backend.get_snapshot("r1", "llm", 0, "g0")
        self.assertEqual(0, snap.payload.retry_count)
        self.assertNotIn(ComponentInvocation("r1", "llm", 0, "g0"), backend._snapshots)

    async def test_sibling_generations_isolated(self):
        """Two ctx.run steps under one (run, component, call_index) don't collide."""
        backend = InMemoryStreamingBackend()
        await backend.write_event(_token_event(subtype=GENERATION_ERROR, generation_id="g0"))
        await backend.write_event(_token_event(subtype=GENERATION_ERROR, generation_id="g1"))
        await backend.write_event(_token_event(subtype=GENERATION_ERROR, generation_id="g1"))

        snap0 = await backend.get_snapshot("r1", "llm", 0, "g0")
        snap1 = await backend.get_snapshot("r1", "llm", 0, "g1")
        self.assertEqual(1, snap0.payload.retry_count)
        self.assertEqual(2, snap1.payload.retry_count)

    async def test_lifecycle_complete_purges_all_generations(self):
        backend = InMemoryStreamingBackend()
        await backend.write_event(_token_event(subtype=GENERATION_ERROR, generation_id="g0"))
        await backend.write_event(_token_event(subtype=GENERATION_ERROR, generation_id="g1"))

        await backend.write_event(_lifecycle_event(subtype=LIFECYCLE_COMPLETE))

        self.assertEqual(0, len(backend._snapshots))


class TestPersistAndSubscribe(unittest.IsolatedAsyncioTestCase):
    """Event persistence, subscriber delivery, and replay."""

    async def test_events_get_monotonic_ids(self):
        backend = InMemoryStreamingBackend()
        e1 = _token_event(content="a")
        e2 = _token_event(content="b")
        await backend.write_event(e1)
        await backend.write_event(e2)

        self.assertEqual("0", e1.event_id)
        self.assertEqual("1", e2.event_id)

    async def test_subscriber_receives_replay_and_live(self):
        backend = InMemoryStreamingBackend()
        # Write an event before subscribing.
        await backend.write_event(_token_event(content="before"))

        collected: list[StreamingEvent] = []

        async def consume():
            async for event in backend.subscribe("r1"):
                collected.append(event)

        task = asyncio.create_task(consume())
        await asyncio.sleep(0)  # let the subscriber start and replay

        # Write a live event.
        await backend.write_event(_token_event(content="live"))
        await asyncio.sleep(0)

        await backend.close_run("r1")
        await task

        self.assertEqual(2, len(collected))
        self.assertEqual("before", collected[0].payload.content)
        self.assertEqual("live", collected[1].payload.content)

    async def test_subscribe_from_event_id_replays_inclusive(self):
        backend = InMemoryStreamingBackend()
        await backend.write_event(_token_event(content="a"))
        await backend.write_event(_token_event(content="b"))
        await backend.write_event(_token_event(content="c"))

        collected: list[StreamingEvent] = []

        async def consume():
            async for event in backend.subscribe("r1", from_event_id="0"):
                collected.append(event)

        task = asyncio.create_task(consume())
        await asyncio.sleep(0)
        await backend.close_run("r1")
        await task

        # With inclusive semantics, from_event_id="0" should include the event with id "0" + later events.
        self.assertEqual(3, len(collected))
        self.assertEqual("a", collected[0].payload.content)
        self.assertEqual("b", collected[1].payload.content)

    async def test_subscribe_gap_injects_snapshots(self):
        """When from_event_id is evicted (bounded log) or not provided, subscribers get snapshot catch-up for active generations + live from current tail."""
        backend = InMemoryStreamingBackend(event_log_max_size=2)

        # Write 3 events — first one will be evicted.
        await backend.write_event(_token_event(content="a"))
        await backend.write_event(_token_event(content="b"))
        await backend.write_event(_token_event(content="c"))

        collected: list[StreamingEvent] = []

        async def consume():
            # from_event_id="0" was evicted (only "1" and "2" remain).
            async for event in backend.subscribe("r1", from_event_id="0"):
                collected.append(event)

        task = asyncio.create_task(consume())
        await asyncio.sleep(0)
        await backend.close_run("r1")
        await task

        # Should get a snapshot catch-up event first, then no replay
        # (gap mode delivers live events only, but the run is already closed).
        # The snapshot contains accumulated content "abc".
        self.assertTrue(len(collected) >= 1)
        first = collected[0]
        self.assertIsInstance(first, StreamingSnapshot)
        self.assertEqual("abc", first.payload.content)

    async def test_subscribe_empty_from_event_id_triggers_fresh_start(self):
        """Empty from_event_id (new connection) should behave like large gap: snapshots + live from current tail."""
        backend = InMemoryStreamingBackend(event_log_max_size=2)

        # Write 3 events — first one will be evicted.
        await backend.write_event(_token_event(content="a"))
        await backend.write_event(_token_event(content="b"))
        await backend.write_event(_token_event(content="c"))

        collected: list[StreamingEvent] = []

        async def consume():
            # New connection: no from_event_id provided.
            async for event in backend.subscribe("r1"):
                collected.append(event)

        task = asyncio.create_task(consume())
        await asyncio.sleep(0)
        await backend.close_run("r1")
        await task

        # Should behave the same as the evicted case: get a snapshot catch-up.
        self.assertTrue(len(collected) >= 1)
        first = collected[0]
        self.assertIsInstance(first, StreamingSnapshot)
        self.assertEqual("abc", first.payload.content)


class TestCloseRunAndClose(unittest.IsolatedAsyncioTestCase):
    async def test_close_run_clears_state(self):
        backend = InMemoryStreamingBackend()
        await backend.write_event(_token_event(content="x"))

        await backend.close_run("r1")

        self.assertNotIn("r1", backend._run_events)
        # Snapshots for that run are also cleaned up.
        self.assertEqual(0, len(backend._snapshots))

    async def test_close_is_idempotent(self):
        backend = InMemoryStreamingBackend()
        await backend.close()
        await backend.close()  # second call should not raise

    async def test_subscribe_after_close_yields_nothing(self):
        backend = InMemoryStreamingBackend()
        await backend.close()

        events = [e async for e in backend.subscribe("r1")]
        self.assertEqual([], events)


if __name__ == "__main__":
    unittest.main()
