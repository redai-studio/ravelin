"""Tests for RedisStreamingBackend using fakeredis."""

import asyncio
import unittest

import fakeredis.aioredis

from ravelin.dataclasses import StreamingChunk
from ravelin.streaming.backends.redis import RedisStreamingBackend
from ravelin.streaming.event import (
    GENERATION_COMPLETE,
    GENERATION_ERROR,
    GENERATION_START,
    LIFECYCLE_COMPLETE,
    LIFECYCLE_START,
    ComponentInvocation,
    EventType,
    StreamingEvent,
    StreamingSnapshot,
)


def _new_backend(event_log_max_size=None, ttl_seconds=3600) -> RedisStreamingBackend:
    """Create a RedisStreamingBackend backed by fakeredis."""
    backend = RedisStreamingBackend(
        redis_url="redis://fake",
        event_log_max_size=event_log_max_size,
        ttl_seconds=ttl_seconds,
    )
    # Replace the real Redis client with a fakeredis instance.
    backend._redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    return backend


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


def _lifecycle_event(run_id="r1", component="llm", call_index=0, subtype=LIFECYCLE_START, payload=None):
    return StreamingEvent(
        run_id=run_id,
        component=component,
        call_index=call_index,
        type=EventType.LIFECYCLE,
        subtype=subtype,
        payload=payload,
    )


class TestWriteAndPersist(unittest.IsolatedAsyncioTestCase):
    """TOKEN_STREAM batching and non-TOKEN_STREAM persistence."""

    async def asyncSetUp(self):
        self.backend = _new_backend()

    async def asyncTearDown(self):
        await self.backend.close()

    async def test_token_stream_assigns_event_id(self):
        event = _token_event(content="hello")
        await self.backend.write_event(event)

        self.assertIsNotNone(event.event_id)

    async def test_lifecycle_event_assigns_event_id(self):
        event = _lifecycle_event(subtype=LIFECYCLE_START, payload={"input": "x"})
        await self.backend.write_event(event)

        self.assertIsNotNone(event.event_id)

    async def test_first_write_sets_ttl_on_stream_key(self):
        await self.backend.close()
        self.backend = _new_backend(ttl_seconds=600)

        await self.backend.write_event(_token_event())

        ttl = await self.backend._redis.ttl("r1:events")
        self.assertGreater(ttl, 0)
        self.assertLessEqual(ttl, 600)

    async def test_second_write_skips_ttl(self):
        """After the first write, subsequent writes don't re-issue EXPIRE."""
        await self.backend.write_event(_token_event(content="a"))
        self.assertIn("r1", self.backend._initialized_runs)

        await self.backend.write_event(_token_event(content="b"))


class TestSnapshot(unittest.IsolatedAsyncioTestCase):
    """Snapshot accumulation, retrieval, and clearing."""

    async def asyncSetUp(self):
        self.backend = _new_backend()

    async def asyncTearDown(self):
        await self.backend.close()

    async def test_accumulate_and_get(self):
        await self.backend.write_event(_token_event(content="hello "))
        await self.backend.write_event(_token_event(content="world"))

        snap = await self.backend.get_snapshot("r1", "llm", 0)
        self.assertIsInstance(snap, StreamingSnapshot)
        self.assertEqual("hello world", snap.payload.content)

    async def test_generation_start_clears_snapshot(self):
        await self.backend.write_event(_token_event(content="old"))
        await self.backend.write_event(_token_event(subtype=GENERATION_START))

        snap = await self.backend.get_snapshot("r1", "llm", 0)
        self.assertEqual("", snap.payload.content)

    async def test_generation_complete_clears_snapshot(self):
        await self.backend.write_event(_token_event(content="data"))
        await self.backend.write_event(_token_event(subtype=GENERATION_COMPLETE))

        snap = await self.backend.get_snapshot("r1", "llm", 0)
        self.assertEqual("", snap.payload.content)

    async def test_clear_snapshot_removes_from_active_components(self):
        await self.backend.write_event(_token_event(content="x"))

        members = await self.backend._redis.smembers("r1:active_components")
        self.assertIn("llm:0", members)

        await self.backend.clear_snapshot_async("r1", "llm", 0)

        members = await self.backend._redis.smembers("r1:active_components")
        self.assertNotIn("llm:0", members)
        self.assertNotIn(ComponentInvocation("r1", "llm", 0), self.backend._initialized_snapshots)

    async def test_lifecycle_complete_clears_snapshot(self):
        await self.backend.write_event(_token_event(content="x"))
        await self.backend.write_event(_lifecycle_event(subtype=LIFECYCLE_COMPLETE))

        snap = await self.backend.get_snapshot("r1", "llm", 0)
        self.assertEqual("", snap.payload.content)

    async def test_get_snapshot_empty_when_not_started(self):
        snap = await self.backend.get_snapshot("r1", "llm", 0)
        self.assertEqual("", snap.payload.content)
        self.assertEqual("", snap.payload.reasoning_content)
        self.assertEqual([], snap.payload.tool_call_deltas)

    async def test_reasoning_content_accumulates(self):
        chunk = StreamingChunk(content="", reasoning_content="think", index=0)
        event = StreamingEvent(
            run_id="r1",
            component="llm",
            call_index=0,
            type=EventType.TOKEN_STREAM,
            subtype=None,
            payload=chunk,
        )
        await self.backend.write_event(event)

        snap = await self.backend.get_snapshot("r1", "llm", 0)
        self.assertEqual("think", snap.payload.reasoning_content)


class TestRetryCountAndGenerationKey(unittest.IsolatedAsyncioTestCase):
    """Retry counting and per-generation_id key isolation."""

    async def asyncSetUp(self):
        self.backend = _new_backend()

    async def asyncTearDown(self):
        await self.backend.close()

    async def test_error_increments_retry_count(self):
        await self.backend.write_event(_token_event(subtype=GENERATION_ERROR, generation_id="g0"))
        await self.backend.write_event(_token_event(subtype=GENERATION_ERROR, generation_id="g0"))

        snap = await self.backend.get_snapshot("r1", "llm", 0, "g0")
        self.assertEqual(2, snap.payload.retry_count)

    async def test_error_before_start_creates_key(self):
        await self.backend.write_event(_token_event(subtype=GENERATION_ERROR, generation_id="g0"))
        await self.backend.write_event(_token_event(content="partial", subtype=GENERATION_START, generation_id="g0"))
        await self.backend.write_event(_token_event(content="new", generation_id="g0"))

        snap = await self.backend.get_snapshot("r1", "llm", 0, "g0")
        self.assertEqual(1, snap.payload.retry_count)
        self.assertEqual("new", snap.payload.content)

    async def test_start_preserves_retry_count(self):
        await self.backend.write_event(_token_event(content="attempt1", generation_id="g0"))
        await self.backend.write_event(_token_event(subtype=GENERATION_ERROR, generation_id="g0"))
        await self.backend.write_event(_token_event(subtype=GENERATION_START, generation_id="g0"))

        snap = await self.backend.get_snapshot("r1", "llm", 0, "g0")
        self.assertEqual(1, snap.payload.retry_count)
        self.assertEqual("", snap.payload.content)

    async def test_complete_deletes_key(self):
        await self.backend.write_event(_token_event(subtype=GENERATION_ERROR, generation_id="g0"))
        await self.backend.write_event(_token_event(subtype=GENERATION_COMPLETE, generation_id="g0"))

        snap = await self.backend.get_snapshot("r1", "llm", 0, "g0")
        self.assertEqual(0, snap.payload.retry_count)
        members = await self.backend._redis.smembers("r1:active_components")
        self.assertNotIn("llm:0:g0", members)

    async def test_sibling_generations_isolated(self):
        await self.backend.write_event(_token_event(subtype=GENERATION_ERROR, generation_id="g0"))
        await self.backend.write_event(_token_event(subtype=GENERATION_ERROR, generation_id="g1"))
        await self.backend.write_event(_token_event(subtype=GENERATION_ERROR, generation_id="g1"))

        snap0 = await self.backend.get_snapshot("r1", "llm", 0, "g0")
        snap1 = await self.backend.get_snapshot("r1", "llm", 0, "g1")
        self.assertEqual(1, snap0.payload.retry_count)
        self.assertEqual(2, snap1.payload.retry_count)

    async def test_lifecycle_complete_purges_all_generations(self):
        await self.backend.write_event(_token_event(content="a", generation_id="g0"))
        await self.backend.write_event(_token_event(subtype=GENERATION_ERROR, generation_id="g1"))

        await self.backend.write_event(_lifecycle_event(subtype=LIFECYCLE_COMPLETE))

        members = await self.backend._redis.smembers("r1:active_components")
        self.assertEqual(set(), members)
        self.assertEqual(0, (await self.backend.get_snapshot("r1", "llm", 0, "g0")).payload.retry_count)
        self.assertEqual("", (await self.backend.get_snapshot("r1", "llm", 0, "g0")).payload.content)
        self.assertEqual(0, (await self.backend.get_snapshot("r1", "llm", 0, "g1")).payload.retry_count)

    async def test_fresh_connect_yields_snapshot_per_generation(self):
        """A new connection gets one catch-up snapshot per live generation, with retry_count."""
        await self.backend.write_event(_token_event(content="a", subtype=GENERATION_ERROR, generation_id="g0"))
        await self.backend.write_event(_token_event(content="body", generation_id="g0"))
        await self.backend.write_event(_token_event(subtype=GENERATION_ERROR, generation_id="g1"))

        collected: list[StreamingEvent] = []

        async def consume():
            async for event in self.backend.subscribe("r1"):
                collected.append(event)

        task = asyncio.create_task(consume())
        await asyncio.sleep(0.05)
        await self.backend.close_run("r1")
        await task

        snapshots = {e.generation_id: e for e in collected if isinstance(e, StreamingSnapshot)}
        self.assertEqual({"g0", "g1"}, set(snapshots))
        self.assertEqual("body", snapshots["g0"].payload.content)
        self.assertEqual(1, snapshots["g0"].payload.retry_count)
        self.assertEqual(1, snapshots["g1"].payload.retry_count)


class TestSubscribeAndReplay(unittest.IsolatedAsyncioTestCase):
    """Subscribe, replay from cursor, and sentinel-based close."""

    async def asyncSetUp(self):
        self.backend = _new_backend()

    async def asyncTearDown(self):
        await self.backend.close()

    async def test_subscribe_replays_and_receives_live(self):
        await self.backend.write_event(_token_event(content="before"))

        collected: list[StreamingEvent] = []

        async def consume():
            async for event in self.backend.subscribe("r1"):
                collected.append(event)

        task = asyncio.create_task(consume())
        await asyncio.sleep(0.2)

        await self.backend.write_event(_token_event(content="live"))
        await asyncio.sleep(0.2)

        await self.backend.close_run("r1")
        await asyncio.wait_for(task, timeout=2)

        contents = [e.payload.content for e in collected if hasattr(e.payload, "content")]
        self.assertIn("before", contents)
        self.assertIn("live", contents)

    async def test_subscribe_from_event_id_resumes(self):
        e1 = _token_event(content="a")
        e2 = _token_event(content="b")
        await self.backend.write_event(e1)
        await self.backend.write_event(e2)

        collected: list[StreamingEvent] = []

        async def consume():
            async for event in self.backend.subscribe("r1", from_event_id=e1.event_id):
                collected.append(event)

        task = asyncio.create_task(consume())
        await asyncio.sleep(0.2)
        await self.backend.close_run("r1")
        await asyncio.wait_for(task, timeout=2)

        # from_event_id is inclusive: the retained cursor event is re-delivered
        # first, followed by later events.
        self.assertEqual(2, len(collected))
        self.assertEqual("a", collected[0].payload.content)
        self.assertEqual("b", collected[1].payload.content)

    async def test_subscribe_empty_from_event_id_triggers_fresh_start(self):
        """Empty from_event_id (new connection) should behave like large gap: snapshots + live from current tail."""
        backend = _new_backend(event_log_max_size=2)

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
        await asyncio.sleep(0.2)
        await backend.close_run("r1")
        await asyncio.wait_for(task, timeout=2)

        first = collected[0]
        self.assertIsInstance(first, StreamingSnapshot)
        self.assertEqual("abc", first.payload.content)

    async def test_deserialization_round_trip(self):
        """Events written and read back preserve type, subtype, component, call_index."""
        original = _lifecycle_event(subtype=LIFECYCLE_START, payload={"key": "value"})
        await self.backend.write_event(original)

        collected: list[StreamingEvent] = []

        async def consume():
            async for event in self.backend.subscribe("r1", from_event_id=original.event_id):
                collected.append(event)

        task = asyncio.create_task(consume())
        await asyncio.sleep(0.2)
        await self.backend.close_run("r1")
        await asyncio.wait_for(task, timeout=2)

        self.assertEqual(1, len(collected))
        event = collected[0]
        self.assertEqual(EventType.LIFECYCLE, event.type)
        self.assertEqual(LIFECYCLE_START, event.subtype)
        self.assertEqual("llm", event.component)
        self.assertEqual(0, event.call_index)


class TestCloseRunAndClose(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.backend = _new_backend()

    async def asyncTearDown(self):
        await self.backend.close()

    async def test_close_run_cleans_up_local_tracking(self):
        await self.backend.write_event(_token_event(content="x"))

        self.assertIn("r1", self.backend._initialized_runs)
        self.assertTrue(len(self.backend._initialized_snapshots) > 0)

        await self.backend.close_run("r1")

        self.assertNotIn("r1", self.backend._initialized_runs)
        self.assertEqual(0, len([inv for inv in self.backend._initialized_snapshots if inv.run_id == "r1"]))

    async def test_close_is_idempotent(self):
        await self.backend.close()
        await self.backend.close()

    async def test_fresh_subscribe_after_close_run_finishes(self):
        await self.backend.write_event(_token_event(content="finished"))
        await self.backend.close_run("r1")

        async def collect():
            return [event async for event in self.backend.subscribe("r1")]

        self.assertEqual([], await asyncio.wait_for(collect(), timeout=1))

    async def test_expired_cursor_after_close_run_finishes(self):
        event = _token_event(content="expired")
        await self.backend.write_event(event)
        await self.backend._redis.xdel("r1:events", event.event_id)
        await self.backend.close_run("r1")

        async def collect():
            return [item async for item in self.backend.subscribe("r1", from_event_id=event.event_id)]

        self.assertEqual([], await asyncio.wait_for(collect(), timeout=1))

    async def test_retained_cursor_after_close_run_replays_then_finishes(self):
        first = _token_event(content="a")
        second = _token_event(content="b")
        await self.backend.write_event(first)
        await self.backend.write_event(second)
        await self.backend.close_run("r1")

        async def collect():
            return [event async for event in self.backend.subscribe("r1", from_event_id=first.event_id)]

        events = await asyncio.wait_for(collect(), timeout=1)
        self.assertEqual(["a", "b"], [event.payload.content for event in events])

    async def test_subscribe_after_close_yields_nothing(self):
        await self.backend.close()

        events = [e async for e in self.backend.subscribe("r1")]
        self.assertEqual([], events)


if __name__ == "__main__":
    unittest.main()
