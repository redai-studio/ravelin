"""Tests for StreamingRelay and EventFilter."""

import asyncio
import unittest

from ravelin.dataclasses import StreamingChunk
from ravelin.streaming.backends.in_memory import InMemoryStreamingBackend
from ravelin.streaming.event import CATCHUP_SUBTYPE, EventType, StreamingEvent, StreamingSnapshot
from ravelin.streaming.relay import EventFilter, StreamingRelay, _intersect


# ---------------------------------------------------------------------------
# EventFilter
# ---------------------------------------------------------------------------


class TestEventFilter(unittest.TestCase):
    def _make_event(self, event_type=EventType.TOKEN_STREAM, component="llm"):
        return StreamingEvent(
            run_id="r1",
            component=component,
            call_index=0,
            type=event_type,
            subtype=None,
            payload=None,
        )

    def test_matches_unconstrained(self):
        f = EventFilter()
        self.assertTrue(f.matches(self._make_event()))

    def test_matches_type_filter(self):
        f = EventFilter(types={EventType.LIFECYCLE})
        self.assertFalse(f.matches(self._make_event(EventType.TOKEN_STREAM)))
        self.assertTrue(f.matches(self._make_event(EventType.LIFECYCLE)))

    def test_matches_component_filter(self):
        f = EventFilter(components={"llm"})
        self.assertTrue(f.matches(self._make_event(component="llm")))
        self.assertFalse(f.matches(self._make_event(component="retriever")))

    def test_matches_combined(self):
        f = EventFilter(types={EventType.LIFECYCLE}, components={"llm"})
        self.assertTrue(f.matches(self._make_event(EventType.LIFECYCLE, "llm")))
        self.assertFalse(f.matches(self._make_event(EventType.LIFECYCLE, "retriever")))
        self.assertFalse(f.matches(self._make_event(EventType.TOKEN_STREAM, "llm")))


class TestIntersect(unittest.TestCase):
    def test_both_none(self):
        self.assertIsNone(_intersect(None, None))

    def test_one_none(self):
        s = {1, 2}
        self.assertEqual(s, _intersect(s, None))
        self.assertEqual(s, _intersect(None, s))

    def test_both_set(self):
        self.assertEqual({2}, _intersect({1, 2}, {2, 3}))


# ---------------------------------------------------------------------------
# StreamingRelay
# ---------------------------------------------------------------------------


class TestStreamingRelay(unittest.IsolatedAsyncioTestCase):
    async def _write_events(self, backend):
        """Write a mix of events and return them."""
        events = [
            StreamingEvent(
                run_id="r1",
                component="llm",
                call_index=0,
                type=EventType.TOKEN_STREAM,
                subtype=None,
                payload=StreamingChunk(content="t1"),
            ),
            StreamingEvent(
                run_id="r1", component="llm", call_index=0, type=EventType.LIFECYCLE, subtype="start", payload={}
            ),
            StreamingEvent(
                run_id="r1",
                component="retriever",
                call_index=0,
                type=EventType.COMPONENT_ACTIVITY,
                subtype="query",
                payload={},
            ),
        ]
        for e in events:
            await backend.write_event(e)
        return events

    async def test_stream_fresh_start_injects_active_snapshot(self):
        backend = InMemoryStreamingBackend()
        await backend.write_event(
            StreamingEvent(
                run_id="r1",
                component="llm",
                call_index=0,
                type=EventType.TOKEN_STREAM,
                subtype=None,
                payload=StreamingChunk(content="hello"),
            )
        )

        relay = StreamingRelay(backend)
        collected = []
        async for event in relay.stream("r1"):
            collected.append(event)
            if len(collected) == 1:
                await backend.close_run("r1")

        self.assertEqual(1, len(collected))
        self.assertIsInstance(collected[0], StreamingSnapshot)
        self.assertEqual(CATCHUP_SUBTYPE, collected[0].subtype)
        self.assertEqual("hello", collected[0].payload.content)

    async def test_stream_no_filter_delivers_live_events(self):
        backend = InMemoryStreamingBackend()
        relay = StreamingRelay(backend)
        collected = []

        async def consume():
            async for event in relay.stream("r1"):
                collected.append(event)

        task = asyncio.create_task(consume())
        await asyncio.sleep(0)
        await self._write_events(backend)
        await backend.close_run("r1")
        await asyncio.wait_for(task, timeout=1)

        self.assertEqual(3, len(collected))

    async def test_stream_with_scope(self):
        backend = InMemoryStreamingBackend()
        relay = StreamingRelay(backend, scope=EventFilter(types={EventType.LIFECYCLE}))
        collected = []

        async def consume():
            async for event in relay.stream("r1"):
                collected.append(event)

        task = asyncio.create_task(consume())
        await asyncio.sleep(0)
        await self._write_events(backend)
        await backend.close_run("r1")
        await asyncio.wait_for(task, timeout=1)

        self.assertEqual(1, len(collected))
        self.assertEqual(EventType.LIFECYCLE, collected[0].type)

    async def test_stream_with_per_call_filter(self):
        backend = InMemoryStreamingBackend()
        relay = StreamingRelay(backend)
        collected = []

        async def consume():
            async for event in relay.stream("r1", event_filter=EventFilter(components={"retriever"})):
                collected.append(event)

        task = asyncio.create_task(consume())
        await asyncio.sleep(0)
        await self._write_events(backend)
        await backend.close_run("r1")
        await asyncio.wait_for(task, timeout=1)

        self.assertEqual(1, len(collected))
        self.assertEqual("retriever", collected[0].component)

    async def test_stream_sse_format(self):
        backend = InMemoryStreamingBackend()
        event = StreamingEvent(
            run_id="r1",
            component="llm",
            call_index=0,
            type=EventType.TOKEN_STREAM,
            subtype=None,
            payload=StreamingChunk(content="hello"),
        )
        await backend.write_event(event)

        relay = StreamingRelay(backend)
        chunks = []
        async for chunk in relay.stream_sse("r1"):
            chunks.append(chunk)
            if len(chunks) == 1:
                await backend.close_run("r1")

        # Fresh subscriptions receive active-generation catch-up snapshots, which
        # are synthetic state and therefore do not carry backend-assigned IDs.
        self.assertFalse(chunks[0].startswith("id: "))
        self.assertIn("data: ", chunks[0])
        self.assertIn(f'"subtype":"{CATCHUP_SUBTYPE}"', chunks[0])
        self.assertEqual(": end\n\n", chunks[-1])

    async def test_effective_filter_merge(self):
        """Scope and per-call filter are intersected."""
        relay = StreamingRelay(
            InMemoryStreamingBackend(),
            scope=EventFilter(types={EventType.LIFECYCLE, EventType.TOKEN_STREAM}),
        )
        effective = relay._effective_filter(EventFilter(types={EventType.LIFECYCLE}))
        self.assertEqual({EventType.LIFECYCLE}, effective.types)

    def test_format_sse_with_and_without_event_id(self):
        relay = StreamingRelay(InMemoryStreamingBackend())

        event_with_id = StreamingEvent(
            run_id="r1",
            component="llm",
            call_index=0,
            type=EventType.TOKEN_STREAM,
            subtype=None,
            payload="x",
            event_id="42",
        )
        result = relay.format_sse(event_with_id)
        self.assertTrue(result.startswith("id: 42\n"))

        event_no_id = StreamingEvent(
            run_id="r1",
            component="llm",
            call_index=0,
            type=EventType.TOKEN_STREAM,
            subtype=None,
            payload="x",
        )
        result = relay.format_sse(event_no_id)
        self.assertFalse(result.startswith("id:"))
        self.assertTrue(result.startswith("data:"))

    def test_format_sse_includes_generation_id(self):
        relay = StreamingRelay(InMemoryStreamingBackend())
        event = StreamingEvent(
            run_id="r1",
            component="llm",
            call_index=0,
            type=EventType.TOKEN_STREAM,
            subtype=None,
            payload="x",
            event_id="7",
            generation_id="g0",
        )
        result = relay.format_sse(event)
        self.assertIn('"generation_id":"g0"', result)


if __name__ == "__main__":
    unittest.main()
