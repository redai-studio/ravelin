"""Tests for _run_with_lifecycle and _run_with_lifecycle_async."""

import unittest
from unittest.mock import AsyncMock

from ravelin.streaming.context import (
    get_component_context,
    streaming_session,
    streaming_session_async,
)
from ravelin.streaming.backends.base import SyncCapableBackend
from ravelin.streaming.event import (
    LIFECYCLE_COMPLETE,
    LIFECYCLE_ERROR,
    LIFECYCLE_START,
    EventType,
    StreamingEvent,
)
from ravelin.streaming.lifecycle import _run_with_lifecycle, _run_with_lifecycle_async


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeSyncBackend(SyncCapableBackend):
    def __init__(self):
        super().__init__()
        self.events: list[StreamingEvent] = []

    def _persist_event_sync(self, event):
        self.events.append(event)


def _make_async_backend():
    backend = AsyncMock()
    backend.write_event = AsyncMock()
    backend.close_run = AsyncMock()
    return backend


# ---------------------------------------------------------------------------
# _run_with_lifecycle_async
# ---------------------------------------------------------------------------


class TestRunWithLifecycleAsync(unittest.IsolatedAsyncioTestCase):
    async def test_success_emits_start_and_complete(self):
        backend = _make_async_backend()
        written: list[StreamingEvent] = []
        backend.write_event.side_effect = lambda e: written.append(e)

        async with streaming_session_async("r1", backend):
            result = await _run_with_lifecycle_async(
                "llm",
                0,
                {"input": "x"},
                self._succeed(),
            )

        self.assertEqual("ok", result)
        lifecycle_events = [e for e in written if e.type == EventType.LIFECYCLE]
        self.assertEqual(2, len(lifecycle_events))
        self.assertEqual(LIFECYCLE_START, lifecycle_events[0].subtype)
        self.assertEqual(LIFECYCLE_COMPLETE, lifecycle_events[1].subtype)

    async def test_error_emits_start_and_error_then_reraises(self):
        backend = _make_async_backend()
        written: list[StreamingEvent] = []
        backend.write_event.side_effect = lambda e: written.append(e)

        with self.assertRaises(ValueError):
            async with streaming_session_async("r1", backend):
                await _run_with_lifecycle_async(
                    "llm",
                    0,
                    {},
                    self._fail(),
                )

        lifecycle_events = [e for e in written if e.type == EventType.LIFECYCLE]
        self.assertEqual(2, len(lifecycle_events))
        self.assertEqual(LIFECYCLE_START, lifecycle_events[0].subtype)
        self.assertEqual(LIFECYCLE_ERROR, lifecycle_events[1].subtype)

    async def test_error_uses_payload_factory_when_provided(self):
        backend = _make_async_backend()
        written: list[StreamingEvent] = []
        backend.write_event.side_effect = lambda e: written.append(e)

        with self.assertRaises(ValueError):
            async with streaming_session_async("r1", backend):
                await _run_with_lifecycle_async(
                    "llm",
                    0,
                    {},
                    self._fail(),
                    error_payload_factory=lambda e: {"error_type": type(e).__name__, "message": str(e)},
                )

        lifecycle_events = [e for e in written if e.type == EventType.LIFECYCLE]
        self.assertEqual({"error_type": "ValueError", "message": "test error"}, lifecycle_events[1].payload)

    async def test_sets_component_context_during_execution(self):
        """ComponentContext is available inside the wrapped coroutine."""
        captured_ctx = None

        async def capture():
            nonlocal captured_ctx
            captured_ctx = get_component_context()
            return "done"

        backend = _make_async_backend()
        async with streaming_session_async("r1", backend):
            await _run_with_lifecycle_async("mycomp", 3, {}, capture())

        self.assertIsNotNone(captured_ctx)
        self.assertEqual("mycomp", captured_ctx.component)
        self.assertEqual(3, captured_ctx.call_index)
        # After execution, context should be reset.
        self.assertIsNone(get_component_context())

    async def test_no_pipeline_context_still_sets_component_context(self):
        """Without a streaming session, lifecycle events are skipped but ComponentContext is set."""
        captured_ctx = None

        async def capture():
            nonlocal captured_ctx
            captured_ctx = get_component_context()
            return "done"

        result = await _run_with_lifecycle_async("llm", 0, {}, capture())
        self.assertEqual("done", result)
        self.assertIsNotNone(captured_ctx)
        self.assertIsNone(get_component_context())

    # -- helpers --

    @staticmethod
    async def _succeed():
        return "ok"

    @staticmethod
    async def _fail():
        raise ValueError("test error")


# ---------------------------------------------------------------------------
# _run_with_lifecycle (sync)
# ---------------------------------------------------------------------------


class TestRunWithLifecycleSync(unittest.TestCase):
    def test_success_emits_start_and_complete(self):
        backend = _FakeSyncBackend()
        with streaming_session("r1", backend):
            result = _run_with_lifecycle("llm", 0, {"input": "x"}, lambda: "ok")

        lifecycle = [e for e in backend.events if e.type == EventType.LIFECYCLE]
        self.assertEqual(2, len(lifecycle))
        self.assertEqual(LIFECYCLE_START, lifecycle[0].subtype)
        self.assertEqual(LIFECYCLE_COMPLETE, lifecycle[1].subtype)
        self.assertEqual("ok", result)

    def test_error_emits_start_and_error_then_reraises(self):
        backend = _FakeSyncBackend()

        def boom():
            raise RuntimeError("kaboom")

        with self.assertRaises(RuntimeError):
            with streaming_session("r1", backend):
                _run_with_lifecycle("llm", 0, {}, boom)

        lifecycle = [e for e in backend.events if e.type == EventType.LIFECYCLE]
        self.assertEqual(2, len(lifecycle))
        self.assertEqual(LIFECYCLE_ERROR, lifecycle[1].subtype)

    def test_sets_component_context(self):
        captured_ctx = None

        def capture():
            nonlocal captured_ctx
            captured_ctx = get_component_context()
            return "done"

        backend = _FakeSyncBackend()
        with streaming_session("r1", backend):
            _run_with_lifecycle("comp", 5, {}, capture)

        self.assertIsNotNone(captured_ctx)
        self.assertEqual("comp", captured_ctx.component)
        self.assertEqual(5, captured_ctx.call_index)
        self.assertIsNone(get_component_context())


if __name__ == "__main__":
    unittest.main()
