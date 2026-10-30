"""Tests for PipelineContext, ComponentContext, and streaming_session."""

import unittest
from unittest.mock import AsyncMock, MagicMock

from ravelin.streaming.backends.base import SyncCapableBackend
from ravelin.streaming.context import (
    ComponentContext,
    PipelineContext,
    get_pipeline_context,
    streaming_session,
    streaming_session_async,
)
from ravelin.streaming.event import EventType, StreamingEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeSyncBackend(SyncCapableBackend):
    """Minimal sync backend that records events."""

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
# PipelineContext
# ---------------------------------------------------------------------------


class TestPipelineContextActiveTracking(unittest.TestCase):
    def test_set_active_and_get_call_index(self):
        ctx = PipelineContext("r1", _make_async_backend())
        ctx.set_active("llm", 2)
        self.assertEqual(2, ctx.get_call_index("llm"))

    def test_deactivate(self):
        ctx = PipelineContext("r1", _make_async_backend())
        ctx.set_active("llm", 1)
        ctx.deactivate("llm")
        self.assertEqual(0, ctx.get_call_index("llm"))

    def test_deactivate_absent_component_no_error(self):
        ctx = PipelineContext("r1", _make_async_backend())
        ctx.deactivate("nonexistent")  # should not raise


class TestPipelineContextEmitAsync(unittest.IsolatedAsyncioTestCase):
    async def test_emit_writes_to_backend(self):
        backend = _make_async_backend()
        ctx = PipelineContext("r1", backend)
        ctx.set_active("llm", 0)

        await ctx.emit_async("llm", EventType.LIFECYCLE, {"input": "x"}, subtype="start")

        backend.write_event.assert_awaited_once()
        event = backend.write_event.call_args[0][0]
        self.assertEqual(EventType.LIFECYCLE, event.type)
        self.assertEqual("start", event.subtype)

    async def test_emit_async_calls_journal_hook_for_non_token_stream(self):
        backend = _make_async_backend()
        hook = AsyncMock()
        ctx = PipelineContext("r1", backend, on_journal_event=hook)
        ctx.set_active("llm", 0)

        await ctx.emit_async("llm", EventType.COMPONENT_ACTIVITY, {}, subtype="tool_start")
        hook.assert_awaited_once()

    async def test_emit_async_skips_journal_hook_for_token_stream(self):
        backend = _make_async_backend()
        hook = AsyncMock()
        ctx = PipelineContext("r1", backend, on_journal_event=hook)
        ctx.set_active("llm", 0)

        await ctx.emit_async("llm", EventType.TOKEN_STREAM, "chunk")
        hook.assert_not_awaited()

    async def test_emit_async_awaits_journal_hook_before_backend_write(self):
        order: list[str] = []
        backend = _make_async_backend()
        backend.write_event.side_effect = lambda event: order.append("backend")

        async def hook(event):
            order.append("journal")

        ctx = PipelineContext("r1", backend, on_journal_event=hook)
        ctx.set_active("llm", 0)

        await ctx.emit_async("llm", EventType.LIFECYCLE, {}, subtype="start")
        self.assertEqual(["journal", "backend"], order)

    async def test_emit_async_propagates_journal_hook_failure_without_backend_write(self):
        backend = _make_async_backend()
        hook = AsyncMock(side_effect=RuntimeError("journal down"))
        ctx = PipelineContext("r1", backend, on_journal_event=hook)
        ctx.set_active("llm", 0)

        with self.assertRaisesRegex(RuntimeError, "journal down"):
            await ctx.emit_async("llm", EventType.COMPONENT_ACTIVITY, {}, subtype="tool_start")
        backend.write_event.assert_not_awaited()

    async def test_emit_async_swallows_exceptions(self):
        backend = _make_async_backend()
        backend.write_event.side_effect = RuntimeError("boom")
        ctx = PipelineContext("r1", backend)
        ctx.set_active("llm", 0)

        # Should not raise — exception is logged and swallowed.
        await ctx.emit_async("llm", EventType.LIFECYCLE, {}, subtype="start")


class TestPipelineContextEmitSync(unittest.TestCase):
    def test_emit_sync_writes_to_sync_backend(self):
        backend = _FakeSyncBackend()
        ctx = PipelineContext("r1", backend)
        ctx.set_active("llm", 0)

        ctx.emit("llm", EventType.LIFECYCLE, {}, subtype="start")
        self.assertEqual(1, len(backend.events))

    def test_emit_sync_drops_event_for_non_sync_backend(self):
        backend = _make_async_backend()
        ctx = PipelineContext("r1", backend)
        ctx.set_active("llm", 0)

        # Should not raise — logs warning and drops.
        ctx.emit("llm", EventType.LIFECYCLE, {}, subtype="start")

    def test_emit_sync_swallows_exceptions(self):
        backend = _FakeSyncBackend()
        backend._persist_event_sync = MagicMock(side_effect=RuntimeError("boom"))
        ctx = PipelineContext("r1", backend)
        ctx.set_active("llm", 0)

        ctx.emit("llm", EventType.LIFECYCLE, {}, subtype="start")

    def test_emit_sync_rejects_non_token_event_with_journal_hook(self):
        backend = _FakeSyncBackend()
        hook = AsyncMock()
        ctx = PipelineContext("r1", backend, on_journal_event=hook)
        ctx.set_active("llm", 0)

        for event_type in (EventType.LIFECYCLE, EventType.COMPONENT_ACTIVITY):
            with self.assertRaisesRegex(RuntimeError, "journal hooks require asynchronous emission"):
                ctx.emit("llm", event_type, {}, subtype="start")
        self.assertEqual([], backend.events)
        hook.assert_not_called()

    def test_emit_sync_allows_token_stream_with_journal_hook(self):
        backend = _FakeSyncBackend()
        hook = AsyncMock()
        ctx = PipelineContext("r1", backend, on_journal_event=hook)
        ctx.set_active("llm", 0)

        ctx.emit("llm", EventType.TOKEN_STREAM, "chunk")
        self.assertEqual(1, len(backend.events))
        hook.assert_not_called()


# ---------------------------------------------------------------------------
# ComponentContext
# ---------------------------------------------------------------------------


class TestComponentContext(unittest.IsolatedAsyncioTestCase):
    async def test_emit_async_delegates_to_pipeline_ctx(self):
        backend = _make_async_backend()
        pipeline_ctx = PipelineContext("r1", backend)
        comp_ctx = ComponentContext("llm", 0, pipeline_ctx)

        await comp_ctx.emit_async(EventType.TOKEN_STREAM, "chunk")
        backend.write_event.assert_awaited_once()

    async def test_emit_async_noop_without_pipeline_ctx(self):
        comp_ctx = ComponentContext("llm", 0, pipeline_ctx=None)
        # Should not raise.
        await comp_ctx.emit_async(EventType.TOKEN_STREAM, "chunk")

    def test_emit_sync_noop_without_pipeline_ctx(self):
        comp_ctx = ComponentContext("llm", 0, pipeline_ctx=None)
        comp_ctx.emit(EventType.TOKEN_STREAM, "chunk")

    async def test_clear_snapshot_async_noop_without_pipeline_ctx(self):
        comp_ctx = ComponentContext("llm", 0, pipeline_ctx=None)
        await comp_ctx.clear_snapshot_async()  # no-op


# ---------------------------------------------------------------------------
# streaming_session context managers
# ---------------------------------------------------------------------------


class TestStreamingSessionAsync(unittest.IsolatedAsyncioTestCase):
    async def test_sets_and_resets_pipeline_context(self):
        backend = _make_async_backend()
        self.assertIsNone(get_pipeline_context())

        async with streaming_session_async("r1", backend) as ctx:
            self.assertIs(ctx, get_pipeline_context())

        self.assertIsNone(get_pipeline_context())

    async def test_close_on_exit_true_calls_close_run(self):
        backend = _make_async_backend()
        async with streaming_session_async("r1", backend, close_on_exit=True):
            pass
        backend.close_run.assert_awaited_once_with("r1")

    async def test_close_on_exit_false_skips_close_run(self):
        backend = _make_async_backend()
        async with streaming_session_async("r1", backend, close_on_exit=False):
            pass
        backend.close_run.assert_not_awaited()

    async def test_context_reset_even_on_exception(self):
        backend = _make_async_backend()
        with self.assertRaises(ValueError):
            async with streaming_session_async("r1", backend):
                raise ValueError("test")
        self.assertIsNone(get_pipeline_context())

    async def test_sync_component_emit_rejected_and_context_reset(self):
        backend = _make_async_backend()
        hook = AsyncMock()
        with self.assertRaisesRegex(RuntimeError, "journal hooks require asynchronous emission"):
            async with streaming_session_async("r1", backend, on_journal_event=hook) as ctx:
                ComponentContext("llm", 0, ctx).emit(EventType.COMPONENT_ACTIVITY, {}, subtype="tool_start")
        hook.assert_not_awaited()
        self.assertIsNone(get_pipeline_context())
        backend.close_run.assert_awaited_once_with("r1")

    async def test_journal_hook_failure_resets_context(self):
        backend = _make_async_backend()
        hook = AsyncMock(side_effect=RuntimeError("journal down"))
        with self.assertRaisesRegex(RuntimeError, "journal down"):
            async with streaming_session_async("r1", backend, on_journal_event=hook) as ctx:
                await ctx.emit_async("llm", EventType.LIFECYCLE, {}, subtype="start")
        self.assertIsNone(get_pipeline_context())
        backend.close_run.assert_awaited_once_with("r1")


class TestStreamingSessionSync(unittest.TestCase):
    def test_sets_and_resets_pipeline_context(self):
        backend = _FakeSyncBackend()
        self.assertIsNone(get_pipeline_context())

        with streaming_session("r1", backend) as ctx:
            self.assertIs(ctx, get_pipeline_context())

        self.assertIsNone(get_pipeline_context())

    def test_rejects_journal_hook_before_installing_context(self):
        backend = _FakeSyncBackend()
        with self.assertRaisesRegex(ValueError, "streaming_session_async"):
            with streaming_session("r1", backend, on_journal_event=AsyncMock()):
                self.fail("session body must not run")
        self.assertIsNone(get_pipeline_context())


if __name__ == "__main__":
    unittest.main()
