"""Sync and async pipelines with streaming lifecycle event support."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from haystack import tracing
from haystack.core.pipeline.async_pipeline import AsyncPipeline as HaystackAsyncPipeline
from haystack.core.pipeline.async_pipeline import Breakpoint
from haystack.core.pipeline.pipeline import Pipeline as HaystackPipeline

from ravelin.streaming.lifecycle import _run_with_lifecycle, _run_with_lifecycle_async


class Pipeline(HaystackPipeline):
    """
    Drop-in replacement for Haystack's sync Pipeline with streaming lifecycle events.

    Emits a lifecycle start/complete/error event around every component execution
    when a streaming session is active. Requires a SyncCapableBackend (e.g.,
    CallbackStreamingBackend). If no session is active, behaviour is identical
    to HaystackPipeline — streaming is fully opt-in.

    The caller establishes a streaming session before calling run()::

        backend = CallbackStreamingBackend(on_event=handle_event)
        with streaming_session(run_id, backend) as ctx:
            pipeline.run(...)
    """

    @staticmethod
    def _run_component(
        component_name: str,
        component: dict[str, Any],
        inputs: dict[str, Any],
        component_visits: dict[str, int],
        parent_span: Optional[tracing.Span] = None,
        *,
        break_point: Optional[Breakpoint] = None,
    ) -> Mapping[str, Any]:
        return _run_with_lifecycle(
            component_name,
            component_visits.get(component_name, 0),
            inputs,
            lambda: HaystackPipeline._run_component(
                component_name,
                component,
                inputs,
                component_visits,
                parent_span,
                break_point=break_point,
            ),
        )


class AsyncPipeline(HaystackAsyncPipeline):
    """
    Drop-in replacement for Haystack's AsyncPipeline with streaming lifecycle events.

    Emits a lifecycle start/complete/error event around every component execution
    when a streaming session is active. If no session is active, behaviour is
    identical to HaystackAsyncPipeline — streaming is fully opt-in.

    The caller establishes a streaming session before calling run()::

        backend = InMemoryStreamingBackend()
        async with streaming_session_async(run_id, backend) as ctx:
            await pipeline.run_async(...)
    """

    @staticmethod
    async def _run_component_async(
        component_name: str,
        component: dict[str, Any],
        component_inputs: dict[str, Any],
        component_visits: dict[str, int],
        parent_span: Optional[tracing.Span] = None,
        *,
        break_point: Optional[Breakpoint] = None,
    ) -> Mapping[str, Any]:
        return await _run_with_lifecycle_async(
            component_name,
            component_visits.get(component_name, 0),
            component_inputs,
            HaystackAsyncPipeline._run_component_async(
                component_name,
                component,
                component_inputs,
                component_visits,
                parent_span,
                break_point=break_point,
            ),
        )
