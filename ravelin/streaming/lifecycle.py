"""_run_with_lifecycle — shared component execution wrapper with streaming events."""

from typing import Any, Awaitable, Callable, Optional, TypeVar

from ravelin.streaming.context import ComponentContext, _component_ctx, get_pipeline_context
from ravelin.streaming.event import LIFECYCLE_COMPLETE, LIFECYCLE_ERROR, LIFECYCLE_START, EventType

_T = TypeVar("_T")


async def _run_with_lifecycle_async(
    component_name: str,
    call_index: int,
    component_inputs: dict[str, Any],
    coro: Awaitable[_T],
    error_payload_factory: Optional[Callable[[BaseException], Any]] = None,
) -> _T:
    """
    Wrap any component execution coroutine with streaming lifecycle events.

    Sets a ComponentContext for the duration so model services and component
    code can emit events via get_component_context() without needing to know
    the component name or call_index explicitly.

    Emits LIFECYCLE_START before and LIFECYCLE_COMPLETE / LIFECYCLE_ERROR
    after the coroutine. All emits are no-ops when no streaming session is
    active.

    On error, the **original** exception is re-raised — callers that need to
    recast it (e.g. Restate executor converting to TerminalError) should wrap
    this call in their own try/except. This preserves the real error in the
    streaming event while letting the caller handle framework-specific concerns.

    Shared by AsyncPipeline and RestatePipeline.

    :param component_name: Name of the component being executed.
    :param call_index: 0-based visit count for this component in the run.
    :param component_inputs: Inputs dict — included as LIFECYCLE_START payload.
    :param coro: The component execution coroutine to wrap.
    :param error_payload_factory: Optional hook for framework-specific structured error payloads.
    :returns: The coroutine's return value.
    """
    pipeline_ctx = get_pipeline_context()

    # Always set ComponentContext so model services and components know their identity,
    # even when no streaming session is active (pipeline_ctx may be None).
    comp_ctx = ComponentContext(component_name, call_index, pipeline_ctx)
    comp_token = _component_ctx.set(comp_ctx)

    if pipeline_ctx:
        pipeline_ctx.set_active(component_name, call_index)
        await pipeline_ctx.emit_async(
            component_name,
            EventType.LIFECYCLE,
            component_inputs,
            subtype=LIFECYCLE_START,
        )

    try:
        outputs = await coro
        if pipeline_ctx:
            await pipeline_ctx.emit_async(
                component_name,
                EventType.LIFECYCLE,
                outputs,
                subtype=LIFECYCLE_COMPLETE,
            )
        return outputs
    except Exception as e:
        if pipeline_ctx:
            await pipeline_ctx.emit_async(
                component_name,
                EventType.LIFECYCLE,
                error_payload_factory(e) if error_payload_factory else str(e),
                subtype=LIFECYCLE_ERROR,
            )
        raise
    finally:
        _component_ctx.reset(comp_token)
        if pipeline_ctx:
            pipeline_ctx.deactivate(component_name)


def _run_with_lifecycle(
    component_name: str,
    call_index: int,
    component_inputs: dict[str, Any],
    fn: Callable[[], _T],
    error_payload_factory: Optional[Callable[[BaseException], Any]] = None,
) -> _T:
    """
    Wrap any synchronous component execution with streaming lifecycle events.

    Sync counterpart to _run_with_lifecycle_async. Sets a ComponentContext for
    the duration and emits lifecycle start/complete/error events via the sync
    PipelineContext.emit() path, which requires a SyncCapableBackend.

    All emits are no-ops when no streaming session is active.

    :param component_name: Name of the component being executed.
    :param call_index: 0-based visit count for this component in the run.
    :param component_inputs: Inputs dict — included as LIFECYCLE_START payload.
    :param fn: The component execution callable to wrap.
    :param error_payload_factory: Optional hook for framework-specific structured error payloads.
    :returns: The callable's return value.
    """
    pipeline_ctx = get_pipeline_context()

    comp_ctx = ComponentContext(component_name, call_index, pipeline_ctx)
    comp_token = _component_ctx.set(comp_ctx)

    if pipeline_ctx:
        pipeline_ctx.set_active(component_name, call_index)
        pipeline_ctx.emit(component_name, EventType.LIFECYCLE, component_inputs, subtype=LIFECYCLE_START)

    try:
        outputs = fn()
        if pipeline_ctx:
            pipeline_ctx.emit(component_name, EventType.LIFECYCLE, outputs, subtype=LIFECYCLE_COMPLETE)
        return outputs
    except Exception as e:
        if pipeline_ctx:
            pipeline_ctx.emit(
                component_name,
                EventType.LIFECYCLE,
                error_payload_factory(e) if error_payload_factory else str(e),
                subtype=LIFECYCLE_ERROR,
            )
        raise
    finally:
        _component_ctx.reset(comp_token)
        if pipeline_ctx:
            pipeline_ctx.deactivate(component_name)
