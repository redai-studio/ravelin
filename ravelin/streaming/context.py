"""PipelineContext and ComponentContext — streaming coordinators."""

import contextvars
import logging
from contextlib import asynccontextmanager, contextmanager
from typing import Any, AsyncGenerator, Callable, Coroutine, Optional

from ravelin.streaming.backends.base import StreamingBackend, SyncCapableBackend
from ravelin.streaming.event import EventType, StreamingEvent

logger = logging.getLogger(__name__)

# Async callable: (event) -> None.
# Awaited before every component_activity and lifecycle event write.
# Token stream events bypass this hook — high-frequency, no journal value.
# Requires asynchronous emission: sync emits reject non-token events when a hook is set.
# Must complete quickly; queue internally for any expensive persistence work.
JournalHook = Callable[[StreamingEvent], Coroutine[Any, Any, None]]

_pipeline_ctx: contextvars.ContextVar[Optional["PipelineContext"]] = contextvars.ContextVar(
    "pipeline_ctx", default=None
)
_component_ctx: contextvars.ContextVar[Optional["ComponentContext"]] = contextvars.ContextVar(
    "component_ctx", default=None
)
# Caller-minted token identifying one durable generation step (e.g. a ctx.run LLM
# call) within a component invocation. Set for the duration of one inference call so
# the generation lifecycle emits (start/delta/complete/error) — which happen deeper
# in provider parse_response and in the model-service error helpers — all stamp the
# same token without threading it through every provider signature.
#
# It is a contextvar rather than a field on ComponentContext for two reasons:
# ComponentContext is created once per component run and is stable identity ("who am
# I"), whereas generation_id changes per inference call ("which sub-operation now");
# and a component may fire concurrent inference calls, so per-task isolation (which
# asyncio gives contextvars by copying context into each Task) is required to avoid
# siblings overwriting a shared field. None outside an inference call.
_generation_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("generation_id", default=None)


def get_pipeline_context() -> Optional["PipelineContext"]:
    """
    Return the active PipelineContext, or None if no streaming session is active.

    :returns: Active pipeline context, or ``None`` outside a streaming session.
    """
    return _pipeline_ctx.get()


def get_generation_id() -> Optional[str]:
    """
    Return the generation_id for the currently-executing inference call, or None.

    :returns: Active generation token, or ``None`` outside an inference call.
    """
    return _generation_id.get()


def set_generation_id(generation_id: Optional[str]) -> "contextvars.Token":
    """
    Set the generation_id for the current context, returning a reset token.

    Callers must ``reset_generation_id(token)`` in a ``finally`` block to restore
    the previous value.

    :param generation_id: Token to set for the current context, or ``None``.
    :returns: Token to pass to :func:`reset_generation_id`.
    """
    return _generation_id.set(generation_id)


def reset_generation_id(token: "contextvars.Token") -> None:
    """
    Restore the generation_id contextvar to its value before :func:`set_generation_id`.

    :param token: The token returned by :func:`set_generation_id`.
    """
    _generation_id.reset(token)


def get_component_context() -> Optional["ComponentContext"]:
    """
    Return the ComponentContext for the currently-executing component, or None.

    Available to model services and component code during component execution.
    Use this to emit token_stream and component_activity events without needing
    to know the component name or call_index explicitly.

    :returns: Active component context, or ``None`` outside component execution.
    """
    return _component_ctx.get()


class PipelineContext:
    """
    Pipeline-side coordinator for the streaming system.

    Owns the streaming session: run_id, backend reference, journal hook, and
    active-component tracking for snapshot management. Created and torn down by
    streaming_session(); managed entirely by the pipeline layer.

    Component code should use get_component_context() instead — it provides a
    scoped emitter without exposing pipeline infrastructure.
    """

    def __init__(
        self,
        run_id: str,
        backend: StreamingBackend,
        on_journal_event: Optional[JournalHook] = None,
    ) -> None:
        """
        Initialize a pipeline streaming context.

        :param run_id: Pipeline run identity.
        :param backend: Streaming backend used by the session.
        :param on_journal_event: Optional hook awaited for lifecycle and component-activity events
            before streaming delivery. Requires asynchronous emission.
        """
        self.run_id = run_id
        self._backend = backend
        self._on_journal_event = on_journal_event
        self._active: dict[str, int] = {}  # component name → current call_index

    @property
    def backend(self) -> StreamingBackend:
        """
        The streaming backend for this session.

        :returns: Backend used by this context.
        """
        return self._backend

    # ------------------------------------------------------------------
    # Active-component tracking (pipeline-managed, for snapshot lifecycle)
    # ------------------------------------------------------------------

    def set_active(self, component: str, call_index: int) -> None:
        """
        Register a component as actively executing with the given call_index.

        :param component: Component name.
        :param call_index: Component invocation index within the run.
        """
        self._active[component] = call_index

    def deactivate(self, component: str) -> None:
        """
        Unregister a component after execution completes.

        :param component: Component name.
        """
        self._active.pop(component, None)

    def get_call_index(self, component: str) -> int:
        """
        Return the current call_index for a component, or 0 if not active.

        :param component: Component name.
        :returns: Active component invocation index, or ``0`` when inactive.
        """
        return self._active.get(component, 0)

    # ------------------------------------------------------------------
    # Snapshot management
    # ------------------------------------------------------------------

    def clear_snapshot(self, component: str) -> None:
        """
        Reset the token stream snapshot for the current generation of component.

        Requires a SyncCapableBackend; raises TypeError otherwise.

        :param component: Component name.
        :raises TypeError: If the active backend is not sync-capable.
        """
        if not isinstance(self._backend, SyncCapableBackend):
            raise TypeError(
                f"clear_snapshot() requires a SyncCapableBackend, "
                f"got {type(self._backend).__name__}. "
                f"Use clear_snapshot_async() instead."
            )
        self._backend.clear_snapshot(self.run_id, component, self.get_call_index(component))

    async def clear_snapshot_async(self, component: str) -> None:
        """
        Reset the token stream snapshot for the current generation of component.

        :param component: Component name.
        """
        await self._backend.clear_snapshot_async(self.run_id, component, self.get_call_index(component))

    # ------------------------------------------------------------------
    # Emit
    # ------------------------------------------------------------------

    def emit(
        self,
        component: str,
        event_type: EventType,
        payload: Any,
        subtype: Optional[str] = None,
    ) -> None:
        """
        Pipeline-layer sync emit. Used by _run_with_lifecycle for lifecycle events.
        Requires a SyncCapableBackend; otherwise the event is logged and dropped.
        Raises RuntimeError for a non-token event when a journal hook is configured.

        :param component: Component name.
        :param event_type: Streaming event type.
        :param payload: Event payload.
        :param subtype: Optional event subtype.
        """
        event = StreamingEvent(
            run_id=self.run_id,
            component=component,
            call_index=self.get_call_index(component),
            type=event_type,
            subtype=subtype,
            payload=payload,
        )
        self._emit_event(event)

    async def emit_async(
        self,
        component: str,
        event_type: EventType,
        payload: Any,
        subtype: Optional[str] = None,
    ) -> None:
        """
        Pipeline-layer async emit. Used by _run_with_lifecycle_async for lifecycle events.
        Component code should use ComponentContext.emit_async() instead.

        :param component: Component name.
        :param event_type: Streaming event type.
        :param payload: Event payload.
        :param subtype: Optional event subtype.
        """
        event = StreamingEvent(
            run_id=self.run_id,
            component=component,
            call_index=self.get_call_index(component),
            type=event_type,
            subtype=subtype,
            payload=payload,
        )
        await self._emit_event_async(event)

    async def _emit_event_async(self, event: StreamingEvent) -> None:
        """
        Dispatch a pre-built event through the journal hook and backend.

        Journal hook exceptions propagate — the hook is the strict persistence
        path and must block advancement on failure. Backend write exceptions
        are logged and swallowed — streaming delivery is a best-effort
        side-channel that must never disrupt the component or pipeline.

        A dropped event is a real problem, not a nuisance: a lost token-stream
        delta produces corrupt content on the receiving SSE client, and a lost
        lifecycle event strands subscribers on an in-flight snapshot. We log at
        ERROR level accordingly. Volume during a backend outage is intended —
        one error per dropped event is the operator's signal to investigate.

        :param event: Pre-built event to dispatch.
        """
        if event.type is not EventType.TOKEN_STREAM and self._on_journal_event is not None:
            await self._on_journal_event(event)
        try:
            await self._backend.write_event(event)
        except Exception:  # pylint: disable=broad-exception-caught
            logger.error(
                "Failed to emit streaming event %s/%s (run_id=%s component=%s call_index=%s); "
                "receiving stream is now corrupt for this event",
                event.type.value,
                event.subtype,
                event.run_id,
                event.component,
                event.call_index,
                exc_info=True,
            )

    def _emit_event(self, event: StreamingEvent) -> None:
        """
        Synchronous dispatch to a SyncCapableBackend.

        The journal hook is async-only, so a non-token event raises
        RuntimeError when a hook is configured instead of silently skipping
        persistence. This covers callers that use PipelineContext directly or
        mix sync emission into an async session.

        Backend exceptions are logged and swallowed — emit is a fire-and-forget
        side-channel that must never disrupt the component or pipeline.
        A missing SyncCapableBackend is a configuration bug that drops the
        event, so it is also logged at ERROR level with the same rationale as
        :meth:`_emit_event_async`.

        :param event: Pre-built event to dispatch.
        :raises RuntimeError: If a journal hook is configured and the event is not a token stream event.
        """
        if event.type is not EventType.TOKEN_STREAM and self._on_journal_event is not None:
            raise RuntimeError(
                f"Cannot emit {event.type.value} event synchronously while a journal hook is configured "
                f"(run_id={event.run_id} component={event.component}): journal hooks require asynchronous "
                "emission. Use emit_async() inside streaming_session_async()."
            )
        try:
            if not isinstance(self._backend, SyncCapableBackend):
                logger.error(
                    "Sync emit requires a SyncCapableBackend, got %s. Event %s/%s dropped "
                    "(run_id=%s component=%s call_index=%s)",
                    type(self._backend).__name__,
                    event.type.value,
                    event.subtype,
                    event.run_id,
                    event.component,
                    event.call_index,
                )
                return
            self._backend.write_event_sync(event)
        except Exception:  # pylint: disable=broad-exception-caught
            logger.error(
                "Failed to emit streaming event %s/%s (run_id=%s component=%s call_index=%s); "
                "receiving stream is now corrupt for this event",
                event.type.value,
                event.subtype,
                event.run_id,
                event.component,
                event.call_index,
                exc_info=True,
            )


class ComponentContext:
    """
    Component-scoped streaming emitter.

    Retrieved via get_component_context() from anywhere in a component's call
    stack — model services, tool helpers, nested calls. Emit events on behalf
    of the current component without knowing its name or call_index explicitly.

    emit() is a no-op when no streaming session is active.
    """

    def __init__(
        self,
        component: str,
        call_index: int,
        pipeline_ctx: Optional[PipelineContext],
    ) -> None:
        """
        Initialize a component-scoped context.

        :param component: Component name.
        :param call_index: Component invocation index within the run.
        :param pipeline_ctx: Owning pipeline context, or ``None`` for no-op emission.
        """
        self.component = component
        self.call_index = call_index
        self._pipeline_ctx = pipeline_ctx

    def clear_snapshot(self) -> None:
        """
        Reset the token stream snapshot for this component's current generation.

        Call from sync methods after a generation completes so the next generation
        starts with a clean snapshot. Raises TypeError if the backend is not
        SyncCapableBackend. No-op if no streaming session is active.

        :raises TypeError: If a streaming session exists but the backend is not sync-capable.
        """
        if self._pipeline_ctx is None:
            return
        self._pipeline_ctx.clear_snapshot(self.component)

    async def clear_snapshot_async(self) -> None:
        """
        Reset the token stream snapshot for this component's current generation.

        Call from async methods after a generation completes so the next generation
        starts with a clean snapshot. Delegates to the backend's async path —
        RedisStreamingBackend DELs snapshot keys so cross-process readers see an
        empty snapshot immediately. No-op if no streaming session is active.
        """
        if self._pipeline_ctx is None:
            return
        await self._pipeline_ctx.clear_snapshot_async(self.component)

    async def emit_async(
        self,
        event_type: EventType,
        payload: Any,
        subtype: Optional[str] = None,
        generation_id: Optional[str] = None,
    ) -> None:
        """
        Emit an event on behalf of this component. Call from async methods.

        No-op if no streaming session is active.

        :param event_type: Streaming event type.
        :param payload: Event payload.
        :param subtype: Optional event subtype.
        :param generation_id: Optional caller-minted token identifying one durable
            generation step within this component invocation. Stable across a
            step's retries, distinct per logical step.
        """
        if self._pipeline_ctx is None:
            return
        event = StreamingEvent(
            run_id=self._pipeline_ctx.run_id,
            component=self.component,
            call_index=self.call_index,
            type=event_type,
            subtype=subtype,
            payload=payload,
            generation_id=generation_id,
        )
        await self._pipeline_ctx._emit_event_async(event)  # pylint: disable=protected-access

    def emit(
        self,
        event_type: EventType,
        payload: Any,
        subtype: Optional[str] = None,
        generation_id: Optional[str] = None,
    ) -> None:
        """
        Emit an event on behalf of this component. Call from sync methods.

        No-op if no streaming session is active. If the active backend is not a
        SyncCapableBackend, the event is logged and dropped. Raises RuntimeError
        for a non-token event when the session has a journal hook, which
        requires :meth:`emit_async`.

        :param event_type: Streaming event type.
        :param payload: Event payload.
        :param subtype: Optional event subtype.
        :param generation_id: Optional caller-minted token identifying one durable
            generation step within this component invocation. Stable across a
            step's retries, distinct per logical step.
        """
        if self._pipeline_ctx is None:
            return
        event = StreamingEvent(
            run_id=self._pipeline_ctx.run_id,
            component=self.component,
            call_index=self.call_index,
            type=event_type,
            subtype=subtype,
            payload=payload,
            generation_id=generation_id,
        )
        self._pipeline_ctx._emit_event(event)  # pylint: disable=protected-access


@asynccontextmanager
async def streaming_session_async(
    run_id: str,
    backend: StreamingBackend,
    on_journal_event: Optional[JournalHook] = None,
    close_on_exit: bool = True,
) -> AsyncGenerator[PipelineContext, None]:
    """
    Async context manager that installs a PipelineContext for a pipeline run.

    Installs the context into the process-local context var. When
    ``close_on_exit`` is True (default), calls ``backend.close_run(run_id)``
    on exit — ending all subscriber iterators.

    Set ``close_on_exit=False`` to participate in a streaming session without
    owning its lifecycle. This is useful when the caller is a subordinate
    executor (e.g. Restate component executor) and a higher-level orchestrator
    is responsible for closing the run.

    Usage::
        # Orchestrator — owns the run lifecycle:
        async with streaming_session_async(run_id, backend) as ctx:
            await pipeline.run_async(...)

        # Subordinate — participates but does not close:
        async with streaming_session_async(run_id, backend, close_on_exit=False) as ctx:
            await run_component(...)

    :param run_id: Pipeline run identity.
    :param backend: Streaming backend used by the session.
    :param on_journal_event: Optional hook awaited for lifecycle and component-activity events
        before streaming delivery. A hook failure propagates to the caller; it does not roll back
        work already performed. Token events bypass the hook.
    :param close_on_exit: Whether to close subscriber iterators when the context exits.
    :returns: Async context manager yielding the installed ``PipelineContext``.
    """
    ctx = PipelineContext(run_id=run_id, backend=backend, on_journal_event=on_journal_event)
    token = _pipeline_ctx.set(ctx)
    try:
        yield ctx
    finally:
        _pipeline_ctx.reset(token)
        if close_on_exit:
            await backend.close_run(run_id)


@contextmanager
def streaming_session(
    run_id: str,
    backend: SyncCapableBackend,
    on_journal_event: Optional[JournalHook] = None,
):
    """
    Synchronous context manager that installs a PipelineContext for a pipeline run.

    Does not call backend.close_run() on exit — close_run() is async-only, and
    SyncCapableBackend.close_run() is a no-op anyway (sync backends deliver events
    inline with no subscriber queues to drain). See SyncCapableBackend.close_run()
    for the full rationale.

    Journal hooks require asynchronous emission, so a non-None
    ``on_journal_event`` is rejected; use :func:`streaming_session_async` to
    journal events.

    Usage::
        with streaming_session(run_id, backend) as ctx:
            pipeline.run(data)

    :param run_id: Pipeline run identity.
    :param backend: Sync-capable streaming backend used by the session.
    :param on_journal_event: Must be None. Accepted only to fail explicitly instead of silently
        skipping persistence.
    :returns: Context manager yielding the installed ``PipelineContext``.
    :raises ValueError: If ``on_journal_event`` is not None.
    """
    if on_journal_event is not None:
        raise ValueError(
            "streaming_session() does not support on_journal_event: journal hooks require asynchronous "
            "emission. Use streaming_session_async() with AsyncPipeline or RestatePipeline."
        )
    ctx = PipelineContext(run_id=run_id, backend=backend, on_journal_event=on_journal_event)
    token = _pipeline_ctx.set(ctx)
    try:
        yield ctx
    finally:
        _pipeline_ctx.reset(token)
