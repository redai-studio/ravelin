"""Base class for the streaming buffer backend."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional

from ravelin.streaming.event import (
    GENERATION_COMPLETE,
    GENERATION_ERROR,
    GENERATION_START,
    LIFECYCLE_COMPLETE,
    LIFECYCLE_ERROR,
    EventType,
    StreamingEvent,
    StreamingSnapshot,
)


class StreamingBackend(ABC):
    """
    Abstract base for streaming buffer backends (in-memory, Redis, etc.).

    One instance may serve many concurrent pipeline runs — ``run_id`` in each
    event routes storage and subscription.

    **Snapshot interface** (override in each backend):

    Snapshots accumulate token stream content for catch-up delivery to reconnecting
    clients. The base class defines the interface with no-op defaults. Concrete
    backends own their snapshot storage:

    - ``InMemoryStreamingBackend``: in-process dict — fast, single-process only.
    - ``RedisStreamingBackend``: Redis APPEND/RPUSH — delta-based, cross-process.
    - ``CallbackStreamingBackend`` and other sync backends: no snapshot support
      (inherit no-ops). Snapshots are only meaningful when a relay can subscribe
      cross-process; sync pipelines have no such consumers.

    ``write_event()`` calls the snapshot hooks automatically — backends only need
    to implement the hooks, not modify ``write_event()`` itself.

    **Event log interface** (abstract — implement in each backend):

    Concrete backends implement ``_persist_event()``, ``subscribe()``,
    ``close_run()``, and ``close()``.
    """

    def __init__(self, event_log_max_size: Optional[int] = None) -> None:
        """
        Initialize the backend.

        :param event_log_max_size: Maximum number of events retained per run for
            cursor replay. ``None`` keeps the log unbounded.
        """
        # Maximum number of events retained per run in the ordered log for from_event_id replay.
        # None means unlimited (default; suitable for dev/test).
        # When set, the log is a bounded FIFO per run: old events are evicted once full.
        # Redis backend: maps to XADD MAXLEN ~ N on the run stream.
        self.event_log_max_size = event_log_max_size

    # ------------------------------------------------------------------
    # Concrete template — subclasses may override for backend-specific batching
    # ------------------------------------------------------------------

    async def write_event(self, event: StreamingEvent) -> None:
        """
        Write an event to the backend.

        Default template: calls snapshot hooks around _persist_event():

        TOKEN_STREAM events (subtype determines snapshot behavior). Snapshots are
        keyed by ``(run_id, component, call_index, generation_id)`` — the optional
        ``generation_id`` separates multiple durable generation steps within one
        component invocation (e.g. two ``ctx.run`` LLM calls).

          - ``generation_start``: reset the accumulated content for this key (a
            retry re-runs the step from scratch, so partial content from the prior
            attempt is discarded) but **preserve** ``retry_count``. Creates the key
            at count 0 if absent.
          - ``generation_error``: increment ``retry_count`` for this key (creating
            it if absent), then persist. Lets a freshly connected client learn how
            many times an in-flight generation has retried.
          - ``generation_complete``: delete the snapshot for this key — the
            generation finished successfully and its transient state is no longer
            needed.
          - no subtype (regular chunk): accumulate into the snapshot, then persist.

        LIFECYCLE events:
          - ``complete`` / ``error``: calls _on_component_complete() after persisting,
            which releases *all* snapshot keys for the component invocation.

        Backends may override this method to batch multiple operations into a
        single round-trip (e.g. RedisStreamingBackend batches snapshot writes
        and XADD for TOKEN_STREAM events). Non-overridden event types should
        delegate to ``super().write_event(event)``.

        :param event: Event to write.
        """
        if event.type == EventType.TOKEN_STREAM:
            if event.subtype == GENERATION_START:
                await self._reset_snapshot_content_async(event)
            elif event.subtype == GENERATION_ERROR:
                await self._increment_retry_async(event)
            elif event.subtype == GENERATION_COMPLETE:
                await self.clear_snapshot_async(event.run_id, event.component, event.call_index, event.generation_id)
            else:
                await self._accumulate_snapshot_async(event)

        await self._persist_event(event)

        if event.type == EventType.LIFECYCLE and event.subtype in (
            LIFECYCLE_COMPLETE,
            LIFECYCLE_ERROR,
        ):
            await self._on_component_complete(event)

    # ------------------------------------------------------------------
    # Snapshot hooks — override in backends that support snapshots
    # ------------------------------------------------------------------

    async def _accumulate_snapshot_async(self, event: StreamingEvent) -> None:
        """
        Called for every TOKEN_STREAM chunk (no subtype) before _persist_event().

        Override to accumulate token content into backend-specific snapshot storage,
        keyed by ``(run_id, component, call_index, generation_id)``.
        Default: no-op (sync backends and backends without snapshot support).

        :param event: Token stream event to include in the snapshot.
        """

    async def _reset_snapshot_content_async(self, event: StreamingEvent) -> None:
        """
        Called on ``generation_start`` before _persist_event().

        Override to clear the accumulated content/reasoning/tool-deltas for this
        key while **preserving** ``retry_count`` (a retry re-runs the step from
        scratch, so partial content is discarded, but the count of prior failed
        attempts must survive). Creates the key at count 0 if absent.
        Default: no-op.

        :param event: The generation_start event identifying the key.
        """

    async def _increment_retry_async(self, event: StreamingEvent) -> None:
        """
        Called on ``generation_error`` before _persist_event().

        Override to increment ``retry_count`` for this key, creating the key if it
        does not yet exist (an initial-send failure can emit ``generation_error``
        before any ``generation_start``). Default: no-op.

        :param event: The generation_error event identifying the key.
        """

    async def _on_component_complete(self, event: StreamingEvent) -> None:
        """
        Called on LIFECYCLE_COMPLETE / LIFECYCLE_ERROR after _persist_event().

        Override to release *all* snapshot keys for the completed component
        invocation — lifecycle events carry no ``generation_id``, so every
        generation under this ``(run_id, component, call_index)`` is dropped.
        Default: no-op.

        :param event: Lifecycle event for the completed or failed component.
        """

    # ------------------------------------------------------------------
    # Snapshot public API — override in backends that support snapshots
    # ------------------------------------------------------------------

    async def clear_snapshot_async(
        self, run_id: str, component: str, call_index: int, generation_id: Optional[str] = None
    ) -> None:
        """
        Reset the token stream snapshot for one generation key.

        Called on ``generation_complete`` so a successfully finished generation
        releases its transient state. The component remains active.

        Default: no-op. Override in backends that maintain snapshot storage.
        RedisStreamingBackend issues DEL commands so cross-process readers
        immediately see an empty snapshot.

        :param run_id: Pipeline run identity.
        :param component: Component name.
        :param call_index: Component invocation index within the run.
        :param generation_id: Optional durable-step token. ``None`` targets the
            snapshot stored without a token (the degraded triple key).
        """

    async def get_snapshot(
        self, run_id: str, component: str, call_index: int, generation_id: Optional[str] = None
    ) -> StreamingSnapshot:
        """
        Return the accumulated snapshot for one generation key.

        Returns an empty StreamingSnapshot if no snapshot is available (component not
        started, already completed, or backend does not support snapshots).

        Default: returns empty snapshot. Override in backends that store snapshots.

        :param run_id: Pipeline run identity.
        :param component: Component name.
        :param call_index: Component invocation index within the run.
        :param generation_id: Optional durable-step token. ``None`` targets the
            snapshot stored without a token (the degraded triple key).
        :returns: Accumulated snapshot, or an empty snapshot when none is available.
        """
        return StreamingSnapshot(run_id=run_id, component=component, call_index=call_index, generation_id=generation_id)

    # ------------------------------------------------------------------
    # Abstract — implement in each backend
    # ------------------------------------------------------------------

    @abstractmethod
    async def _persist_event(self, event: StreamingEvent) -> None:
        """
        Write the event to backend storage and fan out to active subscribers.

        Called by write_event() after snapshot hooks. Responsible for:
          - assigning a monotonic event_id to the event
          - appending to the ordered event log for this run_id
          - delivering to all active subscriber queues for this run_id

        :param event: Event to persist.
        """

    @abstractmethod
    def subscribe(
        self,
        run_id: str,
        from_event_id: Optional[str] = None,
    ) -> AsyncIterator[StreamingEvent]:
        """
        Return an async iterator of all events for ``run_id``, across all components.

        If ``from_event_id`` is provided, the backend will attempt to resume
        starting from that event (inclusive). If the event is still present in
        the retained log, it will be the first event yielded.

        If the ID is not found in the retained window (evicted due to
        ``event_log_max_size`` or TTL), this signals a gap. The caller (usually
        ``StreamingRelay``) is responsible for handling catch-up — typically by
        injecting a ``StreamingSnapshot`` for any active generations before
        continuing with live events.

        The iterator ends when ``close_run(run_id)`` or ``close()`` is called.

        Usage::
            async for event in backend.subscribe(run_id):
                process(event)

        :param run_id: Pipeline run identity to subscribe to.
        :param from_event_id: Optional event ID cursor. Resume starting from
            this event (inclusive) if it is still available.
        :returns: Async iterator of retained and live events.
        """

    @abstractmethod
    async def close_run(self, run_id: str) -> None:
        """
        End subscriptions for one pipeline run.

        Signals all active subscriber iterators for ``run_id`` to stop. Called by
        the streaming_session context manager in its finally block.

        :param run_id: Pipeline run identity to close.
        """

    @abstractmethod
    async def close(self) -> None:
        """
        Shut down the entire backend.

        Signals all active subscriber iterators across all runs to stop and
        releases all resources. Call once at service shutdown.
        """


class SyncCapableBackend(StreamingBackend):
    """
    StreamingBackend subclass for backends that work without a running event loop.

    Concrete sync-capable backends implement _persist_event_sync() once.
    The async _persist_event() delegates to it by default, so the logic lives
    in exactly one place. subscribe(), close_run(), and close() default to
    no-ops — backends that don't support live subscription (e.g.
    CallbackStreamingBackend) inherit these defaults without further work.

    Snapshot support: none. Sync backends deliver events inline to a local
    consumer (e.g. a callback or logger) — there is no cross-process relay
    that would need snapshot catch-up. The snapshot hooks inherited from
    StreamingBackend are no-ops and write_event_sync() does not call them.

    Raises TypeError at emit_sync() time (via PipelineContext) if a non-sync-
    capable backend is passed to a sync streaming session — giving a clear error
    rather than silent failure.
    """

    def clear_snapshot(self, run_id: str, component: str, call_index: int, generation_id: Optional[str] = None) -> None:
        """
        Reset the token stream snapshot for one generation key.

        Sync analog of StreamingBackend.clear_snapshot_async(). No-op by default.
        Override in SyncCapableBackend subclasses that support snapshot storage.
        Retained here for run/run_async symmetry and potential future use cases.

        :param run_id: Pipeline run identity.
        :param component: Component name.
        :param call_index: Component invocation index within the run.
        :param generation_id: Optional durable-step token.
        """

    @abstractmethod
    def _persist_event_sync(self, event: StreamingEvent) -> None:
        """
        Synchronous analog of _persist_event(). No event loop required.

        :param event: Event to persist.
        """

    async def _persist_event(self, event: StreamingEvent) -> None:
        self._persist_event_sync(event)

    def write_event_sync(self, event: StreamingEvent) -> None:
        """
        Synchronous analog of write_event(). Template method — do not override.

        Snapshot hooks are not called on the sync path — sync backends have no
        cross-process relay consumer that would use snapshot catch-up.
        Journal hook is also skipped (it is async-only).

        :param event: Event to write.
        """
        self._persist_event_sync(event)

    def subscribe(
        self,
        run_id: str,
        from_event_id: Optional[str] = None,
    ) -> AsyncIterator[StreamingEvent]:
        """
        Default: no subscription support. Override if needed.

        :param run_id: Pipeline run identity to subscribe to.
        :param from_event_id: Optional event ID cursor.
        :returns: Empty async iterator by default.
        """
        return self._empty_subscribe()

    async def _empty_subscribe(self) -> AsyncIterator[StreamingEvent]:
        return
        yield  # make it an async generator

    async def close_run(self, run_id: str) -> None:
        """
        Default: no-op for sync-capable backends.

        Sync pipelines block the calling thread for their entire duration, leaving
        no concurrent consumer that could drain a subscriber queue. Subscriber queues
        are therefore not a viable pattern for sync backends — events must be delivered
        inline (see CallbackStreamingBackend). With no queues, there is nothing to
        signal on run completion, so this method intentionally does nothing.

        :param run_id: Pipeline run identity to close.
        """

    async def close(self) -> None:
        """Default: no-op. Override if cleanup is needed."""
