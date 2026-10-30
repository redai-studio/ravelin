"""In-process streaming backend for local development and testing."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import replace
from typing import AsyncIterator, Optional

from ravelin.dataclasses import StreamingChunk
from ravelin.streaming.backends.base import StreamingBackend
from ravelin.streaming.event import ComponentInvocation, StreamingEvent, StreamingSnapshot


class _BufferEnd:
    """Internal sentinel signaling a subscriber iterator to stop."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "BufferEnd"


_BUFFER_END = _BufferEnd()


class InMemoryStreamingBackend(StreamingBackend):
    """
    In-process streaming backend backed by asyncio queues.

    One instance serves as a service-level singleton — multiple concurrent
    pipeline runs are distinguished by ``run_id``. Each run has its own event
    log (deque) and subscriber queues so runs are fully isolated within the
    shared backend.

    Intended for local development and unit tests only. Not suitable for
    production:
      - State is process-local; multiple nodes cannot share the same instance.
      - Event logs are released by close_run() (TTL=0 semantics). Whether a
        run_id ever existed is the concern of the journal/orchestration layer,
        not the streaming layer. Call close() at service shutdown to release
        any remaining resources.

    **Snapshots** are stored in the in-process ``_snapshots`` dict. This is the
    full snapshot implementation for single-process use. The Redis backend
    replaces this with Redis-native storage for cross-process access.

    event_log_max_size controls the bounded FIFO behaviour per run:
      - None (default): unlimited log, full replay always available.
      - N: each run's deque retains the last N events; older events are evicted.
        If a subscriber's from_event_id falls outside the window, replay starts from
        the oldest available event (gap detection is the relay's responsibility).

    Thread-safety: single-threaded asyncio only. _accumulate_snapshot_async()
    does only dict operations and does not yield, so it is safe under cooperative
    concurrency. If model services ever emit token stream events via
    run_in_executor, an asyncio.Lock guard would be needed.
    """

    def __init__(self, event_log_max_size: Optional[int] = None) -> None:
        """
        Initialize the in-memory backend.

        :param event_log_max_size: Maximum number of events retained per run for
            cursor replay. ``None`` keeps the log unbounded.
        """
        super().__init__(event_log_max_size)
        # Per-run storage: run_id → ordered event log
        self._run_events: dict[str, deque[StreamingEvent]] = {}
        # Per-run subscriber queues: run_id → list of Queue
        self._run_queues: dict[str, list[asyncio.Queue]] = {}
        # In-process snapshot storage: ComponentInvocation → accumulated snapshot event
        self._snapshots: dict[ComponentInvocation, StreamingSnapshot] = {}
        self._next_id: int = 0
        self._closed: bool = False

    # ------------------------------------------------------------------
    # Snapshot hooks
    # ------------------------------------------------------------------

    async def _accumulate_snapshot_async(self, event: StreamingEvent) -> None:
        snapshot = self._ensure_snapshot(event)
        chunk: StreamingChunk = event.payload
        if chunk.content:
            snapshot.payload.content += chunk.content
        if chunk.reasoning_content:
            snapshot.payload.reasoning_content += chunk.reasoning_content
        if chunk.tool_calls:
            # TODO: merge deltas into a list of in-progress ToolCall objects
            # (keyed by ToolCallDelta.index) rather than accumulating raw fragments.
            snapshot.payload.tool_call_deltas.extend(chunk.tool_calls)

    async def _reset_snapshot_content_async(self, event: StreamingEvent) -> None:
        # A retry re-runs the generation from scratch: discard partial content but
        # preserve retry_count. Creates the key at count 0 if absent.
        snapshot = self._ensure_snapshot(event)
        snapshot.payload.content = ""
        snapshot.payload.reasoning_content = ""
        snapshot.payload.tool_call_deltas.clear()

    async def _increment_retry_async(self, event: StreamingEvent) -> None:
        # generation_error may arrive before any generation_start (initial-send
        # failure), so create the key if absent.
        snapshot = self._ensure_snapshot(event)
        snapshot.payload.retry_count += 1

    def _ensure_snapshot(self, event: StreamingEvent) -> StreamingSnapshot:
        key = ComponentInvocation(event.run_id, event.component, event.call_index, event.generation_id)
        snapshot = self._snapshots.get(key)
        if snapshot is None:
            snapshot = StreamingSnapshot(
                run_id=event.run_id,
                component=event.component,
                call_index=event.call_index,
                generation_id=event.generation_id,
            )
            self._snapshots[key] = snapshot
        return snapshot

    async def _on_component_complete(self, event: StreamingEvent) -> None:
        # Lifecycle events carry no generation_id — drop every generation under
        # this component invocation.
        for key in [
            k
            for k in self._snapshots
            if k.run_id == event.run_id and k.component == event.component and k.call_index == event.call_index
        ]:
            self._snapshots.pop(key, None)

    # ------------------------------------------------------------------
    # Snapshot public API
    # ------------------------------------------------------------------

    async def clear_snapshot_async(
        self, run_id: str, component: str, call_index: int, generation_id: Optional[str] = None
    ) -> None:
        """
        Remove the accumulated snapshot for one generation key.

        :param run_id: Pipeline run identity.
        :param component: Component name.
        :param call_index: Component invocation index within the run.
        :param generation_id: Optional durable-step token.
        """
        self._snapshots.pop(ComponentInvocation(run_id, component, call_index, generation_id), None)

    async def get_snapshot(
        self, run_id: str, component: str, call_index: int, generation_id: Optional[str] = None
    ) -> StreamingSnapshot:
        """
        Return a copy of the accumulated snapshot for one generation key.

        :param run_id: Pipeline run identity.
        :param component: Component name.
        :param call_index: Component invocation index within the run.
        :param generation_id: Optional durable-step token.
        :returns: Snapshot copy, or an empty snapshot when none is available.
        """
        stored = self._snapshots.get(ComponentInvocation(run_id, component, call_index, generation_id))
        if stored is None:
            return StreamingSnapshot(
                run_id=run_id, component=component, call_index=call_index, generation_id=generation_id
            )
        return self._copy_snapshot(stored)

    def _copy_snapshot(self, snap: StreamingSnapshot) -> StreamingSnapshot:
        """Shallow copy of a snapshot — duplicates tool_call_deltas to prevent external mutation."""
        return replace(snap, payload=replace(snap.payload, tool_call_deltas=list(snap.payload.tool_call_deltas)))

    # ------------------------------------------------------------------
    # StreamingBackend abstract method implementations
    # ------------------------------------------------------------------

    async def _persist_event(self, event: StreamingEvent) -> None:
        event.event_id = str(self._next_id)
        self._next_id += 1
        run_id = event.run_id
        if run_id not in self._run_events:
            self._run_events[run_id] = deque(maxlen=self.event_log_max_size)
        self._run_events[run_id].append(event)
        for q in self._run_queues.get(run_id, []):
            q.put_nowait(event)

    def subscribe(
        self,
        run_id: str,
        from_event_id: Optional[str] = None,
    ) -> AsyncIterator[StreamingEvent]:
        """
        Subscribe to retained and live events for one run.

        :param run_id: Pipeline run identity.
        :param from_event_id: Optional event ID cursor. Resume starting from
            this event (inclusive) if it is still available.
        :returns: Async iterator of retained and live events.
        """
        return self._subscribe(run_id, from_event_id)

    async def close_run(self, run_id: str) -> None:
        """
        End subscriptions for one pipeline run and release its event log.

        Signals all active subscriber iterators for ``run_id`` to stop, then
        discards the event log and snapshot state (TTL=0 semantics — no retention
        after the session closes). Whether a run_id ever existed is the
        journal/orchestration layer's concern; the streaming layer does not model
        run lifecycle.

        :param run_id: Pipeline run identity to close.
        """
        run_queues = self._run_queues.pop(run_id, [])
        for q in run_queues:
            q.put_nowait(_BUFFER_END)
        self._run_events.pop(run_id, None)
        # Release snapshots for all component invocations in this run.
        for key in [k for k in self._snapshots if k.run_id == run_id]:
            self._snapshots.pop(key, None)

    async def close(self) -> None:
        """Shut down all runs and mark the backend as closed."""
        if self._closed:
            return
        self._closed = True
        for run_id in list(self._run_queues.keys()):
            await self.close_run(run_id)
        # Discard any event logs or snapshots from runs that had no subscribers.
        self._run_events.clear()
        self._snapshots.clear()

    # ------------------------------------------------------------------
    # Internal async generator
    # ------------------------------------------------------------------

    async def _subscribe(
        self,
        run_id: str,
        from_event_id: Optional[str] = None,
    ) -> AsyncIterator[StreamingEvent]:
        if self._closed:
            return

        # Snapshot the current log once — O(n), acceptable for dev/test.
        log = list(self._run_events.get(run_id, []))

        # Determine whether we do inclusive replay or fresh-start / large-gap recovery.
        # Fresh-start / large-gap happens when:
        #   - from_event_id was provided but is no longer in the buffer, or
        #   - from_event_id was not provided (new connection).
        #
        # In the fresh-start case we discover current snapshots for active
        # generations and only deliver live events (plus those snapshots).
        do_fresh_start = False
        start = 0

        if from_event_id is not None:
            for i, e in enumerate(reversed(log)):
                if e.event_id == from_event_id:
                    start = len(log) - i - 1  # inclusive
                    break
            else:
                do_fresh_start = True
        else:
            # New connection — treat as fresh start / large gap
            do_fresh_start = True

        # Everything below up to the first await contains no yield/await points.
        # Queue registration, replay/snapshot collection all happen atomically.
        q: asyncio.Queue = asyncio.Queue()
        self._run_queues.setdefault(run_id, []).append(q)

        if do_fresh_start:
            # Fresh start / large gap path.
            # Inject snapshots for any currently active generations (last resort).
            # We do *not* replay historical events from the log.
            for key, snap in self._snapshots.items():
                if key.run_id == run_id:
                    q.put_nowait(self._copy_snapshot(snap))
        else:
            for event in log[start:]:
                q.put_nowait(event)

        if self._closed:
            q.put_nowait(_BUFFER_END)

        try:
            while True:
                item = await q.get()
                if isinstance(item, _BufferEnd):
                    return
                yield item
        finally:
            try:
                self._run_queues.get(run_id, []).remove(q)
            except ValueError:
                pass  # already removed by close_run()
