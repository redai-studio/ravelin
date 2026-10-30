"""Redis-backed streaming backend for production deployments."""

import logging
from collections.abc import Awaitable
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Optional, cast

from haystack.lazy_imports import LazyImport

from ravelin.core.serialization import deserialize_value, serialize_value
from ravelin.streaming.backends.base import StreamingBackend
from ravelin.streaming.event import (
    GENERATION_COMPLETE,
    GENERATION_ERROR,
    GENERATION_START,
    ComponentInvocation,
    EventType,
    StreamingEvent,
    StreamingSnapshot,
)
from ravelin.dataclasses import StreamingChunk

logger = logging.getLogger(__name__)


with LazyImport("Run 'pip install ravelin[redis]'") as redis_import:
    import redis.asyncio as aioredis

    if TYPE_CHECKING:
        from redis.typing import EncodableT, FieldT

_json_dumps: Callable[[Any], str]
_json_loads: Callable[[Any], Any]

try:
    import orjson as _orjson

    def _json_dumps(obj: Any) -> str:
        return _orjson.dumps(obj).decode()

    _json_loads = _orjson.loads
except ImportError:
    import json as _json_stdlib

    def _json_dumps(obj: Any) -> str:
        return _json_stdlib.dumps(obj)

    _json_loads = _json_stdlib.loads


class RedisStreamingBackend(StreamingBackend):
    """
    Redis-backed streaming backend for production deployments.

    **Event log**: single ``{run_id}:events`` Redis Stream per run.
      - One stream per run keeps subscription simple (one XREAD key).
      - ``from_event_id`` maps directly to Redis Stream entry IDs (globally monotonic
        within the stream).
      - Bounded by ``XADD MAXLEN ~ N`` when ``event_log_max_size`` is set.
      - Stream key TTL is a crash-safety net; normal cleanup is via sentinel.

    **Snapshots**: per-ComponentInvocation Redis keys, delta-based.
      - ``{run_id}:{component}:{call_index}:content`` — Redis String, APPEND per chunk.
      - ``{run_id}:{component}:{call_index}:reasoning`` — Redis String, APPEND per chunk.
      - ``{run_id}:{component}:{call_index}:tool_deltas`` — Redis List, RPUSH per delta.
      - Each write is O(chunk_size) — no quadratic bandwidth growth as content grows.
      - ``get_snapshot()`` reads all three keys in one pipeline round-trip.
      - ``clear_snapshot_async()`` DELs all three keys immediately so cross-process
        readers (the relay) see an empty snapshot during the inter-generation gap.
      - Snapshot keys carry their own TTL (same ``ttl_seconds``) set on first write.

    **Active components**: ``{run_id}:active_components`` Redis Set.
      - Stores ``{component}:{call_index}`` strings for all invocations with
        accumulated snapshot state. Written by ``write_event()`` on first TOKEN_STREAM
        write, removed by ``clear_snapshot_async()`` on completion.
      - Queried by the relay during gap detection (``_subscribe`` with a stale
        ``from_event_id``) to discover which components need catch-up snapshots.
      - Using Redis rather than local memory ensures correct behaviour across
        nodes: the pipeline node and relay node may be different processes.

    **TOKEN_STREAM batching**: ``write_event()`` is overridden for TOKEN_STREAM events
    to batch snapshot writes (APPEND / RPUSH) and the event log write (XADD) into a
    single pipeline round-trip. Other event types fall through to the base class.

    ``close_run(run_id)`` writes a sentinel entry to the stream; subscriber
    iterators detect it and stop.

    Requires the ``redis`` extra: ``pip install ravelin[redis]``

    :param redis_url: Redis connection URL. Default: ``redis://localhost:6379``.
    :param event_log_max_size: Maximum events retained per run stream.
        Maps to ``XADD MAXLEN ~ N``. ``None`` (default) = unlimited.
    :param ttl_seconds: TTL applied to stream and snapshot keys as a crash-safety net.
        Normal cleanup happens through the component-complete and ``close_run``
        paths; TTL is a fallback for processes that crash before reaching those.
        Default: 86400 (24 h).

        **Set this to cover your longest expected run duration, including maximum
        human-in-the-loop suspension time.** For example, if approval workflows
        can remain suspended for up to 7 days, use ``ttl_seconds=7*24*3600``.
        Snapshot keys are typically cleared by the model service layer before any
        awakeable wait, so an expired ``active_components`` set during suspension
        has no correctness impact, but the stream key must survive until the run
        completes or the subscriber is no longer needed.
    """

    _SENTINEL_FIELD = "__sentinel__"

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        event_log_max_size: Optional[int] = None,
        ttl_seconds: int = 86400,
    ) -> None:
        redis_import.check()
        super().__init__(event_log_max_size)
        self._redis = aioredis.from_url(redis_url, decode_responses=True)
        self._ttl_seconds = ttl_seconds
        # Local TTL-tracking sets — used only to avoid redundant EXPIRE calls
        # from this process instance. Gap detection uses the Redis-backed
        # active_components set instead, so these do not need to be shared.
        self._initialized_runs: set[str] = set()
        self._initialized_snapshots: set[ComponentInvocation] = set()
        self._closed = False

    # ------------------------------------------------------------------
    # write_event override — TOKEN_STREAM batching
    # ------------------------------------------------------------------

    async def write_event(self, event: StreamingEvent) -> None:
        """
        Write an event to Redis.

        TOKEN_STREAM events without a subtype (regular chunks): snapshot writes
        (APPEND / RPUSH) and the event log write (XADD) are batched into a
        single pipeline round-trip.

        TOKEN_STREAM events with generation subtypes (start/complete/error) and
        all other event types delegate to the base class template, which handles
        snapshot clearing and lifecycle hooks.

        :param event: Event to write.
        """
        if event.type != EventType.TOKEN_STREAM or event.subtype in (
            GENERATION_START,
            GENERATION_COMPLETE,
            GENERATION_ERROR,
        ):
            await super().write_event(event)
            return

        chunk: StreamingChunk = event.payload
        inv = ComponentInvocation(event.run_id, event.component, event.call_index, event.generation_id)
        prefix = self._snapshot_prefix(event.run_id, event.component, event.call_index, event.generation_id)
        stream_key = f"{event.run_id}:events"
        is_first_snapshot = inv not in self._initialized_snapshots
        is_first_run = event.run_id not in self._initialized_runs

        fields: dict[FieldT, EncodableT] = {
            "component": event.component,
            "call_index": str(event.call_index),
            "type": event.type.value,
            "subtype": event.subtype or "",
            "generation_id": event.generation_id or "",
            "payload": _json_dumps(serialize_value(event.payload)),
        }
        xadd_kwargs: dict = {}
        if self.event_log_max_size is not None:
            xadd_kwargs["maxlen"] = self.event_log_max_size
            xadd_kwargs["approximate"] = True

        async with self._redis.pipeline(transaction=False) as pipe:
            # XADD is always first → event_id is always results[0].
            pipe.xadd(stream_key, fields, **xadd_kwargs)
            if is_first_run:
                pipe.expire(stream_key, self._ttl_seconds)
            # Snapshot delta writes.
            if chunk.content:
                pipe.append(f"{prefix}:content", chunk.content)
            if chunk.reasoning_content:
                pipe.append(f"{prefix}:reasoning", chunk.reasoning_content)
            if chunk.tool_calls:
                for delta in chunk.tool_calls:
                    pipe.rpush(f"{prefix}:tool_deltas", _json_dumps(serialize_value(delta)))
            # First-write housekeeping: TTL on snapshot keys + register in
            # active_components so other nodes can discover this invocation.
            if is_first_snapshot:
                pipe.expire(f"{prefix}:content", self._ttl_seconds)
                pipe.expire(f"{prefix}:reasoning", self._ttl_seconds)
                pipe.expire(f"{prefix}:tool_deltas", self._ttl_seconds)
                pipe.sadd(
                    f"{event.run_id}:active_components",
                    self._active_member(event.component, event.call_index, event.generation_id),
                )
                pipe.expire(f"{event.run_id}:active_components", self._ttl_seconds)
            results = await pipe.execute()

        event.event_id = results[0]
        if is_first_snapshot:
            self._initialized_snapshots.add(inv)
        if is_first_run:
            self._initialized_runs.add(event.run_id)

    # ------------------------------------------------------------------
    # Snapshot public API
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Snapshot key helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _snapshot_prefix(run_id: str, component: str, call_index: int, generation_id: Optional[str]) -> str:
        """Redis key prefix for one generation's snapshot. Omits the segment when no token."""
        base = f"{run_id}:{component}:{call_index}"
        return f"{base}:{generation_id}" if generation_id else base

    @staticmethod
    def _active_member(component: str, call_index: int, generation_id: Optional[str]) -> str:
        """
        active_components set member encoding: ``component:call_index[:generation_id]``.

        .. warning::
            Component names and ``generation_id`` values must not contain ``:``.
            The colon is the field separator; a component named ``foo:0`` colliding
            with an unrelated invocation of ``foo`` at ``call_index=0`` would cause
            :meth:`_on_component_complete` to mis-attribute snapshots and prematurely
            release live subscribers. Haystack conventionally uses Python-identifier
            component names, so this is a natural constraint rather than an added one.
        """
        base = f"{component}:{call_index}"
        return f"{base}:{generation_id}" if generation_id else base

    # ------------------------------------------------------------------
    # Snapshot public API
    # ------------------------------------------------------------------

    async def clear_snapshot_async(
        self, run_id: str, component: str, call_index: int, generation_id: Optional[str] = None
    ) -> None:
        """
        DEL all snapshot keys for one generation and remove it from active_components.

        Cross-process readers (the relay) immediately see an empty snapshot after
        this call — no stale content served.

        :param run_id: Pipeline run identity.
        :param component: Component name.
        :param call_index: Component invocation index within the run.
        :param generation_id: Optional durable-step token.
        """
        prefix = self._snapshot_prefix(run_id, component, call_index, generation_id)
        async with self._redis.pipeline(transaction=False) as pipe:
            pipe.delete(
                f"{prefix}:content",
                f"{prefix}:reasoning",
                f"{prefix}:tool_deltas",
                f"{prefix}:retry",
            )
            pipe.srem(f"{run_id}:active_components", self._active_member(component, call_index, generation_id))
            await pipe.execute()
        self._initialized_snapshots.discard(ComponentInvocation(run_id, component, call_index, generation_id))

    async def get_snapshot(
        self, run_id: str, component: str, call_index: int, generation_id: Optional[str] = None
    ) -> StreamingSnapshot:
        """
        Read the current snapshot from Redis in one pipeline round-trip.

        Accessible from any process — the relay calls this to build catch-up events
        for reconnecting clients.

        :param run_id: Pipeline run identity.
        :param component: Component name.
        :param call_index: Component invocation index within the run.
        :param generation_id: Optional durable-step token.
        :returns: Snapshot assembled from Redis content, reasoning, tool-delta, and retry keys.
        """
        prefix = self._snapshot_prefix(run_id, component, call_index, generation_id)
        async with self._redis.pipeline(transaction=False) as pipe:
            pipe.get(f"{prefix}:content")
            pipe.get(f"{prefix}:reasoning")
            pipe.lrange(f"{prefix}:tool_deltas", 0, -1)
            pipe.get(f"{prefix}:retry")
            content, reasoning, delta_strs, retry = await pipe.execute()
        tool_call_deltas = [deserialize_value(_json_loads(d)) for d in (delta_strs or [])]
        return StreamingSnapshot(
            run_id=run_id,
            component=component,
            call_index=call_index,
            generation_id=generation_id,
            payload=StreamingSnapshot.Payload(
                content=content or "",
                reasoning_content=reasoning or "",
                tool_call_deltas=tool_call_deltas,
                retry_count=int(retry) if retry else 0,
            ),
        )

    # ------------------------------------------------------------------
    # StreamingBackend abstract method implementations
    # ------------------------------------------------------------------

    async def _persist_event(self, event: StreamingEvent) -> None:
        """
        Persist an event to the Redis stream.

        :param event: Event to persist.
        """
        stream_key = f"{event.run_id}:events"
        fields: dict[FieldT, EncodableT] = {
            "component": event.component,
            "call_index": str(event.call_index),
            "type": event.type.value,
            "subtype": event.subtype or "",
            "generation_id": event.generation_id or "",
            "payload": _json_dumps(serialize_value(event.payload)),
        }

        xadd_kwargs: dict = {}
        if self.event_log_max_size is not None:
            xadd_kwargs["maxlen"] = self.event_log_max_size
            xadd_kwargs["approximate"] = True

        # decode_responses=True on the client guarantees str returns from redis;
        # redis-py 8.x stubs declare `bytes | str`, so we cast at the boundary.
        is_first_write = event.run_id not in self._initialized_runs
        if is_first_write:
            async with self._redis.pipeline(transaction=False) as pipe:
                pipe.xadd(stream_key, fields, **xadd_kwargs)
                pipe.expire(stream_key, self._ttl_seconds)
                results = await pipe.execute()
            event.event_id = cast(str, results[0])
            self._initialized_runs.add(event.run_id)
        else:
            event.event_id = cast(str, await self._redis.xadd(stream_key, fields, **xadd_kwargs))

    async def _reset_snapshot_content_async(self, event: StreamingEvent) -> None:
        # A retry re-runs the generation from scratch: discard partial content but
        # preserve the retry counter. DEL is a no-op when keys are absent, which is
        # the correct "create at 0" behaviour (an absent :retry key reads as 0).
        #
        # We also refresh the :retry TTL here. Without this, retry_count silently
        # expires on any step whose total duration exceeds ttl_seconds: content
        # keys are recreated (and re-EXPIREd) on every retry, but :retry is only
        # touched by _increment_retry_async — so its TTL keeps counting from the
        # very first failure. A client reconnecting mid-step would then see
        # retry_count=0 despite prior failures. EXPIRE on an absent key returns 0
        # (no-op), so the call is safe on retries that begin with a start.
        prefix = self._snapshot_prefix(event.run_id, event.component, event.call_index, event.generation_id)
        async with self._redis.pipeline(transaction=False) as pipe:
            pipe.delete(
                f"{prefix}:content",
                f"{prefix}:reasoning",
                f"{prefix}:tool_deltas",
            )
            pipe.expire(f"{prefix}:retry", self._ttl_seconds)
            await pipe.execute()
        # Drop the invocation from the local first-write tracker so the next chunk
        # re-applies EXPIRE to the freshly-created keys. Without this, the retry's
        # replacement keys would live TTL-less and leak if the process crashed
        # before close_run.
        inv = ComponentInvocation(event.run_id, event.component, event.call_index, event.generation_id)
        self._initialized_snapshots.discard(inv)

    async def _increment_retry_async(self, event: StreamingEvent) -> None:
        # generation_error may arrive before any generation_start (initial-send
        # failure). INCR creates the key at 1 if absent; SADD to active_components
        # so fresh-connect discovery finds this invocation even before the first
        # token chunk. Note: we deliberately do NOT touch _initialized_snapshots
        # here — that local set only tracks whether we've already EXPIREd this
        # invocation's *content* keys, and no content keys are created in this
        # method. The Redis SADD is the source of truth for cross-node discovery.
        prefix = self._snapshot_prefix(event.run_id, event.component, event.call_index, event.generation_id)
        async with self._redis.pipeline(transaction=False) as pipe:
            pipe.incr(f"{prefix}:retry")
            pipe.expire(f"{prefix}:retry", self._ttl_seconds)
            pipe.sadd(
                f"{event.run_id}:active_components",
                self._active_member(event.component, event.call_index, event.generation_id),
            )
            pipe.expire(f"{event.run_id}:active_components", self._ttl_seconds)
            await pipe.execute()

    async def _on_component_complete(self, event: StreamingEvent) -> None:
        # Lifecycle events carry no generation_id — release every generation under
        # this component invocation by matching the active_components members and
        # batching all DEL / SREM into a single pipeline round-trip.
        #
        # Concurrency note: this fires on LIFECYCLE_COMPLETE (after the component's
        # run() returns normally) and LIFECYCLE_ERROR (when it raises). The COMPLETE
        # path is race-free — the component has already stopped. The ERROR path has
        # a narrow window where an in-flight coroutine spawned by the component
        # (e.g. a slow emit still awaiting Redis) could register a new generation
        # after our SMEMBERS snapshot; that late-registered invocation's keys then
        # leak until TTL. This is bounded by TTL (see class docstring) and
        # observability-only — the pipeline's correctness is unaffected.
        members = await cast(
            Awaitable[set[Any]],
            self._redis.smembers(f"{event.run_id}:active_components"),
        )
        base = f"{event.component}:{event.call_index}"
        matches: list[tuple[str, Optional[str]]] = []
        for member in members:
            if member == base:
                matches.append((member, None))
            elif member.startswith(f"{base}:"):
                matches.append((member, member[len(base) + 1 :]))

        if not matches:
            return

        async with self._redis.pipeline(transaction=False) as pipe:
            for member, generation_id in matches:
                prefix = self._snapshot_prefix(event.run_id, event.component, event.call_index, generation_id)
                pipe.delete(
                    f"{prefix}:content",
                    f"{prefix}:reasoning",
                    f"{prefix}:tool_deltas",
                    f"{prefix}:retry",
                )
                pipe.srem(f"{event.run_id}:active_components", member)
            await pipe.execute()

        for _, generation_id in matches:
            self._initialized_snapshots.discard(
                ComponentInvocation(event.run_id, event.component, event.call_index, generation_id)
            )

    def subscribe(
        self,
        run_id: str,
        from_event_id: Optional[str] = None,
    ) -> AsyncIterator[StreamingEvent]:
        """
        Subscribe to retained and live events for one run.

        :param run_id: Pipeline run identity.
        :param from_event_id: Optional Redis Stream entry ID cursor.
            Resume starting from this event (inclusive) if it is still available.
        :returns: Async iterator of retained and live events.
        """
        return self._subscribe(run_id, from_event_id)

    async def close_run(self, run_id: str) -> None:
        """
        End subscriptions for ``run_id`` by writing a sentinel entry to the stream
        and cleaning up the active_components set.

        Active subscriber iterators detect the sentinel and stop. The event log is
        retained (with TTL reset) for replay until the TTL expires. Late fresh
        subscriptions stop when their initial tail read finds the sentinel;
        retained-cursor subscriptions replay through the end of the run.

        :param run_id: Pipeline run identity to close.
        """
        stream_key = f"{run_id}:events"
        async with self._redis.pipeline(transaction=False) as pipe:
            pipe.xadd(stream_key, {self._SENTINEL_FIELD: "1"})
            pipe.expire(stream_key, self._ttl_seconds)
            pipe.delete(f"{run_id}:active_components")
            await pipe.execute()
        self._initialized_runs.discard(run_id)
        # Sweep local snapshot tracking for this run to prevent unbounded growth.
        # These sets are only used for EXPIRE-call deduplication — not ground truth.
        self._initialized_snapshots = {inv for inv in self._initialized_snapshots if inv.run_id != run_id}

    async def close(self) -> None:
        """Shut down the backend and close the Redis connection pool."""
        if self._closed:
            return
        self._closed = True
        await self._redis.aclose()

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

        stream_key = f"{run_id}:events"

        last_id = "0"

        # Determine if we need "fresh start / large gap" handling.
        # This happens in two cases:
        #   1. from_event_id was provided but is no longer in the buffer.
        #   2. from_event_id was not provided at all (new connection / no context).
        #
        # In both cases we want to:
        #   - Discover the current tail from the event queue.
        #   - Inject snapshots for any currently active generations (last resort).
        #   - Start live consumption from the current tail.
        do_fresh_start = False

        if from_event_id is not None:
            retained = await self._redis.xrange(stream_key, from_event_id, from_event_id, count=1)
            if retained:
                # Happy path: inclusive replay from the provided cursor
                entry_id, fields = retained[0]
                # decode_responses=True → entries are always (str, dict[str, str]);
                # cast for redis-py 8.x stubs which widened these to `bytes | str`.
                last_id = cast(str, entry_id)
                yield self._deserialize_event(run_id, last_id, cast(dict, fields))
            else:
                do_fresh_start = True
        else:
            # New connection (no from_event_id) → treat as fresh start
            do_fresh_start = True

        if do_fresh_start:
            # Fresh start / large gap path.
            # Always discover the current tail from the event queue (not snapshots).
            active_members: set[Any] = await cast(
                Awaitable[set[Any]],
                self._redis.smembers(f"{run_id}:active_components"),
            )
            # Member format: "component:call_index" or "component:call_index:generation_id".
            active = []
            for member in active_members:
                parts = member.split(":", 2)
                component = parts[0]
                call_index = int(parts[1])
                generation_id = parts[2] if len(parts) == 3 else None
                active.append(ComponentInvocation(run_id, component, call_index, generation_id))

            async with self._redis.pipeline(transaction=True) as pipe:
                pipe.xrevrange(stream_key, "+", "-", count=1)  # current tail from event queue
                for inv in active:
                    prefix = self._snapshot_prefix(inv.run_id, inv.component, inv.call_index, inv.generation_id)
                    pipe.get(f"{prefix}:content")
                    pipe.get(f"{prefix}:reasoning")
                    pipe.lrange(f"{prefix}:tool_deltas", 0, -1)
                    pipe.get(f"{prefix}:retry")
                results = await pipe.execute()

            tail_entries = results[0]
            if tail_entries and self._SENTINEL_FIELD in tail_entries[0][1]:
                # The run closed before the fresh subscription captured its tail.
                # Starting XREAD after this sentinel would wait indefinitely.
                return
            last_id = tail_entries[0][0] if tail_entries else "0"

            for i, inv in enumerate(active):
                base = 1 + i * 4
                content = results[base] or ""
                reasoning = results[base + 1] or ""
                tool_call_deltas = [deserialize_value(_json_loads(d)) for d in (results[base + 2] or [])]
                retry = results[base + 3]
                yield StreamingSnapshot(
                    run_id=inv.run_id,
                    component=inv.component,
                    call_index=inv.call_index,
                    generation_id=inv.generation_id,
                    payload=StreamingSnapshot.Payload(
                        content=content,
                        reasoning_content=reasoning,
                        tool_call_deltas=tool_call_deltas,
                        retry_count=int(retry) if retry else 0,
                    ),
                )

        while not self._closed:
            try:
                # block=100: wait up to 100 ms for new entries so that close()
                # is noticed within one polling cycle rather than blocking forever.
                # redis-py 8.x widens xread's return to a union of list/dict
                # shapes; we always call it with a single stream and no group,
                # which is the list form.
                results = cast(
                    list,
                    await self._redis.xread(
                        {stream_key: last_id},
                        count=100,
                        block=100,
                    ),
                )
            except Exception:
                if self._closed:
                    return
                raise

            if not results:
                # XREAD timed out — no new entries. Recheck _closed and loop.
                continue

            for _, entries in results:
                for entry_id, fields in entries:
                    if self._SENTINEL_FIELD in fields:
                        # close_run() sentinel — stop the iterator.
                        return
                    last_id = entry_id
                    yield self._deserialize_event(run_id, entry_id, fields)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _deserialize_event(self, run_id: str, entry_id: str, fields: dict) -> StreamingEvent:
        payload_raw = _json_loads(fields["payload"])
        return StreamingEvent(
            run_id=run_id,
            component=fields["component"],
            call_index=int(fields["call_index"]),
            type=EventType(fields["type"]),
            subtype=fields["subtype"] or None,
            payload=deserialize_value(payload_raw),
            event_id=entry_id,
            generation_id=fields.get("generation_id") or None,
        )
