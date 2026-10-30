"""Framework-agnostic streaming relay — event delivery layer for pipeline developers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Optional, Set

from ravelin.core.serialization import serialize_value
from ravelin.streaming.backends.base import StreamingBackend
from ravelin.streaming.event import EventType, StreamingEvent

try:
    import orjson as _orjson

    def _json_dumps(obj: Any) -> str:
        return _orjson.dumps(obj).decode()

except ImportError:
    import json as _json_stdlib

    def _json_dumps(obj: Any) -> str:
        return _json_stdlib.dumps(obj)


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------


@dataclass
class EventFilter:
    """
    Visibility constraint for relay.stream() / relay.stream_sse().

    ``None`` on a field means "no constraint on that dimension" (all values pass).

    Two filters are combined by intersection: the relay's ``scope`` sets the
    upper bound of what clients may see; a per-call ``filter`` narrows within
    that scope. ``None`` is treated as the universal set, so intersecting with
    ``None`` defers to the other side.

    Example::

        # Pipeline developer restricts clients to lifecycle events only.
        relay = StreamingRelay(backend, scope=EventFilter(types={EventType.LIFECYCLE}))

        # Client further narrows to a single component.
        async for event in relay.stream(run_id, filter=EventFilter(components={"llm"})):
            ...
        # Effective filter: types={LIFECYCLE}, components={"llm"}
    """

    types: Optional[Set[EventType]] = None  # None = all event types
    components: Optional[Set[str]] = None  # None = all components

    def matches(self, event: StreamingEvent) -> bool:
        """
        Return True if the event passes this filter.

        :param event: Event to test.
        :returns: Whether the event is visible under this filter.
        """
        if self.types is not None and event.type not in self.types:
            return False
        if self.components is not None and event.component not in self.components:
            return False
        return True


def _intersect(a: Optional[set], b: Optional[set]) -> Optional[set]:
    if a is None:
        return b
    if b is None:
        return a
    return a & b


# ---------------------------------------------------------------------------
# Default serializer
# ---------------------------------------------------------------------------


def _default_serializer(event: StreamingEvent) -> str:
    """
    JSON-serialize a StreamingEvent for SSE delivery.

    Manually flattens the event envelope fields so that EventType (str Enum)
    and init=False fields on StreamingSnapshot are handled correctly without
    relying on dataclasses.asdict. Uses serialize_value() for payload, which
    handles Haystack types (ToolCallDelta, StreamingChunk, etc.) via to_dict().

    :param event: Event to serialize.
    :returns: JSON string suitable for an SSE ``data:`` line.
    """
    return _json_dumps(
        {
            "run_id": event.run_id,
            "component": event.component,
            "call_index": event.call_index,
            "type": event.type.value,
            "subtype": event.subtype,
            "event_id": event.event_id,
            "generation_id": event.generation_id,
            "payload": serialize_value(event.payload),
        }
    )


# ---------------------------------------------------------------------------
# Relay
# ---------------------------------------------------------------------------


class StreamingRelay:
    """
    Framework-agnostic streaming relay.

    Wraps a ``StreamingBackend`` and provides two async generator methods for
    event delivery. Pipeline developers embed these in their own endpoints — the
    relay has no dependency on any web framework.

    **Delivery methods**:

    - ``stream()``: yields ``StreamingEvent`` objects. Use for WebSocket handlers
      or any transport that needs raw events.
    - ``stream_sse()``: thin wrapper over ``stream()`` that yields complete SSE
      chunks (``id:``, ``data:``, blank line). Pipe directly into a
      ``StreamingResponse`` or equivalent.

    **Filtering** is two-layered:

    - ``scope`` (constructor): visibility upper bound set by the pipeline developer.
      Clients cannot see events outside this scope regardless of what they request.
    - ``filter`` (per-call): client-driven narrowing within the scope, applied via
      intersection. ``None`` means no additional constraint.

    **Reconnect**: pass the client's ``Last-Event-ID`` header value as ``from_event_id``.
    The backend replays from that retained cursor inclusively before delivering live
    events. The client must deduplicate or replace the cursor event at the boundary.

    Example (FastAPI)::

        relay = StreamingRelay(backend)

        @app.get("/runs/{run_id}/events")
        async def events(run_id: str, request: Request):
            from_event_id = request.headers.get("last-event-id")
            return StreamingResponse(
                relay.stream_sse(run_id, from_event_id=from_event_id),
                media_type="text/event-stream",
            )
    """

    def __init__(
        self,
        backend: StreamingBackend,
        serializer: Callable[[StreamingEvent], str] = _default_serializer,
        scope: Optional[EventFilter] = None,
    ) -> None:
        """
        :param backend: The streaming backend to subscribe from.
        :param serializer: Converts a StreamingEvent to the SSE ``data:`` payload string.
            Default: JSON-serializes the event envelope + payload.
        :param scope: Visibility upper bound. Clients cannot receive events outside
            this filter regardless of what they pass to stream()/stream_sse().
            None (default) = no restriction.
        """
        self._backend = backend
        self._serializer = serializer
        self._scope = scope

    async def stream(
        self,
        run_id: str,
        from_event_id: Optional[str] = None,
        event_filter: Optional[EventFilter] = None,
    ) -> AsyncIterator[StreamingEvent]:
        """
        Yield StreamingEvents for ``run_id``, filtered and cursor-aware.

        :param run_id: The pipeline run to subscribe to.
        :param from_event_id: Resume starting from this event ID (inclusive).
            Pass the client's Last-Event-ID on SSE reconnect to resume without gaps.
            If the event is still present in the short-term buffer, it will be
            re-delivered as the first event.
        :param event_filter: Additional narrowing applied on top of the relay's scope,
            via intersection. None = no additional constraint.
        """
        effective = self._effective_filter(event_filter)
        async for event in self._backend.subscribe(run_id, from_event_id):
            if effective is None or effective.matches(event):
                yield event

    async def stream_sse(
        self,
        run_id: str,
        from_event_id: Optional[str] = None,
        event_filter: Optional[EventFilter] = None,
    ) -> AsyncIterator[str]:
        """
        Yield SSE-formatted strings for ``run_id``.

        Thin wrapper over ``stream()``. Each yielded string is a complete SSE
        chunk ready to write to an HTTP streaming response::

            id: <event_id>
            data: <serialized payload>
            <blank line>

        Pipe directly into your framework's StreamingResponse. For WebSocket or
        custom transports, use ``stream()`` and call ``format_sse()`` selectively.

        :param run_id: The pipeline run to subscribe to.
        :param from_event_id: Resume starting from this event ID (inclusive).
        :param event_filter: Additional narrowing applied on top of the relay's scope.
        :returns: Async iterator of SSE-formatted chunks.
        """
        async for event in self.stream(run_id, from_event_id=from_event_id, event_filter=event_filter):
            yield self.format_sse(event)
        # Standard SSE practice: a trailing comment flushes any server-side
        # send buffer (ASGI server, reverse proxy) before the connection closes,
        # ensuring all real events reach the client. Clients silently ignore
        # comment lines per the SSE spec.
        yield ": end\n\n"

    def format_sse(self, event: StreamingEvent) -> str:
        """
        Format a single event as a complete SSE chunk string.

        Exposed for callers that iterate via ``stream()`` but occasionally need
        SSE formatting (e.g. injecting synthetic events before the live stream).

        :param event: Event to format.
        :returns: Complete SSE chunk ending in a blank line.
        """
        prefix = f"id: {event.event_id}\n" if event.event_id is not None else ""
        return f"{prefix}data: {self._serializer(event)}\n\n"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _effective_filter(self, call_filter: Optional[EventFilter]) -> Optional[EventFilter]:
        """Merge scope and per-call filter via intersection. Returns None if unconstrained."""
        if self._scope is None and call_filter is None:
            return None
        scope = self._scope or EventFilter()
        flt = call_filter or EventFilter()
        merged = EventFilter(
            types=_intersect(scope.types, flt.types),
            components=_intersect(scope.components, flt.components),
        )
        # Fast-path: fully unconstrained after merge.
        if merged.types is None and merged.components is None:
            return None
        return merged
