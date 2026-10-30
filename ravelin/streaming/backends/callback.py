"""CallbackStreamingBackend — simple sync-capable backend for scripts and tests."""

from typing import Any, Callable

from ravelin.streaming.backends.base import SyncCapableBackend
from ravelin.streaming.event import StreamingEvent


class CallbackStreamingBackend(SyncCapableBackend):
    """
    Streaming backend that fires a callable for each event inline.

    No asyncio.Queue, no subscribers, no event log — events are delivered
    directly to on_event as they arrive. Works in both sync and async contexts.
    One instance serves as a service-level singleton across all pipeline runs;
    the callback receives events from every run and can use ``event.run_id`` to
    filter if needed.

    Intended for:
      - Terminal scripts: print tokens as they stream
      - Logging / debugging: record events to a log
      - Testing: collect events in a list for assertions

    Usage (async pipeline)::

        events = []
        backend = CallbackStreamingBackend(on_event=events.append)
        async with streaming_session_async(run_id, backend):
            await pipeline.run_async(data)

    Usage (sync pipeline)::

        backend = CallbackStreamingBackend(on_event=lambda e: print(e.payload))
        with streaming_session(run_id, backend):
            pipeline.run(data)

    Note: on_event is called synchronously, inline with event emission.
    Keep it fast — queue internally for any expensive work.
    """

    def __init__(
        self,
        on_event: Callable[[StreamingEvent], Any],
    ) -> None:
        """
        Initialize the callback backend.

        :param on_event: Callable invoked synchronously for each emitted event.
        """
        super().__init__()
        self._on_event = on_event

    def _persist_event_sync(self, event: StreamingEvent) -> None:
        """
        Deliver an event to the configured callback.

        :param event: Event to deliver.
        """
        self._on_event(event)

    # subscribe(), close_run(), and close() inherited from SyncCapableBackend (no-ops)
