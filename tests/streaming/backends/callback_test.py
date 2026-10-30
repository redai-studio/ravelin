"""Tests for CallbackStreamingBackend."""

import unittest

from ravelin.streaming.backends.callback import CallbackStreamingBackend
from ravelin.streaming.event import EventType, StreamingEvent


class TestCallbackStreamingBackend(unittest.TestCase):
    def test_callback_receives_event(self):
        collected: list[StreamingEvent] = []
        backend = CallbackStreamingBackend(on_event=collected.append)

        event = StreamingEvent(
            run_id="r1",
            component="llm",
            call_index=0,
            type=EventType.TOKEN_STREAM,
            subtype=None,
            payload="hello",
        )
        backend.write_event_sync(event)

        self.assertEqual(1, len(collected))
        self.assertIs(event, collected[0])


if __name__ == "__main__":
    unittest.main()
