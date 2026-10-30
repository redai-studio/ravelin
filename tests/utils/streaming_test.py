import unittest
from unittest.mock import patch

import httpx

from ravelin.utils.streaming import ServerSentEvent, SSEDecoder, ServerSentEventStream, AsyncServerSentEventStream


class SimpleAsyncIter:
    def __init__(self, items):
        self.items = items

    async def __anext__(self):
        try:
            return self.items.pop(0)
        except IndexError:
            raise StopAsyncIteration

    def __aiter__(self):
        return self


class TestServerSentEvent(unittest.TestCase):
    def test_properties(self):
        sse = ServerSentEvent(event="test_event", data="test_data", id="test_id", retry=123)
        self.assertEqual(sse.event, "test_event")
        self.assertEqual(sse.data, "test_data")
        self.assertEqual(sse.id, "test_id")
        self.assertEqual(sse.retry, 123)

    def test_json(self):
        sse = ServerSentEvent(data='{"key": "value"}')
        self.assertEqual(sse.json(), {"key": "value"})


class TestSSEDecoder(unittest.TestCase):
    def test_iter_bytes(self):
        decoder = SSEDecoder()
        iterator = iter([b"event: test_event\ndata: test_data\n\n"])
        events = list(decoder.iter_bytes(iterator))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event, "test_event")
        self.assertEqual(events[0].data, "test_data")

    def test_decode(self):
        decoder = SSEDecoder()
        self.assertIsNone(decoder.decode("event: test_event"))
        self.assertIsNone(decoder.decode("data: test_data"))
        self.assertIsNone(decoder.decode("id: test_id"))
        self.assertIsNone(decoder.decode("id: \0abc"))
        self.assertIsNone(decoder.decode("retry: 123"))
        sse = decoder.decode("")
        self.assertEqual(sse.event, "test_event")
        self.assertEqual(sse.data, "test_data")
        self.assertEqual(sse.id, "test_id")
        self.assertEqual(sse.retry, 123)


class TestServerSentEventStream(unittest.TestCase):
    @patch("httpx.Response")
    def test_next(self, mock_response):
        mock_response.iter_bytes.return_value = iter([b"event: test_event\ndata: test_data\n\n"])
        stream = ServerSentEventStream(response=mock_response)
        sse = next(stream)
        self.assertEqual(sse.event, "test_event")
        self.assertEqual(sse.data, "test_data")

    @patch("httpx.Response")
    def test_iter(self, mock_response):
        mock_response.iter_bytes.return_value = iter([b"event: test_event\ndata: test_data\n\n"])
        stream = ServerSentEventStream(response=mock_response)
        events = list(iter(stream))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event, "test_event")
        self.assertEqual(events[0].data, "test_data")

    @patch("httpx.Response")
    def test_stream_error(self, mock_response):
        mock_response.iter_bytes.return_value = iter([b'event: error\ndata: {"error": {"message": "test_error"}}\n\n'])
        stream = ServerSentEventStream(response=mock_response)
        with self.assertRaisesRegex(httpx.StreamError, "test_error"):
            next(stream)

        mock_response.iter_bytes.return_value = iter([b'event: error\ndata: {"error": {"foo": "bar"}}\n\n'])
        stream = ServerSentEventStream(response=mock_response)
        with self.assertRaisesRegex(httpx.StreamError, "error message not found"):
            next(stream)


class TestAsyncServerSentEventStream(unittest.IsolatedAsyncioTestCase):
    @patch("httpx.Response")
    async def test_anext(self, mock_response):
        mock_response.aiter_bytes.return_value = SimpleAsyncIter([b"event: test_event\ndata: test_data\n\n"])
        stream = AsyncServerSentEventStream(response=mock_response)
        sse = await stream.__anext__()
        self.assertEqual(sse.event, "test_event")
        self.assertEqual(sse.data, "test_data")

    @patch("httpx.Response")
    async def test_aiter(self, mock_response):
        mock_response.aiter_bytes.return_value = SimpleAsyncIter([b"event: test_event\ndata: test_data\n\n"])
        stream = AsyncServerSentEventStream(response=mock_response)
        events = [event async for event in stream]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event, "test_event")
        self.assertEqual(events[0].data, "test_data")

    @patch("httpx.Response")
    async def test_stream_error(self, mock_response):
        mock_response.aiter_bytes.return_value = SimpleAsyncIter(
            [b'event: error\ndata: {"error": {"message": "test_error"}}\n\n']
        )
        stream = AsyncServerSentEventStream(response=mock_response)
        with self.assertRaisesRegex(httpx.StreamError, "test_error"):
            await stream.__anext__()

        mock_response.aiter_bytes.return_value = SimpleAsyncIter([b'event: error\ndata: {"error": {"foo": "bar"}}\n\n'])
        stream = AsyncServerSentEventStream(response=mock_response)
        with self.assertRaisesRegex(httpx.StreamError, "error message not found"):
            await stream.__anext__()


class TestSSEDecoderAsync(unittest.IsolatedAsyncioTestCase):
    async def test_aiter_bytes(self):
        decoder = SSEDecoder()
        iterator = SimpleAsyncIter([b"event: test_event\ndata: test_data\n\n"])
        events = [event async for event in decoder.aiter_bytes(iterator)]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event, "test_event")
        self.assertEqual(events[0].data, "test_data")


if __name__ == "__main__":
    unittest.main()
