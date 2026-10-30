import asyncio
import unittest
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from ravelin.assets._httpx import _HttpxClientMixin


class _Service(_HttpxClientMixin):
    @property
    def client_kwargs(self) -> Dict[str, Any]:
        return {"headers": {"X-Test": "1"}}


class TestHttpxClientMixin(unittest.TestCase):
    def test_client_kwargs_must_be_implemented(self):
        with self.assertRaises(NotImplementedError):
            _ = _HttpxClientMixin().client_kwargs

    def test_client_is_created_with_client_kwargs(self):
        with patch("httpx.Client") as mock_client_cls:
            service = _Service()
            self.assertIs(mock_client_cls.return_value, service._client)
            mock_client_cls.assert_called_once_with(headers={"X-Test": "1"})

    def test_injected_client_is_used_and_can_be_deleted(self):
        service = _Service()
        injected = MagicMock()
        service._client = injected
        self.assertIs(injected, service._client)

        del service._client
        with patch("httpx.Client") as mock_client_cls:
            self.assertIs(mock_client_cls.return_value, service._client)
        injected.close.assert_not_called()

    def test_close_and_aclose_without_clients_are_noops(self):
        service = _Service()
        service.close()
        asyncio.run(service.aclose())

    def test_async_client_is_recreated_for_a_different_event_loop(self):
        service = _Service()

        async def get_client():
            return service._async_client

        with patch("httpx.AsyncClient", side_effect=lambda **_: MagicMock()) as mock_client_cls:
            first = asyncio.run(get_client())
            second = asyncio.run(get_client())

        self.assertIsNot(first, second)
        self.assertEqual(2, mock_client_cls.call_count)

    def test_async_client_is_reused_within_an_event_loop(self):
        service = _Service()

        async def get_clients():
            return service._async_client, service._async_client

        with patch("httpx.AsyncClient") as mock_client_cls:
            first, second = asyncio.run(get_clients())

        self.assertIs(first, second)
        mock_client_cls.assert_called_once_with(headers={"X-Test": "1"})

    def test_injected_async_client_binds_to_the_first_event_loop(self):
        service = _Service()
        injected = MagicMock()
        service._async_client = injected

        async def get_client():
            return service._async_client

        with patch("httpx.AsyncClient") as mock_client_cls:
            self.assertIs(injected, asyncio.run(get_client()))
            self.assertIs(mock_client_cls.return_value, asyncio.run(get_client()))

        del service._async_client
        with patch("httpx.AsyncClient", side_effect=lambda **_: MagicMock()):
            self.assertIsNot(injected, asyncio.run(get_client()))

    def test_async_clients_are_stable_per_live_event_loop(self):
        service = _Service()

        async def get_client():
            return service._async_client

        first_loop, second_loop = asyncio.new_event_loop(), asyncio.new_event_loop()
        try:
            with patch("httpx.AsyncClient", side_effect=lambda **_: MagicMock()) as mock_client_cls:
                first = first_loop.run_until_complete(get_client())
                second = second_loop.run_until_complete(get_client())
                # Alternating between live loops reuses each loop's client
                self.assertIs(first, first_loop.run_until_complete(get_client()))
                self.assertIs(second, second_loop.run_until_complete(get_client()))
            self.assertIsNot(first, second)
            self.assertEqual(2, mock_client_cls.call_count)
        finally:
            first_loop.close()
            second_loop.close()

    def test_clients_of_closed_event_loops_are_dropped(self):
        service = _Service()

        async def get_client():
            return service._async_client

        with patch("httpx.AsyncClient", side_effect=lambda **_: MagicMock()):
            for _ in range(3):
                asyncio.run(get_client())

        # Each run drops the clients of the previously closed loops, keeping only the latest one
        self.assertEqual(1, len(service._httpx_async_clients))

    def test_aclose_only_closes_the_running_loop_client(self):
        service = _Service()

        async def get_client():
            return service._async_client

        async def aclose():
            await service.aclose()

        def make_client(**_):
            client = MagicMock()
            client.aclose = AsyncMock()
            return client

        first_loop, second_loop = asyncio.new_event_loop(), asyncio.new_event_loop()
        try:
            with patch("httpx.AsyncClient", side_effect=make_client):
                first = first_loop.run_until_complete(get_client())
                second = second_loop.run_until_complete(get_client())
                first_loop.run_until_complete(aclose())
                first.aclose.assert_awaited_once_with()
                second.aclose.assert_not_awaited()
                self.assertIs(second, second_loop.run_until_complete(get_client()))
                # A new client is created when the closed loop's asset is used again
                self.assertIsNot(first, first_loop.run_until_complete(get_client()))
        finally:
            first_loop.close()
            second_loop.close()

    def test_aclose_closes_an_unbound_injected_client(self):
        service = _Service()
        injected = MagicMock()
        injected.aclose = AsyncMock()
        service._async_client = injected

        asyncio.run(service.aclose())

        injected.aclose.assert_awaited_once_with()

    def test_apply_timeout_sets_request_timeout(self):
        request = httpx.Request("GET", "https://example.com")

        _HttpxClientMixin._apply_timeout(request, 3.0)
        self.assertEqual(httpx.Timeout(3.0).as_dict(), request.extensions["timeout"])

        _HttpxClientMixin._apply_timeout(request, None)
        self.assertEqual(httpx.Timeout(None).as_dict(), request.extensions["timeout"])


if __name__ == "__main__":
    unittest.main()
