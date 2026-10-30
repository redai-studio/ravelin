"""
This module provides the HTTPX client lifecycle shared by HTTPX-based assets. Creating an HTTPX client is expensive
(building its SSL context loads the CA bundle), so assets reuse their clients across calls instead of creating one per
request.
"""

import asyncio
import threading
from typing import Any, Dict, Optional

import httpx

_CLIENT_LOCK = threading.Lock()


class _HttpxClientMixin:
    """
    A mixin that lazily creates and reuses HTTPX clients for an asset.

    The synchronous client is created on first use and shared across calls. Asynchronous clients are created on first
    use per event loop, since an ``httpx.AsyncClient`` connection pool cannot be shared across loops: each running loop
    gets its own client, which is reused for every call made on that loop.

    Lifecycle: clients live as long as the asset and are meant to be reused, so long-lived assets on long-lived event
    loops need no explicit cleanup until shutdown, where ``close()`` and ``await aclose()`` release them. When an asset
    is driven from short-lived event loops (e.g. ``asyncio.run`` per call), call ``await aclose()`` before each loop
    ends: once a loop is closed, its client can no longer be closed gracefully. Clients of closed loops are dropped on
    the next use of the asset, and their connections are released only when they are garbage collected.

    Subclasses must provide ``client_kwargs``.
    """

    _httpx_client: Optional[httpx.Client] = None
    _httpx_async_clients: Optional[Dict[asyncio.AbstractEventLoop, httpx.AsyncClient]] = None
    _httpx_unbound_async_client: Optional[httpx.AsyncClient] = None

    @property
    def client_kwargs(self) -> Dict[str, Any]:
        """
        Get the client keyword arguments for the HTTPX client.

        :return: The client keyword arguments.
        """
        raise NotImplementedError("client_kwargs must be implemented")

    @property
    def _client(self) -> httpx.Client:
        """
        Get the reusable synchronous HTTPX client, creating it on first use.

        :return: The synchronous HTTPX client.
        """
        if self._httpx_client is None:
            with _CLIENT_LOCK:
                if self._httpx_client is None:
                    self._httpx_client = httpx.Client(**self.client_kwargs)
        return self._httpx_client

    @_client.setter
    def _client(self, client: httpx.Client) -> None:
        self._httpx_client = client

    @_client.deleter
    def _client(self) -> None:
        self._httpx_client = None

    def _live_async_clients(self) -> Dict[asyncio.AbstractEventLoop, httpx.AsyncClient]:
        """
        Get the asynchronous clients per event loop, dropping those of closed loops. Must be called with the lock held.

        :return: The asynchronous clients keyed by their event loop.
        """
        if self._httpx_async_clients is None:
            self._httpx_async_clients = {}
        for loop in [loop for loop in self._httpx_async_clients if loop.is_closed()]:
            del self._httpx_async_clients[loop]
        return self._httpx_async_clients

    @property
    def _async_client(self) -> httpx.AsyncClient:
        """
        Get the reusable asynchronous HTTPX client of the running event loop, creating it on first use.

        :return: The asynchronous HTTPX client.
        :raises RuntimeError: If there is no running event loop.
        """
        loop = asyncio.get_running_loop()
        with _CLIENT_LOCK:
            clients = self._live_async_clients()
            client = clients.get(loop)
            if client is None:
                client, self._httpx_unbound_async_client = self._httpx_unbound_async_client, None
                clients[loop] = client = client or httpx.AsyncClient(**self.client_kwargs)
            return client

    @_async_client.setter
    def _async_client(self, client: httpx.AsyncClient) -> None:
        """Replace the asynchronous clients with the given one, which is bound to the next event loop that uses it."""
        with _CLIENT_LOCK:
            self._httpx_async_clients = None
            self._httpx_unbound_async_client = client

    @_async_client.deleter
    def _async_client(self) -> None:
        with _CLIENT_LOCK:
            self._httpx_async_clients = None
            self._httpx_unbound_async_client = None

    @staticmethod
    def _apply_timeout(request: httpx.Request, timeout: Optional[float]) -> httpx.Request:
        """
        Set the timeout of a single request, leaving the client's default timeout untouched.

        :param request: The request to update.
        :param timeout: The timeout for the request. ``None`` disables the timeout.
        :return: The updated request.
        """
        request.extensions["timeout"] = httpx.Timeout(timeout).as_dict()
        return request

    def close(self) -> None:
        """Close the reusable synchronous HTTPX client. A new client is created if the asset is used again."""
        with _CLIENT_LOCK:
            client, self._httpx_client = self._httpx_client, None
        if client is not None:
            client.close()

    async def aclose(self) -> None:
        """
        Close the reusable asynchronous HTTPX client of the running event loop. A new client is created if the asset is
        used again on this loop. Clients of other loops are left untouched, since they can only be closed on their own
        loop.
        """
        loop = asyncio.get_running_loop()
        with _CLIENT_LOCK:
            client = self._live_async_clients().pop(loop, None)
            if client is None:
                client, self._httpx_unbound_async_client = self._httpx_unbound_async_client, None
        if client is not None:
            await client.aclose()
