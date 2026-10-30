"""
This module defines the base class for chat model services. A chat model service is a deployed model (or a selection of
models) that can be used for generating replies to chat message inputs.
"""

import abc
import asyncio
import logging
from datetime import datetime
from typing import Any, AsyncIterable, Dict, Iterable, List, Mapping, Optional, Union

import httpx
from haystack.tools import Tool, Toolset
from tenacity import (
    AsyncRetrying,
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from ravelin.assets.model_services._base import _HttpModelService, _ModelService
from ravelin.assets._base import SecretValue
from ravelin.assets._httpx import _HttpxClientMixin
from ravelin.core.restate.context import has_restate_context
from ravelin.streaming import EventType, get_component_context
from ravelin.streaming.context import get_generation_id, reset_generation_id, set_generation_id
from ravelin.streaming.event import GENERATION_COMPLETE, GENERATION_DELTA, GENERATION_ERROR, GENERATION_START
from ravelin.dataclasses import ChatMessage, StreamingChunk

logger = logging.getLogger(__name__)


class _ChatModelService(_ModelService, metaclass=abc.ABCMeta):
    """
    An abstract base class of a generic service for managing and interacting with chat models.
    """

    @abc.abstractmethod
    # pylint: disable=arguments-differ
    def infer(
        self,
        messages: List[ChatMessage],
        *,
        model: Optional[str],
        timeout: Optional[float],
        max_retries: int = 3,
        stream: bool = False,
        tools: Optional[Union[List[Tool], Toolset]] = None,
        inference_kwargs: Optional[Dict[str, Any]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> List[ChatMessage]:
        """
        Infer responses to the given messages using the model service.

        :param messages: The list of ChatMessage instances to send to the model.
        :param model: The model to use for the request.
        :param timeout: The timeout for the request.
        :param max_retries: Maximum number of retry attempts for retryable errors. Defaults to 3.
        :param stream: Whether to stream the response.
        :param tools: A list of tools to include in the request.
        :param inference_kwargs: Additional generation keyword arguments to include in the request.
        :param extra_headers: Additional headers to include in the request.
        :param kwargs: Additional keyword arguments to include in the request.
        :return: The list of ChatMessage instances containing the generated responses.
        """
        raise NotImplementedError("infer method must be implemented")

    async def ainfer(
        self,
        messages: List[ChatMessage],
        *,
        model: Optional[str],
        timeout: Optional[float],
        max_retries: int = 3,
        stream: bool = False,
        tools: Optional[Union[List[Tool], Toolset]] = None,
        inference_kwargs: Optional[Dict[str, Any]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> List[ChatMessage]:
        """
        Asynchronously infer responses to the given messages using the model service.

        :param messages: The list of ChatMessage instances to send to the model.
        :param model: The model to use for the request.
        :param timeout: The timeout for the request.
        :param max_retries: Maximum number of retry attempts for retryable errors. Defaults to 3.
        :param stream: Whether to stream the response.
        :param tools: A list of tools to include in the request.
        :param inference_kwargs: Additional generation keyword arguments to include in the request.
        :param extra_headers: Additional headers to include in the request.
        :param kwargs: Additional keyword arguments to include in the request.
        :return: The list of ChatMessage instances containing the generated responses.
        """
        # By default, we run the synchronous method in a separate thread
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.infer(
                messages,
                model=model,
                timeout=timeout,
                max_retries=max_retries,
                stream=stream,
                tools=tools,
                inference_kwargs=inference_kwargs,
                extra_headers=extra_headers,
                **kwargs,
            ),
        )

    def _collect_streaming_chunks(self, chunk_iter: Iterable[StreamingChunk]) -> List[StreamingChunk]:
        """
        Consume a synchronous stream of chunks, emitting one ``GENERATION_DELTA``
        TOKEN_STREAM event per chunk.

        Generation lifecycle events (``generation_start`` / ``generation_complete`` /
        ``generation_error``) are emitted by :meth:`infer`, which owns the full
        send → parse → collect flow. Failures that surface here — mid-stream
        chunk errors — propagate to that outer scope and become
        ``generation_error`` there.

        :param chunk_iter: Synchronous iterable of StreamingChunk objects.
        :returns: The collected list of StreamingChunk objects.
        """
        comp_ctx = get_component_context()
        generation_id = get_generation_id()
        chunks: List[StreamingChunk] = []
        for chunk in chunk_iter:
            if not chunks:
                chunk.meta["completion_start_time"] = datetime.now().isoformat()
            if comp_ctx:
                comp_ctx.emit(EventType.TOKEN_STREAM, chunk, subtype=GENERATION_DELTA, generation_id=generation_id)
            chunks.append(chunk)
        return chunks

    async def _collect_streaming_chunks_async(self, chunk_iter: AsyncIterable[StreamingChunk]) -> List[StreamingChunk]:
        """
        Consume an asynchronous stream of chunks, emitting one ``GENERATION_DELTA``
        TOKEN_STREAM event per chunk.

        Async counterpart of :meth:`_collect_streaming_chunks`. Lifecycle events
        are owned by :meth:`ainfer`.

        :param chunk_iter: Asynchronous iterable of StreamingChunk objects.
        :returns: The collected list of StreamingChunk objects.
        """
        comp_ctx = get_component_context()
        generation_id = get_generation_id()
        chunks: List[StreamingChunk] = []
        async for chunk in chunk_iter:
            if not chunks:
                chunk.meta["completion_start_time"] = datetime.now().isoformat()
            if comp_ctx:
                await comp_ctx.emit_async(
                    EventType.TOKEN_STREAM, chunk, subtype=GENERATION_DELTA, generation_id=generation_id
                )
            chunks.append(chunk)
        return chunks

    # The four generation-lifecycle emitters below are *best-effort*: any
    # exception raised by the streaming backend (e.g. transient Redis outage
    # while writing the terminal event) is caught and logged, never propagated.
    # Rationale: streaming events are journal-exempt, generation-scoped
    # observability — allowing them to fail an otherwise-successful inference,
    # or to mask the original inference exception on the error path, would let
    # a transient snapshot-backend problem corrupt the primary caller's result.
    # A dropped lifecycle event degrades observability for that one attempt;
    # subscribers reconverge on the next LIFECYCLE_COMPLETE/ERROR at the
    # component boundary and via TTL as a final backstop.

    @staticmethod
    def _emit_generation_start() -> None:
        """
        Emit a ``generation_start`` TOKEN_STREAM event.

        Called by :meth:`infer` at the top of the send → parse → collect flow.
        No-op when no streaming session is active. Best-effort — see the
        block comment above.
        """
        comp_ctx = get_component_context()
        if comp_ctx is None:
            return
        try:
            comp_ctx.emit(EventType.TOKEN_STREAM, None, subtype=GENERATION_START, generation_id=get_generation_id())
        except Exception:  # pylint: disable=broad-exception-caught
            logger.exception("Failed to emit generation_start; continuing")

    @staticmethod
    async def _emit_generation_start_async() -> None:
        """Async counterpart of :meth:`_emit_generation_start`."""
        comp_ctx = get_component_context()
        if comp_ctx is None:
            return
        try:
            await comp_ctx.emit_async(
                EventType.TOKEN_STREAM, None, subtype=GENERATION_START, generation_id=get_generation_id()
            )
        except Exception:  # pylint: disable=broad-exception-caught
            logger.exception("Failed to emit generation_start; continuing")

    @staticmethod
    def _emit_generation_error(error: BaseException) -> None:
        """
        Emit a ``generation_error`` TOKEN_STREAM event for a failed inference attempt.

        Called by :meth:`infer` from a single outer try/except that surrounds the
        request send, response parse, and streaming collector. Every failure in
        that region — non-streaming JSON errors, request-send failures, mid-stream
        chunk errors, post-collector validation errors — routes here once. As a
        TOKEN_STREAM event this is structurally journal-exempt: transient,
        generation-scoped, never persisted (under Restate it is emitted inside a
        retried step).

        Best-effort. An emit failure here would replace the caller's original
        inference exception with a snapshot-backend exception; instead we log
        and return so the caller sees the real inference error.

        No-op when no streaming session is active.

        :param error: The exception that caused the inference attempt to fail.
        """
        comp_ctx = get_component_context()
        if comp_ctx is None:
            return
        try:
            comp_ctx.emit(
                EventType.TOKEN_STREAM,
                {"error": str(error), "error_type": type(error).__name__},
                subtype=GENERATION_ERROR,
                generation_id=get_generation_id(),
            )
        except Exception:  # pylint: disable=broad-exception-caught
            logger.exception("Failed to emit generation_error; original inference error preserved")

    @staticmethod
    async def _emit_generation_error_async(error: BaseException) -> None:
        """
        Async counterpart of :meth:`_emit_generation_error`.

        :param error: The exception that caused the inference attempt to fail.
        """
        comp_ctx = get_component_context()
        if comp_ctx is None:
            return
        try:
            await comp_ctx.emit_async(
                EventType.TOKEN_STREAM,
                {"error": str(error), "error_type": type(error).__name__},
                subtype=GENERATION_ERROR,
                generation_id=get_generation_id(),
            )
        except Exception:  # pylint: disable=broad-exception-caught
            logger.exception("Failed to emit generation_error; original inference error preserved")

    @staticmethod
    def _emit_generation_complete() -> None:
        """
        Emit a ``generation_complete`` TOKEN_STREAM event for a successful inference.

        Called by :meth:`infer` once the response has been fully parsed and
        (for streaming) fully collected. Clears the snapshot key so a subsequent
        retry or generation starts clean.

        Best-effort. A snapshot-backend failure at the terminal emit must not
        turn a successful LLM call into a caller-visible error; observability
        degrades for this attempt but the pipeline's LIFECYCLE_COMPLETE at
        component boundary still sweeps the snapshot, and TTL is the final
        backstop.

        No-op when no streaming session is active.
        """
        comp_ctx = get_component_context()
        if comp_ctx is None:
            return
        try:
            comp_ctx.emit(EventType.TOKEN_STREAM, None, subtype=GENERATION_COMPLETE, generation_id=get_generation_id())
        except Exception:  # pylint: disable=broad-exception-caught
            logger.exception("Failed to emit generation_complete; inference result preserved")

    @staticmethod
    async def _emit_generation_complete_async() -> None:
        """Async counterpart of :meth:`_emit_generation_complete`."""
        comp_ctx = get_component_context()
        if comp_ctx is None:
            return
        try:
            await comp_ctx.emit_async(
                EventType.TOKEN_STREAM, None, subtype=GENERATION_COMPLETE, generation_id=get_generation_id()
            )
        except Exception:  # pylint: disable=broad-exception-caught
            logger.exception("Failed to emit generation_complete; inference result preserved")


class _HttpxChatModelService(_HttpxClientMixin, _HttpModelService, _ChatModelService, metaclass=abc.ABCMeta):
    """
    An abstract base class of a generic service for managing and interacting with chat models using HTTPX.

    :param name: The name of the model service.
    :param api_key: The API key for the model service as a Haystack ``Secret``.
    :param base_url: The base URL of the model service.
    :param organization: The organization for the model service.
    :param routes: Per-instance operation-route overrides.
    :param client_kwargs: Additional keyword arguments to pass to the HTTPX client.
    """

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        api_key: SecretValue = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        routes: Optional[Mapping[str, str]] = None,
        client_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            name=name,
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            routes=routes,
        )
        if client_kwargs:
            self.configs["client_kwargs"] = client_kwargs

    @property
    def client_kwargs(self) -> Dict[str, Any]:
        """
        Get the client keyword arguments for the HTTPX client.

        :return: The client keyword arguments.
        """
        return self.configs.get("client_kwargs", {})

    @staticmethod
    def _is_retryable_error(exception: BaseException) -> bool:
        """
        Determine if an exception should trigger a retry.

        Retryable errors include:
        - Connection errors (ConnectError, ConnectTimeout)
        - Request timeout errors
        - HTTP 429 (Too Many Requests)
        - HTTP 5xx errors: 500, 502, 503, 504

        :param exception: The exception to check.
        :return: True if the error is retryable, False otherwise.
        """
        # Retry on connection errors and timeouts
        if isinstance(exception, (httpx.ConnectError, httpx.ConnectTimeout, httpx.TimeoutException)):
            return True

        # Retry on specific HTTP status codes (429 and specific 5xx)
        if isinstance(exception, httpx.HTTPStatusError):
            status_code = exception.response.status_code
            if status_code == 429:
                return True
            if status_code in {500, 502, 503, 504}:
                return True

        return False

    def _send_request_once(self, client: httpx.Client, request: httpx.Request, stream: bool) -> httpx.Response:
        """Send an HTTP request once without setting up Tenacity retry state."""
        response = client.send(request, stream=stream)
        response.raise_for_status()
        return response

    async def _asend_request_once(
        self, client: httpx.AsyncClient, request: httpx.Request, stream: bool
    ) -> httpx.Response:
        """Asynchronously send an HTTP request once without setting up Tenacity retry state."""
        response = await client.send(request, stream=stream)
        response.raise_for_status()
        return response

    def _send_request_with_retry(
        self, client: httpx.Client, request: httpx.Request, stream: bool, max_retries: int
    ) -> httpx.Response:
        """
        Send an HTTP request with retry logic for retryable errors.

        :param client: The httpx client to use.
        :param request: The request to send.
        :param stream: Whether to stream the response.
        :param max_retries: Maximum number of retry attempts. Ignored inside a Restate context, where local Tenacity
            retries are skipped and Restate owns retry orchestration.
        :return: The HTTP response.
        :raises: The original exception if it's not retryable or max retries exceeded.
        """
        if max_retries <= 1 or has_restate_context():
            return self._send_request_once(client, request, stream)

        retryer = Retrying(
            retry=retry_if_exception(self._is_retryable_error),
            wait=wait_exponential(multiplier=5, min=0, max=60),
            stop=stop_after_attempt(max_retries),
            reraise=True,  # Re-raise original exception when retries exhausted
        )

        for attempt in retryer:
            with attempt:
                try:
                    response = client.send(request, stream=stream)
                    response.raise_for_status()
                    return response
                except Exception as e:
                    # Log retry attempt before raising (tenacity will catch retryable errors)
                    if self._is_retryable_error(e) and attempt.retry_state.attempt_number < max_retries:
                        logger.warning(
                            "Request to %s failed with %s: %s. Retrying (attempt %d/%d)...",
                            self.name,
                            type(e).__name__,
                            str(e),
                            attempt.retry_state.attempt_number,
                            max_retries,
                        )
                    raise

        # This should never be reached, but satisfies type checker
        raise RuntimeError("Retry loop exited unexpectedly without returning or raising")  # pragma: no cover

    async def _asend_request_with_retry(
        self, client: httpx.AsyncClient, request: httpx.Request, stream: bool, max_retries: int
    ) -> httpx.Response:
        """
        Asynchronously send an HTTP request with retry logic for retryable errors.

        :param client: The async httpx client to use.
        :param request: The request to send.
        :param stream: Whether to stream the response.
        :param max_retries: Maximum number of retry attempts. Ignored inside a Restate context, where local Tenacity
            retries are skipped and Restate owns retry orchestration.
        :return: The HTTP response.
        :raises: The original exception if it's not retryable or max retries exceeded.
        """
        if max_retries <= 1 or has_restate_context():
            return await self._asend_request_once(client, request, stream)

        retryer = AsyncRetrying(
            retry=retry_if_exception(self._is_retryable_error),
            wait=wait_exponential(multiplier=5, min=0, max=60),
            stop=stop_after_attempt(max_retries),
            reraise=True,  # Re-raise original exception when retries exhausted
        )

        async for attempt in retryer:
            with attempt:
                try:
                    response = await client.send(request, stream=stream)
                    response.raise_for_status()
                    return response
                except Exception as e:
                    # Log retry attempt before raising (tenacity will catch retryable errors)
                    if self._is_retryable_error(e) and attempt.retry_state.attempt_number < max_retries:
                        logger.warning(
                            "Request to %s failed with %s: %s. Retrying (attempt %d/%d)...",
                            self.name,
                            type(e).__name__,
                            str(e),
                            attempt.retry_state.attempt_number,
                            max_retries,
                        )
                    raise

        # This should never be reached, but satisfies type checker
        raise RuntimeError("Retry loop exited unexpectedly without returning or raising")  # pragma: no cover

    @abc.abstractmethod
    def build_request(
        self,
        client: httpx.Client | httpx.AsyncClient,
        messages: List[ChatMessage],
        *,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        stream: bool = False,
        tools: Optional[Union[List[Tool], Toolset]] = None,
        inference_kwargs: Optional[Dict[str, Any]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> httpx.Request:
        """
        Build a request according to the designated chat completion API format.

        :param client: The httpx client to use for the request.
        :param messages: The list of ChatMessage instances to send to the model.
        :param model: The model to use for the request.
        :param timeout: The timeout for the request.
        :param stream: Whether to stream the response.
        :param tools: A list of tools to include in the request.
        :param inference_kwargs: Additional generation keyword arguments to include in the request.
        :param extra_headers: Additional headers to include in the request.
        :return: The httpx request object.
        """
        raise NotImplementedError("build_request method must be implemented")

    @abc.abstractmethod
    def parse_response(
        self,
        response: httpx.Response,
        *,
        stream: bool = False,
    ) -> List[ChatMessage]:
        """
        Parse the response from the model service to a list of ChatMessage instances.

        :param response: The response from the model service.
        :param stream: Whether the response is streamed.
        :return: The list of ChatMessage instances containing the generated responses.
        """
        raise NotImplementedError("parse_response method must be implemented")

    async def aparse_response(
        self,
        response: httpx.Response,
        *,
        stream: bool = False,
    ) -> List[ChatMessage]:
        """
        Asynchronously parse the response from the model service to a list of ChatMessage instances.

        Default implementation runs the synchronous parse_response in a thread.
        Subclasses can override this with a truly asynchronous implementation.

        .. note::
            Subclasses that emit streaming events inside ``parse_response`` (via
            ``comp_ctx.emit()``) **must** override this method with a true async
            implementation that uses ``comp_ctx.emit_async()`` instead.  This
            default executor path calls the sync ``parse_response`` from a thread
            pool thread: although ``contextvars`` values are correctly copied into
            the thread, calling the sync ``emit()`` path from a thread requires a
            ``SyncCapableBackend``, which async pipelines do not provide.  The four
            built-in model services (OpenAI, Anthropic, Google, Bedrock) all
            override this method and are unaffected.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.parse_response(
                response,
                stream=stream,
            ),
        )

    # pylint: disable=arguments-differ
    def infer(
        self,
        messages: List[ChatMessage],
        *,
        model: Optional[str],
        timeout: Optional[float],
        max_retries: int = 3,
        stream: bool = False,
        tools: Optional[Union[List[Tool], Toolset]] = None,
        inference_kwargs: Optional[Dict[str, Any]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        generation_id: Optional[str] = None,
        **kwargs,
    ) -> List[ChatMessage]:
        """
        Infer responses to the given messages using the model service.

        :param messages: The list of ChatMessage instances to send to the model.
        :param model: The model to use for the request.
        :param timeout: The timeout for the request.
        :param max_retries: Maximum number of retry attempts for retryable errors. Defaults to 3.
        :param stream: Whether to stream the response.
        :param tools: A list of tools to include in the request.
        :param inference_kwargs: Additional generation keyword arguments to include in the request.
        :param extra_headers: Additional headers to include in the request.
        :param generation_id: Optional caller-minted token identifying this durable
            generation step. Stamped on every streaming event so a reconnecting client
            can track the retry count of an in-flight generation. Stable across a
            step's retries, distinct per logical step. See the streaming guide.
        :param kwargs: Additional keyword arguments to include in the request.
        :return: The list of ChatMessage instances containing the generated responses.
        """
        token = set_generation_id(generation_id)
        try:
            # Lifecycle scope covering request send, response parse, and
            # (for streaming) chunk collection. Any failure — HTTP error,
            # malformed JSON, mid-stream chunk error, post-collect validation —
            # routes through the single except and emits exactly one
            # generation_error. Emit helpers themselves are best-effort: they
            # log rather than raise, so a snapshot-backend outage never masks
            # the original inference error or turns a successful inference
            # into a caller-visible failure.
            self._emit_generation_start()
            try:
                request = self.build_request(
                    self._client,
                    messages,
                    model=model,
                    timeout=timeout,
                    stream=stream,
                    tools=tools,
                    inference_kwargs=inference_kwargs,
                    extra_headers=extra_headers,
                )
                try:
                    response = self._send_request_with_retry(self._client, request, stream, max_retries)
                except httpx.HTTPStatusError as e:
                    if stream:
                        response_text = e.response.read().decode("utf-8")
                    else:
                        response_text = e.response.text
                    raise httpx.HTTPStatusError(
                        str(e) + f"\nModel Service: {self.name}\nDetails: {response_text}",
                        request=e.request,
                        response=e.response,
                    ) from e
                chat_messages = self.parse_response(response, stream=stream)
            except Exception as e:
                self._emit_generation_error(e)
                raise
            self._emit_generation_complete()
            return chat_messages
        finally:
            reset_generation_id(token)

    async def ainfer(
        self,
        messages: List[ChatMessage],
        *,
        model: Optional[str],
        timeout: Optional[float],
        max_retries: int = 3,
        stream: bool = False,
        tools: Optional[Union[List[Tool], Toolset]] = None,
        inference_kwargs: Optional[Dict[str, Any]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        generation_id: Optional[str] = None,
        **kwargs,
    ) -> List[ChatMessage]:
        """
        Asynchronously infer responses to the given messages using the model service.

        :param messages: The list of ChatMessage instances to send to the model.
        :param timeout: The timeout for the request.
        :param max_retries: Maximum number of retry attempts for retryable errors. Defaults to 3.
        :param stream: Whether to stream the response.
        :param model: The model to use for the request.
        :param inference_kwargs: Additional generation keyword arguments to include in the request.
        :param tools: A list of tools to include in the request.
        :param extra_headers: Additional headers to include in the request.
        :param generation_id: Optional caller-minted token identifying this durable
            generation step. Stamped on every streaming event so a reconnecting client
            can track the retry count of an in-flight generation. Stable across a
            step's retries, distinct per logical step. See the streaming guide.
        :param kwargs: Additional keyword arguments to include in the request.
        :return: The list of ChatMessage instances containing the generated responses.
        """
        token = set_generation_id(generation_id)
        try:
            # See :meth:`infer` for the lifecycle scope rationale.
            await self._emit_generation_start_async()
            try:
                request = self.build_request(
                    self._async_client,
                    messages,
                    model=model,
                    timeout=timeout,
                    stream=stream,
                    tools=tools,
                    inference_kwargs=inference_kwargs,
                    extra_headers=extra_headers,
                )
                try:
                    response = await self._asend_request_with_retry(self._async_client, request, stream, max_retries)
                except httpx.HTTPStatusError as e:
                    if stream:
                        response_bytes = await e.response.aread()
                        response_text = response_bytes.decode("utf-8")
                    else:
                        response_text = e.response.text
                    raise httpx.HTTPStatusError(
                        str(e) + f"\nModel Service: {self.name}\nDetails: {response_text}",
                        request=e.request,
                        response=e.response,
                    ) from e
                chat_messages = await self.aparse_response(response, stream=stream)
            except Exception as e:
                await self._emit_generation_error_async(e)
                raise
            await self._emit_generation_complete_async()
            return chat_messages
        finally:
            reset_generation_id(token)
