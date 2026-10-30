"""
This module provides a collection of classes for handling the chat model services hosted on Amazon Bedrock. Currently,
only the Anthropic chat model service is supported.
"""

import base64
import json
import logging
from types import MappingProxyType
from typing import Any, AsyncIterable, Dict, Iterable, List, Mapping, Optional, Union

import httpx
from haystack.tools import Tool, Toolset

from ravelin.assets._base import SecretValue, resolve_secret
from ravelin.assets.model_services._base import register_model_service
from ravelin.assets.model_services.chat_model_services.anthropic import AnthropicChatModelService
from ravelin.dataclasses import ChatMessage, ChatRole, StreamingChunk
from ravelin.dataclasses.chat_message import TextContent

logger = logging.getLogger(__name__)


@register_model_service("bedrock-anthropic-chat")
class BedrockAnthropicChatModelService(AnthropicChatModelService):
    """
    A ModelService asset that refers to an Anthropic chat model hosted on AWS Bedrock.

    :param name: The name of the model service.
    :param model: The model hosted on the model service, defaults to None.
    :param api_key: The API key to use for the model service as a Haystack ``Secret``, defaults to None.
    :param base_url: The base URL of the Bedrock Runtime service.
    :param organization: The organization of the model service, defaults to None.
    :param routes: Per-instance operation-route overrides.
    """

    DEFAULT_ROUTES = MappingProxyType(
        {
            "inference": "/model/{model}/invoke",
            "stream": "/model/{model}/invoke-with-response-stream",
        }
    )

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        model: Optional[str] = None,
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
            client_kwargs=client_kwargs,
        )
        self.configs["model"] = model
        self.model = model or "claude-3-5-sonnet-latest"

    @property
    def stream_url(self) -> str:
        """Return the streaming URL for the configured default model."""
        return self.build_request_url("stream", model=self.model)

    def _decode_data(
        self,
        stream_bytes: str | bytes,
        bedrock_content_type: str = "application/json",
        request_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Decode the data from the stream of bytes.

        :param stream_bytes: The stream of bytes.
        :param bedrock_content_type: The actual content type of the data.
        :param request_url: The effective streaming request URL, used in errors.
        :returns: The decoded data.
        :raises NotImplementedError: If the content type is not supported.
        :raises RuntimeError: If the response format is invalid.
        """
        if bedrock_content_type != "application/json":
            raise NotImplementedError(f"Unsupported content type: {bedrock_content_type}")
        content = json.loads(stream_bytes)
        if encoded_data := content.get("chunk", {}).get("bytes"):
            return json.loads(base64.b64decode(encoded_data, validate=True))

        # Error handling if encoded_data is not present
        error_url = request_url or self.stream_url
        if msg := content.get("internalServerException", {}).get("message"):
            raise RuntimeError(f"Internal server exception from {error_url}: {msg}")
        if model_err_info := content.get("modelStreamErrorException"):
            raise RuntimeError(f"Model stream error from {error_url}: {model_err_info}")
        if msg := content.get("validationException", {}).get("message"):
            raise RuntimeError(f"Validation exception from {error_url}: {msg}")
        if msg := content.get("throttlingException", {}).get("message"):
            raise RuntimeError(f"Throttling exception from {error_url}: {msg}")
        if msg := content.get("modelTimeoutException", {}).get("message"):
            raise RuntimeError(f"Model timeout exception from {error_url}: {msg}")
        if msg := content.get("serviceUnavailableException", {}).get("message"):
            raise RuntimeError(f"Service unavailable exception from {error_url}: {msg}")
        raise RuntimeError(f"Invalid response format from {error_url}: {content}")

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
        Build a request according to the Bedrock-Anthropic Chat Completion API format.

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
        inference_kwargs = inference_kwargs or {}
        model_name = model or self.model

        # Extract the system prompt if it exists
        if messages[0].role == ChatRole.SYSTEM:
            if len(messages[0]) != 1 or not isinstance(messages[0].content[0], TextContent):
                raise ValueError("The system prompt must be a single text message.")
            system_field = self._build_system_field(messages[0])
            messages = messages[1:]
        else:
            system_field = None

        # Format the rest of the messages
        anthropic_formatted_messages = [self.format_chat_message(message) for message in messages]

        max_tokens = inference_kwargs.get("max_tokens")
        if max_tokens is None:
            max_tokens = self.default_max_tokens(model_name)
            logger.info(
                "max_tokens is required for AWS Bedrock Anthropic Chat API. "
                "Using default value inferred from model: %s",
                max_tokens,
            )

        # Create the body of the request
        body = {
            "messages": anthropic_formatted_messages,
            "max_tokens": max_tokens,
            "anthropic_version": "bedrock-2023-05-31",
            **inference_kwargs,
        }

        if system_field:
            body["system"] = system_field

        # Create the headers for the request
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        api_key = resolve_secret(self.api_key)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        # Include tool definitions if provided
        if tools:
            body["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools
            ]

        # Build the request
        request = client.build_request(
            "POST",
            url=self.build_request_url("stream" if stream else "inference", model=model_name),
            json=body,
            headers=headers | (extra_headers or {}),
            timeout=timeout,
        )
        return request

    def _iter_streaming_chunks(self, response: httpx.Response) -> Iterable[StreamingChunk]:
        """Yield StreamingChunks from a synchronous Bedrock response stream."""
        content_type = response.headers.get("x-amzn-bedrock-content-type", "application/json")
        request_url = str(response.request.url) if isinstance(response, httpx.Response) else self.stream_url
        for event_stream in response.iter_lines():
            data = self._decode_data(event_stream, bedrock_content_type=content_type, request_url=request_url)
            chunk = self._build_chunk_from_data(data)
            if chunk is not None:
                yield chunk

    async def _aiter_streaming_chunks(self, response: httpx.Response) -> AsyncIterable[StreamingChunk]:
        """Yield StreamingChunks from an asynchronous Bedrock response stream."""
        content_type = response.headers.get("x-amzn-bedrock-content-type", "application/json")
        request_url = str(response.request.url) if isinstance(response, httpx.Response) else self.stream_url
        async for event_stream in response.aiter_lines():
            data = self._decode_data(event_stream, bedrock_content_type=content_type, request_url=request_url)
            chunk = self._build_chunk_from_data(data)
            if chunk is not None:
                yield chunk

    def parse_response(
        self,
        response: httpx.Response,
        *,
        stream: bool = False,
    ) -> List[ChatMessage]:
        """
        Parse the response from the Anthropic Chat Completion API to a list of ChatMessage instances. If streaming is
        enabled, the response is parsed as a stream of ServerSentEvents, and each event is captured as a StreamingChunk.

        :param response: The response from the model service.
        :param stream: Whether the response is streamed.
        :return: The list of ChatMessage instances containing the generated responses.
        """
        if stream:
            chunks = self._collect_streaming_chunks(self._iter_streaming_chunks(response))
            chat_messages = [self._connect_chunks(chunks)]
        else:
            response_json = response.json()
            chat_messages = [
                self._build_message(
                    content=response_json.get("content", []),
                    model=response_json.get("model"),
                    msg_id=response_json.get("id"),
                    stop_reason=response_json.get("stop_reason"),
                    stop_sequence=response_json.get("stop_sequence"),
                    usage=response_json.get("usage"),
                )
            ]
        for message in chat_messages:
            self._check_stop_reason(message)
        return chat_messages

    async def aparse_response(
        self,
        response: httpx.Response,
        *,
        stream: bool = False,
    ) -> List[ChatMessage]:
        """
        Asynchronously parse the response from the Bedrock Anthropic Chat Completion API to a list of ChatMessage
        instances. If streaming is enabled, the response is parsed as a stream of ServerSentEvents, and each event is
        captured as a StreamingChunk.

        :param response: The response from the model service.
        :param stream: Whether the response is streamed.
        :return: The list of ChatMessage instances containing the generated responses.
        """
        if stream:
            chunks = await self._collect_streaming_chunks_async(self._aiter_streaming_chunks(response))
            chat_messages = [self._connect_chunks(chunks)]
        else:
            response_json = response.json()
            chat_messages = [
                self._build_message(
                    content=response_json.get("content", []),
                    model=response_json.get("model"),
                    msg_id=response_json.get("id"),
                    stop_reason=response_json.get("stop_reason"),
                    stop_sequence=response_json.get("stop_sequence"),
                    usage=response_json.get("usage"),
                )
            ]
        for message in chat_messages:
            self._check_stop_reason(message)
        return chat_messages
