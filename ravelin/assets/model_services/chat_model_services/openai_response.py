"""
This module defines the OpenAIResponseModelService class, which is a ModelService asset that refers to a deployed OpenAI
model, or an Azure OpenAI deployment, exposed through the Responses API.

The Responses API differs from the Chat Completions API in a few notable ways: it uses ``input`` (typed items) instead
of ``messages``, a top-level ``instructions`` field for the system prompt, a flat ``tools`` schema (no nested
``function`` wrapper), and a typed SSE event stream (``response.output_text.delta``,
``response.function_call_arguments.delta``, ``response.completed``, ...) rather than choice-index deltas. A single class
covers both OpenAI and Azure OpenAI: the base URL and route are user-configurable, and both ``Authorization: Bearer``
and ``api-key`` headers are set so either provider authenticates without a subclass.
"""

import json
import logging
from base64 import b64encode
from types import MappingProxyType
from typing import Any, AsyncIterable, Dict, Iterable, List, Literal, Optional, Union, cast

import httpx
from haystack.dataclasses import ToolCallDelta
from haystack.tools import Tool, Toolset

from ravelin.assets._base import resolve_secret
from ravelin.assets.model_services._base import register_model_service
from ravelin.assets.model_services.chat_model_services._base import _HttpxChatModelService
from ravelin.dataclasses import ChatMessage, ChatRole, StreamingChunk
from ravelin.dataclasses.chat_message import MediaContent, TextContent, ToolCall, ToolCallResult
from ravelin.utils.streaming import AsyncServerSentEventStream, ServerSentEvent, ServerSentEventStream

logger = logging.getLogger(__name__)


def _convert_message_to_response_items(message: ChatMessage) -> List[Dict[str, Any]]:
    """
    Convert a single ChatMessage into one or more Responses API ``input`` items.

    Assistant messages that carry both text and tool calls yield multiple items — a ``message`` item for the text and
    one ``function_call`` item per tool call — because the Responses API keeps them separate at the top level.

    :param message: The ChatMessage to convert.
    :returns: A list of dicts, each suitable to be appended to the ``input`` array of a Responses API request.
    """
    if message.role == ChatRole.SYSTEM:
        raise ValueError("System messages are handled via the top-level `instructions` field, not `input`.")

    if message.role == ChatRole.TOOL:
        if not message.tool_call_result:
            raise ValueError("Tool messages must carry a `ToolCallResult`.")
        if len(message) > 1:
            raise ValueError("Tool call results should not be included with other content.")
        if message.tool_call_result.origin.id is None:
            raise ValueError("`ToolCall` must have a non-null `id` attribute to be used with OpenAI Responses.")
        return [
            {
                "type": "function_call_output",
                "call_id": message.tool_call_result.origin.id,
                "output": message.tool_call_result.result,
            }
        ]

    role = message.role.value  # "user" or "assistant"
    text_part_type = "output_text" if message.role == ChatRole.ASSISTANT else "input_text"
    content_parts: List[Dict[str, Any]] = []
    tool_call_items: List[Dict[str, Any]] = []

    for item in message.content:
        if isinstance(item, TextContent):
            content_parts.append({"type": text_part_type, "text": item.text})
        elif isinstance(item, MediaContent):
            if item.media.type != "image":
                raise ValueError(f"Unsupported media type '{item.media.mime_type}' for OpenAI Responses.")
            if message.role != ChatRole.USER:
                raise ValueError("Image content is only supported on user messages for OpenAI Responses.")
            base64_data = b64encode(item.media.data).decode("utf-8")
            content_parts.append(
                {
                    "type": "input_image",
                    "image_url": f"data:{item.media.mime_type};base64,{base64_data}",
                    "detail": item.media.meta.get("detail", "auto"),
                }
            )
        elif isinstance(item, ToolCall):
            if message.role != ChatRole.ASSISTANT:
                raise ValueError("Tool calls should only be included for assistant messages.")
            if item.id is None:
                raise ValueError("`ToolCall` must have a non-null `id` attribute to be used with OpenAI Responses.")
            tool_call_items.append(
                {
                    "type": "function_call",
                    "call_id": item.id,
                    "name": item.tool_name,
                    "arguments": json.dumps(item.arguments, ensure_ascii=False),
                }
            )
        elif isinstance(item, ToolCallResult):
            raise ValueError("`ToolCallResult` must be sent on a tool-role message.")
        else:
            raise ValueError(f"Unsupported content type '{type(item).__name__}' for OpenAI Responses.")

    items: List[Dict[str, Any]] = []

    # Reattach reasoning blocks captured from a prior response so multi-turn reasoning can chain statelessly.
    if message.role == ChatRole.ASSISTANT:
        for block in message.meta.get("reasoning_blocks", []) or []:
            items.append(block)

    if content_parts:
        items.append({"role": role, "content": content_parts})

    items.extend(tool_call_items)
    return items


@register_model_service("openai-response")
class OpenAIResponseModelService(_HttpxChatModelService):
    """
    A ModelService asset that refers to a deployed OpenAI or Azure OpenAI model exposed through the Responses API.

    Azure vs OpenAI is an endpoint-routing choice. OpenAI uses the default ``/v1/responses`` route; Azure deployments
    can override it with their deployment path and ``api-version`` query parameter. Both ``Authorization`` and
    ``api-key`` headers are set when an API key is provided.
    """

    DEFAULT_ROUTES = MappingProxyType({"inference": "/v1/responses"})

    @staticmethod
    def _extract_instructions(messages: List[ChatMessage]) -> Optional[str]:
        """
        Return the text of a leading system ChatMessage for use as the top-level ``instructions`` field.

        :param messages: The full list of ChatMessages. Not mutated — callers are responsible for skipping the
            system message when building ``input`` items.
        :returns: The system prompt text, or ``None`` if the first message is not a system message.
        """
        if not messages or messages[0].role != ChatRole.SYSTEM:
            return None
        first = messages[0]
        if len(first) != 1 or not isinstance(first.content[0], TextContent):
            raise ValueError("The system prompt must be a single text message.")
        return first.content[0].text

    @staticmethod
    def _build_message(response_json: Dict[str, Any]) -> ChatMessage:
        """
        Convert a non-streaming Responses API JSON payload into a ChatMessage.

        :param response_json: The parsed JSON body of a Responses API response.
        :returns: The assembled assistant ChatMessage.
        """
        text = ""
        reasoning_content: Optional[str] = None
        reasoning_blocks: List[Dict[str, Any]] = []
        tool_calls: List[ToolCall] = []

        for item in response_json.get("output", []):
            item_type = item.get("type")
            if item_type == "message":
                for part in item.get("content", []):
                    if part.get("type") == "output_text":
                        text += part.get("text", "")
            elif item_type == "reasoning":
                reasoning_blocks.append(item)
                summary_text = "".join(
                    part.get("text", "") for part in item.get("summary", []) if part.get("type") == "summary_text"
                )
                if not summary_text:
                    summary_text = "".join(
                        part.get("text", "") for part in item.get("content", []) if part.get("type") == "reasoning_text"
                    )
                if summary_text:
                    reasoning_content = (reasoning_content or "") + summary_text
            elif item_type == "function_call":
                tool_call = OpenAIResponseModelService._build_tool_call(item)
                if tool_call is not None:
                    tool_calls.append(tool_call)

        meta: Dict[str, Any] = {
            "id": response_json.get("id"),
            "model": response_json.get("model"),
            "status": response_json.get("status"),
            "usage": dict(response_json.get("usage") or {}),
        }
        if response_json.get("incomplete_details"):
            meta["incomplete_details"] = response_json["incomplete_details"]
        finish_reason = OpenAIResponseModelService._derive_finish_reason(response_json, tool_calls)
        if finish_reason:
            meta["finish_reason"] = finish_reason
        if reasoning_blocks:
            meta["reasoning_blocks"] = reasoning_blocks

        return ChatMessage.from_assistant(
            text, reasoning_content=reasoning_content, tool_calls=tool_calls or None, meta=meta
        )

    @staticmethod
    def _build_tool_call(item: Dict[str, Any]) -> Optional[ToolCall]:
        """
        Build a ToolCall from a Responses API ``function_call`` output item.

        :param item: The raw output item from the Responses API.
        :returns: The ToolCall, or ``None`` if the arguments are not valid JSON.
        """
        try:
            call_id = item["call_id"]
            tool_name = item["name"]
            arguments_str = item.get("arguments", "") or "{}"
        except KeyError as err:
            raise ValueError(f"Invalid tool call format: {item}") from err
        try:
            arguments = json.loads(arguments_str)
        except json.JSONDecodeError:
            logger.warning(
                "OpenAI Responses returned malformed JSON for tool call arguments. This tool call will be skipped. "
                "Tool call ID: %s, Tool name: %s, Arguments: %s",
                call_id,
                tool_name,
                arguments_str,
            )
            return None
        return ToolCall(id=call_id, tool_name=tool_name, arguments=arguments)

    @staticmethod
    def _derive_finish_reason(response_json: Dict[str, Any], tool_calls: List[ToolCall]) -> Optional[str]:
        """
        Derive a Chat-Completions-style ``finish_reason`` from a Responses API payload.

        The Responses API uses ``status`` (``completed``/``incomplete``/``failed``) and
        ``incomplete_details.reason``, but downstream callers still key off ``finish_reason``. We map into the same
        vocabulary used by ``OpenAIChatModelService._check_finish_reason``.
        """
        status = response_json.get("status")
        if status == "incomplete":
            reason = (response_json.get("incomplete_details") or {}).get("reason")
            if reason == "max_output_tokens":
                return "length"
            if reason == "content_filter":
                return "content_filter"
            return "incomplete"
        if status == "completed":
            return "tool_calls" if tool_calls else "stop"
        return status

    # pylint: disable=too-many-return-statements,too-many-branches
    @staticmethod
    def _build_chunk(sse: ServerSentEvent) -> Optional[StreamingChunk]:
        """
        Build a StreamingChunk from a Responses API SSE event.

        See https://platform.openai.com/docs/api-reference/responses-streaming for the typed event catalog.

        :param sse: The raw ServerSentEvent.
        :returns: A StreamingChunk, or ``None`` when the event carries no incremental content of interest.
        """
        data = sse.json()
        event_type = sse.event or data.get("type")

        if event_type == "response.created":
            response = data.get("response", {})
            return StreamingChunk(
                content="",
                meta={
                    "id": response.get("id"),
                    "model": response.get("model"),
                },
                start=True,
            )
        if event_type == "response.output_text.delta":
            return StreamingChunk(
                content=data.get("delta", ""),
                meta={"item_id": data.get("item_id")},
                index=data.get("output_index"),
            )
        if event_type in ("response.reasoning_summary_text.delta", "response.reasoning_text.delta"):
            return StreamingChunk(
                content="",
                reasoning_content=data.get("delta", ""),
                meta={"item_id": data.get("item_id")},
                index=data.get("output_index"),
            )
        if event_type == "response.output_item.added":
            item = data.get("item", {})
            if item.get("type") == "function_call":
                output_index = data.get("output_index", 0) or 0
                return StreamingChunk(
                    content="",
                    tool_calls=[
                        ToolCallDelta(
                            index=int(output_index),
                            id=item.get("call_id"),
                            tool_name=item.get("name") or "",
                            arguments=item.get("arguments", "") or "",
                        )
                    ],
                    meta={"item_id": item.get("id")},
                    index=output_index,
                    start=True,
                )
            if item.get("type") == "reasoning":
                output_index = data.get("output_index", 0) or 0
                # No reasoning_content on the start marker itself — the reasoning_block is what
                # carries the item's identity; actual reasoning text arrives via subsequent
                # reasoning_summary_text.delta / reasoning_text.delta events.
                return StreamingChunk(
                    content="",
                    meta={"item_id": item.get("id"), "reasoning_block": item},
                    index=output_index,
                    start=True,
                )
            return None
        if event_type == "response.function_call_arguments.delta":
            output_index = data.get("output_index", 0) or 0
            return StreamingChunk(
                content="",
                tool_calls=[ToolCallDelta(index=int(output_index), arguments=data.get("delta", ""))],
                meta={"item_id": data.get("item_id")},
                index=output_index,
            )
        if event_type == "response.completed":
            response = data.get("response", {})
            tool_calls_present = any(item.get("type") == "function_call" for item in response.get("output", []))
            finish_reason = OpenAIResponseModelService._derive_finish_reason(
                response,
                [ToolCall(id="_", tool_name="_", arguments={})] if tool_calls_present else [],
            )
            return StreamingChunk(
                content="",
                meta={
                    "id": response.get("id"),
                    "model": response.get("model"),
                    "status": response.get("status"),
                    "usage": response.get("usage"),
                },
                finish_reason=cast(
                    Optional[Literal["stop", "length", "tool_calls", "content_filter", "tool_call_results"]],
                    finish_reason,
                ),
            )
        if event_type in ("response.failed", "error"):
            error = data.get("error") or (data.get("response", {}).get("error"))
            message = (error or {}).get("message") if isinstance(error, dict) else str(error)
            raise RuntimeError(f"Error received during Responses streaming: {message or data}")
        # Every other event type — response.in_progress, response.output_item.done,
        # response.content_part.added/done, response.output_text.done, response.function_call_arguments.done,
        # response.reasoning_summary_text.done, response.reasoning_summary_part.added/done — carries no
        # incremental content beyond what the delta / added events already provided.
        return None

    @staticmethod
    def _connect_chunks(chunks: List[StreamingChunk]) -> ChatMessage:
        """
        Reassemble a list of streaming chunks into a single ChatMessage.

        :param chunks: The chunks captured during streaming, in order.
        :returns: The assembled assistant ChatMessage.
        """
        if not chunks:
            raise ValueError("Cannot reassemble an empty chunk list.")

        text = ""
        reasoning_content: Optional[str] = None
        raw_tool_calls: Dict[int, Dict[str, Any]] = {}
        reasoning_blocks: List[Dict[str, Any]] = []
        meta: Dict[str, Any] = {}
        finish_reason: Optional[str] = None
        usage: Optional[Dict[str, Any]] = None

        for chunk in chunks:
            if chunk.finish_reason:
                finish_reason = chunk.finish_reason
            chunk_usage = chunk.meta.get("usage")
            if chunk_usage:
                usage = chunk_usage
            if chunk.tool_calls:
                for tc_delta in chunk.tool_calls:
                    tc_index = tc_delta.index
                    if tc_index not in raw_tool_calls:
                        raw_tool_calls[tc_index] = {
                            "call_id": tc_delta.id,
                            "name": tc_delta.tool_name or "",
                            "arguments": tc_delta.arguments or "",
                        }
                    else:
                        if tc_delta.id and not raw_tool_calls[tc_index]["call_id"]:
                            raw_tool_calls[tc_index]["call_id"] = tc_delta.id
                        if tc_delta.tool_name and not raw_tool_calls[tc_index]["name"]:
                            raw_tool_calls[tc_index]["name"] = tc_delta.tool_name
                        if tc_delta.arguments:
                            raw_tool_calls[tc_index]["arguments"] += tc_delta.arguments
            if chunk.reasoning_content:
                reasoning_content = (reasoning_content or "") + chunk.reasoning_content
            if "reasoning_block" in chunk.meta:
                reasoning_blocks.append(chunk.meta["reasoning_block"])
            if chunk.content:
                text += chunk.content
            for key, value in chunk.meta.items():
                if key in ("usage", "reasoning_block"):
                    continue
                if value is not None:
                    meta[key] = value

        if usage is not None:
            meta["usage"] = dict(usage)
        if finish_reason:
            meta["finish_reason"] = finish_reason
        if reasoning_blocks:
            meta["reasoning_blocks"] = reasoning_blocks

        tool_calls: List[ToolCall] = []
        for raw_tc in raw_tool_calls.values():
            tc = OpenAIResponseModelService._build_tool_call(
                {"call_id": raw_tc["call_id"], "name": raw_tc["name"], "arguments": raw_tc["arguments"]}
            )
            if tc is not None:
                tool_calls.append(tc)

        return ChatMessage.from_assistant(
            text, reasoning_content=reasoning_content, tool_calls=tool_calls or None, meta=meta
        )

    @staticmethod
    def _check_finish_reason(message: ChatMessage) -> None:
        """
        Warn on notable ``finish_reason`` values, mirroring ``OpenAIChatModelService._check_finish_reason``.

        :param message: The reassembled assistant message.
        """
        finish_reason = message.meta.get("finish_reason")
        match finish_reason:
            case "length":
                logger.warning(
                    "The Responses completion %s was truncated before reaching a natural stopping point. "
                    "Increase max_output_tokens to allow for longer completions.",
                    message.meta.get("id"),
                )
            case "content_filter":
                logger.warning(
                    "The Responses completion %s was truncated due to the content filter.",
                    message.meta.get("id"),
                )
            case "tool_calls":
                if message.tool_call is None:
                    logger.warning(
                        "The Responses completion %s claimed to have tool calls, but none were found in the response.",
                        message.meta.get("id"),
                    )
            case _:
                pass

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
        Build an httpx request targeting the OpenAI Responses API.

        :param client: The httpx client used to construct the request.
        :param messages: The list of ChatMessage instances to send.
        :param model: The model or Azure deployment name.
        :param timeout: The per-request timeout.
        :param stream: Whether to stream the response.
        :param tools: Tools to expose to the model.
        :param inference_kwargs: Additional body fields (``reasoning``, ``max_output_tokens``, ``temperature``, ...).
        :param extra_headers: Additional headers to merge into the request.
        :returns: The built httpx request.
        """
        model = model or ""
        inference_kwargs = inference_kwargs or {}
        instructions = self._extract_instructions(messages)
        remaining = messages[1:] if instructions is not None else messages

        input_items: List[Dict[str, Any]] = []
        for message in remaining:
            input_items.extend(_convert_message_to_response_items(message))

        body: Dict[str, Any] = {
            "model": model,
            "input": input_items,
            "stream": stream,
            **inference_kwargs,
        }
        if instructions is not None:
            body["instructions"] = instructions

        if tools:
            body["tools"] = [
                {
                    "type": "function",
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                }
                for t in tools
            ]

        headers = {"Content-Type": "application/json"}
        api_key = resolve_secret(self.api_key)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
            headers["api-key"] = api_key  # For Azure OpenAI API compatibility
            if self.organization:
                headers["OpenAI-Organization"] = self.organization

        return client.build_request(
            "POST",
            url=self.build_request_url("inference"),
            json=body,
            headers=headers | (extra_headers or {}),
            timeout=timeout,
        )

    def _iter_streaming_chunks(self, response: httpx.Response) -> Iterable[StreamingChunk]:
        """Yield StreamingChunks from a synchronous Responses SSE stream."""
        for sse in ServerSentEventStream(response=response):
            chunk = self._build_chunk(sse)
            if chunk is not None:
                yield chunk

    async def _aiter_streaming_chunks(self, response: httpx.Response) -> AsyncIterable[StreamingChunk]:
        """Yield StreamingChunks from an asynchronous Responses SSE stream."""
        async for sse in AsyncServerSentEventStream(response=response):
            chunk = self._build_chunk(sse)
            if chunk is not None:
                yield chunk

    def parse_response(
        self,
        response: httpx.Response,
        *,
        stream: bool = False,
    ) -> List[ChatMessage]:
        """
        Parse a Responses API response into ChatMessages.

        :param response: The httpx response from the model service.
        :param stream: Whether the response is a stream.
        :returns: A single-element list containing the assembled assistant ChatMessage.
        """
        if stream:
            chunks = self._collect_streaming_chunks(self._iter_streaming_chunks(response))
            chat_messages = [self._connect_chunks(chunks)]
        else:
            try:
                response_json = response.json()
            except json.JSONDecodeError as err:
                raise ValueError(
                    f"The model service '{self.name}' returned an invalid JSON response: {response.text}"
                ) from err
            chat_messages = [self._build_message(response_json)]
        for message in chat_messages:
            self._check_finish_reason(message)
        return chat_messages

    async def aparse_response(
        self,
        response: httpx.Response,
        *,
        stream: bool = False,
    ) -> List[ChatMessage]:
        """Asynchronously parse a Responses API response into ChatMessages."""
        if stream:
            chunks = await self._collect_streaming_chunks_async(self._aiter_streaming_chunks(response))
            chat_messages = [self._connect_chunks(chunks)]
        else:
            try:
                response_json = response.json()
            except json.JSONDecodeError as err:
                raise ValueError(
                    f"The model service '{self.name}' returned an invalid JSON response: {response.text}"
                ) from err
            chat_messages = [self._build_message(response_json)]
        for message in chat_messages:
            self._check_finish_reason(message)
        return chat_messages
