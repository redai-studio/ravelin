"""
This module contains ModelServices provided by Google, or other services that are compatible with Google's API schemas.
Currently, this includes the GeminiChatModelService class, which is responsible for interacting with the Gemini model or
any model service that follows the same API structure.
"""

import base64
import json
import logging
import uuid
from collections import defaultdict
from types import MappingProxyType
from typing import Any, AsyncIterable, Dict, Iterable, List, Optional, Generator, Union

import httpx
from haystack.tools import Tool, Toolset
from haystack.dataclasses import ToolCallDelta

from ravelin.assets._base import resolve_secret
from ravelin.assets.model_services._base import register_model_service
from ravelin.assets.model_services.chat_model_services._base import _HttpxChatModelService
from ravelin.assets.model_services.chat_model_services.anthropic import AnthropicChatModelService
from ravelin.dataclasses import ByteStream, ChatMessage, ChatRole, StreamingChunk
from ravelin.dataclasses.chat_message import (
    TextContent,
    MediaContent,
    ToolCall,
    ToolCallResult,
    ChatMessageContent,
)
from ravelin.utils.streaming import ServerSentEventStream, AsyncServerSentEventStream, ServerSentEvent

logger = logging.getLogger(__name__)


_GEMINI_SUPPORTED_MEDIA_TYPES = {
    "text": {"plain"},
    "image": {"jpeg", "png", "webp"},
    "video": {"x-flv", "quicktime", "mpeg", "mpgs", "mpg", "mp4", "webm", "wmv", "3gpp"},
    "audio": {"x-aac", "flac", "mp3", "m4a", "mpeg", "mpga", "mp4", "opus", "pcm", "wav", "webm"},
    "application": {"pdf"},
}

_GEMINI_PARAM_MAPPING = {
    "temperature": "temperature",
    "max_tokens": "maxOutputTokens",
    "n": "candidateCount",
    "top_p": "topP",
    "top_k": "topK",
    "stop": "stopSequences",
    "presence_penalty": "presencePenalty",
    "frequency_penalty": "frequencyPenalty",
}


@register_model_service("gemini-chat")
class GeminiChatModelService(_HttpxChatModelService):
    """
    A service for interacting with Google's Gemini API.
    """

    DEFAULT_ROUTES = MappingProxyType(
        {
            "inference": "/v1beta/models/{model}:generateContent",
            "stream": "/v1beta/models/{model}:streamGenerateContent",
        }
    )

    @staticmethod
    def format_chat_message(message: ChatMessage) -> Optional[Dict[str, Any]]:
        """
        Convert a ChatMessage to the Gemini API format.

        :param message: The ChatMessage to convert.
        :return: The Gemini API format of the ChatMessage or None for system messages.
        """

        gemini_msg: Dict[str, Any] = {}

        # System messages are handled separately in build_request
        if message.role not in (ChatRole.USER, ChatRole.ASSISTANT):
            if message.role == ChatRole.TOOL and message.tool_call_result is not None:
                logger.debug("Skipping role for tool call results as per Gemini guidelines.")
            else:
                logger.warning(
                    "The role %s is not supported by the Gemini API. Falling back to user role.", message.role
                )
                gemini_msg["role"] = ChatRole.USER.value
        else:
            gemini_msg["role"] = message.role.value if message.role != ChatRole.ASSISTANT else "model"

        # Process content
        parts: List[Dict[str, Any]] = []
        _signature_unattached = "thoughtSignature" in message.meta

        for item in message.content:
            if isinstance(item, TextContent):
                if item.text:  # Only include non-empty text
                    parts.append({"text": item.text})
            elif isinstance(item, MediaContent):
                # Check if the media type is supported
                if not (
                    item.media.type in _GEMINI_SUPPORTED_MEDIA_TYPES
                    and item.media.subtype in _GEMINI_SUPPORTED_MEDIA_TYPES[item.media.type]
                ):
                    raise ValueError(f"Unsupported media type '{item.media.mime_type}' for Gemini completions.")

                media_part = {
                    "inlineData": {
                        "mimeType": item.media.mime_type,
                        "data": base64.b64encode(item.media.data).decode("utf-8"),
                    }
                }

                # Add video metadata information if available
                if item.media.type == "video":
                    video_metadata = {}
                    if "startOffset" in item.media.meta:
                        video_metadata["startOffset"] = item.media.meta["startOffset"]
                    if "endOffset" in item.media.meta:
                        video_metadata["endOffset"] = item.media.meta["endOffset"]
                    if video_metadata:
                        media_part["videoMetadata"] = video_metadata

                parts.append(media_part)

            elif isinstance(item, ToolCall):
                tool_call_part = {"functionCall": {"name": item.tool_name, "args": item.arguments}}
                if _signature_unattached:
                    # Attach the thought signature to the first functionCall part
                    # See https://ai.google.dev/gemini-api/docs/thought-signatures#function-calling for more information
                    tool_call_part["thoughtSignature"] = message.meta["thoughtSignature"]
                    _signature_unattached = False
                parts.append(tool_call_part)
            elif isinstance(item, ToolCallResult):
                if item.error:
                    tool_response = {"error": item.result, "output": "N/A"}
                else:
                    tool_response = {"output": item.result}
                parts.append({"functionResponse": {"name": item.origin.tool_name, "response": tool_response}})
            else:
                raise ValueError(f"Unsupported content type '{type(item).__name__}' for Gemini completions.")

        if _signature_unattached:
            # If signature is not attached to functionCalls, it should be attached to the final content part
            parts[-1]["thoughtSignature"] = message.meta["thoughtSignature"]

        gemini_msg["parts"] = parts
        return gemini_msg

    def _build_chunks(self, sse: ServerSentEvent) -> Generator[StreamingChunk, None, None]:
        """
        Build StreamingChunks from the Gemini API response. Multiple chunks can be yielded from a single SSE event if
        the response contains multiple candidates.

        :param sse: The Server-Sent Event data.
        :return: A generator yielding StreamingChunk instances.
        """
        data = sse.json()
        response_meta = {"model": data.get("modelVersion")}
        if "responseId" in data:
            response_meta["msg_id"] = data["responseId"]
        if "usageMetadata" in data:
            # We preserve the original usage format from the API response for transparency.
            # This might need to be converted later for specific tracing platforms.
            response_meta["usage"] = data["usageMetadata"]

        for candidate in data.get("candidates", []):
            content = ""
            reasoning_content = None
            tool_call_deltas = []
            candidate_meta = self._extract_candidate_meta(candidate)
            meta = {**response_meta, **{k: v for k, v in candidate_meta.items() if k != "finish_reason"}}
            chunk_index = candidate.get("index", 0)
            finish_reason = candidate_meta.get("finish_reason")
            # is_start flag is currently not tracked, since sse can't see previous contents
            is_start = True

            for extracted_content in self._extract_candidate_contents(candidate):
                if "thought" in extracted_content:
                    if reasoning_content is None:
                        reasoning_content = extracted_content["thought"]
                    else:
                        reasoning_content += extracted_content["thought"]
                if "text" in extracted_content:
                    content += extracted_content["text"]
                if "inlineData" in extracted_content:
                    meta.setdefault("media_contents", []).append(extracted_content["inlineData"])
                if "functionCall" in extracted_content:
                    func_call = extracted_content["functionCall"]
                    tool_call_delta = ToolCallDelta(
                        index=chunk_index,
                        id=None,  # Gemini doesn't provide tool call IDs
                        tool_name=func_call["name"],
                        arguments=json.dumps(func_call["args"]),
                    )
                    tool_call_deltas.append(tool_call_delta)
                if "thoughtSignature" in extracted_content:
                    meta["thoughtSignature"] = extracted_content["thoughtSignature"]

            # We can't have both content and tool_calls in the same chunk
            # so we need to create separate chunks if both are present
            if tool_call_deltas:
                yield StreamingChunk(
                    content="",  # Empty content for tool call chunks
                    tool_calls=tool_call_deltas,
                    meta=meta,
                    index=chunk_index,
                    start=is_start,
                    finish_reason=finish_reason,
                )
            else:
                # For unsupported content or if no content was extracted, we still need to return a chunk
                yield StreamingChunk(
                    content=content,
                    reasoning_content=reasoning_content,
                    tool_calls=None,  # Properly use None instead of empty list
                    meta=meta,
                    index=chunk_index,
                    start=is_start,
                    finish_reason=finish_reason,
                )

    @staticmethod
    def _extract_candidate_contents(candidate: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
        """
        Parse a candidate from the Gemini API response to extract the content.

        :param candidate: The candidate data from the Gemini API response.
        :return: A generator yielding dictionaries with the parsed content.
        """
        if not candidate["content"]:  # Edge case check for empty content due to safety filters
            return
        for part in candidate["content"].get("parts", []):
            if part.get("thought", False):
                # This part is thought from the model instead of the actual response
                if "text" not in part:
                    raise ValueError("Thought content must contain text.")
                yield {"thought": part["text"]}
            else:
                # Check if part contains any of the supported types
                matching_keys = part.keys() & {"text", "inlineData", "functionCall", "thoughtSignature"}
                if len(matching_keys) > 0:
                    # Yield all matching keys together to preserve the part structure
                    yield {key: part[key] for key in matching_keys}
                else:
                    # No supported type found, log warning
                    unsupported_keys = list(part.keys())
                    logger.warning('Unsupported content types "%s" from Gemini completions.', unsupported_keys)

    @staticmethod
    def _extract_candidate_meta(
        candidate: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Extract metadata from the candidate.

        :param candidate: The candidate data from the Gemini API response.
        :return: A dictionary with the extracted metadata.
        """
        # Create metadata
        meta = {
            "index": candidate.get("index"),
            "finish_reason": candidate.get("finishReason"),
        }

        # Handle optional metadata fields from candidate if they exist
        for key in (
            "avgLogprobs",
            "logprobsResult",
            "safetyRatings",
            "citationMetadata",
            "groundingMetadata",
            "finishMessage",
        ):
            if key in candidate:
                meta[key] = candidate[key]
        return meta

    def _connect_chunks(self, chunks: List[StreamingChunk]) -> List[ChatMessage]:
        """
        Connect the streaming chunks to lists of ChatMessage, each representing a candidate of the response.

        :param chunks: The list of StreamingChunks.
        :return: The list of ChatMessage instances.
        """
        index_list = set()
        # mapping from candidate index to the chat message contents
        message_contents: Dict[int, List[ChatMessageContent]] = defaultdict(list)
        reasoning_contents: Dict[int, str] = defaultdict(str)
        meta_contents: Dict[int, Dict[str, Any]] = defaultdict(dict)

        for chunk in chunks:
            # Use chunk.index directly when available
            index = chunk.index if chunk.index is not None else chunk.meta.get("index", 0)
            index_list.add(index)
            contents = message_contents[index]

            if chunk.content:
                if contents and isinstance(contents[-1], TextContent):
                    # If the last content is a TextContent, append to it
                    contents[-1].text += chunk.content
                else:
                    contents.append(TextContent(text=chunk.content))

            if chunk.reasoning_content:
                reasoning_contents[index] += chunk.reasoning_content

            if chunk.tool_calls:
                for tc_delta in chunk.tool_calls:
                    try:
                        # Parse the arguments string into a Python dictionary
                        arguments = json.loads(tc_delta.arguments) if tc_delta.arguments else {}
                        contents.append(
                            ToolCall(
                                id=tc_delta.id or str(uuid.uuid4()),  # Gemini may not return an ID
                                tool_name=tc_delta.tool_name or "",
                                arguments=arguments,
                            )
                        )
                    except json.JSONDecodeError as e:
                        logger.warning(
                            "Could not parse tool call arguments: %s. Error: %s",
                            tc_delta.arguments,
                            str(e),
                        )

            for raw_media in chunk.meta.get("media_contents", []):
                contents.append(
                    MediaContent(media=ByteStream.from_base64(raw_media["data"], mime_type=raw_media["mimeType"]))
                )

            meta_contents[index].update(chunk.meta)

            # Make sure we preserve index and finish_reason in meta
            meta_contents[index]["index"] = index
            if chunk.finish_reason:
                meta_contents[index]["finish_reason"] = chunk.finish_reason

        messages = []
        for index in sorted(index_list):
            contents = message_contents[index]
            reasoning_content = reasoning_contents[index]
            meta = meta_contents[index]
            # Create the ChatMessage from scratch since content might contain unorthodox data types in a specific order
            messages.append(
                ChatMessage(
                    _role=ChatRole.ASSISTANT,
                    _content=contents,
                    _meta=meta,
                    _reasoning_content=reasoning_content,
                )
            )
        return messages

    def _build_message(
        self,
        candidate: Dict[str, Any],
        model: str,
        msg_id: Optional[str] = None,
        usage: Optional[Dict[str, Any]] = None,
    ) -> ChatMessage:
        """
        Build a ChatMessage from the Gemini API response.

        :param candidate: The content candidate of the message.
        :param model: The model used to generate the message.
        :param msg_id: The ID of the message.
        :param usage: The usage information from the API response.
        :return: A ChatMessage instance.
        """
        reasoning_content = None
        chat_message_content: List[ChatMessageContent] = []
        thought_signature: Optional[str] = None

        for extracted_content in self._extract_candidate_contents(candidate):
            if "thought" in extracted_content:
                reasoning_content = extracted_content["thought"]
            if "text" in extracted_content:
                chat_message_content.append(TextContent(text=extracted_content["text"]))
            if "inlineData" in extracted_content:
                inline_data = extracted_content["inlineData"]
                chat_message_content.append(
                    MediaContent(media=ByteStream.from_base64(inline_data["data"], mime_type=inline_data["mimeType"]))
                )
            if "functionCall" in extracted_content:
                func_call = extracted_content["functionCall"]
                chat_message_content.append(
                    ToolCall(
                        tool_name=func_call["name"],
                        arguments=func_call["args"],
                        # Gemini does not return tool call IDs. Assign a synthetic UUID so
                        # downstream deduplication (e.g. HiTL strategies) can distinguish
                        # multiple calls to the same tool within one response.
                        id=str(uuid.uuid4()),
                    )
                )
            if "thoughtSignature" in extracted_content:
                thought_signature = extracted_content["thoughtSignature"]

        meta = self._extract_candidate_meta(candidate)
        meta["model"] = model
        if msg_id:
            meta["msg_id"] = msg_id
        if usage:
            # We preserve the original usage format from the API response for transparency.
            # This might need to be converted later for specific tracing platforms.
            meta["usage"] = usage
        if thought_signature is not None:
            meta["thoughtSignature"] = thought_signature

        # Create the ChatMessage from scratch since content might contain unorthodox data types in a specific order
        return ChatMessage(
            _role=ChatRole.ASSISTANT,
            _content=chat_message_content,
            _meta=meta,
            _reasoning_content=reasoning_content,
        )

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
        Build a request according to the Gemini API format.

        :param client: The httpx client to use for the request.
        :param messages: The list of ChatMessage instances to send to the model.
        :param model: The model to use for the request.
        :param timeout: The timeout for the request.
        :param stream: Whether to stream the response.
        :param tools: A list of tools to include in the request.
        :param inference_kwargs: Additional generation keyword arguments to include in the request.
        :param extra_headers: Additional headers to include in the request.
        """
        query_params = {}
        headers = {"Content-Type": "application/json"}

        # Use provided model or fall back to default
        model_name = model or self.configs.get("model")
        if not model_name:
            raise ValueError("A model must be provided either during initialization or when calling the API")

        # Extract the system prompt if it exists
        if messages[0].role == ChatRole.SYSTEM:
            if len(messages[0].content) == 0:
                raise ValueError("The system prompt must contain at least one text message.")
            if not all(isinstance(item, TextContent) for item in messages[0].content):
                raise ValueError("Only text should be included in system instructions for Gemini completions.")
            system_prompts = messages[0].texts
            messages = messages[1:]
        else:
            system_prompts = None

        # Format messages for Gemini API
        gemini_formatted_messages = [self.format_chat_message(message) for message in messages]

        # Prepare request body
        request_body: Dict[str, Any] = {"contents": gemini_formatted_messages}

        # Add system instruction if present
        if system_prompts:
            request_body["systemInstruction"] = {"parts": [{"text": paragraph} for paragraph in system_prompts]}

        # Add generation config from inference_kwargs if provided
        if inference_kwargs:
            generation_config: Dict[str, Any] = {}

            for key, value in inference_kwargs.items():
                # Map common parameters since Gemini uses an uncommon naming convention
                if key in _GEMINI_PARAM_MAPPING:
                    generation_config[_GEMINI_PARAM_MAPPING[key]] = value
                else:
                    # Could convert camelCase to snake_case if needed, but trying to keep it simple for now
                    generation_config[key] = value

            if generation_config:
                request_body["generationConfig"] = generation_config

        if tools:
            # Now we use the function_declarations field for tools.
            # Gemini offers a few integrated tools (like Vertex RAG and Google search), but they are handled in a very
            # different way, and it is very unlikely that we will use those directly.
            function_declarations = [tool.tool_spec for tool in tools]
            request_body["tools"] = [{"functionDeclarations": function_declarations}]

        # Use the streaming endpoint if needed
        if stream:
            operation = "stream"
            query_params["alt"] = "sse"
        else:
            operation = "inference"

        # Authenticate through the documented header so credentials are not exposed in request URLs or HTTP errors.
        api_key = resolve_secret(self.api_key)
        if api_key:
            headers["x-goog-api-key"] = api_key

        route_model = model_name.removeprefix("models/")

        # Create and return the request
        return client.build_request(
            method="POST",
            url=self.build_request_url(operation, model=route_model),
            params=query_params,
            headers=headers | (extra_headers or {}),
            json=request_body,
            timeout=timeout,
        )

    @staticmethod
    def _check_finish_reason(message: ChatMessage) -> None:
        """
        Check the `finish_reason` returned with the Gemini completions, and log a warning if it did not reach a natural
        stopping point.

        :param message: The message returned by the LLM.
        """
        finish_reason = message.meta.get("finish_reason")
        match finish_reason:
            case "STOP":
                pass
            case "MAX_TOKENS":
                logger.warning(
                    "The assistant completion %s has been truncated before reaching a natural stopping point. "
                    "Increase the max_tokens parameter or pick another model to allow for longer completions.",
                    message.meta.get("index"),
                )
            case "SAFETY":
                logger.warning(
                    "The assistant completion %s was stopped due to safety filters. "
                    "Please check the safety ratings and try again: %s.",
                    message.meta.get("index"),
                    message.meta.get("safetyRatings"),
                )
            case "RECITATION":
                logger.warning(
                    "The assistant completion %s was stopped due to potential recitation. "
                    "Please try again with different prompt or inference parameters.",
                    message.meta.get("index"),
                )
            case "MALFORMED_FUNCTION_CALL":
                logger.warning(
                    "The assistant completion %s was stopped since it generated a malformed function call.",
                    message.meta.get("index"),
                )
            case None:
                logger.warning(
                    "The assistant completion %s has stopped without a finish reason. "
                    "This might be due to a bug in the Gemini API.",
                    message.meta.get("index"),
                )
            case _:
                logger.warning(
                    "The assistant completion %s has stopped with finish reason: %s. "
                    "Please check the original Gemini API documentation for more information.",
                    message.meta.get("index"),
                    finish_reason,
                )

    def _iter_streaming_chunks(self, response: httpx.Response) -> Iterable[StreamingChunk]:
        """Yield StreamingChunks from a synchronous Gemini SSE response."""
        for sse in ServerSentEventStream(response=response):
            yield from self._build_chunks(sse)

    async def _aiter_streaming_chunks(self, response: httpx.Response) -> AsyncIterable[StreamingChunk]:
        """Yield StreamingChunks from an asynchronous Gemini SSE response."""
        async for sse in AsyncServerSentEventStream(response=response):
            for chunk in self._build_chunks(sse):
                yield chunk

    def parse_response(
        self,
        response: httpx.Response,
        *,
        stream: bool = False,
    ) -> List[ChatMessage]:
        """
        Parse the response from the Gemini generateResponse API to a list of ChatMessage instances. If streaming is
        enabled, the response is parsed as a stream of ServerSentEvents, and each event is captured as a StreamingChunk.

        :param response: The response from the model service.
        :param stream: Whether the response is streamed.
        :return: The list of ChatMessage instances containing the generated responses.
        """
        if stream:
            chunks = self._collect_streaming_chunks(self._iter_streaming_chunks(response))
            chat_messages = self._connect_chunks(chunks)
        else:
            response_json = response.json()
            chat_messages = [
                self._build_message(
                    candidate=candidate,
                    model=response_json.get("modelVersion"),
                    msg_id=response_json.get("responseId"),
                    usage=response_json.get("usageMetadata"),
                )
                for candidate in response_json["candidates"]
            ]
        for message in chat_messages:
            self._check_finish_reason(message)
        return chat_messages

    async def aparse_response(
        self,
        response: httpx.Response,
        *,
        stream: bool = False,
    ) -> List[ChatMessage]:
        """
        Asynchronously parse the response from the Gemini generateResponse API to a list of ChatMessage instances. If
        streaming is enabled, the response is parsed as an async stream of ServerSentEvents, and each event is captured
        as a StreamingChunk.

        :param response: The response from the model service.
        :param stream: Whether the response is streamed.
        :return: The list of ChatMessage instances containing the generated responses.
        """
        if stream:
            chunks = await self._collect_streaming_chunks_async(self._aiter_streaming_chunks(response))
            chat_messages = self._connect_chunks(chunks)
        else:
            response_json = response.json()
            chat_messages = [
                self._build_message(
                    candidate=candidate,
                    model=response_json.get("modelVersion"),
                    msg_id=response_json.get("responseId"),
                    usage=response_json.get("usageMetadata"),
                )
                for candidate in response_json["candidates"]
            ]
        for message in chat_messages:
            self._check_finish_reason(message)
        return chat_messages


@register_model_service("vertex-anthropic-chat")
class VertexAnthropicChatModelService(AnthropicChatModelService):
    """
    A service for interacting with Vertex AI's Anthropic API.
    """

    DEFAULT_ROUTES = MappingProxyType(
        {
            "inference": "/v1/{model}:rawPredict",
            "stream": "/v1/{model}:streamRawPredict",
        }
    )

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
        Build a request according to the Vertex AI Anthropic Chat Completion API format.

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
        model_name = model or self.configs.get("model")
        if not model_name:
            raise ValueError("A Vertex AI model resource must be provided when calling the API")

        # Extract the system prompt if it exists
        if messages[0].role == ChatRole.SYSTEM:
            if len(messages[0]) != 1 or not isinstance(messages[0].content[0], TextContent):
                raise ValueError("The system prompt must be a single text message.")
            system_prompt = messages[0].text
            messages = messages[1:]
        else:
            system_prompt = None

        # Format the rest of the messages
        anthropic_formatted_messages = [self.format_chat_message(message) for message in messages]

        max_tokens = inference_kwargs.get("max_tokens")
        if max_tokens is None:
            max_tokens = self.default_max_tokens(model or "claude-3-5-sonnet")
            logger.info(
                "max_tokens is required for Vertex Anthropic Chat API. Using default value inferred from model: %s",
                max_tokens,
            )

        # Create the body of the request
        body = {
            "anthropic_version": "vertex-2023-10-16",
            "messages": anthropic_formatted_messages,
            "max_tokens": max_tokens,
            "stream": stream,
            **inference_kwargs,
        }

        if system_prompt:
            body["system"] = system_prompt

        # Create the headers for the request
        headers = {
            "Content-Type": "application/json",
        }
        api_key = resolve_secret(self.api_key)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
            headers["api-key"] = api_key

        # Include tool definitions if provided
        if tools:
            body["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools
            ]

        # Build the request
        request = client.build_request(
            "POST",
            url=self.build_request_url("stream" if stream else "inference", model=model_name.removeprefix("v1/")),
            json=body,
            headers=headers | (extra_headers or {}),
            timeout=timeout,
        )
        return request
