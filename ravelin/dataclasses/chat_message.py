# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
"""
Data classes for representing chat messages in a multimodal conversation. A chat message can contain text, media, Tool
calls, and Tool call results. The chat message can be sent by the user, the system, the assistant, or a tool.
"""

import logging
from base64 import b64decode, b64encode
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union, cast

from haystack.dataclasses.chat_message import (
    ChatMessage as HaystackChatMessage,
    ChatRole,
    TextContent,
    ReasoningContent,
    ToolCall,
    ToolCallResult,
    _serialize_content_part,
    _deserialize_content_part,
)
from haystack.dataclasses.image_content import ImageContent
from typing_extensions import Self

from ravelin.dataclasses.byte_stream import ByteStream


@dataclass
class MediaContent:
    """
    The media content of a chat message.

    :param media: The media content of the message.
    """

    media: ByteStream


ChatMessageContent = Union[TextContent, MediaContent, ToolCall, ToolCallResult]


@dataclass
class ChatMessage(HaystackChatMessage):
    """
    Represents a message in a LLM chat conversation.
    This class is a subclass of Haystack's ChatMessage to provide compatibility with Haystack components,
    while also supporting custom features like `MediaContent` and `reasoning_content`.
    :param content: The content of the message.
    :param role: The role of the entity sending the message.
    :param meta: Additional metadata associated with the message.
    :param name: An optional name for the message participant. Default to None.
    """

    # We redefine the fields from the parent dataclass to control the order and types.
    _role: ChatRole
    _content: Sequence[ChatMessageContent]  # type: ignore # This is intentionally different from the parent class
    _reasoning_content: Optional[str] = field(default=None)
    _name: Optional[str] = field(default=None)
    _meta: Dict[str, Any] = field(default_factory=dict, hash=False)

    def __getattribute__(self, name: str) -> Any:
        """
        We override this method to bypass the parent's block on the `content` attribute.
        This allows us to define our own `content` property.
        """
        return object.__getattribute__(self, name)

    @property
    def content(self) -> Sequence[ChatMessageContent]:
        """
        Returns the content of the message.
        """
        return self._content

    @property
    def reasoning_content(self) -> Optional[str]:
        """
        Returns the reasoning content of the message.
        """
        return self._reasoning_content

    @property
    def reasonings(self) -> List["ReasoningContent"]:
        """
        Returns the list of all reasoning content contained in the message.
        This is for compatibility with Haystack components.
        """
        return [ReasoningContent(reasoning_text=self._reasoning_content)] if self._reasoning_content else []

    @property
    def reasoning(self) -> Optional["ReasoningContent"]:
        """
        Returns the reasoning content of the message, converted to Haystack's ReasoningContent.
        This is for compatibility with Haystack components.
        """
        if self._reasoning_content:
            return ReasoningContent(reasoning_text=self._reasoning_content)
        return None

    @property
    def media(self) -> List[ByteStream]:
        """
        Returns the list of all media content contained in the message.
        :return: List of ByteStream objects.
        """
        return [content.media for content in self._content if isinstance(content, MediaContent)]

    @property
    def images(self) -> List["ImageContent"]:
        """
        Returns the list of all images contained in the message, converted to Haystack's ImageContent.
        This is for compatibility with Haystack components.
        """
        image_contents = []
        for content in self._content:
            if isinstance(content, MediaContent) and content.media.mime_type and "image" in content.media.mime_type:
                try:
                    base64_data = b64encode(content.media.data).decode("utf-8")
                    img_content = ImageContent(base64_image=base64_data, mime_type=content.media.mime_type)
                    image_contents.append(img_content)
                except Exception as e:
                    # Could fail if Pillow is not installed or image data is corrupt.
                    raise ValueError(
                        f"Failed to convert media content to ImageContent: {content.media.mime_type}"
                    ) from e
        return image_contents

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts ChatMessage into a dictionary.
        This method overrides Haystack's `to_dict` to support `MediaContent` and `reasoning_content`.
        :returns:
            Serialized version of the object.
        """
        serialized: Dict[str, Any] = {"role": self._role.value, "meta": self._meta, "name": self._name}

        content: List[Dict[str, Any]] = []
        for part in self._content:
            if isinstance(part, MediaContent):
                content.append({"media": part.media.to_dict()})
            else:
                # For Haystack native types, use its own serializer
                content.append(_serialize_content_part(part))

        serialized["content"] = content
        if self._reasoning_content:
            serialized["reasoning_content"] = self._reasoning_content
        return serialized

    @classmethod
    def from_haystack_message(cls, haystack_message: HaystackChatMessage) -> Self:
        """
        Create a `ChatMessage` from a `HaystackChatMessage`.
        This is a convenience method to convert messages from Haystack components.
        It handles the conversion of `ImageContent` to `MediaContent`.
        :param haystack_message: The Haystack message to convert.
        :return: A new `ChatMessage` instance.
        """
        # We access the private _content member to preserve the order of the content parts.
        new_content: List[ChatMessageContent] = []
        reasoning_content: Optional[str] = None
        for part in haystack_message._content:  # type: ignore # pylint: disable=protected-access
            if isinstance(part, ImageContent):
                image_bytes = b64decode(part.base64_image)
                byte_stream = ByteStream(data=image_bytes, mime_type=part.mime_type)
                new_content.append(MediaContent(media=byte_stream))
            elif isinstance(part, ReasoningContent):
                reasoning_content = part.reasoning_text
            elif isinstance(part, (TextContent, ToolCall, ToolCallResult)):
                new_content.append(part)
            else:
                logging.warning("Unexpected content type: %s", type(part))

        return cls(
            _role=haystack_message.role,
            _content=new_content,
            _name=haystack_message.name,
            _meta=haystack_message.meta,
            _reasoning_content=reasoning_content,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatMessage":
        """
        Creates a new ChatMessage object from a dictionary.
        This method overrides Haystack's `from_dict` to support `MediaContent` and `reasoning_content`.

        :param data:
            The dictionary to build the ChatMessage object.
        :returns:
            The created object.
        """
        content_data = data.get("content", [])
        content: List[ChatMessageContent] = []

        if isinstance(content_data, str):
            content.append(TextContent(text=content_data))
        else:
            for part in content_data:
                if "media" in part:
                    content.append(MediaContent(media=ByteStream.from_dict(part["media"])))
                else:
                    try:
                        # For Haystack native types, use its own deserializer with a cast to ensure type safety
                        deserialized_part = cast(ChatMessageContent, _deserialize_content_part(part))
                        content.append(deserialized_part)
                    except ValueError as e:
                        raise ValueError(f"Unsupported content in serialized ChatMessage: `{part}`") from e

        return cls(
            _role=ChatRole(data["role"]),
            _content=content,
            _name=data.get("name"),
            _meta=data.get("meta", {}),
            _reasoning_content=data.get("reasoning_content"),
        )

    @classmethod
    # pylint: disable=arguments-differ,arguments-renamed
    def from_user(  # type: ignore # We deliberately override the parent method with different signature
        cls,
        text: Optional[str] = None,
        media: Optional[Sequence[ByteStream]] = None,
        meta: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
    ) -> "ChatMessage":
        """
        Create a message from the user.

        :param text: The text content of the message.
        :param media: The media contents of the message, if any.
        :param meta: Additional metadata associated with the message.
        :param name: An optional name for the message participant.
        :returns: A new ChatMessage instance.
        """
        media_contents = [MediaContent(media=m) for m in media] if media else []
        content: List[ChatMessageContent] = [*media_contents]
        if text:
            content.append(TextContent(text=text))
        return cls(_role=ChatRole.USER, _content=content, _name=name, _meta=meta or {})

    @classmethod
    def from_system(cls, text: str, meta: Optional[Dict[str, Any]] = None, name: Optional[str] = None) -> "ChatMessage":
        """
        Create a message from the system.
        :param text: The text content of the message.
        :param meta: Additional metadata associated with the message.
        :param name: An optional name for the message participant.
        :returns: A new ChatMessage instance.
        """
        return cls(_role=ChatRole.SYSTEM, _content=[TextContent(text=text)], _name=name, _meta=meta or {})

    @classmethod
    # pylint: disable=arguments-differ
    def from_assistant(  # type: ignore
        cls,
        text: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        tool_calls: Optional[List[ToolCall]] = None,
        *,
        reasoning_content: Optional[str] = None,
    ) -> "ChatMessage":
        """
        Create a message from the assistant.

        :param text: The text content of the message.
        :param meta: Additional metadata associated with the message.
        :param name: An optional name for the message participant.
        :param tool_calls: The Tool calls to include in the message.
        :param reasoning_content: The reasoning text content of the message.
        :returns: A new ChatMessage instance.
        """
        content: List[ChatMessageContent] = []
        if text is not None:
            content.append(TextContent(text=text))
        if tool_calls:
            content.extend(tool_calls)

        return cls(
            _role=ChatRole.ASSISTANT,
            _content=content,
            _reasoning_content=reasoning_content,
            _name=name,
            _meta=meta or {},
        )

    @classmethod
    def from_tool(
        cls,
        tool_result: Union[str, Sequence[Union[TextContent, ImageContent]]],
        origin: ToolCall,
        error: bool = False,
        meta: Optional[Dict[str, Any]] = None,
    ) -> "ChatMessage":
        """
        Create a message from a Tool.

        :param tool_result: The result of the Tool invocation. Only ``str`` is currently supported;
            passing a sequence of content parts will raise ``NotImplementedError``.
        :param origin: The Tool call that produced this result.
        :param error: Whether the Tool invocation resulted in an error.
        :param meta: Additional metadata associated with the message.
        :returns: A new ChatMessage instance.
        """
        # TODO: Support Sequence[TextContent | ImageContent] tool results.
        #  Haystack now accepts multimodal tool results, but we replace Haystack's ImageContent with
        #  our MediaContent (ByteStream-based). Until the two content models are aligned, non-str
        #  tool results are not supported. MediaContent should map to a content-part list here,
        #  and ImageContent inputs (from upstream Haystack callers) should be converted to MediaContent.
        if not isinstance(tool_result, str):
            raise NotImplementedError(
                "Only str tool results are currently supported. "
                "Multimodal tool results (ImageContent sequences) are not yet implemented."
            )
        return cls(
            _role=ChatRole.TOOL,
            _content=[ToolCallResult(result=tool_result, origin=origin, error=error)],
            _meta=meta or {},
        )

    def _to_trace_dict(self) -> dict[str, Any]:
        """
        Convert the ChatMessage to a dictionary representation for tracing.

        :returns:
            Serialized version of the object only for tracing purposes.
        """

        serialized: Dict[str, Any] = {"role": self._role.value, "meta": self._meta, "name": self._name}

        content: List[Dict[str, Any]] = []
        for part in self._content:
            if isinstance(part, MediaContent):
                content.append({"media": part.media._to_trace_dict()})  # pylint: disable=protected-access
            else:
                # For Haystack native types, use its own serializer
                content.append(_serialize_content_part(part))

        serialized["content"] = content
        if self._reasoning_content:
            serialized["reasoning_content"] = self._reasoning_content
        return serialized
