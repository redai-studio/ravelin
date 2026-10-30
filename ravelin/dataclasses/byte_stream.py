"""
Data classes for representing binary data in the Haystack API. The ByteStream class can be used to represent binary data
in the API, and can be converted to and from base64 encoded strings, dictionaries, and files. This is particularly
useful for representing media files in chat messages.
"""

import logging
from base64 import b64encode, b64decode
from dataclasses import dataclass
from typing import Any, Dict, Optional

from haystack.dataclasses import ByteStream as HaystackByteStream


logger = logging.getLogger(__name__)


@dataclass
class ByteStream(HaystackByteStream):
    """
    Ravelin customized ByteStream with enhanced support in base64, (de)serialization, and mime types.

    :param data: The binary data stored in Bytestream.
    :param meta: Additional metadata to be stored with the ByteStream.
    :param mime_type: The mime type of the binary data.
    """

    # # Field definitions from base class
    # data: bytes
    # meta: dict[str, Any] = field(default_factory=dict, hash=False)
    # mime_type: Optional[str] = field(default=None)

    @property
    def type(self) -> Optional[str]:
        """
        Return the type of the ByteStream. This is the first part of the mime type, or None if the mime type is not set.

        :return: The type of the ByteStream.
        """
        if self.mime_type:
            return self.mime_type.split("/", maxsplit=1)[0]
        return None

    @property
    def subtype(self) -> Optional[str]:
        """
        Return the subtype of the ByteStream. This is the second part of the mime type,
        or None if the mime type is not set.

        :return: The subtype of the ByteStream.
        """
        if self.mime_type:
            return self.mime_type.split("/", maxsplit=1)[-1]
        return None

    @classmethod
    def from_base64(
        cls,
        base64_string: str,
        encoding: str = "utf-8",
        meta: Optional[Dict[str, Any]] = None,
        mime_type: Optional[str] = None,
    ) -> "ByteStream":
        """
        Create a ByteStream from a base64 encoded string.

        :param base64_string: The base64 encoded string representation of the ByteStream data.
        :param encoding: The encoding used to convert the base64 string into bytes.
        :param meta: Additional metadata to be stored with the ByteStream.
        :param mime_type: The mime type of the file.
        :returns: A new ByteStream instance.
        """
        return cls(data=b64decode(base64_string.encode(encoding)), meta=meta or {}, mime_type=mime_type)

    def to_base64(self, encoding: str = "utf-8") -> str:
        """
        Convert the ByteStream data to a base64 encoded string.

        :returns: The base64 encoded string representation of the ByteStream data.
        """
        return b64encode(self.data).decode(encoding)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], encoding: str = "utf-8") -> "ByteStream":
        """
        Create a ByteStream from a dictionary.

        :param data: The dictionary representation of the ByteStream.
        :param encoding: The encoding used to convert the base64 string into bytes.
        :returns: A new ByteStream instance.
        """
        return cls.from_base64(data["data"], encoding=encoding, meta=data.get("meta"), mime_type=data.get("mime_type"))

    def to_dict(self, encoding: str = "utf-8"):
        """
        Convert the ByteStream to a dictionary.

        :param encoding: The encoding used to convert the bytes to a string. Defaults to "utf-8".
        :returns: The dictionary representation of the ByteStream.
        :raises: UnicodeDecodeError: If the ByteStream data cannot be decoded with the specified encoding.
        """
        return {"data": self.to_base64(encoding=encoding), "meta": self.meta, "mime_type": self.mime_type}

    def _to_trace_dict(self, encoding: str = "utf-8") -> dict[str, Any]:
        """
        Convert the ByteStream to a dictionary representation for tracing.

        Binary data is replaced with a placeholder string to avoid sending large payloads to the tracing backend.

        To be deprecated when Haystack 2.20.1 goes live, as this is integrated in the base ByteStream class.

        :param encoding: The encoding used to convert the bytes to a string. Defaults to "utf-8".
        :returns:
            Serialized version of the object only for tracing purposes.
        """
        return {
            "data": f"Base 64 data ({len(self.to_base64(encoding=encoding))} bytes)",
            "meta": self.meta,
            "mime_type": self.mime_type,
        }
