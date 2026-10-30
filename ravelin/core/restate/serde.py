"""Custom Serde implementations and DefaultSerde monkey-patching for Restate."""

import json
import logging
import types
from dataclasses import fields, is_dataclass
from typing import Any, Optional, Type, Union, get_args, get_origin, get_type_hints

from restate.serde import Serde, I, Msgspec, is_pydantic, DefaultSerde, DaciteFromDict
from haystack.core.serialization import generate_qualified_class_name, import_class_by_name

from ravelin.core.serialization import _try_match_union_type, deserialize_value, serialize_value


logger = logging.getLogger(__name__)


def _deserialize_nested_metadata(data: Any) -> Any:
    """Reconstruct nested metadata without swallowing custom decoder errors."""
    if isinstance(data, list):
        return [_deserialize_nested_metadata(value) for value in data]
    if isinstance(data, dict):
        type_name = data.get("__type__")
        if type_name:
            try:
                hint = import_class_by_name(type_name)
            except ImportError:
                # Match the existing metadata fallback when no declared type
                # is available, while still reconstructing known nested values.
                data = {key: value for key, value in data.items() if key != "__type__"}
            else:
                return _deserialize_typed_value(data, hint)
        return {key: _deserialize_nested_metadata(value) for key, value in data.items()}
    return data


def _deserialize_union(data: Any, hint: Any, args: tuple) -> Any:
    """Select a union member before decoding; never retry a failed decoder."""
    non_none = [arg for arg in args if arg is not type(None)]
    if len(non_none) == 1:
        return _deserialize_typed_value(data, non_none[0])
    if isinstance(data, dict) and data.get("__type__"):
        encoded_type = import_class_by_name(data["__type__"])
        if not any(arg is Any or (isinstance(arg, type) and issubclass(encoded_type, arg)) for arg in args):
            raise TypeError(f"Encoded type is not a member of Union {hint}")
        return _deserialize_typed_value(data, encoded_type)
    matched_type = _try_match_union_type(data, args)
    if matched_type is None:
        raise ValueError(f"Cannot deserialize ambiguous Union {hint} without __type__ metadata")
    return _deserialize_typed_value(data, matched_type)


def _deserialize_typed_value(data: Any, hint: Any) -> Any:
    """Preserve dictionary protocols while validating ordinary typed dataclasses.

    Custom ``from_dict`` methods own their wire format and take precedence over
    dataclass construction. Their failures must propagate, not fall back to a
    constructor that might accept data the custom decoder rejected.
    """
    if data is None:
        return None
    if hint in (str, int, float, bool, type(None)):
        # Let the enclosing dataclass validator check scalars. The generic
        # reconstruction fallback could otherwise turn {} into str() or bool().
        return data
    origin = get_origin(hint)
    args = get_args(hint)
    if origin in (Union, types.UnionType):
        return _deserialize_union(data, hint, args)
    if origin is list and isinstance(data, list):
        return [_deserialize_typed_value(item, args[0] if args else Any) for item in data]
    if origin is dict and isinstance(data, dict) and "__type__" not in data:
        return {key: _deserialize_typed_value(value, args[1] if args else Any) for key, value in data.items()}

    if isinstance(hint, type) and isinstance(data, dict) and (is_dataclass(hint) or hasattr(hint, "from_dict")):
        type_name = data.get("__type__")
        if type_name and type_name != generate_qualified_class_name(hint):
            try:
                encoded_type = import_class_by_name(type_name)
            except ImportError:
                # The explicit hint is sufficient for a local or renamed class.
                pass
            else:
                if not issubclass(encoded_type, hint):
                    raise TypeError(f"Encoded type is not a subclass of {hint.__name__}")
                hint = encoded_type

    if callable(getattr(hint, "from_dict", None)) and isinstance(data, dict):
        # Metadata-bearing objects historically received reconstructed nested
        # values. Plain custom wire dictionaries remain owned by from_dict.
        values = {key: value for key, value in data.items() if key != "__type__"}
        if data.get("__type__"):
            values = {key: _deserialize_nested_metadata(value) for key, value in values.items()}
        return hint.from_dict(values)
    if isinstance(hint, type) and is_dataclass(hint):
        if not isinstance(data, dict):
            raise TypeError(f"Expected an object for {hint.__name__}")
        hints = get_type_hints(hint)
        # Decode known nested fields first so Haystack dictionary protocols also
        # work inside ordinary dataclasses. Never decode ignored extra fields.
        values = {
            field.name: _deserialize_typed_value(data[field.name], hints[field.name])
            for field in fields(hint)
            if field.name in data
        }
        # Retain InitVar inputs for Dacite, which also handles constructor
        # defaults, missing fields and type checks without coercing values.
        values.update({name: data[name] for name in hints if name in data and name not in values})
        return DaciteFromDict(hint, values)
    return deserialize_value(data, hint)


# Helper to recursively convert objects to dicts
def to_dict_encoder(o: Any) -> Any:
    """
    Encode objects for JSON serialization using serialize_value().

    :param o: Object passed by ``json.dumps(default=...)``.
    :returns: JSON-compatible representation with ``__type__`` metadata when available.
    :raises TypeError: If the object cannot be serialized by the project serializer.
    """
    serialized = serialize_value(o)
    if serialized is o:
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")
    return serialized


class DictProtocolSerde(Serde[I]):
    """
    Serde for classes that implement to_dict() and from_dict().
    """

    def __init__(self, type_hint: Type[I]) -> None:
        """
        Initialize the serde with the expected payload type.

        :param type_hint: Type used when deserializing JSON-compatible data.
        """
        self.type_hint = type_hint

    def deserialize(self, buf: bytes) -> Optional[I]:
        """
        Deserializes a bytearray to an object using from_dict.

        :param buf: Serialized JSON bytes.
        :returns: Deserialized object, or ``None`` when ``buf`` is empty.
        """
        if not buf:
            return None

        # 1. Parse bytes to dict (JSON)
        data = json.loads(buf)

        # 2. Use the class method to instantiate
        # This relies on the type_hint passed during initialization
        return _deserialize_typed_value(data, self.type_hint)

    def serialize(self, obj: Optional[I]) -> bytes:
        """
        Serializes an object to a bytearray using to_dict.

        :param obj: Object to serialize.
        :returns: Serialized JSON bytes, or empty bytes when ``obj`` is ``None``.
        """
        if obj is None:
            return bytes()

        # json.dumps will traverse Lists and Dicts automatically.
        # When it hits your custom object, it uses 'default_encoder'
        return json.dumps(obj, default=to_dict_encoder).encode("utf-8")


################################
# Monkey patching DefaultSerde #
################################


def patch_default_serde() -> None:
    """Monkey-patch DefaultSerde to support DictProtocol-based deserialization."""

    def patched_deserialize(self, buf: bytes) -> Optional[I]:
        """
        Deserializes a byte array into a Python object.

        :param buf: The byte array to deserialize.
        :returns: The resulting Python object, or ``None`` if the input is empty.
        """
        if not buf:
            return None
        hint = self.type_hint
        if not hint:
            return json.loads(buf)
        if Msgspec.is_struct(hint):
            return Msgspec.decode(buf, type=hint)
        if is_pydantic(hint):
            return hint.model_validate_json(buf)  # type: ignore

        # Keep Haystack dictionary protocols, with native validation for ordinary
        # dataclasses instead of the generic serializer's dictionary fallback.
        json_data = json.loads(buf)
        return _deserialize_typed_value(json_data, hint)

    def patched_serialize(self, obj: Optional[I]) -> bytes:
        """
        Serializes a Python object into a byte array.

        If the object is a msgspec Struct or Pydantic BaseModel, uses their respective methods.

        :param obj: The Python object to serialize.
        :returns: The serialized byte array.
        """
        if obj is None:
            return bytes()
        hint = self.type_hint
        if not hint:
            return json.dumps(obj).encode("utf-8")
        if Msgspec.is_struct(hint):
            return Msgspec.encode(obj)
        if is_pydantic(hint):
            return obj.model_dump_json().encode("utf-8")  # type: ignore[attr-defined]

        # Patched code start #
        # This replaces both the original `is_dataclass` branch (which used Restate's DaciteToDict)
        # and the original json.dumps fallback. All non-msgspec/pydantic types go through our own
        # serialize_value, which handles dataclasses, ChatMessage, lists, dicts, etc. consistently.
        return json.dumps(obj, default=to_dict_encoder).encode("utf-8")
        # Patched code end #

    # Apply the patch
    DefaultSerde.deserialize = patched_deserialize  # type: ignore
    DefaultSerde.serialize = patched_serialize  # type: ignore

    logger.debug("DefaultSerde patched successfully with DictProtocol support.")
