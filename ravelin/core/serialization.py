"""
Serialization utilities for RestatePipeline.

This module provides unified JSON-compatible serialization/deserialization for:
1. HTTP boundary (PipelineIOHandler with type hints from InputSpec/OutputSpec)
2. Internal service communication (executor ↔ workflow with socket type hints)

Deserialization strategy (type-hint driven with __type__ fallback):
1. If type_hint provided: use it to guide deserialization
   - Optional[X]: extract X, recurse
   - List[X] / Dict[K, V]: recurse with inner types
   - Union[A, B, ...]: try to match, fall back to __type__ if ambiguous
   - Concrete type: reconstruct using type_hint
2. If no type_hint or ambiguous Union:
   - If __type__ metadata present: use it to determine type
   - Else: return as-is (primitives) or recurse (containers)

Serialization always adds __type__ metadata for complex objects (to_dict,
model_dump, dataclass) to enable reconstruction when type hints are unavailable.
"""

import base64
import dataclasses
import datetime
import decimal
import enum
import inspect
import json
import logging
import types
import uuid
from typing import Any, Callable, List, Dict, Union, get_origin, get_args

from haystack.core.serialization import generate_qualified_class_name, import_class_by_name

logger = logging.getLogger(__name__)


# =============================================================================
# Type matching utilities for Union disambiguation
# =============================================================================

# Primitive types that can be matched by isinstance check
_PRIMITIVE_TYPES = (str, int, float, bool, type(None))


def _is_primitive_type(t: type) -> bool:
    """Check if a type is a primitive that can be matched directly."""
    return t in _PRIMITIVE_TYPES


def _try_match_union_type(value: Any, union_args: tuple) -> type | None:
    """
    Try to match a value against Union type arguments.

    Returns the matching type if unambiguous, None if ambiguous or no match.

    Matching rules:
    - None value matches NoneType
    - Primitives (str, int, float, bool) match their types directly
    - list values match List[X] types
    - dict values WITHOUT __type__ match Dict[K, V] types
    - several matching List[X] / Dict[K, V] types are narrowed by their element types
    - dict values WITH __type__ are serialized objects → return None to signal
      that __type__ metadata should be used instead

    :param value: The value to match.
    :param union_args: The type arguments from Union[A, B, ...].
    :returns: The matching type, or None if should use __type__ metadata.
    """
    if value is None:
        if type(None) in union_args:
            return type(None)
        return None

    # dict with __type__ is always a serialized object
    # Return None to signal: use __type__ metadata for deserialization
    if isinstance(value, dict) and "__type__" in value:
        return None

    matches = []

    for arg in union_args:
        origin = get_origin(arg)

        # Check primitive types
        if _is_primitive_type(arg):
            if isinstance(value, arg):
                matches.append(arg)
            continue

        # Check List[X]
        if origin in (list, List):
            if isinstance(value, list):
                matches.append(arg)
            continue

        # Check Dict[K, V] - only for dicts WITHOUT __type__
        if origin in (dict, Dict):
            if isinstance(value, dict):
                matches.append(arg)
            continue

    # Unambiguous match
    if len(matches) == 1:
        return matches[0]

    # Several container types accept the same container (e.g. List[str] and List[UrlParse] both accept a list).
    # Narrow by element types, which serialization preserves: primitives stay primitives, objects carry __type__.
    if len(matches) > 1 and all(get_origin(m) in (list, List, dict, Dict) for m in matches):
        return _match_container_by_elements(value, matches)

    return None


def _element_matches(element: Any, element_type: Any) -> bool | None:
    """
    Check whether a serialized element decodes to ``element_type``.

    :returns: True or False when decidable, None when it cannot be decided (e.g. ``Any`` or generic element types).
    """
    if element_type is Any:
        return None
    origin = get_origin(element_type)
    if origin in (list, List, dict, Dict):
        return _container_matches(element, element_type)
    if not isinstance(element_type, type):
        return None
    if isinstance(element, dict) and "__type__" in element:
        try:
            encoded_type = import_class_by_name(element["__type__"])
        except ImportError:
            return None
        return issubclass(encoded_type, element_type)
    if _is_primitive_type(element_type):
        # bool is an int subclass, but JSON keeps them apart.
        if element_type in (int, float) and isinstance(element, bool):
            return False
        return isinstance(element, element_type)
    # An object type: primitives cannot decode to it, plain dicts/lists are undecidable without __type__.
    return False if _is_primitive_type(type(element)) else None


def _container_matches(value: Any, container_type: Any) -> bool | None:
    """
    Check whether a serialized container decodes to ``container_type`` (``List[X]`` or ``Dict[K, V]``), recursing into
    its elements so nested generics such as ``List[Dict[str, int]]`` are checked all the way down.

    :returns: True or False when decidable, None when an element cannot be decided. An empty container matches.
    """
    is_dict_type = get_origin(container_type) in (dict, Dict)
    if not isinstance(value, dict if is_dict_type else list):
        return False
    if is_dict_type and "__type__" in value:
        # A serialized object, not a plain mapping.
        return False
    args = get_args(container_type)
    elements: list[Any]
    if isinstance(value, dict):
        element_type = args[1] if len(args) >= 2 else Any
        elements = list(value.values())
    else:
        element_type = args[0] if args else Any
        elements = list(value)
    results = [_element_matches(element, element_type) for element in elements]
    if False in results:
        return False
    if None in results:
        return None
    return True


def _match_container_by_elements(value: Any, candidates: list) -> type | None:
    """
    Pick the single container type that accepts every element of ``value``, recursing into nested containers.

    An empty container decodes to the same value under every candidate, so the first candidate is returned.

    :returns: The matching container type, or None if still ambiguous or undecidable.
    """
    if not value:
        return candidates[0]

    accepted = []
    for candidate in candidates:
        result = _container_matches(value, candidate)
        if result is None:
            return None
        if result:
            accepted.append(candidate)

    if len(accepted) == 1:
        return accepted[0]
    return None


# =============================================================================
# Standard library value types
# =============================================================================


# Value types without to_dict()/dataclass support that components commonly return. Each is encoded as
# {"__type__": <qualified name>, "value": <JSON value>} so it survives the JSON boundary and keeps its type.
def _serialize_items(values: Any) -> list[Any]:
    return [serialize_value(item) for item in values]


def _serialize_set_items(values: Any) -> list[Any]:
    # Set elements must be hashable when restored, but tuples otherwise serialize as plain lists; tag them so they come
    # back as tuples. Set iteration order depends on hashing, which varies between processes; sort by the encoded form
    # so the same set always produces the same bytes (Restate compares journaled call parameters on replay).
    items = [_serialize_hashable(item) for item in values]
    return sorted(items, key=lambda item: json.dumps(item, sort_keys=True))


def _serialize_hashable(value: Any) -> Any:
    if type(value) is tuple:  # pylint: disable=unidiomatic-typecheck  # namedtuples keep their own encoding
        return {"__type__": "builtins.tuple", "value": [_serialize_hashable(item) for item in value]}
    return serialize_value(value)


def _deserialize_items(values: Any) -> Any:
    return deserialize_value(values)


_STDLIB_ENCODERS: dict[type, Callable[[Any], Any]] = {
    datetime.datetime: lambda v: v.isoformat(),
    datetime.date: lambda v: v.isoformat(),
    datetime.time: lambda v: v.isoformat(),
    datetime.timedelta: lambda v: v.total_seconds(),
    decimal.Decimal: str,
    uuid.UUID: str,
    bytes: lambda v: base64.b64encode(v).decode("ascii"),
    set: _serialize_set_items,
    frozenset: _serialize_set_items,
}
_STDLIB_DECODERS: dict[type, Callable[[Any], Any]] = {
    datetime.datetime: datetime.datetime.fromisoformat,
    datetime.date: datetime.date.fromisoformat,
    datetime.time: datetime.time.fromisoformat,
    datetime.timedelta: lambda v: datetime.timedelta(seconds=v),
    decimal.Decimal: decimal.Decimal,
    uuid.UUID: uuid.UUID,
    bytes: base64.b64decode,
    tuple: lambda v: tuple(_deserialize_items(v)),
    set: lambda v: set(_deserialize_items(v)),
    frozenset: lambda v: frozenset(_deserialize_items(v)),
}


def _encode_stdlib_value(value: Any) -> dict[str, Any] | None:
    # Exact type lookup: datetime is a date subclass, and bool/str subclasses must keep their own handling.
    encoder = _STDLIB_ENCODERS.get(type(value))
    if encoder is not None:
        return {"__type__": generate_qualified_class_name(type(value)), "value": encoder(value)}
    if isinstance(value, enum.Enum) and not isinstance(value, (str, int)):
        # Member values are looked up by equality, so tuples must come back as tuples.
        return {"__type__": generate_qualified_class_name(type(value)), "value": _serialize_hashable(value.value)}
    return None


def _decode_stdlib_value(cls: type, data: dict[str, Any]) -> tuple[bool, Any]:
    """Decode a value encoded by ``_encode_stdlib_value``. Returns (handled, value)."""
    if set(data) != {"value"}:
        return False, None
    decoder = _STDLIB_DECODERS.get(cls)
    if decoder is not None:
        return True, decoder(data["value"])
    if isinstance(cls, type) and issubclass(cls, enum.Enum):
        # The member value was encoded recursively (e.g. a tuple or date), so restore it before the lookup.
        return True, cls(deserialize_value(data["value"]))
    return False, None


def find_json_incompatible_value(value: Any, path: str = "$") -> str | None:
    """
    Locate the first value that ``json.dumps`` cannot encode in an already serialized structure.

    :param value: Output of ``serialize_value``.
    :param path: Path of ``value`` used in the result, e.g. ``"outputs.text"``.
    :returns: ``"<path>: <type name>"`` for the first incompatible value, or None if the structure is JSON compatible.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return None
    if isinstance(value, list):
        for index, item in enumerate(value):
            found = find_json_incompatible_value(item, f"{path}[{index}]")
            if found:
                return found
        return None
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, (str, int, float, bool)) and key is not None:
                return f"{path}: dict key of type {type(key).__name__}"
            found = find_json_incompatible_value(item, f"{path}.{key}")
            if found:
                return found
        return None
    return f"{path}: {type(value).__name__}"


def serialize_value(value: Any) -> Any:
    """
    Serialize a value for JSON transmission.

    Adds `__type__` metadata for complex objects to enable reconstruction.

    Supports (in order of priority):
    1. Objects with to_dict() method (Haystack convention)
    2. Pydantic models (.model_dump())
    3. Dataclasses (dataclasses.asdict())
    4. Lists/tuples (recursive, converted to list)
    5. Dicts (recursive)
    6. Standard library values (datetime, date, time, timedelta, Decimal, UUID, bytes, set, frozenset, Enum)
    7. Primitives (pass through)

    :param value: The value to serialize.
    :returns: JSON-serializable representation.
    """
    if value is None:
        return None

    # Primitives first. Subclasses (e.g. str/int Enums) are normalized to the plain JSON type: they are encoded as
    # their primitive value anyway, and later code must not depend on the subclass surviving the boundary.
    if isinstance(value, (str, int, float, bool)):
        if type(value) in (str, int, float, bool):
            return value
        for primitive in (bool, int, float, str):
            if isinstance(value, primitive):
                return primitive(value.value if isinstance(value, enum.Enum) else value)

    # 1. Haystack convention: to_dict()
    if hasattr(value, "to_dict") and callable(value.to_dict):
        serialized = value.to_dict()
        if isinstance(serialized, dict):
            serialized["__type__"] = generate_qualified_class_name(value.__class__)
        return serialized

    # 2. Pydantic models
    if hasattr(value, "model_dump") and callable(value.model_dump):
        serialized = value.model_dump()
        if isinstance(serialized, dict):
            serialized["__type__"] = generate_qualified_class_name(value.__class__)
        return serialized

    # 3. Dataclasses
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        serialized = {f.name: serialize_value(getattr(value, f.name)) for f in dataclasses.fields(value)}
        serialized["__type__"] = generate_qualified_class_name(value.__class__)
        return serialized

    # 4. Recursive for lists/tuples
    if isinstance(value, (list, tuple)):
        return [serialize_value(v) for v in value]

    # 5. Recursive for dicts
    if isinstance(value, dict):
        return {k: serialize_value(v) for k, v in value.items()}

    # 6. Standard library value types
    encoded = _encode_stdlib_value(value)
    if encoded is not None:
        return encoded

    # 7. Primitives (str, int, float, bool) and other JSON-serializable types
    return value


def deserialize_value(value: Any, type_hint: type | None = None) -> Any:
    """
    Deserialize a value from JSON.

    Deserialization strategy:
    1. If type_hint provided: use it to guide deserialization (for HTTP boundary)
    2. Else if __type__ metadata present: infer type from metadata (for internal calls)
    3. Else: return as-is (primitives) or recurse into containers

    :param value: The serialized value.
    :param type_hint: Optional explicit type hint to guide deserialization.
    :returns: Reconstructed Python object.
    """
    if value is None:
        return None

    # If type_hint provided, use type-aware deserialization
    # Note: inspect._empty is returned when function has no type annotation
    if (
        type_hint is not None
        and type_hint is not Any
        and type_hint is not inspect._empty  # pylint: disable=protected-access
    ):
        return _deserialize_with_type_hint(value, type_hint)

    # Try to infer from __type__ metadata
    if isinstance(value, dict) and "__type__" in value:
        return _deserialize_with_metadata(value)

    # Recurse into containers
    if isinstance(value, list):
        return [deserialize_value(v) for v in value]

    if isinstance(value, dict):
        return {k: deserialize_value(v) for k, v in value.items()}

    # Primitives pass through
    return value


def _deserialize_with_type_hint(value: Any, type_hint: type) -> Any:  # pylint: disable=too-many-return-statements
    """
    Deserialize using an explicit type hint.

    Used for both HTTP boundary (PipelineIOHandler) and internal communication
    (component sockets) where we know the expected type.

    :param value: The value to deserialize.
    :param type_hint: The expected type (can be generic like List[T], Optional[T], etc.).
    :returns: Deserialized value.
    :raises ValueError: If Union type is ambiguous and no __type__ metadata present.
    """
    # Handle None
    if value is None:
        return None

    # Get origin for generic types (list, dict, Union, etc.)
    origin = get_origin(type_hint)
    args = get_args(type_hint)

    # Handle List[T] / list[T]
    if origin in (list, List):
        if not isinstance(value, list):
            return value
        inner_type = args[0] if args else None
        if inner_type is None or inner_type is Any:
            return [deserialize_value(item) for item in value]
        return [_deserialize_with_type_hint(item, inner_type) for item in value]

    # Handle Dict[K, V] / dict[K, V]
    if origin in (dict, Dict):
        if not isinstance(value, dict):
            return value
        # If dict has __type__, it's a serialized object, use metadata
        if "__type__" in value:
            return _deserialize_with_metadata(value)
        # Otherwise, recursively deserialize values if we have a value type hint
        if len(args) >= 2:
            value_type = args[1]
            if value_type is not Any:
                return {k: _deserialize_with_type_hint(v, value_type) for k, v in value.items()}
        return {k: deserialize_value(v) for k, v in value.items()}

    # Handle Union types (including Optional[T] which is Union[T, None])
    if origin is Union or isinstance(type_hint, types.UnionType):
        # Get args for both typing.Union and types.UnionType (Python 3.10+ X | Y syntax)
        if not args:
            args = get_args(type_hint)

        # Handle Optional[T] (Union[T, None]) - extract the non-None type
        non_none_args = [a for a in args if a is not type(None)]
        if len(non_none_args) == 1:
            # This is Optional[X], use X
            return _deserialize_with_type_hint(value, non_none_args[0])

        # Multiple non-None types - try to match
        matched_type = _try_match_union_type(value, args)
        if matched_type is not None:
            if matched_type is type(None):
                return None
            return _deserialize_with_type_hint(value, matched_type)

        # Ambiguous Union - need __type__ metadata
        if isinstance(value, dict) and "__type__" in value:
            return _deserialize_with_metadata(value)

        # Cannot determine type - raise error
        raise ValueError(
            f"Cannot deserialize value for Union type {type_hint}: "
            f"value type '{type(value).__name__}' is ambiguous and no __type__ metadata present"
        )

    # For non-generic types, if value is a dict, try to reconstruct
    if isinstance(value, dict):
        # If has __type__, use that (it's more specific than the hint)
        if "__type__" in value:
            return _deserialize_with_metadata(value)
        # Otherwise, reconstruct using the type hint
        return _reconstruct_object(type_hint, value)

    # Value is already the right type or a primitive
    return value


def _deserialize_with_metadata(value: dict) -> Any:
    """
    Deserialize a dict using its __type__ metadata.

    Used for internal service communication where we control serialization
    and always add __type__ metadata.

    :param value: Dict with __type__ key containing fully qualified class name.
    :returns: Reconstructed object, or dict without __type__ if reconstruction fails.
    """
    type_name = value.get("__type__")
    if not type_name:
        return {k: deserialize_value(v) for k, v in value.items()}

    try:
        cls = import_class_by_name(type_name)
        # Standard library values keep their raw JSON payload; decoding it recursively would change it.
        handled, decoded = _decode_stdlib_value(cls, {k: v for k, v in value.items() if k != "__type__"})
        if handled:
            return decoded
        # Remove __type__ before reconstruction, recursively deserialize other values
        data = {k: deserialize_value(v) for k, v in value.items() if k != "__type__"}
        return _reconstruct_object(cls, data)
    except ImportError:
        # Failed to import class, return dict without __type__
        logger.warning(
            "Could not reconstruct object of type '%s', returning dict",
            type_name,
        )
        return {k: deserialize_value(v) for k, v in value.items() if k != "__type__"}


def _reconstruct_object(cls: type, data: dict[str, Any]) -> Any:
    """
    Reconstruct an object from its class and data dict.

    Tries multiple reconstruction strategies in order of preference:
    1. from_dict() class method (Haystack convention)
    2. model_validate() for Pydantic models
    3. Constructor for dataclasses
    4. Generic constructor fallback

    :param cls: The class to instantiate.
    :param data: The data dict (already recursively deserialized).
    :returns: Reconstructed object, or data dict if reconstruction fails.
    """
    # 1. Haystack convention: from_dict()
    if hasattr(cls, "from_dict") and callable(cls.from_dict):
        try:
            return cls.from_dict(data)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning(
                "from_dict() failed for %s: %s",
                cls.__name__,
                str(e),
            )

    # 2. Pydantic models
    if hasattr(cls, "model_validate") and callable(cls.model_validate):
        try:
            return cls.model_validate(data)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning(
                "model_validate() failed for %s: %s",
                cls.__name__,
                str(e),
            )

    # 3. Dataclasses
    if dataclasses.is_dataclass(cls):
        try:
            return cls(**data)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning(
                "Dataclass constructor failed for %s: %s",
                cls.__name__,
                str(e),
            )

    # 4. Generic constructor fallback
    try:
        return cls(**data)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning(
            "Constructor failed for %s: %s, returning dict: %s",
            cls.__name__,
            str(e),
            json.dumps(data),
        )
        return data


# =============================================================================
# Convenience wrappers for component inputs/outputs
# =============================================================================


def serialize_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """
    Serialize component inputs for service call.

    :param inputs: Component input dict.
    :returns: Serialized inputs safe for JSON transmission.
    """
    return {k: serialize_value(v) for k, v in inputs.items()}


def deserialize_inputs(
    inputs: dict[str, Any],
    type_hints: dict[str, type | types.UnionType] | None = None,
) -> dict[str, Any]:
    """
    Deserialize component inputs from service call.

    :param inputs: Serialized input dict.
    :param type_hints: Optional dict mapping input names to expected types.
    :returns: Deserialized inputs with reconstructed objects.
    """
    if type_hints:
        return {k: deserialize_value(v, type_hints.get(k)) for k, v in inputs.items()}  # type: ignore[arg-type]
    return {k: deserialize_value(v) for k, v in inputs.items()}


def serialize_outputs(outputs: dict[str, Any]) -> dict[str, Any]:
    """
    Serialize component outputs for service response.

    :param outputs: Component output dict.
    :returns: Serialized outputs safe for JSON transmission.
    """
    return {k: serialize_value(v) for k, v in outputs.items()}


def deserialize_outputs(
    outputs: dict[str, Any],
    type_hints: dict[str, type] | None = None,
) -> dict[str, Any]:
    """
    Deserialize component outputs from service response.

    :param outputs: Serialized output dict.
    :param type_hints: Optional dict mapping output names to expected types.
    :returns: Deserialized outputs with reconstructed objects.
    """
    if type_hints:
        return {k: deserialize_value(v, type_hints.get(k)) for k, v in outputs.items()}
    return {k: deserialize_value(v) for k, v in outputs.items()}
