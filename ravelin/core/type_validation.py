"""
Runtime type validation utilities.

This module provides runtime type checking for pipeline inputs and component inputs.
Python doesn't enforce type hints at runtime, so this module fills that gap to catch
type errors early (as TerminalError) instead of letting them fail inside component.run().

Usage:
    from ravelin.core.type_validation import validate_types

    # Validate data against type hints
    validate_types(data, type_hints)  # Raises TypeError if mismatch
"""

import types
from typing import Any, Dict, List, Union, get_origin, get_args

from haystack.core.type_utils import _type_name


def validate_types(
    data: dict[str, Any],
    type_hints: dict[str, type | types.UnionType],
) -> None:
    """
    Validate that data values match their expected types.

    :param data: Dict of values to validate.
    :param type_hints: Dict mapping keys to expected types.
    :raises TypeError: If any value doesn't match its expected type.
    """
    for name, expected_type in type_hints.items():
        if name not in data:
            continue
        value = data[name]
        if value is None:
            # None is allowed for Optional types, handled in _check_type
            # For non-Optional, _check_type will catch it
            pass
        if not _check_type(value, expected_type):
            raise TypeError(
                f"Input '{name}' expected {_type_name(expected_type)}, "
                f"got {type(value).__name__}: {_value_repr(value)}"
            )


def _check_type(value: Any, expected: type | types.UnionType) -> bool:  # pylint: disable=too-many-return-statements
    """
    Check if value matches expected type.

    Handles:
    - Primitives (str, int, float, bool)
    - None / NoneType
    - List[T] - checks all elements
    - Dict[K, V] - checks all values
    - Optional[T] (Union[T, None])
    - Union[A, B, ...] - value must match at least one type
    - Complex objects - isinstance check

    :param value: The value to check.
    :param expected: The expected type (can be generic).
    :returns: True if value matches expected type.
    """
    # Handle Any first - accepts everything including None
    if expected is Any:
        return True

    # Handle None
    if value is None:
        # Check if None is allowed (Optional or Union with None)
        origin = get_origin(expected)
        if origin is Union or isinstance(expected, types.UnionType):
            args = get_args(expected)
            return type(None) in args
        return expected is type(None)

    # Get origin for generic types
    origin = get_origin(expected)
    args = get_args(expected)

    # Handle Union types (including Optional[T] which is Union[T, None])
    if origin is Union or isinstance(expected, types.UnionType):
        if not args:
            args = get_args(expected)
        # Value must match at least one of the union types
        return any(_check_type(value, arg) for arg in args)

    # Handle List[T]
    if origin in (list, List):
        if not isinstance(value, list):
            return False
        if not args or args[0] is Any:
            return True  # List without type param or Any element type
        return all(_check_type(item, args[0]) for item in value)

    # Handle Dict[K, V]
    if origin in (dict, Dict):
        if not isinstance(value, dict):
            return False
        if len(args) < 2 or args[1] is Any:
            return True  # Dict without type params or Any value type
        return all(_check_type(v, args[1]) for v in value.values())

    # For non-generic types, use isinstance
    # origin is set for generic aliases (e.g., list[int]), otherwise use expected directly
    check_against = origin if origin is not None else expected
    try:
        return isinstance(value, check_against)
    except TypeError:
        # Some types don't support isinstance (e.g., special forms)
        return True  # Be permissive for unknown types


def _value_repr(value: Any, max_len: int = 50) -> str:
    """Get a truncated string representation of a value."""
    repr_str = repr(value)
    if len(repr_str) > max_len:
        return repr_str[:max_len] + "..."
    return repr_str
