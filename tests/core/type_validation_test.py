"""Tests for type validation utilities."""

import dataclasses
from typing import Any, Dict, List, Optional, Union

import pytest

from ravelin.core.type_validation import (
    _check_type,
    validate_types,
)


# Module-level test classes
@dataclasses.dataclass
class Config:
    name: str
    value: int


class TestCheckType:
    """Tests for _check_type function."""

    # Primitives
    def test_str(self):
        assert _check_type("hello", str) is True
        assert _check_type(123, str) is False

    def test_int(self):
        assert _check_type(42, int) is True
        assert _check_type("42", int) is False

    def test_float(self):
        assert _check_type(3.14, float) is True
        assert _check_type(42, float) is False  # int is not float

    def test_bool(self):
        assert _check_type(True, bool) is True
        assert _check_type(False, bool) is True
        assert _check_type(1, bool) is False  # int is not bool

    # None
    def test_none_with_nonetype(self):
        assert _check_type(None, type(None)) is True

    def test_none_with_optional(self):
        assert _check_type(None, Optional[str]) is True
        assert _check_type(None, Optional[int]) is True

    def test_none_with_non_optional(self):
        assert _check_type(None, str) is False
        assert _check_type(None, int) is False

    # List
    def test_list_basic(self):
        assert _check_type([1, 2, 3], list) is True
        assert _check_type([1, 2, 3], List) is True
        assert _check_type("not a list", list) is False

    def test_list_with_type_param(self):
        assert _check_type([1, 2, 3], List[int]) is True
        assert _check_type(["a", "b"], List[str]) is True
        assert _check_type([1, "mixed"], List[int]) is False
        assert _check_type([1, 2, 3], List[str]) is False

    def test_list_empty(self):
        assert _check_type([], List[int]) is True
        assert _check_type([], List[str]) is True

    def test_list_nested(self):
        assert _check_type([[1, 2], [3, 4]], List[List[int]]) is True
        assert _check_type([[1, 2], ["a"]], List[List[int]]) is False

    # Dict
    def test_dict_basic(self):
        assert _check_type({"a": 1}, dict) is True
        assert _check_type({"a": 1}, Dict) is True
        assert _check_type("not a dict", dict) is False

    def test_dict_with_type_params(self):
        assert _check_type({"a": 1, "b": 2}, Dict[str, int]) is True
        assert _check_type({"a": "x", "b": "y"}, Dict[str, str]) is True
        assert _check_type({"a": 1, "b": "mixed"}, Dict[str, int]) is False

    def test_dict_empty(self):
        assert _check_type({}, Dict[str, int]) is True

    # Optional
    def test_optional_with_value(self):
        assert _check_type("hello", Optional[str]) is True
        assert _check_type(42, Optional[int]) is True

    def test_optional_with_none(self):
        assert _check_type(None, Optional[str]) is True
        assert _check_type(None, Optional[int]) is True

    def test_optional_wrong_type(self):
        assert _check_type(123, Optional[str]) is False
        assert _check_type("hello", Optional[int]) is False

    # Union
    def test_union_matches_first(self):
        assert _check_type("hello", Union[str, int]) is True

    def test_union_matches_second(self):
        assert _check_type(42, Union[str, int]) is True

    def test_union_no_match(self):
        assert _check_type([1, 2], Union[str, int]) is False

    def test_union_with_list(self):
        assert _check_type([1, 2], Union[str, List[int]]) is True
        assert _check_type("hello", Union[str, List[int]]) is True

    # Complex objects
    def test_dataclass(self):
        obj = Config(name="test", value=42)
        assert _check_type(obj, Config) is True
        assert _check_type({"name": "test", "value": 42}, Config) is False

    # Any
    def test_any(self):
        assert _check_type("anything", Any) is True
        assert _check_type(123, Any) is True
        assert _check_type([1, 2, 3], Any) is True
        assert _check_type(None, Any) is True


class TestValidateTypes:
    """Tests for validate_types function."""

    def test_valid_primitives(self):
        data = {"name": "test", "count": 42}
        type_hints = {"name": str, "count": int}
        # Should not raise
        validate_types(data, type_hints)

    def test_invalid_type_raises(self):
        data = {"name": 123}  # Should be str
        type_hints = {"name": str}
        with pytest.raises(TypeError, match="Input 'name' expected str"):
            validate_types(data, type_hints)

    def test_missing_key_is_ok(self):
        data = {"name": "test"}
        type_hints = {"name": str, "count": int}  # count not in data
        # Should not raise - missing keys are allowed
        validate_types(data, type_hints)

    def test_extra_key_is_ok(self):
        data = {"name": "test", "extra": "value"}
        type_hints = {"name": str}  # extra not in hints
        # Should not raise - extra keys are ignored
        validate_types(data, type_hints)

    def test_list_type(self):
        data = {"items": [1, 2, 3]}
        type_hints = {"items": List[int]}
        validate_types(data, type_hints)

    def test_list_type_invalid(self):
        data = {"items": [1, "two", 3]}
        type_hints = {"items": List[int]}
        with pytest.raises(TypeError, match="Input 'items' expected"):
            validate_types(data, type_hints)

    def test_optional_with_value(self):
        data = {"name": "test"}
        type_hints = {"name": Optional[str]}
        validate_types(data, type_hints)

    def test_optional_with_none(self):
        data = {"name": None}
        type_hints = {"name": Optional[str]}
        validate_types(data, type_hints)

    def test_complex_nested(self):
        data = {
            "configs": [
                Config(name="a", value=1),
                Config(name="b", value=2),
            ]
        }
        type_hints = {"configs": List[Config]}
        validate_types(data, type_hints)

    def test_complex_nested_invalid(self):
        data = {
            "configs": [
                {"name": "a", "value": 1},  # dict, not Config
            ]
        }
        type_hints = {"configs": List[Config]}
        with pytest.raises(TypeError):
            validate_types(data, type_hints)


class TestPython310UnionSyntax:
    """Tests for Python 3.10+ union syntax (X | Y)."""

    def test_union_pipe_syntax(self):
        # Python 3.10+ allows str | int instead of Union[str, int]
        assert _check_type("hello", str | int) is True
        assert _check_type(42, str | int) is True
        assert _check_type([1, 2], str | int) is False

    def test_optional_pipe_syntax(self):
        # str | None is equivalent to Optional[str]
        assert _check_type("hello", str | None) is True
        assert _check_type(None, str | None) is True
        assert _check_type(123, str | None) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
