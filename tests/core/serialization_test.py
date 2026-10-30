"""Tests for serialization utilities."""

import dataclasses
import datetime
import decimal
import enum
import json
import uuid
from typing import List, Optional, Union

import pytest
from haystack.core.serialization import generate_qualified_class_name, import_class_by_name

from ravelin.core.serialization import (
    _try_match_union_type,
    deserialize_inputs,
    deserialize_outputs,
    deserialize_value,
    find_json_incompatible_value,
    serialize_inputs,
    serialize_outputs,
    serialize_value,
)


# Module-level classes for roundtrip tests (can be imported by name)
@dataclasses.dataclass
class RoundtripConfig:
    """Test dataclass at module level for roundtrip testing."""

    name: str
    value: int


@dataclasses.dataclass
class AnotherConfig:
    """Another test dataclass for Union type testing."""

    label: str
    count: int


class TestGenerateQualifiedClassName:
    def test_builtin_type(self):
        assert generate_qualified_class_name(str) == "builtins.str"

    def test_custom_class(self):
        class MyClass:
            pass

        result = generate_qualified_class_name(MyClass)
        assert "MyClass" in result


class TestImportClassByName:
    """Tests for Haystack's import_class_by_name (used by our serialization)."""

    def test_import_builtin(self):
        cls = import_class_by_name("builtins.str")
        assert cls is str

    def test_import_stdlib(self):
        cls = import_class_by_name("collections.OrderedDict")
        from collections import OrderedDict

        assert cls is OrderedDict

    def test_import_nonexistent_module_raises(self):
        # Haystack's import_class_by_name raises ImportError on failure
        with pytest.raises(ImportError):
            import_class_by_name("nonexistent.module.Class")

    def test_import_nonexistent_class_raises(self):
        with pytest.raises(ImportError):
            import_class_by_name("builtins.NonexistentClass")

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError):
            import_class_by_name("no_dots_here")


class TestSerializeValue:
    def test_none(self):
        assert serialize_value(None) is None

    def test_primitives(self):
        assert serialize_value("hello") == "hello"
        assert serialize_value(42) == 42
        assert serialize_value(3.14) == 3.14
        assert serialize_value(True) is True

    def test_list(self):
        result = serialize_value([1, 2, 3])
        assert result == [1, 2, 3]

    def test_nested_list(self):
        result = serialize_value([[1, 2], [3, 4]])
        assert result == [[1, 2], [3, 4]]

    def test_dict(self):
        result = serialize_value({"a": 1, "b": 2})
        assert result == {"a": 1, "b": 2}

    def test_nested_dict(self):
        result = serialize_value({"outer": {"inner": 42}})
        assert result == {"outer": {"inner": 42}}

    def test_object_with_to_dict(self):
        class MyObject:
            def to_dict(self):
                return {"value": 123}

        obj = MyObject()
        result = serialize_value(obj)
        assert result["value"] == 123
        assert "__type__" in result
        assert "MyObject" in result["__type__"]

    def test_dataclass(self):
        @dataclasses.dataclass
        class Point:
            x: int
            y: int

        point = Point(x=10, y=20)
        result = serialize_value(point)
        assert result["x"] == 10
        assert result["y"] == 20
        assert "__type__" in result
        assert "Point" in result["__type__"]


class TestDeserializeValue:
    def test_none(self):
        assert deserialize_value(None) is None

    def test_primitives(self):
        assert deserialize_value("hello") == "hello"
        assert deserialize_value(42) == 42
        assert deserialize_value(3.14) == 3.14
        assert deserialize_value(True) is True

    def test_list(self):
        result = deserialize_value([1, 2, 3])
        assert result == [1, 2, 3]

    def test_dict_without_type(self):
        result = deserialize_value({"a": 1, "b": 2})
        assert result == {"a": 1, "b": 2}

    def test_nested_structures(self):
        result = deserialize_value({"items": [{"a": 1}, {"b": 2}]})
        assert result == {"items": [{"a": 1}, {"b": 2}]}


class TestRoundTrip:
    def test_primitives_roundtrip(self):
        values = ["hello", 42, 3.14, True, None]
        for v in values:
            assert deserialize_value(serialize_value(v)) == v

    def test_list_roundtrip(self):
        original = [1, "two", 3.0, [4, 5]]
        result = deserialize_value(serialize_value(original))
        assert result == original

    def test_dict_roundtrip(self):
        original = {"a": 1, "b": {"c": 2}}
        result = deserialize_value(serialize_value(original))
        assert result == original

    def test_dataclass_roundtrip(self):
        # Uses module-level RoundtripConfig class (can be imported by name)
        original = RoundtripConfig(name="test", value=42)
        serialized = serialize_value(original)
        result = deserialize_value(serialized)

        assert isinstance(result, RoundtripConfig)
        assert result.name == "test"
        assert result.value == 42

    def test_local_class_returns_dict(self):
        # Local classes can't be imported, so deserialization returns dict
        @dataclasses.dataclass
        class LocalConfig:
            x: int

        original = LocalConfig(x=99)
        serialized = serialize_value(original)
        result = deserialize_value(serialized)

        # Can't import local class, so returns dict without __type__
        assert isinstance(result, dict)
        assert result["x"] == 99


class TestSerializeInputsOutputs:
    def test_serialize_inputs(self):
        inputs = {"text": "hello", "count": 5}
        result = serialize_inputs(inputs)
        assert result == {"text": "hello", "count": 5}

    def test_deserialize_inputs(self):
        inputs = {"text": "hello", "count": 5}
        result = deserialize_inputs(inputs)
        assert result == {"text": "hello", "count": 5}

    def test_serialize_outputs(self):
        outputs = {"result": "done", "score": 0.95}
        result = serialize_outputs(outputs)
        assert result == {"result": "done", "score": 0.95}

    def test_deserialize_outputs(self):
        outputs = {"result": "done", "score": 0.95}
        result = deserialize_outputs(outputs)
        assert result == {"result": "done", "score": 0.95}


class TestPydanticSupport:
    def test_pydantic_model_serialize(self):
        pytest.importorskip("pydantic")
        from pydantic import BaseModel

        class User(BaseModel):
            name: str
            age: int

        user = User(name="Alice", age=30)
        result = serialize_value(user)

        assert result["name"] == "Alice"
        assert result["age"] == 30
        assert "__type__" in result

    def test_pydantic_local_class_returns_dict(self):
        # Local pydantic classes can't be imported, so deserialization returns dict
        pytest.importorskip("pydantic")
        from pydantic import BaseModel

        class LocalItem(BaseModel):
            id: int
            label: str

        original = LocalItem(id=1, label="test")
        serialized = serialize_value(original)
        result = deserialize_value(serialized)

        # Can't import local class, so returns dict
        assert isinstance(result, dict)
        assert result["id"] == 1
        assert result["label"] == "test"


class TestTypeHintDeserialization:
    """Tests for type-hint guided deserialization."""

    def test_deserialize_with_list_type_hint(self):
        # Module-level class so it can be imported
        data = [{"name": "test1", "value": 1}, {"name": "test2", "value": 2}]
        result = deserialize_value(data, list[RoundtripConfig])

        assert len(result) == 2
        assert isinstance(result[0], RoundtripConfig)
        assert result[0].name == "test1"
        assert result[1].value == 2

    def test_deserialize_with_single_type_hint(self):
        data = {"name": "test", "value": 42}
        result = deserialize_value(data, RoundtripConfig)

        assert isinstance(result, RoundtripConfig)
        assert result.name == "test"
        assert result.value == 42

    def test_deserialize_primitives_ignores_type_hint(self):
        # Primitives should pass through
        assert deserialize_value("hello", str) == "hello"
        assert deserialize_value(42, int) == 42

    def test_deserialize_none_with_type_hint(self):
        assert deserialize_value(None, RoundtripConfig) is None

    def test_deserialize_optional_type_hint(self):
        from typing import Optional

        data = {"name": "test", "value": 42}
        result = deserialize_value(data, Optional[RoundtripConfig])

        assert isinstance(result, RoundtripConfig)
        assert result.name == "test"

    def test_deserialize_dict_value_type_hint(self):
        data = {
            "item1": {"name": "a", "value": 1},
            "item2": {"name": "b", "value": 2},
        }
        result = deserialize_value(data, dict[str, RoundtripConfig])

        assert isinstance(result["item1"], RoundtripConfig)
        assert result["item1"].name == "a"
        assert result["item2"].value == 2

    def test_type_hint_with_metadata_prefers_metadata(self):
        # When __type__ is present, it should be used even with type_hint
        from haystack.core.serialization import generate_qualified_class_name

        data = {
            "name": "test",
            "value": 99,
            "__type__": generate_qualified_class_name(RoundtripConfig),
        }
        # Pass a different type hint - __type__ should win
        result = deserialize_value(data, dict)  # type_hint says dict

        assert isinstance(result, RoundtripConfig)
        assert result.name == "test"

    def test_deserialize_inputs_with_type_hints(self):
        inputs = {
            "config": {"name": "test", "value": 42},
            "count": 5,
        }
        type_hints = {"config": RoundtripConfig}
        result = deserialize_inputs(inputs, type_hints)

        assert isinstance(result["config"], RoundtripConfig)
        assert result["config"].name == "test"
        assert result["count"] == 5


class TestUnionTypeMatching:
    """Tests for Union type matching utilities."""

    def test_match_none_in_union(self):
        result = _try_match_union_type(None, (str, type(None)))
        assert result is type(None)

    def test_match_str_in_union(self):
        result = _try_match_union_type("hello", (str, int))
        assert result is str

    def test_match_int_in_union(self):
        result = _try_match_union_type(42, (str, int))
        assert result is int

    def test_match_list_in_union(self):
        result = _try_match_union_type([1, 2, 3], (str, List[int]))
        assert result == List[int]

    def test_match_dict_without_type_in_union(self):
        result = _try_match_union_type({"a": 1}, (str, dict[str, int]))
        assert result == dict[str, int]

    def test_dict_with_type_returns_none(self):
        # dict with __type__ should return None (use metadata instead)
        result = _try_match_union_type({"__type__": "some.Class", "value": 1}, (str, RoundtripConfig))
        assert result is None

    def test_ambiguous_union_returns_none(self):
        # Elements match both list types (bool is accepted by neither int nor str here, but mixed lists are)
        result = _try_match_union_type([1, "a"], (List[int], List[str]))
        assert result is None

    def test_list_types_narrowed_by_primitive_elements(self):
        assert _try_match_union_type([1, 2], (List[int], List[str])) == List[int]
        assert _try_match_union_type(["a"], (List[str], List[RoundtripConfig])) == List[str]

    def test_list_types_narrowed_by_object_elements(self):
        value = [serialize_value(RoundtripConfig(name="a", value=1))]
        assert _try_match_union_type(value, (List[str], List[RoundtripConfig])) == List[RoundtripConfig]
        assert _try_match_union_type(value, (List[AnotherConfig], List[RoundtripConfig])) == List[RoundtripConfig]

    def test_empty_list_matches_first_candidate(self):
        assert _try_match_union_type([], (List[str], List[RoundtripConfig])) == List[str]

    def test_list_types_with_untyped_object_elements_stay_ambiguous(self):
        # Plain dicts cannot be told apart from serialized objects without __type__
        assert _try_match_union_type([{"name": "a"}], (List[str], List[RoundtripConfig])) is None

    def test_dict_types_narrowed_by_values(self):
        assert _try_match_union_type({"k": 1}, (dict[str, int], dict[str, str])) == dict[str, int]

    def test_list_of_primitives_vs_list_of_dicts(self):
        hint = (List[str], List[dict[str, int]])
        assert _try_match_union_type(["a"], hint) == List[str]
        assert _try_match_union_type([{"k": 1}], hint) == List[dict[str, int]]

    def test_nested_containers_narrowed_by_inner_types(self):
        hint = (List[dict[str, int]], List[dict[str, str]])
        assert _try_match_union_type([{"k": "v"}], hint) == List[dict[str, str]]
        assert _try_match_union_type([{"k": 1}], hint) == List[dict[str, int]]

    def test_deeply_nested_containers(self):
        hint = (List[List[int]], List[List[str]])
        assert _try_match_union_type([[1, 2], []], hint) == List[List[int]]
        assert _try_match_union_type([["a"]], hint) == List[List[str]]

    def test_dict_of_lists_narrowed_by_inner_elements(self):
        hint = (dict[str, List[int]], dict[str, List[RoundtripConfig]])
        configs = {"k": [serialize_value(RoundtripConfig(name="a", value=1))]}
        assert _try_match_union_type({"k": [1]}, hint) == dict[str, List[int]]
        assert _try_match_union_type(configs, hint) == dict[str, List[RoundtripConfig]]

    def test_nested_containers_with_only_empty_inner_values_stay_ambiguous(self):
        # Both candidates accept [{}]: the inner dict is empty, so nothing distinguishes them.
        assert _try_match_union_type([{}], (List[dict[str, int]], List[dict[str, str]])) is None

    def test_deserialize_nested_union_of_list_types(self):
        hint = Union[List[dict[str, int]], List[dict[str, str]]]

        assert deserialize_value([{"k": "v"}], hint) == [{"k": "v"}]

    def test_bool_elements_do_not_match_int(self):
        assert _try_match_union_type([True], (List[int], List[bool])) == List[bool]


class TestUnionDeserialization:
    """Tests for Union type deserialization."""

    def test_deserialize_union_with_str(self):
        result = deserialize_value("hello", Union[str, int])
        assert result == "hello"

    def test_deserialize_union_with_int(self):
        result = deserialize_value(42, Union[str, int])
        assert result == 42

    def test_deserialize_union_with_list(self):
        result = deserialize_value([1, 2, 3], Union[str, List[int]])
        assert result == [1, 2, 3]

    def test_deserialize_union_with_object_using_metadata(self):
        # Object in Union uses __type__ metadata
        serialized = serialize_value(RoundtripConfig(name="test", value=42))
        result = deserialize_value(serialized, Union[str, RoundtripConfig])

        assert isinstance(result, RoundtripConfig)
        assert result.name == "test"

    def test_deserialize_union_ambiguous_raises(self):
        # dict without __type__ when Union has object types is ambiguous
        with pytest.raises(ValueError, match="ambiguous"):
            deserialize_value({"name": "test", "value": 42}, Union[RoundtripConfig, AnotherConfig])

    def test_deserialize_optional_with_value(self):
        data = {"name": "test", "value": 42}
        result = deserialize_value(data, Optional[RoundtripConfig])

        assert isinstance(result, RoundtripConfig)
        assert result.name == "test"

    def test_deserialize_optional_with_none(self):
        result = deserialize_value(None, Optional[RoundtripConfig])
        assert result is None

    def test_deserialize_union_str_or_list(self):
        # str value
        assert deserialize_value("hello", Union[str, List[str]]) == "hello"
        # list value
        assert deserialize_value(["a", "b"], Union[str, List[str]]) == ["a", "b"]

    def test_deserialize_union_of_list_types(self):
        hint = Union[List[str], List[RoundtripConfig]]

        assert deserialize_value(["https://a"], hint) == ["https://a"]
        configs = serialize_value([RoundtripConfig(name="a", value=1)])
        assert deserialize_value(configs, hint) == [RoundtripConfig(name="a", value=1)]


class TestNestedStructures:
    """Tests for complex nested structures."""

    def test_serialize_mixed_dict(self):
        """Test serializing dict with object and primitive values."""
        data = {
            "config": RoundtripConfig(name="test", value=42),
            "items": [RoundtripConfig(name="a", value=1), RoundtripConfig(name="b", value=2)],
            "label": "simple string",
        }
        result = serialize_value(data)

        assert result["label"] == "simple string"
        assert "__type__" in result["config"]
        assert result["config"]["name"] == "test"
        assert len(result["items"]) == 2
        assert "__type__" in result["items"][0]

    def test_deserialize_mixed_dict_with_metadata(self):
        """Test deserializing dict with __type__ metadata on nested objects."""
        data = {
            "config": RoundtripConfig(name="test", value=42),
            "items": [RoundtripConfig(name="a", value=1), RoundtripConfig(name="b", value=2)],
            "label": "simple string",
        }
        serialized = serialize_value(data)
        result = deserialize_value(serialized)

        assert result["label"] == "simple string"
        assert isinstance(result["config"], RoundtripConfig)
        assert result["config"].name == "test"
        assert len(result["items"]) == 2
        assert isinstance(result["items"][0], RoundtripConfig)

    def test_roundtrip_nested_dict_with_objects(self):
        """Full roundtrip test for nested structures."""
        original = {
            "a": RoundtripConfig(name="first", value=1),
            "b": [RoundtripConfig(name="second", value=2), "plain string"],
            "c": {"nested": RoundtripConfig(name="third", value=3)},
        }
        serialized = serialize_value(original)
        result = deserialize_value(serialized)

        assert isinstance(result["a"], RoundtripConfig)
        assert result["a"].name == "first"
        assert isinstance(result["b"][0], RoundtripConfig)
        assert result["b"][1] == "plain string"
        assert isinstance(result["c"]["nested"], RoundtripConfig)

    def test_deserialize_nested_with_type_hints(self):
        """Test deserialization with type hints for nested structures."""
        data = {
            "items": [
                {"name": "a", "value": 1},
                {"name": "b", "value": 2},
            ]
        }
        # With type hint, we can deserialize without __type__
        result = deserialize_value(data, dict[str, List[RoundtripConfig]])

        assert len(result["items"]) == 2
        assert isinstance(result["items"][0], RoundtripConfig)
        assert result["items"][0].name == "a"

    def test_mixed_list_with_primitives_and_objects(self):
        """Test list containing both primitives and objects."""
        original = [
            RoundtripConfig(name="obj", value=1),
            "string",
            42,
            {"plain": "dict"},
        ]
        serialized = serialize_value(original)
        result = deserialize_value(serialized)

        assert isinstance(result[0], RoundtripConfig)
        assert result[1] == "string"
        assert result[2] == 42
        assert result[3] == {"plain": "dict"}


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_list(self):
        assert serialize_value([]) == []
        assert deserialize_value([]) == []

    def test_empty_dict(self):
        assert serialize_value({}) == {}
        assert deserialize_value({}) == {}

    def test_deeply_nested_structure(self):
        original = {"l1": {"l2": {"l3": {"l4": RoundtripConfig(name="deep", value=99)}}}}
        serialized = serialize_value(original)
        result = deserialize_value(serialized)

        assert isinstance(result["l1"]["l2"]["l3"]["l4"], RoundtripConfig)
        assert result["l1"]["l2"]["l3"]["l4"].name == "deep"

    def test_list_of_lists(self):
        original = [[1, 2], [3, 4], [5, 6]]
        result = deserialize_value(serialize_value(original))
        assert result == original

    def test_bool_not_confused_with_int(self):
        # In Python, bool is subclass of int - make sure we handle correctly
        assert serialize_value(True) is True
        assert serialize_value(False) is False
        assert deserialize_value(True) is True
        assert deserialize_value(False) is False

    def test_float_precision(self):
        original = 3.141592653589793
        result = deserialize_value(serialize_value(original))
        assert result == original

    def test_unicode_strings(self):
        original = {"emoji": "🎉", "chinese": "你好", "arabic": "مرحبا"}
        result = deserialize_value(serialize_value(original))
        assert result == original


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class Shade(enum.Enum):
    """Module-level enum for roundtrip tests."""

    DARK = "dark"


class Layout(enum.Enum):
    """Module-level enum whose member values do not survive JSON as-is."""

    GRID = (2, 3)
    SINCE = datetime.date(2020, 1, 1)


class TestStandardLibraryValues:
    """Values without to_dict() that components commonly return must survive the JSON boundary."""

    @pytest.mark.parametrize(
        "value",
        [
            datetime.datetime(2026, 10, 27, 12, 30, tzinfo=datetime.timezone.utc),
            datetime.datetime(2026, 10, 27, 12, 30),
            datetime.date(2026, 1, 2),
            datetime.time(8, 15),
            datetime.timedelta(seconds=90.5),
            decimal.Decimal("1.10"),
            uuid.UUID("12345678-1234-5678-1234-567812345678"),
            b"\x00\xff",
            {1, 2},
            frozenset({"a"}),
            Shade.DARK,
        ],
    )
    def test_roundtrip_through_json(self, value):
        encoded = json.loads(json.dumps(serialize_value(value)))

        restored = deserialize_value(encoded)

        assert restored == value
        assert type(restored) is type(value)

    def test_nested_values_roundtrip(self):
        value = {"items": [{"when": datetime.date(2020, 1, 1), "amount": decimal.Decimal("3")}]}

        assert deserialize_value(json.loads(json.dumps(serialize_value(value)))) == value

    def test_encoded_type_takes_precedence_over_type_hint(self):
        # The hint names a different type that could be built from the payload; the encoded type must win.
        encoded = serialize_value(datetime.date(2020, 1, 1))

        restored = deserialize_value(encoded, str)

        assert type(restored) is datetime.date
        assert restored == datetime.date(2020, 1, 1)

    def test_str_enum_becomes_a_plain_string(self):
        class Mode(str, enum.Enum):
            FAST = "fast"

        serialized = serialize_value(Mode.FAST)

        assert type(serialized) is str
        assert serialized == "fast"

    def test_int_enum_becomes_a_plain_int(self):
        class Level(enum.IntEnum):
            HIGH = 3

        serialized = serialize_value(Level.HIGH)

        assert type(serialized) is int
        assert serialized == 3

    def test_set_of_tuples_roundtrip(self):
        value = {(1, 2), (3, (4, 5))}

        restored = deserialize_value(json.loads(json.dumps(serialize_value(value))))

        assert restored == value

    def test_set_encoding_is_order_independent(self):
        # Iteration order is stable within one process but varies across hash seeds; the encoding must be canonical.
        assert serialize_value({"z", "a", "m"})["value"] == ["a", "m", "z"]

    @pytest.mark.parametrize("member", list(Layout))
    def test_enum_with_compound_values_roundtrip(self, member):
        restored = deserialize_value(json.loads(json.dumps(serialize_value(member))))

        assert restored is member


class TestFindJsonIncompatibleValue:
    def test_compatible_structure(self):
        assert find_json_incompatible_value({"a": [1, "b", None, {"c": True}]}) is None

    def test_reports_path_of_first_incompatible_value(self):
        class Opaque:
            pass

        result = find_json_incompatible_value(serialize_value({"x": [1, {"y": Opaque()}]}), "outputs")

        assert result == "outputs.x[1].y: Opaque"

    def test_reports_non_string_keys(self):
        assert find_json_incompatible_value({(1, 2): "v"}) == "$: dict key of type tuple"
