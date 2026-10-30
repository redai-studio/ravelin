"""Tests for serde utilities."""

import json
from dataclasses import InitVar, dataclass, field
from typing import Dict, List, Optional, Union

import pytest

from ravelin.core.restate.serde import to_dict_encoder, DictProtocolSerde
from restate.serde import DefaultSerde
from dacite.exceptions import MissingValueError, WrongTypeError


@dataclass
class ResponsePayload:
    response: str
    tags: list[str] = field(default_factory=list)


@dataclass
class CustomPayload:
    response: str

    def to_dict(self):
        return {"answer": self.response}

    @classmethod
    def from_dict(cls, data):
        return cls(response=data["answer"])


@dataclass
class ExtendedCustomPayload(CustomPayload):
    pass


@dataclass
class NestedWrapper:
    child: ResponsePayload
    items: list[dict[str, CustomPayload]]

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


@dataclass
class CheckedPayload:
    response: str

    @classmethod
    def from_dict(cls, data):
        if data["response"] != "allowed":
            raise ValueError("response denied")
        return cls(**data)


@dataclass
class UnionEnvelope:
    decision: str | CheckedPayload


@pytest.mark.parametrize("serde_cls", [DefaultSerde, DictProtocolSerde])
class TestRecursiveDecoderContracts:
    def test_custom_wrapper_reconstructs_nested_metadata(self, serde_cls):
        serde = serde_cls(NestedWrapper)
        original = NestedWrapper(ResponsePayload("yes"), [{"custom": CustomPayload("nested")}])
        result = serde.deserialize(serde.serialize(original))
        assert result == original
        assert isinstance(result.child, ResponsePayload)
        assert isinstance(result.items[0]["custom"], CustomPayload)

    def test_nested_metadata_decoder_failure_propagates(self, serde_cls):
        raw = {
            "__type__": f"{__name__}.NestedWrapper",
            "child": {"__type__": f"{__name__}.CheckedPayload", "response": "denied"},
            "items": [],
        }
        with pytest.raises(ValueError, match="response denied"):
            serde_cls(NestedWrapper).deserialize(json.dumps(raw).encode())

    def test_plain_custom_wire_dictionary_is_left_to_the_hook(self, serde_cls):
        child = {"__type__": f"{__name__}.ResponsePayload", "response": "yes"}
        raw = {"child": child, "items": []}
        result = serde_cls(NestedWrapper).deserialize(json.dumps(raw).encode())
        assert result.child == child

    def test_unknown_nested_metadata_keeps_known_children(self, serde_cls):
        raw = {
            "__type__": f"{__name__}.NestedWrapper",
            "child": {
                "__type__": "unknown_module.MissingClass",
                "known": {"__type__": f"{__name__}.ResponsePayload", "response": "yes"},
            },
            "items": [],
        }
        result = serde_cls(NestedWrapper).deserialize(json.dumps(raw).encode())
        assert result.child == {"known": ResponsePayload("yes")}

    @pytest.mark.parametrize("hint", [str | CheckedPayload, Union[str, CheckedPayload], UnionEnvelope])
    def test_union_cannot_bypass_custom_validation(self, serde_cls, hint):
        raw = {"__type__": f"{__name__}.CheckedPayload", "response": "denied"}
        if hint is UnionEnvelope:
            raw = {"decision": raw}
        with pytest.raises(ValueError, match="response denied"):
            serde_cls(hint).deserialize(json.dumps(raw).encode())

    def test_union_reconstructs_valid_custom_payload(self, serde_cls):
        serde = serde_cls(UnionEnvelope)
        original = UnionEnvelope(CheckedPayload("allowed"))
        assert serde.deserialize(serde.serialize(original)) == original

    def test_union_rejects_unrelated_metadata(self, serde_cls):
        raw = serde_cls(ResponsePayload).serialize(ResponsePayload("yes"))
        with pytest.raises(TypeError, match="Union"):
            serde_cls(str | CheckedPayload).deserialize(raw)

    @pytest.mark.parametrize(
        "hint, data, expected",
        [
            (str | CheckedPayload, "text", "text"),
            (str | list[CheckedPayload], [{"response": "allowed"}], [CheckedPayload("allowed")]),
            (str | dict[str, CheckedPayload], {"item": {"response": "allowed"}}, {"item": CheckedPayload("allowed")}),
        ],
    )
    def test_union_container_members_use_the_typed_decoder(self, serde_cls, hint, data, expected):
        assert serde_cls(hint).deserialize(json.dumps(data).encode()) == expected

    @pytest.mark.parametrize(
        "hint, data",
        [
            (str | list[CheckedPayload], [{"response": "denied"}]),
            (str | dict[str, CheckedPayload], {"item": {"response": "denied"}}),
        ],
    )
    def test_union_container_members_preserve_decoder_errors(self, serde_cls, hint, data):
        with pytest.raises(ValueError, match="response denied"):
            serde_cls(hint).deserialize(json.dumps(data).encode())

    def test_ambiguous_union_requires_metadata(self, serde_cls):
        with pytest.raises(ValueError, match="ambiguous"):
            serde_cls(ResponsePayload | CheckedPayload).deserialize(b'{"response":"allowed"}')


@pytest.mark.parametrize("serde_cls", [DefaultSerde, DictProtocolSerde])
class TestTypedDataclassSerde:
    """Exercise real JSON decoding through both public serde entry points."""

    @pytest.mark.parametrize("metadata", [{}, {"__type__": f"{__name__}.ResponsePayload"}])
    def test_additive_fields_and_defaults(self, serde_cls, metadata):
        data = {"response": "yes", "future": {"__type__": "must.not.import"}, **metadata}
        assert serde_cls(ResponsePayload).deserialize(json.dumps(data).encode()) == ResponsePayload("yes")

    @pytest.mark.parametrize("data", [{}, {"future": 1}])
    def test_missing_required_field_raises(self, serde_cls, data):
        with pytest.raises(MissingValueError, match="response"):
            serde_cls(ResponsePayload).deserialize(json.dumps(data).encode())

    @pytest.mark.parametrize("value", [None, False, 1, [], {}])
    def test_wrong_field_type_raises(self, serde_cls, value):
        with pytest.raises(WrongTypeError, match="response"):
            serde_cls(ResponsePayload).deserialize(json.dumps({"response": value}).encode())

    @pytest.mark.parametrize("raw", [b'"yes"', b"[]", b"true"])
    def test_wrong_top_level_shape_raises(self, serde_cls, raw):
        with pytest.raises(TypeError, match="Expected an object"):
            serde_cls(ResponsePayload).deserialize(raw)

    def test_nested_dataclasses_and_protocols(self, serde_cls):
        from ravelin.dataclasses import ChatMessage

        @dataclass
        class Envelope:
            response: ResponsePayload
            custom: CustomPayload
            messages: list[ChatMessage]

        original = Envelope(ResponsePayload("yes"), CustomPayload("custom"), [ChatMessage.from_user("hello")])
        serde = serde_cls(Envelope)
        assert serde.deserialize(serde.serialize(original)) == original

    @pytest.mark.parametrize("hint", [list[ResponsePayload], dict[str, ResponsePayload], Optional[ResponsePayload]])
    def test_container_type_hints(self, serde_cls, hint):
        data = {"response": "yes", "extra": 1}
        if hint == list[ResponsePayload]:
            data, expected = [data], [ResponsePayload("yes")]
        elif hint == dict[str, ResponsePayload]:
            data, expected = {"item": data}, {"item": ResponsePayload("yes")}
        else:
            expected = ResponsePayload("yes")
        assert serde_cls(hint).deserialize(json.dumps(data).encode()) == expected

    def test_custom_wire_format_has_precedence(self, serde_cls):
        serde = serde_cls(CustomPayload)
        raw = serde.serialize(CustomPayload("yes"))
        assert json.loads(raw)["answer"] == "yes"
        assert "response" not in json.loads(raw)
        assert serde.deserialize(raw) == CustomPayload("yes")

    def test_custom_decoder_failure_cannot_fall_back_to_constructor(self, serde_cls):
        with pytest.raises(KeyError, match="answer"):
            serde_cls(CustomPayload).deserialize(b'{"response":"constructor would accept this"}')

    def test_metadata_preserves_custom_subclasses(self, serde_cls):
        serde = serde_cls(CustomPayload)
        value = serde.deserialize(serde.serialize(ExtendedCustomPayload("yes")))
        assert type(value) is ExtendedCustomPayload
        assert value.response == "yes"

    def test_unrelated_metadata_cannot_override_declared_type(self, serde_cls):
        raw = serde_cls(CustomPayload).serialize(CustomPayload("yes"))
        with pytest.raises(TypeError, match="not a subclass"):
            serde_cls(ResponsePayload).deserialize(raw)

    def test_declared_type_survives_unimportable_metadata(self, serde_cls):
        raw = b'{"response":"yes","__type__":"unknown_module.OldName"}'
        assert serde_cls(ResponsePayload).deserialize(raw) == ResponsePayload("yes")

    def test_constructor_only_fields(self, serde_cls):
        @dataclass
        class WithInitVar:
            response: InitVar[str]
            normalized: str = field(init=False)

            def __post_init__(self, response):
                self.normalized = response.upper()

        assert serde_cls(WithInitVar).deserialize(b'{"response":"yes"}').normalized == "YES"

    @pytest.mark.parametrize("hint, raw, expected", [(List, b"[1]", [1]), (Dict, b'{"a":1}', {"a": 1})])
    def test_unparameterized_containers(self, serde_cls, hint, raw, expected):
        assert serde_cls(hint).deserialize(raw) == expected


class TestDefaultSerdeCompatibility:
    """The extension retains Restate's other native serde branches."""

    def test_untyped_json(self):
        serde = DefaultSerde()
        assert serde.deserialize(serde.serialize({"a": [1, False]})) == {"a": [1, False]}
        assert serde.deserialize(b"") is None
        assert serde.serialize(None) == b""

    def test_pydantic(self):
        from pydantic import BaseModel, ValidationError

        class Payload(BaseModel):
            response: str

        serde = DefaultSerde(Payload)
        value = Payload(response="yes")
        assert serde.deserialize(serde.serialize(value)) == value
        with pytest.raises(ValidationError):
            serde.deserialize(b"{}")

    def test_msgspec(self):
        msgspec = pytest.importorskip("msgspec")

        class Payload(msgspec.Struct):
            response: str

        serde = DefaultSerde(Payload)
        value = Payload(response="yes")
        assert serde.deserialize(serde.serialize(value)) == value
        with pytest.raises(msgspec.ValidationError):
            serde.deserialize(b"{}")


class TestToDictEncoder:
    """Tests for to_dict_encoder function.

    to_dict_encoder delegates to serialize_value, which includes a __type__ key
    for round-trip deserialization of nested structures. Tests here assume
    serialize_value is correct (tested in serialization tests) and just verify
    the encoder contract.
    """

    def test_object_with_to_dict(self):
        """Should delegate to serialize_value, which includes __type__ metadata."""

        class HasToDict:
            def __init__(self, value: int):
                self.value = value

            def to_dict(self):
                return {"value": self.value}

        obj = HasToDict(42)
        result = to_dict_encoder(obj)
        assert result["value"] == 42
        assert "__type__" in result

    def test_object_without_to_dict_raises(self):
        """Should raise TypeError for objects without to_dict()."""

        class NoToDict:
            def __init__(self, value: int):
                self.value = value

        obj = NoToDict(42)
        with pytest.raises(TypeError, match="not JSON serializable"):
            to_dict_encoder(obj)

    def test_used_with_json_dumps(self):
        """Should work as json.dumps default encoder, including __type__ metadata."""

        class HasToDict:
            def __init__(self, name: str, count: int):
                self.name = name
                self.count = count

            def to_dict(self):
                return {"name": self.name, "count": self.count}

        obj = HasToDict("test", 5)
        result = json.loads(json.dumps(obj, default=to_dict_encoder))
        assert result["name"] == "test"
        assert result["count"] == 5
        assert "__type__" in result

    def test_nested_objects_with_to_dict(self):
        """Should handle nested objects, with __type__ metadata at each level."""

        class Inner:
            def __init__(self, x: int):
                self.x = x

            def to_dict(self):
                return {"x": self.x}

        class Outer:
            def __init__(self, items: list):
                self.items = items

            def to_dict(self):
                return {"items": [item.to_dict() for item in self.items]}

        outer = Outer([Inner(1), Inner(2)])
        result = json.loads(json.dumps(outer, default=to_dict_encoder))
        assert result["items"] == [{"x": 1}, {"x": 2}]
        assert "__type__" in result


class TestDictProtocolSerde:
    """Tests for DictProtocolSerde class."""

    def test_serialize_none(self):
        """Should return empty bytes for None."""
        serde = DictProtocolSerde(dict)
        result = serde.serialize(None)
        assert result == b""

    def test_serialize_dict(self):
        """Should serialize dict to JSON bytes."""
        serde = DictProtocolSerde(dict)
        result = serde.serialize({"key": "value", "num": 42})
        assert json.loads(result) == {"key": "value", "num": 42}

    def test_serialize_list(self):
        """Should serialize list to JSON bytes."""
        serde = DictProtocolSerde(list)
        result = serde.serialize([1, 2, 3])
        assert json.loads(result) == [1, 2, 3]

    def test_serialize_object_with_to_dict(self):
        """Should serialize object using to_dict(), including __type__ metadata."""

        class MyClass:
            def __init__(self, name: str):
                self.name = name

            def to_dict(self):
                return {"name": self.name}

        serde = DictProtocolSerde(MyClass)
        obj = MyClass("test")
        result = json.loads(serde.serialize(obj))
        assert result["name"] == "test"
        assert "__type__" in result

    def test_deserialize_empty_bytes(self):
        """Should return None for empty bytes."""
        serde = DictProtocolSerde(dict)
        result = serde.deserialize(b"")
        assert result is None

    def test_deserialize_dict(self):
        """Should deserialize JSON bytes to dict."""
        serde = DictProtocolSerde(dict)
        data = b'{"key": "value", "num": 42}'
        result = serde.deserialize(data)
        assert result == {"key": "value", "num": 42}

    def test_deserialize_list(self):
        """Should deserialize JSON bytes to list."""
        serde = DictProtocolSerde(list)
        data = b"[1, 2, 3]"
        result = serde.deserialize(data)
        assert result == [1, 2, 3]

    def test_deserialize_with_type_hint(self):
        """Should use type_hint for deserialization."""

        @dataclass
        class Config:
            name: str
            value: int

        serde = DictProtocolSerde(Config)
        data = b'{"name": "test", "value": 42}'
        result = serde.deserialize(data)

        assert isinstance(result, Config)
        assert result.name == "test"
        assert result.value == 42

    def test_roundtrip_dict(self):
        """Should successfully roundtrip a dict."""
        serde = DictProtocolSerde(dict)
        original = {"nested": {"a": 1, "b": [2, 3]}, "flag": True}

        serialized = serde.serialize(original)
        deserialized = serde.deserialize(serialized)

        assert deserialized == original

    def test_serialize_dataclass_without_to_dict(self):
        """Plain dataclasses are serialized via serialize_value (Haystack serialization).

        Note: Plain dataclasses should prefer DefaultSerde (with patch_default_serde),
        but DictProtocolSerde handles them too since serialize_value covers dataclasses.
        """

        @dataclass
        class Person:
            name: str
            age: int

        serde = DictProtocolSerde(Person)
        original = Person(name="Alice", age=30)

        result = json.loads(serde.serialize(original))
        assert result["name"] == "Alice"
        assert result["age"] == 30
        assert "__type__" in result

    def test_roundtrip_approval_dataclass(self):
        """Plain approval dataclasses round-trip without custom dictionary methods."""
        from ravelin.core.restate import ToolExecutionApprovalPayload

        serde = DictProtocolSerde(ToolExecutionApprovalPayload)
        original = ToolExecutionApprovalPayload(approved=True, reason="looks good")

        serialized = serde.serialize(original)
        deserialized = serde.deserialize(serialized)

        assert isinstance(deserialized, ToolExecutionApprovalPayload)
        assert deserialized.approved == original.approved
        assert deserialized.reason == original.reason

    def test_serialize_nested_objects(self):
        """Should serialize nested objects with to_dict, including __type__ at each level."""

        class Inner:
            def __init__(self, x: int):
                self.x = x

            def to_dict(self):
                return {"x": self.x}

        class Outer:
            def __init__(self, inner: "Inner"):
                self.inner = inner

            def to_dict(self):
                return {"inner": self.inner.to_dict()}

        serde = DictProtocolSerde(Outer)
        obj = Outer(Inner(42))

        result = json.loads(serde.serialize(obj))
        assert result["inner"] == {"x": 42}
        assert "__type__" in result


class TestDictProtocolSerdeWithOptional:
    """Tests for DictProtocolSerde with Optional types."""

    def test_optional_none(self):
        """Should handle Optional[T] with None value."""
        serde = DictProtocolSerde(Optional[dict])
        assert serde.serialize(None) == b""
        assert serde.deserialize(b"") is None

    def test_optional_with_value(self):
        """Should handle Optional[T] with actual value."""
        serde = DictProtocolSerde(Optional[dict])
        original = {"key": "value"}

        serialized = serde.serialize(original)
        deserialized = serde.deserialize(serialized)

        assert deserialized == original


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
