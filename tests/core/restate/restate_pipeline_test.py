import os

import pytest
from haystack import component

from ravelin.core.restate.restate_pipeline import RestatePipeline


@component
class _Noop:
    @component.output_types(out=int)
    def run(self, value: int = 0):
        return {"out": value}


class TestRestatePipelineBasics:
    """Test basic RestatePipeline functionality without Restate server."""

    def test_pipeline_inherits_from_pipeline_base(self):
        """RestatePipeline should inherit from PipelineBase."""
        from haystack.core.pipeline.base import PipelineBase

        pipeline = RestatePipeline()
        assert isinstance(pipeline, PipelineBase)

    def test_create_component_future_returns_callable(self):
        """_create_component_future should be a method on the pipeline."""
        pipeline = RestatePipeline()
        assert hasattr(pipeline, "_create_component_future")
        assert callable(pipeline._create_component_future)

    def test_executor_handler_initially_none(self):
        """_executor_handler should be None before create_restate_app is called."""
        pipeline = RestatePipeline()
        assert pipeline._executor_handler is None


class TestMaxRunsPerComponent:
    def test_dispatch_allowed_up_to_limit_completed_runs(self):
        # visits counts completed runs; with a limit of 3 the 4th run (visits=3) is still dispatched.
        pipeline = RestatePipeline(max_runs_per_component=3)
        pipeline._check_max_runs("a", 0)
        pipeline._check_max_runs("a", 3)

    def test_dispatch_refused_once_visits_exceed_limit(self):
        import restate

        pipeline = RestatePipeline(max_runs_per_component=3)
        with pytest.raises(restate.TerminalError, match="Maximum run count 3 reached for component 'a'"):
            pipeline._check_max_runs("a", 4)

    @pytest.mark.parametrize("visits", [0, 2, 3, 4, 5])
    def test_matches_haystack_pipeline_check(self, visits):
        """The limit must behave the same as Haystack's Pipeline, which RestatePipeline replaces."""
        import restate
        from haystack.core.errors import PipelineMaxComponentRuns
        from haystack.core.pipeline.base import ComponentPriority, PipelineBase

        class _Queue:
            def get(self):
                return (ComponentPriority.READY, "a")

        pipeline = RestatePipeline(max_runs_per_component=3)
        pipeline.add_component("a", _Noop())

        def haystack_refuses() -> bool:
            try:
                PipelineBase._get_next_runnable_component(pipeline, _Queue(), {"a": visits})
            except PipelineMaxComponentRuns:
                return True
            return False

        def restate_refuses() -> bool:
            try:
                pipeline._check_max_runs("a", visits)
            except restate.TerminalError:
                return True
            return False

        assert restate_refuses() == haystack_refuses()


class TestDeterministicComponentCall:
    def test_component_call_inputs_are_sorted(self):
        """Replay compares the journaled call byte-for-byte, so input order must not depend on socket order."""
        from unittest.mock import MagicMock

        pipeline = RestatePipeline()
        pipeline._executor_handler = MagicMock()
        ctx = MagicMock()
        ctx.key.return_value = "run-1"

        pipeline._create_component_future(ctx, "builder", {"query": "q", "context": "c", "agent": "a"}, 0)

        arg = ctx.service_call.call_args.kwargs["arg"]
        assert list(arg["inputs"]) == ["agent", "context", "query"]

    def test_nested_dict_order_is_preserved(self):
        """Components may depend on the key order of the values they receive; only top-level keys are sorted."""
        from unittest.mock import MagicMock

        pipeline = RestatePipeline()
        pipeline._executor_handler = MagicMock()
        ctx = MagicMock()

        pipeline._create_component_future(
            ctx, "c", {"text": {"z": 1, "a": {"y": 2, "b": 3}}, "meta": [{"q": 1, "c": 2}]}, 0
        )

        inputs = ctx.service_call.call_args.kwargs["arg"]["inputs"]
        assert list(inputs) == ["meta", "text"]
        assert list(inputs["text"]) == ["z", "a"]
        assert list(inputs["text"]["a"]) == ["y", "b"]
        assert list(inputs["meta"][0]) == ["q", "c"]

    def test_json_incompatible_inputs_are_terminal(self):
        """Encoding fails after the call is issued and would be retried forever; reject it before."""
        from unittest.mock import MagicMock

        import restate

        class Opaque:
            pass

        pipeline = RestatePipeline()
        pipeline._executor_handler = MagicMock()
        ctx = MagicMock()

        with pytest.raises(restate.TerminalError, match=r"inputs\.query: Opaque"):
            pipeline._create_component_future(ctx, "builder", {"query": Opaque()}, 0)
        ctx.service_call.assert_not_called()


# Routes one journaled output from "sender" to "receiver" and prints the receiver's call parameter. Tags.to_dict()
# iterates a set, so re-encoding a decoded Tags gives a different key order under each hash seed.
_ROUTING_SCRIPT = """
import json
from unittest.mock import MagicMock
from haystack import component
from ravelin.core.restate.restate_pipeline import RestatePipeline, _outputs_for_routing

class Tags:
    def __init__(self, values):
        self.values = set(values)
    def to_dict(self):
        return {value: True for value in self.values}
    @classmethod
    def from_dict(cls, data):
        return cls(key for key in data if key != "__type__")

@component
class Sender:
    @component.output_types(tags=Tags)
    def run(self):
        return {}

@component
class Receiver:
    @component.output_types(out=int)
    def run(self, tags: Tags):
        return {}

pipeline = RestatePipeline()
pipeline.add_component("sender", Sender())
pipeline.add_component("receiver", Receiver())
pipeline.connect("sender.tags", "receiver.tags")
pipeline._executor_handler = MagicMock()
ctx = MagicMock()

journaled = json.loads(input())
receivers = pipeline._find_receivers_from("sender")
inputs_state = {}
pipeline._write_component_outputs(
    component_name="sender",
    component_outputs=_outputs_for_routing(journaled, receivers),
    inputs=inputs_state,
    receivers=receivers,
    include_outputs_from=set(),
)
receiver = pipeline._get_component_with_graph_metadata_and_visits("receiver", 0)
pipeline._create_component_future(ctx, "receiver", pipeline._consume_component_inputs("receiver", receiver, inputs_state), 0)
print(json.dumps(ctx.service_call.call_args.kwargs["arg"]["inputs"]))
"""


class TestOutputRouting:
    def test_routed_outputs_reach_the_next_call_as_journaled(self):
        """A replay in a new process must rebuild the downstream call parameter byte-for-byte."""
        import json
        import subprocess
        import sys
        from pathlib import Path

        import ravelin

        tags = {name: True for name in ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]}
        journaled = json.dumps({"tags": {**tags, "__type__": "__main__.Tags"}})
        # Import the ravelin under test, not whichever copy the environment has installed.
        package_root = str(Path(ravelin.__file__).resolve().parents[1])

        call_parameters = {
            subprocess.run(
                [sys.executable, "-c", _ROUTING_SCRIPT],
                input=journaled,
                capture_output=True,
                text=True,
                check=True,
                env={**os.environ, "PYTHONHASHSEED": str(seed), "PYTHONPATH": package_root},
            ).stdout.strip()
            for seed in range(1, 6)
        }

        assert call_parameters == {json.dumps(json.loads(journaled))}

    def test_plain_outputs_are_not_decoded(self):
        from ravelin.core.restate.restate_pipeline import _outputs_for_routing

        pipeline = RestatePipeline()
        pipeline.add_component("a", _Noop())
        pipeline.add_component("b", _Noop())
        pipeline.connect("a.out", "b.value")
        raw = {"out": {"__type__": "haystack.dataclasses.chat_message.ChatMessage"}}

        assert _outputs_for_routing(raw, pipeline._find_receivers_from("a")) is raw

    def test_structural_conversions_keep_the_journaled_json(self):
        """WRAP and UNWRAP only restructure lists, so the element keeps its journaled encoding."""
        from ravelin.core.restate.restate_pipeline import _outputs_for_routing

        @component
        class Many:
            @component.output_types(items=list[int])
            def run(self):
                return {}

        pipeline = RestatePipeline()
        pipeline.add_component("many", Many())
        pipeline.add_component("one", _Noop())
        pipeline.connect("many.items", "one.value")
        receivers = pipeline._find_receivers_from("many")
        element = {"z": 1, "__type__": "tests.Unimportable"}
        raw = {"items": [element]}
        inputs_state: dict = {}

        routed = _outputs_for_routing(raw, receivers)
        pipeline._write_component_outputs(
            component_name="many",
            component_outputs=routed,
            inputs=inputs_state,
            receivers=receivers,
            include_outputs_from=set(),
        )

        assert routed is raw
        assert inputs_state["one"]["value"][0]["value"] == element

    def test_str_to_chat_message_conversion_builds_the_message(self):
        from haystack.dataclasses import ChatMessage

        from ravelin.core.restate.restate_pipeline import _outputs_for_routing

        @component
        class Prompt:
            @component.output_types(prompt=str)
            def run(self):
                return {}

        @component
        class Chat:
            @component.output_types(out=int)
            def run(self, messages: list[ChatMessage]):
                return {}

        pipeline = RestatePipeline()
        pipeline.add_component("prompt", Prompt())
        pipeline.add_component("chat", Chat())
        pipeline.connect("prompt.prompt", "chat.messages")
        receivers = pipeline._find_receivers_from("prompt")
        inputs_state: dict = {}

        pipeline._write_component_outputs(
            component_name="prompt",
            component_outputs=_outputs_for_routing({"prompt": "hi"}, receivers),
            inputs=inputs_state,
            receivers=receivers,
            include_outputs_from=set(),
        )

        assert inputs_state["chat"]["messages"][0]["value"] == [ChatMessage.from_user("hi")]

    def test_chat_message_to_str_conversion_gets_the_object(self):
        """The conversion reads ChatMessage.text, which the journaled JSON does not have."""
        from haystack.dataclasses import ChatMessage

        from ravelin.core.restate.restate_pipeline import _outputs_for_routing
        from ravelin.core.serialization import serialize_outputs

        @component
        class Chat:
            @component.output_types(message=ChatMessage, other=int)
            def run(self):
                return {}

        @component
        class Text:
            @component.output_types(out=int)
            def run(self, text: str):
                return {}

        pipeline = RestatePipeline()
        pipeline.add_component("chat", Chat())
        pipeline.add_component("text", Text())
        pipeline.connect("chat.message", "text.text")
        receivers = pipeline._find_receivers_from("chat")
        raw = serialize_outputs({"message": ChatMessage.from_user("hi"), "other": 1})
        inputs_state: dict = {}

        routed = _outputs_for_routing(raw, receivers)
        pipeline._write_component_outputs(
            component_name="chat",
            component_outputs=routed,
            inputs=inputs_state,
            receivers=receivers,
            include_outputs_from=set(),
        )

        assert list(routed) == ["message", "other"]
        assert routed["other"] == 1
        assert inputs_state["text"]["text"][0]["value"] == "hi"

    def test_non_list_flattened_into_lazy_variadic_socket_is_decoded(self):
        """Flattening iterates the value; iterating an encoded set would yield its keys."""
        from ravelin.core.restate.restate_pipeline import _outputs_for_routing
        from ravelin.core.serialization import serialize_outputs

        @component
        class Items:
            @component.output_types(items=list[str])
            def run(self):
                return {}

        @component
        class Collect:
            @component.output_types(out=int)
            def run(self, items: list[str]):
                return {}

        pipeline = RestatePipeline()
        pipeline.add_component("first", Items())
        pipeline.add_component("second", Items())
        pipeline.add_component("collect", Collect())
        pipeline.connect("first.items", "collect.items")
        pipeline.connect("second.items", "collect.items")
        inputs_state: dict = {}

        for name, items in [("first", {"x"}), ("second", ["y"])]:
            receivers = pipeline._find_receivers_from(name)
            pipeline._write_component_outputs(
                component_name=name,
                component_outputs=_outputs_for_routing(serialize_outputs({"items": items}), receivers),
                inputs=inputs_state,
                receivers=receivers,
                include_outputs_from=set(),
            )
        collect = pipeline._get_component_with_graph_metadata_and_visits("collect", 0)

        assert pipeline._consume_component_inputs("collect", collect, inputs_state) == {"items": ["x", "y"]}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
