"""Typed approval responses through real Restate resolution and replay.

These tests require the Restate test harness and a reachable Docker daemon.
"""

import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import dataclass
from queue import Queue
from uuid import uuid4

import pytest

pytestmark = pytest.mark.integration

restate = pytest.importorskip("restate")


@dataclass
class HumanResponse:
    response: str


@dataclass
class NestedResponse:
    child: HumanResponse

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


@dataclass
class CheckedResponse:
    response: str

    @classmethod
    def from_dict(cls, data):
        if data["response"] != "allowed":
            raise ValueError("response denied")
        return cls(**data)


@dataclass
class UnionResponse:
    decision: str | CheckedResponse


@pytest.fixture(scope="module")
def approval_harness():
    docker = pytest.importorskip("docker")
    try:
        with closing(docker.from_env()) as client:
            client.ping()
    except docker.errors.DockerException as exc:
        pytest.skip(f"Restate test harness needs a reachable Docker daemon: {exc}")

    from haystack import component
    from ravelin.core.restate import RestatePipeline, create_restate_app, require_approval
    from ravelin.utils.pipeline_io_handler import PipelineIOHandler

    notifications = Queue()

    def notify(notification):
        notifications.put(notification)

    @component
    class Gate:
        @component.output_types(response=str)
        def run(self, timeout_seconds: int, payload_kind: str = "plain") -> dict[str, str]:
            raise NotImplementedError("Use run_async")

        @component.output_types(response=str)
        async def run_async(self, timeout_seconds: int, payload_kind: str = "plain") -> dict[str, str]:
            payload_type = {"plain": HumanResponse, "nested": NestedResponse, "union": UnionResponse}[payload_kind]
            decision = await require_approval(
                payload_type,
                timeout_seconds=timeout_seconds or None,
                notifier=notify,
            )
            if payload_kind == "nested":
                assert isinstance(decision.child, HumanResponse)
                return {"response": decision.child.response}
            if payload_kind == "union":
                assert isinstance(decision.decision, CheckedResponse)
                return {"response": decision.decision.response}
            return {"response": decision.response}

    pipeline = RestatePipeline()
    pipeline.add_component("gate", Gate())
    io_handler = PipelineIOHandler(
        inputs={
            "timeout_seconds": {"type": int, "connections": ["gate.timeout_seconds"], "required": True},
            "payload_kind": {"type": str, "connections": ["gate.payload_kind"]},
        },
        outputs={"response": {"type": str, "connection": "gate.response"}},
    )
    workflow, executor = create_restate_app(pipeline, io_handler, "ApprovalTest")
    # Retry behavior stays enabled: malformed durable responses must terminate,
    # rather than merely fail because the harness disabled retries.
    with restate.test_harness(restate.app([workflow, executor]), always_replay=True) as harness:
        yield harness, notifications


@pytest.mark.parametrize("route", ["executor", "direct"])
@pytest.mark.parametrize("timeout_seconds", [0, 60])
@pytest.mark.parametrize(
    "payload, valid",
    [
        ({"response": "yes"}, True),
        ({"response": "yes", "extra": "future field"}, True),
        ({}, False),
        ({"response": 42}, False),
    ],
)
def test_typed_awakeable_resolution(approval_harness, route, timeout_seconds, payload, valid):
    harness, notifications = approval_harness
    with harness.ingress_client() as caller, harness.ingress_client() as resolver, ThreadPoolExecutor() as pool:
        result = pool.submit(
            caller.post,
            f"/ApprovalTest/{uuid4()}/run",
            json={"timeout_seconds": timeout_seconds},
            timeout=30,
        )
        notification = notifications.get(timeout=15)
        awakeable_id = notification["awakeable_id"]
        if route == "executor":
            response = resolver.post(
                "/ApprovalTest_Executor/resolve_awakeable",
                json={"awakeable_id": awakeable_id, "payload": payload},
                timeout=10,
            )
        else:
            response = resolver.post(f"/restate/awakeables/{awakeable_id}/resolve", json=payload, timeout=10)
        assert response.is_success, response.text
        completed = result.result(timeout=30)
        if valid:
            assert completed.status_code == 200, completed.text
            assert completed.json() == {"response": "yes"}
        else:
            assert completed.status_code == 500, completed.text
            assert "invalid payload" in completed.text
            assert "HumanResponse" in completed.text


def test_missing_envelope_payload_leaves_awakeable_resolvable(approval_harness):
    harness, notifications = approval_harness
    with harness.ingress_client() as caller, harness.ingress_client() as resolver, ThreadPoolExecutor() as pool:
        result = pool.submit(caller.post, f"/ApprovalTest/{uuid4()}/run", json={"timeout_seconds": 0}, timeout=30)
        awakeable_id = notifications.get(timeout=15)["awakeable_id"]
        rejected = resolver.post(
            "/ApprovalTest_Executor/resolve_awakeable", json={"awakeable_id": awakeable_id}, timeout=10
        )
        assert rejected.status_code == 400, rejected.text
        assert "payload is required" in rejected.text
        accepted = resolver.post(
            "/ApprovalTest_Executor/resolve_awakeable",
            json={"awakeable_id": awakeable_id, "payload": {"response": "corrected"}},
            timeout=10,
        )
        assert accepted.is_success, accepted.text
        completed = result.result(timeout=30)
        assert completed.status_code == 200, completed.text
        assert completed.json() == {"response": "corrected"}


@pytest.mark.parametrize("route", ["executor", "direct"])
@pytest.mark.parametrize("payload_kind, response", [("nested", "yes"), ("union", "allowed"), ("union", "denied")])
def test_custom_decoder_contracts_through_replay(approval_harness, route, payload_kind, response):
    from restate.serde import DefaultSerde

    harness, notifications = approval_harness
    payload = (
        NestedResponse(HumanResponse(response))
        if payload_kind == "nested"
        else UnionResponse(CheckedResponse(response))
    )
    raw = DefaultSerde(type(payload)).serialize(payload)
    with harness.ingress_client() as caller, harness.ingress_client() as resolver, ThreadPoolExecutor() as pool:
        result = pool.submit(
            caller.post,
            f"/ApprovalTest/{uuid4()}/run",
            json={"timeout_seconds": 60, "payload_kind": payload_kind},
            timeout=30,
        )
        awakeable_id = notifications.get(timeout=15)["awakeable_id"]
        if route == "executor":
            resolved = resolver.post(
                "/ApprovalTest_Executor/resolve_awakeable",
                json={"awakeable_id": awakeable_id, "payload": json.loads(raw)},
                timeout=10,
            )
        else:
            resolved = resolver.post(
                f"/restate/awakeables/{awakeable_id}/resolve",
                content=raw,
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
        assert resolved.is_success, resolved.text
        completed = result.result(timeout=30)
        if response == "denied":
            assert completed.status_code == 500, completed.text
            assert "invalid payload" in completed.text
            assert "UnionResponse" in completed.text
        else:
            assert completed.status_code == 200, completed.text
            assert completed.json() == {"response": response}
