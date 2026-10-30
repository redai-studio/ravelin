"""Core event types and data structures for the streaming system."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, NamedTuple, Optional

from haystack.dataclasses import ToolCallDelta


class EventType(str, Enum):
    """Streaming event type — determines routing and persistence semantics."""

    TOKEN_STREAM = "token_stream"  # LLM token chunks (high-frequency), not journaled
    COMPONENT_ACTIVITY = "component_activity"  # structured component notifications
    LIFECYCLE = "lifecycle"  # pipeline-level start/complete/error


# Lifecycle subtype constants
LIFECYCLE_START = "start"
LIFECYCLE_COMPLETE = "complete"
LIFECYCLE_ERROR = "error"

# Catch-up subtype for synthetic snapshot events sent to reconnecting clients
CATCHUP_SUBTYPE = "catch_up"

# Generation subtypes for TOKEN_STREAM events (streaming lifecycle markers)
# These are fire-and-forget signals — not journaled, not COMPONENT_ACTIVITY.
# The backend uses them to manage snapshot lifecycle automatically.
GENERATION_START = "generation_start"
GENERATION_DELTA = "generation_delta"
GENERATION_COMPLETE = "generation_complete"
GENERATION_ERROR = "generation_error"


class ComponentInvocation(NamedTuple):
    """
    Globally unique key for one specific execution of a component.

    Used as the snapshot storage key throughout the streaming system. Including
    ``run_id`` means a single backend instance can hold snapshots for concurrent
    runs without collision. Exposed as a first-class type so developers have a
    named, self-documenting handle rather than a raw (str, str, int) tuple.

    ``generation_id`` distinguishes multiple durable generation steps that share
    one ``(run_id, component, call_index)`` — e.g. two ``ctx.run`` LLM calls in
    the same component invocation. It is a caller-minted token, stable across a
    step's retries and distinct per logical step. ``None`` (the default) means no
    token was supplied; the key then degrades to the triple and retry counting is
    not tracked separately per step.
    """

    run_id: str
    component: str
    call_index: int
    generation_id: Optional[str] = None


@dataclass
class StreamingEvent:
    """
    Unified envelope for all streaming events emitted within a pipeline run.

    (component, call_index) uniquely identifies one invocation of a component
    within the run. call_index is 0-based and incremented by the pipeline layer
    each time the same component starts executing.

    event_id is assigned by the backend on write; None before the event is written.
    """

    run_id: str
    component: str
    call_index: int
    type: EventType
    # LIFECYCLE_* for lifecycle events; GENERATION_* for token streams; component-defined for component activities
    subtype: Optional[str]
    payload: Any
    event_id: Optional[str] = None
    # Caller-minted token identifying one durable generation step within a
    # (run_id, component, call_index). Stable across a step's retries, distinct
    # per logical step. None when no token was supplied (e.g. AsyncPipeline).
    generation_id: Optional[str] = None


@dataclass
class StreamingSnapshot(StreamingEvent):
    """
    Catch-up event delivering the full accumulated token stream state to a reconnecting client.

    IS a StreamingEvent — relay yields it alongside live events with no wrapping.
    type and subtype are fixed (TOKEN_STREAM / catch_up); set via init=False so
    callers cannot override them.

    payload is a StreamingSnapshot.Payload instance. Backends accumulate token
    chunks into Payload directly; get_snapshot() wraps a copy into StreamingSnapshot
    with the full event identity (run_id, component, call_index) filled in.
    """

    @dataclass
    class Payload:
        """Accumulated token stream state for catch-up delivery."""

        content: str = ""
        reasoning_content: str = ""
        tool_call_deltas: List[ToolCallDelta] = field(default_factory=list)
        # Number of failed generation attempts observed for this key so far.
        # Incremented on GENERATION_ERROR, preserved across GENERATION_START
        # (content resets per attempt, the count does not), dropped on
        # GENERATION_COMPLETE. Lets a freshly connected client learn how many
        # times an in-flight generation has retried.
        retry_count: int = 0

    # pylint: disable=unnecessary-lambda
    payload: "StreamingSnapshot.Payload" = field(default_factory=lambda: StreamingSnapshot.Payload())
    type: EventType = field(default=EventType.TOKEN_STREAM, init=False)
    subtype: Optional[str] = field(default=CATCHUP_SUBTYPE, init=False)
