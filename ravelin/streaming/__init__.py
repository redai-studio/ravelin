"""
Streaming system public API.

    Events
        StreamingEvent       unified event envelope
        EventType            token_stream | component_activity | lifecycle
        ComponentInvocation  named (run_id, component, call_index) key
        StreamingSnapshot    catch-up event carrying accumulated token-stream state
        LIFECYCLE_START/COMPLETE/ERROR   lifecycle subtype constants
        GENERATION_START/COMPLETE/ERROR generation subtype constants (TOKEN_STREAM)
        CATCHUP_SUBTYPE      subtype for synthetic catch-up events (relay, Phase 5)

    Relay
        StreamingRelay      framework-agnostic delivery layer; yields events or SSE strings
        EventFilter         visibility filter for stream()/stream_sse(); merged by intersection

    Pipeline infrastructure
        PipelineContext      run-scoped coordinator; created by streaming_session()
        get_pipeline_context()   returns the active PipelineContext or None
        streaming_session()  async context manager; establishes a streaming session
        JournalHook          type alias for the optional journal callback

    Component API
        ComponentContext     component-scoped emitter; use this in model services and components
        get_component_context()  returns the ComponentContext for the current component or None
"""

from ravelin.streaming.context import (
    ComponentContext,
    JournalHook,
    PipelineContext,
    get_component_context,
    get_pipeline_context,
    streaming_session,
    streaming_session_async,
)
from ravelin.streaming.event import (
    CATCHUP_SUBTYPE,
    GENERATION_COMPLETE,
    GENERATION_ERROR,
    GENERATION_START,
    LIFECYCLE_COMPLETE,
    LIFECYCLE_ERROR,
    LIFECYCLE_START,
    ComponentInvocation,
    EventType,
    StreamingSnapshot,
    StreamingEvent,
)
from ravelin.streaming.relay import EventFilter, StreamingRelay

__all__ = [
    # Events
    "StreamingEvent",
    "EventType",
    "ComponentInvocation",
    "StreamingSnapshot",
    "LIFECYCLE_START",
    "LIFECYCLE_COMPLETE",
    "LIFECYCLE_ERROR",
    "GENERATION_START",
    "GENERATION_COMPLETE",
    "GENERATION_ERROR",
    "CATCHUP_SUBTYPE",
    # Relay
    "StreamingRelay",
    "EventFilter",
    # Pipeline infrastructure
    "PipelineContext",
    "get_pipeline_context",
    "streaming_session",
    "streaming_session_async",
    "JournalHook",
    # Component API
    "ComponentContext",
    "get_component_context",
]
