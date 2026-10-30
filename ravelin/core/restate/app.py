"""Restate Workflow and Service creation for RestatePipeline."""

import asyncio
import sys
from typing import Any, Mapping, Optional, TYPE_CHECKING

from haystack import logging, tracing
from haystack.core.component import Component
from haystack.lazy_imports import LazyImport

from ravelin.core.restate.context import _restate_ctx
from ravelin.core.restate.errors import (
    ErrorClassification,
    ErrorClassifier,
    ErrorContext,
    build_error_event_payload,
    classify_restate_error,
    classify_restate_error_fallback,
    terminal_error_message,
)
from ravelin.core.serialization import deserialize_inputs, find_json_incompatible_value, serialize_outputs
from ravelin.streaming.backends import StreamingBackend
from ravelin.streaming.context import JournalHook, streaming_session_async
from ravelin.streaming.lifecycle import _run_with_lifecycle_async
from ravelin.core.type_validation import validate_types
from ravelin.utils.pipeline_io_handler import PipelineIOHandler

if TYPE_CHECKING:
    from ravelin.core.restate.restate_pipeline import RestatePipeline

with LazyImport("Run 'pip install restate_sdk[serde]'") as restate_import:
    import restate


logger = logging.getLogger(__name__)

# Deprecated compatibility export. Component runtime failures are now classified
# by ravelin.core.restate.errors and are retryable by default.
NON_RETRYABLE_ERRORS: tuple[type[Exception], ...] = (
    TypeError,
    ValueError,
    KeyError,
    AttributeError,
    IndexError,
    AssertionError,
    ImportError,
    SyntaxError,
    OSError,
)

# Add PythonFinalizationError for Python 3.13+
if sys.version_info >= (3, 13):
    # pylint: disable-next=undefined-variable
    NON_RETRYABLE_ERRORS = (*NON_RETRYABLE_ERRORS, PythonFinalizationError)  # type: ignore[name-defined]  # noqa: F821

# Header key to control type validation (default: enabled, set to "false" to disable)
_TYPE_VALIDATION_HEADER = "x-type-validation"


def create_restate_app(
    pipeline: "RestatePipeline",
    io_handler: PipelineIOHandler,
    name: str | None = None,
    streaming_backend: Optional[StreamingBackend] = None,
    on_journal_event: Optional[JournalHook] = None,
    error_classifier: Optional[ErrorClassifier] = None,
    invocation_retry_policy: Optional["restate.InvocationRetryPolicy"] = None,
) -> tuple["restate.Workflow", "restate.Service"]:
    """
    Create Restate Workflow and Service for a RestatePipeline.

    This function creates:
    1. A Workflow for pipeline orchestration (durable, long-running)
    2. A Service for component execution (durable request-response)

    The separation ensures proper Restate context access:
    - Workflow handles orchestration (which component runs next)
    - Service handles component execution (components can use ctx freely)

    The io_handler defines the HTTP interface:
    - Input specs define request schema and type hints for deserialization
    - Output specs define response schema
    - Handles mapping between pipeline-level and component-level I/O

    Usage::
        pipeline = RestatePipeline()
        pipeline.add_component("retriever", retriever)
        pipeline.add_component("llm", llm)
        pipeline.connect("retriever", "llm")

        io_handler = PipelineIOHandler(
            inputs={"query": {"type": str, "connections": ["retriever.query"]}},
            outputs={"answer": {"type": str, "connection": "llm.replies"}},
        )

        workflow, executor = create_restate_app(pipeline, io_handler, "MyRAGPipeline")
        app = restate.app([workflow, executor])

        # Start: python -m hypercorn app:app --bind 0.0.0.0:9080
        # Register: restate deployments register http://localhost:9080
        # Invoke: curl -X POST http://localhost:8080/MyRAGPipeline/run-1/run -d '{"query": "..."}'

    :param pipeline: The RestatePipeline instance to wrap.
    :param io_handler: PipelineIOHandler defining the HTTP interface.
    :param name: Base name for Workflow and Service. Defaults to "Pipeline".
    :param streaming_backend: Optional shared :class:`StreamingBackend` instance. When provided,
        all streaming events (token-stream, component-activity, lifecycle) are emitted
        by the *executor* handler (``execute_component``) via a ``PipelineContext``
        installed on the contextvar. The executor is the Restate retry boundary, so
        lifecycle events (LIFECYCLE_START / COMPLETE / ERROR) fire on every attempt.

        The *workflow* handler is responsible only for calling ``backend.close_run(run_id)``
        when the pipeline finishes, which signals subscriber iterators to stop.

        A single backend instance serves all concurrent runs — use ``event.run_id``
        to distinguish them. On process restart the backend's in-memory state is lost;
        use ``RedisStreamingBackend`` for durability across restarts.
    :param on_journal_event: Optional :data:`JournalHook` awaited for every non-token
        event *before* the backend write. Exceptions propagate (strict persistence
        semantics). Use this to persist domain events to a store — the hook receives
        the full ``StreamingEvent`` and can inspect ``event.payload`` for a
        ``domain_id`` key to distinguish domain events from best-effort ones.
    :param error_classifier: Optional callback for classifying component runtime
        failures. Restate internal exceptions bypass this callback. Returning
        ``None`` falls back to the built-in classifier.
    :param invocation_retry_policy: Optional Restate invocation retry policy passed
        through to the created workflow and executor. When omitted, Restate server
        defaults apply.
    :returns: Tuple of (Workflow, Service) ready to register with restate.app().
    """
    restate_import.check()

    base_name = name or pipeline.metadata.get("name", "Pipeline")
    workflow_name = base_name
    executor_name = f"{base_name}_Executor"

    # Create the component executor Service
    executor = restate.Service(executor_name, invocation_retry_policy=invocation_retry_policy)

    def _classify_execution_error(error: BaseException, context: ErrorContext) -> ErrorClassification:
        try:
            return classify_restate_error(error, context, error_classifier)
        except Exception:  # pylint: disable=broad-exception-caught
            logger.warning("Restate error classifier failed; falling back to unknown_retryable", exc_info=True)
            return classify_restate_error_fallback(error, context)

    # no cover: start - Restate handlers tested via TestRestateIntegration harness tests
    @executor.handler()
    async def warm_up_component(_: "restate.Context", request: dict[str, Any]) -> dict[str, Any]:
        """
        Warm up a component before execution.

        This ensures component initialization (loading models, etc.) happens
        in the executor where the component will actually run.

        Request format: {"component_name": str}
        """
        component_name = request["component_name"]

        if component_name not in pipeline.graph.nodes:
            raise restate.TerminalError(
                f"Component '{component_name}' not found. Available: {list(pipeline.graph.nodes.keys())}"
            )

        instance: Component = pipeline.graph.nodes[component_name]["instance"]

        # Call warm_up if the component has it
        if hasattr(instance, "warm_up") and callable(instance.warm_up):
            logger.info("Warming up component {component_name}", component_name=component_name)
            instance.warm_up()

        return {"success": True, "component_name": component_name}

    # no cover: stop

    # no cover: start - Restate handler tested via TestRestateIntegration harness tests
    @executor.handler()
    async def execute_component(ctx: "restate.Context", request: dict[str, Any]) -> dict[str, Any]:
        """
        Execute a single component with full Restate context access.

        The component can use get_restate_context() to access ctx for:
        - ctx.run() for durable side effects
        - ctx.awakeable() for human-in-the-loop
        - ctx.sleep() for durable timers

        Request format: {"component_name": str, "inputs": dict, "workflow_id": str}
        """
        component_name = request["component_name"]
        raw_inputs = request["inputs"]
        workflow_id = request.get("workflow_id", "unknown")
        call_index = request.get("call_index", 0)

        if component_name not in pipeline.graph.nodes:
            raise restate.TerminalError(
                f"Component '{component_name}' not found. Available: {list(pipeline.graph.nodes.keys())}"
            )

        instance: Component = pipeline.graph.nodes[component_name]["instance"]

        # Extract type hints from component sockets
        # __haystack_input__ is a Haystack internal attribute for component input sockets
        # pylint: disable=protected-access
        sockets = instance.__haystack_input__._sockets_dict  # type: ignore[attr-defined]
        input_type_hints = {name_: socket.type for name_, socket in sockets.items()}

        # Deserialize inputs using socket type hints (TerminalError if fails)
        try:
            inputs = deserialize_inputs(raw_inputs, type_hints=input_type_hints)
        except Exception as e:
            raise restate.TerminalError(f"Failed to deserialize inputs for component '{component_name}': {e}") from e

        # Validate types unless disabled via header (internal calls disable to save resources)
        if ctx.request().headers.get(_TYPE_VALIDATION_HEADER, "true").lower() != "false":
            try:
                validate_types(inputs, input_type_hints)
            except TypeError as e:
                raise restate.TerminalError(f"Component '{component_name}' input type validation failed: {e}") from e

        async def _execute_component() -> Mapping[str, Any]:
            """Run the component with tracing. Restate error recasting happens outside."""
            # TODO: Full span hierarchy requires Haystack Tracer abstraction changes
            # to support span ID serialization across service boundaries. For now,
            # component spans are independent. Restate provides execution tracking
            # via its admin API and journal.
            with tracing.tracer.trace(
                "haystack.component.run",
                tags={
                    "haystack.component.name": component_name,
                    "restate.workflow_id": workflow_id,
                },
            ) as span:
                logger.info("Executing component {component_name}", component_name=component_name)
                if getattr(instance, "__haystack_supports_async__", False):
                    outputs = await instance.run_async(**inputs)  # type: ignore
                else:
                    loop = asyncio.get_running_loop()
                    outputs = await loop.run_in_executor(None, lambda: instance.run(**inputs))

                if not isinstance(outputs, Mapping):
                    raise restate.TerminalError(
                        f"Component '{component_name}' returned invalid output type: {type(outputs)}"
                    )
                span.set_tag("haystack.component.output_keys", list(outputs.keys()))
            return outputs

        # Expose Restate ctx to component code via ContextVar.
        error_context = ErrorContext(
            component_name=component_name,
            workflow_id=workflow_id,
            call_index=call_index,
            phase="component_execution",
        )
        classification_cache: dict[int, ErrorClassification] = {}

        def _classification_for(error: BaseException) -> ErrorClassification:
            cached = classification_cache.get(id(error))
            if cached is not None:
                return cached
            classification = _classify_execution_error(error, error_context)
            classification_cache[id(error)] = classification
            return classification

        def _error_payload_for(error: BaseException) -> dict[str, object]:
            return build_error_event_payload(error, error_context, _classification_for(error))

        restate_token = _restate_ctx.set(ctx)
        try:
            if streaming_backend is not None:
                # Lifecycle events are emitted in the executor — not the workflow —
                # because the executor is the Restate retry boundary. On crash + retry,
                # Restate re-invokes execute_component directly; the workflow is still
                # blocked on the service_call future. _run_with_lifecycle_async makes
                # every attempt visible (LIFECYCLE_START / COMPLETE / ERROR).
                #
                # streaming_session_async(close_on_exit=False) installs a PipelineContext
                # without owning the run lifecycle — the workflow calls close_run().
                async with streaming_session_async(
                    workflow_id,
                    backend=streaming_backend,
                    on_journal_event=on_journal_event,
                    close_on_exit=False,
                ):
                    outputs = await _run_with_lifecycle_async(
                        component_name,
                        call_index,
                        inputs,
                        _execute_component(),
                        error_payload_factory=_error_payload_for,
                    )
            else:
                outputs = await _execute_component()
        except Exception as e:
            if restate.is_internal_exception(e):
                raise
            classification = _classification_for(e)
            logger.error(
                "Restate component error classified as {category}/{label} "
                "(terminal={terminal}, component={component_name}, workflow_id={workflow_id})",
                category=classification.category,
                label=classification.label,
                terminal=classification.terminal,
                component_name=component_name,
                workflow_id=workflow_id,
                exc_info=True,
            )
            if classification.terminal:
                if isinstance(e, restate.TerminalError):
                    raise
                raise restate.TerminalError(terminal_error_message(e, classification)) from e
            raise
        finally:
            _restate_ctx.reset(restate_token)

        # Serialize outputs for JSON transmission (TerminalError if fails). The Restate SDK encodes the handler result
        # after the handler returns, where a failure would be retried and re-run the component, so incompatible values
        # are rejected here: the same output fails the same way on every attempt.
        try:
            serialized = serialize_outputs(dict(outputs))
        except Exception as e:
            raise restate.TerminalError(f"Failed to serialize outputs for component '{component_name}': {e}") from e
        incompatible = find_json_incompatible_value(serialized, "outputs")
        if incompatible:
            raise restate.TerminalError(
                f"Failed to serialize outputs for component '{component_name}': "
                f"value at {incompatible} is not JSON serializable"
            )
        return serialized

    # no cover: stop

    # no cover: start - Restate handler tested via harness
    @executor.handler()
    async def resolve_awakeable(ctx: "restate.Context", request: dict[str, Any]) -> dict[str, Any]:
        """
        Resolve an awakeable with a payload.

        This provides a convenient way to resolve awakeables created by components
        without using the Restate admin API directly.

        Request format: {"awakeable_id": str, "payload": JSON value}.
        The envelope is validated here; the waiter owns the payload schema.
        Successful submission does not imply that the waiter accepted the payload.
        """
        if not isinstance(request, dict):
            raise restate.TerminalError("Expected an object with awakeable_id and payload", 400)
        awakeable_id = request.get("awakeable_id")

        if not isinstance(awakeable_id, str) or not awakeable_id.strip():
            raise restate.TerminalError("awakeable_id must be a non-empty string", 400)
        if "payload" not in request:
            raise restate.TerminalError("payload is required", 400)

        ctx.resolve_awakeable(awakeable_id, request["payload"])
        return {"resolved": True, "awakeable_id": awakeable_id}

    # no cover: stop

    # no cover: start - Restate handler tested via harness
    @executor.handler()
    async def reject_awakeable(ctx: "restate.Context", request: dict[str, Any]) -> dict[str, Any]:
        """
        Reject an awakeable, causing a TerminalError at the awaiter.

        This is for **system-level termination**, not business-level denial.
        The awaiting workflow/component will receive a TerminalError and stop.

        Use this for:
        - Operator manually canceling a stuck workflow
        - System cleanup or timeout at infrastructure level
        - Catastrophic errors that should abort the workflow

        Do NOT use this for normal approval/denial decisions. Instead, use
        resolve_awakeable with ToolExecutionApprovalPayload(approved=False), which raises
        ApprovalRejected that callers can catch and handle gracefully.

        Request format: {"awakeable_id": str, "failure_message": str, "failure_code": int}
        """
        awakeable_id = request.get("awakeable_id")
        failure_message = request.get("failure_message", "No message is provided")
        failure_code = request.get("failure_code", 500)

        if not awakeable_id:
            raise restate.TerminalError("awakeable_id is required")
        if not isinstance(failure_message, str):
            raise restate.TerminalError("failure_message must be a string")
        if not isinstance(failure_code, int):
            raise restate.TerminalError("failure_code must be an integer")

        ctx.reject_awakeable(awakeable_id, failure_message=failure_message, failure_code=failure_code)
        return {"rejected": True, "awakeable_id": awakeable_id}

    # no cover: stop

    # Create the pipeline orchestration Workflow
    workflow = restate.Workflow(workflow_name, invocation_retry_policy=invocation_retry_policy)

    # Store executor handler references for service calls
    pipeline._executor_handler = execute_component  # type: ignore[assignment]  # pylint: disable=protected-access
    pipeline._warm_up_handler = warm_up_component  # type: ignore[assignment]  # pylint: disable=protected-access

    # Get output components from io_handler
    include_outputs_from = io_handler.get_output_components()

    # no cover: start - Restate workflow handler tested via TestRestateIntegration harness tests
    @workflow.main()
    async def run(ctx: "restate.WorkflowContext", data: dict[str, Any]) -> dict[str, Any]:
        """
        Main workflow handler - orchestrates pipeline execution.

        Flow:
        1. Deserialize HTTP request using io_handler type hints
        2. Parse pipeline-level inputs to component-level inputs
        3. Execute pipeline with durable component calls
        4. Parse component outputs to pipeline-level outputs
        5. Serialize response for HTTP

        All IO handler operations are wrapped as TerminalError since they indicate
        client errors (bad input) or configuration errors (bad pipeline wiring).
        """
        # 1. Deserialize request using type hints (handles objects like ChatMessage)
        try:
            deserialized_data = io_handler.deserialize_request(data)
        except Exception as e:
            raise restate.TerminalError(f"Failed to deserialize request: {e}") from e

        # 1b. Validate types at pipeline level unless disabled via header (catches type errors early)
        if ctx.request().headers.get(_TYPE_VALIDATION_HEADER, "true").lower() != "false":
            try:
                validate_types(deserialized_data, io_handler.get_input_type_hints())  # type: ignore[arg-type]
            except TypeError as e:
                raise restate.TerminalError(f"Pipeline input type validation failed: {e}") from e

        # 2. Parse pipeline-level inputs to component-level inputs
        try:
            component_inputs = io_handler.parse_input(deserialized_data)
        except Exception as e:
            raise restate.TerminalError(f"Failed to parse pipeline inputs: {e}") from e

        # 3. Execute pipeline. Lifecycle events (Type 3) are emitted by the
        # executor handler, not the workflow — see execute_component above.
        # The workflow is only responsible for calling close_run() when the
        # pipeline finishes, which signals all subscriber iterators to stop.
        #
        # close_run() writes the "run over" sentinel to the event stream, so it
        # must fire only when the invocation is genuinely done — never when it
        # will keep producing events. That mirrors Restate's own retry rule: an
        # invocation ends iff it succeeds or raises TerminalError; everything else
        # (awakeable suspension, retryable crash, retry-then-pause) resumes the
        # same invocation. Writing the sentinel on those non-terminal unwinds would
        # strand it in the stream: a client reconnecting with Last-Event-ID replays
        # the cursor, hits the stale sentinel, and EOFs before the events emitted
        # after resume. So we anchor on TerminalError — the one thing Restate
        # treats as fatal — rather than trying to enumerate the continue cases.
        run_id = ctx.key()
        try:
            raw_outputs = await pipeline.run(
                ctx,
                component_inputs,
                include_outputs_from=include_outputs_from,
            )
        except BaseException as e:
            if streaming_backend is not None and isinstance(e, restate.TerminalError):
                await streaming_backend.close_run(run_id)
            raise
        # Reached only on success — the except above always re-raises.
        if streaming_backend is not None:
            await streaming_backend.close_run(run_id)

        # 4. Parse component outputs to pipeline-level outputs
        try:
            pipeline_outputs = io_handler.parse_output(raw_outputs)
        except Exception as e:
            raise restate.TerminalError(f"Failed to parse pipeline outputs: {e}") from e

        # 5. Serialize response
        try:
            return io_handler.serialize_response(pipeline_outputs)
        except Exception as e:
            raise restate.TerminalError(f"Failed to serialize response: {e}") from e

    # no cover: stop

    # no cover: start - Restate workflow handler tested via harness
    @workflow.handler()
    async def get_status(ctx: "restate.WorkflowSharedContext") -> dict[str, Any]:
        """Get current pipeline execution status."""
        return {
            "workflow_id": ctx.key(),
            "outputs": await ctx.get("pipeline_outputs", type_hint=dict) or {},
            "completed_components": await ctx.get("completed_components", type_hint=list) or [],
        }

    # no cover: stop

    return workflow, executor
