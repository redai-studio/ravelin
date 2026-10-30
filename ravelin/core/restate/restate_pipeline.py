"""
RestatePipeline - Haystack AsyncPipeline with Restate durable execution.

This module provides integration between Haystack's pipeline execution and
Restate's durable execution framework.

Architecture:
- Pipeline orchestration runs as a Workflow (durable, long-running)
- Component execution runs as a Service (durable request-response)
- Components have access to Restate context for fine-grained durability

This separation ensures:
- Pipeline-level durability: which component runs next (via service_call)
- Component-level durability: fine-grained steps within a component (opt-in via ctx.run)
- Proper Restate context access: components can use ctx.run(), ctx.awakeable(), etc.

Usage::
    from ravelin.core.restate_pipeline import RestatePipeline, create_restate_app

    pipeline = RestatePipeline()
    pipeline.add_component("retriever", retriever)
    pipeline.add_component("llm", llm)
    pipeline.connect("retriever", "llm")

    workflow, executor = create_restate_app(pipeline, "MyRAGPipeline")
    app = restate.app([workflow, executor])
"""

from typing import Any, Callable, Optional, Mapping

from haystack import logging, tracing
from haystack.core.pipeline.base import ComponentPriority, PipelineBase
from haystack.core.pipeline.utils import FIFOPriorityQueue
from haystack.core.type_utils import ConversionStrategy
from haystack.lazy_imports import LazyImport

from ravelin.core.restate.app import _TYPE_VALIDATION_HEADER
from ravelin.core.serialization import (
    deserialize_outputs,
    find_json_incompatible_value,
    serialize_inputs,
    serialize_outputs,
)


with LazyImport("Run 'pip install restate_sdk[serde]'") as restate_import:
    import restate


logger = logging.getLogger(__name__)


def _serialize_pipeline_outputs(outputs: dict[str, Any]) -> dict[str, Any]:
    """
    Serialize pipeline outputs for Restate state storage.

    Pipeline outputs have structure: {component_name: {output_key: value, ...}}
    Each component's outputs need to be serialized for JSON compatibility.

    :param outputs: Pipeline outputs dict.
    :returns: JSON-serializable dict.
    """
    return {comp_name: serialize_outputs(comp_outputs) for comp_name, comp_outputs in outputs.items()}


# Connection conversions that only restructure lists, and so work on JSON as well as on objects. Every other
# conversion receives the decoded value: the ChatMessage-to-str ones read ``.text``, the str-to-ChatMessage ones decode
# a JSON string to itself, and a conversion added by a later Haystack version gets the objects it was written for.
_STRUCTURAL_CONVERSIONS = frozenset({ConversionStrategy.WRAP, ConversionStrategy.UNWRAP})


def _outputs_for_routing(raw_outputs: dict[str, Any], receivers: Any) -> dict[str, Any]:
    """
    Prepare a component's journaled outputs for distribution to its receivers.

    Outputs are routed as the JSON the executor returned, not as decoded objects. Restate compares a journaled call
    parameter byte-for-byte on replay, and re-encoding a decoded object runs its ``to_dict()`` again, which need not
    reproduce the journaled bytes in a new process (e.g. a dict built by iterating a set). ``serialize_inputs`` leaves
    JSON unchanged, so routed values reach the next call exactly as journaled.

    A socket is decoded only where the pipeline itself may read the value: a connection conversion other than
    ``WRAP``/``UNWRAP``, and a non-list value flattened into a lazy variadic socket, which would otherwise iterate its
    encoded form. Decoded values are encoded again for the next call, so only these depend on a deterministic
    ``to_dict()``, as pipeline inputs do.

    :param raw_outputs: Component outputs as returned by the executor service.
    :param receivers: The component's receivers, as returned by ``_find_receivers_from``.
    :returns: The outputs, with only the sockets the pipeline reads decoded.
    """
    to_decode = {
        sender_socket.name
        for _, sender_socket, receiver_socket, conversion_strategy in receivers
        if sender_socket.name in raw_outputs
        and (
            (conversion_strategy is not None and conversion_strategy not in _STRUCTURAL_CONVERSIONS)
            or (
                receiver_socket.is_lazy_variadic
                and not receiver_socket.wrap_input_in_list
                and not isinstance(raw_outputs[sender_socket.name], list)
            )
        )
    }
    if not to_decode:
        return raw_outputs
    decoded = deserialize_outputs({name: raw_outputs[name] for name in to_decode})
    return {name: decoded.get(name, value) for name, value in raw_outputs.items()}


class RestatePipeline(PipelineBase):
    """
    A Haystack AsyncPipeline with Restate integration for durable execution.

    Architecture:
    - Pipeline orchestration: Workflow with service_call per component
    - Component execution: Separate Service with full ctx access
    - Component durability: Opt-in via ctx.run() within components

    This ensures:
    - Invocation-level durability: Each component call is durable (auto-retry)
    - Fine-grained durability: Components can checkpoint with ctx.run() (opt-in)
    - Awakeable support: Components can use ctx.awakeable() for human-in-the-loop

    Usage::
        pipeline = RestatePipeline()
        pipeline.add_component("retriever", retriever)
        pipeline.add_component("llm", llm)
        pipeline.connect("retriever", "llm")

        workflow, executor = create_restate_app(pipeline, "RAGPipeline")
        app = restate.app([workflow, executor])
    """

    restate_import.check()

    # Set by create_restate_app to reference the executor handlers
    _executor_handler: Optional[Callable[..., Any]] = None
    _warm_up_handler: Optional[Callable[..., Any]] = None

    def _check_max_runs(self, component_name: str, visits: int) -> None:
        """
        Enforce ``max_runs_per_component`` before dispatching a component.

        Uses the same check as Haystack's ``Pipeline`` (``visits > max_runs_per_component``), so a pipeline behaves
        the same under both: ``visits`` counts completed runs, and dispatch is refused only once that count exceeds
        the limit. A component therefore runs at most ``max_runs_per_component + 1`` times, e.g. 4 runs with a limit
        of 3, and the next dispatch fails. Each of those runs is a durable, journaled call. Raises a ``TerminalError``
        so Restate stops the workflow instead of retrying it.

        :param component_name: Name of the component about to run.
        :param visits: Number of completed runs of the component so far.
        :raises restate.TerminalError: If ``visits`` exceeds ``max_runs_per_component``.
        """
        if visits > self._max_runs_per_component:
            raise restate.TerminalError(
                f"Maximum run count {self._max_runs_per_component} reached for component '{component_name}'"
            )

    def _create_component_future(  # pragma: no cover - tested via TestRestateIntegration harness tests
        self,
        ctx: "restate.WorkflowContext",
        component_name: str,
        inputs: dict[str, Any],
        call_index: int,
    ) -> "restate.RestateDurableFuture[dict[str, Any]]":
        """
        Create a durable future for component execution via service call.

        This uses ctx.service_call() to invoke the component executor Service,
        which provides proper Restate context access for the component.

        :param ctx: Restate WorkflowContext.
        :param component_name: Name of the component to run.
        :param inputs: Inputs for the component.
        :param call_index: The current visit count for this component (0-based).
        :returns: RestateDurableFuture that resolves to component outputs.
        """
        if self._executor_handler is None:
            raise RuntimeError("Pipeline not initialized. Use create_restate_app() first.")

        # Serialize inputs for JSON transmission. Restate compares the journaled call parameter byte-for-byte on replay,
        # so the top-level keys are sorted: their order follows component socket order, which can vary between
        # processes (e.g. ChatPromptBuilder derives sockets from a set of template variables). Values keep their own
        # order, which components may depend on (e.g. scripts that build an output dict in a fixed key order).
        # Values from upstream components are routed as their journaled JSON (see _outputs_for_routing) and pass through
        # unchanged. Values still held as objects - pipeline inputs from the IO handler, socket defaults, and the few
        # outputs decoded for routing - are encoded here by their to_dict(), which must therefore give the same output
        # in every process.
        serialized_inputs = serialize_inputs(dict(sorted(inputs.items())))
        # Restate encodes the call parameter outside this code, where an encoding failure would be retried forever on
        # the same data. Reject it here as terminal instead.
        incompatible = find_json_incompatible_value(serialized_inputs, "inputs")
        if incompatible:
            raise restate.TerminalError(
                f"Failed to serialize inputs for component '{component_name}': "
                f"value at {incompatible} is not JSON serializable"
            )

        return ctx.service_call(
            self._executor_handler,
            arg={
                "component_name": component_name,
                "inputs": serialized_inputs,
                "workflow_id": ctx.key(),
                "call_index": call_index,
            },
            # Disable type validation for internal calls - already validated at pipeline level
            headers={_TYPE_VALIDATION_HEADER: "false"},
        )

    def _create_warm_up_future(  # pragma: no cover - tested via TestRestateIntegration harness tests
        self,
        ctx: "restate.WorkflowContext",
        component_name: str,
    ) -> "restate.RestateDurableFuture[dict[str, Any]]":
        if self._warm_up_handler is None:
            raise RuntimeError("Pipeline not initialized. Use create_restate_app() first.")
        return ctx.service_call(self._warm_up_handler, arg={"component_name": component_name})

    async def warm_up_executor(  # pragma: no cover - tested via TestRestateIntegration harness tests
        self, ctx: "restate.WorkflowContext"
    ) -> "list[restate.RestateDurableFuture[dict[str, Any]]]":
        """
        Make sure all nodes are warm in the executor.

        It's the node's responsibility to make sure this method can be called at every `Pipeline.run()`
        without re-initializing everything.

        :param ctx: Restate workflow context used to issue warm-up service calls.
        :returns: Warm-up results returned by the executor service.
        """
        futures = []
        for node in self.graph.nodes:
            if hasattr(self.graph.nodes[node]["instance"], "warm_up"):
                logger.info("Warming up component {node}...", node=node)
                futures.append(self._create_warm_up_future(ctx, node))
        return await restate.gather(*futures)

    async def run(  # pragma: no cover - tested via TestRestateIntegration harness tests  # pylint: disable=too-many-branches
        self,
        ctx: "restate.WorkflowContext",
        data: dict[str, Any],
        include_outputs_from: set[str] | None = None,
        concurrency_limit: int = 4,
    ) -> dict[str, Any]:
        """
        Execute the pipeline with Restate durability.

        Components are executed via service_call to the executor Service,
        providing proper Restate context access for fine-grained durability.

        :param ctx: Restate WorkflowContext.
        :param data: Input data for the pipeline.
        :param include_outputs_from: Components whose outputs to include.
        :param concurrency_limit: Maximum parallel component executions.
        :returns: Final pipeline outputs.
        """
        if include_outputs_from is None:
            include_outputs_from = set()

        # Initialize pipeline
        # pipeline_running(self)
        await self.warm_up_executor(ctx)

        # Prepare state
        inputs_state: dict[str, dict[str, list[dict[str, Any]]]] = {}
        pipeline_outputs: dict[str, Any] = {}
        completed_components: list[str] = []

        # Convert and validate input (TerminalError if validation fails)
        try:
            prepared_data = self._prepare_component_input_data(data)
            self.validate_input(prepared_data)
            inputs_state = self._convert_to_internal_format(prepared_data)
        except restate.TerminalError:
            raise
        except Exception as e:
            raise restate.TerminalError(f"Pipeline input validation failed: {e}") from e

        # Setup component tracking
        ordered_names = sorted(self.graph.nodes.keys())
        cached_receivers = {n: self._find_receivers_from(n) for n in ordered_names}
        component_visits = dict.fromkeys(ordered_names, 0)

        # Validate pipeline can run (TerminalError if graph misconfigured)
        try:
            self.validate_pipeline(self._fill_queue(ordered_names, inputs_state, component_visits))
        except restate.TerminalError:
            raise
        except Exception as e:
            raise restate.TerminalError(f"Pipeline graph validation failed: {e}") from e

        with tracing.tracer.trace(
            "haystack.restate_pipeline.run",
            tags={
                "haystack.pipeline.input_data": prepared_data,
                "haystack.pipeline.metadata": self.metadata,
                "haystack.pipeline.max_runs_per_component": self._max_runs_per_component,
                "restate.workflow_id": ctx.key(),
            },
        ):
            # Cache for topological sort (used in DEFER tiebreaking)
            cached_topological_sort = None

            # Track running component futures for parallel execution
            running_futures: dict[str, restate.RestateDurableFuture[dict[str, Any]]] = {}

            while True:
                priority_queue = self._fill_queue(ordered_names, inputs_state, component_visits)

                # Collect components by priority
                highest_components: list[str] = []
                ready_components: list[str] = []
                defer_components: list[tuple[str, ComponentPriority]] = []
                blocked_components: list[tuple[str, dict]] = []

                while len(priority_queue) > 0:
                    priority, comp_name = priority_queue.pop()
                    if priority == ComponentPriority.HIGHEST:
                        highest_components.append(comp_name)
                    elif priority == ComponentPriority.READY:
                        ready_components.append(comp_name)
                    elif priority in (ComponentPriority.DEFER, ComponentPriority.DEFER_LAST):
                        defer_components.append((comp_name, ComponentPriority(priority)))
                    elif priority == ComponentPriority.BLOCKED:
                        comp_dict = self.graph.nodes[comp_name]
                        blocked_components.append((comp_name, comp_dict))

                # 1. Run HIGHEST priority components in isolation
                if highest_components and running_futures:
                    # Drain all running futures before running HIGHEST
                    futures_list = list(running_futures.items())
                    # restate.gather returns futures in the same order they were passed (not completion order)
                    gathered = await restate.gather(*[f for _, f in futures_list])
                    for (comp_name, _), future_result in zip(futures_list, gathered):
                        raw_outputs = await future_result
                        outputs = _outputs_for_routing(raw_outputs, cached_receivers[comp_name])
                        component_visits[comp_name] += 1

                        pruned = self._write_component_outputs(
                            component_name=comp_name,
                            component_outputs=outputs,
                            inputs=inputs_state,
                            receivers=cached_receivers[comp_name],
                            include_outputs_from=include_outputs_from,
                        )
                        if pruned or comp_name in include_outputs_from:
                            pipeline_outputs[comp_name] = pruned

                        completed_components.append(comp_name)

                    running_futures.clear()
                    ctx.set("completed_components", completed_components)
                    # Serialize outputs for Restate state storage (JSON-compatible)
                    ctx.set("pipeline_outputs", _serialize_pipeline_outputs(pipeline_outputs))

                for comp_name in highest_components:
                    await self._execute_component_via_service(
                        ctx=ctx,
                        component_name=comp_name,
                        inputs_state=inputs_state,
                        component_visits=component_visits,
                        cached_receivers=cached_receivers,
                        include_outputs_from=include_outputs_from,
                        pipeline_outputs=pipeline_outputs,
                    )
                    completed_components.append(comp_name)

                if highest_components:
                    ctx.set("completed_components", completed_components)
                    ctx.set("pipeline_outputs", _serialize_pipeline_outputs(pipeline_outputs))
                    continue

                # 2. Run READY components with first-completed parallelism
                if ready_components or running_futures:
                    # Schedule READY components up to concurrency limit
                    while ready_components and len(running_futures) < concurrency_limit:
                        comp_name = ready_components.pop(0)
                        self._check_max_runs(comp_name, component_visits[comp_name])
                        comp_dict = self._get_component_with_graph_metadata_and_visits(
                            comp_name, component_visits[comp_name]
                        )
                        component_inputs = self._consume_component_inputs(comp_name, comp_dict, inputs_state)
                        component_inputs = self._add_missing_input_defaults(
                            component_inputs, comp_dict["input_sockets"]
                        )

                        future = self._create_component_future(
                            ctx, comp_name, component_inputs, component_visits[comp_name]
                        )
                        running_futures[comp_name] = future

                    if running_futures:
                        # Wait for first completion using restate.select()
                        completed_name, raw_outputs = await restate.select(**running_futures)
                        del running_futures[completed_name]

                        outputs = _outputs_for_routing(raw_outputs, cached_receivers[completed_name])
                        component_visits[completed_name] += 1

                        pruned = self._write_component_outputs(
                            component_name=completed_name,
                            component_outputs=outputs,
                            inputs=inputs_state,
                            receivers=cached_receivers[completed_name],
                            include_outputs_from=include_outputs_from,
                        )
                        if pruned or completed_name in include_outputs_from:
                            pipeline_outputs[completed_name] = pruned

                        completed_components.append(completed_name)
                        ctx.set("completed_components", completed_components)
                        ctx.set("pipeline_outputs", _serialize_pipeline_outputs(pipeline_outputs))

                    continue

                # 3. Run DEFER components when nothing else is ready
                if defer_components:
                    comp_name, priority = defer_components[0]

                    if len(defer_components) > 1:
                        temp_queue = FIFOPriorityQueue()
                        for name, prio in defer_components[1:]:
                            temp_queue.push(name, prio)

                        comp_name, cached_topological_sort = self._tiebreak_waiting_components(
                            component_name=comp_name,
                            priority=priority,
                            priority_queue=temp_queue,
                            topological_sort=cached_topological_sort,
                        )

                    await self._execute_component_via_service(
                        ctx=ctx,
                        component_name=comp_name,
                        inputs_state=inputs_state,
                        component_visits=component_visits,
                        cached_receivers=cached_receivers,
                        include_outputs_from=include_outputs_from,
                        pipeline_outputs=pipeline_outputs,
                    )
                    completed_components.append(comp_name)

                    ctx.set("completed_components", completed_components)
                    ctx.set("pipeline_outputs", _serialize_pipeline_outputs(pipeline_outputs))
                    continue

                # 4. Nothing runnable - check if pipeline is blocked
                if blocked_components:
                    if self._is_pipeline_possibly_blocked(current_pipeline_outputs=pipeline_outputs):
                        blocked_name, blocked_comp = blocked_components[0]
                        logger.warning(
                            "Cannot run pipeline - component is blocked.\n"
                            "Component name: '{component_name}'\n"
                            "Component type: '{component_type}'\n"
                            "Check connections and ensure all required inputs are provided.",
                            component_name=blocked_name,
                            component_type=blocked_comp["instance"].__class__.__name__,
                        )

                break

            ctx.set("completed_components", completed_components)
            ctx.set("pipeline_outputs", _serialize_pipeline_outputs(pipeline_outputs))

        # Outputs were routed as journaled JSON; callers receive decoded objects.
        return {name: deserialize_outputs(outputs) for name, outputs in pipeline_outputs.items()}

    async def _execute_component_via_service(  # pragma: no cover - tested via TestRestateIntegration harness tests
        self,
        ctx: "restate.WorkflowContext",
        component_name: str,
        inputs_state: dict[str, dict[str, list[dict[str, Any]]]],
        component_visits: dict[str, int],
        cached_receivers: dict[str, Any],
        include_outputs_from: set[str],
        pipeline_outputs: dict[str, Any],
    ) -> Mapping[str, Any]:
        """Prepare inputs, execute component via service call, and distribute outputs."""
        self._check_max_runs(component_name, component_visits[component_name])
        comp_dict = self._get_component_with_graph_metadata_and_visits(component_name, component_visits[component_name])
        component_inputs = self._consume_component_inputs(component_name, comp_dict, inputs_state)
        component_inputs = self._add_missing_input_defaults(component_inputs, comp_dict["input_sockets"])

        future = self._create_component_future(ctx, component_name, component_inputs, component_visits[component_name])
        raw_outputs = await future
        outputs = _outputs_for_routing(raw_outputs, cached_receivers[component_name])

        component_visits[component_name] += 1

        pruned = self._write_component_outputs(
            component_name=component_name,
            component_outputs=outputs,
            inputs=inputs_state,
            receivers=cached_receivers[component_name],
            include_outputs_from=include_outputs_from,
        )
        if pruned or component_name in include_outputs_from:
            pipeline_outputs[component_name] = pruned

        return pruned
