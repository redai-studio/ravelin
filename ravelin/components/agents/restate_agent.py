"""
RestateAgent - A durable agent powered by Restate.

RestateAgent records LLM-call and tool-execution results through Restate durable
steps. During recovery, recorded results are reused to reconstruct the agent's
execution. Operations without a durably recorded result may run again, so
external side effects require appropriate idempotency or reconciliation. The
Restate execution journal supports recovery; applications remain responsible for
their durable conversation history.

Key features:
- Each LLM call result is recorded as a Restate durable step
- Each tool execution result is recorded as a durable step (via RestateToolInvoker)
- Human-in-the-loop approval via confirmation_strategies + Restate awakeables
- State support for tool state injection and output merging
- Automatic retry for transient failures
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union, cast

from haystack import component, logging
from haystack.components.agents.agent import _ExecutionContext
from haystack.components.agents.state.state import State, _validate_schema
from haystack.components.agents.state.state_utils import merge_lists, replace_values
from haystack.core.pipeline.utils import _deepcopy_with_exceptions
from haystack.human_in_the_loop.strategies import _process_confirmation_strategies_async
from haystack.human_in_the_loop.types import ConfirmationStrategy
from haystack.lazy_imports import LazyImport
from haystack.tools import Tool, Toolset, ToolsType, flatten_tools_or_toolsets

from ravelin.components.generators import ChatGenerator
from ravelin.components.tools.restate_tool_invoker import RestateToolInvoker
from ravelin.core.restate.context import get_restate_context
from ravelin.streaming import EventType, get_component_context
from ravelin.dataclasses import ChatMessage, ChatRole

with LazyImport("Run 'pip install restate_sdk[serde]'") as restate_import:
    from restate.context import RunOptions

logger = logging.getLogger(__name__)


class AgentComponentEvent(str, Enum):
    """COMPONENT_ACTIVITY subtypes emitted by RestateAgent during execution."""

    LLM_RESPONSE = "llm_response"
    TOOL_CALLS_PENDING = "tool_calls_pending"
    TOOL_RESULTS = "tool_results"


@component
class RestateAgent:
    """
    A Restate-powered agent with durable execution.

    This component provides the same agent functionality as the regular Agent,
    with Restate-backed recovery:

    - **Durable LLM steps**: Each chat generator result is recorded in the Restate journal
    - **Durable tool steps**: Each tool invocation result is recorded (via RestateToolInvoker)
    - **Human approval**: Optional confirmation strategies with Restate awakeables
    - **Replay-based recovery**: After a crash, execution is reconstructed from recorded
      step results. A call whose result was not recorded may run again, so external
      side effects need idempotency or reconciliation.
    - **State support**: Tool state injection and output merging

    Unlike the regular Agent, there's no need for breakpoints or snapshots for
    recovery. The Restate journal is not a conversation store; persist durable
    conversation history in application-owned storage.

    ### Usage example
    ```python
    from ravelin.components.agents import RestateAgent
    from ravelin.components.generators import ChatGenerator
    from ravelin.dataclasses import ChatMessage
    from haystack.tools import Tool

    # Define tools
    tools = [
        Tool(name="search", description="Search the web", ...),
        Tool(name="calculator", description="Do math", ...),
    ]

    # Create agent with optional human approval
    agent = RestateAgent(
        chat_generator=ChatGenerator(...),
        tools=tools,
        confirmation_strategies={"search": RestateConfirmationStrategy(...)},
    )

    # Run within a RestatePipeline
    pipeline = RestatePipeline()
    pipeline.add_component("agent", agent)
    # ... configure and run
    ```
    """

    restate_import.check()

    def __init__(
        self,
        *,
        chat_generator: ChatGenerator,
        tools: Optional[Union[List[Tool], Toolset]] = None,
        system_prompt: Optional[str] = None,
        exit_conditions: Optional[List[str]] = None,
        state_schema: Optional[Dict[str, Any]] = None,
        max_agent_steps: int = 100,
        raise_on_tool_invocation_failure: bool = False,
        tool_invoker_kwargs: Optional[Dict[str, Any]] = None,
        confirmation_strategies: Dict[str | tuple[str, ...], ConfirmationStrategy] | None = None,
    ) -> None:
        """
        Initialize the RestateAgent.

        :param chat_generator: The chat generator component for LLM calls.
        :param tools: List of Tool objects or a Toolset that the agent can use.
        :param system_prompt: Optional system prompt prepended to messages.
        :param exit_conditions: Tool names that trigger agent exit. Defaults to ["text"].
        :param state_schema: The schema for the runtime state used by the tools.
        :param max_agent_steps: Maximum agent iterations before stopping.
        :param raise_on_tool_invocation_failure: If True, raise tool failures so the outer Restate
            execution boundary can classify and retry/pause them.
        :param tool_invoker_kwargs: Additional keyword arguments to pass to the RestateToolInvoker.
        :param confirmation_strategies: Optional dict mapping tool names (or tuples of names) to
            ConfirmationStrategy instances for human-in-the-loop approval.
        """
        valid_exits = ["text"] + [tool.name for tool in tools or []]
        if exit_conditions is None:
            exit_conditions = ["text"]
        if not all(condition in valid_exits for condition in exit_conditions):
            raise ValueError(
                f"Invalid exit conditions provided: {exit_conditions}. "
                f"Valid exit conditions must be a subset of {valid_exits}. "
                "Ensure that each exit condition corresponds to either 'text' or a valid tool name."
            )

        # Validate state schema if provided
        if state_schema is not None:
            _validate_schema(state_schema)
        self._state_schema = state_schema or {}

        # Initialize state schema (with messages merge handler)
        resolved_state_schema = _deepcopy_with_exceptions(self._state_schema)
        if resolved_state_schema.get("messages") is None:
            resolved_state_schema["messages"] = {"type": list[ChatMessage], "handler": merge_lists}
        self.state_schema = resolved_state_schema

        self.chat_generator = chat_generator
        self.tools = tools or []
        self.system_prompt = system_prompt
        self.exit_conditions = exit_conditions
        self.max_agent_steps = max_agent_steps
        self.raise_on_tool_invocation_failure = raise_on_tool_invocation_failure
        self._confirmation_strategies = confirmation_strategies or {}

        # Set up output types (same as Agent)
        output_types = {
            "last_message": ChatMessage,
            "generated_messages": List[ChatMessage],
        }
        for param, config in self.state_schema.items():
            output_types[param] = config["type"]
            if param in ["messages"]:
                continue
            component.set_input_type(self, name=param, type=config["type"], default=None)
        component.set_output_types(self, **output_types)

        # Create RestateToolInvoker (like Agent creates ToolInvoker)
        self.tool_invoker_kwargs = tool_invoker_kwargs
        self._tool_invoker = None
        if self.tools:
            resolved_tool_invoker_kwargs = {
                "tools": self.tools,
                "raise_on_failure": self.raise_on_tool_invocation_failure,
                "combine_tool_results_to_single_message": (
                    self.chat_generator.model_service.inference_api_format not in {"openai-chat", "openai-response"}
                ),
                **(tool_invoker_kwargs or {}),
            }
            self._tool_invoker = RestateToolInvoker(**resolved_tool_invoker_kwargs)
        else:
            logger.warning(
                "No tools provided to the RestateAgent. The agent will behave like a ChatGenerator and only return "
                "text responses. To enable tool usage, pass tools directly to the Agent, not to the chat_generator."
            )

        self._is_warmed_up = False

    def warm_up(self) -> None:
        """Warm up the agent (delegates to chat generator)."""
        if not self._is_warmed_up:
            if hasattr(self.chat_generator, "warm_up"):
                self.chat_generator.warm_up()
            self._is_warmed_up = True

    def _select_tools(self, tools: ToolsType | list[str] | None = None) -> ToolsType:
        """
        Select tools for the current run based on the provided tools parameter. This method is identical to
        Agent._select_tools.

        :param tools: Optional list of Tool objects, a Toolset, or list of tool names to use for this run.
            When passing tool names, tools are selected from the Agent's originally configured tools.
        :returns: Selected tools for the current run.
        :raises ValueError: If tool names are provided but no tools were configured at initialization,
            or if any provided tool name is not valid.
        :raises TypeError: If tools is not a list of Tool objects, a Toolset, or a list of tool names (strings).
        """
        if tools is None:
            return self.tools

        if isinstance(tools, list) and all(isinstance(t, str) for t in tools):
            if not self.tools:
                raise ValueError("No tools were configured for the RestateAgent at initialization.")
            available_tools = flatten_tools_or_toolsets(self.tools)
            selected_tool_names = cast(list[str], tools)
            valid_tool_names = {tool.name for tool in available_tools}
            invalid_tool_names = {name for name in selected_tool_names if name not in valid_tool_names}
            if invalid_tool_names:
                raise ValueError(
                    f"The following tool names are not valid: {invalid_tool_names}. "
                    f"Valid tool names are: {valid_tool_names}."
                )
            return [tool for tool in available_tools if tool.name in selected_tool_names]

        if isinstance(tools, Toolset):
            return tools

        if isinstance(tools, list):
            return cast(list[Tool | Toolset], tools)

        raise TypeError(
            "tools must be a list of Tool and/or Toolset objects, a Toolset, or a list of tool names (strings)."
        )

    def _initialize_fresh_execution(
        self,
        messages: list[ChatMessage],
        *,
        system_prompt: str | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        tools: ToolsType | list[str] | None = None,
        confirmation_strategy_context: dict[str, Any] | None = None,
        **kwargs,
    ) -> _ExecutionContext:
        """
        Initialize execution context for a fresh run. Mirrors Agent._initialize_fresh_execution.

        :param messages: List of ChatMessage objects to start the agent with.
        :param system_prompt: System prompt override.
        :param generation_kwargs: Additional keyword arguments for chat generator.
        :param tools: Optional runtime tool selection.
        :param confirmation_strategy_context: Optional dictionary for passing request-scoped resources
            to confirmation strategies.
        :param kwargs: Additional data to pass to the State used by the Agent.
        """
        system_prompt = system_prompt or self.system_prompt
        if system_prompt is not None:
            messages = [ChatMessage.from_system(system_prompt)] + messages

        if all(m.is_from(ChatRole.SYSTEM) for m in messages):
            logger.warning("All messages provided to the RestateAgent are system messages. This is not recommended.")

        state = State(schema=self.state_schema, data=kwargs)
        state.set("messages", messages)

        selected_tools = self._select_tools(tools)
        tool_invoker_inputs: dict[str, Any] = {"tools": selected_tools}
        generator_inputs: dict[str, Any] = {"tools": selected_tools}
        if generation_kwargs is not None:
            generator_inputs["generation_kwargs"] = generation_kwargs

        return _ExecutionContext(
            state=state,
            component_visits=dict.fromkeys(["chat_generator", "tool_invoker"], 0),
            chat_generator_inputs=generator_inputs,
            tool_invoker_inputs=tool_invoker_inputs,
            confirmation_strategy_context=confirmation_strategy_context,
        )

    async def run_async(  # noqa: PLR0915
        self,
        messages: List[ChatMessage],
        *,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[Any] = None,
        confirmation_strategy_context: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Run the agent with durable execution.

        Each step result (LLM call, tool execution) is recorded through Restate's
        journal. After a crash, recovery replays the run and reuses recorded
        results; a step whose result was not recorded may run again.

        :param messages: Initial conversation messages.
        :param generation_kwargs: Optional runtime LLM parameter overrides.
        :param system_prompt: Optional runtime system prompt override.
        :param tools: Optional runtime tool selection — list of Tool objects or tool name strings.
        :param confirmation_strategy_context: Optional request-scoped context for confirmation strategies.
        :param kwargs: Additional data to pass to the State schema used by the Agent.
        :returns: Dict with 'messages', 'last_message', 'generated_messages',
            and any state schema outputs.
        """
        ctx = get_restate_context()
        comp_ctx = get_component_context()

        exe_context = self._initialize_fresh_execution(
            messages=messages,
            system_prompt=system_prompt,
            generation_kwargs=generation_kwargs,
            tools=tools,
            confirmation_strategy_context=confirmation_strategy_context,
            **kwargs,
        )
        initial_message_count = len(messages)

        while exe_context.counter < self.max_agent_steps:
            # 1. Call LLM (durable via ctx.run_typed)
            # Wrap to return List[ChatMessage] directly so RunOptions type_hint
            # guides deserialization on journal replay without needing __type__.
            llm_inputs = {
                "messages": exe_context.state.data["messages"],
                **exe_context.chat_generator_inputs,
            }

            async def _llm_call() -> List[ChatMessage]:
                result = await self.chat_generator.run_async(**llm_inputs)
                return result["replies"]

            llm_messages = await ctx.run_typed(
                f"llm_call_step_{exe_context.counter}",
                _llm_call,
                RunOptions(type_hint=List[ChatMessage]),
            )
            exe_context.state.set("messages", llm_messages)
            if comp_ctx:
                await comp_ctx.emit_async(
                    EventType.COMPONENT_ACTIVITY, llm_messages, subtype=AgentComponentEvent.LLM_RESPONSE
                )

            # 2. Check if any LLM responses contain tool calls
            if not any(msg.tool_call for msg in llm_messages) or self._tool_invoker is None:
                exe_context.counter += 1
                break

            if comp_ctx:
                await comp_ctx.emit_async(
                    EventType.COMPONENT_ACTIVITY, llm_messages, subtype=AgentComponentEvent.TOOL_CALLS_PENDING
                )

            # 3. Process confirmation strategies (short-circuits if empty)
            modified_tool_call_messages, new_chat_history = await _process_confirmation_strategies_async(
                confirmation_strategies=self._confirmation_strategies,
                messages_with_tool_calls=llm_messages,  # type: ignore[arg-type]
                execution_context=exe_context,
            )
            # Haystack strategy functions (e.g. _apply_tool_execution_decisions) create parent-class
            # ChatMessage instances. Convert back to our subclass so downstream code (google.py, etc.)
            # can access our .content property without hitting Haystack's deprecation guard.
            modified_tool_call_messages = [
                m if isinstance(m, ChatMessage) else ChatMessage.from_haystack_message(m)
                for m in modified_tool_call_messages
            ]
            new_chat_history = [
                m if isinstance(m, ChatMessage) else ChatMessage.from_haystack_message(m) for m in new_chat_history
            ]
            exe_context.state.set(key="messages", value=new_chat_history, handler_override=replace_values)

            # 4. Execute tools via RestateToolInvoker (handles per-call durability internally)
            tool_invoker_result = await self._tool_invoker.run_async(
                messages=modified_tool_call_messages,  # type: ignore[arg-type]
                state=exe_context.state,
                **exe_context.tool_invoker_inputs,
            )
            tool_messages = tool_invoker_result["tool_messages"]
            exe_context.state = tool_invoker_result["state"]
            exe_context.state.set("messages", tool_messages)
            if comp_ctx:
                await comp_ctx.emit_async(
                    EventType.COMPONENT_ACTIVITY, tool_messages, subtype=AgentComponentEvent.TOOL_RESULTS
                )

            # 5. Check exit conditions
            if self.exit_conditions != ["text"] and self._check_exit_conditions(llm_messages, tool_messages):
                exe_context.counter += 1
                break

            # Increment the step counter
            exe_context.counter += 1

        if exe_context.counter >= self.max_agent_steps:
            logger.warning(
                "RestateAgent reached max_agent_steps ({max_agent_steps}), stopping.",
                max_agent_steps=self.max_agent_steps,
            )

        # Build result (same format as Agent)
        result = {**exe_context.state.data}
        if msgs := result.get("messages"):
            result["last_message"] = msgs[-1]
            generated_messages = msgs[initial_message_count:] if len(msgs) > initial_message_count else []
            result.update({"generated_messages": generated_messages})
        return result

    def _check_exit_conditions(self, llm_messages: list[ChatMessage], tool_messages: list[ChatMessage]) -> bool:
        """
        Check if any tool call matches an exit condition.

        Only exits if the matching tool executed without error.

        :param llm_messages: Messages from the LLM containing tool calls.
        :param tool_messages: Messages from the tool invoker with results.
        :returns: True if an exit condition was matched without error.
        """
        matched_exit_conditions = set()
        has_errors = False

        for msg in llm_messages:
            if msg.tool_call and msg.tool_call.tool_name in self.exit_conditions:
                matched_exit_conditions.add(msg.tool_call.tool_name)

                tool_errors = [
                    tool_msg.tool_call_result.error
                    for tool_msg in tool_messages
                    if tool_msg.tool_call_result is not None
                    and tool_msg.tool_call_result.origin.tool_name == msg.tool_call.tool_name
                ]
                if any(tool_errors):
                    has_errors = True
                    break

        return bool(matched_exit_conditions) and not has_errors

    def run(
        self,
        messages: List[ChatMessage],
        *,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[Any] = None,
        confirmation_strategy_context: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Sync run is not supported - RestateAgent requires async execution.

        :raises RuntimeError: Always, as sync execution is not supported.
        """
        raise RuntimeError(
            "RestateAgent requires async execution within a RestatePipeline. "
            "Use run_async() or run within a RestatePipeline."
        )
