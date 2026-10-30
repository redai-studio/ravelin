"""
Agent component — extends Haystack's Agent with Ravelin customizations.

All agent loop logic (run, run_async, confirmation strategies, breakpoints, snapshots,
state management, exit conditions) is inherited unchanged from Haystack's Agent.

Customizations:
- Uses ravelin ChatGenerator and ToolInvoker
- Streaming via ComponentContext.emit() / emit_sync() (no per-call streaming_callbacks)
- Automatically sets combine_tool_results_to_single_message from model format
- Returns generated_messages (messages added during this run) in output

Streaming limitation — component activities events:
    This Agent only emits token stream (LLM token stream, via the model service) events. It does
    NOT emit component-activity events such as `llm_complete`, `tool_calls_pending`, or
    `tool_results` for each agent loop iteration.

    The reason is structural: the entire agent loop lives inside Haystack's Agent base class,
    which this component wraps with a thin override. Hooking into intermediate loop states
    (after each LLM call, after each tool invocation) would require overriding or duplicating
    Haystack internals, which conflicts with the goal of inheriting loop upgrades automatically.

    If per-step component-activity events are needed, use RestateAgent instead — its loop is fully
    owned by this codebase and emits `llm_complete`, `tool_calls_pending`, and `tool_results`
    at each durable step boundary.

    Future work: explore upstream Haystack hook points (e.g., overridable step callbacks)
    that could enable component event emission without duplicating loop logic.

Note: @component is NOT re-applied to avoid breaking super() via types.new_class().
The class inherits ComponentMeta from the parent and is manually registered below.
"""

from typing import Any, Dict, List, Optional, Union

from haystack import component, logging
from haystack.components.agents.agent import Agent as _HaystackAgent
from haystack.human_in_the_loop.types import ConfirmationStrategy
from haystack.tools import Tool, Toolset

from ravelin.components.generators import ChatGenerator
from ravelin.components.tools import ToolInvoker
from ravelin.dataclasses import ChatMessage

logger = logging.getLogger(__name__)


class Agent(_HaystackAgent):
    """
    Ravelin Agent extending Haystack's Agent.

    Adds streaming via ComponentContext.emit(), automatic `combine_tool_results_to_single_message`
    based on model format, and a `generated_messages` output listing messages added during this run.

    All other behaviour — agent loop, confirmation strategies, breakpoints, snapshots, state
    management — is inherited from Haystack's Agent and upgrades automatically with Haystack.

    ### Usage example
    ```python
    from ravelin.components.agents import Agent
    from ravelin.components.generators import ChatGenerator
    from ravelin.dataclasses import ChatMessage

    agent = Agent(
        chat_generator=ChatGenerator(...),
        tools=[...],
        exit_conditions=["my_tool"],
    )
    agent.warm_up()
    result = agent.run([ChatMessage.from_user("Hello")])
    # result["generated_messages"] — messages added during this run
    ```
    """

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
        confirmation_strategies: Optional[Dict[str | tuple[str, ...], ConfirmationStrategy]] = None,
    ) -> None:
        super().__init__(
            chat_generator=chat_generator,  # type: ignore[arg-type]
            tools=tools,
            system_prompt=system_prompt,
            exit_conditions=exit_conditions,
            state_schema=state_schema,
            max_agent_steps=max_agent_steps,
            streaming_callback=None,  # No per-call streaming override; use ComponentContext.emit()
            raise_on_tool_invocation_failure=raise_on_tool_invocation_failure,
            tool_invoker_kwargs=tool_invoker_kwargs,
            confirmation_strategies=confirmation_strategies,
        )

        # Extend parent's output types to include generated_messages
        existing = {name: sock.type for name, sock in self.__haystack_output__._sockets_dict.items()}  # type: ignore[attr-defined]  # pylint: disable=no-member
        component.set_output_types(self, **existing, generated_messages=List[ChatMessage])

        # Replace Haystack's ToolInvoker with our extended version
        if self.tools:
            combine = hasattr(
                chat_generator, "model_service"
            ) and chat_generator.model_service.inference_api_format not in {"openai-chat", "openai-response"}
            self._tool_invoker = ToolInvoker(  # type: ignore[assignment]
                tools=self.tools,  # type: ignore[arg-type]
                raise_on_failure=self.raise_on_tool_invocation_failure,
                combine_tool_results_to_single_message=combine,
                **(tool_invoker_kwargs or {}),
            )

    def _initialize_fresh_execution(self, *args: Any, **kwargs: Any) -> Any:
        # Haystack reads self._tool_invoker.enable_streaming_callback_passthrough to decide
        # whether to forward a runtime streaming_callback into tools. We don't use runtime
        # passthrough (init-time streaming_callbacks only), so we expose False and strip the
        # key before it reaches our ToolInvoker.run() which doesn't accept it.
        if self._tool_invoker is not None:
            self._tool_invoker.enable_streaming_callback_passthrough = False
        exe_context = super()._initialize_fresh_execution(*args, **kwargs)
        exe_context.tool_invoker_inputs.pop("enable_streaming_callback_passthrough", None)
        return exe_context

    def _initialize_from_snapshot(self, *args: Any, **kwargs: Any) -> Any:
        if self._tool_invoker is not None:
            self._tool_invoker.enable_streaming_callback_passthrough = False
        exe_context = super()._initialize_from_snapshot(*args, **kwargs)
        exe_context.tool_invoker_inputs.pop("enable_streaming_callback_passthrough", None)
        return exe_context

    def run(  # type: ignore[override]  # pylint: disable=arguments-differ
        self,
        messages: List[ChatMessage],
        *,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        effective_prompt = system_prompt or self.system_prompt
        initial_count = len(messages) + (1 if effective_prompt is not None else 0)
        result = super().run(messages, system_prompt=system_prompt, **kwargs)  # type: ignore[arg-type]
        msgs = result.get("messages", [])
        result["generated_messages"] = msgs[initial_count:] if len(msgs) > initial_count else []
        return result

    async def run_async(  # type: ignore[override]  # pylint: disable=arguments-differ
        self,
        messages: List[ChatMessage],
        *,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        effective_prompt = system_prompt or self.system_prompt
        initial_count = len(messages) + (1 if effective_prompt is not None else 0)
        result = await super().run_async(messages, system_prompt=system_prompt, **kwargs)  # type: ignore[arg-type]
        msgs = result.get("messages", [])
        result["generated_messages"] = msgs[initial_count:] if len(msgs) > initial_count else []
        return result

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        params = d["init_parameters"]
        # Haystack serializes streaming_callback (singular); remove it — we don't use it
        params.pop("streaming_callback", None)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Agent":
        init_params = data.get("init_parameters", {})
        # Remove singular form (Haystack's or old serializations) — our __init__ doesn't accept it
        init_params.pop("streaming_callback", None)
        # Remove deprecated streaming_callbacks if present in old serializations
        init_params.pop("streaming_callbacks", None)
        return super().from_dict(data)  # type: ignore[return-value]


# Register in Haystack's component registry for pipeline serialization/deserialization.
# @component is intentionally NOT applied — it uses types.new_class() which breaks super() in subclasses.
# The class inherits ComponentMeta from _HaystackAgent, so sockets and pipeline integration work correctly.
component.registry["ravelin.components.agents.agent.Agent"] = Agent
