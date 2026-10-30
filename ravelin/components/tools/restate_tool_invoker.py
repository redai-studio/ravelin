"""
Restate-durable tool invoker component.

Provides the same tool invocation interface as ToolInvoker but replaces the
ThreadPoolExecutor execution with ctx.run_typed() for per-call durability
and restate.gather() for parallel execution.

Uses composition: holds a HaystackToolInvoker instance internally for
preparation and result formatting, only replacing the execution mechanism.
"""

from typing import Any, Dict, List, Optional

from haystack import component, logging
from haystack.components.agents.state import State
from haystack.components.tools.tool_invoker import ToolInvoker as HaystackToolInvoker, ToolOutputMergeError
from haystack.lazy_imports import LazyImport
from haystack.tools import ToolsType

from ravelin.core.restate.context import get_restate_context
from ravelin.dataclasses import ChatMessage, ChatRole

with LazyImport("Run 'pip install restate_sdk[serde]'") as restate_import:
    import restate

logger = logging.getLogger(__name__)


@component
class RestateToolInvoker:
    """
    A tool invoker with Restate durable execution.

    Like ``ToolInvoker``, but replaces ThreadPoolExecutor with ``ctx.run_typed()``
    for per-call durability and ``restate.gather()`` for parallel execution.
    Gets the Restate context internally via ``get_restate_context()``.

    Holds a ``HaystackToolInvoker`` instance internally for preparation (tool
    validation, state injection, tool-not-found handling) and result formatting
    (output processing, state merging). Only the execution mechanism differs.

    This component is used internally by ``RestateAgent`` and should not normally
    be instantiated directly by users.
    """

    def __init__(  # pylint: disable=unused-argument
        self,
        tools: ToolsType,
        raise_on_failure: bool = True,
        convert_result_to_json_string: bool = False,
        combine_tool_results_to_single_message: bool = False,
        **kwargs: Any,
    ):
        """
        Initialize the RestateToolInvoker.

        :param tools: A list of tools that can be invoked or a Toolset instance.
        :param raise_on_failure: If True, raise the original tool failure so the outer Restate
            execution boundary can classify it.
        :param convert_result_to_json_string: If True, tool results are converted to a JSON string.
        :param combine_tool_results_to_single_message: If True, tool results are combined into a
            single message (to comply with Claude & Gemini protocol instead of OpenAI).
        :param kwargs: Additional keyword arguments (accepted and ignored for compatibility
            with ToolInvoker-specific params like max_workers, streaming_callback, etc.).
        """
        restate_import.check()
        self.tools = tools
        self.raise_on_failure = raise_on_failure
        self.convert_result_to_json_string = convert_result_to_json_string
        self.combine_tool_results_to_single_message = combine_tool_results_to_single_message

        # Hold a HaystackToolInvoker for its logic methods (preparation + result formatting).
        # We never call its run()/run_async() — only its internal methods.
        self._haystack_invoker = HaystackToolInvoker(
            tools=self.tools,
            raise_on_failure=self.raise_on_failure,
            convert_result_to_json_string=self.convert_result_to_json_string,
        )

    def _combine_tool_results(self, tool_result_msgs: List[ChatMessage]) -> ChatMessage:
        """
        Combine tool result messages into a single message.

        :param tool_result_msgs: A list of tool result messages.
        :returns: A single tool result message with the combined results.
        """
        tool_results = []
        for tool_result_msg in tool_result_msgs:
            tool_results.extend(tool_result_msg.tool_call_results)
        return ChatMessage(_role=ChatRole.TOOL, _content=tool_results)

    @component.output_types(tool_messages=List[ChatMessage], state=State)
    def run(
        self,
        messages: List[ChatMessage],
        state: Optional[State] = None,
        *,
        tools: ToolsType | None = None,
    ) -> Dict[str, Any]:
        """
        Sync run is not supported — RestateToolInvoker requires async execution.

        :raises RuntimeError: Always, as sync execution is not supported.
        """
        raise RuntimeError("RestateToolInvoker requires async execution within Restate. Use run_async() instead.")

    @component.output_types(tool_messages=List[ChatMessage], state=State)
    async def run_async(
        self,
        messages: List[ChatMessage],
        state: State | None = None,
        *,
        tools: ToolsType | None = None,
    ) -> Dict[str, Any]:
        """
        Process ChatMessage objects with tool calls and invoke tools durably.

        Mirrors the structure of upstream ``ToolInvoker.run_async`` but replaces
        ThreadPoolExecutor with ``ctx.run_typed()`` + ``restate.gather()`` for
        durable parallel execution.

        :param messages: A list of ChatMessage objects, some containing tool calls.
        :param state: Optional runtime state for state-aware tools.
        :param tools: Optional runtime tool override. If set, overrides the tools
            provided at initialization for this invocation only.
        :returns: A dictionary with ``tool_messages`` and ``state``.
        """
        ctx = get_restate_context()

        # Mirror upstream: warm_up check
        if not self._haystack_invoker._is_warmed_up:  # pylint: disable=protected-access
            self._haystack_invoker.warm_up()

        # Mirror upstream: runtime tools override
        tools_with_names = self._haystack_invoker._tools_with_names  # pylint: disable=protected-access
        if tools is not None:
            tools_with_names = HaystackToolInvoker._validate_and_prepare_tools(  # pylint: disable=protected-access
                tools
            )

        if state is None:
            state = State(schema={})

        # Only keep messages with tool calls
        messages_with_tool_calls = [message for message in messages if message.tool_calls]

        if not messages_with_tool_calls:
            return {"tool_messages": [], "state": state}

        # 1) Collect all tool calls and their parameters for parallel execution
        tool_messages: list[ChatMessage] = []
        (
            tool_calls,
            tool_call_params,
            error_messages,
        ) = self._haystack_invoker._prepare_tool_call_params(  # pylint: disable=protected-access
            messages_with_tool_calls=messages_with_tool_calls,  # type: ignore[arg-type]
            state=state,
            streaming_callback=None,
            enable_streaming_passthrough=False,
            tools_with_names=tools_with_names,
        )
        tool_messages.extend(ChatMessage.from_haystack_message(m) for m in error_messages)

        if not tool_call_params:
            return {"tool_messages": tool_messages, "state": state}

        # 2) Execute valid tool calls durably via Restate  # no cover: start
        # Journal name is a label for observability — Restate matches entries by sequential position.
        futures = []
        for params in tool_call_params:
            tool_to_invoke = params["tool_to_invoke"]
            final_args = params["final_args"]
            future = ctx.run_typed(f"invoke_{tool_to_invoke.name}", tool_to_invoke.invoke, **final_args)
            futures.append(future)

        gathered = await restate.gather(*futures)

        # 3) Gather and process results: handle errors and merge outputs into state
        for future_result, tool_call, params in zip(gathered, tool_calls, tool_call_params):
            tool_to_invoke = params["tool_to_invoke"]
            try:
                result = await future_result
            except Exception as e:  # pylint: disable=broad-exception-caught
                # Restate internal exceptions (suspension, cancellation) must always propagate
                if restate.is_internal_exception(e):
                    raise
                # Tool invocation error
                if self.raise_on_failure:
                    raise
                logger.error("{error_exception}", error_exception=e)
                tool_messages.append(ChatMessage.from_tool(tool_result=str(e), origin=tool_call, error=True))
                continue

            # Success: merge outputs into state and prepare result message
            try:
                HaystackToolInvoker._merge_tool_outputs(  # pylint: disable=protected-access
                    tool=tool_to_invoke, result=result, state=state
                )
                tool_messages.append(
                    ChatMessage.from_haystack_message(
                        self._haystack_invoker._prepare_tool_result_message(  # pylint: disable=protected-access
                            result=result, tool_call=tool_call, tool_to_invoke=tool_to_invoke
                        )
                    )
                )
            except Exception as e:  # pylint: disable=broad-exception-caught
                error = ToolOutputMergeError.from_exception(tool_name=tool_call.tool_name, error=e)
                if self.raise_on_failure:
                    raise error from e
                logger.error("{error_exception}", error_exception=error)
                tool_messages.append(ChatMessage.from_tool(tool_result=str(error), origin=tool_call, error=True))

        # Combine if requested (for Claude/Gemini protocol)
        if self.combine_tool_results_to_single_message and tool_messages:
            tool_messages = [self._combine_tool_results(tool_messages)]

        return {"tool_messages": tool_messages, "state": state}  # no cover: stop
