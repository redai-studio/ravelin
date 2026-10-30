"""
This module contains a component that invokes tools based on prepared tool calls and returns the results.
"""

from typing import Any, Dict, List, Optional, Union

from haystack import component, default_from_dict, default_to_dict
from haystack.components.agents.state import State
from haystack.components.tools.tool_invoker import ToolInvoker as HaystackToolInvoker
from haystack.tools import Tool, Toolset, ToolsType, deserialize_tools_or_toolset_inplace
from typing_extensions import Self

from ravelin.dataclasses import ChatRole, ChatMessage


@component
class ToolInvoker:
    """
    A component that invokes tools based on prepared tool calls and returns the results.

    This component wraps the Haystack `ToolInvoker` to provide a consistent API within the Ravelin SDK.
    It supports multiple streaming callbacks and a simplified interface.
    """

    def __init__(
        self,
        tools: Union[List[Tool], Toolset],
        raise_on_failure: bool = True,
        convert_result_to_json_string: bool = False,
        combine_tool_results_to_single_message: bool = False,
    ):
        """
        Initialize the ToolInvoker component.

        :param tools: A list of tools that can be invoked or a Toolset instance.
        :param raise_on_failure: If True, the component will raise an exception in case of errors.
        :param convert_result_to_json_string: If True, tool results are converted to a JSON string.
        :param combine_tool_results_to_single_message: If True, tool results are combined into a single message
            (to comply with Claude & Gemini protocol instead of OpenAI).
        """
        self.tools = tools
        self.raise_on_failure = raise_on_failure
        self.convert_result_to_json_string = convert_result_to_json_string
        self.combine_tool_results_to_single_message = combine_tool_results_to_single_message

        self._haystack_invoker = HaystackToolInvoker(
            tools=self.tools,
            raise_on_failure=self.raise_on_failure,
            convert_result_to_json_string=self.convert_result_to_json_string,
        )

    def _combine_tool_results(self, tool_result_msgs: List[ChatMessage]) -> ChatMessage:
        """
        Combine tool result messages into a single message.

        :param tool_result_msgs: A list of tool result messages.
        :return: A single tool result message with the combined results.
        """
        tool_results = []
        for tool_result_msg in tool_result_msgs:
            tool_results.extend(tool_result_msg.tool_call_results)

        return ChatMessage(_role=ChatRole.TOOL, _content=tool_results)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the component to a dictionary.
        """
        return default_to_dict(
            self,
            tools=self.tools,
            raise_on_failure=self.raise_on_failure,
            convert_result_to_json_string=self.convert_result_to_json_string,
            combine_tool_results_to_single_message=self.combine_tool_results_to_single_message,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        """
        Deserialize the component from a dictionary.
        """
        init_params = data.get("init_parameters", {})
        deserialize_tools_or_toolset_inplace(init_params, key="tools")
        return default_from_dict(cls, data)

    @component.output_types(tool_messages=List[ChatMessage], state=State)
    def run(
        self, messages: List[ChatMessage], state: Optional[State] = None, *, tools: Optional[ToolsType] = None
    ) -> Dict[str, Any]:
        """
        Process `ChatMessage` objects containing tool calls and invoke the corresponding tools.

        :param messages: A list of `ChatMessage` objects, some containing tool calls.
        :param state: The runtime state to be used by the tools.
        :param tools: Tools selected for this invocation; None uses the configured tools.
        :return: A dictionary with the key `tool_messages` containing a list of `ChatMessage` objects
                 with the results of the tool invocations.
        """
        # Use type ignore as our ChatMessage class is a subclass of HaystackChatMessage
        result = self._haystack_invoker.run(messages=messages, state=state, tools=tools)  # type: ignore
        # Convert the HaystackChatMessages back to our ChatMessage type
        result["tool_messages"] = [ChatMessage.from_haystack_message(m) for m in result["tool_messages"]]

        # Combine tool results into a single message if requested
        if self.combine_tool_results_to_single_message:
            result["tool_messages"] = [self._combine_tool_results(result["tool_messages"])]
        return result

    @component.output_types(tool_messages=List[ChatMessage], state=State)
    async def run_async(
        self, messages: List[ChatMessage], state: Optional[State] = None, *, tools: Optional[ToolsType] = None
    ) -> Dict[str, Any]:
        """
        Asynchronously process `ChatMessage` objects containing tool calls and invoke the corresponding tools.

        :param messages: A list of `ChatMessage` objects, some containing tool calls.
        :param state: The runtime state to be used by the tools.
        :param tools: Tools selected for this invocation; None uses the configured tools.
        :return: A dictionary with the key `tool_messages` containing a list of `ChatMessage` objects
                 with the results of the tool invocations.
        """
        result = await self._haystack_invoker.run_async(messages=messages, state=state, tools=tools)  # type: ignore
        # Convert the HaystackChatMessages back to our ChatMessage type
        result["tool_messages"] = [ChatMessage.from_haystack_message(m) for m in result["tool_messages"]]

        # Combine tool results into a single message if requested
        if self.combine_tool_results_to_single_message:
            result["tool_messages"] = [self._combine_tool_results(result["tool_messages"])]
        return result
