"""Restate confirmation strategy for human-in-the-loop tool approval via awakeables."""

from typing import Any, Optional

from haystack import default_from_dict, default_to_dict, logging
from haystack.human_in_the_loop import ToolExecutionDecision
from haystack.utils import deserialize_callable, serialize_callable

from ravelin.core.restate.approvals import (
    ApprovalNotifier,
    ApprovalTimeout,
    ToolExecutionApprovalPayload,
    require_approval,
)
from ravelin.streaming import EventType, get_component_context

logger = logging.getLogger(__name__)


class RestateConfirmationStrategy:
    """
    A ConfirmationStrategy that uses Restate awakeables for human approval.

    Implements the upstream Haystack ConfirmationStrategy protocol. The ``run_async()``
    method creates a Restate awakeable, optionally notifies an external system, and
    waits for resolution with an ``ToolExecutionApprovalPayload``.

    Mapping to ``ToolExecutionDecision``:
    - ``ToolExecutionApprovalPayload(approved=True)`` → ``ToolExecutionDecision(execute=True)``
    - ``ToolExecutionApprovalPayload(approved=False, reason=...)`` →
        ``ToolExecutionDecision(execute=False, feedback=...)``
    - ``ApprovalTimeout`` → ``ToolExecutionDecision(execute=False, feedback="Approval timed out...")``

    Usage::

        from ravelin.core.restate import RestateConfirmationStrategy

        strategy = RestateConfirmationStrategy(
            notifier=my_webhook_notifier,
            timeout_seconds=300,
        )

        agent = RestateAgent(
            chat_generator=chat_generator,
            tools=tools,
            confirmation_strategies={"dangerous_tool": strategy},
        )
    """

    def __init__(
        self,
        notifier: Optional[ApprovalNotifier] = None,
        timeout_seconds: Optional[int] = None,
    ):
        """
        Initialize the RestateConfirmationStrategy.

        :param notifier: Optional async callback to notify external systems of approval requests.
            Receives an ``ApprovalNotification`` with ``awakeable_id`` and ``metadata``.
        :param timeout_seconds: Optional timeout in seconds. If exceeded, the tool execution
            is rejected with a timeout feedback message.
        """
        self.notifier = notifier
        self.timeout_seconds = timeout_seconds

    def run(
        self,
        *,
        tool_name: str,
        tool_description: str,
        tool_params: dict[str, Any],
        tool_call_id: Optional[str] = None,
        confirmation_strategy_context: Optional[dict[str, Any]] = None,
    ) -> "ToolExecutionDecision":
        """
        Sync execution is not supported — Restate requires async execution.

        :raises RuntimeError: Always, as sync execution is not supported.
        """
        raise RuntimeError(
            "RestateConfirmationStrategy requires async execution within Restate. Use run_async() instead."
        )

    async def run_async(
        self,
        *,
        tool_name: str,
        tool_description: str,
        tool_params: dict[str, Any],
        tool_call_id: Optional[str] = None,
        confirmation_strategy_context: Optional[dict[str, Any]] = None,  # pylint: disable=unused-argument
    ) -> "ToolExecutionDecision":
        """
        Request human approval for tool execution via a Restate awakeable.

        Creates an awakeable, optionally notifies an external system, and waits
        for the awakeable to be resolved with an ``ToolExecutionApprovalPayload``.

        :param tool_name: Name of the tool requiring approval.
        :param tool_description: Description of the tool.
        :param tool_params: Parameters the LLM wants to pass to the tool.
        :param tool_call_id: Optional tool call ID from the LLM response.
        :param confirmation_strategy_context: Optional request-scoped context (unused by this strategy).
        :returns: A ``ToolExecutionDecision`` indicating whether to execute the tool.
        """
        metadata = {
            "tool_name": tool_name,
            "tool_description": tool_description,
            "tool_params": tool_params,
            "tool_call_id": tool_call_id,
        }

        comp_ctx = get_component_context()
        if comp_ctx:
            await comp_ctx.emit_async(EventType.COMPONENT_ACTIVITY, metadata, subtype="suspended")

        try:
            approval: ToolExecutionApprovalPayload = await require_approval(
                payload_type=ToolExecutionApprovalPayload,
                metadata=metadata,
                timeout_seconds=self.timeout_seconds,
                notifier=self.notifier,
            )
        except ApprovalTimeout as e:
            if comp_ctx:
                await comp_ctx.emit_async(EventType.COMPONENT_ACTIVITY, metadata, subtype="resumed")
            return ToolExecutionDecision(
                tool_name=tool_name,
                execute=False,
                tool_call_id=tool_call_id,
                feedback=f"Approval timed out after {e.timeout_seconds}s",
            )

        if comp_ctx:
            await comp_ctx.emit_async(
                EventType.COMPONENT_ACTIVITY, {**metadata, "approved": approval.approved}, subtype="resumed"
            )

        if approval.approved:
            return ToolExecutionDecision(
                tool_name=tool_name,
                execute=True,
                tool_call_id=tool_call_id,
                final_tool_params=tool_params,
            )
        return ToolExecutionDecision(
            tool_name=tool_name,
            execute=False,
            tool_call_id=tool_call_id,
            feedback=f"Tool execution rejected: {approval.reason}" if approval.reason else None,
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize this strategy to a Haystack component dictionary.

        :returns: Serialized strategy configuration.
        """
        return default_to_dict(
            self,
            timeout_seconds=self.timeout_seconds,
            notifier=serialize_callable(self.notifier) if self.notifier else None,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RestateConfirmationStrategy":
        """
        Deserialize this strategy from a Haystack component dictionary.

        :param data: Serialized strategy configuration.
        :returns: Restored ``RestateConfirmationStrategy`` instance.
        """
        init_params = data.get("init_parameters", {})
        if init_params.get("notifier"):
            init_params["notifier"] = deserialize_callable(init_params["notifier"])
        return default_from_dict(cls, data)
