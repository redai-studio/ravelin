"""Approval utilities for human-in-the-loop patterns with Restate awakeables."""

import logging
from dataclasses import dataclass
from datetime import timedelta
from typing import TypedDict, TypeVar, Any, Callable, Awaitable, Optional, Type

from haystack.lazy_imports import LazyImport

from ravelin.core.restate.context import get_restate_context

with LazyImport("Run 'pip install restate_sdk[serde]'") as restate_import:
    import restate
    from restate.serde import DefaultSerde, Serde

T = TypeVar("T")

logger = logging.getLogger(__name__)


class _ApprovalPayloadSerde(Serde[T]):
    """Make malformed one-shot responses terminal at the approval boundary."""

    def __init__(self, payload_type: Type[T]) -> None:
        self.payload_type = payload_type
        self.delegate = DefaultSerde(payload_type)

    def serialize(self, obj: Optional[T]) -> bytes:
        return self.delegate.serialize(obj)

    def deserialize(self, buf: bytes) -> Optional[T]:
        try:
            return self.delegate.deserialize(buf)
        except restate.TerminalError:
            raise
        except Exception as exc:
            # The resolved bytes are durable: retrying cannot repair them. Do
            # not include payload values in the public error message.
            raise restate.TerminalError(
                f"Approval awakeable resolved with invalid payload for {self.payload_type.__name__} "
                f"({type(exc).__name__})"
            ) from exc


class ApprovalNotification(TypedDict):
    """Arguments passed to the approval notifier callback."""

    awakeable_id: str
    """The unique Restate awakeable ID. Use this to resolve/reject the awakeable."""

    metadata: dict[str, Any]
    """Optional metadata provided by the caller (e.g., response preview, context)."""


# Type alias for the notifier callback
ApprovalNotifier = Callable[[ApprovalNotification], Awaitable[None]] | Callable[[ApprovalNotification], None]


@dataclass
class ToolExecutionApprovalPayload:
    """
    Expected payload structure when resolving a tool execution approval.

    Awakeables should be resolved (not rejected) with this structure so callers
    can handle approval denial as business logic instead of Restate termination.
    """

    approved: bool
    """Required. True to approve, False to reject."""

    reason: str = ""
    """Optional. Explanation for approval/rejection decision."""


class ApprovalTimeout(Exception):
    """
    Raised when approval times out.

    This is raised by require_approval() when the configured timeout expires
    before the awakeable is resolved. Unlike TerminalError, callers can catch
    and handle this (e.g., retry with longer timeout, proceed without approval).

    :ivar timeout_seconds: The timeout duration that was exceeded.
    :ivar awakeable_id: The awakeable ID for debugging or logging.
    """

    def __init__(self, timeout_seconds: int, awakeable_id: str):
        self.timeout_seconds = timeout_seconds
        self.awakeable_id = awakeable_id
        super().__init__(f"Approval timed out after {timeout_seconds} seconds")


async def require_approval(
    payload_type: Type[T] = ToolExecutionApprovalPayload,  # type: ignore[assignment]
    metadata: dict[str, Any] | None = None,
    timeout_seconds: int | None = None,
    notifier: ApprovalNotifier | None = None,
    notifier_run_options: Optional["restate.RunOptions"] = None,
) -> T:
    """
    Require external approval before proceeding.

    This utility can be called from within any component's run method to
    pause execution and wait for external approval via Restate awakeables.

    Protocol:
        External systems should **resolve** (not reject) the awakeable with a
        payload matching ``payload_type``. The awakeable's type hint and the
        runtime guard are adjusted accordingly.

        Using resolve (instead of reject) allows callers to distinguish rejection
        from other terminal errors and handle it gracefully.

    Note: This must be called at the component level (not inside ctx.run),
    as it uses ctx.awakeable() which cannot be nested inside ctx.run().

    The notification mechanism is injectable - provide a notifier callback
    to send the awakeable ID to your external system (webhook, message queue,
    WebSocket, etc.). If no notifier is provided, the awakeable ID is logged.

    Example::

        try:
            decision = await require_approval(
                metadata={"action": "delete_user", "user_id": 123},
                timeout_seconds=300,
            )
            if decision.approved:
                # Proceed with the action
                ...
            else:
                # User explicitly rejected
                logger.info(f"Rejected: {decision.reason}")
        except ApprovalTimeout as e:
            # Timed out waiting
            logger.warning(f"No response after {e.timeout_seconds}s")

    :param payload_type: Expected payload class. Ordinary dataclasses ignore extra
        fields and validate required fields and types. A custom ``from_dict``
        method takes precedence and defines its own decoding rules.
    :param metadata: Optional metadata to include with the notification.
    :param timeout_seconds: Optional timeout in seconds.
    :param notifier: Optional async callback to notify external systems.
                     Receives ApprovalNotification with awakeable_id and metadata.
    :param notifier_run_options: Optional RunOptions for the notifier ctx.run call.
                                 Use to customize retry behavior, timeouts, etc.
    :returns: The deserialized payload of type ``payload_type``.
    :raises ApprovalTimeout: If timeout_seconds is set and exceeded.
    :raises restate.TerminalError: If the resolved payload cannot be decoded as
        ``payload_type``. This also applies to direct Restate console resolution.
    """
    restate_import.check()

    ctx = get_restate_context()
    # Create awakeable for external response
    awakeable_id, promise = ctx.awakeable(serde=_ApprovalPayloadSerde(payload_type), type_hint=payload_type)

    notification = ApprovalNotification(
        awakeable_id=awakeable_id,
        metadata=metadata or {},
    )

    if notifier is not None:
        # Durable notification - survives crashes
        await ctx.run_typed("notify_approval", notifier, notifier_run_options or restate.RunOptions(), notification)
    else:
        # Default: log only
        logger.info(
            "Awaiting approval. Awakeable ID: %s",
            awakeable_id,
        )

    # Wait for response (with optional timeout)
    if timeout_seconds is not None:
        match await restate.select(
            approval=promise,
            timeout=ctx.sleep(timedelta(seconds=timeout_seconds)),
        ):
            case ["approval", result]:
                decision = result
            case ["timeout", _]:
                raise ApprovalTimeout(timeout_seconds, awakeable_id)
            case _:
                raise restate.TerminalError("Unexpected select result for approval")
    else:
        decision = await promise

    # Both the executor endpoint and direct Restate resolution can supply values
    # of the wrong shape (including null). Decoding errors are handled by the
    # approval serde; this guard also checks the decoder's result type.
    if not isinstance(decision, payload_type):
        raise restate.TerminalError(
            f"Approval awakeable '{awakeable_id}' resolved with invalid payload "
            f"(got {type(decision).__name__!r}, expected {payload_type.__name__}). "
            f"Send a valid {payload_type.__name__} payload when resolving the awakeable."
        )

    return decision
