"""Error classification helpers for Restate execution boundaries."""

from __future__ import annotations

import hashlib
import logging
import re
import traceback
from dataclasses import dataclass
from typing import Callable, Optional
from urllib.error import HTTPError as UrllibHTTPError, URLError

import httpx

try:  # requests is not a direct dependency, but many third-party tools use it.
    import requests
except ImportError:  # pragma: no cover - depends on optional environment packages
    requests = None  # type: ignore[assignment]


_MAX_MESSAGE_CHARS = 1200
_MAX_STACK_CHARS = 12000
logger = logging.getLogger(__name__)
_SECRET_PATTERN = re.compile(r"(?i)\b(api[_-]?key|authorization|bearer|password|secret|token)(\s*[:=]\s*|\s+)[^\s,;]+")


class TransientError(Exception):
    """Explicit marker for failures that should be retried by Restate."""

    def __init__(self, message: str, *, label: str = "transient_error", status_code: Optional[int] = None) -> None:
        self.label = label
        self.status_code = status_code
        super().__init__(message)


class OperatorFixableError(Exception):
    """Explicit marker for failures that need operator action before resume."""

    def __init__(
        self,
        message: str,
        *,
        label: str = "operator_fixable_error",
        operator_action: Optional[str] = None,
        status_code: Optional[int] = None,
    ) -> None:
        self.label = label
        self.operator_action = operator_action
        self.status_code = status_code
        super().__init__(message)


class NonRecoverableError(Exception):
    """Explicit marker for failures where retry/resume cannot make progress."""

    def __init__(
        self,
        message: str,
        *,
        label: str = "non_recoverable_error",
        operator_action: Optional[str] = None,
        status_code: Optional[int] = None,
    ) -> None:
        self.label = label
        self.operator_action = operator_action
        self.status_code = status_code
        super().__init__(message)


@dataclass
class ErrorContext:
    """Context used when classifying a Restate execution error."""

    component_name: str
    workflow_id: str
    call_index: int
    phase: str


@dataclass
class ErrorClassification:
    """Structured classification for Restate retry/terminal decisions."""

    category: str
    label: str
    terminal: bool
    operator_action: Optional[str] = None
    status_code: Optional[int] = None
    fingerprint: Optional[str] = None


ErrorClassifier = Callable[[BaseException, ErrorContext], Optional[ErrorClassification]]


def classify_restate_error(
    error: BaseException,
    context: ErrorContext,
    custom_classifier: Optional[ErrorClassifier] = None,
) -> ErrorClassification:
    """
    Classify a Restate execution error.

    Custom classifiers may return ``None`` to fall back to the built-in rules.
    If the custom classifier itself fails, classification falls back to the
    unknown retryable bucket.
    """
    if custom_classifier is not None:
        try:
            custom_result = custom_classifier(error, context)
        except Exception:  # pylint: disable=broad-exception-caught
            logger.warning("Restate error classifier failed; falling back to unknown_retryable", exc_info=True)
            return classify_restate_error_fallback(error, context)
        if custom_result is not None:
            return _with_fingerprint(custom_result, error, context)

    return _with_fingerprint(_default_classification(error), error, context)


def classify_restate_error_fallback(error: BaseException, context: ErrorContext) -> ErrorClassification:
    """Return the default unknown retryable classification after classifier failure."""
    return _with_fingerprint(
        ErrorClassification(
            category="unknown_retryable",
            label="unclassified_exception",
            terminal=False,
            operator_action="Inspect stack trace and deployment/configuration, then resume after fixing.",
        ),
        error,
        context,
    )


def build_error_event_payload(
    error: BaseException,
    context: ErrorContext,
    classification: ErrorClassification,
) -> dict[str, object]:
    """Build the structured lifecycle error payload used by Restate journal hooks."""
    message = sanitize_error_message(str(error))
    stack = sanitize_error_message("".join(traceback.format_exception(type(error), error, error.__traceback__)))
    if len(stack) > _MAX_STACK_CHARS:
        stack = stack[: _MAX_STACK_CHARS - 14] + "...<truncated>"

    return {
        "category": classification.category,
        "label": classification.label,
        "exception_type": exception_type_name(error),
        "message": message,
        "status_code": classification.status_code,
        "terminal": classification.terminal,
        "retryable": not classification.terminal,
        "operator_action": classification.operator_action,
        "fingerprint": classification.fingerprint or error_fingerprint(error, context, classification.label),
        "component": context.component_name,
        "workflow_id": context.workflow_id,
        "call_index": context.call_index,
        "phase": context.phase,
        "stack": stack,
    }


def terminal_error_message(error: BaseException, classification: ErrorClassification) -> str:
    """Return a compact TerminalError message; stack traces live in structured events."""
    return (
        f"{classification.category}/{classification.label}: "
        f"{exception_type_name(error)}: {sanitize_error_message(str(error))}"
    )


def sanitize_error_message(message: str) -> str:
    """Redact obvious secrets and trim noisy response bodies/details from error messages."""
    message = _SECRET_PATTERN.sub(lambda m: f"{m.group(1)}{m.group(2)}<redacted>", message)
    if "\nDetails:" in message:
        prefix, _ = message.split("\nDetails:", 1)
        message = f"{prefix}\nDetails: <redacted>"
    if len(message) > _MAX_MESSAGE_CHARS:
        message = message[: _MAX_MESSAGE_CHARS - 14] + "...<truncated>"
    return message


def exception_type_name(error: BaseException) -> str:
    """Return a stable fully-qualified exception type name."""
    return f"{type(error).__module__}.{type(error).__qualname__}"


def error_fingerprint(error: BaseException, context: ErrorContext, label: str) -> str:
    """Build a stable fingerprint for journal-hook deduplication."""
    frames = traceback.extract_tb(error.__traceback__)
    top_frame = frames[-1] if frames else None
    frame_key = (
        f"{top_frame.filename}:{top_frame.name}:{top_frame.lineno}" if top_frame is not None else "<no-traceback>"
    )
    basis = "|".join(
        [
            context.component_name,
            context.phase,
            label,
            exception_type_name(error),
            frame_key,
            sanitize_error_message(str(error)),
        ]
    )
    digest = hashlib.sha256(basis.encode("utf-8")).hexdigest()[:12]
    return f"{context.component_name}:{label}:{digest}"


def _default_classification(error: BaseException) -> ErrorClassification:
    if isinstance(error, NonRecoverableError):
        return ErrorClassification(
            category="non_recoverable",
            label=error.label,
            terminal=True,
            operator_action=error.operator_action,
            status_code=error.status_code,
        )

    if _is_restate_terminal_error(error):
        return ErrorClassification(category="non_recoverable", label="terminal_error", terminal=True)

    if isinstance(error, OperatorFixableError):
        return ErrorClassification(
            category="operator_fixable",
            label=error.label,
            terminal=False,
            operator_action=error.operator_action,
            status_code=error.status_code,
        )

    if isinstance(error, TransientError):
        return ErrorClassification(
            category="transient",
            label=error.label,
            terminal=False,
            status_code=error.status_code,
        )

    status_code = _http_status_code(error)
    if status_code is not None:
        return _classify_http_status(status_code)

    if _is_network_or_timeout_error(error):
        return ErrorClassification(category="transient", label="network_error", terminal=False)

    return ErrorClassification(
        category="unknown_retryable",
        label="unclassified_exception",
        terminal=False,
        operator_action="Inspect stack trace and deployment/configuration, then resume after fixing.",
    )


def _classify_http_status(status_code: int) -> ErrorClassification:
    if status_code in {408, 429, 500, 502, 503, 504}:
        label = "rate_limited" if status_code == 429 else "upstream_unavailable"
        if status_code == 408:
            label = "request_timeout"
        return ErrorClassification(category="transient", label=label, terminal=False, status_code=status_code)
    if status_code in {401, 403}:
        return ErrorClassification(
            category="operator_fixable",
            label="auth_error",
            terminal=False,
            status_code=status_code,
            operator_action="Check provider credentials, permissions, or API key configuration.",
        )
    if 400 <= status_code < 500:
        return ErrorClassification(
            category="operator_fixable",
            label="http_client_error",
            terminal=False,
            status_code=status_code,
            operator_action="Inspect request configuration and upstream API contract.",
        )
    return ErrorClassification(
        category="unknown_retryable",
        label="http_error",
        terminal=False,
        status_code=status_code,
    )


def _http_status_code(error: BaseException) -> Optional[int]:
    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code
    if isinstance(error, UrllibHTTPError):
        return error.code
    response = getattr(error, "response", None)
    status_code = getattr(response, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    status_code = getattr(error, "status_code", None)
    return status_code if isinstance(status_code, int) else None


def _is_network_or_timeout_error(error: BaseException) -> bool:
    request_error_types: tuple[type[BaseException], ...] = (
        httpx.TimeoutException,
        httpx.TransportError,
        TimeoutError,
        ConnectionError,
        URLError,
    )
    if requests is not None:
        request_error_types = (
            *request_error_types,
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
        )
    return isinstance(error, request_error_types)


def _is_restate_terminal_error(error: BaseException) -> bool:
    return type(error).__module__.startswith("restate") and type(error).__name__ == "TerminalError"


def _with_fingerprint(
    classification: ErrorClassification,
    error: BaseException,
    context: ErrorContext,
) -> ErrorClassification:
    if classification.fingerprint is None:
        classification.fingerprint = error_fingerprint(error, context, classification.label)
    return classification
