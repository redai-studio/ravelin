"""Core pipeline utilities with optional Restate integration."""

import sys
from importlib.util import find_spec
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

from ravelin.core.type_validation import validate_types

# Preserve serde setup when the optional SDK is installed. Context probes from
# base model services must also work without the Restate extra.
if find_spec("restate") is not None:
    from ravelin.core.restate.serde import patch_default_serde as _patch_default_serde

    _patch_default_serde()

_import_structure = {
    "restate_pipeline": ["RestatePipeline"],
    "app": ["NON_RETRYABLE_ERRORS", "create_restate_app"],
    "context": ["get_restate_context", "has_restate_context"],
    "errors": [
        "ErrorClassification",
        "ErrorClassifier",
        "ErrorContext",
        "NonRecoverableError",
        "OperatorFixableError",
        "TransientError",
        "build_error_event_payload",
        "classify_restate_error",
    ],
    "approvals": [
        "ApprovalNotification",
        "ApprovalNotifier",
        "ApprovalTimeout",
        "ToolExecutionApprovalPayload",
        "require_approval",
    ],
    "confirmation": ["RestateConfirmationStrategy"],
    "serde": ["DictProtocolSerde", "patch_default_serde"],
}

if TYPE_CHECKING:
    from ravelin.core.restate.restate_pipeline import RestatePipeline
    from ravelin.core.restate.app import NON_RETRYABLE_ERRORS, create_restate_app
    from ravelin.core.restate.context import get_restate_context, has_restate_context
    from ravelin.core.restate.errors import (
        ErrorClassification,
        ErrorClassifier,
        ErrorContext,
        NonRecoverableError,
        OperatorFixableError,
        TransientError,
        build_error_event_payload,
        classify_restate_error,
    )
    from ravelin.core.restate.approvals import (
        ApprovalNotification,
        ApprovalNotifier,
        ApprovalTimeout,
        ToolExecutionApprovalPayload,
        require_approval,
    )
    from ravelin.core.restate.confirmation import RestateConfirmationStrategy
    from ravelin.core.restate.serde import DictProtocolSerde, patch_default_serde

    __all__ = [
        "ApprovalNotification",
        "ApprovalNotifier",
        "ApprovalTimeout",
        "DictProtocolSerde",
        "ErrorClassification",
        "ErrorClassifier",
        "ErrorContext",
        "NonRecoverableError",
        "NON_RETRYABLE_ERRORS",
        "OperatorFixableError",
        "RestateConfirmationStrategy",
        "RestatePipeline",
        "TransientError",
        "ToolExecutionApprovalPayload",
        "build_error_event_payload",
        "classify_restate_error",
        "create_restate_app",
        "get_restate_context",
        "has_restate_context",
        "patch_default_serde",
        "require_approval",
        "validate_types",
    ]

else:
    sys.modules[__name__] = LazyImporter(
        name=__name__,
        module_file=__file__,
        import_structure=_import_structure,
        extra_objects={"validate_types": validate_types},
    )
