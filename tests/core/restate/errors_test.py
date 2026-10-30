import unittest
from urllib.error import HTTPError as UrllibHTTPError

import httpx

from ravelin.core.restate.errors import (
    ErrorClassification,
    ErrorContext,
    NonRecoverableError,
    OperatorFixableError,
    TransientError,
    build_error_event_payload,
    classify_restate_error,
)


class TestRestateErrorClassification(unittest.TestCase):
    def setUp(self):
        self.context = ErrorContext(
            component_name="retriever",
            workflow_id="wf-1",
            call_index=2,
            phase="component_execution",
        )

    def test_explicit_exception_types(self):
        transient = classify_restate_error(TransientError("temporary"), self.context)
        self.assertEqual("transient", transient.category)
        self.assertFalse(transient.terminal)

        operator = classify_restate_error(
            OperatorFixableError("bad key", operator_action="Rotate API key"),
            self.context,
        )
        self.assertEqual("operator_fixable", operator.category)
        self.assertEqual("Rotate API key", operator.operator_action)
        self.assertFalse(operator.terminal)

        non_recoverable = classify_restate_error(NonRecoverableError("corrupt state"), self.context)
        self.assertEqual("non_recoverable", non_recoverable.category)
        self.assertTrue(non_recoverable.terminal)

    def test_http_status_classification(self):
        self.assert_http_status(429, "transient", "rate_limited")
        self.assert_http_status(503, "transient", "upstream_unavailable")
        self.assert_http_status(403, "operator_fixable", "auth_error")
        self.assert_http_status(400, "operator_fixable", "http_client_error")

    def test_urllib_http_error_uses_status_code_before_oserror_lineage(self):
        error = UrllibHTTPError("https://example.test", 403, "forbidden", hdrs=None, fp=None)

        classification = classify_restate_error(error, self.context)

        self.assertEqual("operator_fixable", classification.category)
        self.assertEqual("auth_error", classification.label)
        self.assertFalse(classification.terminal)

    def test_network_errors_are_transient(self):
        classification = classify_restate_error(httpx.TimeoutException("timeout"), self.context)

        self.assertEqual("transient", classification.category)
        self.assertEqual("network_error", classification.label)
        self.assertFalse(classification.terminal)

    def test_generic_python_errors_are_unknown_retryable(self):
        for error in (ValueError("bug"), OSError("socket-ish or file-ish"), RuntimeError("third party")):
            classification = classify_restate_error(error, self.context)
            self.assertEqual("unknown_retryable", classification.category)
            self.assertEqual("unclassified_exception", classification.label)
            self.assertFalse(classification.terminal)

    def test_custom_classifier_precedence(self):
        def custom_classifier(_error, _context):
            return ErrorClassification(
                category="operator_fixable",
                label="custom_database_config",
                terminal=False,
                operator_action="Check database settings",
            )

        classification = classify_restate_error(RuntimeError("db nope"), self.context, custom_classifier)

        self.assertEqual("custom_database_config", classification.label)
        self.assertEqual("Check database settings", classification.operator_action)
        self.assertIsNotNone(classification.fingerprint)

    def test_custom_classifier_failure_falls_back(self):
        def broken_classifier(_error, _context):
            raise RuntimeError("classifier failed")

        classification = classify_restate_error(RuntimeError("original"), self.context, broken_classifier)

        self.assertEqual("unknown_retryable", classification.category)
        self.assertEqual("unclassified_exception", classification.label)

    def test_payload_contains_stack_trace_and_sanitized_message(self):
        try:
            raise ValueError("bad api_key=secret\nDetails: raw provider body")
        except ValueError as error:
            classification = classify_restate_error(error, self.context)
            payload = build_error_event_payload(error, self.context, classification)

        self.assertEqual("unknown_retryable", payload["category"])
        self.assertEqual("builtins.ValueError", payload["exception_type"])
        self.assertIn("api_key=<redacted>", payload["message"])
        self.assertIn("Details: <redacted>", payload["message"])
        self.assertNotIn("raw provider body", payload["message"])
        self.assertIn("ValueError", payload["stack"])
        self.assertTrue(str(payload["fingerprint"]).startswith("retriever:unclassified_exception:"))

    def assert_http_status(self, status_code: int, category: str, label: str):
        request = httpx.Request("GET", "https://example.test")
        response = httpx.Response(status_code=status_code, request=request)
        error = httpx.HTTPStatusError("bad status", request=request, response=response)

        classification = classify_restate_error(error, self.context)

        self.assertEqual(category, classification.category)
        self.assertEqual(label, classification.label)
        self.assertEqual(status_code, classification.status_code)
        self.assertFalse(classification.terminal)


if __name__ == "__main__":
    unittest.main()
