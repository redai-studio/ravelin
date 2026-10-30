"""Tests for RestateConfirmationStrategy."""

import unittest
from unittest.mock import AsyncMock, patch


from ravelin.core.restate.approvals import ApprovalTimeout
from ravelin.core.restate import ToolExecutionApprovalPayload
from ravelin.core.restate.confirmation import RestateConfirmationStrategy


# Module-level notifier so Haystack can serialize it via dotted-path lookup
async def _module_level_notifier(notification):
    pass


class TestRestateConfirmationStrategyInit(unittest.TestCase):
    def test_init_defaults(self):
        strategy = RestateConfirmationStrategy()
        self.assertIsNone(strategy.notifier)
        self.assertIsNone(strategy.timeout_seconds)

    def test_init_with_params(self):
        async def my_notifier(n):
            pass

        strategy = RestateConfirmationStrategy(notifier=my_notifier, timeout_seconds=120)
        self.assertIs(strategy.notifier, my_notifier)
        self.assertEqual(strategy.timeout_seconds, 120)


class TestRestateConfirmationStrategySync(unittest.TestCase):
    def test_run_raises_runtime_error(self):
        strategy = RestateConfirmationStrategy()
        with self.assertRaisesRegex(RuntimeError, "requires async execution"):
            strategy.run(
                tool_name="my_tool",
                tool_description="does something",
                tool_params={"key": "value"},
            )


class TestRestateConfirmationStrategyAsync(unittest.IsolatedAsyncioTestCase):
    async def test_approved(self):
        strategy = RestateConfirmationStrategy()
        approval = ToolExecutionApprovalPayload(approved=True, reason=None)

        with patch(
            "ravelin.core.restate.confirmation.require_approval",
            new_callable=AsyncMock,
            return_value=approval,
        ):
            decision = await strategy.run_async(
                tool_name="my_tool",
                tool_description="does something",
                tool_params={"key": "val"},
                tool_call_id="tc1",
            )

        self.assertTrue(decision.execute)
        self.assertEqual(decision.tool_name, "my_tool")
        self.assertEqual(decision.tool_call_id, "tc1")

    async def test_rejected_with_reason(self):
        strategy = RestateConfirmationStrategy()
        approval = ToolExecutionApprovalPayload(approved=False, reason="too risky")

        with patch(
            "ravelin.core.restate.confirmation.require_approval",
            new_callable=AsyncMock,
            return_value=approval,
        ):
            decision = await strategy.run_async(
                tool_name="my_tool",
                tool_description="does something",
                tool_params={},
                tool_call_id="tc1",
            )

        self.assertFalse(decision.execute)
        self.assertIn("too risky", decision.feedback)

    async def test_rejected_without_reason(self):
        strategy = RestateConfirmationStrategy()
        approval = ToolExecutionApprovalPayload(approved=False, reason=None)

        with patch(
            "ravelin.core.restate.confirmation.require_approval",
            new_callable=AsyncMock,
            return_value=approval,
        ):
            decision = await strategy.run_async(
                tool_name="my_tool",
                tool_description="does something",
                tool_params={},
            )

        self.assertFalse(decision.execute)
        self.assertIsNone(decision.feedback)

    async def test_timeout(self):
        strategy = RestateConfirmationStrategy(timeout_seconds=60)

        with patch(
            "ravelin.core.restate.confirmation.require_approval",
            new_callable=AsyncMock,
            side_effect=ApprovalTimeout(timeout_seconds=60, awakeable_id="awk-1"),
        ):
            decision = await strategy.run_async(
                tool_name="my_tool",
                tool_description="does something",
                tool_params={},
                tool_call_id="tc2",
            )

        self.assertFalse(decision.execute)
        self.assertIn("60", decision.feedback)

    async def test_passes_metadata_to_require_approval(self):
        strategy = RestateConfirmationStrategy(timeout_seconds=30)
        approval = ToolExecutionApprovalPayload(approved=True)

        with patch(
            "ravelin.core.restate.confirmation.require_approval",
            new_callable=AsyncMock,
            return_value=approval,
        ) as mock_require:
            await strategy.run_async(
                tool_name="search",
                tool_description="Search the web",
                tool_params={"query": "test"},
                tool_call_id="tc99",
            )

        mock_require.assert_called_once()
        call_kwargs = mock_require.call_args.kwargs
        self.assertEqual(call_kwargs["timeout_seconds"], 30)
        meta = call_kwargs["metadata"]
        self.assertEqual(meta["tool_name"], "search")
        self.assertEqual(meta["tool_params"], {"query": "test"})
        self.assertEqual(meta["tool_call_id"], "tc99")


class TestRestateConfirmationStrategySerialization(unittest.TestCase):
    def test_to_dict_no_notifier(self):
        strategy = RestateConfirmationStrategy(timeout_seconds=120)
        d = strategy.to_dict()
        self.assertEqual(d["init_parameters"]["timeout_seconds"], 120)
        self.assertIsNone(d["init_parameters"]["notifier"])

    def test_to_dict_with_notifier(self):
        strategy = RestateConfirmationStrategy(notifier=_module_level_notifier)
        d = strategy.to_dict()
        self.assertIsNotNone(d["init_parameters"]["notifier"])

    def test_from_dict_roundtrip(self):
        strategy = RestateConfirmationStrategy(timeout_seconds=45)
        d = strategy.to_dict()
        restored = RestateConfirmationStrategy.from_dict(d)
        self.assertEqual(restored.timeout_seconds, 45)
        self.assertIsNone(restored.notifier)


if __name__ == "__main__":
    unittest.main()
