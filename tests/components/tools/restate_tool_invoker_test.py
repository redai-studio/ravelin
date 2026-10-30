"""Unit tests for RestateToolInvoker — non-Restate paths only.

The Restate-native execution block (ctx.run_typed / restate.gather) is excluded
from unit coverage via `# no cover: start/stop` in the source and is exercised
by harness tests instead.
"""

import unittest
from unittest.mock import AsyncMock, MagicMock

from haystack.tools import Tool

from ravelin.core.restate.context import _restate_ctx
from ravelin.dataclasses import ChatMessage
from ravelin.dataclasses.chat_message import ToolCall, ChatRole


def _make_tool(name: str) -> MagicMock:
    t = MagicMock(spec=Tool)
    t.name = name
    return t


class TestRestateToolInvokerInit(unittest.TestCase):
    """Test initialization and defaults."""

    def test_init_defaults(self):
        from ravelin.components.tools.restate_tool_invoker import RestateToolInvoker

        tool = _make_tool("search")
        invoker = RestateToolInvoker(tools=[tool])

        self.assertEqual(invoker.tools, [tool])
        self.assertTrue(invoker.raise_on_failure)
        self.assertFalse(invoker.convert_result_to_json_string)
        self.assertFalse(invoker.combine_tool_results_to_single_message)

    def test_init_custom_flags(self):
        from ravelin.components.tools.restate_tool_invoker import RestateToolInvoker

        tool = _make_tool("calc")
        invoker = RestateToolInvoker(
            tools=[tool],
            raise_on_failure=False,
            convert_result_to_json_string=True,
            combine_tool_results_to_single_message=True,
        )

        self.assertFalse(invoker.raise_on_failure)
        self.assertTrue(invoker.convert_result_to_json_string)
        self.assertTrue(invoker.combine_tool_results_to_single_message)

    def test_init_accepts_extra_kwargs(self):
        """Extra kwargs (e.g. max_workers, streaming_callback) are silently accepted."""
        from ravelin.components.tools.restate_tool_invoker import RestateToolInvoker

        tool = _make_tool("tool")
        # Should not raise
        invoker = RestateToolInvoker(
            tools=[tool],
            max_workers=4,
            streaming_callbacks=["some_cb"],
        )
        self.assertIsNotNone(invoker)

    def test_haystack_invoker_created(self):
        from ravelin.components.tools.restate_tool_invoker import RestateToolInvoker
        from haystack.components.tools.tool_invoker import ToolInvoker as HaystackToolInvoker

        tool = _make_tool("search")
        invoker = RestateToolInvoker(tools=[tool])

        self.assertIsInstance(invoker._haystack_invoker, HaystackToolInvoker)


class TestRestateToolInvokerSync(unittest.TestCase):
    """Sync run must raise RuntimeError."""

    def test_run_raises_runtime_error(self):
        from ravelin.components.tools.restate_tool_invoker import RestateToolInvoker

        tool = _make_tool("search")
        invoker = RestateToolInvoker(tools=[tool])

        with self.assertRaisesRegex(RuntimeError, "requires async execution"):
            invoker.run(messages=[])


class TestCombineToolResults(unittest.TestCase):
    """Test _combine_tool_results merges individual results into one message."""

    def setUp(self):
        from ravelin.components.tools.restate_tool_invoker import RestateToolInvoker

        tool = _make_tool("search")
        self.invoker = RestateToolInvoker(tools=[tool])

    def _make_tool_msg(self, result: str, tool_name: str = "search") -> ChatMessage:
        tc = ToolCall(tool_name=tool_name, arguments={}, id="tc1")
        return ChatMessage.from_tool(tool_result=result, origin=tc)

    def test_combine_single_message(self):
        msg = self._make_tool_msg("result1")
        combined = self.invoker._combine_tool_results([msg])

        self.assertEqual(combined.role, ChatRole.TOOL)
        self.assertEqual(len(combined.tool_call_results), 1)
        self.assertEqual(combined.tool_call_results[0].result, "result1")

    def test_combine_multiple_messages(self):
        msg1 = self._make_tool_msg("result1", "search")
        msg2 = self._make_tool_msg("result2", "calc")
        combined = self.invoker._combine_tool_results([msg1, msg2])

        self.assertEqual(combined.role, ChatRole.TOOL)
        self.assertEqual(len(combined.tool_call_results), 2)
        results = {r.result for r in combined.tool_call_results}
        self.assertEqual(results, {"result1", "result2"})


class TestRestateToolInvokerRunAsyncEarlyReturns(unittest.IsolatedAsyncioTestCase):
    """Test run_async paths that return before hitting the Restate-native block."""

    def setUp(self):
        self.mock_ctx = MagicMock()
        self.mock_ctx.run_typed = AsyncMock()

    async def test_no_messages_with_tool_calls_returns_empty(self):
        from ravelin.components.tools.restate_tool_invoker import RestateToolInvoker

        tool = _make_tool("search")
        invoker = RestateToolInvoker(tools=[tool])

        # Plain user message — no tool calls
        messages = [ChatMessage.from_user("hello")]

        token = _restate_ctx.set(self.mock_ctx)
        try:
            result = await invoker.run_async(messages=messages)
        finally:
            _restate_ctx.reset(token)

        self.assertEqual(result["tool_messages"], [])
        self.mock_ctx.run_typed.assert_not_called()

    async def test_state_none_defaults_to_empty_state(self):
        from ravelin.components.tools.restate_tool_invoker import RestateToolInvoker
        from haystack.components.agents.state import State

        tool = _make_tool("search")
        invoker = RestateToolInvoker(tools=[tool])

        token = _restate_ctx.set(self.mock_ctx)
        try:
            result = await invoker.run_async(messages=[], state=None)
        finally:
            _restate_ctx.reset(token)

        self.assertIsInstance(result["state"], State)

    async def test_unknown_tool_produces_error_message(self):
        """Message with a tool call for an unknown tool → error message, no Restate call."""
        from ravelin.components.tools.restate_tool_invoker import RestateToolInvoker

        known_tool = _make_tool("known")
        invoker = RestateToolInvoker(tools=[known_tool], raise_on_failure=False)

        unknown_tc = ToolCall(tool_name="unknown_tool", arguments={}, id="tc-unknown")
        msg_with_unknown = ChatMessage.from_assistant("calling unknown", tool_calls=[unknown_tc])

        token = _restate_ctx.set(self.mock_ctx)
        try:
            result = await invoker.run_async(messages=[msg_with_unknown])
        finally:
            _restate_ctx.reset(token)

        # _prepare_tool_call_params returns an error_message for the unknown tool
        # tool_call_params will be empty → early return at line 160
        self.assertEqual(len(result["tool_messages"]), 1)
        error_msg = result["tool_messages"][0]
        self.assertTrue(error_msg.tool_call_result.error)
        self.mock_ctx.run_typed.assert_not_called()

    async def test_tools_runtime_override(self):
        """Passing tools at runtime overrides the init-time tool list.

        With raise_on_failure=False: a tool call for a name not in the runtime
        override produces an error message (not an exception), confirming that the
        runtime override is applied before _prepare_tool_call_params.
        """
        from ravelin.components.tools.restate_tool_invoker import RestateToolInvoker
        from haystack.tools import create_tool_from_function

        def my_func(x: int) -> str:
            return str(x)

        runtime_tool = create_tool_from_function(my_func, name="runtime_tool", description="runtime")
        init_tool = _make_tool("init_tool")
        # raise_on_failure=False so unknown-tool errors go to error_messages instead of raising
        invoker = RestateToolInvoker(tools=[init_tool], raise_on_failure=False)

        # Call init_tool but override with runtime_tool → init_tool is now unknown
        unknown_tc = ToolCall(tool_name="init_tool", arguments={}, id="tc1")
        msg = ChatMessage.from_assistant("calling", tool_calls=[unknown_tc])

        token = _restate_ctx.set(self.mock_ctx)
        try:
            result = await invoker.run_async(messages=[msg], tools=[runtime_tool])
        finally:
            _restate_ctx.reset(token)

        # init_tool is not in runtime override → error message returned, no Restate call
        self.assertEqual(len(result["tool_messages"]), 1)
        self.assertTrue(result["tool_messages"][0].tool_call_result.error)
        self.mock_ctx.run_typed.assert_not_called()


if __name__ == "__main__":
    unittest.main()
