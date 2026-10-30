"""Tests for RestateAgent component."""

import unittest
from unittest.mock import AsyncMock, MagicMock, create_autospec

from ravelin.core.restate.context import _restate_ctx


class TestRestateAgentInit(unittest.TestCase):
    """Test RestateAgent initialization."""

    def test_init_basic(self):
        """Test basic initialization with defaults."""
        from ravelin.components.agents import RestateAgent

        mock_generator = MagicMock()
        agent = RestateAgent(chat_generator=mock_generator)

        self.assertIs(agent.chat_generator, mock_generator)
        self.assertListEqual(agent.tools, [])
        self.assertIsNone(agent.system_prompt)
        self.assertListEqual(agent.exit_conditions, ["text"])
        self.assertEqual(agent.max_agent_steps, 100)
        self.assertDictEqual(agent._confirmation_strategies, {})

    def test_init_with_tools(self):
        """Test initialization with tools creates a RestateToolInvoker."""
        from ravelin.components.agents import RestateAgent
        from haystack.tools import Tool

        mock_generator = MagicMock()
        tool1 = MagicMock(spec=Tool)
        tool1.name = "search"
        tool2 = MagicMock(spec=Tool)
        tool2.name = "calculator"

        agent = RestateAgent(
            chat_generator=mock_generator,
            tools=[tool1, tool2],
            exit_conditions=["text", "search"],
        )

        self.assertEqual(len(agent.tools), 2)
        self.assertIsNotNone(agent._tool_invoker)
        self.assertListEqual(agent.exit_conditions, ["text", "search"])

    def test_init_with_toolset(self):
        """Test initialization with a Toolset."""
        from ravelin.components.agents import RestateAgent
        from haystack.tools import Tool, Toolset

        mock_generator = MagicMock()
        tool1 = MagicMock(spec=Tool)
        tool1.name = "search"
        tool2 = MagicMock(spec=Tool)
        tool2.name = "calculator"

        toolset = Toolset([tool1, tool2])
        agent = RestateAgent(chat_generator=mock_generator, tools=toolset)

        self.assertIsInstance(agent.tools, Toolset)
        self.assertIsNotNone(agent._tool_invoker)

    def test_init_invalid_exit_condition(self):
        """Test that invalid exit conditions raise ValueError."""
        from ravelin.components.agents import RestateAgent

        mock_generator = MagicMock()

        with self.assertRaisesRegex(ValueError, "Invalid exit conditions"):
            RestateAgent(
                chat_generator=mock_generator,
                exit_conditions=["nonexistent_tool"],
            )

    def test_init_with_confirmation_strategies(self):
        """Test initialization stores confirmation strategies."""
        from ravelin.components.agents import RestateAgent
        from haystack.human_in_the_loop.types import ConfirmationStrategy

        mock_generator = MagicMock()
        mock_strategy = create_autospec(ConfirmationStrategy, instance=True)

        agent = RestateAgent(
            chat_generator=mock_generator,
            confirmation_strategies={"my_tool": mock_strategy},
        )

        self.assertIn("my_tool", agent._confirmation_strategies)
        self.assertIs(agent._confirmation_strategies["my_tool"], mock_strategy)


class TestRestateAgentSync(unittest.TestCase):
    """Test that sync run raises RuntimeError."""

    def test_run_raises_runtime_error(self):
        """Sync run should raise RuntimeError."""
        from ravelin.components.agents import RestateAgent
        from ravelin.dataclasses import ChatMessage

        mock_generator = MagicMock()
        agent = RestateAgent(chat_generator=mock_generator)

        with self.assertRaisesRegex(RuntimeError, "requires async execution"):
            agent.run([ChatMessage.from_user("Hello")])


class TestRestateAgentWarmUp(unittest.TestCase):
    """Test warm-up functionality."""

    def test_warm_up_delegates_to_generator(self):
        """Warm up should delegate to chat generator."""
        from ravelin.components.agents import RestateAgent

        mock_generator = MagicMock()
        agent = RestateAgent(chat_generator=mock_generator)
        self.assertFalse(agent._is_warmed_up)

        agent.warm_up()

        mock_generator.warm_up.assert_called_once()
        self.assertTrue(agent._is_warmed_up)

    def test_warm_up_only_once(self):
        """Warm up should only happen once."""
        from ravelin.components.agents import RestateAgent

        mock_generator = MagicMock()
        agent = RestateAgent(chat_generator=mock_generator)

        agent.warm_up()
        agent.warm_up()

        mock_generator.warm_up.assert_called_once()


class TestRestateAgentAsync(unittest.IsolatedAsyncioTestCase):
    """Test async execution."""

    def setUp(self):
        self.mock_ctx = MagicMock()
        self.mock_ctx.run_typed = AsyncMock()

        self.mock_generator = MagicMock()
        self.mock_generator.warm_up = MagicMock()
        self.mock_generator.run_async = AsyncMock()

    async def test_run_async_text_exit(self):
        """Text exit: LLM returns no tool calls → agent exits after one step."""
        from ravelin.components.agents import RestateAgent
        from ravelin.dataclasses import ChatMessage

        response = ChatMessage.from_assistant("Hello! How can I help?")
        # run_typed wraps the LLM call and returns its result directly (List[ChatMessage])
        self.mock_ctx.run_typed.return_value = [response]

        agent = RestateAgent(chat_generator=self.mock_generator)
        agent.warm_up()

        token = _restate_ctx.set(self.mock_ctx)
        try:
            result = await agent.run_async([ChatMessage.from_user("Hello")])

            self.assertEqual(result["last_message"], response)
            self.assertIn(response, result["generated_messages"])
        finally:
            _restate_ctx.reset(token)

    async def test_run_async_with_system_prompt(self):
        """System prompt is prepended to the message list."""
        from ravelin.components.agents import RestateAgent
        from ravelin.dataclasses import ChatMessage

        response = ChatMessage.from_assistant("I'm a helpful assistant!")
        self.mock_ctx.run_typed.return_value = [response]

        agent = RestateAgent(
            chat_generator=self.mock_generator,
            system_prompt="You are a helpful assistant.",
        )
        agent.warm_up()

        token = _restate_ctx.set(self.mock_ctx)
        try:
            result = await agent.run_async([ChatMessage.from_user("Who are you?")])

            first_msg = result["messages"][0]
            self.assertEqual(first_msg.role.value, "system")
            self.assertIn("helpful assistant", first_msg.text)
        finally:
            _restate_ctx.reset(token)

    async def test_run_async_max_agent_steps(self):
        """Agent stops after max_agent_steps iterations even if LLM keeps calling tools."""
        from ravelin.components.agents import RestateAgent
        from ravelin.dataclasses import ChatMessage
        from haystack.components.agents.state import State
        from haystack.dataclasses import ToolCall
        from haystack.tools import Tool

        mock_tool = MagicMock(spec=Tool)
        mock_tool.name = "search"

        tool_call = ToolCall(tool_name="search", arguments={"query": "test"}, id="tc1")
        tool_response = ChatMessage.from_assistant("Searching...", tool_calls=[tool_call])

        # LLM always returns a tool call → agent must hit max_agent_steps
        self.mock_ctx.run_typed.return_value = [tool_response]

        agent = RestateAgent(
            chat_generator=self.mock_generator,
            tools=[mock_tool],
            max_agent_steps=2,
        )
        agent.warm_up()

        tool_result = ChatMessage.from_tool("search result", origin=tool_call)
        state = State(schema=agent.state_schema)
        agent._tool_invoker.run_async = AsyncMock(return_value={"tool_messages": [tool_result], "state": state})

        token = _restate_ctx.set(self.mock_ctx)
        try:
            await agent.run_async([ChatMessage.from_user("Search for something")])
            self.assertEqual(self.mock_ctx.run_typed.call_count, 2)
        finally:
            _restate_ctx.reset(token)

    async def test_run_async_tool_invoker_error_propagates(self):
        """Tool invoker errors propagate so the outer Restate boundary can classify them."""
        from ravelin.components.agents import RestateAgent
        from ravelin.dataclasses import ChatMessage
        from haystack.dataclasses import ToolCall
        from haystack.tools import Tool

        mock_tool = MagicMock(spec=Tool)
        mock_tool.name = "failing_tool"

        tool_call = ToolCall(tool_name="failing_tool", arguments={}, id="tc1")
        tool_response = ChatMessage.from_assistant("Using tool", tool_calls=[tool_call])
        self.mock_ctx.run_typed.return_value = [tool_response]

        agent = RestateAgent(
            chat_generator=self.mock_generator,
            tools=[mock_tool],
            raise_on_tool_invocation_failure=True,
        )
        agent.warm_up()

        agent._tool_invoker.run_async = AsyncMock(side_effect=RuntimeError("crash"))

        token = _restate_ctx.set(self.mock_ctx)
        try:
            with self.assertRaisesRegex(RuntimeError, "crash"):
                await agent.run_async([ChatMessage.from_user("Use the failing tool")])
        finally:
            _restate_ctx.reset(token)


class TestRestateAgentConfirmationStrategies(unittest.IsolatedAsyncioTestCase):
    """Test that confirmation strategies are correctly wired into the agent loop.

    Uses a real Tool (so Haystack's argument-inspection machinery works) and a
    mock ConfirmationStrategy, letting _process_confirmation_strategies_async run
    naturally. RestateConfirmationStrategy is tested in its own file.
    """

    def setUp(self):
        self.mock_ctx = MagicMock()
        self.mock_ctx.run_typed = AsyncMock()

    async def test_approved_strategy_calls_tool_invoker(self):
        """When strategy approves, tool invoker receives the tool call message."""
        from ravelin.components.agents import RestateAgent
        from ravelin.dataclasses import ChatMessage
        from haystack.components.agents.state import State
        from haystack.dataclasses import ToolCall
        from haystack.human_in_the_loop.types import ConfirmationStrategy
        from haystack.human_in_the_loop.types.protocol import ToolExecutionDecision
        from haystack.tools import create_tool_from_function

        def dangerous_action() -> str:
            return "action completed"

        tool = create_tool_from_function(
            dangerous_action, name="dangerous_action", description="Do something dangerous"
        )

        mock_generator = MagicMock()
        tool_call = ToolCall(tool_name="dangerous_action", arguments={}, id="tc1")
        tool_response = ChatMessage.from_assistant("I'll do it", tool_calls=[tool_call])
        final_response = ChatMessage.from_assistant("Done!")

        call_count = [0]

        async def mock_run_typed(*args, **kwargs):
            call_count[0] += 1
            return [tool_response] if call_count[0] == 1 else [final_response]

        self.mock_ctx.run_typed.side_effect = mock_run_typed

        mock_strategy = create_autospec(ConfirmationStrategy, instance=True)
        mock_strategy.run_async.return_value = ToolExecutionDecision(tool_name="dangerous_action", execute=True)

        agent = RestateAgent(
            chat_generator=mock_generator,
            tools=[tool],
            confirmation_strategies={"dangerous_action": mock_strategy},
        )
        agent.warm_up()

        tool_result = ChatMessage.from_tool("action completed", origin=tool_call)
        state = State(schema=agent.state_schema)
        tool_invoker_mock = AsyncMock(return_value={"tool_messages": [tool_result], "state": state})
        agent._tool_invoker.run_async = tool_invoker_mock

        token = _restate_ctx.set(self.mock_ctx)
        try:
            await agent.run_async([ChatMessage.from_user("Do it")])

            mock_strategy.run_async.assert_called_once_with(
                tool_name="dangerous_action",
                tool_description="Do something dangerous",
                tool_params={},
                tool_call_id="tc1",
                confirmation_strategy_context=None,
            )
            self.assertEqual(len(tool_invoker_mock.call_args.kwargs["messages"]), 1)
        finally:
            _restate_ctx.reset(token)

    async def test_rejected_strategy_skips_tool_invocation(self):
        """When strategy rejects all tool calls, tool invoker receives no messages to execute."""
        from ravelin.components.agents import RestateAgent
        from ravelin.dataclasses import ChatMessage
        from haystack.components.agents.state import State
        from haystack.dataclasses import ToolCall
        from haystack.human_in_the_loop.types import ConfirmationStrategy
        from haystack.human_in_the_loop.types.protocol import ToolExecutionDecision
        from haystack.tools import create_tool_from_function

        def dangerous_action() -> str:
            return "action completed"

        tool = create_tool_from_function(
            dangerous_action, name="dangerous_action", description="Do something dangerous"
        )

        mock_generator = MagicMock()
        tool_call = ToolCall(tool_name="dangerous_action", arguments={}, id="tc1")
        tool_response = ChatMessage.from_assistant("I'll do it", tool_calls=[tool_call])
        final_response = ChatMessage.from_assistant("OK, I won't do it.")

        call_count = [0]

        async def mock_run_typed(*args, **kwargs):
            call_count[0] += 1
            return [tool_response] if call_count[0] == 1 else [final_response]

        self.mock_ctx.run_typed.side_effect = mock_run_typed

        mock_strategy = create_autospec(ConfirmationStrategy, instance=True)
        mock_strategy.run_async.return_value = ToolExecutionDecision(
            tool_name="dangerous_action", execute=False, feedback="Too risky"
        )

        agent = RestateAgent(
            chat_generator=mock_generator,
            tools=[tool],
            confirmation_strategies={"dangerous_action": mock_strategy},
        )
        agent.warm_up()

        state = State(schema=agent.state_schema)
        tool_invoker_mock = AsyncMock(return_value={"tool_messages": [], "state": state})
        agent._tool_invoker.run_async = tool_invoker_mock

        token = _restate_ctx.set(self.mock_ctx)
        try:
            await agent.run_async([ChatMessage.from_user("Do the dangerous thing")])

            # Tool invoker was called with empty modified_tool_call_messages (all rejected)
            self.assertListEqual(tool_invoker_mock.call_args.kwargs["messages"], [])
        finally:
            _restate_ctx.reset(token)


class TestRestateAgentExitConditions(unittest.IsolatedAsyncioTestCase):
    """Test exit condition handling."""

    def setUp(self):
        self.mock_ctx = MagicMock()
        self.mock_ctx.run_typed = AsyncMock()

    async def test_exit_on_tool(self):
        """Agent exits after one step when a tool matching an exit condition succeeds."""
        from ravelin.components.agents import RestateAgent
        from ravelin.dataclasses import ChatMessage
        from haystack.components.agents.state import State
        from haystack.dataclasses import ToolCall
        from haystack.tools import Tool

        mock_generator = MagicMock()
        final_answer_tool = MagicMock(spec=Tool)
        final_answer_tool.name = "final_answer"

        tool_call = ToolCall(tool_name="final_answer", arguments={"answer": "42"}, id="tc1")
        tool_response = ChatMessage.from_assistant("Here's the answer", tool_calls=[tool_call])
        self.mock_ctx.run_typed.return_value = [tool_response]

        agent = RestateAgent(
            chat_generator=mock_generator,
            tools=[final_answer_tool],
            exit_conditions=["text", "final_answer"],
        )
        agent.warm_up()

        tool_result = ChatMessage.from_tool("The answer is 42", origin=tool_call)
        state = State(schema=agent.state_schema)
        agent._tool_invoker.run_async = AsyncMock(return_value={"tool_messages": [tool_result], "state": state})

        token = _restate_ctx.set(self.mock_ctx)
        try:
            await agent.run_async([ChatMessage.from_user("What is the meaning of life?")])
            # Only one LLM call before exit condition triggered
            self.assertEqual(self.mock_ctx.run_typed.call_count, 1)
        finally:
            _restate_ctx.reset(token)


class TestRestateAgentSelectTools(unittest.TestCase):
    """Test _select_tools with various input types."""

    def setUp(self):
        from ravelin.components.agents import RestateAgent
        from haystack.tools import Tool

        mock_generator = MagicMock()
        self.tool1 = MagicMock(spec=Tool)
        self.tool1.name = "search"
        self.tool2 = MagicMock(spec=Tool)
        self.tool2.name = "calc"
        self.agent = RestateAgent(
            chat_generator=mock_generator,
            tools=[self.tool1, self.tool2],
        )

    def test_select_tools_none_returns_all(self):
        result = self.agent._select_tools(None)
        self.assertEqual(result, [self.tool1, self.tool2])

    def test_select_tools_by_name(self):
        result = self.agent._select_tools(["search"])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "search")

    def test_select_tools_by_name_invalid(self):
        with self.assertRaisesRegex(ValueError, "not valid"):
            self.agent._select_tools(["nonexistent"])

    def test_select_tools_by_name_no_tools_configured(self):
        from ravelin.components.agents import RestateAgent

        mock_generator = MagicMock()
        agent = RestateAgent(chat_generator=mock_generator)

        with self.assertRaisesRegex(ValueError, "No tools were configured"):
            agent._select_tools(["search"])

    def test_select_tools_toolset(self):
        from haystack.tools import Toolset

        toolset = Toolset([self.tool1])
        result = self.agent._select_tools(toolset)
        self.assertIsInstance(result, Toolset)

    def test_select_tools_list_of_tools(self):
        result = self.agent._select_tools([self.tool1])
        self.assertEqual(result, [self.tool1])

    def test_select_tools_invalid_type(self):
        with self.assertRaises(TypeError):
            self.agent._select_tools("search")  # type: ignore[arg-type]


class TestRestateAgentStateSchema(unittest.TestCase):
    """Test state_schema validation and custom schema handling."""

    def test_invalid_state_schema_raises(self):
        from ravelin.components.agents import RestateAgent

        mock_generator = MagicMock()
        with self.assertRaises(Exception):
            RestateAgent(
                chat_generator=mock_generator,
                state_schema={"bad_key": "not_a_dict"},  # type: ignore[dict-item]
            )

    def test_custom_state_schema_output(self):
        """Custom state schema fields are added as outputs."""
        from ravelin.components.agents import RestateAgent
        from haystack.components.agents.state.state_utils import merge_lists

        mock_generator = MagicMock()
        agent = RestateAgent(
            chat_generator=mock_generator,
            state_schema={"my_list": {"type": list, "handler": merge_lists}},
        )
        # Custom schema param should appear as an output socket
        self.assertIn("my_list", agent.__haystack_output__)


class TestRestateAgentInitializeFreshExecution(unittest.TestCase):
    """Test _initialize_fresh_execution covers generation_kwargs and system prompt paths."""

    def test_all_system_messages_warning(self):
        """Warning is emitted when all provided messages are system messages."""
        from ravelin.components.agents import RestateAgent

        mock_generator = MagicMock()
        agent = RestateAgent(chat_generator=mock_generator, system_prompt="sys")

        from ravelin.dataclasses import ChatMessage

        with self.assertLogs("ravelin", level="WARNING") as log:
            agent._initialize_fresh_execution(
                messages=[ChatMessage.from_system("only system")],
                system_prompt=None,  # use init system_prompt → prepend, leaving only system msgs
            )
        self.assertTrue(any("system messages" in msg for msg in log.output))

    def test_generation_kwargs_added_to_generator_inputs(self):
        """generation_kwargs are forwarded to the generator inputs dict."""
        from ravelin.components.agents import RestateAgent

        mock_generator = MagicMock()
        agent = RestateAgent(chat_generator=mock_generator)

        from ravelin.dataclasses import ChatMessage

        ctx = agent._initialize_fresh_execution(
            messages=[ChatMessage.from_user("hi")],
            generation_kwargs={"temperature": 0.5},
        )
        self.assertEqual(ctx.chat_generator_inputs.get("generation_kwargs"), {"temperature": 0.5})


class TestRestateAgentCheckExitConditions(unittest.TestCase):
    """Test _check_exit_conditions edge cases."""

    def setUp(self):
        from ravelin.components.agents import RestateAgent
        from haystack.tools import Tool
        from ravelin.dataclasses.chat_message import ToolCall

        mock_generator = MagicMock()
        self.tool = MagicMock(spec=Tool)
        self.tool.name = "final_answer"
        self.agent = RestateAgent(
            chat_generator=mock_generator,
            tools=[self.tool],
            exit_conditions=["text", "final_answer"],
        )
        self.tc = ToolCall(tool_name="final_answer", arguments={}, id="tc1")

    def test_exit_matched_no_error(self):
        from ravelin.dataclasses import ChatMessage

        llm_msgs = [ChatMessage.from_assistant("done", tool_calls=[self.tc])]
        tool_msgs = [ChatMessage.from_tool("result", origin=self.tc, error=False)]
        self.assertTrue(self.agent._check_exit_conditions(llm_msgs, tool_msgs))

    def test_exit_matched_with_error(self):
        from ravelin.dataclasses import ChatMessage

        llm_msgs = [ChatMessage.from_assistant("done", tool_calls=[self.tc])]
        tool_msgs = [ChatMessage.from_tool("error", origin=self.tc, error=True)]
        self.assertFalse(self.agent._check_exit_conditions(llm_msgs, tool_msgs))

    def test_exit_not_matched(self):
        from ravelin.dataclasses import ChatMessage
        from ravelin.dataclasses.chat_message import ToolCall

        other_tc = ToolCall(tool_name="other_tool", arguments={}, id="tc2")
        llm_msgs = [ChatMessage.from_assistant("", tool_calls=[other_tc])]
        tool_msgs = [ChatMessage.from_tool("result", origin=other_tc, error=False)]
        self.assertFalse(self.agent._check_exit_conditions(llm_msgs, tool_msgs))


if __name__ == "__main__":
    unittest.main()
