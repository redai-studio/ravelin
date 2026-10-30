import unittest
from typing import Dict, List, Optional, Any
from unittest.mock import AsyncMock, Mock, patch, MagicMock

from haystack.components.agents.state.state_utils import merge_lists
from haystack.tools import Tool

from ravelin.components.agents.agent import Agent
from ravelin.components.generators import ChatGenerator
from ravelin.components.tools import ToolInvoker
from ravelin.dataclasses import ChatMessage
from ravelin.dataclasses.chat_message import ToolCall


# Minimal run/run_async stubs with the right signatures
def chat_generator_run(messages: List[ChatMessage], tools: Optional[List[Tool]] = None) -> Dict[str, Any]:
    return {"replies": []}


async def chat_generator_run_async(messages: List[ChatMessage], tools: Optional[List[Tool]] = None) -> Dict[str, Any]:
    return {"replies": []}


def _make_generator(inference_api_format: str = "openai-chat") -> Mock:
    gen = Mock(spec=ChatGenerator)
    gen.run = chat_generator_run
    gen.run_async = chat_generator_run_async
    ms = Mock()
    ms.inference_api_format = inference_api_format
    gen.model_service = ms
    return gen


# ---------------------------------------------------------------------------
# 1. Initialisation tests
# ---------------------------------------------------------------------------
class TestAgentInit(unittest.TestCase):
    def setUp(self):
        self.gen = _make_generator()
        self.tool1 = Mock(spec=Tool)
        self.tool1.name = "tool1"
        self.tool2 = Mock(spec=Tool)
        self.tool2.name = "tool2"

    def test_init_basic(self):
        agent = Agent(chat_generator=self.gen, tools=[self.tool1, self.tool2])
        self.assertIs(agent.chat_generator, self.gen)
        self.assertEqual(agent.tools, [self.tool1, self.tool2])
        self.assertIsNone(agent.system_prompt)
        self.assertEqual(agent.exit_conditions, ["text"])
        self.assertEqual(agent.max_agent_steps, 100)
        self.assertFalse(agent.raise_on_tool_invocation_failure)

    def test_init_no_tools(self):
        agent = Agent(chat_generator=self.gen)
        self.assertEqual(agent.tools, [])
        self.assertIsNone(agent._tool_invoker)

    def test_init_invalid_exit_condition(self):
        with self.assertRaises(ValueError):
            Agent(chat_generator=self.gen, tools=[self.tool1], exit_conditions=["nonexistent_tool"])

    def test_init_output_types_include_generated_messages(self):
        agent = Agent(chat_generator=self.gen, tools=[self.tool1])
        self.assertIn("generated_messages", agent.__haystack_output__)
        self.assertIn("last_message", agent.__haystack_output__)
        self.assertIn("messages", agent.__haystack_output__)

    def test_init_uses_our_tool_invoker(self):
        agent = Agent(chat_generator=self.gen, tools=[self.tool1, self.tool2])
        self.assertIsInstance(agent._tool_invoker, ToolInvoker)

    def test_init_combine_flag_openai(self):
        gen = _make_generator("openai-chat")
        agent = Agent(chat_generator=gen, tools=[self.tool1])
        self.assertFalse(agent._tool_invoker.combine_tool_results_to_single_message)

    def test_init_combine_flag_anthropic(self):
        gen = _make_generator("anthropic-chat")
        agent = Agent(chat_generator=gen, tools=[self.tool1])
        self.assertTrue(agent._tool_invoker.combine_tool_results_to_single_message)

    def test_init_combine_flag_gemini(self):
        gen = _make_generator("gemini-chat")
        agent = Agent(chat_generator=gen, tools=[self.tool1])
        self.assertTrue(agent._tool_invoker.combine_tool_results_to_single_message)

    def test_init_no_streaming_callbacks_attribute(self):
        agent = Agent(chat_generator=self.gen, tools=[self.tool1])
        self.assertFalse(hasattr(agent, "streaming_callbacks"))


# ---------------------------------------------------------------------------
# 2. Serialisation tests
# ---------------------------------------------------------------------------
class TestAgentSerialization(unittest.TestCase):
    def setUp(self):
        self.gen = _make_generator()
        self.tool1 = Mock(spec=Tool)
        self.tool1.name = "tool1"

    def test_to_dict_excludes_streaming_callback_singular(self):
        """to_dict strips the Haystack-internal streaming_callback key."""
        agent = Agent(chat_generator=self.gen, tools=[self.tool1])
        with (
            patch("haystack.components.agents.agent.component_to_dict", return_value={}),
            patch("haystack.components.agents.agent.serialize_tools_or_toolset", return_value=[]),
            patch("haystack.components.agents.agent._schema_to_dict", return_value={}),
        ):
            d = agent.to_dict()
        self.assertNotIn("streaming_callback", d.get("init_parameters", {}))

    def test_from_dict_removes_streaming_callback_singular(self):
        data = {
            "type": "ravelin.components.agents.agent.Agent",
            "init_parameters": {
                "chat_generator": {"type": "ChatGenerator", "init_parameters": {}},
                "streaming_callback": "some.path",  # old/Haystack singular form
            },
        }
        with patch("haystack.components.agents.agent.default_from_dict", return_value=Mock()) as mock_from_dict:
            with patch("haystack.components.agents.agent.deserialize_component_inplace"):
                with patch("haystack.components.agents.agent.deserialize_tools_or_toolset_inplace"):
                    Agent.from_dict(data)
        # streaming_callback should have been removed before default_from_dict is called
        passed_data = mock_from_dict.call_args[0][1]
        self.assertNotIn("streaming_callback", passed_data["init_parameters"])


# ---------------------------------------------------------------------------
# 3. Warm-up tests
# ---------------------------------------------------------------------------
class TestAgentWarmUp(unittest.TestCase):
    def setUp(self):
        self.gen = _make_generator()
        self.tool1 = Mock(spec=Tool)
        self.tool1.name = "tool1"
        self.agent = Agent(chat_generator=self.gen, tools=[self.tool1])

    def test_warm_up_calls_generator(self):
        self.gen.warm_up = Mock()
        self.agent._is_warmed_up = False
        self.agent.warm_up()
        self.gen.warm_up.assert_called_once()
        self.assertTrue(self.agent._is_warmed_up)

    def test_warm_up_only_once(self):
        self.gen.warm_up = Mock()
        self.agent._is_warmed_up = False
        self.agent.warm_up()
        self.agent.warm_up()
        self.gen.warm_up.assert_called_once()


# ---------------------------------------------------------------------------
# 4. Exit-condition helper (pure logic, no mocking needed)
# ---------------------------------------------------------------------------
class TestAgentExitConditions(unittest.TestCase):
    def setUp(self):
        gen = _make_generator()
        tool1 = Mock(spec=Tool)
        tool1.name = "tool1"
        tool2 = Mock(spec=Tool)
        tool2.name = "tool2"
        self.agent = Agent(chat_generator=gen, tools=[tool1, tool2], exit_conditions=["text", "tool1"])

    def test_exit_condition_matched_no_error(self):
        tc = ToolCall(tool_name="tool1", arguments={}, id="c1")
        llm = [ChatMessage.from_assistant(text="", tool_calls=[tc])]
        tool = [ChatMessage.from_tool(tool_result="ok", origin=tc, error=False)]
        self.assertTrue(self.agent._check_exit_conditions(llm, tool))

    def test_exit_condition_matched_with_error(self):
        tc = ToolCall(tool_name="tool1", arguments={}, id="c2")
        llm = [ChatMessage.from_assistant(text="", tool_calls=[tc])]
        tool = [ChatMessage.from_tool(tool_result="err", origin=tc, error=True)]
        self.assertFalse(self.agent._check_exit_conditions(llm, tool))

    def test_exit_condition_not_matched(self):
        tc = ToolCall(tool_name="tool2", arguments={}, id="c3")
        llm = [ChatMessage.from_assistant(text="", tool_calls=[tc])]
        tool = [ChatMessage.from_tool(tool_result="ok", origin=tc, error=False)]
        self.assertFalse(self.agent._check_exit_conditions(llm, tool))


# ---------------------------------------------------------------------------
# 5. run() / run_async() — generated_messages injection
#    Patches target Haystack's agent namespace since the loop lives there.
# ---------------------------------------------------------------------------
class TestAgentRun(unittest.TestCase):
    def setUp(self):
        self.gen = _make_generator()
        self.tool1 = Mock(spec=Tool)
        self.tool1.name = "tool1"
        self.tool2 = Mock(spec=Tool)
        self.tool2.name = "tool2"
        self.agent = Agent(
            chat_generator=self.gen,
            tools=[self.tool1, self.tool2],
            system_prompt="You are a helpful assistant",
            exit_conditions=["text", "tool1"],
            state_schema={"custom_data": {"type": list, "handler": merge_lists}},
            max_agent_steps=50,
        )
        self.agent.warm_up()
        self.agent._tool_invoker = Mock(spec=ToolInvoker)
        self.messages = [ChatMessage.from_user("Hello, agent!")]

    @patch("haystack.components.agents.agent.Pipeline._run_component")
    @patch("haystack.components.agents.agent.State")
    @patch("haystack.components.agents.agent.tracing.tracer.trace")
    def test_run_text_exit_adds_generated_messages(self, mock_trace, mock_state, mock_run_component):
        mock_span = MagicMock()
        mock_trace.return_value.__enter__.return_value = mock_span

        llm_messages = [ChatMessage.from_assistant("Hello!")]
        mock_run_component.return_value = {"replies": llm_messages}

        system_message = ChatMessage.from_system("You are a helpful assistant")
        all_messages = [system_message] + self.messages + llm_messages
        mock_state_instance = MagicMock()
        mock_state.return_value = mock_state_instance
        mock_state_instance.data = {"messages": all_messages, "custom_data": []}
        mock_state_instance.get.return_value = all_messages

        result = self.agent.run(self.messages)

        mock_run_component.assert_called_once()
        # initial_count = len(messages) + 1 (system prompt) = 2
        # generated_messages = all_messages[2:] = [assistant_message]
        self.assertEqual(result["generated_messages"], llm_messages)
        self.assertEqual(result["last_message"], llm_messages[-1])

    @patch("haystack.components.agents.agent.Pipeline._run_component")
    @patch("haystack.components.agents.agent.State")
    @patch("haystack.components.agents.agent.tracing.tracer.trace")
    def test_run_with_tool_exit_adds_generated_messages(self, mock_trace, mock_state, mock_run_component):
        mock_span = MagicMock()
        mock_trace.return_value.__enter__.return_value = mock_span

        tc = ToolCall(tool_name="tool1", arguments={"arg": "val"}, id="tc1")
        tool_call_msg = ChatMessage.from_assistant(text="Using tool1", tool_calls=[tc])
        tool_response = ChatMessage.from_tool(tool_result="result", origin=tc, error=False)

        system_message = ChatMessage.from_system("You are a helpful assistant")
        user_message = self.messages[0]
        all_messages = [system_message, user_message, tool_call_msg, tool_response]

        mock_state_instance = MagicMock()
        mock_state.return_value = mock_state_instance
        mock_state_instance.data = {"messages": all_messages, "custom_data": []}
        mock_state_instance.get.return_value = all_messages

        # Haystack replaces exe_context.state with tool_invoker_result["state"] after tool runs.
        # Pass mock_state_instance so that exe_context.state.data still yields our all_messages.
        mock_run_component.side_effect = [
            {"replies": [tool_call_msg]},
            {"tool_messages": [tool_response], "state": mock_state_instance},
        ]

        result = self.agent.run([user_message])

        self.assertEqual(mock_run_component.call_count, 2)
        # initial_count = 1 (user_message) + 1 (system_prompt) = 2
        # generated_messages = all_messages[2:] = [tool_call_msg, tool_response]
        self.assertEqual(result["generated_messages"], [tool_call_msg, tool_response])
        self.assertEqual(result["last_message"], tool_response)

    @patch("haystack.components.agents.agent.Pipeline._run_component")
    @patch("haystack.components.agents.agent.State")
    @patch("haystack.components.agents.agent.tracing.tracer.trace")
    @patch("haystack.components.agents.agent.logger.warning")
    def test_run_max_steps_stops_and_warns(self, mock_warn, mock_trace, mock_state, mock_run_component):
        mock_trace.return_value.__enter__.return_value = MagicMock()
        mock_state_instance = MagicMock()
        mock_state.return_value = mock_state_instance
        mock_state_instance.data = {"messages": self.messages, "custom_data": []}

        tc = ToolCall(tool_name="tool2", arguments={}, id="tc2")
        tool_call_msg = ChatMessage.from_assistant(text="", tool_calls=[tc])
        tool_resp = ChatMessage.from_tool(tool_result="r", origin=tc, error=False)

        mock_run_component.side_effect = lambda **kw: (
            {"replies": [tool_call_msg]}
            if kw.get("component_name") == "chat_generator"
            else {"tool_messages": [tool_resp], "state": mock_state_instance}
        )

        self.agent.max_agent_steps = 2
        result = self.agent.run(self.messages)

        mock_warn.assert_called_with(
            "Agent reached maximum agent steps of {max_agent_steps}, stopping.", max_agent_steps=2
        )
        self.assertIn("generated_messages", result)

    @patch("haystack.components.agents.agent.Pipeline._run_component")
    @patch("haystack.components.agents.agent.State")
    @patch("haystack.components.agents.agent.tracing.tracer.trace")
    def test_run_auto_warms_up_if_needed(self, mock_trace, mock_state, mock_run_component):
        # Haystack auto-calls warm_up() in run() if not already warmed up
        self.agent._is_warmed_up = False
        self.gen.warm_up = Mock()
        mock_trace.return_value.__enter__.return_value = MagicMock()
        llm_messages = [ChatMessage.from_assistant("Hello!")]
        mock_run_component.return_value = {"replies": llm_messages}
        system_message = ChatMessage.from_system("You are a helpful assistant")
        all_messages = [system_message] + self.messages + llm_messages
        mock_state_instance = MagicMock()
        mock_state.return_value = mock_state_instance
        mock_state_instance.data = {"messages": all_messages, "custom_data": []}
        mock_state_instance.get.return_value = all_messages

        self.agent.run(self.messages)

        self.gen.warm_up.assert_called_once()
        self.assertTrue(self.agent._is_warmed_up)


class TestAgentRunAsync(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.gen = _make_generator()
        self.tool1 = Mock(spec=Tool)
        self.tool1.name = "tool1"
        self.tool2 = Mock(spec=Tool)
        self.tool2.name = "tool2"
        self.agent = Agent(
            chat_generator=self.gen,
            tools=[self.tool1, self.tool2],
            system_prompt="You are a helpful assistant",
            exit_conditions=["text", "tool1"],
            state_schema={"custom_data": {"type": list, "handler": merge_lists}},
            max_agent_steps=50,
        )
        self.agent.warm_up()
        self.agent._tool_invoker = AsyncMock(spec=ToolInvoker)
        self.messages = [ChatMessage.from_user("Hello, agent!")]

    @patch("haystack.components.agents.agent.AsyncPipeline._run_component_async")
    @patch("haystack.components.agents.agent.State")
    @patch("haystack.components.agents.agent.tracing.tracer.trace")
    async def test_run_async_text_exit_adds_generated_messages(self, mock_trace, mock_state, mock_run_async):
        mock_trace.return_value.__enter__.return_value = MagicMock()
        llm_messages = [ChatMessage.from_assistant("Hello!")]
        mock_run_async.return_value = {"replies": llm_messages}

        system_message = ChatMessage.from_system("You are a helpful assistant")
        all_messages = [system_message] + self.messages + llm_messages
        mock_state_instance = MagicMock()
        mock_state.return_value = mock_state_instance
        mock_state_instance.data = {"messages": all_messages, "custom_data": []}
        mock_state_instance.get.return_value = all_messages

        result = await self.agent.run_async(self.messages)

        mock_run_async.assert_called_once()
        self.assertEqual(result["generated_messages"], llm_messages)
        self.assertEqual(result["last_message"], llm_messages[-1])

    @patch("haystack.components.agents.agent.AsyncPipeline._run_component_async")
    @patch("haystack.components.agents.agent.State")
    @patch("haystack.components.agents.agent.tracing.tracer.trace")
    async def test_run_async_with_tool_exit_adds_generated_messages(self, mock_trace, mock_state, mock_run_async):
        mock_trace.return_value.__enter__.return_value = MagicMock()

        tc = ToolCall(tool_name="tool1", arguments={"arg": "val"}, id="tc1")
        tool_call_msg = ChatMessage.from_assistant(text="Using tool1", tool_calls=[tc])
        tool_response = ChatMessage.from_tool(tool_result="result", origin=tc, error=False)

        system_message = ChatMessage.from_system("You are a helpful assistant")
        user_message = self.messages[0]
        all_messages = [system_message, user_message, tool_call_msg, tool_response]
        mock_state_instance = MagicMock()
        mock_state.return_value = mock_state_instance
        mock_state_instance.data = {"messages": all_messages, "custom_data": []}
        mock_state_instance.get.return_value = all_messages

        # Haystack replaces exe_context.state with tool_invoker_result["state"] after tool runs.
        # Pass mock_state_instance so that exe_context.state.data still yields our all_messages.
        def side_effect(**kw):
            if kw.get("component_name") == "chat_generator":
                return {"replies": [tool_call_msg]}
            return {"tool_messages": [tool_response], "state": mock_state_instance}

        mock_run_async.side_effect = side_effect

        result = await self.agent.run_async([user_message])

        self.assertEqual(mock_run_async.call_count, 2)
        self.assertEqual(result["generated_messages"], [tool_call_msg, tool_response])
        self.assertEqual(result["last_message"], tool_response)

    @patch("haystack.components.agents.agent.AsyncPipeline._run_component_async")
    @patch("haystack.components.agents.agent.State")
    @patch("haystack.components.agents.agent.tracing.tracer.trace")
    async def test_run_async_auto_warms_up_if_needed(self, mock_trace, mock_state, mock_run_async):
        # Haystack auto-calls warm_up() in run_async() if not already warmed up
        self.agent._is_warmed_up = False
        self.gen.warm_up = Mock()
        mock_trace.return_value.__enter__.return_value = MagicMock()
        llm_messages = [ChatMessage.from_assistant("Hello!")]
        mock_run_async.return_value = {"replies": llm_messages}
        system_message = ChatMessage.from_system("You are a helpful assistant")
        all_messages = [system_message] + self.messages + llm_messages
        mock_state_instance = MagicMock()
        mock_state.return_value = mock_state_instance
        mock_state_instance.data = {"messages": all_messages, "custom_data": []}
        mock_state_instance.get.return_value = all_messages

        await self.agent.run_async(self.messages)

        self.gen.warm_up.assert_called_once()
        self.assertTrue(self.agent._is_warmed_up)


class TestAgentSnapshotInit(unittest.TestCase):
    """Test _initialize_from_snapshot strips the passthrough key."""

    def setUp(self):
        self.gen = _make_generator()
        self.tool1 = Mock(spec=Tool)
        self.tool1.name = "tool1"
        self.agent = Agent(chat_generator=self.gen, tools=[self.tool1])
        self.agent.warm_up()

    @patch("haystack.components.agents.agent.Pipeline._run_component")
    @patch("haystack.components.agents.agent.State")
    @patch("haystack.components.agents.agent.tracing.tracer.trace")
    def test_initialize_from_snapshot_strips_passthrough_key(self, mock_trace, mock_state, mock_run_component):
        """_initialize_from_snapshot should strip enable_streaming_callback_passthrough."""
        mock_trace.return_value.__enter__.return_value = MagicMock()

        # Build a minimal snapshot dict that Haystack's from_snapshot accepts
        user_msg = ChatMessage.from_user("hello")
        llm_reply = ChatMessage.from_assistant("reply")
        mock_state_instance = MagicMock()
        mock_state.return_value = mock_state_instance
        mock_state_instance.data = {"messages": [user_msg, llm_reply]}
        mock_state_instance.get.return_value = [user_msg, llm_reply]
        mock_run_component.return_value = {"replies": [llm_reply]}

        # Run normally first so there's a snapshot to restore from
        result = self.agent.run([user_msg])
        # Smoke-test: the agent returned successfully and our override was effective
        self.assertIn("generated_messages", result)
        # _tool_invoker.enable_streaming_callback_passthrough was set to False
        self.assertFalse(self.agent._tool_invoker.enable_streaming_callback_passthrough)


if __name__ == "__main__":
    unittest.main()
