"""Real Agent/provider/tool execution with only the HTTP transport substituted."""

import json

import httpx
import pytest
from haystack.dataclasses import ChatMessage as HaystackChatMessage

from ravelin.assets.model_services.chat_model_services import OpenAIChatModelService
from ravelin.components.agents import Agent
from ravelin.components.generators import ChatGenerator
from ravelin.components.tools import ToolInvoker
from ravelin.dataclasses import ChatMessage
from ravelin.dataclasses.chat_message import ToolCall
from ravelin.tools import Tool, create_tool_from_function


def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


def unused() -> str:
    """A tool that must not be selected for the test run."""
    raise AssertionError("Unselected tool was invoked")


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_agent_system_prompt_and_selected_tool_reach_real_provider(asynchronous):
    """Exercise the inherited loop and Ravelin adapter without mocking components."""
    requests = []

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        assert request.url.path == "/v1/chat/completions"
        assert body["messages"][0] == {"role": "system", "content": "Use the selected tool."}
        assert [tool["function"]["name"] for tool in body["tools"]] == ["multiply"]
        if len(requests) == 1:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "multiply-1",
                        "type": "function",
                        "function": {"name": "multiply", "arguments": '{"a":17,"b":23}'},
                    }
                ],
            }
            finish_reason = "tool_calls"
        else:
            tool_results = [item for item in body["messages"] if item["role"] == "tool"]
            assert tool_results == [{"role": "tool", "tool_call_id": "multiply-1", "content": "391"}]
            message = {"role": "assistant", "content": "391"}
            finish_reason = "stop"
        return httpx.Response(
            200,
            json={
                "model": "test-model",
                "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
                "usage": {},
            },
        )

    service = OpenAIChatModelService(
        base_url="https://example.test", client_kwargs={"transport": httpx.MockTransport(respond)}
    )
    generator = ChatGenerator(model_service=service, model="test-model", max_retries=0)
    agent = Agent(
        chat_generator=generator,
        tools=[create_tool_from_function(multiply), create_tool_from_function(unused)],
        system_prompt="Use the selected tool.",
        max_agent_steps=3,
        raise_on_tool_invocation_failure=True,
    )
    try:
        inputs = {"messages": [ChatMessage.from_user("17 times 23")], "tools": ["multiply"]}
        result = await agent.run_async(**inputs) if asynchronous else agent.run(**inputs)
    finally:
        service.close()
        await service.aclose()
    assert len(requests) == 2
    assert result["last_message"].text == "391"
    assert len(result["generated_messages"]) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_tool_invoker_honors_runtime_tool_selection(asynchronous):
    """A selected runtime implementation must reach the upstream invoker."""
    original = create_tool_from_function(multiply)
    replacement = Tool(
        name="multiply", description="Replacement", parameters=original.parameters, function=lambda a, b: 42
    )
    invoker = ToolInvoker(tools=[original])
    messages = [ChatMessage.from_assistant(tool_calls=[ToolCall("multiply", {"a": 2, "b": 3}, "id-1")])]
    if asynchronous:
        result = await invoker.run_async(messages=messages, tools=[replacement])
    else:
        result = invoker.run(messages=messages, tools=[replacement])
    assert result["tool_messages"][0].tool_call_results[0].result == "42"


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_generator_normalizes_only_upstream_messages(asynchronous):
    """Existing Ravelin message identity and reasoning must survive normalization."""
    from unittest.mock import AsyncMock, Mock

    service = Mock()
    service.infer.return_value = []
    service.ainfer = AsyncMock(return_value=[])
    generator = ChatGenerator(model_service=service)
    upstream = HaystackChatMessage.from_system("system")
    native = ChatMessage.from_assistant("answer", reasoning_content="reasoning")
    if asynchronous:
        await generator.run_async([upstream, native])
        converted = service.ainfer.call_args.args[0]
    else:
        generator.run([upstream, native])
        converted = service.infer.call_args.args[0]
    assert isinstance(converted[0], ChatMessage)
    assert converted[0].text == "system"
    assert converted[1] is native
