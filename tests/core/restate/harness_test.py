"""
Integration tests using Restate test harness.

These tests require:
- restate_sdk[harness] to be installed
- Docker to be running (DOCKER_HOST set if non-default socket)

They exercise the full flow: create_restate_app → HTTP request → workflow → service → response,
testing both app.py handlers and restate_pipeline.py orchestration together.
"""

import pytest

pytestmark = pytest.mark.integration

try:
    import restate

    RESTATE_AVAILABLE = True
except ImportError:
    RESTATE_AVAILABLE = False


@pytest.fixture()
def harness_env():
    """Skip if Restate test harness environment isn't available."""
    pytest.importorskip("restate", reason="restate_sdk required")
    try:
        pytest.importorskip("docker", reason="docker package required for test harness")
    except pytest.skip.Exception:
        raise
    except Exception:
        pytest.skip("docker package not available")


def _run_harness(app, endpoint, json_data):
    """
    Run a request through the test harness and return the response.

    Raises pytest.skip if the harness can't start (e.g. Docker not running).
    """
    try:
        from docker.errors import DockerException
    except ImportError:
        pytest.skip("docker package required for test harness")

    try:
        with restate.test_harness(app) as harness:
            client = harness.ingress_client()
            return client.post(endpoint, json=json_data)
    except (DockerException, ConnectionError, OSError) as e:
        pytest.skip(f"Restate test harness not available: {e}")


@pytest.mark.skipif(not RESTATE_AVAILABLE, reason="restate_sdk not installed")
class TestHarnessBasicPipeline:
    """Test basic single-component pipeline through the harness."""

    @pytest.mark.asyncio
    async def test_single_component_passthrough(self, harness_env):
        """Test a simple single-component pipeline end-to-end."""
        from ravelin.core.restate.app import create_restate_app
        from ravelin.core.restate.restate_pipeline import RestatePipeline
        from ravelin.utils.pipeline_io_handler import PipelineIOHandler
        from haystack.components.converters import OutputAdapter

        pipeline = RestatePipeline()
        adapter = OutputAdapter(template="{{ text }}", output_type=str)
        pipeline.add_component("adapter", adapter)

        io_handler = PipelineIOHandler(
            inputs={"text": {"type": str, "connections": ["adapter.text"], "required": True}},
            outputs={"output": {"type": str, "connection": "adapter.output"}},
        )

        workflow, executor = create_restate_app(pipeline, io_handler, "BasicTest")
        app = restate.app([workflow, executor])

        response = _run_harness(app, "/BasicTest/test-1/run", {"text": "hello"})

        assert response.status_code == 200
        result = response.json()
        assert "output" in result


@pytest.mark.skipif(not RESTATE_AVAILABLE, reason="restate_sdk not installed")
class TestHarnessMultiComponent:
    """Test multi-component connected pipelines through the harness."""

    @pytest.mark.asyncio
    async def test_multi_component_pipeline(self, harness_env):
        """Test a 3-component connected pipeline: uppercase → repeat → wrap."""
        from haystack import component
        from ravelin.core.restate.app import create_restate_app
        from ravelin.core.restate.restate_pipeline import RestatePipeline
        from ravelin.utils.pipeline_io_handler import PipelineIOHandler

        @component
        class UpperCaser:
            """Converts text to uppercase."""

            @component.output_types(text=str)
            def run(self, text: str) -> dict[str, str]:
                return {"text": text.upper()}

        @component
        class Repeater:
            """Repeats text N times."""

            @component.output_types(text=str)
            def run(self, text: str, times: int = 2) -> dict[str, str]:
                return {"text": " ".join([text] * times)}

        @component
        class Wrapper:
            """Wraps text with delimiters."""

            @component.output_types(result=str)
            def run(self, text: str) -> dict[str, str]:
                return {"result": f"[{text}]"}

        pipeline = RestatePipeline()
        pipeline.add_component("upper", UpperCaser())
        pipeline.add_component("repeat", Repeater())
        pipeline.add_component("wrap", Wrapper())
        pipeline.connect("upper.text", "repeat.text")
        pipeline.connect("repeat.text", "wrap.text")

        io_handler = PipelineIOHandler(
            inputs={"text": {"type": str, "connections": ["upper.text"], "required": True}},
            outputs={"result": {"type": str, "connection": "wrap.result"}},
        )

        workflow, executor = create_restate_app(pipeline, io_handler, "MultiTest")
        app = restate.app([workflow, executor])

        response = _run_harness(app, "/MultiTest/test-multi/run", {"text": "hello"})

        assert response.status_code == 200
        result = response.json()
        assert result["result"] == "[HELLO HELLO]"


@pytest.mark.skipif(not RESTATE_AVAILABLE, reason="restate_sdk not installed")
class TestHarnessSerialization:
    """Test serialization of complex types across service boundaries."""

    @pytest.mark.asyncio
    async def test_chat_message_roundtrip(self, harness_env):
        """Test that ChatMessage serializes correctly across service boundaries."""
        from haystack import component
        from ravelin.core.restate.app import create_restate_app
        from ravelin.core.restate.restate_pipeline import RestatePipeline
        from ravelin.dataclasses import ChatMessage
        from ravelin.utils.pipeline_io_handler import PipelineIOHandler

        @component
        class MessageEchoComponent:
            """Component that echoes back ChatMessages."""

            @component.output_types(messages=list[ChatMessage])
            def run(self, messages: list[ChatMessage]) -> dict[str, list[ChatMessage]]:
                response = ChatMessage.from_assistant(f"Echo: {messages[0].text}")
                return {"messages": messages + [response]}

        pipeline = RestatePipeline()
        pipeline.add_component("echo", MessageEchoComponent())

        io_handler = PipelineIOHandler(
            inputs={
                "messages": {
                    "type": list[ChatMessage],
                    "connections": ["echo.messages"],
                    "required": True,
                }
            },
            outputs={"messages": {"type": list[ChatMessage], "connection": "echo.messages"}},
        )

        workflow, executor = create_restate_app(pipeline, io_handler, "SerdeTest")
        app = restate.app([workflow, executor])

        response = _run_harness(
            app,
            "/SerdeTest/test-serde/run",
            {
                "messages": [
                    {
                        "__type__": "ravelin.dataclasses.chat_message.ChatMessage",
                        "role": "user",
                        "content": [{"text": "Hello"}],
                        "name": None,
                        "meta": {},
                    }
                ]
            },
        )

        assert response.status_code == 200
        result = response.json()
        assert "messages" in result
        messages = result["messages"]
        assert len(messages) == 2
        assert "__type__" in messages[0]
        assert "ChatMessage" in messages[0]["__type__"]


@pytest.mark.skipif(not RESTATE_AVAILABLE, reason="restate_sdk not installed")
class TestHarnessErrorHandling:
    """Test error handling behavior through the harness."""

    @pytest.mark.asyncio
    async def test_type_validation_error(self, harness_env):
        """Type validation errors should return 500 with TerminalError."""
        from ravelin.core.restate.app import create_restate_app
        from ravelin.core.restate.restate_pipeline import RestatePipeline
        from ravelin.utils.pipeline_io_handler import PipelineIOHandler
        from haystack.components.converters import OutputAdapter

        pipeline = RestatePipeline()
        adapter = OutputAdapter(template="{{ text }}", output_type=str)
        pipeline.add_component("adapter", adapter)

        io_handler = PipelineIOHandler(
            inputs={"text": {"type": str, "connections": ["adapter.text"], "required": True}},
            outputs={"output": {"type": str, "connection": "adapter.output"}},
        )

        workflow, executor = create_restate_app(pipeline, io_handler, "TypeErrTest")
        app = restate.app([workflow, executor])

        # Send wrong type (list instead of str)
        response = _run_harness(
            app,
            "/TypeErrTest/test-type-error/run",
            {"text": ["not", "a", "string"]},
        )

        assert response.status_code == 500
        error_text = response.text
        assert "type" in error_text.lower() or "validation" in error_text.lower()

    @pytest.mark.asyncio
    async def test_non_recoverable_error_becomes_terminal(self, harness_env):
        """Component raising NonRecoverableError should result in TerminalError."""
        from haystack import component
        from ravelin.core.restate import NonRecoverableError
        from ravelin.core.restate.app import create_restate_app
        from ravelin.core.restate.restate_pipeline import RestatePipeline
        from ravelin.utils.pipeline_io_handler import PipelineIOHandler

        @component
        class NonRecoverableErrorComponent:
            @component.output_types(result=str)
            def run(self, text: str) -> dict[str, str]:
                raise NonRecoverableError("Invalid durable state")

        pipeline = RestatePipeline()
        pipeline.add_component("validator", NonRecoverableErrorComponent())

        io_handler = PipelineIOHandler(
            inputs={"text": {"type": str, "connections": ["validator.text"], "required": True}},
            outputs={"result": {"type": str, "connection": "validator.result"}},
        )

        workflow, executor = create_restate_app(pipeline, io_handler, "NonRecoverableTest")
        app = restate.app([workflow, executor])

        response = _run_harness(
            app,
            "/NonRecoverableTest/test-non-recoverable/run",
            {"text": "trigger_error"},
        )

        assert response.status_code == 500
        assert "NonRecoverableError" in response.text or "non_recoverable" in response.text

    @pytest.mark.asyncio
    async def test_custom_error_is_retryable(self, harness_env):
        """Component raising custom error should be retried by Restate."""
        from haystack import component
        from ravelin.core.restate.app import create_restate_app
        from ravelin.core.restate.restate_pipeline import RestatePipeline
        from ravelin.utils.pipeline_io_handler import PipelineIOHandler

        class TransientError(Exception):
            pass

        call_count = 0

        @component
        class TransientErrorComponent:
            @component.output_types(result=str)
            def run(self, text: str) -> dict[str, str]:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise TransientError("Transient failure")
                return {"result": f"Success on attempt {call_count}"}

        pipeline = RestatePipeline()
        pipeline.add_component("transient", TransientErrorComponent())

        io_handler = PipelineIOHandler(
            inputs={"text": {"type": str, "connections": ["transient.text"], "required": True}},
            outputs={"result": {"type": str, "connection": "transient.result"}},
        )

        workflow, executor = create_restate_app(pipeline, io_handler, "RetryTest")
        app = restate.app([workflow, executor])

        response = _run_harness(
            app,
            "/RetryTest/test-retry/run",
            {"text": "test"},
        )

        assert response.status_code == 200
        result = response.json()
        assert "result" in result
        assert "Success on attempt 2" in result["result"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
