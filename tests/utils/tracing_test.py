import datetime
import unittest
from unittest.mock import patch, MagicMock

from haystack_integrations.tracing.langfuse.tracer import (
    _PIPELINE_INPUT_KEY,
    _PIPELINE_OUTPUT_KEY,
    _COMPONENT_OUTPUT_KEY,
    SpanContext,
    tracing_context_var,
)

from ravelin.dataclasses import ChatMessage
from ravelin.utils import tracing


class TestLangfuseSpan(unittest.TestCase):
    @patch("haystack.tracing.tracer")
    def test_set_content_tag(self, mock_tracer):
        mock_tracer.is_content_tracing_enabled = True
        # Create a mock context manager that returns a mock span when entered
        mock_context_manager = MagicMock()
        mock_span = MagicMock()
        mock_context_manager.__enter__ = MagicMock(return_value=mock_span)
        mock_context_manager.__exit__ = MagicMock(return_value=False)

        span = tracing.LangfuseSpan(mock_context_manager)
        span.set_content_tag("test_key", "test_value")
        self.assertEqual({"test_key": "test_value"}, span._data)

    @patch("haystack.tracing.tracer")
    def test_set_content_tag_disabled(self, mock_tracer):
        mock_tracer.is_content_tracing_enabled = False
        mock_context_manager = MagicMock()
        mock_span = MagicMock()
        mock_context_manager.__enter__ = MagicMock(return_value=mock_span)
        mock_context_manager.__exit__ = MagicMock(return_value=False)

        span = tracing.LangfuseSpan(mock_context_manager)
        span.set_content_tag("test_key", "test_value")
        self.assertEqual({}, span._data)

    @patch("haystack.tracing.tracer")
    @patch("ravelin.utils.serialization.coerce_tag_value")
    def test_set_content_tag_input_and_output(self, mock_coerce, mock_tracer):
        mock_tracer.is_content_tracing_enabled = True
        mock_coerce.side_effect = lambda x: x

        mock_context_manager = MagicMock()
        mock_span = MagicMock()
        mock_context_manager.__enter__ = MagicMock(return_value=mock_span)
        mock_context_manager.__exit__ = MagicMock(return_value=False)

        span = tracing.LangfuseSpan(mock_context_manager)

        # Test input
        span.set_content_tag("test.input", "foo")
        mock_coerce.assert_called_with("foo")
        mock_span.update.assert_called_with(input="foo")
        self.assertEqual({"test.input": "foo"}, span._data)

        span.set_content_tag("test.output", "bar")
        mock_coerce.assert_called_with("bar")
        mock_span.update.assert_called_with(output="bar")
        self.assertEqual({"test.input": "foo", "test.output": "bar"}, span._data)


class TestRavelinSpanHandler(unittest.TestCase):
    def _create_mock_tracer(self):
        """Helper to create a mock tracer with context manager methods."""
        mock_tracer = MagicMock()

        def create_context_manager():
            cm = MagicMock()
            mock_span = MagicMock()
            cm.__enter__ = MagicMock(return_value=mock_span)
            cm.__exit__ = MagicMock(return_value=False)
            return cm

        mock_tracer.start_as_current_observation = MagicMock(side_effect=lambda **kwargs: create_context_manager())
        mock_tracer.start_as_current_span = MagicMock(side_effect=lambda **kwargs: create_context_manager())
        return mock_tracer

    def test_create_span_root_trace(self):
        """Test creating a root span (no parent)."""
        handler = tracing.RavelinSpanHandler()
        handler.tracer = self._create_mock_tracer()

        context = MagicMock(spec=SpanContext)
        context.parent_span = None
        context.trace_name = "test_trace"
        context.operation_name = "haystack.pipeline.run"
        context.public = False

        # Set up tracing context
        token = tracing_context_var.set(
            {"user_id": "test_user", "session_id": "test_session", "trace_id": "test_trace_id"}
        )
        try:
            result = handler.create_span(context)

            # Assert tracer was called with correct parameters
            handler.tracer.start_as_current_observation.assert_called_once_with(
                trace_context={"trace_id": "test_trace_id"},
                name="test_trace",
                version=None,
                as_type="span",
            )
            call_kwargs = handler.tracer.start_as_current_observation.call_args[1]
            self.assertEqual("test_trace", call_kwargs["name"])
            self.assertEqual("span", call_kwargs["as_type"])

            # Assert result is a LangfuseSpan
            self.assertIsInstance(result, tracing.LangfuseSpan)
        finally:
            tracing_context_var.reset(token)

    def test_create_span_agent(self):
        """Test creating an agent span."""
        handler = tracing.RavelinSpanHandler()
        handler.tracer = self._create_mock_tracer()

        context = MagicMock(spec=SpanContext)
        context.parent_span = None
        context.trace_name = "agent_trace"
        context.operation_name = "haystack.agent.run"
        context.public = False

        token = tracing_context_var.set({})
        try:
            handler.create_span(context)

            call_kwargs = handler.tracer.start_as_current_observation.call_args[1]
            self.assertEqual("agent", call_kwargs["as_type"])
        finally:
            tracing_context_var.reset(token)

    def test_create_span_child_generator(self):
        """Test creating a child span for a Generator component."""
        handler = tracing.RavelinSpanHandler()
        handler.tracer = self._create_mock_tracer()

        context = MagicMock(spec=SpanContext)
        context.parent_span = MagicMock()  # Has parent
        context.name = "generator_span"
        context.component_type = "OpenAIChatGenerator"
        context.operation_name = "component.run"

        result = handler.create_span(context)

        handler.tracer.start_as_current_observation.assert_called_once()
        call_kwargs = handler.tracer.start_as_current_observation.call_args[1]
        self.assertEqual("generator_span", call_kwargs["name"])
        self.assertEqual("generation", call_kwargs["as_type"])
        self.assertIsInstance(result, tracing.LangfuseSpan)

    def test_create_span_child_retriever(self):
        """Test creating a child span for a Retriever component."""
        handler = tracing.RavelinSpanHandler()
        handler.tracer = self._create_mock_tracer()

        context = MagicMock(spec=SpanContext)
        context.parent_span = MagicMock()
        context.name = "retriever_span"
        context.component_type = "InMemoryRetriever"
        context.operation_name = "component.run"

        handler.create_span(context)

        call_kwargs = handler.tracer.start_as_current_observation.call_args[1]
        self.assertEqual("retriever", call_kwargs["as_type"])

    def test_create_span_child_default(self):
        """Test creating a child span for an unknown component type."""
        handler = tracing.RavelinSpanHandler()
        handler.tracer = self._create_mock_tracer()

        context = MagicMock(spec=SpanContext)
        context.parent_span = MagicMock()
        context.name = "default_span"
        context.component_type = "SomeUnknownComponent"
        context.operation_name = "component.run"

        handler.create_span(context)

        # Should use start_as_current_span for unknown types
        handler.tracer.start_as_current_span.assert_called_once()
        call_kwargs = handler.tracer.start_as_current_span.call_args[1]
        self.assertEqual("default_span", call_kwargs["name"])

    def test_create_span_tracer_not_initialized(self):
        """Test that RuntimeError is raised when tracer is None."""
        handler = tracing.RavelinSpanHandler()
        handler.tracer = None

        context = MagicMock(spec=SpanContext)

        with self.assertRaises(RuntimeError) as ctx:
            handler.create_span(context)

        self.assertIn("Tracer is not initialized", str(ctx.exception))

    @patch("ravelin.utils.serialization.coerce_tag_value")
    def test_handle_pipeline_level(self, mock_coerce):
        handler = tracing.RavelinSpanHandler()
        span = MagicMock(spec=tracing.LangfuseSpan)

        # Mock span.get_data() to return pipeline input/output data
        span.get_data.return_value = {_PIPELINE_INPUT_KEY: "input_data", _PIPELINE_OUTPUT_KEY: "output_data"}

        # Mock coerce function to return the input unchanged
        mock_coerce.side_effect = lambda x: x

        handler.handle(span, "SomeComponent")

        # Check if coerce was called for both input and output
        mock_coerce.assert_any_call("input_data")
        mock_coerce.assert_any_call("output_data")

        # Check if span.raw_span().update_trace was called with the coerced values
        span.raw_span.return_value.update_trace.assert_called_once_with(input="input_data", output="output_data")

    @patch("ravelin.utils.tracing.normalize_usage_for_langfuse")
    def test_handle_chat_generator(self, mock_normalize):
        handler = tracing.RavelinSpanHandler()
        span = MagicMock(spec=tracing.LangfuseSpan)

        # Create a mock chat message with metadata
        completion_time = "2021-07-27T16:02:08.012345"
        usage_data = {"input_tokens": 123, "output_tokens": 456}
        model_name = "gpt-4-turbo"

        chat_message = ChatMessage.from_assistant(
            "response", meta={"completion_start_time": completion_time, "usage": usage_data, "model": model_name}
        )

        # Mock span.get_data() to return component output with replies
        span.get_data.return_value = {_COMPONENT_OUTPUT_KEY: {"replies": [chat_message]}}

        # Mock normalize to return sanitized usage
        mock_normalize.return_value = {"input_tokens": 123, "output_tokens": 456}

        # Call handle with a chat generator component type
        handler.handle(span, "ChatGenerator")

        # Check normalize was called with usage data
        mock_normalize.assert_called_once_with(usage_data)

        # Check if span.raw_span().update was called with the expected metadata
        expected_model = "gpt-4-turbo"
        expected_time = datetime.datetime.fromisoformat(completion_time)

        span.raw_span().update.assert_called_once_with(
            usage_details={"input_tokens": 123, "output_tokens": 456},
            model=expected_model,
            completion_start_time=expected_time,
        )

    @patch("ravelin.utils.tracing.normalize_usage_for_langfuse")
    def test_handle_chat_generator_invalid_time(self, mock_normalize):
        handler = tracing.RavelinSpanHandler()
        span = MagicMock(spec=tracing.LangfuseSpan)

        # Create a mock chat message with invalid completion time
        usage_data = {"input": 123, "output": 456}
        chat_message = ChatMessage.from_assistant(
            "response",
            meta={
                "completion_start_time": "invalid_time",
                "usage": usage_data,
                "model": "model-name",
            },
        )

        # Mock span.get_data() to return component output with replies
        span.get_data.return_value = {_COMPONENT_OUTPUT_KEY: {"replies": [chat_message]}}

        mock_normalize.return_value = usage_data

        # Call handle with a chat generator component type
        with patch("ravelin.utils.tracing.logger") as mock_logger:
            handler.handle(span, "ChatGenerator")

            # Check if error was logged for invalid time format
            mock_logger.error.assert_called_once_with("Invalid completion start time format: %s", "invalid_time")

        # Check if span.raw_span().update was called with None for completion_start_time
        span.raw_span().update.assert_called_once_with(
            usage_details=usage_data, model="model-name", completion_start_time=None
        )

    @patch("ravelin.utils.tracing.normalize_usage_for_langfuse")
    def test_handle_non_chat_generator(self, mock_normalize):
        """Test handling non-chat Generator components."""
        handler = tracing.RavelinSpanHandler()
        span = MagicMock(spec=tracing.LangfuseSpan)

        usage_data = {"prompt_tokens": 100, "completion_tokens": 50}
        span.get_data.return_value = {_COMPONENT_OUTPUT_KEY: {"meta": [{"usage": usage_data, "model": "text-davinci"}]}}

        mock_normalize.return_value = usage_data

        handler.handle(span, "SomeGenerator")

        mock_normalize.assert_called_once_with(usage_data)
        span.raw_span().update.assert_called_once_with(usage_details=usage_data, model="text-davinci")


def test_enable_content_tracing():
    with patch("haystack.tracing.tracer") as mock_tracer:
        tracing.enable_content_tracing()
        assert mock_tracer.is_content_tracing_enabled is True


@patch("ravelin.utils.tracing.Langfuse")
@patch("ravelin.utils.tracing.haystack.tracing.enable_tracing")
@patch("ravelin.utils.tracing.LangfuseTracer")
@patch("ravelin.utils.tracing.RavelinSpanHandler")
def test_enable_langfuse_tracing(mock_handler, mock_tracer, mock_enable_tracing, mock_langfuse):
    # Create a mock span handler
    mock_handler_instance = MagicMock()
    mock_handler.return_value = mock_handler_instance

    # Call the function
    tracing.enable_langfuse_tracing("test_name", release="release", public=True)

    # Check that Langfuse was initialized correctly
    mock_langfuse.assert_called_once_with(release="release")

    # Check that LangfuseTracer was created with the right parameters
    mock_tracer.assert_called_once_with(
        tracer=mock_langfuse.return_value, name="test_name", public=True, span_handler=mock_handler_instance
    )

    # Check that tracing was enabled
    mock_enable_tracing.assert_called_once_with(mock_tracer.return_value)


class TestNormalizeUsageForLangfuse(unittest.TestCase):
    def test_normalize_usage_none(self):
        """Test normalization when usage is None."""
        result = tracing.normalize_usage_for_langfuse(None)
        self.assertIsNone(result)

    @patch("ravelin.utils.tracing._sanitize_usage_data")
    def test_normalize_usage_gemini_format(self, mock_sanitize):
        """Test normalization from Gemini format."""
        mock_sanitize.side_effect = lambda x: x

        usage = {"promptTokenCount": 100, "candidatesTokenCount": 200, "totalTokenCount": 300}
        result = tracing.normalize_usage_for_langfuse(usage)

        # Check that keys were normalized
        expected = {"input_tokens": 100, "output_tokens": 200, "total_tokens": 300}
        mock_sanitize.assert_called_once_with(expected)
        self.assertEqual(expected, result)

    @patch("ravelin.utils.tracing._sanitize_usage_data")
    def test_normalize_usage_gemini_partial(self, mock_sanitize):
        """Test normalization from partial Gemini format."""
        mock_sanitize.side_effect = lambda x: x

        usage = {"promptTokenCount": 100}
        result = tracing.normalize_usage_for_langfuse(usage)

        expected = {"input_tokens": 100}
        mock_sanitize.assert_called_once_with(expected)
        self.assertEqual(expected, result)

    def test_normalize_usage_openai_format_not_sanitized(self):
        """Test that OpenAI format is returned as-is without sanitization."""
        # OpenAI format with nested details - should NOT be sanitized
        usage = {
            "prompt_tokens": 100,
            "completion_tokens": 200,
            "total_tokens": 300,
            "prompt_tokens_details": {"cached_tokens": 50},
            "completion_tokens_details": {"reasoning_tokens": 10},
        }
        result = tracing.normalize_usage_for_langfuse(usage)

        # Should return the original usage unchanged
        self.assertEqual(usage, result)

    @patch("ravelin.utils.tracing._sanitize_usage_data")
    def test_normalize_usage_anthropic_format(self, mock_sanitize):
        """Test that Anthropic format is sanitized."""
        mock_sanitize.side_effect = lambda x: x

        usage = {"input_tokens": 100, "output_tokens": 200}
        result = tracing.normalize_usage_for_langfuse(usage)

        # Should be passed to sanitize
        mock_sanitize.assert_called_once_with(usage)
        self.assertEqual(usage, result)

    @patch("ravelin.utils.tracing._sanitize_usage_data")
    def test_normalize_usage_generic_format(self, mock_sanitize):
        """Test that generic format is sanitized."""
        mock_sanitize.side_effect = lambda x: x

        usage = {"input": 100, "output": 200}
        result = tracing.normalize_usage_for_langfuse(usage)

        mock_sanitize.assert_called_once_with(usage)
        self.assertEqual(usage, result)


if __name__ == "__main__":
    unittest.main()
