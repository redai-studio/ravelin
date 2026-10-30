"""
This module provides utility functions for enabling and customizing tracing in the application.
"""

import logging
from datetime import datetime
from typing import Any, Dict, Literal, Optional, cast

import haystack.tracing

# from haystack.tracing import utils as tracing_utils
from haystack_integrations.tracing.langfuse.tracer import (
    LangfuseSpan as _LangfuseSpan,
    LangfuseTracer,
    DefaultSpanHandler,
    SpanContext,
    _PIPELINE_INPUT_KEY,
    _PIPELINE_OUTPUT_KEY,
    _COMPONENT_OUTPUT_KEY,
    _sanitize_usage_data,
    ObservationSpanType,
    SpanHandler,
    tracing_context_var,
)
from langfuse import Langfuse
from langfuse.types import TraceContext

from ravelin.utils import serialization as tracing_utils

logger = logging.getLogger(__name__)


class LangfuseSpan(_LangfuseSpan):
    """
    A customized Langfuse span that supports additional metadata tracking.
    """

    def set_content_tag(self, key: str, value: Any) -> None:
        """
        Set a content-specific tag for this span.

        :param key: The content tag key.
        :param value: The content tag value.
        """
        if not haystack.tracing.tracer.is_content_tracing_enabled:
            return
        # Langfuse does fancy rendering of input/output messages, so we need to convert them to the format it expects.
        # However, this will cause all other input/output data to be dropped, and this is not ideal.
        # We will need to find a better way to handle this in the future.
        if key.endswith(".input"):
            coerced_value = tracing_utils.coerce_tag_value(value)
            self._span.update(input=coerced_value)
        elif key.endswith(".output"):
            coerced_value = tracing_utils.coerce_tag_value(value)
            self._span.update(output=coerced_value)

        self._data[key] = value


class RavelinSpanHandler(DefaultSpanHandler):
    """
    A customized span handler that supports additional metadata tracking for multimodal and tool call support.
    """

    def create_span(self, context: SpanContext) -> LangfuseSpan:
        """
        Create a customized LangfuseSpan for the given context.

        This method replicates the parent's span creation logic to return our custom LangfuseSpan class,
        which supports additional metadata tracking for multimodal and tool call support.

        :param context: The span context containing information about the span to create.
        :returns: A customized LangfuseSpan instance.
        :raises RuntimeError: If the tracer is not initialized.
        """
        if self.tracer is None:
            raise RuntimeError(
                "Tracer is not initialized. "
                "Make sure the environment variable HAYSTACK_CONTENT_TRACING_ENABLED is set to true before "
                "importing Haystack."
            )

        # Get external tracing context for root trace creation (correlation metadata)
        tracing_ctx = tracing_context_var.get({})

        if not context.parent_span:
            # Create a new trace when there's no parent span
            root_span_type: Literal["agent", "span"] = (
                "agent" if context.operation_name == "haystack.agent.run" else "span"
            )
            trace_context: Optional[TraceContext] = cast(
                TraceContext, {"trace_id": tracing_ctx.get("trace_id")} if tracing_ctx.get("trace_id") else None
            )
            context_manager = self.tracer.start_as_current_observation(
                trace_context=trace_context,
                name=context.trace_name,
                version=tracing_ctx.get("version"),
                as_type=root_span_type,
            )

            # Create our custom LangfuseSpan which will handle entering the context manager
            span = LangfuseSpan(context_manager)

            # Build and apply trace metadata from context
            trace_attrs = {
                k: v
                for k, v in {
                    "name": context.trace_name,
                    "user_id": tracing_ctx.get("user_id"),
                    "session_id": tracing_ctx.get("session_id"),
                    "version": tracing_ctx.get("version"),
                    "tags": tracing_ctx.get("tags"),
                    "public": context.public,
                }.items()
                if v is not None
            }
            if trace_attrs:
                span._span.update_trace(**trace_attrs)  # pylint: disable=protected-access

            return span

        # Determine span type for child spans based on component type
        span_type: Optional[ObservationSpanType] = None
        if context.component_type == "ToolInvoker":
            span_type = "tool"
        elif context.operation_name == "haystack.agent.run":
            span_type = "agent"
        elif context.component_type and context.component_type.endswith("Retriever"):
            span_type = "retriever"
        elif context.component_type and context.component_type.endswith("Embedder"):
            span_type = "embedding"
        elif context.component_type and context.component_type.endswith("Generator"):
            span_type = "generation"

        if span_type:
            return LangfuseSpan(
                self.tracer.start_as_current_observation(
                    name=context.name, as_type=cast(ObservationSpanType, span_type)
                )
            )

        return LangfuseSpan(self.tracer.start_as_current_span(name=context.name))

    def handle(self, span: _LangfuseSpan, component_type: Optional[str]) -> None:
        """
        Process span data after component execution.

        Adds metadata, metrics, and other information to the span based on the component type.

        :param span: The span to process.
        :param component_type: The type of component that created the span.
        """
        # If the span is at the pipeline level, we add input and output keys to the trace
        at_pipeline_level = span.get_data().get(_PIPELINE_INPUT_KEY) is not None
        if at_pipeline_level:
            coerced_input = tracing_utils.coerce_tag_value(span.get_data().get(_PIPELINE_INPUT_KEY))
            coerced_output = tracing_utils.coerce_tag_value(span.get_data().get(_PIPELINE_OUTPUT_KEY))
            span.raw_span().update_trace(input=coerced_input, output=coerced_output)

        if component_type and component_type.endswith("ChatGenerator"):
            replies = span.get_data().get(_COMPONENT_OUTPUT_KEY, {}).get("replies")
            if replies:
                meta = replies[0].meta
                completion_start_time = meta.get("completion_start_time")
                if completion_start_time:
                    try:
                        completion_start_time = datetime.fromisoformat(completion_start_time)
                    except ValueError:
                        logger.error("Invalid completion start time format: %s", completion_start_time)
                        completion_start_time = None
                usage = meta.get("usage")
                normalized_usage = normalize_usage_for_langfuse(usage)
                model = meta.get("model")
                span.raw_span().update(
                    usage_details=normalized_usage,
                    model=model,
                    completion_start_time=completion_start_time,
                )
        elif component_type and component_type.endswith("Generator"):
            # Handle non-chat generators (e.g., completion generators)
            meta = span.get_data().get(_COMPONENT_OUTPUT_KEY, {}).get("meta")
            if meta:
                usage = meta[0].get("usage")
                normalized_usage = normalize_usage_for_langfuse(usage)
                span.raw_span().update(usage_details=normalized_usage, model=meta[0].get("model"))


def enable_content_tracing() -> None:
    """
    Enable content tracing in the application.
    """
    haystack.tracing.tracer.is_content_tracing_enabled = True


def enable_langfuse_tracing(
    name: str, release: Optional[str] = None, public: bool = False, span_handler: Optional[SpanHandler] = None, **kwargs
) -> None:
    """
    Enable langfuse tracing in the application.

    :param name: the name of the trace to be created.
    :param release: the identifier of the current usage of the trace. Default is None.
    :param public: if public, a public link will be generated for the trace. Default is False.
    :param span_handler: the span handler to be used. Default is None.
    :param kwargs: additional arguments to pass to the Langfuse tracer.
    """
    langfuse_tracer = LangfuseTracer(
        tracer=Langfuse(release=release, **kwargs),
        name=name,
        public=public,
        span_handler=span_handler or RavelinSpanHandler(),
    )
    haystack.tracing.enable_tracing(langfuse_tracer)


def normalize_usage_for_langfuse(usage: Optional[Dict[str, Any]]) -> Dict[str, Any] | None:
    """
    Normalize and sanitize usage data for Langfuse.

    This function converts provider-specific usage formats to Langfuse-compatible keys,
    then applies sanitization to flatten nested dicts and keep only numeric values.

    Langfuse natively supports:
    - OpenAI format: prompt_tokens, completion_tokens, total_tokens
    - Anthropic format: input_tokens, output_tokens
    - Generic format: input, output

    Gemini format (promptTokenCount, candidatesTokenCount) needs normalization.

    :param usage: The usage object from various LLM providers. If None, returns None.
    :returns: A Langfuse-compatible usage dict with only numeric values, or None.
    """
    if usage is None:
        return None

    # Gemini format: {"promptTokenCount": int, "candidatesTokenCount": int, "totalTokenCount": int}
    # Langfuse doesn't recognize these keys, so we normalize them
    if "promptTokenCount" in usage or "candidatesTokenCount" in usage:
        normalized: Dict[str, Any] = {}
        if "promptTokenCount" in usage:
            normalized["input_tokens"] = usage["promptTokenCount"]
        if "candidatesTokenCount" in usage:
            normalized["output_tokens"] = usage["candidatesTokenCount"]
        if "totalTokenCount" in usage:
            normalized["total_tokens"] = usage["totalTokenCount"]
        # Copy any other fields that might be present
        for key, value in usage.items():
            if key not in ("promptTokenCount", "candidatesTokenCount", "totalTokenCount"):
                normalized[key] = value
        return _sanitize_usage_data(normalized)

    # Skip sanitizing OpenAI Usage data (https://github.com/deepset-ai/haystack-core-integrations/issues/2587)
    if "prompt_tokens" in usage and "completion_tokens" in usage and "total_tokens" in usage:
        return usage

    # Anthropic and generic formats are natively supported by Langfuse
    return _sanitize_usage_data(usage)
