"""Context variable management for exposing Restate context to component code."""

import contextvars
from typing import Optional

from haystack.lazy_imports import LazyImport

with LazyImport("Run 'pip install restate_sdk[serde]'") as restate_import:
    from restate import Context

# Context variable to expose Restate context to component code
# This is set by the component executor Service, giving components access to ctx
_restate_ctx: contextvars.ContextVar[Optional["Context"]] = contextvars.ContextVar("restate_ctx", default=None)


def get_restate_context() -> "Context":
    """
    Get the current Restate Context.

    This can be called from within component code to access Restate features:
    - ctx.run() / ctx.run_typed() for durable side effects
    - ctx.awakeable() for human-in-the-loop patterns
    - ctx.sleep() for durable timers

    Note: Components are responsible for their own fine-grained durability.
    Use ctx.run() to wrap non-deterministic operations (API calls, DB queries).

    Example Restate-aware component::

        async def run_async(self, content: str) -> dict:
            ctx = get_restate_context()

            # Durable LLM call
            result = await ctx.run_typed("llm", self._call_llm, content)

            # Human approval (awakeable)
            if self._needs_review(result):
                awakeable_id, promise = ctx.awakeable(type_hint=dict)
                await ctx.run_typed("notify", self._notify, awakeable_id)
                review = await promise
                if not review.get("approved"):
                    raise ValueError("Rejected")

            return {"result": result}

    :raises RuntimeError: If called outside of a Restate execution context.
    :returns: The current Restate Context.
    """
    restate_import.check()

    ctx = _restate_ctx.get()
    if ctx is None:
        raise RuntimeError(
            "No Restate context available. This function must be called "
            "within a RestatePipeline component execution."
        )
    return ctx


def has_restate_context() -> bool:
    """
    Return whether the current execution is inside a Restate component context.

    This helper intentionally does not call ``restate_import.check()`` so code
    paths that merely want to adjust behavior can remain lazy-safe when the
    Restate extra is not installed.
    """
    return _restate_ctx.get() is not None
