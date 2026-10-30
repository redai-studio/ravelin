"""
Agent components for Ravelin.

This module offers two implementations with different durability and observability profiles:

**Agent**
    A thin, compatible wrapper around Haystack's upstream Agent.

    - Best for: local development, simple agents, or when you want to stay
      maximally aligned with future Haystack Agent improvements.
    - Limitations: The agent loop lives inside Haystack. You get token streaming,
      but not rich per-iteration component activity events (e.g. ``llm_response``,
      ``tool_calls_pending``). No automatic durability.

**RestateAgent**
    A durable agent whose entire loop is owned by this library and executed as
    Restate durable steps.

    - Every LLM call and tool invocation result is recorded (``ctx.run_typed``).
    - Human-in-the-loop via ``confirmation_strategies`` + Restate awakeable waits.
    - Restate-backed recovery: replay reuses recorded step results. A call whose
      result was not recorded may run again, so external side effects need
      idempotency or reconciliation.
    - Emits detailed ``COMPONENT_ACTIVITY`` events at every step boundary
      (usable with the unified streaming system).
    - **Requires** the ``restate`` extra **and** must be used inside a
      ``RestatePipeline`` (regular ``Pipeline`` / ``AsyncPipeline`` will not work):
      ``pip install ravelin[restate]``.

**When to choose which**

- Use ``RestateAgent`` when you care about durability, long-running workflows,
  human approvals, or want high-fidelity streaming of agent progress.
- Use ``Agent`` when you want zero Restate dependency and are happy with the
  simpler (non-durable) execution model.

Both classes share a very similar public surface (``tools``, ``confirmation_strategies``,
``state_schema``, ``generated_messages`` output, etc.) to keep migration cheap.
"""

import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure = {
    "agent": ["Agent"],
    "restate_agent": ["RestateAgent"],
}

if TYPE_CHECKING:
    from ravelin.components.agents.agent import Agent
    from ravelin.components.agents.restate_agent import RestateAgent

    __all__ = ["Agent", "RestateAgent"]
else:
    sys.modules[__name__] = LazyImporter(name=__name__, module_file=__file__, import_structure=_import_structure)
