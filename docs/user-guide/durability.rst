Suspendable and Recoverable Pipeline Execution
==============================================

Some LLM pipelines finish inside a single request. Others need to keep going
after the request has ended, after a provider has failed, after a process has
restarted, or after a human has approved the next step.

This guide is about **suspendable and recoverable pipeline execution for
long-running tasks**. The application and its Haystack graph continue to define
what the workflow means. Ravelin uses Restate to supply durable component
invocation, explicit durable substeps, resumable waits, timers, and retry
integration without replacing that workflow with a Ravelin-owned run model.

The granularity is explicit: a completed component invocation is a durable
boundary; work inside an unfinished component is independently protected only
when it uses ``ctx.run_typed()`` or another Restate-aware primitive. External
side effects still require an application or downstream idempotency,
transaction, status-reconciliation, or compensation contract. See
:doc:`architecture-boundaries` for the division of responsibility.

Production Contract at a Glance
-------------------------------

- **Completed component invocations are durable boundaries.** Restate can reuse
  their recorded results after a retry or process replacement instead of
  invoking the component again.
- **An unfinished component is not automatically atomic.** Work inside it can
  run again unless it is expressed through ``ctx.run_typed()`` or another
  Restate-aware primitive.
- **Pipeline inputs must be deterministic.** Restate compares each component
  call with the recorded one on replay. Ravelin forwards values passed between
  components as recorded, whatever their type. Pipeline inputs cannot be
  forwarded that way: they are decoded from the request and encoded again on
  every replay, so the same request must encode to the same JSON in any
  process. The same holds for a sender that returns a non-list value to a
  list-typed socket collecting several senders, which Ravelin decodes to
  flatten. How an application guarantees determinism is its own choice:
  deterministic ``to_dict()``/``from_dict()``, preprocessing before the
  request, or its own Restate call that records the value. Typical violations
  are encoding a set in iteration order and filling an omitted field with a
  fresh value on decode (``uuid4()``, ``datetime.now()``). A mismatch pauses
  the workflow instead of resuming it.
- **External effects keep their own safety contract.** Email, payments, database
  updates, and third-party tool calls still need idempotency, transactions,
  reconciliation, or compensation at the application or downstream boundary.
- **Awakeables preserve a wait, not its product meaning.** The process that
  requested approval can disappear and the workflow can still resume later.
  The application authenticates the responder and defines approve, reject,
  edit, timeout, and audit semantics.
- **The Restate journal is execution infrastructure.** It is not the durable
  conversation or audit history queried by the product UI.

These contracts are intentionally narrower than "the whole workflow is safe."
They make the recovery unit visible so an application can test the precise
failure windows it cares about. See :doc:`from-prototype-to-production` for a
composed failure-path walkthrough.

Why This Is Worth Adding
------------------------

Recoverable execution matters when the cost of starting over is high.

In an LLM application, a single run may include retrieval, several model calls,
tool execution, external API updates, user-visible progress, and human review.
If the process crashes during the last step, rerunning from the beginning may be
slow, expensive, confusing to the user, or unsafe if side effects have already
happened.

Ravelin adapts declared work in those runs to Restate's durable execution
model. That makes the feature useful for:

- Long-running pipelines where work may take seconds, minutes, or hours.
- Pipelines with flaky execution, such as model providers, search APIs, web
  scrapers, tool services, or databases that may fail transiently.
- Pipelines involving external handlers, including human-in-the-loop approvals,
  review queues, webhooks, background workers, or operator actions.
- Agents that call tools repeatedly before producing a final answer.
- Workflows where some errors should pause for a fix instead of immediately
  becoming failed user-visible work.

Human-in-the-loop is one of the most concrete examples. The pipeline can suspend
while a user or reviewer decides what should happen next, then resume from that
point with the same workflow identity. The waiting period is part of the
execution model, not an ad hoc state machine hidden in the product backend.

Execution State Is Not Product History
--------------------------------------

Recoverable execution, live streaming, and application history solve related but
different problems:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Need
     - Use
     - Why
   * - Recover pipeline execution
     - ``RestatePipeline`` and ``create_restate_app``
     - Resume work after failures, restarts, pauses, retries, or external input.
   * - Show live progress
     - Streaming backend and ``StreamingRelay``
     - Deliver tokens, progress, lifecycle events, and short reconnect behavior.
   * - Query product history
     - Application journal or database
     - Store conversation history, audit records, user-visible events, and query state.

The Restate journal is for workflow recovery. Treat it as runtime
infrastructure, not as the product database your UI queries for long-term
history. If the UI or backend needs durable records, persist those through your
application journal or normal storage path. If the pipeline execution itself
must resume safely, use the recoverable execution layer.

The journal hook is strict under every execution model: its failures propagate.
What differs is recovery. Under a plain pipeline, no automatic recovery repairs
events missed because the process crashed. Under ``RestatePipeline``, recovery
may invoke the hook again for the same logical event, so wrap the write in
``ctx.run`` or use an idempotent sink. See :doc:`streaming` for the two routes
and their guarantees.

When To Adopt It
----------------

We recommend adopting recoverable execution for long-running pipelines,
pipelines with flaky execution, or pipelines that depend on external handlers.

It may be more than you need for a short request-response flow. It becomes much
more valuable when a run can outlive a single process, request, network
connection, or user interaction. For live UX without recoverable execution,
streaming may be enough. For workflow recovery, use the Restate-backed layer.

What Ravelin Provides
---------------------------

``RestatePipeline`` adapts the Haystack-style pipeline graph to a durable
workflow. ``create_restate_app`` exposes that pipeline as a Restate workflow plus
an executor service. ``PipelineIOHandler`` keeps the external request/response
contract separate from component-level sockets.

.. list-table::
   :header-rows: 1
   :widths: 28 32 40

   * - Boundary
     - Ravelin structure
     - Why it matters
   * - Pipeline orchestration
     - ``RestatePipeline``
     - Tracks which components are ready, running, completed, retried, or waiting.
   * - Component invocation
     - Executor service from ``create_restate_app``
     - Turns component calls into durable service calls with retry and recovery behavior.
   * - Component internals
     - ``get_restate_context()``, ``ctx.run_typed()``, ``ctx.sleep()``, ``ctx.awakeable()``
     - Lets Restate-aware components make substeps, waits, timers, and side effects durable.
   * - External boundary
     - ``PipelineIOHandler``
     - Keeps HTTP-facing inputs and outputs separate from pipeline wiring.
   * - Live visibility
     - Optional streaming backend and journal hook
     - Emits lifecycle and activity events while Restate owns execution recovery.

The distinction between component invocation and component internals matters. A
component can be called through the durable executor boundary. If that component
does expensive work, performs side effects, or calls tools internally, make
those substeps Restate-aware with ``get_restate_context()`` and
``ctx.run_typed()``, or use built-in Restate-aware components such as
``RestateAgent`` and ``RestateToolInvoker``.

Human and External Intervention
-------------------------------

Human intervention is not just a pause button. It is an integration boundary
between the pipeline and the outside world.

A run may need to ask a user, reviewer, operator, or external service for a
decision. That decision might come in minutes or weeks, or it might never arrive.
The pipeline should be able to suspend while that decision is pending, preserve
durable memory of the current pipeline progress, publish enough context for
someone to act, and then resume with the decision once it arrives.

Restate models this with an **awakeable**. An awakeable is a durable wait point:
the component creates it, receives an ID, and waits until some external system
resolves it. If the worker restarts while waiting, the wait still belongs to the
workflow. The run does not need to poll a database or keep an in-memory callback
alive.

An awakeable is the convenient one-shot form of external coordination. Restate
also provides durable **signals**, addressed by invocation ID and signal name. A
signal can arrive before or after the invocation begins waiting, and repeated
signals can support a capture queue for steering, requested pause, or other
active-run input. Ravelin does not currently wrap signals into a product control
router: the application still defines signal names, input intent, target,
admission, effective boundary, and propagation.

Ravelin adds an approval notifier hook around this mechanism. The notifier
is the active bridge to your product: it can send the awakeable ID and metadata
to a review queue, WebSocket, webhook, notification service, or backend endpoint
that knows how to reach the user. The user or external handler then resolves the
awakeable with an approval payload.

``require_approval(payload_type=...)`` decodes the response using the expected
class. Ordinary dataclasses need no ``from_dict()`` boilerplate: extra fields
are ignored, while missing required fields and incompatible field types fail.
Custom ``to_dict()`` / ``from_dict()`` methods retain precedence and define their
own wire format. Ravelin continues to install its Restate serde extension so
Haystack objects remain supported by default.

Custom ``from_dict()`` methods must accept their declared wire format or raise.
Their errors are propagated instead of falling back to a dataclass constructor.
For ``ToolExecutionApprovalPayload``, ``approved`` must be a boolean. Resolve
with ``approved=False`` for a normal denial; rejecting the awakeable is a
terminal error. A configured approval timeout raises ``ApprovalTimeout``, which
the application can handle separately.

The executor's ``resolve_awakeable`` handler requires a non-empty string
``awakeable_id`` and an explicit ``payload`` key. It validates that envelope,
not the waiter's payload schema; successful submission does not mean the
response was accepted by the waiter. Invalid responses produce a
``TerminalError`` at ``require_approval()``, including responses sent directly
through Restate. A resolved awakeable is a one-shot response, so retrying the
same recorded bytes does not repair a malformed payload.

The notifier runs as a Restate durable substep, so clean replay reuses its
recorded completion. It is still an external side effect: if the process dies
after delivery commits but before Restate records completion, delivery can
repeat. Use the awakeable ID as an idempotency key when duplicate notification
is unacceptable.

The awakeable and notifier provide the mechanism, not the product control
plane. Your application decides who may resolve the wait, how intent and target
are represented, what information is shown, how the action is audited, and what
approve, reject, edit, or response means to the workflow. Ravelin does not infer
those semantics from the arrival of a payload.

This is why the feature is interesting:

- The pipeline can ask for approval without owning the UI or notification
  system.
- The product can decide how approval is displayed, authorized, audited, and
  resolved.
- Approval denial can be normal business logic, not a crashed workflow.
- The same pattern works for external handlers beyond humans, such as webhook
  callbacks, delayed jobs, operator remediation, or third-party review systems.

For generic approval points, use ``require_approval()`` in the awaited async call
tree of a Restate-aware component. Nested async helpers and replay-stable runtime
branches inherit the context. The helper cannot run in a synchronous component,
detached task, or inside ``ctx.run()`` / ``ctx.run_typed()``; an awakeable must be
a handler-level journal entry. For tool approval inside an agent, use
``RestateConfirmationStrategy`` with ``RestateAgent``. It places the gate before
the durable tool invocation rather than inside the tool body.

How It Is Wired
---------------

The minimal shape is intentionally small:

.. code-block:: python

   import restate

   from ravelin.core.restate import RestatePipeline, create_restate_app
   from ravelin.utils.pipeline_io_handler import PipelineIOHandler


   pipeline = RestatePipeline()
   # Add components and connect them as you would for a pipeline.

   io_handler = PipelineIOHandler(
       inputs={"query": {"type": str, "connections": ["retriever.query"]}},
       outputs={"answer": {"type": str, "connection": "generator.replies"}},
   )

   workflow, executor = create_restate_app(
       pipeline,
       io_handler,
       name="ResearchPipeline",
   )

   app = restate.app([workflow, executor])

The important pieces are:

- ``RestatePipeline`` owns durable pipeline orchestration.
- ``PipelineIOHandler`` defines the service boundary for external callers.
- ``create_restate_app`` creates the Restate workflow and executor service.
- The workflow key used by the caller becomes the durable run identity.

In production, you usually combine this with the streaming layer and an
application journal:

- Restate owns workflow recovery.
- The streaming backend owns live delivery and short reconnect.
- Your application journal owns durable product history.

This lets a UI watch a run live while the workflow remains recoverable, and it
lets your product query completed history without reading Restate's recovery
journal as an application database.

Choosing Recovery Behavior
--------------------------

Recoverable execution is useful only if failures are treated deliberately.
Ravelin classifies component failures before handing them back to Restate:

- Transient failures, network errors, timeouts, rate limits, and many unknown
  failures are retryable by default.
- Operator-fixable failures can carry an action such as "check credentials" or
  "fix upstream configuration".
- Non-recoverable failures become terminal errors and stop the workflow.
- A custom ``error_classifier`` can override or refine these decisions for your
  application.

Pass an ``invocation_retry_policy`` to ``create_restate_app`` when you need to
control Restate's retry behavior for the workflow and executor. Use explicit
errors such as ``TransientError``, ``OperatorFixableError``, and
``NonRecoverableError`` when your components know the correct recovery category.

Recommended Shape
-----------------

Use the lightest execution model that matches the workflow:

.. list-table::
   :header-rows: 1
   :widths: 32 38 30

   * - Use case
     - Recommended pieces
     - Suitable for
   * - Short request-response flow
     - Normal pipeline execution
     - Simple calls where restart means rerun.
   * - Live chat or agent UI
     - Streaming backend, relay, and application journal
     - Live progress, reconnect, and durable product history.
   * - Recoverable long-running pipeline
     - ``RestatePipeline``, ``create_restate_app``, and ``PipelineIOHandler``
     - Runs that must survive failure, retries, pauses, or restarts.
   * - Human-in-the-loop or external-handler workflow
     - Recoverable execution, streaming, journal hook, awakeables, and notifier
     - Runs that suspend for approval, callbacks, operator action, or delayed external work.
   * - Durable agent
     - ``RestateAgent`` and ``RestateConfirmationStrategy``
     - Agents that need durable model calls, tool calls, approval, and rich progress events.

The durable runtime should protect execution. Your application should still own
product state: user records, conversation history, audit trails, permissions,
and query models.

Further Reading
---------------

- `Restate signals and external events
  <https://docs.restate.dev/develop/ts/external-events>`__ for the underlying
  awakeable, signal, and workflow-promise primitives.
- `Restate Python SDK 1.0 release
  <https://github.com/restatedev/sdk-python/releases/tag/v1.0.0>`__ for the
  signal API and server compatibility requirements.
- :doc:`why-ravelin` for the overall production-layer philosophy.
- :doc:`streaming` for live delivery, reconnect, and application journaling.
- :class:`ravelin.core.restate.restate_pipeline.RestatePipeline`
- :func:`ravelin.core.restate.app.create_restate_app`
- :class:`ravelin.utils.pipeline_io_handler.PipelineIOHandler`
- :class:`ravelin.components.agents.restate_agent.RestateAgent`
