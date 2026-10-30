Comparing Runtime Boundaries
============================

Scope and Method
----------------

Start with :doc:`why-ravelin` for the adoption decision. This page compares the
execution objects and engineering contracts behind similar headline
capabilities, focusing on ownership and failure contracts rather than feature
count. Integrated runtimes and application-owned orchestration suit different
products. In the latter case, a team can build these contracts directly,
compose narrow infrastructure services, or use Ravelin as a packaged
production-boundary layer.

The comparison first identifies each route's execution objects, then examines
human control; continuation, commit, and recovery; and observation and delivery.
A durable run or reconnectable stream does not by itself cover environment,
definition version, trust, resources, or evaluation.

The five sections distinguish live delivery, recovery, intervention, permanent
product history, and resource binding. Each table presents the strongest
documented route for a project and the boundary an adopter still needs to
verify.

Execution Routes Covered
------------------------

.. list-table::
   :header-rows: 1
   :widths: 18 33 49

   * - Route
     - Native center of gravity
     - Product posture supported by the route
   * - Haystack
     - Components, pipelines, loops, and branches embedded in application code
     - The application wants broad freedom over execution composition and is
       prepared to own service lifecycle, durable control, and delivery.
   * - LangGraph OSS and LangSmith Deployment
     - LangGraph threads, graph state, checkpoints, and interrupts; Agent Server
       runs in a Deployment
     - The product can adopt graph checkpoints and thread/run identities as its
       execution lifecycle. Agent Server adds background run control and
       resumable delivery beyond the embedded OSS iterator; LangSmith tracing is
       a separate observation surface.
   * - Dify
     - Published applications and workflows, forms, provider workspaces, UI,
       versions, and logs
     - The product fits a Workflow or Chatflow and benefits from one integrated
       authoring, execution, human-input, and product-observation surface.
   * - DeerFlow
     - Research/coding threads, runs, sandboxes, artifacts, memory, and
       process-scoped subagents
     - The product wants DeerFlow's opinionated long-horizon super-agent
       lifecycle, including its Gateway and product stores.
   * - AgentScope Agent Service
     - Agent state and session persistence, tool-boundary human-input events,
       teams, and workspaces
     - The product wants session-oriented agents, named team members, and
       portable workspace providers, while supplying any stronger durable
       continuation or reconnect contract its service requires.
   * - Agno and AgentOS
     - Database-backed sessions, runs, workflow steps, human requirements, and
       product APIs
     - The product fits Agno's Agent, Team, or Workflow objects and wants
       integrated history, background execution, and workflow review gates.
   * - AG2 Network
     - Participants, Hub channels, ordered envelopes, inboxes, and a write-ahead
       log
     - The application needs addressable participants, durable messaging,
       packet-boundary steering, and explicit at-least-once delivery semantics.
   * - OpenAI Agents SDK
     - Runner execution, tool approvals, serializable paused ``RunState``, tools,
       and handoffs
     - The application wants a compact agent loop with a strong approval
       continuation and is comfortable supplying its service and product stores.
   * - Google ADK
     - Invocations, sessions, events, workflow agents, graphs, and resumability
     - The product can adopt ADK's invocation/session event model and
       ADK-defined workflow semantics, while making consequential tools
       idempotent.

These centers of gravity are different architectural choices, not disguised
versions of the same runtime.

Design Question 1: How Does Live Delivery Survive Disconnects?
--------------------------------------------------------------

Execution must no longer belong to the browser request, and later observers need
a current view without re-executing the run. Long-term history may join that
view from a different store rather than extending the live replay window
forever.

.. list-table::
   :header-rows: 1
   :widths: 21 42 37

   * - Route
     - Native answer
     - Boundary to verify
   * - `LangGraph Agent Server
       <https://docs.langchain.com/langsmith/streaming>`__
     - Agent Server can start background runs. Its resumable **thread** stream
       can continue from ``Last-Event-ID`` and covers all runs on that thread.
     - Joining one specific run is a different API whose earlier output is not
       buffered. Neither contract belongs to an embedded OSS ``stream()``
       iterator.
   * - `Dify streaming
       <https://docs.dify.ai/en/api-reference/guides/streaming>`__
     - A Workflow or Chatflow exposes a persistent workflow-run ID; the client
       can reconnect and request a snapshot of executed node state before live
       events.
     - The documented contract does not provide an event cursor proving lossless
       replay of every transient delta.
   * - `DeerFlow Gateway
       <https://github.com/bytedance/deer-flow/blob/main/backend/docs/STREAMING.md>`__
     - The Gateway provides bounded ``Last-Event-ID`` replay. An explicit gap
       directs the client to reload durable thread, run, and message state.
     - Replay is bounded, and the Gateway/product-store lifecycle is part of the
       adopted runtime.
   * - `AgentScope Agent Service
       <https://docs.agentscope.io/latest/en/deploy/agent-service>`__
     - Agent Service separates serving from agent code and carries a stable
       session ID into AgentScope's persistence model.
     - Its public ``/process`` response is a live SSE stream. Persisted agent
       state does not by itself provide a cursor-based replay contract for the
       stream produced by a worker.
   * - `Agno background execution
       <https://docs.agno.com/background-execution/overview>`__
     - Background streaming assigns sequential ``event_index`` values and can
       replay a bounded in-process buffer when a client resumes the same run.
     - The task and replay buffer are process-local. Persisted sessions and run
       records are a different surface, not a shared delivery cursor that lets
       another replica follow the original live task.
   * - `Google ADK resumability <https://adk.dev/runtime/resume/>`__
     - ADK's Agent Runtime and API server can own a stopped invocation. With
       resumability enabled, the same invocation ID can continue recorded
       workflow progress after a server or worker interruption.
     - Invocation resume recovers execution; it is not a cursor-based replay of
       transient stream deltas. Cross-process survival also requires a
       persistent SessionService; partial streaming events are not that durable
       session record.
   * - Embedded SDKs
     - Haystack and OpenAI Agents SDK expose live iterators or events from which
       an application can build delivery.
     - The application must provide background ownership, a shared log or
       snapshot, reconnect API, retention, and gap behavior.
   * - Ravelin
     - A shared streaming backend supports multiple subscribers, active-
       generation catch-up, and inclusive bounded cursor replay through an
       application endpoint.
     - Once the retained real-time cursor window expires, the subscriber starts
       from current snapshots for active generations and continues live.
       Completed history joins from the application journal rather than
       extending the live buffer indefinitely.

The key difference is who owns delivery state and how clients recover after the
retained window expires. DeerFlow exposes an explicit gap and reloads its
product stores. Ravelin resumes from active-generation snapshots while the
application journal supplies older history.

Parallel Work Changes the Identity Contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Parallel execution is not one capability. A runtime must first say which work
items are independent, which results have committed, and what ordering an
observer may infer.

LangGraph checkpoints a superstep and preserves pending writes from successful
sibling tasks, so a failed sibling need not make them run again. Its stream can
identify subgraph namespaces, but parallel completion order is not a semantic
branch order. ADK resumability similarly records completed ``ParallelAgent``
subagents and successful tool results so unfinished work can be rerun without
repeating all siblings. Other workflow and agent SDKs commonly expose
completion-order events without documenting a separately committed recovery
unit for every sibling.

Ravelin separates two ownership levels:

- **Across Haystack components**, the pipeline and Restate integration assign
  component name and call index and schedule independent component invocations
  as Restate service calls. Reuse of completed calls follows Restate's journal
  contract.
- **Inside one custom component**, the component owns the work it creates. For
  concurrent model generations it must mint a distinct ``generation_id`` for
  each logical generation. It also chooses whether to wait for the first result,
  gather all results, poll, cancel siblings, or accept partial completion.

The delivery layer keeps snapshots for those explicit identities; it does not
invent the component's join or cancellation semantics. Redis retains one
arrival-ordered event stream per run while the cursor window exists, whereas
active-generation catch-up is a set of current snapshots rather than a replay
of one causal order.

Design Question 2: Which Work Is Reused After Recovery?
-------------------------------------------------------

The important question is the commit boundary: which completed results are
facts during recovery, and which operations may execute again?

A hard worker loss is not the only reason execution re-enters code. A run may
also suspend deliberately for human input and resume hours later on another
worker. Clean suspension is a product feature rather than a failure, but it
still exposes the runtime's replay unit: what is restored, what is re-executed,
and which earlier effects must already be idempotent?

.. list-table::
   :header-rows: 1
   :widths: 21 42 37

   * - Route
     - Native answer
     - Boundary to verify
   * - `LangGraph persistence
       <https://docs.langchain.com/oss/python/langgraph/persistence>`__
     - Persistent checkpointers record graph supersteps. Pending writes preserve
       successful siblings, and Functional API tasks can record finer results.
     - An incomplete task may run again. Effects and replayed work still need
       idempotency, reconciliation, or compensation.
   * - `Google ADK resumability <https://adk.dev/runtime/resume/>`__
     - Invocation resumability reinstates completed workflow agents and recorded
       successful tool results; unfinished work is rerun.
     - ADK documents tool execution as at least once. Custom agents must define
       their own savable steps, and the workflow definition must remain
       compatible with the continuation.
   * - `AG2 Network workflow
       <https://docs.ag2.ai/docs/user-guide/network/workflow/>`__
     - A workflow round commits one packet atomically. A mid-packet failure rolls
       back that packet and redispatches its input.
     - Tools inside the packet execute again. AG2 explicitly recommends stable
       idempotency keys for consequential effects.
   * - AgentScope and Agno
     - AgentScope can persist agent state; Agno persists runs, workflow
       requirements, and step outputs in its database.
     - Their reviewed contracts do not establish a general per-operation journal
       that reuses an uncommitted model or tool result after hard worker loss.
   * - `OpenAI Agents SDK HITL
       <https://openai.github.io/openai-agents-python/human_in_the_loop/>`__
     - A serialized ``RunState`` preserves a paused approval continuation, model
       items, usage, and nested-agent resume state.
     - It is not documented as an automatic crash checkpoint after every local
       model or tool effect.
   * - Ravelin with Restate
     - Completed Haystack component service calls are recorded, and selected
       work inside a component can use ``ctx.run_typed()``. ``RestateAgent`` and
       ``RestateToolInvoker`` use finer durable steps for their loops.
     - Undeclared work inside an unfinished boundary may run again. External
       systems still determine whether an effect is idempotent, reconcilable, or
       compensatable.

"Checkpointed," "resumable," and "durable" are incomplete descriptions unless
the unit of replay and the external-effect contract are named.

Crash Recovery and Suspension Share a Boundary, Not an Identity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some systems use the same persistence substrate for both cases without making
them the same operation:

- LangGraph stores graph checkpoints for failure recovery and pending
  interrupts. A dynamic ``interrupt()`` may occur at an application-chosen
  point inside a node, helper, or tool, but resume restarts the containing node.
- Agno stores workflow runs and human requirements, then resumes through
  ``continue_run``. Its workflow and executor gates are declared on workflow
  primitives or tools; that does not establish a per-effect crash journal.
- OpenAI Agents SDK serializes a deliberately paused ``RunState``. The base SDK
  does not automatically persist arbitrary in-flight work before a process
  disappears.
- Ravelin uses Restate journals for both, but not one run-wide journal. The
  pipeline workflow invocation records component service calls; each component
  executor invocation has its own journal. Within a Restate-aware component,
  ``ctx.run_typed()`` records declared work and an awakeable records a one-shot
  pending resolution. The former answers "which work is already committed?"
  and the latter answers "which continuation should this response resolve?"

Keeping those identities separate matters near the gate boundary. An expensive
action can have committed while its approval notification has not, or a gate
can be pending even though no expensive action preceded it.

Design Question 3: Where Can Durable Human Intervention Live?
-------------------------------------------------------------

Several frameworks have a stronger integrated answer than an application should
rebuild casually:

.. list-table::
   :header-rows: 1
   :widths: 21 42 37

   * - Route
     - Native answer
     - Boundary to verify
   * - `LangGraph interrupts
       <https://docs.langchain.com/oss/python/langgraph/interrupts>`__
     - ``interrupt()`` creates a persisted pending continuation addressed by
       thread and interrupt identity; parallel interrupts can be resolved by ID.
     - The containing node restarts on resume. Code before the interrupt must be
       replay-safe, and application or platform policy owns responder authority.
   * - `Dify Human Input
       <https://docs.dify.ai/en/cloud/use-dify/nodes/human-input>`__
     - A Human Input node persists a workflow pause with form, action, timeout,
       and submission tokens.
     - First successful submission closes the form. Email links and API users
       need an authority model appropriate for the product.
   * - `OpenAI Agents SDK HITL
       <https://openai.github.io/openai-agents-python/human_in_the_loop/>`__
     - Tool-call IDs and serializable ``RunState`` preserve approval/rejection;
       nested-agent approvals surface on the outer run.
     - The application owns storage, external addressability, locking, expiry,
       authorization, and compatible agent-version routing.
   * - `Google ADK graph human input
       <https://adk.dev/graphs/human-input/>`__
     - Graph ``RequestInput`` uses interrupt and invocation identity; tool
       confirmation correlates a function response with its call.
     - Survival across replacement depends on enabling resumability and using a
       persistent SessionService. Response correlation is still separate from
       the external-effect and responder-authority contracts.
   * - `AgentScope HITL
       <https://docs.agentscope.io/versions/2.0.7dev/en/building-blocks/agent/human-in-the-loop>`__
     - AgentScope emits ``RequireUserConfirmEvent`` or
       ``RequireExternalExecutionEvent`` at a tool boundary and correlates the
       result with reply and tool-call IDs.
     - The reviewed core contract parks the active reply flow; it does not
       establish a crash-durable pending continuation or shared responder inbox.
       Service deployments must provide that stronger ownership when required.
   * - `Agno workflow HITL
       <https://docs.agno.com/workflows/hitl/overview>`__
     - Workflow requirements are stored in the database and resumed with
       ``continue_run``. Gates exist at step, router, loop, and tool-executor
       boundaries.
     - This is a workflow/run requirement model, not a general participant
       mailbox or active-run control ledger.
   * - `AG2 Network
       <https://docs.ag2.ai/docs/user-guide/network/overview/>`__
     - Stable participants receive envelopes through a Hub channel and can
       reattach through the write-ahead log.
     - The mailbox makes delivery durable; the application still defines which
       envelope resolves which domain continuation.
   * - Ravelin
     - ``require_approval()`` is a shortcut over a Restate awakeable: an
       identity-bound one-shot wait and resolution path that survives the worker
       that created it.
     - It may be placed in the awaited async call tree of a Restate-aware
       component, but not inside ``ctx.run()`` or ``ctx.run_typed()``. Ravelin
       does not define responder authority, expiry, or product decision meaning.

Hooks, Durable Gates, and Active Steering Are Different
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A hook, middleware, or interceptor is an attachment point. It becomes a gated
continuation only when it creates a pending object, preserves the state needed
to resume, and binds the answer to that object. A synchronous callback can
provide excellent local approval without promising survival after its process
is replaced. Conversely, an in-operation request may appear after a normal
``before_tool`` hook and resolve before a corresponding ``after_tool`` hook;
ordinary before/after interception does not by itself model that continuation.

Placement is part of the contract. LangGraph and Restate both permit a dynamic
gate at a runtime-chosen point, but they recover differently. LangGraph saves
graph/task state and restarts the containing node; interrupt order within a task
must remain replay-stable. Restate records the wait as a handler journal entry,
and preceding operations are reused only when they were themselves durable
calls or ``ctx.run`` steps.

Ravelin exposes that Restate context through the awaited async call tree of a
component, so a nested async helper or replay-stable branch may call
``require_approval()`` without becoming a special Haystack node. The boundary is
not arbitrary Python code: synchronous components, detached work, and code
inside a ``ctx.run`` action are outside it. Built-in ``RestateAgent`` tool
approval therefore occurs before the durable tool invocation; a tool body or
agent invoked as a tool cannot currently open a nested awakeable mid-operation.

Other runtimes intentionally choose more declared positions. Dify persists a
Human Input workflow node or deferred agent-tool request. Agno attaches review
requirements to workflow primitives and tool executors. Google ADK uses graph
``RequestInput`` and long-running tool boundaries. OpenAI Agents SDK approvals
belong to tools and agents-as-tools and surface on the outer ``RunState``. Those
positions are often simpler when they cover the product; they become a design
constraint when the pending continuation arises elsewhere.

Resolving a pending gate is also not the same as steering active execution:

- **Resolve a pending continuation** consumes a continuation the runtime already
  declared.
- **Steer, cancel, or request a pause** targets work that is still
  progressing; acceptance and actual application may happen at different times.
- **Queue the next turn** records future input and must not accidentally consume
  a gate or alter the active scope.

LangSmith Deployment's double-texting strategies and AG2 Network's
packet-boundary messages are examples of integrated active-run policies. A
parallel capture queue is another valid design; it need not block the run.
Restate signals are an available substrate for this design: they durably route
one or more named signals to an invocation, including signals that arrive before
the handler begins waiting. Ravelin currently packages the awakeable shortcut
for one-shot gates while leaving the product free to define a router for all
three intents.

Once subagents or parallel branches exist, every operation additionally needs a
target and a propagation rule. Does cancelling a parent cancel children? Does
steering a lead agent invalidate branch results? Can resolving one gate create
side effects or new gates elsewhere? Neither a hook nor a mailbox answers those
questions by itself.

Design Question 4: Who Owns Permanent Product History?
------------------------------------------------------

Integrated products often have a good answer precisely because they own more of
the product lifecycle. Their persistence is not merely an implementation
detail; it defines the conversations, runs, events, checkpoints, requirements,
and feedback that the product can retrieve later:

- Dify owns conversations, workflow runs, node traces, feedback, and pause
  records when the application adopts its product model.
- DeerFlow owns threads, runs, run events, messages, checkpoints, artifacts,
  uploads, and memory for its super-agent product.
- Agno AgentOS persists sessions, runs, requirements, and related records in its
  configured database. AgentScope persists agent state through a configurable
  store, but does not document the same breadth of product-history schema.
- LangGraph threads and checkpoints are execution state. LangSmith traces are
  operational observation. An application must decide whether either schema is
  also its authoritative product history.
- OpenAI Agents SDK Sessions and ADK SessionService persist conversation or
  invocation context. They can become part of a product record, but do not by
  themselves establish the product's retention, authorization, audit, search,
  and branch-presentation contract.

That ownership is a material advantage when the native records are the product
records the application wants. When they are not, the application must either
adopt the framework lifecycle as its domain model or translate it into a
separate authoritative model and keep the two views consistent.

Other routes allow more schema freedom through LangGraph state and Store values,
a custom OpenAI Agents SDK Session, a custom ADK SessionService, AG2 storage
protocols, or Haystack's application-defined stores. The application still maps
execution facts to product facts.

Ravelin separates bounded live delivery from an application-supplied journal
hook. Its callback receives lifecycle facts plus
run, component, and call coordinates and can write directly into the product's
existing database or event model. The application chooses the stable product
event identity, ordering, atomicity, and projection rules in its existing
schema.

Design Question 5: Can Resources Change Without Rewriting the Flow?
-------------------------------------------------------------------

Ravelin Assets separate runtime resource binding from flow code:

.. list-table::
   :header-rows: 1
   :widths: 21 42 37

   * - Route
     - Native answer
     - Boundary to verify
   * - `AgentScope Workspace Manager
       <https://docs.agentscope.io/versions/2.0.7dev/en/deploy/workspace-manager>`__
     - ``WorkspaceManager`` presents one lifecycle over local, container, and
       managed workspace providers.
     - This is a strong execution-environment abstraction. Model, tool, data,
       and credential binding remain separate concerns.
   * - Dify
     - DSL export/import, provider and plugin identities, workspace credentials,
       and environment variables provide a substantial deployment-binding
       layer.
     - Target workspaces still need compatible plugins, datasets, credentials,
       and provider behavior.
   * - DeerFlow
     - ``SandboxProvider`` and configuration bind per-thread workspaces, model
       providers, tools, and MCP services.
     - Sandbox portability does not prove resource or continuation compatibility
       across configurations.
   * - Google ADK and OpenAI Agents SDK
     - Service interfaces, model providers, credentials, sandbox or environment
       tools, and application factories support environment-specific binding.
     - The reviewed APIs do not form one uniform logical reference protocol for
       every external resource.
   * - Ravelin Assets
     - One ``AssetReference`` and resolver contract covers model services,
       embeddings, vector databases, and tool services; asset-type registries
       adapt different provider protocols.
     - The serialized reference retains logical name, source, and version while
       leaving the resolved implementation, configuration, credentials, and
       deployment binding outside the flow structure. It does not prove
       behavioral compatibility, migrate data, or choose rollout and failover
       policy.

The main execution benefit is that a compatible logical resource can use a
different deployment-specific implementation without changing graph structure.
The application still decides whether behavior is compatible with existing
recovery and delivery contracts.
Registering a resolver changes where a definition comes from; registering an
asset type changes the protocol spoken by the resolved resource.

The Composition You Operate
---------------------------

Independent adoption also makes the operational shape explicit:

.. list-table::
   :header-rows: 1
   :widths: 27 32 41

   * - Selected mechanism
     - Required runtime system
     - What it is for
   * - Assets only
     - No additional service for the local resolver
     - Reconstructing logical resources without serializing live clients or
       credential values.
   * - Shared live delivery
     - A shared backend such as Redis for multi-replica use
     - Bounded event replay and active-generation catch-up across API replicas.
   * - Durable work and waits
     - Restate
     - Recording declared results and preserving addressable continuations
       across worker replacement.
   * - Permanent product history
     - The application's database or event store
     - Product retention, authorization, audit, querying, and presentation.

An integrated runtime may fold several of these responsibilities into one
server and schema. Ravelin instead keeps their contracts independently
selectable, so the application can adopt each mechanism where it adds value and
keep its permanent product model authoritative.

Beyond the Five Contracts
-------------------------

The five cases above mostly exercise one run and its production boundaries.
The larger runtime landscape introduces objects Ravelin intentionally does not
own:

.. list-table::
   :header-rows: 1
   :widths: 27 31 42

   * - Added object
     - Strong native examples
     - Contract still worth checking
   * - Delegated child scope
     - AgentScope team workers, Agno team members, DeerFlow subagents, OpenAI
       handoffs or agents-as-tools, and ADK subagents
     - Child identity, independent continuation, cancel propagation, orphan
       behavior, and whether a "background" child actually outlives its parent.
   * - Coordinated branch set
     - LangGraph supersteps and ``Send``, Agno workflow primitives, ADK workflow
       agents, and AG2 transition graphs
     - All/any/quorum joins, stale-result policy, partial commit, replacement,
       and recovery of only part of the set.
   * - Lifecycle-scoped addressable participant
     - AG2 Network Hub participants and AgentScope Agent Service team sessions
     - Incarnation identity, mailbox durability, direct-control semantics, and
       what survives participant replacement.
   * - Remote task or service
     - A2A ``Task`` identity, interrupted task states, cancellation,
       subscription, and Agent Cards
     - A2A deliberately leaves the remote executor opaque. Its task and service
       addressability do not imply an internal effect journal, child lineage,
       join policy, or lossless delivery cursor.
   * - Shared-state collaboration
     - Framework session/team state, DeerFlow artifact revisions, and AG2 channel
       context provide useful pieces
     - Ownership, base version, attributable mutation, conflict detection,
       reconciliation, and stale reads remain separate semantics.
   * - Multiple principals and authorities
     - Platform authorization, RBAC, tool permissions, and approval policy
     - Delegation provenance, revocation effects, conflicts between legitimate
       authorities, and audit across scope lineage.

These questions belong to the application and its execution substrate: child
and branch semantics, participant lifecycle, shared-state policy, authority, and
execution-definition migration. When Ravelin mechanisms attach to such a flow,
they preserve the relevant identities.

Substrates That Shape a Long-Lived Run
--------------------------------------

Execution topology is only part of production behavior. The same run or
participant can become invalid when one of its substrates changes:

.. list-table::
   :header-rows: 1
   :widths: 21 37 42

   * - Substrate
     - Useful native answers
     - Question that remains
   * - Context and memory
     - Framework sessions, conversations, memory services, and summaries
     - Which record is execution state, which is a model-visible projection,
       and how state is copied or shared across scopes?
   * - Sandbox and environment
     - AgentScope workspaces, DeerFlow sandboxes, ADK environment tools, and
       provider-specific code executors
     - Is environment identity preserved across pause and resume, and what does
       cancel, release, or human out-of-band mutation do to it?
   * - Execution definition
     - Platform assistant or application versions, deployment revisions,
       database migrations, and Restate deployment pinning provide pieces
     - Is a pending continuation pinned, migrated, restarted, or forked when
       prompts, graph structure, models, tools, or policy change?
   * - Trust
     - RBAC, guardrails, approval policy, tool permissions, and sandboxing
     - Who authorized the action, what authority reached a child, and does a
       later revocation affect already-running work or only future calls?
   * - Resources and budgets
     - Turn, token, tool-call, recursion, concurrency, and time limits
     - Are budgets merely local caps, or allocated and reclaimed across child
       scopes and participants?
   * - Evaluation
     - Traces, run records, callbacks, and framework-specific eval suites
     - Does evaluation check only answer quality, or also replay, intervention,
       propagation, recovery, and delivery contracts?

Ravelin's Assets cover part of environment binding, Restate pins the deployed
handler used by an invocation, and the streaming and journal contracts expose
facts that can be evaluated. They do not turn these substrate questions into a
universal policy. A long-lived application must answer them at the level where
its product semantics live.

Source Notes
------------

This comparison was reviewed against public first-party documentation and
repositories on 20 August 2026. These projects change quickly; the links below
are part of the claim and should be rechecked when this page is updated.

**Haystack**

- `Pipelines <https://docs.haystack.deepset.ai/docs/pipelines>`__
- `Agents <https://docs.haystack.deepset.ai/docs/agent>`__
- `Serialization <https://docs.haystack.deepset.ai/docs/serialization>`__
- `Haystack 3 migration guide
  <https://github.com/deepset-ai/haystack/blob/main/MIGRATION.md>`__

**LangGraph and LangSmith**

- `Interrupts <https://docs.langchain.com/oss/python/langgraph/interrupts>`__
- `Persistence <https://docs.langchain.com/oss/python/langgraph/persistence>`__
- `Functional API <https://docs.langchain.com/oss/python/langgraph/functional-api>`__
- `Double texting <https://docs.langchain.com/langsmith/double-texting>`__
- `LangSmith streaming <https://docs.langchain.com/langsmith/streaming>`__

**Dify and DeerFlow**

- `Dify Human Input
  <https://docs.dify.ai/en/cloud/use-dify/nodes/human-input>`__
- `Dify streaming guide
  <https://docs.dify.ai/en/api-reference/guides/streaming>`__
- `Dify version control
  <https://docs.dify.ai/en/cloud/use-dify/build/version-control>`__
- `Dify application data models
  <https://github.com/langgenius/dify/blob/main/api/models/model.py>`__
- `DeerFlow streaming contract
  <https://github.com/bytedance/deer-flow/blob/main/backend/docs/STREAMING.md>`__
- `DeerFlow backend architecture
  <https://github.com/bytedance/deer-flow/blob/main/backend/README.md>`__

**AgentScope, Agno, and AG2**

- `AgentScope Agent Service
  <https://docs.agentscope.io/latest/en/deploy/agent-service>`__
- `AgentScope human-in-the-loop
  <https://docs.agentscope.io/versions/2.0.7dev/en/building-blocks/agent/human-in-the-loop>`__
- `AgentScope event identities
  <https://github.com/agentscope-ai/agentscope/blob/main/src/agentscope/event/_event.py>`__
- `AgentScope Workspace Manager
  <https://docs.agentscope.io/versions/2.0.7dev/en/deploy/workspace-manager>`__
- `Agno background execution
  <https://docs.agno.com/background-execution/overview>`__
- `Agno workflow human-in-the-loop
  <https://docs.agno.com/workflows/hitl/overview>`__
- `Agno session storage <https://docs.agno.com/database/session-storage>`__
- `AG2 Network
  <https://docs.ag2.ai/docs/user-guide/network/overview/>`__
- `AG2 Network workflow
  <https://docs.ag2.ai/docs/user-guide/network/workflow/>`__

**OpenAI Agents SDK, Google ADK, and A2A**

- `OpenAI Agents SDK human-in-the-loop
  <https://openai.github.io/openai-agents-python/human_in_the_loop/>`__
- `OpenAI Agents SDK streaming
  <https://openai.github.io/openai-agents-python/streaming/>`__
- `OpenAI Agents SDK sessions
  <https://openai.github.io/openai-agents-python/sessions/>`__
- `Google ADK resumability <https://adk.dev/runtime/resume/>`__
- `Google ADK graph human input
  <https://adk.dev/graphs/human-input/>`__
- `Google ADK session state
  <https://google.github.io/adk-docs/sessions/state/>`__
- `A2A specification <https://a2a-protocol.org/latest/specification/>`__

**Ravelin's durability substrate**

- `Restate durable agents <https://docs.restate.dev/ai/patterns/durable-agents>`__
- `Restate actions <https://docs.restate.dev/foundations/actions>`__
- `Restate signals and external events
  <https://docs.restate.dev/develop/ts/external-events>`__
- `Restate Python SDK 1.0 signal release
  <https://github.com/restatedev/sdk-python/releases/tag/v1.0.0>`__
- `Restate service versioning <https://docs.restate.dev/services/versioning>`__

Where To Go Next
----------------

- :doc:`why-ravelin` for the failure-first project rationale.
- :doc:`architecture-boundaries` for Ravelin's ownership and non-implication
  rules.
- :doc:`durability` for exact recovery and replay boundaries.
- :doc:`streaming` for delivery, replay, gaps, and product-history separation.
- :doc:`assets` for the logical resource and resolver model.
