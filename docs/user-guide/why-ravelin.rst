Why Ravelin
===========

The Adoption Decision
---------------------

AI application flows change with new models, conventions, and product
requirements. Once work outlives its initiating request, later reconnects,
inputs, or approvals may reach another replica after the original worker is
gone. Long-running AI products need background work, intervention, delegation,
history, and recovery. Each product assembles those capabilities around its own
runtime and deployment model.

Ravelin provides composable production mechanisms for teams that keep workflow
and product semantics in application code or a chosen orchestrator. They form
three independently adoptable pillars:

- **Assets** separate serialized flow structure from the live resource
  implementations supplied by each environment.
- **Streaming and application journals** separate reconnectable live delivery
  from the permanent history the product chooses to preserve.
- **Durable recovery and intervention** let selected work and pending input
  survive beyond the worker that started them.

The pillars address different service boundaries, but share one ownership
choice: they attach around the flow without deciding its orchestration semantics.
When combined, they agree on explicit run, generation, and pending-input
identities. None is a prerequisite for another.

In an early application, the request, execution, live stream, and visible
history can look like one lifecycle. As they separate, the team must decide
which system owns its runs, branches, decisions, and records.

Teams can take either of two orchestration approaches. An integrated runtime
works well when its native thread, run, checkpoint, or workflow objects also fit
the product model and the team wants those concerns integrated there.
Alternatively, the application or a chosen orchestrator can remain authoritative
for workflow and product semantics. If bounded calls can restart safely as
units, the application or chosen orchestrator may already provide enough
recovery. When selected production boundaries need stronger contracts, a team
can implement them directly or compose narrow infrastructure services.
Ravelin packages those contracts into independently adoptable pillars and tests
the shared identities and lifecycle transitions where they meet.

The trigger is an ownership boundary, not branch count. Parallel branches,
subagents, or a long business process need an application or orchestration
framework to define their lineage, joins, control propagation, and authority.
Ravelin preserves the identities and bounded contracts around those decisions.

When Correct Subsystems Disagree
--------------------------------

An earlier Ravelin integration durably suspended execution for an approval and
resumed it correctly. But suspension also wrote an end-of-stream marker. A
reconnecting client saw a completed stream even though the run remained pending
and later continued. Recovery was correct inside one subsystem; the product
lifecycle was wrong across them.

Parallel generations exposed the same class of mistake. Without a distinct
generation identity, two logical generations inside one component could share
catch-up state, and completing one could remove state the other still needed.
The component must own its wait-first, gather-all, cancellation, and
partial-result policy, while delivery and recovery preserve the identities that
policy creates.

Together, these failures show why independent mechanisms still need shared
identities. When the mechanisms meet, they must agree on the run, generation,
pending decision, and lifecycle transition being observed.

Five Production Boundaries
--------------------------

Integrated runtimes can solve each of these inside their native execution
model. Ravelin instead lets a team select the contracts separately around an
application-defined flow.

The Browser Returns Through Another Replica
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A prototype streams directly from the request that started the work. Once the
work moves into a background service, a browser can disconnect while execution
continues and reconnect through a different API replica. The response lifecycle
and execution lifecycle are no longer the same.

Ravelin provides shared live delivery, bounded replay, and catch-up for work
that is still active. The application decides whether the run continues and
joins older completed history from its product store. Given a stable ``run_id``
and independently owned execution, a reconnecting client can subscribe without
restarting the run. See :doc:`streaming` for the cursor, snapshot, and backend
contracts.

The Worker Dies After Expensive Work Finishes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A model call or tool step completes, but the worker disappears before the whole
run returns. Retrying the handler may repeat paid work or an external effect.
Retry, resume, and replay are different contracts: recovery needs an explicit
answer about which completed results are already facts.

If the whole unit is safe to restart, an application orchestrator may already
be sufficient. When selected results must survive process loss, Ravelin maps
declared component and substep boundaries to Restate so recorded results can be
reused. Work outside a completed durable boundary may run again, and external
effects still need application-level idempotency, reconciliation, or
compensation. See :doc:`durability` for the exact recovery units.

The Answer Arrives After the Asking Process Is Gone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An approval or clarification begins as an in-memory callback. It becomes a
different product feature when the decision remains pending across a deployment,
worker replacement, or a long period with no process assigned to the run. The
continuation must belong to a stable pending object rather than the process that
happened to ask.

Ravelin lets awaited application logic create an identity-bound durable wait at
a chosen execution boundary. The application still authenticates the responder
and decides what approval, rejection, editing, timeout, cancellation, or
propagation means. A packaged awakeable covers a one-shot gate; Restate signals
are available when the application needs repeated or nonblocking input. See
:doc:`durability` for placement rules and resolution APIs.

Product History Needs Its Own Shape
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A live buffer serves current viewers. A recovery journal decides which work may
be reused. Product history may need its own schema, retention, authorization,
querying, audit, branching, and deletion policy. These stores can cooperate
without becoming the same record.

Integrated products can make native conversations, runs, and events the
permanent product model when those objects fit. Ravelin exposes journal hooks
for application-owned storage, allowing the application's permanent-history
schema to remain canonical. The application still owns stable event identity,
deduplication, sink atomicity, ordering, and projection rules.

A Deployment Change Rewrites the Flow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A graph moves from development to production, one tenant uses another
deployment, or a service endpoint, provider adapter, vector database, or tool
service configuration changes. If the serialized graph contains a live client,
endpoint, and resolved credential, deployment binding has leaked into execution
structure.

Ravelin components written for the Assets contract retain logical resource
references and resolve supported services in the current environment. A resource
change that satisfies the same component contract need not become a
graph-structure change. Changed behavior, schemas, policies, model selection, or
stored data can still require an explicit compatibility or migration decision.
See :doc:`assets` and :doc:`asset-resolution` for the resolver and secret
contracts.

One Product Lifecycle, Several Systems
--------------------------------------

The number of stores is not the ownership problem. Redis can own a bounded live
delivery window, Restate can own recorded execution progress and pending waits,
and the application database can own the permanent product record. They answer
different questions.

The harder boundary appears when two systems can independently answer the same
lifecycle question: whether a run may continue, which branch is current, whether
a decision still applies, or which result has committed. A framework-native
checkpoint and a product record then become two semantic authorities that need
reconciliation.

Ravelin makes the ownership choice explicit. The application decides branch and
completion semantics, control propagation, approval authority, canonical state,
and external-effect safety. The selected production mechanisms retain their
narrow delivery, recovery, intervention, or resource contracts around those
decisions.

How Mature AI Products Combine These Capabilities
-------------------------------------------------

The public interfaces of mature AI products show these boundaries at product
scale. OpenAI documents `hosted long-running work in ChatGPT
<https://learn.chatgpt.com/docs/long-running-work>`_, including later steering
through the same chat. Its `Codex app-server
<https://learn.chatgpt.com/docs/app-server>`_ separates stored thread history,
streamed agent events, active-turn steering and interruption, and pending
approval requests. `ChatGPT Work and Codex subagents
<https://learn.chatgpt.com/docs/agent-configuration/subagents>`_ add parallel
delegated scopes whose results return to a main conversation.

Claude Code combines these capabilities differently. Its `agentic harness
<https://code.claude.com/docs/en/how-claude-code-works>`_ supports interruption
and steering, resumable sessions, and local, cloud, and remote-control execution
environments. Its `subagents
<https://code.claude.com/docs/en/sub-agents>`_ have independent contexts and can
run in foreground or background within the harness's session model.

Each of these products integrates long-running work, intervention, delegation,
history, and recovery with its own execution model. Ravelin makes a narrower set
of service-side mechanisms available around an application-defined flow.

Adopt Only the Boundary the Product Needs
-----------------------------------------

Ravelin's mechanisms are independently selectable. A local pipeline can use
only Assets. A service can use shared streaming without Restate. A durable
workflow can run without publishing live events, and journal hooks are relevant
when permanent product history belongs in application storage.

A team can start with the simplest path that covers its current failure
boundaries. Adopt a mechanism when the product needs independent control,
recovery, accounting, or observation across a real boundary, and add others as
those needs emerge.

The flow can still evolve in ways that require new identities, attachment
points, durable boundaries, or continuation migrations. Explicit attachment
points make the resulting compatibility decisions visible at the delivery,
recovery, intervention, resource, or product-history boundary they affect.

Current Integration
-------------------

Ravelin implements these attachment points through Haystack today. Haystack's
component and graph scaffolding leaves broad freedom for the application to
define flow semantics and service lifecycle, which makes it a practical first
execution substrate for this design. The current component APIs, data models,
and durable pipeline adapters are Haystack-specific. Each additional execution
substrate requires a dedicated integration.

Evidence and Maturity
---------------------

Ravelin's maintainers use Assets, shared streaming and journal integration, and
Restate-backed durability and waits in production. The public repository tests
the bounded contracts behind reference serialization, secret handling, cursor
replay, late join, active-generation catch-up, generation identity, retry-state
isolation, and the distinction between successful completion, suspension,
retryable failure, and terminal error. Ravelin remains beta software, so teams
must still validate the topology and product policies chosen for each deployment.

Where To Go Next
----------------

- Read :doc:`from-prototype-to-production` to compose the mechanisms in one
  service and walk through their failure windows.
- Read :doc:`architecture-boundaries` for the exact ownership and
  non-implication rules.
- Read :doc:`comparison` for the strongest documented answers in current
  integrated runtimes and the execution models behind them.
- Read :doc:`assets`, :doc:`streaming`, or :doc:`durability` for implementation
  details of the mechanism you need.
