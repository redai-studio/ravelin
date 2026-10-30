From Prototype to Production
============================

This guide follows one small application through the service boundaries that
appear when execution outlives a request. The application owns the workflow,
product state, and control plane; this walkthrough composes independently
adoptable Ravelin mechanisms around them.

The Example Service
-------------------

Start with *summarize a document*. It becomes a service that fetches a document,
generates a summary, waits for approval, and emails the result. Users watch the
run in a browser and can return later to see what happened.

That ordinary feature set creates four independent boundaries:

.. list-table::
   :header-rows: 1
   :widths: 26 34 40

   * - Pressure
     - Mechanism
     - Contract
   * - Staging and production use different providers
     - Assets
     - The component retains a logical reference; each environment resolves a
       compatible runtime resource.
   * - A browser refreshes during an independently hosted run
     - Streaming plus application journal
     - Replay a bounded live window, then obtain completed history from the
       application's store.
   * - A worker dies after expensive work
     - Restate-backed execution
     - Reuse completed durable boundaries; explicitly protect important work
       inside an unfinished component.
   * - Approval arrives after the original process is gone
     - Awakeable plus notifier
     - Preserve the wait while the product owns authorization and decision
       semantics.

Each mechanism can be adopted independently. A local batch job might use only
Assets. A live service might use Redis streaming without Restate. A recoverable
background workflow might publish no live events at all. This page composes them
to make their interactions and failure windows visible.

Name the Identities
-------------------

The mechanisms meet only if the application can correlate them. For this
service, keep the identities explicit:

``AssetReference``
   Names a deployment resource such as ``primary-chat-model``. Its source and
   optional version select configuration, not workflow behavior.

``run_id``
   Names the product run. When using ``create_restate_app``, use the same value
   as the Restate workflow key and streaming run ID.

``generation_id``
   Distinguishes a durable model-generation step within the run. It lets active
   catch-up state and retry counts refer to the same logical generation across
   its retries.

Application event identity
   Gives a journal write a stable idempotency key. Ravelin does not yet generate
   that key for every built-in durable event; the application must supply one
   when duplicates are unacceptable.

Do not collapse these into one identifier. They describe a resource, a run, a
logical generation across retries, and a product record respectively.

Compose the Service Boundary
----------------------------

Assets enter through normal components. Persist the component or pipeline and
it retains the reference; the process that loads it resolves the live resource
again. A provider change can therefore be a deployment configuration change
when the replacement satisfies the same component-facing contract. It is not a
promise that providers behave identically or that stored vector data migrates.
See :doc:`assets` for the runnable first example and :doc:`asset-resolution`
for the complete protocol.

For a shared live stream, journal hook, and recoverable pipeline, the outer
wiring has this shape:

.. code-block:: python

   import restate

   from ravelin.core.restate import RestatePipeline, create_restate_app
   from ravelin.streaming.backends import RedisStreamingBackend
   from ravelin.utils.pipeline_io_handler import PipelineIOHandler


   backend = RedisStreamingBackend(
       redis_url="redis://localhost:6379",
       event_log_max_size=10_000,
       ttl_seconds=86_400,
   )

   pipeline = RestatePipeline()
   # Add and connect the application's components.

   io_handler = PipelineIOHandler(
       inputs={"document_id": {"type": str, "connections": ["fetch.document_id"]}},
       outputs={"summary": {"type": str, "connection": "summarize.summary"}},
   )

   workflow, executor = create_restate_app(
       pipeline,
       io_handler,
       name="DocumentSummary",
       streaming_backend=backend,
       on_journal_event=persist_product_event,
   )

   app = restate.app([workflow, executor])

``persist_product_event`` is not supplied by Ravelin. It writes the non-token
facts the product needs to its own database or event store. Under recoverable
execution the hook can see the same logical event more than once. Wrap the
write in a durable ``ctx.run`` step for the simplest setup, use a stable event
identity with an idempotent sink when the crash-window duplicate is
unacceptable, or combine both. :doc:`streaming` defines those guarantees.

Expose the live side through ``StreamingRelay`` and pass the browser's
``Last-Event-ID`` value as ``from_event_id``. The relay does not start, cancel,
or keep the workflow alive. The application service decides whether a run
continues after its initiating request or client connection ends.

Walk the Failure Path
---------------------

1. **The run starts.** The application chooses a ``run_id`` and invokes the
   Restate workflow with that key. Components resolve their Asset references in
   the worker environment. Missing resolver registration, credentials, or
   protocol implementations fail explicitly rather than silently choosing a
   fallback.
2. **The browser disconnects.** Execution follows the application's run-lifecycle
   policy; streaming itself neither cancels nor preserves it. If the workflow
   continues, ``RedisStreamingBackend`` keeps a bounded event window and active
   generation snapshots that another API replica can serve.
3. **The browser reconnects with a retained cursor.** The relay re-delivers the
   cursor event itself, then later retained events, then live events. The client
   deduplicates or replaces the inclusive boundary event by ``event_id``.
4. **The cursor is gone, or the browser has no cursor.** The relay emits catch-up
   snapshots only for generations still active. The API loads completed
   summaries, component facts, and previous turns from the product journal and
   merges them according to application semantics. The snapshot is not a full
   workflow-state reconstruction.
5. **The worker dies.** Restate retries or resumes the invocation. Results from
   completed component invocations and explicit durable substeps can be reused.
   Work inside an unfinished component that was not protected by a
   Restate-aware primitive may run again. Streaming lifecycle events and journal
   writes can therefore repeat across attempts.
6. **The workflow asks for approval.** A Restate-aware component calls
   ``require_approval()``. It runs the notifier as a durable substep with an
   ``awakeable_id`` and application metadata, then the component suspends. As
   with any external effect, the notifier should tolerate duplicate delivery
   across the commit-and-record crash window. No worker or in-memory callback
   must remain alive while the workflow waits.
7. **A later request resolves the decision.** After authenticating and
   authorizing the responder, the product sends ``awakeable_id`` plus the typed
   decision payload to the executor's ``resolve_awakeable`` handler. Normal
   denial is resolved as ``approved=False`` so it remains business logic;
   infrastructure rejection is reserved for terminating the wait.
8. **The run completes.** Active generation snapshots are cleared and the live
   stream closes. The application journal remains the queryable product record;
   the Restate journal remains execution-recovery infrastructure.

What Each Store Owns
--------------------

.. list-table::
   :header-rows: 1
   :widths: 24 36 40

   * - Store
     - Owns
     - Does not replace
   * - Asset configuration source
     - Deployment resource definitions and Secret descriptors selected by an
       ``AssetReference``.
     - Provider rollout policy, behavioral compatibility testing, or migrated
       provider data.
   * - Redis streaming backend
     - Low-latency events, bounded cursor replay, and active-generation
       snapshots across API replicas.
     - Permanent conversation, audit, or product history.
   * - Application journal or database
     - Durable, authorized, queryable product records in an application-defined
       schema.
     - Workflow recovery or automatic execution replay.
   * - Restate journal
     - Recorded execution progress, durable calls and substeps, retries, timers,
       and awakeable waits.
     - The product read model or external-side-effect safety.

Failure-Injection Checklist
---------------------------

Before calling this path production-ready, test the boundaries against the
deployment topology you actually operate:

- Change the resolver's configuration for ``primary-chat-model``, restart the
  worker, and deserialize the same pipeline. Verify that structure is unchanged
  and the resolved resource is the intended one.
- Remove a required environment credential or resolver registration. Verify
  startup or deserialization fails closed with an actionable error.
- Disconnect and reconnect with a retained ``Last-Event-ID``. Verify inclusive
  replay and client deduplication at the boundary.
- Reconnect after the event window expires. Verify that only active generations
  come from catch-up and completed history comes from application storage.
- Route the reconnect to a different API replica. Verify that both replicas use
  the shared Redis backend rather than process memory.
- Kill a worker after a completed component invocation and during an unfinished
  one. Verify which results are reused and which internal work can repeat.
- Crash after a journal write commits but before its durable step records
  completion. Verify the sink's idempotency rule if duplicates are unacceptable.
- Restart every application process while approval is pending, then resolve the
  awakeable through a later authenticated request. Verify approve, deny, timeout,
  duplicate response, and unauthorized response behavior.
- Complete a run, let the streaming TTL expire, and verify the product history
  remains available without reading Redis or Restate as the product database.

Passing these checks does not certify every application built with Ravelin. It
does make the claimed boundaries observable, and it exposes the policies the
application must still supply.

Where To Go Next
----------------

- Read :doc:`why-ravelin` for the operating model behind these mechanisms.
- Read :doc:`architecture-boundaries` for application, Haystack, Ravelin, and
  infrastructure ownership.
- Read :doc:`assets` and :doc:`asset-resolution` for resource portability and
  the Secret contract.
- Read :doc:`streaming` for live delivery, reconnect, catch-up, and application
  journal guarantees.
- Read :doc:`durability` for component boundaries, durable substeps, recovery,
  and external intervention.
