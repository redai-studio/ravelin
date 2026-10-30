Streaming in Production
=======================

Once execution outlives the request that started it, streaming is no longer just
"send tokens as fast as possible". A production UI also needs progress
visibility during long-running work, reconnect behavior for refreshed or
disconnected clients, and a durable place to read history after the live run has
moved on.

Ravelin is designed around that production shape. The streaming layer
handles the live experience: low-latency delivery, short-term replay, and
active-generation catch-up. Your application journal handles durable history:
the records you keep, the format you query, the retention policy you need, and
the authorization rules your product requires.

Live delivery can stay optimized for current viewers while durable history
follows the product's own schema and lifecycle.

This is a delivery contract, not an application state model. Ravelin does not
decide which events constitute canonical product state, how a UI interprets
them, or how completed history joins a fresh view of active work. The
application owns those semantics and can persist the facts it needs through the
journal hook. See :doc:`architecture-boundaries` for the broader ownership
rules.

Production Contract at a Glance
-------------------------------

- **Live delivery is a side channel.** Backend write failures are logged and do
  not fail component execution. The application decides whether a run continues
  after its initiating client disconnects.
- **Recent replay is inclusive.** If ``from_event_id`` is retained, that event
  is delivered again, followed by later retained events and the live stream.
  Clients need a deduplication or replacement rule at the cursor boundary.
- **Outside the replay window, catch up active work and join history.** Ravelin
  supplies current snapshots for active generations and then continues live.
  Completed generations, prior tool activity, and product state load from the
  application journal.
- **The journal hook is strict; recovery follows execution.** The hook requires
  asynchronous emission and is awaited before a non-token event reaches the
  streaming backend. Its failures propagate under every execution model. A plain
  pipeline has no recovery for events missed because the process crashed;
  recoverable execution can invoke the hook again for the same logical event,
  so the write must be protected by ``ctx.run`` or an idempotent sink.
- **Retention is bounded.** Redis stream length and TTL are operational recovery
  windows, not product-history policy.

The sections below define each boundary in detail. :doc:`from-prototype-to-production`
shows how the live stream, product journal, and recovery journal compose during
disconnects and worker replacement.

Recommended Production Shape
----------------------------

For most production chat or agent UIs, start with this shape:

- ``RedisStreamingBackend`` as the shared streaming backend
- ``StreamingRelay`` in your API layer, usually exposed as SSE or WebSocket
- ``on_journal_event`` to persist the component activity and lifecycle events you
  care about
- A normal history endpoint backed by your application database or event store

This lets each layer do the job it is good at:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Need
     - Use
     - Why
   * - Live tokens and progress
     - Streaming backend + ``StreamingRelay``
     - Low-latency delivery to currently connected clients.
   * - Short disconnect or refresh
     - ``from_event_id`` / ``Last-Event-ID``
     - Replay retained events and continue with the live stream.
   * - Large gap during an active generation
     - Streaming catch-up snapshot
     - Recover accumulated text for in-progress LLM generations when normal replay is no longer possible.
   * - History after completion
     - Application journal
     - Durable storage, retention policy, querying, authorization, and audit.

Live Streaming and Durable History
----------------------------------

The streaming system (backends plus ``StreamingRelay``) is designed for
**immediate subscription and short-term recovery**:

- Delivering events to currently connected clients with low latency
- Allowing clients to reconnect and catch up on recent or in-progress work,
  within the backend's retention limits

Your application journal is the durable companion to that live stream. Once a
component finishes and its short-term recovery window closes, the data needed to
reconstruct what happened during that step should come from the storage layer
you control.

If you need history that survives long disconnects, process restarts, or needs
to be queried days later, persist those records through ``on_journal_event`` or
another application-owned storage path. That keeps live delivery simple while
leaving your product free to choose its own schema, retention, access control,
and audit strategy.

What Reconnect Can Recover
--------------------------

When a client subscribes or reconnects using ``from_event_id`` (typically the
value from the ``Last-Event-ID`` header), the streaming system behaves
differently depending on whether that event is still retained.

Small Gap: Cursor Still Available
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the event matching ``from_event_id`` is still available, the system
re-delivers that event first, followed by subsequent retained events, and then
continues with the live stream. The replay is inclusive so the client has a
clear boundary for deduplication or replacement logic.

Simplified event payload after SSE serialization:

.. code-block:: json

   {
     "event_id": "137",
     "run_id": "run_abc123",
     "component": "llm",
     "call_index": 0,
     "type": "token_stream",
     "subtype": "generation_delta",
     "generation_id": "llm_step_0",
     "payload": { "content": "..." }
   }

The ``generation_id`` is present when the emitting call supplied one (a durable
generation step, e.g. an LLM call wrapped in ``ctx.run``); it is ``null``
otherwise. It distinguishes multiple generations that share one
``(run_id, component, call_index)`` — for example two ``ctx.run`` LLM calls
inside a single agent component.

Fresh Join or Expired Cursor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The backend only keeps a bounded window of history, controlled by settings such
as ``event_log_max_size`` and backend TTL. If ``from_event_id`` is no longer in
that window, or no cursor is provided, the streaming layer cannot reconstruct
all past events by itself.

In this case:

- For each generation still in progress, the system provides a catch-up snapshot
  containing accumulated state from the start of that generation, including how
  many times that generation has retried so far (``retry_count``).
- For completed generations and other past events, the streaming layer provides
  no automatic recovery. Load that context from your application journal.

Catch-up snapshot events are a last-resort mechanism for active LLM generations.
They are a delivery projection of in-progress generation state, not a snapshot
of the full workflow or product. They are not a replacement for durable history.

The ``retry_count`` is worth calling out. Under recoverable execution an LLM call
wrapped in ``ctx.run`` retries silently — Restate owns the retry loop and does not
surface a per-step attempt count. Each failed attempt emits a ``generation_error``
event and increments the snapshot's ``retry_count``; a retry's ``generation_start``
resets the partial content but preserves the count, and ``generation_complete``
clears the whole snapshot. So a client that reconnects *after* the per-attempt
``generation_error`` events have scrolled out of the buffer still learns, from the
snapshot, that an in-flight generation is on (say) its third attempt — a fact that
is otherwise unobservable. This requires the emitting call to have supplied a
``generation_id``; without one, retries of the same logical step cannot be told
apart and the count is not tracked separately.

Example snapshot:

.. code-block:: json

   {
     "event_id": null,
     "run_id": "run_abc123",
     "component": "llm",
     "call_index": 0,
     "type": "token_stream",
     "subtype": "catch_up",
     "generation_id": "llm_step_0",
     "payload": {
       "content": "Accumulated text from the start of this generation...",
       "reasoning_content": "...",
       "tool_call_deltas": [],
       "retry_count": 2
     }
   }

The Application Journal
-----------------------

Implement an ``on_journal_event`` hook when you need:

- History beyond the streaming buffer's retention period
- Previous agent turns or tool calls after a long disconnect
- Audit or compliance records
- A source of truth that your API can combine with live streaming data

The hook receives non-token events, such as component activity and lifecycle
events. Token stream events are intentionally left to the streaming backend —
they are high-frequency and, under recoverable execution, transient (a token
stream is emitted inside a retried step, so it repeats and rolls back with that
step). Persist only the durable events the hook delivers.

Journal hooks require asynchronous emission. For lifecycle and
component-activity events, the hook is awaited before streaming delivery. A hook
failure propagates to the caller; it does not roll back work already performed.
Token events bypass the journal hook. Treat the hook as your strict persistence
path.

Because the hook is async, ``streaming_session()`` rejects ``on_journal_event``
and a synchronous non-token emit raises ``RuntimeError`` while a hook is
configured. Use ``streaming_session_async()`` with ``AsyncPipeline`` or
``RestatePipeline``, and ``emit_async()`` from component code, when you journal
events.

Recovery, and why it differs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The hook is strict in both execution models. What changes is recovery — a
property of the door you opened, not something the journal imposes.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Execution
     - Contract
   * - Plain async pipeline (``AsyncPipeline``)
     - Hook failures propagate. No automatic recovery repairs events missed
       because the process crashed.
   * - Recoverable execution (``RestatePipeline``)
     - Recovery may invoke the hook again for the same logical event.
       Persistence must account for repeated attempts by choosing one of the
       two routes below.

Two routes under RestatePipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 26 44 30

   * - Route
     - How
     - Guarantee
   * - Wrap the hook in ``ctx.run``
     - Persist inside a ``ctx.run`` step so the write becomes a durable step:
       Restate records its completion and skips it on clean replay. Simplest —
       nothing to design.
     - A recorded completion skips the write on replay. A crash between the
       external commit and the recorded completion can repeat the write.
   * - Custom idempotency
     - Give each durable event a stable identity and make your sink idempotent
       (same identity + same payload = no-op; conflicting payload = terminal).
       Closes the crash-window duplicate above.
     - Application-defined deduplicated effect, if identity stability and sink
       atomicity hold.

These are the two recommended routes, and they trade the same axis in opposite
directions:

- **Wrap the hook in** ``ctx.run`` **— for easy setup.** Nothing to design: the
  write becomes a durable step, and a recorded completion skips it on replay. A
  crash between the external commit and the recorded completion can repeat the
  write on recovery.
- **Custom idempotency — for robustness.** Give each durable event a stable
  identity and make your sink idempotent (same identity + same payload = no-op;
  conflicting payload = terminal). This can close the crash window above and
  produce a deduplicated effect, but only under the identity, payload-equivalence,
  and sink-atomicity contract your application defines.

The two compose: a ``ctx.run``-wrapped write *and* an idempotent sink is the
strongest setup. Reach for ``ctx.run`` first; adopt custom idempotency when a rare
duplicate across a crash is unacceptable.

.. note::

   Custom idempotency currently means you supply the event identity yourself.
   Built-in deterministic idempotency-key construction is not yet implemented.
   Therefore Ravelin does not currently claim an exactly-once journal effect.
   Until built-in identity lands, the ``ctx.run`` route is the recommended
   default and custom idempotency requires your own identity scheme and atomic
   sink behavior.

Example:

.. code-block:: python

   from ravelin.streaming import EventType


   async def persist_to_journal(event):
       if event.type == EventType.COMPONENT_ACTIVITY:
           await my_storage.record(
               run_id=event.run_id,
               component=event.component,
               call_index=event.call_index,
               subtype=event.subtype,
               data=event.payload,
           )

When you later build a Relay or history endpoint, combine:

- Recent and in-progress data from the streaming system, including active
  generation snapshots
- Historical data from your own journal

Minimal Wiring Example
----------------------

The relay is framework-agnostic. It yields raw events through ``stream()`` and
SSE-formatted chunks through ``stream_sse()``. In an HTTP framework, pass the
client's ``Last-Event-ID`` header into ``from_event_id`` and return the chunks
as a ``text/event-stream`` response.

.. code-block:: python

   from ravelin.streaming import EventType, StreamingRelay, streaming_session_async
   from ravelin.streaming.backends import RedisStreamingBackend


   backend = RedisStreamingBackend(redis_url="redis://localhost:6379")
   relay = StreamingRelay(backend)


   async def persist_to_journal(event):
       if event.type == EventType.COMPONENT_ACTIVITY:
           await my_storage.record(
               run_id=event.run_id,
               component=event.component,
               call_index=event.call_index,
               subtype=event.subtype,
               data=event.payload,
           )


   async def run_pipeline(run_id, pipeline, inputs):
       async with streaming_session_async(
           run_id,
           backend,
           on_journal_event=persist_to_journal,
       ):
           await pipeline.run_async(inputs)


   async def stream_run_events(run_id, last_event_id=None):
       async for chunk in relay.stream_sse(run_id, from_event_id=last_event_id):
           yield chunk

For example, an SSE endpoint would call ``stream_run_events()`` with the
request's ``Last-Event-ID`` header value and stream the yielded strings to the
client.

Choosing an Approach
--------------------

Use the lightest setup that matches the experience you need.

.. list-table::
   :header-rows: 1
   :widths: 28 42 30

   * - Use case
     - Recommended pieces
     - Suitable for
   * - Live tokens or progress only
     - ``InMemoryStreamingBackend`` or ``CallbackStreamingBackend``
     - Internal tools, scripts, prototypes, or pages where refresh can lose live state.
   * - Live UX with short reconnect
     - ``RedisStreamingBackend`` plus ``StreamingRelay`` and ``from_event_id``
     - Production UIs where users may refresh or briefly lose connectivity.
   * - Durable history and long reconnect
     - Redis streaming, relay, ``on_journal_event``, and a history endpoint backed by your storage
     - Products with conversation history, audit, compliance, or multi-client requirements.

Production Notes
----------------

``RedisStreamingBackend`` accepts ``event_log_max_size`` to bound the retained
event stream. It also accepts ``ttl_seconds`` for Redis stream and snapshot keys;
the default is 24 hours. Set TTL long enough to cover your expected run duration
and any period where clients may still need streaming-layer recovery.

``StreamingRelay`` can also be scoped with ``EventFilter`` so an API can expose
only selected event types or components to clients. A per-call filter can narrow
that scope further.

For completed work, prefer loading history from your application storage instead
of trying to stretch the streaming buffer into permanent storage.

Further Reading
---------------

- :mod:`ravelin.streaming` API reference
- :class:`ravelin.streaming.relay.StreamingRelay`
- :class:`ravelin.streaming.backends.redis.RedisStreamingBackend`
- :func:`ravelin.core.restate.create_restate_app`
- :class:`ravelin.components.agents.restate_agent.RestateAgent`
