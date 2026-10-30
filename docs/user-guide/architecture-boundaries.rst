Architecture and Responsibility Boundaries
==========================================

Ravelin is designed to be used inside applications with different execution
models. This page states the roles and ownership boundaries that the rest of the
documentation assumes.

The Three Roles
---------------

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Role
     - Responsibilities
   * - Application
     - Workflow and business semantics; canonical product state; actor
       authorization; control policy and propagation; user experience;
       long-term history; and external side-effect policy.
   * - Execution substrate (Haystack today)
     - Component or step contracts, graph construction, connections, loops,
       branches, input and output routing, and execution.
   * - Ravelin
     - Self-contained mechanisms for asset resolution and provider behavior,
       durable execution boundaries and waits, live event delivery, and
       application journal integration.

These responsibilities can interact without collapsing into one role. A
Ravelin event can describe activity at an execution substrate's component
boundary, but Ravelin does not thereby own the application's state machine. A
Restate-backed wait can pause execution, but it does not decide what the
resulting human action means.

Current Integration Boundary
----------------------------

Ravelin's component-facing APIs, extended data models, and pipeline adapters
integrate with Haystack today. Another execution substrate would need a real
integration that maps its resource injection, execution lifecycle, event,
serialization, durable-work, and external-wait boundaries onto the relevant
Ravelin contracts.

The mechanism contracts state which invariants a future integration must
preserve. Until that integration exists, the current components, data models,
and pipeline adapters remain Haystack-specific.

Mechanism-level Invariants
--------------------------

Ravelin owns the invariants explicitly promised by each building block. For
example:

- An ``AssetReference`` remains a serializable dependency reference while a
  resolver supplies environment-specific configuration.
- Secret-bearing asset definitions cross declared resolution and
  serialization boundaries according to the asset contract.
- A streaming backend associates events with a run identifier and applies its
  documented retention, replay, and active-generation catch-up behavior.
- A journal hook is invoked at the documented event boundary and with the
  delivery guarantees of the selected execution model.
- A Restate-backed component invocation or explicit durable step can reuse a
  recorded result during replay.
- An awakeable supplies an identity-bound durable wait and resolution path.

These are local contracts. They remain useful in many different application
models because they do not require Ravelin to reinterpret the entire workflow.

What the Primitives Do Not Imply
--------------------------------

No individual primitive silently provides a broader product contract:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Primitive
     - Does not by itself imply
   * - Pipeline or graph
     - A standard agent loop, run model, multi-agent topology, or application
       control plane.
   * - Awakeable or approval helper
     - Actor authorization, decision policy, audit completeness, cancellation
       propagation, or governance.
   * - Durable component or step
     - Checkpointing of arbitrary code, rollback, or exactly-once effects in an
       external system.
   * - Streaming cursor
     - Canonical execution recovery, permanent event retention, or replay of
       work rather than replay of delivery.
   * - Journal hook
     - A framework-owned product database, schema, retention policy, or query
       model.
   * - Asset resolver
     - Model-routing policy, automatic failover, health management, quota
       policy, or a credential-management service.

Application Integration Is Explicit
-----------------------------------

The application composes these primitives into its own semantics. A human
review flow, for example, commonly combines:

1. Application logic that decides a review is required.
2. A Restate-aware durable wait created inside a Haystack component.
3. A notifier hook that exposes the wait to the product backend.
4. Product authorization and UI that collect a decision.
5. Application logic that interprets the decision and chooses the next
   transition.
6. Optional Ravelin events and an application journal that make the transition
   observable and durable in the product's own data model.

Ravelin makes steps 2, 3, and 6 reusable. It does not absorb steps 1, 4, or 5.

The same rule applies when switching a model, reconnecting a stream, retrying a
component, or recording history: Ravelin provides a mechanism and its boundary;
the application decides why and when to use it.

Reading the Guides
------------------

Each technical guide should be read in four parts:

- **Primitive:** the API or integration point Ravelin provides.
- **Invariant:** what Ravelin promises at that boundary.
- **Application responsibility:** the surrounding semantics the application
  must define.
- **Non-implication:** the larger runtime or product behavior the primitive
  does not claim to provide.

Start with :doc:`assets`, :doc:`streaming`, or :doc:`durability` according to
the cross-cutting problem you need to solve. The modules can be adopted
independently.
