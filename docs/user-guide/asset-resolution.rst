Asset Resolution and Secret Contract
====================================

This guide specifies the technical contract behind Ravelin assets. Read
:doc:`assets` first if you only need to configure and load a resource. This page
is for authors of configuration integrations, custom resolvers, asset types, and
serialization tooling.

The contract begins after an application has selected an ``AssetReference``.
Ravelin resolves and validates that reference; it does not prescribe the
application's model-routing, failover, experimentation, tenant, or credential-
management policy.

Production Contract at a Glance
-------------------------------

- **Persist the reference, not the live resource.** A serialized component
  retains ``AssetReference(name, source, version)``. The next process resolves
  the reference again; it does not deserialize a provider client or credential
  value.
- **Register before deserialization.** Resolver and custom asset-type
  registration is process-local startup work. Unknown sources, families, and
  protocol selectors fail explicitly.
- **Keep secrets typed.** The local resolver accepts serialized environment-
  variable ``Secret`` descriptors, rejects plaintext and runtime token secrets,
  and never turns the asset directory into a credential store.
- **Treat portability as structural.** Component wiring can remain unchanged
  when the replacement satisfies the component-facing contract. Ravelin does
  not promise provider-equivalent behavior, model quality, data migration, or
  compatible operational limits.

These statements are the Asset layer's production claim. Routing policy,
rollout decisions, and compatibility testing remain application concerns.

The Resolution Lifecycle
------------------------

A serializable component retains an ``AssetReference``, not a live service
object or its credentials. Deserialization follows this path:

.. code-block:: text

   AssetReference(name, source, version)
       -> resolver registry[source]
       -> resolver.resolve(reference)
       -> asset-definition mapping
       -> family factory (model, vector database, or tool)
       -> registered concrete asset type
       -> live asset with the original reference attached

There are two independent dispatch decisions:

1. ``AssetReference.source`` selects **where configuration comes from**.
2. The selector inside ``configs`` selects **which runtime adapter interprets
   that configuration**.

Changing storage does not require a new model or database protocol. Adding a
new protocol does not require a new storage resolver.

Asset Reference Contract
------------------------

An ``AssetReference`` contains:

``name``
   The resolver-specific logical asset name.

``source``
   A non-empty string used as the resolver registry key. It defaults to
   ``local``.

``version``
   An optional resolver-specific version. The reference preserves it, but each
   resolver decides how version lookup works. The built-in local resolver loads
   ``<name>.json`` and does not interpret versions.

The wire representation is deliberately small:

.. code-block:: json

   {
     "name": "primary-chat-model",
     "source": "configuration-service",
     "version": "2026-08-14"
   }

Source values are strings rather than a closed enum so an application can add
private or domain-specific configuration systems without changing Ravelin.

Asset Definition Contract
-------------------------

A resolver returns a mapping with this common shape:

.. code-block:: json

   {
     "name": "primary-chat-model",
     "type": "model-service",
     "configs": {
       "inference_api_format": "openai-chat",
       "base_url": "https://api.example.com"
     },
     "secrets": {
       "api_key": {
         "type": "env_var",
         "env_vars": ["MODEL_API_KEY"],
         "strict": true
       }
     }
   }

``name``
   The asset's runtime name. A resolver should normally return the logical name
   requested by the reference.

``type``
   The asset family: currently ``model-service``, ``vector-database``, or
   ``tool-service``.

``configs``
   Non-secret constructor configuration. It contains the family selector —
   ``inference_api_format``, ``vector_db_type``, or ``call_api_format`` — plus
   the settings understood by the selected implementation.

``secrets``
   Credential fields. Every present, non-null runtime value must be a Haystack
   ``Secret``. Serialized environment-variable descriptors are rehydrated into
   ``Secret`` objects before construction.

The resolver owns retrieval and any decision to accept plaintext at a trusted
boundary. Resolver helpers and asset deserialization both normalize supported
``Secret`` objects and serialized environment descriptors. The family factory
validates ``type`` and its selector, then the selected asset constructor
validates protocol-specific configuration.

Resolver Interface and Registry
-------------------------------

A resolver structurally implements one method:

.. code-block:: python

   from typing import Any, Mapping

   from ravelin.dataclasses import AssetReference


   class AssetResolver:
       def resolve(self, reference: AssetReference) -> Mapping[str, Any]:
           ...

Register an instance under a source string:

.. code-block:: python

   from ravelin.assets import register_asset_resolver


   register_asset_resolver("configuration-service", ConfigurationServiceResolver())

Registry behavior is intentionally strict:

- ``local`` is reserved for Ravelin's built-in resolver.
- A source must be a non-empty string.
- Registering the same source twice raises instead of silently replacing the
  existing resolver.
- Registration is process-local and should normally happen once during
  application startup, before components or pipelines are deserialized.
- There is no public replacement or unregister operation. Tests that need
  isolation should patch or otherwise isolate process state.

Unknown sources fail before factory construction with an error naming the
unregistered source.

The Built-In Local Resolver
---------------------------

``set_local_asset_storage_path(path)`` selects a directory. The local resolver
loads ``<path>/<reference.name>.json`` and applies the strictest secret policy:

- Serialized environment-variable secrets are accepted and rehydrated.
- Plaintext strings are rejected.
- Runtime token secrets are rejected.
- A missing ``secrets`` mapping is treated as empty.

The local directory is configuration storage, not a secret store. Ravelin does
not create the directory automatically, add encryption, or provide versioned
local files.

Local names must be non-empty relative names. Forward-slash-separated
subdirectories such as ``models/primary-chat-model`` are supported; absolute
paths, empty segments, ``.`` or ``..`` segments, backslashes, colons and NUL
characters are rejected with ``ValueError``. Names are rejected rather than
silently rewritten. Both reads and writes resolve symlinks and require the
resulting file path to remain beneath the configured directory. Symlinks within
that directory, and a symlink used as the configured root itself, are allowed.
Directories must already exist.

These rules apply only to local asset storage; custom resolvers own their name
syntax. The directory and its ancestors must be managed by the application:
path validation does not protect against another process changing filesystem
links between validation and file access.

Secret Normalization
--------------------

``normalize_asset_secrets`` copies the top-level asset mapping and normalizes
its ``secrets`` values. By default it accepts only safe runtime or serialized
representations:

.. list-table::
   :header-rows: 1
   :widths: 32 28 40

   * - Input value
     - Result
     - Default behavior
   * - Haystack ``Secret``
     - Same ``Secret`` object
     - Accepted
   * - Serialized environment descriptor
     - Rehydrated environment ``Secret``
     - Accepted
   * - ``None``
     - ``None``
     - Accepted as an absent optional credential
   * - Plain string
     - Runtime token ``Secret``
     - Rejected unless ``allow_plaintext=True``
   * - Any other value
     - No result
     - Raises ``TypeError``

Allowing plaintext is an explicit trust decision. It is not a default resolver
behavior and is not applied automatically by the registry. A resolver that
retrieves an already-authorized response containing credential strings must opt
in at that boundary:

.. code-block:: python

   from typing import Any, Mapping

   from ravelin.assets import normalize_asset_secrets
   from ravelin.dataclasses import AssetReference


   class ConfigurationServiceResolver:
       def resolve(self, reference: AssetReference) -> Mapping[str, Any]:
           response = fetch_asset_definition(
               name=reference.name,
               version=reference.version,
           )
           return normalize_asset_secrets(response, allow_plaintext=True)

The conversion produces a runtime-only token secret. It does not make the value
safe to persist.

Serialization Rules
-------------------

Secret behavior is deliberately asymmetric:

.. list-table::
   :header-rows: 1
   :widths: 26 28 24 22

   * - Runtime value
     - Resolve for a service call
     - ``asset.to_dict()``
     - Save locally
   * - Environment ``Secret``
     - Reads the environment variable
     - Stores only the descriptor
     - Allowed
   * - Token ``Secret``
     - Returns the token
     - Raises
     - Raises before opening the file
   * - ``None``
     - Returns ``None``
     - Stores ``null`` if present
     - Allowed
   * - Plain string
     - Rejected by supported APIs
     - Rejected
     - Rejected

An asset resolved from a trusted service may therefore contain a token secret
and still be used safely by a serializable component: the component stores the
original ``AssetReference``, not ``asset.to_dict()``. On the next process, the
reference is resolved again.

Do not put credentials under ``configs``. The secret contract applies to the
``secrets`` mapping; arbitrary configuration values are not inspected for
credential material.

Factory Dispatch and Asset-Type Registration
--------------------------------------------

After resolution, a family factory validates the asset ``type`` and chooses a
registered implementation using the selector inside ``configs``:

.. list-table::
   :header-rows: 1
   :widths: 28 32 40

   * - Family factory
     - Selector
     - Registration API
   * - ``ModelServiceFactory``
     - ``inference_api_format``
     - ``register_model_service``
   * - ``VectorDatabaseFactory``
     - ``vector_db_type``
     - ``register_vector_database``
   * - ``ToolServiceFactory``
     - ``call_api_format``
     - ``register_tool_service``

Register a new asset type only when runtime behavior or wire protocol differs.
For example, a model endpoint with a custom request and streaming format needs
a new ``ChatModelService`` implementation. The same OpenAI-compatible protocol
at another URL needs only different configuration.

.. code-block:: python

   from ravelin.assets.model_services import ChatModelService, register_model_service


   @register_model_service("my-chat-protocol")
   class MyChatModelService(ChatModelService):
       ...

The resolver that returns ``my-chat-protocol`` does not need to know how the
class performs inference. The factory and model-service registry own that step.

Complete External-Resolver Example
----------------------------------

The application registers its resolver during startup, then pipeline
deserialization uses it through the normal factory path:

.. code-block:: python

   from typing import Any, Mapping

   from ravelin.assets import normalize_asset_secrets, register_asset_resolver
   from ravelin.assets.asset_factory import ModelServiceFactory
   from ravelin.components.generators import ChatGenerator
   from ravelin.dataclasses import AssetReference


   class ConfigurationServiceResolver:
       def resolve(self, reference: AssetReference) -> Mapping[str, Any]:
           payload = fetch_asset_definition(reference.name, reference.version)
           return normalize_asset_secrets(payload, allow_plaintext=True)


   register_asset_resolver("configuration-service", ConfigurationServiceResolver())

   reference = AssetReference(
       name="primary-chat-model",
       source="configuration-service",
       version="2026-08-14",
   )
   model_service = ModelServiceFactory.load_from_reference(reference)
   generator = ChatGenerator(model_service=model_service, model="my-chat-model")

   assert generator.to_dict()["init_parameters"]["model_service"] == reference.to_dict()

``fetch_asset_definition`` represents application-owned I/O and authentication.
Ravelin only requires the final mapping and the explicit normalization decision.

Failure Modes and Migration
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 34 32 34

   * - Failure
     - Meaning
     - Remedy
   * - No resolver registered for a source
     - Startup registration did not run, or the serialized source is wrong.
     - Register the resolver before deserialization.
   * - Resolver source already registered
     - Startup ran twice or two integrations claimed the same source.
     - Make registration ownership explicit and run it once.
   * - Plaintext local secret rejected
     - An old local asset embeds a credential value.
     - Move the value to an environment variable and store its descriptor.
   * - Token secret cannot serialize
     - Runtime-only credential data reached a persistence operation.
     - Persist the reference, not the resolved asset.
   * - Unsupported asset type or selector
     - The family or concrete protocol is not registered.
     - Correct the record or register the corresponding asset type.
   * - Environment secret cannot resolve
     - A strict environment variable is unset.
     - Supply the variable in the runtime environment.

The Secret-only contract is intentionally breaking for older code that passes
raw strings to asset constructors or stores plaintext in local asset JSON. Wrap
programmatic values in a Haystack ``Secret`` and migrate local files to
environment-variable descriptors.

Supported Mutation Boundary
---------------------------

Constructors, family-specific secret setters, resolver normalization,
deserialization, and serialization validate the contract. ``asset.secrets`` is
currently a public dictionary for compatibility; mutating it directly bypasses
eager validation. Runtime secret resolution and serialization still reject a
plain string, but integrations should use constructors or the named properties
instead of writing the dictionary directly.

API Pointers
------------

- :mod:`ravelin.assets.resolvers`
- :mod:`ravelin.assets.asset_factory`
- :mod:`ravelin.dataclasses.asset_reference`
- :mod:`ravelin.assets.model_services`
- :mod:`ravelin.assets.vector_databases`
- :mod:`ravelin.assets.tool_services`
