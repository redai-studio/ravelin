Assets and Runtime Services
===========================

An LLM application has two kinds of structure:

- **Application behavior** — prompts, retrieval, generation, tools, and how
  components connect in a pipeline.
- **Runtime resources** — the model endpoint, vector database, external tool
  service, credentials, and environment-specific settings used by those
  components.

Ravelin assets keep the second category out of the first. A pipeline component
depends on a stable service interface, while an asset describes the concrete
runtime resource behind that interface.

The asset layer does not decide which resource an application should use, when
to switch it, or what availability and rollout policy applies. It provides the
portable reference, resolution boundary, secret contract, and runtime adapter;
the application or deployment owns resource-selection policy. See
:doc:`architecture-boundaries` for the general responsibility model.

The Four Pieces
---------------

.. list-table::
   :header-rows: 1
   :widths: 22 32 46

   * - Piece
     - Example
     - Responsibility
   * - Asset
     - ``OpenAIChatModelService``
     - A configured runtime adapter for a model, database, or tool service.
   * - ``AssetReference``
     - ``primary-chat-model`` from ``local``
     - A portable coordinate that a serializable component can retain.
   * - Resolver
     - The built-in ``local`` resolver
     - Turns a reference into an asset definition for the current runtime.
   * - ``Secret``
     - ``Secret.from_env_var("MODEL_API_KEY")``
     - Keeps credential values out of serializable configuration.

You can construct an asset directly when you only need it in the current
process. Use a reference and resolver when a component must be serialized or
the same pipeline must run with different resource configuration in another
environment.

Start with a Direct Asset
-------------------------

The shortest path is to construct a service and pass it to a component:

.. code-block:: python

   from haystack.utils import Secret

   from ravelin.assets.model_services.chat_model_services import OpenAIChatModelService
   from ravelin.components.generators import ChatGenerator


   model_service = OpenAIChatModelService(
       name="primary-chat-model",
       base_url="https://api.example.com",
       api_key=Secret.from_env_var("MODEL_API_KEY"),
   )
   generator = ChatGenerator(model_service=model_service, model="my-chat-model")

This object can make calls immediately. It does not yet have an
``AssetReference``, so a component containing it cannot be serialized.

Move the Resource into Local Configuration
------------------------------------------

For a serializable pipeline, put the resource definition in the configured
local asset directory. Save the following as ``primary-chat-model.json``:

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

The JSON stores the environment-variable *name*, never its value. Point
Ravelin at the directory, load the asset, and use it normally:

.. code-block:: python

   from ravelin.assets.asset_factory import ModelServiceFactory
   from ravelin.components.generators import ChatGenerator
   from ravelin.utils.asset_storage import set_local_asset_storage_path


   set_local_asset_storage_path("./assets")
   model_service = ModelServiceFactory.load_from_local("primary-chat-model")
   generator = ChatGenerator(model_service=model_service, model="my-chat-model")

   serialized = generator.to_dict()
   assert serialized["init_parameters"]["model_service"] == {
       "name": "primary-chat-model",
       "source": "local",
       "version": None,
   }

The component stores only the reference. When it is deserialized,
``ModelServiceFactory`` resolves that reference again and reconstructs the
service for the current process.

Use the Same Pipeline in Another Environment
--------------------------------------------

The serialized component above always asks for ``primary-chat-model`` from the
``local`` source. Each environment can configure a different local asset
directory and provide a record with that name:

- Development may point at a local or inexpensive OpenAI-compatible endpoint.
- Staging may use a shared test deployment.
- Production may use a regional endpoint and a different environment variable.

The provider-facing configuration changes, but the ``ChatGenerator`` and its
pipeline connections do not.

If definitions come from a configuration service, database, or secret manager,
register another resolver source instead of copying them into local files. See
:doc:`asset-resolution` for that extension contract.

Secret Rules at a Glance
------------------------

- Asset credentials are Haystack ``Secret`` objects at runtime. A missing
  optional credential may be absent or ``None``.
- Local JSON may contain serialized environment-variable references only.
- Plaintext values in local files are rejected.
- A trusted external resolver may explicitly convert a returned plaintext value
  into a runtime-only token secret.
- Token secrets resolve for runtime use, but cannot be serialized or saved to
  local storage.

These rules make configuration portable without turning an asset file or a
serialized pipeline into a credential store.

Built-In Asset Families
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 24 30 46

   * - Family
     - Selector
     - Used by
   * - Model service
     - ``inference_api_format``
     - Chat generators and model-backed embedding functions.
   * - Vector database
     - ``vector_db_type``
     - Vector database retrievers and embedding-field configuration.
   * - Tool service
     - ``call_api_format``
     - Web search, web scraping, and custom tool components.

Model Endpoints and Client Lifecycle
------------------------------------

Model services use ``base_url`` for the deployment root and ``routes`` for
operation paths. Configure a route override when a deployment uses a different
path or requires an API-version query parameter. The default OpenAI chat route
is ``/v1/chat/completions``; the Responses route is ``/v1/responses``.
Changing endpoints does not make incompatible provider protocols equivalent.

HTTPX-backed chat, embedding, web search and web scraper services reuse their
HTTPX clients across calls instead of creating one per request, which avoids
rebuilding the TLS context and lets connections be pooled. Clients are created
on first use: one synchronous client per service, and one asynchronous client
per event loop the service is used on. Pass ``client_kwargs`` to configure the
clients (for example ``verify``, ``proxy`` or default headers); the per-call
``timeout`` applies to each request.

The application owns the client lifecycle:

- Reuse a service across calls. A long-lived service on a long-lived event
  loop needs no cleanup until shutdown, where you call ``close()`` and await
  ``aclose()`` to release its clients.
- ``aclose()`` closes only the client of the event loop it runs on, because an
  asynchronous client can only be closed on its own loop.
- If you drive a service from short-lived event loops, such as one
  ``asyncio.run`` per call, await ``aclose()`` before each loop ends. A client
  left behind on a closed loop cannot be closed gracefully: the service drops
  it on its next use, and its connections are released only when it is
  garbage collected, with ``ResourceWarning`` messages.
- A closed service can be used again; it creates new clients on demand.

.. code-block:: python

   async def handle(messages):
       try:
           return await chat_model_service.ainfer(messages, model="gpt-4o", timeout=60.0)
       finally:
           await chat_model_service.aclose()  # only needed when the loop is short-lived

   asyncio.run(handle(messages))

Outside Restate, ``max_retries`` controls local request attempts (default 3,
including the initial attempt). Values of 0 or 1 send one request without a
local retry. Inside a Restate context, local retries are disabled; configure
recovery through Restate instead. See :doc:`durability`.

OpenAI-compatible chat endpoints must support the request options you enable.
Streaming requests include ``stream_options.include_usage`` by default. Set
``support_stream_options=False`` when constructing the service if an endpoint
does not support that option. Token streaming currently supports a single
completion per request.

Milvus and Tracing Compatibility
--------------------------------

Install ``ravelin[milvus]`` for the Milvus adapter. It uses PyMilvus 2.5.x and
targets Milvus 2.5.x APIs for sparse search and native hybrid-search grouping;
there is no client-side fallback for older servers. Existing collections and
their embedding fields must match the asset configuration. For sparse fields,
``MilvusVectorDatabase.search_by_data()`` expects raw query text; dense fields
take embedding vectors.

The current tracing integration requires ``langfuse`` and ``langfuse-haystack``
3.x; their upper bounds are part of the package requirements. The ``tracing``
extra adds the OpenTelemetry SDK and OTLP exporter. Usage and model metadata
depend on what the selected provider returns, so validate the fields your
observability setup relies on with that provider.

Configure, Resolve, or Register?
--------------------------------

Use the smallest extension point that matches the change:

- **Change configuration** when the resource speaks an existing protocol and
  only its endpoint, credential, model, collection, or other settings differ.
- **Register a resolver** when asset definitions come from another storage or
  configuration system.
- **Register an asset type** when the resource protocol itself differs — for
  example, a model service has a different request, response, streaming, or
  tool-call format.

Resolver registration and asset-type registration solve different problems.
The technical guide explains both dispatch steps and their failure modes.

Where to Go Next
----------------

- Read :doc:`asset-resolution` for the complete asset record, resolver, factory,
  and secret contract.
- Explore :mod:`ravelin.assets.asset_factory` for built-in factories.
- Explore :mod:`ravelin.assets.model_services` for model-service interfaces and
  protocol adapters.
- Explore :mod:`ravelin.assets.vector_databases` and
  :mod:`ravelin.assets.tool_services` for the other asset families.
- Read :doc:`why-ravelin` for how assets fit with streaming and durable
  execution.
