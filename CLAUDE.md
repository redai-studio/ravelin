# CLAUDE.md

Guidance for working in this repository.

## What this is

Ravelin is not an agent runtime. It is a Python SDK of modular, cross-cutting
building blocks that help teams build a production-ready agent runtime around
an application-defined flow. The application owns its execution semantics,
product state, service lifecycle, and control plane. Ravelin provides
self-contained asset, intervention, delivery, journal-integration, and
durability mechanisms that can remain robust inside custom workflows.

The current execution integration is
[Haystack](https://haystack.deepset.ai/). It was chosen because its component
graph and workflow scaffolding leave applications in control of flow semantics.
Ravelin's component-facing APIs, extended data models, and pipeline adapters are
therefore Haystack-specific today, but Haystack is not the conceptual boundary
of the mechanism contracts. Do not claim compatibility with another execution
substrate without a real integration. Provider-specific behavior belongs to the
asset model. Ravelin does not prescribe an agent loop, run/task model,
multi-agent topology, canonical product state, or application control policy.
See [docs/user-guide/why-ravelin.rst](docs/user-guide/why-ravelin.rst) and
[docs/user-guide/architecture-boundaries.rst](docs/user-guide/architecture-boundaries.rst)
for the full responsibility model.

Requires Python 3.10+. See `VERSION.txt` for the current version.

## Module layout (`ravelin/`)

- `assets/` — config-driven resource abstraction. Every external resource is an
  asset reference: `model_services/` (chat + embedding, with per-provider
  implementations under `chat_model_services/`: openai, anthropic, google,
  amazon_bedrock), `vector_databases/` (milvus), `tool_services/`
  (websearch, webscraper). `asset_factory.py` loads assets and switches
  providers; `_base.py` holds the core interface. Register new types with
  `register_model_service` / `register_vector_database` / `register_tool_service`.
- `components/` — extended Haystack components: `generators/` (ChatGenerator),
  `agents/` (agent, restate_agent), `retrievers/`, `builders/`, `rankers/`,
  `preprocessors/`, `joiners/`, `tools/` (tool_invoker, restate_tool_invoker,
  web_scraper, websearch), `others/` (custom_function_executor).
- `core/` — `pipeline.py`, `serialization.py`, `type_validation.py`, and
  `restate/` (durable execution: app, restate_pipeline, context, approvals,
  confirmation, errors, serde).
- `streaming/` — reconnectable live events: `context.py` (PipelineContext /
  ComponentContext context vars — presence is the opt-in signal), `event.py`,
  `lifecycle.py`, `relay.py`, and `backends/` (in_memory, redis, callback).
- `dataclasses/` — extended data models: chat_message, streaming_chunk,
  byte_stream, asset_reference, embedding_fields.
- `embedding_functions/` — unified dense/sparse embedding abstraction.
- `tools/` — tool definitions (from_function).
- `utils/` — asset_storage, pipeline_io_handler, registry, serialization,
  streaming, tracing.

Tests mirror this layout under `tests/`.

## Build & dev commands

Uses [Hatch](https://hatch.pypa.io/). See
[README.md](README.md#development) for environment setup and contribution guidance.

```bash
hatch build
hatch run test:unit       # unit tests + total coverage (at least 90%)
hatch run test:diff-cov   # coverage of newly added code (at least 90%)
hatch run test:lint       # pylint + ruff
hatch run test:lint-fix   # auto-fix (black + ruff --fix)
hatch run test:types      # mypy
hatch run test:all        # all of the above
hatch run docs:build      # Sphinx docs -> docs/_build
```

When installed, `pre-commit` runs the configured checks on every commit.

## Conventions

- Code, comments, variable names, and function names are **English only**.
- Add type hints; `mypy` must pass.
- `black` with line-length 120 (configured in `pyproject.toml`).
- Commits follow [Conventional Commits](https://www.conventionalcommits.org/):
  `<type>[scope]: <description>` (feat, fix, docs, refactor, test, perf, revert).
  Append `!` for breaking changes (`feat!`, `fix!`).
- Pair a release note with user-visible changes — generate it with
  `hatch run reno new <change-description>` and fill in the relevant section
  (`features`, `fixes`, `upgrade`, etc.) in the same commit as the change,
  rather than batching notes after. Keep notes in `releasenotes/notes/`.
  Documentation-only and internal maintenance changes do not need a note
  unless they affect users.
