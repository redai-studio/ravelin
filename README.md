# Ravelin

**Let the flow evolve. Keep the product yours.**

Ravelin adds independently adoptable production mechanisms to the flow your
product needs, while workflow and product semantics remain in your application.
They keep runtime resources portable, reconnect users to live work while
permanent product history stays application-owned, make selected work
recoverable, and let pending input outlive a worker.

Choose the mechanism that matches the boundary you need:

- **Assets** let a serialized component retain a logical reference instead of a
  live provider client, endpoint, and credentials. Each environment resolves
  that reference into a chat or embedding model service, vector database, or
  tool service. When the replacement satisfies the same component contract,
  component wiring and control flow can remain unchanged.
- **Streaming and application journals** separate transient live delivery,
  bounded reconnect, active-generation catch-up, and the durable product
  history your application chooses to preserve.
- **Durable execution and intervention** adapt declared component and step
  boundaries to Restate, reuse recorded results during recovery, and let work
  wait for a person or external system without retaining the original process.

Each mechanism can be adopted on its own, as the corresponding boundary
appears:

| When the product changes... | The boundary that appears... | Attach when needed... |
| --- | --- | --- |
| The same flow reaches another environment, tenant, or provider | Live deployment resources should not become serialized flow structure | Assets |
| Execution continues after the initiating request | A reconnecting client may land on another replica | Shared streaming and catch-up |
| A completed model or tool call becomes too costly to repeat | Recovery needs a declared reusable unit | Restate-backed components or explicit steps |
| A person or external system answers later | The pending continuation must outlive its worker | Durable waits and resolution |
| Results become product records, branches, or audit history | Live delivery and recovery state should not dictate the permanent schema | Application journal hooks |

When combined, each mechanism retains a distinct role: Assets supply runtime
dependencies and provider behavior; Restate protects declared work; streaming
and journal hooks expose lifecycle and activity. The application still decides
what a run, branch, join, decision, or permanent record means.

Follow those transitions through provider changes, parallel work, browser
reconnect, worker replacement, and late approval in
[From prototype to production](docs/user-guide/from-prototype-to-production.rst).

The same mechanisms can also serve different products. A small tool loop, a
branching review workflow, and parallel research can share resource resolution,
delivery, selected recovery, and durable input without sharing one thread,
checkpoint, or product-history schema.

**Current integration.** Ravelin implements these attachment points through
Haystack, whose component and graph scaffolding leaves flow semantics and service
lifecycle to the application. The component-facing APIs, extended data models,
and durable pipeline adapters are Haystack-specific today. Each additional
execution substrate requires a dedicated integration.

## Installation

Ravelin requires Python 3.10 or later.

```bash
pip install ravelin
```

Install optional integrations only when you need them:

```bash
pip install "ravelin[milvus]"
pip install "ravelin[redis]"
pip install "ravelin[restate]"
```

## Quick start: portable resources

Assets are the smallest independent path into Ravelin. This example deliberately
does not require a streaming backend or a durable executor: it shows how one
serialized component can retain a logical resource identity while each
environment supplies the live implementation.

Put the deployment-specific model service in `assets/primary-chat-model.json`.
The credential remains an environment-variable reference rather than a
plaintext value:

```json
{
  "name": "primary-chat-model",
  "type": "model-service",
  "configs": {
    "inference_api_format": "openai-chat",
    "base_url": "https://api.openai.com"
  },
  "secrets": {
    "api_key": {
      "type": "env_var",
      "env_vars": ["OPENAI_API_KEY"],
      "strict": true
    }
  }
}
```

Load that resource into a normal Haystack component:

```python
from ravelin.assets.asset_factory import ModelServiceFactory
from ravelin.components.generators import ChatGenerator
from ravelin.dataclasses import ChatMessage
from ravelin.utils.asset_storage import set_local_asset_storage_path


set_local_asset_storage_path("./assets")
model_service = ModelServiceFactory.load_from_local("primary-chat-model")
generator = ChatGenerator(model_service=model_service, model="gpt-5.6-terra")

# Persist structure, not the live client or its resolved credential.
serialized = generator.to_dict()

# Deserialization resolves the logical resource again in the current environment.
restored = ChatGenerator.from_dict(serialized)
result = restored.run([ChatMessage.from_user("Why make runtime resources portable?")])
print(serialized["init_parameters"]["model_service"])
print(result["replies"][0].text)
```

The component retains an `AssetReference`, so serialized pipeline structure does
not absorb the live service or its credentials. Each environment can provide a
different local definition with the same name. To load definitions from another
configuration system, serialize a reference naming that registered resolver
source.

See [Assets and runtime services](docs/user-guide/assets.rst) for the basic
workflow and [Asset resolution and secret contract](docs/user-guide/asset-resolution.rst)
for the complete protocol.

## Build the production path

When one logical run crosses several lifecycles, live delivery, execution
recovery, pending input, and permanent product history need related identities
without becoming the same store or lifecycle.

[From prototype to production](docs/user-guide/from-prototype-to-production.rst)
walks that path boundary by boundary, including reconnect, process failure, and
late approval. The mechanisms remain independently adoptable:

| When your application needs to... | Ravelin provides... | You continue to control... |
| --- | --- | --- |
| Run the same graph against another provider or environment | [Asset references, resolvers, secrets, and runtime implementations](docs/user-guide/asset-resolution.rst) | Resource-selection and rollout policy |
| Reattach a live UI across refreshes and replicas after execution has been decoupled from the client request | [Shared streaming backends, cursor replay, active-generation catch-up, and relay APIs](docs/user-guide/streaming.rst) | Run lifecycle, product presentation, and canonical state |
| Preserve the history your product needs | [Journal event hooks at documented execution boundaries](docs/user-guide/streaming.rst#the-application-journal) | Schema, retention, authorization, and queries |
| Recover declared work after failure or process replacement | [Restate-backed component invocations and opt-in durable steps](docs/user-guide/durability.rst) | Workflow meaning, commit policy, and external side-effect safety |
| Wait for human or external input | [Durable waits, resolution primitives, and notifier hooks](docs/user-guide/durability.rst#human-and-external-intervention) | Who may respond, what the decision means, and what happens next |

These are independent adoption paths, not maturity stages. A local pipeline can
use only Assets; an application service can use shared streaming without
Restate; a durable workflow can run without publishing live events.

### What the current contracts cover

Ravelin is currently beta software. Its public production contracts are
specific and tested:

- **Assets:** serialized components retain a logical `AssetReference`; a new
  process resolves and validates the live resource again. Environment-secret
  descriptors may cross serialization; resolved token values and provider
  clients do not.
- **Streaming and journals:** a retained cursor is replayed inclusively. After
  that real-time window expires, a subscription catches up to generations that
  are still active and continues with live events; completed history joins from
  application storage. Recoverable execution requires an idempotent or
  Restate-protected journal write.
- **Durability and intervention:** completed component invocations and explicit
  durable substeps can be reused after process failure. Unprotected work inside
  an unfinished component may run again, and external side effects still need
  an application-level safety contract.

The repository tests the contract boundaries behind those statements. Streaming
tests cover late join, inclusive cursor replay, and active-generation catch-up;
Restate tests distinguish successful or terminal completion from an awakeable
suspension or retryable failure, which must not close the live run stream. See
[streaming relay tests](tests/streaming/relay_test.py),
[Redis backend tests](tests/streaming/backends/redis_test.py), and
[Restate application tests](tests/core/restate/app_test.py).

Ravelin is designed for teams that keep flow semantics and permanent product
history application-owned.

Each mechanism brings only the infrastructure appropriate to its contract: use
a shared backend such as Redis for multi-replica reconnectable streaming, keep
permanent product history in application-owned storage, and add Restate when
work needs durable recovery or long-lived waits. [Why Ravelin](docs/user-guide/why-ravelin.rst)
explains the architecture behind that choice.

## Documentation

- [Why Ravelin](docs/user-guide/why-ravelin.rst) — the ownership decision among
  integrated runtimes, bounded application orchestration, and Ravelin's
  composable mechanisms.
- [Choosing the runtime boundary](docs/user-guide/comparison.rst) — how current
  integrated frameworks answer the same design questions, and how their
  execution contracts differ.
- [Architecture and boundaries](docs/user-guide/architecture-boundaries.rst) —
  what the application, execution substrate, and Ravelin each own.
- [Assets and runtime services](docs/user-guide/assets.rst) — a practical first
  asset, reference, and resolver walkthrough.
- [Streaming](docs/user-guide/streaming.rst) — live events, reconnect, and
  backend responsibilities.
- [Durable execution](docs/user-guide/durability.rst) — Restate-backed recovery
  and human-in-the-loop workflows.
- [From prototype to production](docs/user-guide/from-prototype-to-production.rst)
  — one composed service path, its stores, and failure-injection checks.
- [Asset resolution and secret contract](docs/user-guide/asset-resolution.rst)
  — resolver registration, trust boundaries, and serialization behavior.

Build the complete documentation locally with:

```bash
hatch run docs:build
```

## Development

This project uses Hatch for its development environments. From the repository
root, install the development tools and create an environment:

```bash
pip install hatch pre-commit
hatch env create

# Optional: run the configured checks before each commit
pre-commit install
```

The configured hooks include the full test, coverage, type, and lint checks, plus
release-note validation. The Black hook requires Python 3.10 to be available.
Hatch installs the development dependencies and downloads the English and
Chinese spaCy models when creating the environment.

```bash
# Unit tests and total coverage
hatch run test:unit

# Changed-line coverage, types, and lint
hatch run test:diff-cov
hatch run test:types
hatch run test:lint

# Complete validation
hatch run test:all

# Build source and wheel distributions
hatch build
```

`test:unit` and `test:all` exclude integration tests. Run the Restate harness
tests separately with `hatch run test:integration`; they require Docker and
access to the required container images. This command leaves unit-test coverage
unchanged.

Run unit tests before changed-line coverage so that `coverage.xml` is current.
Both coverage checks require at least 90%. Changed-line coverage compares with
`origin/master` by default; pass another available base ref when needed, for
example `hatch run test:diff-cov origin/main`.

Use English for code, comments, and contribution discussions. English is the
default documentation language; translated documentation can be maintained
alongside it. Add type hints and follow the Black formatting configuration in
`pyproject.toml` (120-character lines). Run `hatch run format` to format code.

## Contributing

Keep changes focused and explain their purpose, observable behavior, and how
they were validated. Include regression tests for bug fixes and update relevant
documentation when behavior changes. Use the [issue templates](.github/ISSUE_TEMPLATE/)
and [pull request template](.github/pull_request_template.md) to structure reports
and reviews.

Use Conventional Commits for commit messages:
`<type>[optional scope]: <description>` (for example, `fix(streaming): ...`).
Mark breaking changes with `!`, such as `feat!: ...`, and explain their impact.

For user-visible changes, add a release note in `releasenotes/notes/`:

```bash
hatch run reno new <change-description>
hatch run reno lint
```

Fill in the generated note before running the lint command. Documentation-only
and internal maintenance changes do not need a note unless they affect users.

## License

Ravelin is licensed under the [Apache License 2.0](LICENSE.txt).
