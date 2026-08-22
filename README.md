# Atrex Kernel Agent Runtime

English | [中文](README.zh.md)

Atrex Kernel Agent Runtime is the trusted Python control plane for self-evolving GPU Kernel
Agents. It bootstraps commit-pinned Optimizer and Evolver repositories, schedules isolated
lineage-local competitions, evaluates Kernel candidates through Agate, records every Agent,
Kernel, Attempt, Session, and measurement, and promotes only Runtime-authoritative results.

The Runtime does not implement an Agent framework. Core and Evolver own their prompts, tools,
workflow, and Claude/Codex/QoderCLI/Pi adapters. Runtime owns immutable provenance, model/backend
binding, capabilities, evaluation policy, persistence, sandbox policy, and promotion.

## Main lifecycle

```text
Campaign bootstrap -> agent-v0 + Kernel v0
        |
        v
Epoch: build Challengers -> run Branch trajectories -> compare Kernels -> select Agent
        |
        v
immutable Evidence -> next Epoch
```

- A Campaign freezes Core/Evolver commits, the Evaluation Contract, Gate Policy, hardware, and DSL
  Lineages.
- Each Lineage evolves an independent Agent (`agent-vN`) and Kernel history (`vN`).
- Each Attempt is a fresh Agent Session and may run multiple exploratory Gateway evaluations.
- Runtime finalization applies correctness, optional Production Gate, and configured Evaluate or
  same-allocation ABBA comparison before retaining a Kernel.
- An Evolver can create, reuse, or evolve from a historical Agent revision. Active and Challenger
  Branches then run concurrently within configured limits.
- Optimizers may query the external GPU Wiki live. Runtime freezes the complete query interaction
  before returning the knowledge; no Agent history or Session trace is uploaded to the Wiki.

## Repository layout

- `src/atrex_runtime/`: trusted Runtime package
- `src/atrex-kernel-agent-core/`: separately versioned Optimizer Git submodule
- `src/atrex-kernel-agent-evolver/`: separately versioned Evolver Git submodule
- `third_party/atrex-bench/`: pinned evaluator source submodule
- `examples/`: self-contained runnable workflows
- `docs/`: design, usage, interface, configuration, operation, and release documentation
- `workspaces/local-wiki/`: development-only wire-compatible GPU Wiki test service

## Quick start

Requirements: Python 3.12+, Git, an Agate endpoint, and one supported Agent CLI. Production
sandboxing additionally requires Linux, bubblewrap, and cgroup v2/systemd. Sandboxed Workers share
the host network directly.

```bash
git submodule update --init --recursive
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'

export AGATE_URL='https://your-agate.example.com'
export AGATE_AK='...'
export AGATE_SK='...'
export AGATE_GPU='H20'
export QODER_PERSONAL_ACCESS_TOKEN='...'

bash examples/bootstrap/run.sh
```

The examples generate isolated configuration and state under `workspaces/`; they do not reuse a
root `runtime.json`. For a real deployment, copy `runtime.example.json`, select `development` only
for trusted local debugging or configure the Linux `sandbox` launcher, then follow the
[User Guide](docs/user-guide.md).

## Documentation

- [Documentation index](docs/README.md)
- [Architecture and trust design](docs/architecture.md)
- [Module design](docs/module-design.md)
- [User Guide](docs/user-guide.md)
- [CLI, HTTP, and Runtime Tools interfaces](docs/interfaces.md)
- [Configuration reference](docs/configuration.md)
- [Performance and correctness gates](docs/performance-gates.md)
- [Deployment and operations](docs/operations.md)
- [Production runner](scripts/production/README.md)
- [Release checklist](docs/release-checklist.md)
- [Runnable examples](examples/README.md)

## Development verification

```bash
.venv/bin/ruff check src tests scripts
.venv/bin/mypy src
.venv/bin/pytest
.venv/bin/python scripts/smoke-wheel-independence.py
```

Core, Evolver, local Wiki, Linux sandbox, and external-service acceptance commands are listed in
[Testing and Production Acceptance](docs/testing-and-acceptance.md). Passing local tests does not
replace target Linux, Agent provider, Agate, Wiki, and GPU acceptance.

License: Apache-2.0.
