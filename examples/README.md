# Examples

English | [中文](README.zh.md)

- [`shared/`](shared/README.md): canonical read-only VecAdd fixtures and generic helpers used by
  multiple runnable examples; it is not itself a runnable workflow.
- [`bootstrap/`](bootstrap/README.md): run a real single-DSL VecAdd Campaign Bootstrap through
  Core, Runtime Tools, the configured Agent Backend, and a Agate service.
- [`source-tree/`](source-tree/README.md): prepare a multi-file source Campaign from a GDN-style
  source manifest and an ATREX task, with an immutable adapter and explicit editable roots.
  Includes the same seven enabled Campaign instances as the single-file production ablation suite.
- [`lineage/`](lineage/README.md): bootstrap one Triton VecAdd Lineage and run one Epoch with
  configurable Challenger, Trajectory, and serial Attempt counts.
- [`evolution/`](evolution/README.md): run three Epochs with one Attempt per Branch and create a
  Challenger only after each of the first two Epochs.
- [`optimizer-dev-shell/`](optimizer-dev-shell/README.md): create a disposable Optimizer-compatible
  workspace with live Gateway authority, without Bootstrap, durable lineage state, or an Agent.
- [`evolver-dev-shell/`](evolver-dev-shell/README.md): open a disposable synthetic Evolution
  workspace without Bootstrap, Runtime service, or an Agent process.
- [`agate/`](agate/README.md): call a real Agate service with the official CLI for
  evaluation, profiling, compilation checks, disassembly, development commands, and job control.
- [`local-wiki/`](local-wiki/README.md): start the standalone Local GPU Wiki for browser/API queries.
  The Agent-facing Wiki tool and its shell walkthroughs are temporarily unavailable.
- [`kernel-design-agents/kernel-agent.example.json`](kernel-design-agents/kernel-agent.example.json):
  a `kernel_agent` configuration section for the KDA Optimizer, including approved Skill submodules
  and expanded Bundle limits. This is not a complete Runtime config or a runnable example;
  see [KDA Optimizer](../docs/user-guide.md#kda-optimizer) for usage.

Every Runtime example selects `campaign.optimizer.agent_backend` and
`campaign.evolver.agent_backend` independently in its own `runtime.json`. Supported values are
`claude`, `codex`, `qodercli`, and `pi`; the checked-in Runtime default is QoderCLI for both
Optimizer and Evolver.

Example wrappers resolve `python3`, `atrex-kernel-agent-runtime`, and `agate` from the active
`PATH`. `ATREX_PYTHON`, `ATREX_RUNTIME_CLI`, and `AGATE_BIN` provide explicit overrides. Do not
reuse a macOS virtual environment from a Lima mount; create and activate a Linux environment first.
