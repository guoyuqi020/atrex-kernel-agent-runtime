# Temporary Evolver dev shell

English | [中文](README.zh.md)

This example opens a disposable, sandboxed Evolver-compatible workspace. It does not start the
Runtime HTTP service, Bootstrap, a Campaign, a Lineage, an Epoch, an Optimizer, or the Evolver.

The Runtime imports the pinned Core base as a synthetic `agent-v0`, wraps the configured initial
Evidence, and prepares the normal Evolver layout. No Kernel evaluation is run or fabricated, so
the temporary Runtime Tools Kernel catalog is intentionally empty. The Agent catalog contains the
single active `agent-v0` parent.

## Run

Run on Linux with passwordless `sudo`, `bwrap`, `systemd-run`, and the configured Coding Agent CLIs
installed. QoderCLI is the default label, but the interactive shell exposes all available Claude,
Codex, QoderCLI, and Pi login states:

```bash
bash examples/evolver-dev-shell/run.sh zsh qodercli
bash examples/evolver-dev-shell/run.sh bash codex
```

The Backend argument controls the Evolver metadata in the synthetic input; no Backend process is
started. The example does not require Agate credentials. `AGATE_GPU`, if set, is retained as the
hardware label; otherwise `nvidia-h100` is used.

Inside the shell, inspect `evolution-input.json`, `input/parent/`, `input/agents/`,
`input/evidence/`, `runtime-tools/`, and the writable `candidate/` and `scratch/` directories.

```bash
python runtime-tools/evolver_tools.py inspect-agents
python runtime-tools/evolver_tools.py inspect-kernels
```

Exiting destroys the inner Evolution workspace and the outer temporary state directory. The JSON
summary reports `workspace_destroyed: true`; no durable Registry objects are created.

The existing `evolver-dev-shell --lineage ... --epoch ...` CLI remains available for reconstructing
real Epoch history. This example deliberately uses `temporary-evolver-dev-shell` instead.
