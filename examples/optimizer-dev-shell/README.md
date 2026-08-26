# Temporary Optimizer dev shell

English | [中文](README.zh.md)

This example opens one disposable Optimizer-compatible workspace. It does not run Bootstrap,
create a durable Campaign or Lineage, or start an Agent backend.

Runtime injects a pinned Core bundle, a known-correct Triton VecAdd Kernel, empty first-Attempt
Evidence, writable candidate and scratch directories, and a temporary scoped Gateway capability.
The shell can use the normal Runtime tools while it is open.

The shell always uses the production `sandbox` launcher: bubblewrap filesystem isolation, a
host networking, and systemd/cgroup-v2 resource controls. The wrapper requires Linux,
the configured system tools, and passwordless `sudo`; it re-enters the trusted Runtime/launcher as
root and runs the interactive shell as the invoking non-root user. The invoking user's Backend
Home and PATH are preserved explicitly so user-local CLIs such as `qodercli` and their read-only
login state remain available inside the sandbox.

## Run

Export the remote Agate settings first:

```bash
export AGATE_URL="https://your-agate.example"
export AGATE_AK="..."
export AGATE_SK="..."
export AGATE_GPU="H100"
bash examples/optimizer-dev-shell/run.sh zsh qodercli
```

The shell and Backend arguments are independently optional and may appear in either order. The
Backend argument selects the Runtime binding shown in the Session context; `qodercli` is the
default. Unlike production Agent Sessions, the interactive dev shell projects all available
`claude`, `codex`, `qodercli`, and `pi` login states, so every installed CLI can be invoked from the
same shell. Selecting a Backend explicitly remains useful when debugging its exact Runtime binding:

```bash
bash examples/optimizer-dev-shell/run.sh bash claude
bash examples/optimizer-dev-shell/run.sh bash codex
```

Every invocation creates a unique directory with `mktemp`,
starts its own Runtime, and opens the shell without invoking Qoder, Claude, Codex, or another Agent
backend. Exiting the shell revokes the capability, stops that Runtime, and deletes the complete
temporary directory, including its SQLite databases and Artifact Store.

For example, the default shell can run both providers directly:

```bash
claude -p "Reply with exactly: hello"
codex exec --ephemeral --skip-git-repo-check "Reply with exactly: hello"
```

Inside the shell, inspect `attempt.json`, `input/`, `agent/`, `work/kernel/`, `sessions/`, and
`scratch/`. For example, Runtime tools remain available through:

```bash
python3 agent/optimizer/src/runtime_tools.py gateway-execute --request scratch/request.json
```
