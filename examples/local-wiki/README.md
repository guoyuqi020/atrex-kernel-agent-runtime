# Local Wiki and Runtime Tools example

English | [中文](README.zh.md)

This example starts the wire-compatible Local GPU Wiki and demonstrates the exact `wiki-query`
workflow available to a Core Agent. The Local Wiki is a development test double. Production uses
the same external API through a remote `gpu_wiki.base_url`.

When running from a Lima-mounted checkout, create a Linux-local environment instead of reusing the
repository's macOS `.venv`:

```bash
python3 -m venv ~/.venvs/atrex-runtime
source ~/.venvs/atrex-runtime/bin/activate
python -m pip install -U pip
python -m pip install -e '.[dev]' -e './workspaces/local-wiki[dev]'
```

The wrapper uses `python3` from `PATH`. Set `ATREX_PYTHON` to another Linux interpreter when needed.
The Local Wiki query subprocess inherits that same interpreter by default; no absolute `.venv` path
is stored in the example configuration.

## 1. Pinned test corpus

The exact upstream `gpu-wiki` tree is vendored under `workspaces/local-wiki/corpus/gpu-wiki` and
ships with the Runtime repository. `workspaces/local-wiki/reference.lock.json` records its source
repository and commit. No preparation, sparse checkout, or network access is required at startup.

## 2. Start Local Wiki

From the Runtime repository root:

```bash
bash examples/local-wiki/start-local-wiki.sh
```

The process listens on `http://127.0.0.1:8091`. Verify it from another terminal:

```bash
curl -fsS http://127.0.0.1:8091/healthz
```

Startup keeps the vendored corpus unchanged and synchronizes it into the ignored writable Store at
`workspaces/local-wiki/state/gpu-wiki`. Query uses that Store.

Open [http://127.0.0.1:8091/](http://127.0.0.1:8091/) for the optional browser client. Each Query
result displays complete safe served Records and their stable IDs.

This example's checked-in [`runtime.json`](runtime.json) contains:

```json
{
  "gpu_wiki": {
    "base_url": "http://127.0.0.1:8091"
  }
}
```

The real configuration contains the remaining timeout and byte-limit fields;
do not replace the complete object with this abbreviated fragment.

## 3. Fast path: open a disposable Wiki Agent shell

For Wiki Tool development, a complete Campaign Bootstrap and Agate Gateway are unnecessary. Keep
only Local Wiki running, then open a temporary shell:

```bash
bash examples/local-wiki/start-temporary-agent-shell.sh
```

The wrapper creates a temporary Campaign/Lineage/Epoch/Attempt identity and starts an ephemeral
Runtime Wiki Proxy on a random loopback port. It injects the scoped Capability and the strict Core
Attempt environment, but does not start an Agent backend. Inside the shell, use the unmodified Core
Runtime Tool:

```bash
python agent/optimizer/src/runtime_tools.py \
  wiki-query --request scratch/wiki-query.json
```

This path still uses the Runtime Wiki Proxy for trusted context and response freezing. It merely
avoids the persistent Runtime Service, Kernel baseline, and Agate. Exiting the shell stops the
temporary Proxy and deletes its Registry, Artifacts, Capability, and Workspace.

Use `--dsl cuda`, `--dsl triton`, or `--dsl cutedsl` to change the temporary knowledge scope, and
`--shell bash` when preferred.

## 4. Full path: start this example's Runtime and Bootstrap

On first use, the local example creates `workspaces/local-wiki/state/demo.env` with mode `0600`.
It contains only the local capability signing key and admin token. The Runtime, Bootstrap, and Agent
Shell wrappers automatically load the same file. You may prepare it explicitly, but never need to
source it by hand:

```bash
bash examples/local-wiki/prepare-demo-env.sh
```

Agent-provider credentials are never written there. The example Runtime configuration binds the
Optimizer to QoderCLI and inherits `QODER_PERSONAL_ACCESS_TOKEN` from the launching environment.
Change `campaign.optimizer.agent_backend` in this example's `runtime.json` to select another
supported Backend. Credential values are never printed or copied into the workspace.

Keep the external Agate-compatible Gateway listening on `127.0.0.1:9000`, then run:

```bash
# Terminal 2
bash examples/local-wiki/start-runtime.sh

# Terminal 3
bash examples/local-wiki/bootstrap-campaign.sh
```

The Bootstrap output contains the `lineage_id` used by the debug shell.
This flow uses only this directory's `runtime.json` and `campaign.json`; the Campaign points to the
canonical VecAdd fixtures under `examples/shared/vecadd/`.

## 5. Use Runtime Tools inside a persistent Agent session

Runtime Tools are not direct Local Wiki clients. They call the Attempt-scoped Runtime Proxy at
`/v1/wiki/query`. Therefore these commands must run inside a Core baseline or
Optimizer session launched by Runtime. Runtime injects the trusted manifest, proxy URL, and scoped
capability. Do not manually mint or copy capability tokens into a host shell.

Inside that Agent workspace, the current directory is the workspace root and Runtime has made
`agent/optimizer/src/runtime_tools.py` available. Copy or reproduce the example query under the
workspace-owned `scratch/` directory:

```bash
cp /path/to/atrex-runtime/examples/local-wiki/wiki-query.json scratch/wiki-query.json
python agent/optimizer/src/runtime_tools.py \
  wiki-query --request scratch/wiki-query.json
```

Agent-visible output is the upstream GPU Wiki `records/notes` envelope:

```json
{
  "records": {
    "nvidia.hopper.any.kernel-opt.mbarrier-software-pipeline.pipeline-in-gluon": {
      "store": "gpu_wiki",
      "source": "kernel_wiki",
      "type": "technique-card",
      "applies_to": {},
      "match": {"arch": "exact"},
      "payload": {}
    }
  },
  "notes": []
}
```

Each mapping value is already the complete safe served Record. Preserve the exact mapping key in
`research_sources` when a Record materially informs the work; no second read operation exists.
The Agent never receives protocol versions, snapshot identities, interaction
Artifact Digests, content Digests, external credentials, or trusted control context. Runtime still
freezes all of those internally for idempotent replay and audit.

## 6. Open a real managed Agent debug session

The debug session creates the real Optimizer workspace, Attempt manifest, Evidence view, working
Kernel, and Core Bundle, and injects Attempt-scoped Gateway/Wiki capabilities. It deliberately does
**not** start the Agent backend; the final process is an interactive `zsh` or `bash`.

Keep Local Wiki, the Agate-compatible Gateway, and Runtime service running, and complete the
previous Bootstrap step. Use the returned `lineage_id` in another terminal:

```bash
bash examples/local-wiki/start-agent-shell.sh
```

The no-argument form reads the first `lineage_id` from
`workspaces/local-wiki/state/last-bootstrap.json`, which the Bootstrap wrapper writes atomically.
An explicit Lineage and shell remain supported:

```bash
bash examples/local-wiki/start-agent-shell.sh \
  lineage_0123456789abcdef0123456789abcdef zsh
```

The script always uses this directory's `runtime.json`. The equivalent direct command is:

```bash
atrex-kernel-agent-runtime dev-shell \
  --config examples/local-wiki/runtime.json \
  --lineage lineage_0123456789abcdef0123456789abcdef \
  --shell zsh
```

`--lineage` creates or reuses the first Active Attempt in the current Epoch. Alternatively,
`--attempt attempt_...` creates a fresh `run-<uuid>` workspace for an existing running Attempt. The
shell holds the lineage fencing lease. Exiting retains both the workspace and the running Attempt;
it does not fabricate a session trace, token report, or terminal result.

Inside the shell, inspect `$ATREX_ATTEMPT_MANIFEST`, `input/`, `work/kernel/`, `agent/optimizer/`,
and `scratch/`. Use the active platform-local Python to invoke a tool:

```bash
python3 \
  agent/optimizer/src/runtime_tools.py \
  wiki-query --request scratch/wiki-query.json
```

The capability is not printed, but it is present in the process environment; do not copy `env`
output into logs or chats. No token quota is charged because no Agent starts. Explicit `submit`,
`evaluate`, or Wiki tool calls are still real controlled operations: they consume capability call
budget and create audit records. Do not run a Campaign scheduler for the same lineage concurrently.
