# One-Epoch Lineage Example

English | [中文](README.zh.md)

Defaults connect to the remote Agate service; explicit environment overrides remain supported.

This example runs a complete single-DSL Lineage through exactly one Epoch. The Campaign grants the
Agent-owned Workflow no Challenger slots and an exact budget of three Optimizer Attempts. The
selected `workflow/isolated.py` program spends that budget on one Trajectory with three serial
rounds, explicitly routes each retained Kernel to the next round, and resets adaptive State before
every Attempt.

Runtime therefore starts exactly three fresh Optimizer sessions, but does not decide their topology
or routing. Because the Workflow never requests a Challenger, no Evolver session or Evolver Git
Bundle is needed. The Optimizer uses the Runtime default QoderCLI credential.

`run-campaign` prints each durably completed Attempt immediately to stderr, for example:

```text
[2026-08-18T12:01:02+00:00] active trajectory 1 attempt 1 finished
[2026-08-18T12:03:04+00:00] challenger-1 trajectory 3 attempt 2 finished
```

`active` and `challenger-N` identify competing Agent Branches; `trajectory-N` identifies one
independent optimization chain inside that Branch. Concurrent trajectories appear in actual
completion order. Progress stays out of stdout so the saved Epoch result remains one valid JSON
document.

On an interactive terminal, the timestamped events remain above an in-place chart such as:

```text
Epoch 1 branch progress (lineage_...)
  active
    trajectory 1   [██░] 2/3
    trajectory 2   [█░░] 1/3
  challenger-1
    trajectory 1   [░░░] 0/3
    trajectory 2   [░░░] 0/3
```

When stderr is redirected, Runtime automatically emits only plain timestamped lines and never
writes terminal control sequences.

This directory owns its `runtime.json` and one-Epoch `campaign.json`. They use only the canonical
VecAdd inputs in `examples/shared/vecadd/`; no Bootstrap example scripts or configuration are reused.

Export the Agate settings, then run:

```bash
export AGATE_URL="https://atrex-gateway.alibaba-inc.com"
export AGATE_AK="..."
export AGATE_SK="..."
export AGATE_GPU="L20N"
export QODER_PERSONAL_ACCESS_TOKEN="..."
bash examples/lineage/run.sh
```

The wrapper prepares an isolated state directory and, when `ATREX_WIKI_URL` is unset, starts and
waits for the Local Wiki at `http://127.0.0.1:8091`. It then starts Runtime, bootstraps the Triton
VecAdd Lineage if necessary, runs or resumes Epoch 1, prints Attempt/Kernel/Agent histories, and
stops the Runtime and Local Wiki processes it owns. Set `ATREX_WIKI_URL` explicitly to use an
already-running local or remote Wiki; the wrapper never stops a Wiki it did not start. Results are
retained under `workspaces/lineage-example/`. Re-running an interrupted
Epoch 1 resumes it; re-running after completion reports the existing result without creating Epoch
2. The Bootstrap identity, Optimizer commit, and Evolver commit stay pinned. Delete or move the
workspace only when intentionally starting a fresh Campaign.

For step-by-step debugging, prepare the generated files and keep Runtime in the first terminal:

```bash
bash examples/lineage/prepare.sh
bash examples/lineage/start-runtime.sh
```

In a second terminal with the same exported Agate variables, Bootstrap as needed and run or resume
Epoch 1:

```bash
bash examples/lineage/run-epoch.sh
```

The inspection command is offline: after the run completes it reads the durable Registry directly,
so Runtime does not need to remain running.

```bash
bash examples/lineage/inspect.sh
```

`inspect.sh` prints the saved scheduling result, the Epoch winner decision, every scheduled Attempt (including `pivot`,
`blocked`, and other no-Candidate outcomes), the versioned terminal-Kernel table, and the Lineage
Agent revision table. `X` counts rows in Attempt history; it does not promise `X` new Kernel
versions.

Search organization is executable Agent code. To test another organization, select another
Workflow implementation and give it matching `max_challengers` and `optimizer_attempt_budget`
resource limits in a new Campaign workspace.
