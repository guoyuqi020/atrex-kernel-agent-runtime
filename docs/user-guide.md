# User Guide

English | [中文](user-guide.zh.md)

## 1. Prerequisites

Development mode requires Python 3.12+, Git, a supported Agent CLI (`claude`, `codex`,
`qodercli`, or `pi`), and access to an Agate service/GPU environment. Production sandbox mode also
requires Linux, bubblewrap, systemd with cgroup v2, and a dedicated non-root
Worker user.

Clone all pinned repositories before installation:

```bash
git submodule update --init --recursive
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

A release wheel can be installed instead of editable mode. Core, Evolver, and Atrex Bench are not
inside the wheel; Runtime imports their configured Git commits at execution time.

## 2. Choose a first workflow

The shortest end-to-end check is:

```bash
export AGATE_URL='https://your-agate.example.com'
export AGATE_AK='...'
export AGATE_SK='...'
export AGATE_GPU='H20'
export QODER_PERSONAL_ACCESS_TOKEN='...'
bash examples/bootstrap/run.sh
```

The script creates an isolated workspace, generates local secrets and config, starts Runtime,
bootstraps one Triton VecAdd Lineage, inspects the result, and stops Runtime. Other examples are
listed in [examples/README.md](../examples/README.md).

## 3. Configure a deployment

Copy [`runtime.example.json`](../runtime.example.json) to a private deployment file and edit it.
Runtime config is strict schema v1. Important choices are:

- storage databases and Artifact/workspace roots;
- Agate URL, GPU target, credentials, timeouts, and health interval;
- Core base repository and Evolver repository/full commit;
- Optimizer/Evolver Backend, executable command, environment allowlist, and Session policy;
- Gate policy, comparison method, Roofline builder, and Production Gate;
- `development` or Linux `sandbox` launcher;
- optional GPU Wiki query service;
- administration and maintenance limits.

Create a Campaign schema-v3 file from an example. It selects operator, hardware, Evaluation
Contract, exact Core commit, Epoch topology, per-DSL seed Kernel/Evidence, and optional per-Lineage
models. There is no separate Bootstrap JSON.

Runtime config stores environment-variable names, not secret values. Export the values in the
Runtime process environment:

```bash
export ATREX_CAPABILITY_SIGNING_KEY="$(openssl rand -base64 32)"
export ATREX_ADMIN_BEARER_TOKEN="$(openssl rand -hex 32)"
export AGATE_AK='...'
export AGATE_SK='...'
# Export only the provider credential required by the selected Backend.
```

The checked-in examples generate signing/admin values automatically. Production values must be
stable across process restarts and stored in a secret manager.

## 4. Start and verify Runtime

```bash
atrex-kernel-agent-runtime serve --config /absolute/path/runtime.json
curl -fsS http://127.0.0.1:8765/healthz
curl -fsS http://127.0.0.1:8765/readyz
```

`healthz` checks the process. `readyz` probes local Registry, Gateway control, Agate job database,
and Artifact staging. Runtime logs external Agate health separately and does not make temporary
Agate unavailability fail local readiness.

## 5. Bootstrap a Campaign

Keep the service running because Core Sessions call Runtime Tools over its HTTP endpoint:

```bash
atrex-kernel-agent-runtime bootstrap \
  --config /absolute/path/runtime.json \
  --campaign /absolute/path/campaign.json
```

Bootstrap is idempotent by `creation_key`. It freezes the Campaign contract and commits, optionally
generates a public Agent Problem, runs one Core baseline Session per configured DSL, preserves every
Bootstrap execution generation/evaluation, then publishes `agent-v0` and authoritative Kernel `v0`.
Retrying the same Campaign resumes completed work; changing immutable inputs is rejected.

## 6. Run Epochs

Run one Campaign or explicitly selected Lineages through an absolute Epoch target:

```bash
atrex-kernel-agent-runtime run-campaign \
  --config /absolute/path/runtime.json \
  --campaign campaign_0123456789abcdef0123456789abcdef \
  --target-epoch 3
```

`--target-epoch` is absolute, so repeating the same command is safe. Add `--finalize` only when no
further Epoch should be scheduled. Runtime builds the configured Challenger pool serially, then
runs Branches within `max_parallel_branches`; Trajectories within a Branch may run concurrently,
while Attempts in one Trajectory remain serial.

For queue-based operation, submit `POST /v1/admin/tasks` and run one or more:

```bash
atrex-kernel-agent-runtime run-task-worker --config runtime.json --watch
```

## 7. Inspect results

```bash
atrex-kernel-agent-runtime list-epochs --config runtime.json --campaign "$CAMPAIGN" --format table
atrex-kernel-agent-runtime list-attempts --config runtime.json --campaign "$CAMPAIGN" --format table
atrex-kernel-agent-runtime list-kernels --config runtime.json --campaign "$CAMPAIGN" --format table
atrex-kernel-agent-runtime list-agent-revisions --config runtime.json --campaign "$CAMPAIGN" --format table
atrex-kernel-agent-runtime list-worker-sessions --config runtime.json --campaign "$CAMPAIGN" --format table
```

Use `show-attempt`, `show-kernel`, `show-agent-revision`, and `show-worker-session` for exact
records. `list-evaluations`/`show-evaluation --source --result` expose every exploratory and
Runtime-authoritative evaluated Kernel/result pair. `list-bootstrap-runs` and `show-bootstrap-run`
expose all physical Bootstrap generations, including failed generations.

## 8. Run production tasks

The production scripts separate the persistent control plane from operator tasks. Start Runtime
and Wiki once:

```bash
bash scripts/production/services.sh start \
  --workspace workspaces/production/control-l20n \
  --hardware-target L20N \
  --env-file env.sh
```

Then start an independent background task for each operator/backend. By default, each task creates
separate CUDA, Triton, and CuteDSL Campaigns, bootstraps them concurrently, and runs them through an
absolute Epoch target:

```bash
bash scripts/production/campaign.sh start \
  --service-workspace workspaces/production/control-l20n \
  --kernel production_qwen35_35b_inhouse_4k256/flash_attention \
  --backend qodercli \
  --target-epoch 10 \
  --env-file env.sh
```

Inspect one DSL's Epoch, Kernel, and Agent history:

```bash
bash scripts/production/inspect.sh \
  --workspace workspaces/production/production-qwen35-35b-inhouse-4k256--flash-attention--l20n--qodercli \
  --dsl triton
```

Do not wrap the whole start command in `sudo`. The scripts escalate only transient-service and
sandbox execution, preserving the caller's Home, provider login state, and Worker workspace owner.
See the [Production runner](../scripts/production/README.md) for layouts, logs, status, and recovery.

## 9. Seed a new Lineage

Copy [`lineage-seed.example.json`](../lineage-seed.example.json). Choose either sealed Agent/Kernel
Artifact digests or existing Revision IDs, then run:

```bash
atrex-kernel-agent-runtime seed-lineage \
  --config runtime.json \
  --campaign "$CAMPAIGN" \
  --spec lineage-seed.json
```

Runtime revalidates the Agent repository, independently evaluates the Kernel against the target
Campaign contract, and creates a new independent `agent-v0`/`v0` Lineage. This does not alter the
Campaign's frozen Core or Evolver commits.

## 10. Debug Agent workspaces

These commands create/reconstruct the real workspace and authority but do not start an Agent:

```bash
atrex-kernel-agent-runtime dev-shell --config runtime.json --lineage "$LINEAGE"
atrex-kernel-agent-runtime evolver-dev-shell --config runtime.json --lineage "$LINEAGE" --epoch 2
```

Use them only for trusted debugging. Optimizer Runtime Tools and Evolver inspection tools are
documented in [Interface Reference](interfaces.md).

## 11. Recovery and maintenance

A failed Epoch is not automatically rewritten. After inspection, authorize an idempotent retry:

```bash
atrex-kernel-agent-runtime recover-epoch --config runtime.json --epoch "$EPOCH" \
  --recovery-key incident-2026-08-20 --reason 'Agate allocation interruption'
```

Artifact and workspace GC are dry-run by default. Stop all Runtime/Worker processes before
using `--apply --confirm-runtime-stopped`. Backup/restore, event export/pruning, cancellation,
credential rotation, and Linux sandbox setup are covered in [Deployment and Operations](operations.md).
