# Production runner

English | [中文](README.zh.md)

These scripts create three independent, commit-pinned single-DSL Campaigns from one
`third_party/atrex-bench/data` operator. CUDA, Triton, and CuteDSL run as independent
`Bootstrap -> Campaign` pipelines: each DSL advances into Epoch execution immediately after its own
Bootstrap succeeds. One DSL failure does not cancel or gate its peers. Production content policy is
mandatory; startup rejects any workspace whose `gate_policy.production_gate` is not `true`.

## Choose an entrypoint

- Persistent service with multiple tasks: start `services.sh` once, then use `campaign.sh start`
  for each operator/backend. This is the recommended production topology.
- Self-contained single-task run: use `run.sh`.
- Configuration only: use `prepare.sh`.
- Read-only result inspection: use `inspect.sh --workspace TASK [--dsl DSL]`.

Do not wrap the complete command in `sudo bash ...`. In `sandbox` mode the scripts escalate only
transient-service and sandbox execution. In `container` mode they never escalate.

## Fixed schedule

- Default absolute target: Epoch 10.
- Epoch 1: Active only, two serial fresh Optimizer Attempts.
- Epoch 2 onward: one Evolver creates one Challenger; Active and Challenger run concurrently, with
  two serial Attempts on each Branch.
- Epoch completion independently compares Kernels and selects the next Active Agent.

`--target-epoch N` overrides the absolute target. Repeating a command resumes durable state.

## Prerequisites

Choose one Worker boundary:

- `sandbox` requires Linux, bwrap, systemd with cgroup v2, root/sudo authority, and the configured
  non-root Worker user.
- `container` requires Linux and bwrap, but no systemd, writable cgroup hierarchy, sudo, or
  per-Session cgroup. The OCI policy must allow bwrap's namespace operations. Runtime isolates each
  Worker filesystem/namespace; the outer container supplies their shared memory, CPU, and PID total
  limits. Do not mount the Docker socket, Runtime secrets, private evaluator data, or unrelated paths.

Both modes require one supported Agent CLI and Agate credentials.
In `container` mode, run the outer container as a non-root user when practical and verify that bwrap
can create its user/PID/IPC/UTS namespaces; managed Runtime and Local Wiki processes retain that
container identity.

Preparing a new Campaign also requires clean Core and Evolver Git worktrees. Production records
only commits, so staged, unstaged, or untracked Bundle files cause preparation to fail instead of
silently running an older `HEAD`. Commit and push Bundle changes before creating a new workspace.
An existing workspace remains pinned to the commits in its `production-manifest.json`; use a new
workspace when intentionally adopting new Agent commits.

Copy the environment template outside the repository:

```bash
cp scripts/production/environment.example /secure/atrex-production.env
chmod 0600 /secure/atrex-production.env
```

When no `ATREX_WIKI_URL` is configured, `services.sh` manages the repository's local Wiki test
service. A configured remote Wiki is health-checked but never managed by these scripts.

### Lima and virtiofs

Lima repository mounts commonly use `virtiofs`, where `chown` may report success without changing
the visible UID/GID. Runtime therefore asks systemd to create Worker roots and probes directly as
`worker_user`; it does not rely on root-create-then-chown. Empty foreign-owned roots left by a
failed preparation can be recreated safely. If root and the Worker see different numeric owners,
Runtime verifies the path again through systemd as `worker_user`; an exact Worker UID/GID match
preserves and reuses the non-empty root. Real mismatches fail closed and are never deleted.

Do not pre-create `TASK/state/*-workspaces` with sudo. Let Runtime create them or create them as the
normal login user. Runtime's rollback-journal SQLite layout supports Lima virtiofs only when POSIX
locking is reliable; arbitrary remote filesystems are unsupported.

## Persistent service and managed tasks

Start Runtime and Wiki once:

```bash
bash scripts/production/services.sh start \
  --workspace workspaces/production/control-l20n \
  --hardware-target L20N \
  --launcher-mode container \
  --env-file /secure/atrex-production.env
```

Then start one background task:

```bash
bash scripts/production/campaign.sh start \
  --service-workspace workspaces/production/control-l20n \
  --kernel production_qwen35_35b_inhouse_4k256/flash_attention \
  --backend qodercli \
  --target-epoch 10 \
  --env-file /secure/atrex-production.env
```

The service workspace pins its launcher mode. Campaigns attached to it inherit that mode, so the
Campaign command does not need to repeat `--launcher-mode`.

Manage it independently of the shared services:

```bash
bash scripts/production/campaign.sh status  --workspace TASK
bash scripts/production/campaign.sh stop    --workspace TASK
bash scripts/production/campaign.sh restart --workspace TASK
```

`restart` reuses the exact persisted arguments. To change the target Epoch, stop and issue a new
complete `start` command. A task owns its inputs, per-DSL definitions/results/logs, Worker
workspaces, and `campaign-run/` lifecycle state. Registry, Gateway/Agate Job databases, Artifact
Store, signing secrets, and Wiki belong to the service workspace.

## Self-contained task

For a single task that manages its own Runtime/Wiki:

```bash
bash scripts/production/run.sh \
  --kernel production_qwen35_35b_inhouse_4k256/causal_conv1d \
  --backend qodercli \
  --env-file /secure/atrex-production.env
```

`--kernel` accepts an operator directory, a `suite/operator` path relative to Atrex-Bench `data/`,
or a unique operator name. Bootstrap uses `reference.py` by default; pass
`--seed-source solution.py` only when intentionally starting from that implementation.

## Per-DSL inspection

Inspect all DSLs or one of `cuda`, `triton`, and `cutedsl`:

```bash
bash scripts/production/inspect.sh --workspace TASK
bash scripts/production/inspect.sh --workspace TASK --dsl triton
```

The command prints Epoch, Kernel, and Agent histories and can run while the Campaign is active. A
missing `bootstrap-result.json` means that DSL has not completed Bootstrap successfully.

Follow progress directly:

```bash
tail -f TASK/dsls/triton/bootstrap.log
tail -f TASK/dsls/triton/campaign.log
```

For Attempt and Session detail, read the DSL Campaign ID and use the Runtime CLI:

```bash
TASK=/data/atrex/tasks/flash-attention-qoder
DSL=triton
CAMPAIGN_ID="$(python3 -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["campaign_id"])' \
  "$TASK/dsls/$DSL/bootstrap-result.json")"

atrex-kernel-agent-runtime list-attempts \
  --config "$TASK/runtime.json" --campaign "$CAMPAIGN_ID" --format table
atrex-kernel-agent-runtime list-worker-sessions \
  --config "$TASK/runtime.json" --campaign "$CAMPAIGN_ID" --format table
```

Use `show-attempt`, `list-evaluations`, and `show-worker-session` to drill into IDs from those
tables. These inspection commands are read-only and do not require Runtime quiescence.

## Workspace layout

```text
runtime.json
production-manifest.json
runtime.env
dsls/<dsl>/
  campaign.json
  evaluation-contract.json
  production-manifest.json
  inputs/
  bootstrap-result.json
  campaign-result.json
  bootstrap.log
  campaign.log
campaign-run/
state/
```

In shared-control-plane mode, trusted databases and Artifact paths in task `runtime.json` point to
the service workspace. The task-local `state/` contains only Attempt, Evolution, Generalization,
and Bootstrap Worker workspaces. Always pass the task workspace—not the service workspace—to
`inspect.sh`.
