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

- Main arm `evolve-3`: default absolute target Epoch 5 (its existing workspace stays at `dsls/DSL/`).
- Epoch 1: Active and a replica run the same Agent revision, v0 Kernel and starting State in two
  isolated Branches, each with three serial fresh Optimizer Attempts. No Evolver runs.
- Epoch 2 onward: one Evolver creates one Challenger; Active and Challenger run concurrently, with
  three serial Attempts on each Branch.
- Epoch completion independently compares Kernels and selects the next Active Agent.

With the default `event_only=true`, each DSL runs seven Campaign instances including the main arm.
All share the same frozen Bootstrap v0; controls do not repeat Bootstrap or baseline measurement.
All default schedules run 5 Epochs with 3 serial Attempts per Trajectory per Epoch:

| Arm | Parallel structure | Total Optimizer Attempts | Retain Runtime State | Evolutions |
|---|---|---:|---|---:|
| `evolve-3` (main) | Active + Challenger, one Trajectory each | 30 | yes | 4 |
| `ablation-isolated-01/02` | Two independent instances, one Trajectory each | 15 each, 30 combined | no | 0 |
| `ablation-retained-01/02` | Two independent instances, one Trajectory each | 15 each, 30 combined | yes | 0 |
| `ablation-pool-3` | Two Trajectories in one Branch | 30 | no | 0 |
| `ablation-pool-retained-3` | Two Trajectories in one Branch | 30 | yes | 0 |

Runtime State includes Memory/Knowledge/Skills/Tools. Resetting State restores the pinned Core's initial
contents; it does not erase Kernel progress or Runtime history. Isolated and Retained instances
share only the Bootstrap baseline, not subsequent history or mutable State. Their replica counts
follow the configured Active/Challenger Trajectory count.

Pool Trajectories run independently within each Epoch and share completed history at the next Epoch,
restarting from the selected best Kernel. Pool-Retained also inherits its producing Trajectory's
terminal State; State is selected, not merged or synchronized live. Source stays fixed in all controls.
Both Pool arms always use two Trajectories and three Attempts per Epoch.

Only the main arm runs Evolver. With `first_epoch_same_agent=true`, its first Epoch uses a same-Agent
replica without creating a new Agent revision or Evolution Report. Branch States are independent;
the next Active and Evolver inherit the best-Kernel Trajectory's terminal State. Bootstrap and Evolver
Sessions are excluded from Optimizer Attempt counts.

The main arm lives at `dsls/DSL/`; control files are under `dsls/DSL/ablation-*/`.
The generated `ablation.json` freezes the control schedules with 15 post-Bootstrap Attempts per
Trajectory. `--target-epoch` changes only the main arm's target. Task-level `campaign-results.json`
summarizes all results and budgets. Existing Workspaces retain their frozen plans; preparation
rejects a changed Arm set. Use a new Workspace for this topology; existing experiment data is not removed.

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
For current Atrex-Bench layouts, preparation exposes `shape_train.json` to the Agent and seals
`shape_valid.json` as the exact Evaluation Contract. Legacy `agent_problem.json` and `shapes.json`
remain fallback-only. `metadata.json` is forwarded privately, including `mutates_inputs` and
`scratch_inputs`, so the remote correctness gate enforces declared input side effects.

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
