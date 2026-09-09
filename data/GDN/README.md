# GDN source-tree task inputs

English | [中文](README.zh.md)

For the counterpart retaining the original optimization hints, see [GDN-full](../GDN-full/README.md).

GPU target: **L20D**. Operator: `chunk_gated_delta_rule`. This kit creates one **CuteDSL**
Lineage. Preparation does not start Runtime, Wiki, model sessions, or GPU jobs.
**Both Optimizer and Evolver use the Claude backend, reusing the Lima user's `.claude` configuration.**

`data/GDN/` contains task inputs and configuration templates only. Launch scripts live in
`scripts/gdn/`; their default workspace is `workspaces/GDN/`. Preparation snapshots the task,
Campaign definitions, and initial evidence into that workspace and reconstructs the seed there.
Generated configuration, credentials, databases, sessions, logs, and results never go into `data/`.
Both scripts accept `--workspace workspaces/GDN-clean` for a separate experiment. Use the same
workspace for preparation, serving, and execution. Once registered, an experiment retains its
snapshot; editing `data/GDN/` does not change its inputs. To use revised inputs, prepare a new
workspace. Do not rerun preparation over historical state to update its task definition.

`campaign.optimizer.max_session_tokens` is **100,000,000 (100M) per Session** for this
source-tree kit, including Bootstrap and every Active/Challenger Attempt in all ablation
arms. This is not a shared Epoch or Campaign budget. Evolver has no token quota; existing
timeouts still apply. Newly started Campaign/Bootstrap processes load this configuration;
already running processes do not hot-reload it, and failed Attempts are not automatically
retried when the quota changes. Single-file configurations remain unchanged.

## Ablation entrypoint

With this kit prepared and Runtime/Wiki already running, execute in Lima:

```bash
cd ~/atrex-runtime
source ~/.venvs/atrex-runtime/bin/activate
source env.sh
python scripts/gdn/run.py ablation
```

This starts tasks only, with the same Sandbox scheduling permissions as `campaign`.
`ablation-campaign.json` uses a new `gdn-source-tree-l20d-claude-ablation` creation key;
it neither changes nor takes over the existing trial. After one full Bootstrap, six controls
reuse the new experiment's exact v0, Agent, edit boundaries, contract and initial evidence.
No repeated baseline measurement or old trial experience is imported.

`ablation.json` uses the same plan builder as single-file production. Defaults:

| Campaign instance | Per-Epoch topology | Optimizer Attempts | Retain State | Evolutions |
|---|---|---:|---|---:|
| `evolve-3` | Active + Challenger, 1 trajectory × 3 Attempts each | 30 | yes | 4 |
| `ablation-isolated-01/02` | Two independent instances, 1 trajectory × 3 each | 15 each | no | 0 |
| `ablation-retained-01/02` | Two independent instances, 1 trajectory × 3 each | 15 each | yes | 0 |
| `ablation-pool-3` | One Active Branch, 2 trajectories × 3 | 30 | no | 0 |
| `ablation-pool-retained-3` | One Active Branch, 2 trajectories × 3 | 30 | yes | 0 |

Seven Campaigns, five Epochs each, 150 Optimizer Attempts excluding Bootstrap/Evolver.
The ablation main arm uses two independent copies of the same Agent in Epoch 1; evolution
starts in Epoch 2. The original `campaign` role retains its Active-only first Epoch.
No external original-AKA control is launched. Resetting State preserves Kernel progress and
Runtime journals. Pools restart from the best Kernel at Epoch boundaries; Pool-Retained also
inherits that trajectory's terminal State, without merging. Arms share no subsequent history
or writable files.

Outputs live under `workspaces/GDN/ablation/`: frozen inputs, Bootstrap, `campaign-results.json`,
and per-arm `campaign-result.json` / `campaign.log` (plus control seed definitions/results).
The summary exposes Campaign/Lineage IDs for inspect. Sessions/Artifacts remain in the shared
`workspaces/GDN/state/`. Attempt progress streams to each log; arm completion prints a timestamp.
Failures do not cancel siblings. Rerunning resumes the same identities and reports completed
results. `--target-epoch` affects only the main arm; controls retain 15 Attempts per trajectory.
Changed frozen inputs require a new workspace and creation key.

Up to ten Optimizer workers run concurrently. On memory-limited Lima, finish the old trial
before explicitly launching this suite. Adding these configs does not start/stop/restart tasks.

## Contents and provenance

- `task/`: imported adapter, Source Manifest, Torch Reference, input generator, public Shape
  Train, 10 private test shapes, Metadata, and Roofline. The public objective and evidence
  annotations have been revised; the Source Manifest pins the updated seed provenance.
- `source.bundle`: an offline Git bundle of the seed; no external repro directory
  or network checkout is needed.
- `campaign.json`: pinned L20D task, source and Optimizer revision, and Epoch scheduling.
- `runtime.template.json`: this task's independent configuration, not a cross-example import.
- `initial-evidence/`: seed provenance only, without historical optimization experience.

The workspace holds `task/` and `initial-evidence/` snapshots, `source/` (the reconstructed seed;
do not optimize here), generated `runtime.json` / `evaluation-contract.json`, Campaign definitions,
and `prepared.json` (content hashes, pinned commits, and local validation results).

Copied from these locations inside `GDN_AKA_REPRO_20260907`:

- `gdn_fi_initial_seed/task/` and `gdn_fi_initial_seed/source/`.
- `atrex-bench/data/aka/gdn_prefill_sm103_m64_20260904/chunk_gated_delta_rule/`.

Seed commit: `60c83174e82e4566e6ee360fd38b85c5bb0794b6`, derived from the original seed
`a39405536f178689d7f60b551c17b2252bcee61d` and FlashInfer commit
`2ab910c58fdd2392914ea05e2a8714946ac0eef6`. The seed update changes only
`UPSTREAM_PROVENANCE.json` to remove an implementation-name hint; Kernel source bytes are
unchanged. Original Metadata and Roofline already specify `NVIDIA L20D`; measurements for
another GPU were not relabeled. Existing Campaigns retain their frozen seed and task inputs.

## Prepare in Lima

Enter Lima from the host and use the **Linux venv**, not the shared macOS `.venv`:

```bash
limactl shell ubuntu
cd ~/atrex-runtime
source ~/.venvs/atrex-runtime/bin/activate
python scripts/gdn/prepare.py --backend claude
```

Preparation selects the current non-root Linux user as Sandbox Worker and reuses CLI
configuration from that user's Home. Use `--worker-user` for another existing non-root
account. bwrap + cgroup isolation remains enabled. Runtime and preparation use the Linux
venv; sandboxed Optimizer, Evolver, and Runtime Tools use the global Python interpreter
(resolving the venv symlink to `/usr/bin/python3.x`), not a venv hidden under Home.

The script validates source locks, editable scope, Production Policy, public/private shape
contracts, and the three pinned repository commits. Remote GPU images and model connectivity
are not checked; those require a subsequent live run. Generated files and `source/` are ignored
by Git and can be reconstructed from the bundle after copying or cloning this repository.
Once `state/` exists, preparation refuses to overwrite differing run configuration.

## Later execution

Runtime is configured at `http://127.0.0.1:8766`; the independent Wiki service is expected at
`http://127.0.0.1:8091`. Wiki is not started automatically. State, sessions, and artifacts live
under `workspaces/GDN/state/`. Agate uses `AGATE_AK` / `AGATE_SK`; set `AGATE_URL` during preparation
to override its endpoint. No credentials are copied into the kit. Runtime and Campaign
processes need matching `ATREX_CAPABILITY_SIGNING_KEY` and `ATREX_ADMIN_BEARER_TOKEN` values.
The selected model CLI must already be installed and authenticated.

Alternatively, run `python scripts/gdn/run.py serve` and
`python scripts/gdn/run.py campaign --target-epoch 5` in separate Linux processes with Agate
environment variables loaded and the Sandbox scheduling privileges described below. They
automatically share persistent `runtime-secrets.json` (mode 0600, Git-ignored), run Bootstrap
before the requested Epoch, and save `bootstrap-result.json` / `epoch-result.json`. A failed
Bootstrap prevents optimization from starting. Resume with the same configuration and secrets;
do not run a second scheduler for the same Campaign concurrently.

If managed with systemd, the GDN service units can be named `atrex-gdn-runtime` and
`atrex-gdn-campaign`, with logs redirected under `workspaces/GDN/services/`:

```bash
systemctl status atrex-gdn-runtime atrex-gdn-campaign --no-pager
sudo tail -n 60 workspaces/GDN/services/campaign.log
```

These are the subsequent run entry points, **not commands executed during preparation**.
Sandbox scheduling requires Linux privileges for system-level systemd services and switching
to the Worker user, as in the production scripts' root scheduler:

```bash
# Service process; Bootstrap and Campaign run in another terminal.
atrex-kernel-agent-runtime serve --config workspaces/GDN/runtime.json

# Run a complete Claude Bootstrap Session on the seed, then finalize and register v0.
atrex-kernel-agent-runtime bootstrap \
  --config workspaces/GDN/runtime.json --campaign workspaces/GDN/campaign.json

# Substitute the returned campaign_id. Run through Epoch 5.
atrex-kernel-agent-runtime run-campaign \
  --config workspaces/GDN/runtime.json \
  --campaign CAMPAIGN_ID_FROM_BOOTSTRAP --target-epoch 5
```

Defaults: five Epochs total, three serial Attempts per branch per Epoch, one Trajectory;
Active only in Epoch 1, with one Challenger starting in Epoch 2 (three Active Attempts and
three Challenger Attempts per Epoch). The Epoch target and per-Trajectory Attempt count match
the single-file production defaults; the existing Active-only first Epoch is unchanged.
The target is absolute: rerunning after Epoch 5 reports the completed result rather than adding
five more Epochs. No Evolver is launched solely for an unused Epoch 6.
The Attempt count is frozen when a Lineage is registered. Existing one-Attempt Lineages are not
changed by editing this config: use a new Campaign creation key for the new schedule, retaining
the old history. Do not overwrite Registry rows or treat an old Campaign as a three-Attempt run.
Kernel Retention and Agent Promotion use
same-allocation ABBA, with Production Gate and clock locking enabled. Legacy Manifest fields
such as `measurement` remain provenance; Runtime Gate and Campaign settings control execution.

The tree is materialized directly under the Agent's `work/kernel/`, with a fixed `kernel.py`
adapter. Only `flashinfer/gdn_kernels/blackwell/` may change; other files remain frozen.
Bootstrap uses Claude to evaluate the original tree, repair permitted sources if necessary,
and submit a standard report with Direction/Experiment Journals. Runtime then independently
finalizes v0. The Campaign key `gdn-source-tree-l20d-claude-bootstrap` starts this full Agent
flow separately from the former model-free trial; its historical v0 and Session records are
preserved. Subsequent runs with the new key reuse the new baseline normally.
Runtime Tools support Evaluate, Profile, Check, and Disassemble; see the
[source-tree interface](../../docs/source-trees.md).

The remote L20D environment must satisfy the task's SM103 requirement, `torch>=2.9.0`, and
`nvidia-cutlass-dsl>=4.4.2`. Profile / Disassemble require NCU; sanitized Check requires
Compute Sanitizer.
