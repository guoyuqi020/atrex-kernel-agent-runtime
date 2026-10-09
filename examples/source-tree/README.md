# Source-tree optimization

English | [中文](README.zh.md)

This preparation example uses its own task-derived Campaign and Evaluation Contract. It does
not import another example's config or start services. Use an already configured Runtime service
and its matching `ATREX_RUNTIME_CONFIG`. That deployment must provide `gate_policy.evaluator`,
an Optimizer/Evolver backend, and an Agate environment with the manifest's runtime requirements.

For source-tree tasks, set `campaign.optimizer.max_session_tokens` to `100000000` in that
Runtime configuration: 100M per Bootstrap or Optimizer Session, independently for every
arm/Attempt, not shared across an Epoch. The GDN kit already sets this value. This preparation
script does not modify your supplied Runtime configuration; use a separate configuration from
single-file tasks to keep their limits unchanged. Evolver remains without a token quota.
Restarting the Campaign process is required to load a changed limit; running Sessions do not
hot-reload it.

For the supplied GDN repro, import **only its initial seed**:

```bash
export GDN_REPRO="$HOME/GDN_AKA_REPRO_20260907"
python3 examples/source-tree/prepare.py \
  --source-manifest "$GDN_REPRO/gdn_fi_initial_seed/task/source_manifest.json" \
  --source-repository "$GDN_REPRO/gdn_fi_initial_seed/source" \
  --task "$GDN_REPRO/atrex-bench/data/aka/gdn_prefill_sm103_m64_20260904/chunk_gated_delta_rule" \
  --optimizer-commit "$(git -C src/kernel-design-agents rev-parse HEAD)" \
  --hardware-target "$AGATE_GPU" \
  --output workspaces/gdn-source-tree

# Uses the existing services; Bootstrap once, then eight three-Trajectory Campaigns:
python scripts/source-tree/run.py \
  --config "$ATREX_RUNTIME_CONFIG" \
  --campaign workspaces/gdn-source-tree/campaign.json \
  --plan workspaces/gdn-source-tree/ablation.json \
  --workspace workspaces/gdn-source-tree/run --target-epoch 5
```

Select hardware capable of the original SM103 task; this example does not reinterpret it as an
L20N task. The source commit comes from the existing manifest; the Optimizer commit must belong
to the deployment's configured base repository. Output must be a new directory. Resume with the
generated Campaign unchanged, rather than rerunning preparation over an existing workspace.

Default: CuteDSL only, using the same shared plan builder as single-file production.
The current plan enables two communication modes for each of four Direction/Experiment tool
settings: both enabled, both disabled, Experiment only, and Direction only. Each setting/mode owns
one Campaign/Lineage with three retained Trajectories. All eight arms share one frozen Bootstrap v0.
Default: 5 Epochs, three serial Attempts per Trajectory per Epoch; 15 Attempts per Trajectory,
45 per arm, and 360 in total, excluding Bootstrap. No Evolver runs.

| Mode | Workflow | Within the Epoch | After the Epoch |
|---|---|---|---|
| `epoch-shared` | `epoch_shared_3.py` | Each Trajectory reads only its own current history and carries its own Kernel | All three share completed history and the selected best Kernel |
| `broadcast` | `broadcast_3.py` | Recorded measurements, Artifacts and enabled Journals are immediately queryable across Trajectories; each round routes the best accepted Kernel so far | All three share completed history and the selected best Kernel |

Each mode has arm labels `ablation-MODE-3`, `ablation-MODE-no-modules-3`,
`ablation-MODE-experiments-3`, and `ablation-MODE-directions-3`. Broadcast also exposes sealed
reports and conversations from completed earlier-round peers in the next Attempt's filesystem
snapshot. Working source, scratch and live conversations remain private; retained Tool State is
routed independently per Trajectory. History is accessible when relevant, not a requirement to
read every prior conversation. Different arms do not share post-Bootstrap history.

Pool-Retained, single-Trajectory Retained, Retained-Evolve and other Workflow templates remain
available but are disabled in new plans. Each enabled arm owns a Lineage-local `agent-v0` that
freezes its chosen Workflow; Runtime does not infer topology from labels. Actual concurrency is
bounded by deployment settings.

All arms start from one frozen source-tree v0; controls never repeat Bootstrap/evaluation.

The runner saves per-arm seed IDs, logs/results and `campaign-results.json` under `run/`.
Actual Session/Artifact storage remains in the supplied Runtime. Failures do not cancel
other arms; rerunning resumes the same identities. Do not change frozen inputs to resume.
New control plans spend 15 Attempts per trajectory. `--target-epoch` is retained only for
compatibility with plans that enable the main arm.
Existing workspaces retain their frozen plan; pass the original absolute target when resuming
an older main-arm run.
This declares 24 Trajectories across eight arms; actual concurrency follows deployment limits. Nothing starts
during preparation. To run only the main arm, use the raw `bootstrap` and `run-campaign`
CLI commands instead of this runner.

Bootstrap launches the configured Agent on the imported source tree, records its
Direction/Experiment Journal and standard report, then independently finalizes v0. The normal
Runtime Gate controls staging, repeats and clocks. The original manifest's
`measurement`, `bringup`, and Repository Horizon lifecycle settings do not override it.

Inside an Optimizer workspace, Core/KDA Runtime Tools accept these requests. Every full
`evaluate` needs `latency_prediction`: choose `improved` for more than 1% lower geometric-mean
latency, `retained` for a change within ±1% (inclusive), or `degraded` for more than 1% higher
latency. ABBA compares B with A; ordinary Evaluate compares with the Kernel at the start of this
Attempt. `correctness_only` does not need a prediction. Runtime keeps the prediction for human
assessment without including it in the Agent result:

```json
{"operation":"evaluate","latency_prediction":"retained"}
```

```json
{"operation":"evaluate","mode":"correctness_only","input_path":"scratch/input.py","shapes_path":"scratch/shapes.json"}
```

```json
{"operation":"evaluate","latency_prediction":"improved","comparison":{"method":"abba","baseline_path":"scratch/previous-kernel-tree","repeats":2}}
```

Copy the **entire** historical Kernel Artifact for `comparison.baseline_path`. The fixed adapter is
`work/kernel/kernel.py`; editable files remain at their original paths directly below
`work/kernel/`. The same tools support source-tree diagnostics on NVIDIA:

```json
{"operation":"profile","level":"sol"}
{"operation":"profile","level":"deep","kernel_regex":".*GatedDelta.*","source":true}
{"operation":"check","sanitize":"memcheck"}
{"operation":"disassemble","fmt":"sass"}
```

For these diagnostics, Runtime stages the whole tree and fixed drivers through Agate Dev.
Ordinary Evaluate and ABBA use native Eval source archives. Provision NCU (Profile and
Disassemble) and Compute Sanitizer (sanitized Check) in the GPU image. No local Agent shell/GPU
execution is needed. Check triggers a one-case compile/launch probe, not full correctness.
Inspect the diagnostic `passed` field even when the operation completed. PTX output requires
toolchain support. Details and remaining limits: [source-tree contract](../../docs/source-trees.md).
