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

# Uses the existing services; Bootstrap once, then all seven ablation Campaigns:
python scripts/source-tree/run.py \
  --config "$ATREX_RUNTIME_CONFIG" \
  --campaign workspaces/gdn-source-tree/campaign.json \
  --plan workspaces/gdn-source-tree/ablation.json \
  --workspace workspaces/gdn-source-tree/run --target-epoch 100
```

Select hardware capable of the original SM103 task; this example does not reinterpret it as an
L20N task. The source commit comes from the existing manifest; the Optimizer commit must belong
to the deployment's configured base repository. Output must be a new directory. Resume with the
generated Campaign unchanged, rather than rerunning preparation over an existing workspace.

Default: CuteDSL only, with the same enabled matrix as single-file production: three independent
replicas each of Isolated, Retained, Pool-3, Pool-Retained-3, and Isolated-Evolve.
The legacy single-Lineage Evolve-3, Retained-Evolve, and Isolated-Pool-Evolve Workflows remain
available but are disabled.
The generated
`ablation.json` and checked-in `ablation.example.json`
use the shared production plan builder, not a second set of scheduling rules.
All Lineages start from one frozen source-tree v0; controls never repeat Bootstrap/evaluation.
Each runs 100 Epochs with three serial Attempts per trajectory. Isolated, Retained, and
Isolated-Evolve replicas have one trajectory (300 Attempts each); both pools have two (600 each). Isolated/Pool reset
adaptive State; Retained/Pool-Retained keep it. Pool trajectories share the winning Kernel
at Epoch boundaries; Pool-Retained also inherits the winning trajectory's terminal State.
The three Isolated-Evolve replicas run only their replicated/evolved Challenger, reset State per
Attempt, and perform no same-Epoch Active comparison. Each observes the matching Isolated replica
only through its preceding Epoch and waits if that observer is behind. Total: 6,300 Optimizer Attempts; no external
original-AKA run. Arms have independent later history/state.

The runner saves per-arm seed IDs, logs/results and `campaign-results.json` under `run/`.
Actual Session/Artifact storage remains in the supplied Runtime. Failures do not cancel
other arms; rerunning resumes the same identities. Do not change frozen inputs to resume.
New control plans spend 300 Attempts per trajectory. `--target-epoch` is retained only for
compatibility with plans that enable the main arm.
Existing workspaces retain their frozen plan; explicitly use `--target-epoch 5` to resume an
old five-Epoch experiment without extending its main arm. Single-file defaults are unchanged.
This can run twenty-one Optimizers concurrently; provision sufficient host memory. Nothing starts
during preparation. To run only the main arm, use the raw `bootstrap` and `run-campaign`
CLI commands instead of this runner.

Bootstrap launches the configured Agent on the imported source tree, records its
Direction/Experiment Journal and standard report, then independently finalizes v0. The normal
Runtime Gate controls staging, repeats and clocks. The original manifest's
`measurement`, `bringup`, and Repository Horizon lifecycle settings do not override it.

Inside an Optimizer workspace, existing Core/KDA Runtime Tools work unchanged:

```json
{"operation":"evaluate"}
```

```json
{"operation":"evaluate","mode":"correctness_only","input_path":"scratch/input.py","shapes_path":"scratch/shapes.json"}
```

```json
{"operation":"evaluate","baseline_path":"scratch/previous-kernel-tree","comparison":{"method":"abba","repeats":2}}
```

Copy the **entire** historical Kernel Artifact for `baseline_path`. The fixed adapter is
`work/kernel/kernel.py`; editable files remain at their original paths directly below
`work/kernel/`. The same tools support source-tree diagnostics on NVIDIA:

```json
{"operation":"profile","level":"sol"}
{"operation":"profile","level":"deep","kernel_regex":".*GatedDelta.*","source":true}
{"operation":"check","sanitize":"memcheck"}
{"operation":"disassemble","fmt":"sass"}
```

Runtime stages the whole tree and fixed drivers through Agate Dev. Provision NCU (Profile and
Disassemble) and Compute Sanitizer (sanitized Check) in the GPU image. No local Agent shell/GPU
execution is needed. Check triggers a one-case compile/launch probe, not full correctness.
Inspect the diagnostic `passed` field even when the operation completed. PTX output requires
toolchain support. Details and remaining limits: [source-tree contract](../../docs/source-trees.md).
