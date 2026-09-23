# FA4 prefill source-tree optimization task

English | [中文](README.zh.md)

This independent task combines the supplied Qwen3.8-Max Attention Atrex-Bench problem and
pristine FA4 R0 source package. Defaults: **L20D, CuteDSL, Claude**, one DSL Lineage, Wiki off.
Inputs and offline Bundles live here; all generated state lives in `workspaces/FA4/`.

## Objective and source boundary

Preserve the production ABI of `flashinfer.prefill.trtllm_batch_context_with_kv_cache`, but
implement it with the supplied FlashAttention CuTe source, not an installed FlashInfer operator.
The target has FP8 E4M3 Query/KV, BF16 output, 16 Query Heads, one KV Head, Head Dim 256,
P64 KV pages and causal Attention. `shape_train.json` supplies the public domain; the 30
precise validation Shapes, Metadata and Roofline remain private.

The complete CuTe tree is pinned to upstream FlashAttention
`b54df166ebb69b896892826014759d09b9c3c9c6`, with Quack 0.5.3 source support. R0 supports the
HD256 two-CTA P128 path but lacks the target capabilities. Bootstrap must compose two independently
mapped P64 pages into an N128 Tile, implement authoritative `seqused_k` and dedicated PackGQA,
then pass the complete Bootstrap Gate before Runtime registers `v0`. R0 is not an accepted baseline.

Only `vendor/flash_attention/flash_attn/cute/` is editable. Runtime locks the adapter, Quack and
other supplied files. The initial Evidence contains task instructions, not another run's
optimized implementation, Journal or Conversation. Agents edit Session `work/kernel/`, not the
fixed source Checkout. No C05/Increment optimization is included.

These locks are this task's conservative packaging policy. The original instructions explicitly
permit the complete CuTe tree and prohibit bypassing capability gaps in the adapter; they do not
explicitly require the adapter or `vendor_support` to remain entirely read-only. A frozen starting
snapshot is not a runtime immutability rule; the Runtime Manifest defines that rule here.

## Inputs and provenance

- `task/`: fixed `adapter.py`, Source Manifest, Reference, input generator, public Shape Train,
  private Shape Valid, Metadata and the supplied Roofline. The adapter becomes Candidate `kernel.py`.
- `source.bundle`: offline FA4/Quack package, Commit `7b077cf391a98a06ad9464530f0a4c9c7be3f477`.
  This packaging Commit differs from upstream because it includes dependencies/provenance;
  the FA4 source bytes are unchanged.
- `evaluator.bundle`: supplied Atrex-Bench evaluator with metadata-owned elementwise tolerances,
  Commit `ed449b63ecd8aeff4db23be0d9f658d7d50b1cfa`. It does not use or alter the existing
  `third_party/atrex-bench`.
- `source-provenance.json` and `source-validation.json`: original source and historical smoke
  records. They are not acceptance results for this Campaign.
- `smoke/`: unchanged original Agate smoke scripts and legacy Shape document.
- `asset-integrity.json`: hashes checked against both supplied packages, including all 70 Source
  files and all 32 Evaluator files. Preparation verifies these and the offline Bundle hashes.
- `initial-evidence/`, `campaign.json`, `runtime.template.json`: task instructions and independent
  configuration. Preparation also generates a frozen `ablation.json` from the shared production
  topology. Optimizer and Evolver are pinned to the current KDA/Evolver Commits.

Preparation verifies offline Commits, source locks, public/private Shapes and the actual
Optimizer, Evolver and Evaluator import paths. Nothing depends on the original directories
outside this repository.
See [alignment audit](ALIGNMENT.md) for exact file checks and evaluation-policy differences.

## Evaluation policy and limitations

Both returned output and mutated `out` must satisfy elementwise
`abs(candidate-reference) <= 0.06 + 0.04 * abs(reference)`. Metadata owns these tolerances;
legacy L2/mismatch-rate options cannot weaken them. `workspace_buffer` is declared scratch,
`out` is declared mutable; remaining inputs retain the evaluator's side-effect checks.

Bootstrap uses 1-Case then 5-Case stages; ordinary Evaluate uses 5 Cases, 100 Bench Iters and
one logical Evaluate. Retention and Agent Promotion use same-allocation ABBA. One Shape per
batch, up to 16 concurrent batches, clocks locked by default. Agent/Evolver history and
normal Runtime tools remain available. In eager mode `warmup_iters=10` and `bench_iters=100`
are 10ms and 100ms budgets, not fixed run counts. ABBA executes its complete schedule once,
without an extra cross-job median.

The **Production static source Gate is off for this task**. It scans every editable file;
the complete upstream FA4 tree contains test/benchmark helpers and dependencies incompatible
with that single-DSL scan. Global policy is unchanged; correctness, source locks, edit scope
and Runtime comparisons remain enforced. Adapting that scan is required before enabling it
for this whole-library task.

The supplied smoke record shows upstream P128 execution succeeded and the target ABI hit
the expected R0 assertion. The old smoke L2 threshold is not used for acceptance. The GPU image
needs `torch>=2.9.0`, `nvidia-cutlass-dsl==4.6.1`; Quack is bundled. Preparation neither installs
GPU dependencies nor verifies the remote image, model login or connectivity.

The supplied Roofline is reused; no builder is launched. Its label is `NVIDIA B300 (SM100)`.
In this deployment Agate resource `L20D` denotes B300. Preserve the original values; transport
only strips the parenthesized hardware suffix according to the existing Runtime rule. Carrying
Roofline does not guarantee a returned SOL result. Use `profile` for bottleneck evidence when needed.

## Prepare and run

Use Linux/Lima with runtime installed in the active Python environment, not the shared macOS
`.venv`. The selected CLI must be installed/authenticated, and bwrap must work. Container mode
requires neither systemd nor a per-Session cgroup. Export `AGATE_AK`, `AGATE_SK`; `AGATE_URL`
optionally overrides the endpoint.

```bash
cd ~/atrex-runtime
source env.sh
python3 scripts/source-tree/task.py prepare --backend claude
```

Preparation starts no service, model or GPU job. The default Runtime port is 8770; override
with `--port` during preparation. Supported backends: claude/codex/qodercli/pi. Every command
accepts `--workspace workspaces/FA4-trial-2`. Inputs/backend/port are frozen; existing workspaces
are never overwritten. Use a new workspace for a revised experiment.

Optional: run the unchanged original smoke scripts. These submit Agate Dev jobs directly and
require `agate` on PATH, but no Runtime service. They never register `v0`; target R0 is expected
to fail:

```bash
python3 scripts/source-tree/task.py smoke --smoke-mode upstream-p128
python3 scripts/source-tree/task.py smoke --smoke-mode target --shape-id 0
```

The runner stages pristine R0, its fixed adapter and original `reference/shapes.json` in a
temporary directory, executes the original scripts and cleans up afterward. It does not test
an Agent-modified Candidate and is not a replacement for full evaluation.

Terminal one (Ctrl-C stops the service):

```bash
python3 scripts/source-tree/task.py serve
```

Terminal two, first validate source bring-up:

```bash
python3 scripts/source-tree/task.py bootstrap
```

Then optimize through the first Epoch; `campaign` also runs/reuses Bootstrap automatically:

```bash
python3 scripts/source-tree/task.py campaign --target-epoch 1
python3 scripts/source-tree/task.py inspect
```

The main Campaign has one Trajectory per Branch and three serial Attempts per Epoch. Epoch 1 runs
the same Agent Revision in isolated Active and Challenger-replica Branches; no Evolver runs.
From Epoch 2, one Evolver-generated Challenger competes with Active. Resume through a later
absolute target with `campaign --target-epoch 3`; it does not add three Epochs.

To launch the complete production ablation against the same frozen Bootstrap seed, keep the
Runtime service running and use:

```bash
python3 scripts/source-tree/task.py ablation
```

This starts fifteen independent Campaign schedulers: three replicas each of Isolated, Retained,
Pool-3, Pool-Retained-3, and Isolated-Evolve. The default target is Epoch 5, giving every Trajectory
exactly 15 post-Bootstrap Optimizer Attempts. Main Evolve-3, Retained-Evolve, and
Isolated-Pool-Evolve remain implemented but are disabled.
All arms share `v0`, but do not share later history or writable State. Results and per-arm logs live
under `workspaces/FA4/ablation-run/`. Never run `campaign` and `ablation`, or two schedulers for the
same Campaign, concurrently.

The workspace contains task snapshots, fixed source/evaluator Git Checkouts, generated Runtime
and Evaluation Contract JSON, `prepared.json`, Bootstrap/Epoch results and `state/` with Registry,
Artifacts, Sessions and worker workspaces. Runtime credentials are generated once into
`runtime-secrets.json` (0600), shared by service and task processes; manual key generation is
unnecessary. Resume on the same execution host; prepare there so worker interpreter paths are valid.

Inspect further with the normal CLI; IDs come from `bootstrap-result.json`:

```bash
atrex-kernel-agent-runtime list-attempts --config workspaces/FA4/runtime.json --lineage <lineage_id>
atrex-kernel-agent-runtime list-worker-sessions --config workspaces/FA4/runtime.json --campaign <campaign_id>
```

Before successful Bootstrap, that result file may not exist. Read the terminal error and Sessions
under `state/lineage-bootstrap-workspaces/` instead. Passing preparation checks does not mean R0
passes correctness; the first model session must still implement the missing capabilities.
