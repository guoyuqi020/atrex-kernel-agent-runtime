# FA4 SM120 implementation task

English | [中文](README.zh.md)

This task keeps the production operator, Reference, input generator, Shape population, evaluator,
and correctness policy from `data/FA4`, but targets **L20N / SM120**. It no longer supplies an
editable Vendor implementation or an SM120 bridge.

## Candidate and reference boundary

The Candidate source tree contains:

```text
kernel.py             fixed production-ABI adapter
implementation/       writable SM120 implementation
reference_sm103/      immutable original SM103-family implementation reference
PROVENANCE.json       immutable source provenance
```

`kernel.py` validates the captured ABI and calls
`implementation.sm120.flash_attention_sm120`. The seed function is deliberately unimplemented;
Bootstrap must create the first correct SM120 CuTe implementation. The reference tree retains the
pinned upstream FlashAttention CuTe and Quack sources for study, but it must not be imported as a
runtime fallback. Runtime permits changes only under `implementation/`.

The contract is FP8 E4M3 Query and P64 paged KV, BF16 output, 16 Query Heads, one KV Head, Head
Dim 256, ragged batches, PackGQA, bottom-right causal masking, and mutation of `out`. Returned
output and mutated `out` use the exact elementwise policy
`abs(candidate-reference) <= 0.06 + 0.04 * abs(reference)`.

Production source policy is enabled because it now scans only the self-authored implementation,
not the immutable reference library. Bootstrap and ordinary Evaluate use the shared Runtime Gate;
Retention and Agent Promotion use same-allocation ABBA. The L20N/SM120 Roofline retains the same
workload semantics and uses the configured SM120 peaks.

## Prepare and run

Run on Linux with Runtime, the model CLI, bwrap, and Agate credentials available:

```bash
cd ~/atrex-runtime
source env.sh
python3 scripts/source-tree/task.py prepare \
  --inputs data/FA4-SM120 \
  --workspace workspaces/FA4-SM120 \
  --backend claude
```

Preparation is offline and submits no GPU work. Start the service and Campaign separately:

```bash
python3 scripts/source-tree/task.py serve --workspace workspaces/FA4-SM120
python3 scripts/source-tree/task.py bootstrap --workspace workspaces/FA4-SM120
python3 scripts/source-tree/task.py campaign --workspace workspaces/FA4-SM120 --target-epoch 1
python3 scripts/source-tree/task.py inspect --workspace workspaces/FA4-SM120
```

The standard ablation remains available:

```bash
python3 scripts/source-tree/task.py ablation --workspace workspaces/FA4-SM120
```

Generated state lives only under the selected workspace; task inputs remain immutable.
