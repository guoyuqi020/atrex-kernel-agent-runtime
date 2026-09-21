# FA4 SM120 source-tree optimization task

English | [中文](README.zh.md)

This is the SM120 counterpart of `data/FA4`. It keeps the same production
`flashinfer.prefill.trtllm_batch_context_with_kv_cache` ABI, Reference, input generator and
30 private Shapes, but executes on **L20N / SM120** and starts from an SM120 FA4 Vendor route.
It does not optimize an installed FlashInfer implementation.

## Starting implementation

The fixed adapter receives FP8 E4M3 Query and P64 paged KV, produces BF16 output, and preserves
16 Query Heads, one KV Head, Head Dim 256, ragged lengths, PackGQA and causal semantics.

Upstream FlashAttention Commit `b54df166ebb69b896892826014759d09b9c3c9c6` only provides an
SM120 FP16/BF16 dense kernel. The packaged Source Commit
`b6bfe3d177aab2b930f4d6485227002b65cbb2de` therefore adds a correctness-first bridge inside
the editable Vendor:

1. synchronize `seqused_k` to the Host;
2. gather independently mapped P64 pages into dense K/V;
3. convert Q/K/V from FP8 to BF16;
4. run the upstream SM120 M64/N64 FA4 kernel.

This creates an explicit, measurable R0. The intended optimization is to remove those costs and
build an SM120-native FP8 paged-KV path. SM100 HD256 2CTA, Tensor Memory, `tcgen05` and its TMA
route are not valid assumptions on SM120.

Only `vendor/flash_attention/flash_attn/cute/` is editable. The adapter, Quack support,
Reference and contracts are fixed. See [initial Evidence](initial-evidence/README.md) for the
optimization brief and [alignment audit](ALIGNMENT.md) for the exact relationship to `data/FA4`.

## Evaluation

- Hardware target: `L20N`; Agent-visible architecture: `sm_120`.
- Correctness: returned output and mutated `out` use the elementwise rule
  `abs(candidate-reference) <= 0.06 + 0.04 * abs(reference)`. Runtime injects this exact policy
  into every Bootstrap and Optimizer session.
- Bootstrap: 1 Case then 5 Cases; Optimizer Evaluate: 5 Cases and a 100ms benchmark budget.
- Retention and Agent Promotion: same-allocation ABBA, one Shape per batch, up to 16 batches.
- Roofline: recomputed for the L20N SM120 SKU using 549 TFLOP/s FP8 and 1344 GB/s HBM peaks.
- Production static source Gate remains off because it scans the complete editable CuTe library;
  correctness, source locking, edit scope and Runtime comparisons remain enforced.

The Shape values originate from the same production callable captured on L20D. They define the
workload, not the target hardware performance. Per-shape historical L20D measurements remain in
Metadata as provenance; the task target and Roofline are explicitly L20N/SM120.

## Prepare and run

Use a Linux host with the Runtime, selected model CLI and bwrap installed. Export Agate
credentials first.

```bash
cd ~/atrex-runtime
source env.sh
python3 scripts/source-tree/task.py prepare \
  --inputs data/FA4-SM120 \
  --workspace workspaces/FA4-SM120 \
  --backend claude
```

Preparation is offline and starts no service, model or GPU Job. The generated Runtime defaults
to port 8771. An optional direct Agate smoke can validate either the upstream SM120 BF16 kernel
or one target Shape:

```bash
python3 scripts/source-tree/task.py smoke --workspace workspaces/FA4-SM120 \
  --smoke-mode sm120-bf16
python3 scripts/source-tree/task.py smoke --workspace workspaces/FA4-SM120 \
  --smoke-mode target --shape-id 0
```

Run the Runtime and Campaign in separate terminals:

```bash
python3 scripts/source-tree/task.py serve --workspace workspaces/FA4-SM120
```

```bash
python3 scripts/source-tree/task.py bootstrap --workspace workspaces/FA4-SM120
python3 scripts/source-tree/task.py campaign --workspace workspaces/FA4-SM120 --target-epoch 1
python3 scripts/source-tree/task.py inspect --workspace workspaces/FA4-SM120
```

The normal ablation entry is also available:

```bash
python3 scripts/source-tree/task.py ablation --workspace workspaces/FA4-SM120
```

All generated state remains under `workspaces/FA4-SM120`; the inputs under `data/FA4-SM120`
are never modified.
