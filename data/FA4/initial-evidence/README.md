# FA4 prefill source-tree task

Implement and optimize the supplied upstream FlashAttention CuTe implementation for the
public production Attention contract. The complete source tree is already in `work/kernel/`.
Read its fixed `PROVENANCE.json` for the source version and intended capability gaps.

The intended starting route is HD256, two CTA, M128/N128, with paged FP8 K/V. Upstream
supports one P128 page; the target uses P64 pages with independently mapped physical pages.
Extend the N128 path to compose two P64 pages, respect the actual KV lengths (`seqused_k`),
and implement dedicated PackGQA semantics. The public contract is authoritative for tensor
layout, scaling, ragged sequences, causal masking, and mutation of the provided `out` tensor.

Likely relevant files under `vendor/flash_attention/flash_attn/cute/` include
`sm100_hd256_2cta_fmha_forward.py`, `interface.py`, `mask.py`, `tile_scheduler.py`,
`paged_kv.py`, `pack_gqa.py`, `flash_fwd_sm100.py`, and `utils.py`.
All changes must remain in the Runtime-declared editable source directory. The adapter and
Quack support are fixed. Do not bypass the missing capabilities in the adapter, replace this
task with a single-file implementation, or fetch a previously optimized implementation.

The unchanged R0 is expected to fail the target contract. Bootstrap must repair it and obtain
a correct measured candidate before Runtime can register `v0`; R0 is not an accepted baseline.
The source package's earlier P128 smoke test establishes only compilation/execution of its
original capability. Its relative-L2 smoke threshold is not the target correctness policy.
Use Runtime tools for target evaluation and profiling; do not infer acceptance from a smoke
test. No prior optimization history or optimized C05/Increment source is included.
