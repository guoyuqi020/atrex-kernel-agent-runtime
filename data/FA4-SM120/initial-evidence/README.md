# FA4 SM120 implementation task

Implement and optimize the public production Attention contract on **SM120**. The fixed
`work/kernel/kernel.py` preserves the production ABI and calls the implementation you own in
`work/kernel/implementation/`.

The target is FP8 E4M3 Query and paged KV, BF16 output, head dimension 256, 16 Query heads, one
KV head, P64 HND pages at the production boundary, ragged batches, bottom-right causal masking,
and mutation of the supplied `out` tensor. The public contract is authoritative for layout,
scaling, and Shape semantics.

This is a **native FP8-compute FA4 task**, not merely an FP8-input compatibility task. The
dominant QK and probability-V matrix products must use SM120 FP8 Tensor Core MMA, or an
equivalent native FP8 MMA data path. Eagerly converting the complete Query, Key, or Value
tensors to BF16/FP32 and then running a generic higher-precision Attention implementation does
not satisfy the intended computation. Use higher precision where Attention requires it—for
example softmax, scaling, reductions, and accumulation needed for numerical correctness—but
keep the principal matrix-multiply data path in FP8.

`work/kernel/reference_sm103/` is an immutable copy of the original implementation used by the
SM103-family task. Read it for its HD256, paged-KV, PackGQA, masking, scheduling, and launch
design. The most relevant files include `flash_fwd_sm100.py`,
`sm100_hd256_2cta_fmha_forward.py`, `paged_kv.py`, `pack_gqa.py`, `mask.py`,
`tile_scheduler.py`, and `utils.py`. SM103 Tensor Memory, `tcgen05`, TMA, cluster, and two-CTA
assumptions are design references only; they must not be copied blindly to SM120.

The reference tree is not a runtime dependency and is not editable. Do not import it from the
Candidate. Port or redesign the necessary ideas into `implementation/` and build a self-authored
SM120 CuTe operator. You may freely add, replace, or remove files under `implementation/`; do not
modify the fixed adapter, reference, or provenance files.

The seed implementation intentionally raises `NotImplementedError`. Bootstrap must author the
first correct measured SM120 implementation before Runtime can register `v0`. Use Runtime
Evaluate/Profile/Check results as evidence. SM120 source and the mixed-architecture dispatch
interface are deliberately absent. The task includes no prior SM120 implementation,
optimized result, Journal, or Conversation history.
