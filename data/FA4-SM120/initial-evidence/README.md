# FA4 SM120 prefill source-tree task

Implement and optimize the supplied FlashAttention-4 CuTe implementation for the public
production Attention contract on **SM120**. The complete source tree is already in
`work/kernel/`; read its fixed `PROVENANCE.json` before changing it.

The target contract is FP8 E4M3 Query and paged KV, BF16 output, head dimension 256, 16 Query
heads, one KV head, P64 NHD pages, ragged batches, bottom-right causal masking, and mutation of
the supplied `out` tensor. The public contract is authoritative for layout, scaling and shape
semantics. Do not infer SM100 capabilities from the similar Blackwell product name.

The R0 Vendor deliberately uses a correctness-first SM120 bridge in
`vendor/flash_attention/flash_attn/cute/interface.py`: it synchronizes `seqused_k` to the Host,
gathers P64 pages into dense K/V, converts Q/K/V from FP8 to BF16, then launches the SM120 FA4
M64/N64 warp-MMA kernel. This is a functional starting route, not an acceptable performance
design. Measure before editing and preserve correctness while removing these costs.

High-value SM120 directions include:

- eliminate Host length synchronization and Python per-request assembly;
- load the independently mapped P64 KV pages directly in the CuTe mainloop;
- consume FP8 Q/K/V without materializing BF16 tensors;
- preserve authoritative `seqused_k`, PackGQA and causal tail alignment;
- tune SM120 tiles, stages, warp roles and scheduling within its 99-KiB shared-memory limit.

The SM100 HD256 2CTA/Tensor-Memory path, `tcgen05`, and its TMA assumptions are not a valid
starting design for SM120. The relevant starting files are `interface.py`,
`flash_fwd_sm120.py`, `flash_fwd.py`, `paged_kv.py`, `pack_gqa.py`, `mask.py`,
`tile_scheduler.py`, and `utils.py`. All changes must remain in the Runtime-declared editable
Vendor directory. The fixed adapter and Quack support must not be modified, and the task must
not be replaced by a single-file fallback or a fetched optimized implementation.

Use Runtime Evaluate/Profile results as evidence. The supplied shapes were captured from the
same production callable on L20D, but this task's execution target and Roofline are L20N/SM120.
No previous optimized SM120 implementation or optimization history is included.
