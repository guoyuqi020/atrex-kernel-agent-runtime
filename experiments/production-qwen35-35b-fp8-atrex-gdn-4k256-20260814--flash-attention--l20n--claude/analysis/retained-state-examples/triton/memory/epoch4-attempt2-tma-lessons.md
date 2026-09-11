# Epoch 4 / Attempt 2 — TMA async KV staging: probe-only NO-GO (measured facts vs interpretation marked)

Outcome: champion gtrial_aa6c9768a152055e45c900fff31e049b UNCHANGED (PASS 90/90, geomean 258.57 us
stands). direction_df3c223ba8a3482d96dfb08861fcbb33 abandoned with
experiment_b4f87fed8ab84c9ab815089df826ebd3 (action=abandon_direction, before=null after=null —
probe-only format reconfirmed). Terminal report status=pivot. ZERO evaluate/ABBA/profile budget
spent; all facts from 3 Dev probes (dv_7bfeb386b913 legality-a1, dv_f5effb3c5521 legality-a2,
dv_670ff7308761 bench-b; all exit 0, FAILURES 0) on scratch/kernel_tma.py + scratch/probe_tma.py.

## MEASURED FACTS

### sm_120 / Triton 3.7.1 TMA support (the errata update that outlives this attempt)
- Host-side `TensorDescriptor.from_tensor(t2d, [BN, HD])` + in-kernel `desc.load([row, col])`
  WORKS on sm_120 (RTX PRO 5000, torch 2.9.0+cu130): PTX contains real `cp.async.bulk.tensor`
  (x2 sites BN=64 eff-st1, x4 BN=32 st2-4) with `mbarrier` (x10/x14). TMA_OK import from
  `triton.tools.tensor_descriptor`. Descriptors passed by value; creation is host-only
  (view + from_tensor, no device sync, no set_allocator) → CUDA-graph capture/replay verified
  clean, replay output bit-identical to eager.
- warp_specialize still hard-fails this compiler/pattern (E2T2A2) — TMA descriptors are the ONLY
  working async-copy route on sm_120, and they do lower.

### Envelope (all 0 spills; regs BELOW champion's 200-204)
- BN=64 TMA st2 requested → smem OOR (compile metadata 106520 B > 101376 wall) → _launch_k1
  fallback to eff (8,1): smem 73728, 182-190 regs.
- BN=32 TMA achieves eff (8,2), (8,3), (8,4): smem 69656-69668, 168-174 regs. NOTE st3 and st4
  report identical smem AND identical bench times — Triton's pipeliner saturates buffer depth at
  st3 for this loop (requested stages beyond that are no-ops).
- Paged-KV-in-2D-view pattern validated: descriptors over `cache.view(pages*64, num_g*256)`,
  runtime row offset `page*64 + n0%64`, col offset `g*256`; requires EVEN_PAGE (BN ≤ PAGE and
  PAGE % BN == 0) so each box stays in one page. Phase-2 (masked diagonal/tail) must keep regular
  masked loads (garbage-column safety).

### Correctness
- BN=64 TMA phase-1 outputs are BIT-IDENTICAL to champion (torch.equal on prefill case; same
  abs AND rel digits on all 5 probe cases) — descriptor loads at unchanged tile shape/order do
  not perturb arithmetic.
- BN=32 TMA: abs margins equal-or-better than champion (worst_mixed 0.000362 vs 0.000401;
  ragged/pack8192 identical), all contract-PASS; rel-digit drift only (shorter softmax chunks).
- Gate-off bit-identity (TMA_ON=0 and gate-on-regime-off) held in all 3 regimes.
- Forced split/nosplit PASS with TMA firing on the split path (effcfg key split=True → (8,3)).

### Bench (do_bench warmup=25 rep=100; base = champion, same session)
Big-prefill proxies (pre_q1024, q2048x2, q3072x3, q4319, pack8192, b8_q1024):
- tma-bn64-w8-st2 (eff st1): median 0.921x, range 0.920-0.960 across ALL fired proxies (incl.
  pre_q256 0.955x, ragged_dev3 0.960x). Uniform ~8% loss at equal effective stage depth.
- tma-bn32-w8-st3: median 0.772x; st4: 0.772x (identical times — saturation); st2: 0.752x.
- Warts (mixed batches, gate fires whole-batch): mix_dev2 1.001x / worst_mixed 1.008x (bn64,
  noise-level); 0.84-0.88x (bn32).
- Decode/small sanity with gate off by regime: 0.997-1.002x neutral.
- Firing during bench proven via _CFG_OK keys (use_tma=True position) + PTX grep on the same
  compiled binaries.

## INTERPRETATION (the reusable negative result)

1. **The memory-supply hypothesis class is now CLOSED for the big tier.** E4A1: cutting KV
   traffic 0.75x was neutral (savings eaten by 1.5x MMA). E4A2: multiplying async staging depth
   (1 → 3-4 tiles in flight, 0 spills, lower regs) LOSES 8-25%. Two orthogonal memory-side
   interventions both failed to convert → the >1000us wall is NOT memory supply (neither volume
   nor in-flight bytes). It is the serial per-iteration dot→softmax→dot dependency chain at
   1 CTA × 8 warps/SM (occ 16.64%) — a compute-latency/ILP wall. Any future big-tier idea must
   change the DEPENDENCE STRUCTURE (more independent work per SM: multi-CTA rows, wider M with
   host-decidable gating, 2-CTA cooperation, softmax/MMA overlap), not the memory path.
2. **TMA at equal geometry is pure overhead here**: BN=64 TMA eff-st1 vs champion regular-load
   eff-st1 = 0.92x (descriptor/mbarrier bookkeeping + possibly worse smem swizzle for MMA
   operands, with no depth gain since st2 is OOR either way). BN=32 TMA st3 (0.772x) == historical
   regular-load BN=32 w8 st3/st4 (0.77x, E3A3-era): async depth added nothing at equal tile
   geometry; the BN=32 loss is tile-width doubling (2x iterations, 2x page-table loads, 2x
   softmax rescale passes).
3. **Process wins that made this cheap**: gate-decidability screen BEFORE implementation (E4A1
   lesson applied — host-visible regime keys, no device-side loser class); PTX-grep proof of real
   lowering as the first legality gate (would have caught a silent sync fallback); champion margin
   calibration on shared fp32 refs; pre-registered REJECT clause ("deterministic <0.98x loser one
   refinement cannot remove") made the NO-GO mechanical — no refinement existed (BN=64 at max
   smem-legal stage; BN=32 dominated by geometry; st4≡st3 saturation), so no re-deciding job was
   burned. 3 Dev probes, zero trusted budget, ~1 session.
4. Bit-identity of a same-geometry data-path swap (BN=64) collapses correctness validation to
   torch.equal — use this pattern for any future load-path change.

Companion tools: `tools/probe_tma.py` + `tools/kernel_tma.py` (variant kept as the reference
TMA integration — do NOT re-bench to re-litigate; reuse only if a future structure needs
descriptor loads, e.g. 2-CTA or multi-row-CTA designs where smem staging is a means, not the
hypothesis).
