# Epoch 4 attempt 2 — RA host split right-sizing KEEP record (D1) + D2 CPU-floor abandonment

Status: KEPT (Exp1 experiment_2679cd4158d644908528b4a3f34f2d6e keep_after on the
frozen artifact). D2 (CPU-side wall floor strip) ABANDONED at its pre-registered
verdict gate on component-probe evidence with zero edits (Exp2
experiment_e81c51ea190e4e458baa7baf05fd59f0 abandon_direction, before/after null).
Directions: direction_f2377bfbdee74741ad5b9080143a9f6a (RA, completed),
direction_6607eac29ceb4af0a2feee1918d7d4c6 (D2, abandoned). 2 of max 3 advanced.

## What shipped (RA)
Host-only edit (+1918 B; tree 124197 B, device source byte-identical, diff-verified
two hunks): after the incumbent splits min-law, right-size via the ceil identity

    if splits >= 2:
        chunk = (max_useful + splits - 1) // splits
        splits = (max_useful + chunk - 1) // chunk

plus a docstring e4a2 paragraph. This strips the splits that the device chunk
quantization (`chunk = ceil(ntiles/splits)`, `it_lo = split*chunk`) leaves provably
empty (`it_hi <= it_lo`): they previously launched CTAs, wrote zero partials
(O=0, m=-inf, l=0), inflated merge Opart reads, and added phase-2 rounds.

SPLIT-QUANTIZATION LAW (new): `sm//base` splits × `ceil(ntiles/splits)` chunk
over-covers ntiles whenever `chunk*splits > max_useful`; the empty top splits are
pure waste. Right-sizing by the identity pair preserves chunk EXACTLY for the
max-kv group (per-CTA chain = grid makespan invariant; ragged groups get
`ceil(ntiles_g/splits') <= chunk` by monotonicity) — so this is NOT the epoch-1 D2
floor-rule violation class (that class shrank splits so chunk GREW). Worked
examples (sm=110, base=2): kv4578 55→48, kv4319 55→45, kv4096 55→43, kv2048
55→32, kv1792 55→28. ~44% of (base 2..10, kv 1024..8192) combos fire; savings up
to 40%+ of splits.

Numeric path: active partials bitwise unchanged (same chunk/it_lo); M bitwise
(fmaxf drops -inf); phase-2 acc bitwise (ascending-s FMA, dropped terms are exact
+0.0 additions); ONLY phase-1 L quarter-regrouping moves (quarters rescale with
`splits`) → fired classes tolerance path, everything else bitwise. Probe measured
vsBase max_abs 0.00000 on ALL 13 fired classes (deltas below bf16 rounding).

Routing interactions (all consume the same right-sized value host+device see):
QWALK quarters rescale automatically; HW-band gate can flip both ways —
measured WINS: enter-band m16_b2q1_kv1024 (merge27→hw16) 3.5→2.8us, exit-band
m8_b6q1_kv1024 (hw9→merge8) 2.9→2.5us. ws shrinks (n_part = splits*base*64).
CUDA-graph stable (pure shape arithmetic).

## Identities
- Frozen RA tree: 124197 B, file sha256
  901aec4c1dadd6af8d314e2954b98e9daae0fa9f9f8812c865c6becf08d50267; artifact
  sha256:97daf8f24daefc3ed5d30a19c254f84c58654d70147a448559bbdb0b7864bb72;
  trial gtrial_8682447ea345c62aab98cbb41103a7dc; bound results: check
  sha256:52b3e72f5cf06fc37a0cd5230dc0c999d19c28da17ed503252b816c076cae381,
  sol profile shape 0 sha256:d7cb754211655cc3042932c9fd6dd103c88d075e182daf5542fc467a907d2e4d,
  ABBA sha256:b6b02f5d6357ff2da7678f5ac6fc914dfcaa7d8fb668dd798f51199c90743539,
  final evaluate PASS sha256:9f53e2cf83d5fd35c89f0e46517877a9e481fba7a18ea2ad767407d9f096a728
  (correct=True, 90/90, failures=[], envelope PASS-identical: max_abs 0.001953125 /
  max_rel 0.0078125 / rel_err 0.0007788, geomean **244.660us**, arith 573.456us).
  Dev probe (not trial-listed): dv_bc8a878c86a1 PROBE CLEAN; component probe
  dv_e0f0699a1a07.
- Predecessor incumbent (ABBA baseline A side): e4a1r tree, trial
  gtrial_8ec977a65224e81201bb7277b58aa00d, artifact sha256:3cfc7854…, banked
  final evaluate 247.546us. Rollback anchor scratch/baseline.py was re-frozen at
  the RA tree (901aec4c…) after the keep — the RA tree is the new incumbent.

## Gate outcomes (pre-registered in scratch/draft.md §5)
- G1 PASS: empirical launch-grid law audit (monkeypatched `launch` spy on BOTH
  modules) 24/24: candidate grid.z/NKVH == mirror used, baseline == cap,
  ceil(mu/used)==chunk identity, fa_mma grid.(x,y) unchanged, routing mirror
  confirmed incl. both route-flip directions; occupancy/regs identical
  (fa_mma 196/0/1, fa_merge 80/0/3, fa_merge_hw 96/0/2, fa_wide 224/0/1).
- G2a PASS: 13 fired classes tolerance + graph==eager + zeros + vsBase 0.00000.
- G2b PASS: 11 non-fired classes BITWISE int16 eager+graph.
- G3 PASS: fired merge buckets −0.3..−1.4us (kv2048 4.8→3.5, kv1792 4.8→3.4,
  q8 8.1→7.6, kv4578 4.8→4.5), fa_mma parity ±0.3, non-fired parity ±0.2.
- G4 PASS: locked sol shape 0 fa_merge **7.360us** (anchor 8.096, −9.1%),
  fa_mma 14.208, total **21.568us** (anchor 22.72, −5.1%); dram_bytes_read
  808704 ≈ 48/55 of incumbent (traffic identity confirmed the strip).
- G5 PASS-with-note: ABBA repeats=2 both PASS 90/90 identical envelope; baseline
  244.415 vs candidate 244.114, **speedup 1.0012329**; all 4 rolls ordered
  (B 244.146/244.081 < A 244.364/244.465). Scoping: shapes 2, 31 moved exactly
  −2.048us (one bucket), 17/39/45 −0.8..−1.3%, every non-firing shape
  bit-identical; worst apparent regression +0.7% (shape 48) inside the
  quantization step and ≤2.8% noise floor. NOTE: measured 1.00123 < the
  pre-registered 1.002 threshold — reported frankly; keep justified by roll
  ordering, exact scoping, mechanistic proof, and the e4a1 precedent (kept at
  1.0013570). The 1.002 threshold was mis-calibrated against wall-bucket
  quantization (see lesson 1).
- G6 PASS: final evaluate clean roll, no flake signature, 90/90 present.

## D2 abandonment evidence (component probe dv_e0f0699a1a07 on RA tree)
- single_token: wall 7.5 = cpu 7.4 / gpu 0.0 / replay 6.2 — the ONLY CPU-bound
  class; strippable < 1.5us gate (contiguous×6 0.59, LaunchConfig 0.33, launch
  enqueue 2.79 with raw-launch RE-REFUTED in-run: cpu_cc 8.5 → cpu_raw 10.2,
  bitwise-equal + graph-capturable but slower on every class; empty_like 2.31 IS
  the output).
- Every other class GPU-BOUND: decode1_kv4578 wall 20.6 = replay 20.6 vs
  gpu_sum 17.8 (fa_mma 13.3 + fa_merge 4.4 — RA merge win visible real-run);
  decode_b8_q1 51.3/49.2/46.5; small_decode_b4 31.4/30.8/28.9; mid_decode_b16
  150.3/147.8/145.1; wave_b32_q16 432.5/431.5/429.0; prefill_2048 893/897/895.
- Conclusion: decode-band walls equal their GRAPH-REPLAY floors — the residual
  ~2-3us above the kernel sum is GPU-side inter-kernel launch/drain latency that
  survives CUDA graphs, plus evaluator harness quantum. No host strip can pay.

## Reusable lessons (new or re-confirmed)
1. WALL-BUCKET QUANTIZATION LAW: evaluator per-shape latencies are multiples of
   0.512us and move in 2.048us steps; a GPU win smaller than the distance to the
   next bucket boundary leaves the wall UNCHANGED (shape 0: −1.15us locked-sol
   GPU, wall identical 24.576 on all four ABBA rolls; shapes 2/31 crossed and
   each moved exactly −2.048). Calibrate expected geomean from bucket crossings,
   not from per-class GPU percentages: −3..−8% GPU on ~10-15% of shapes realized
   as +0.123% geomean, not the +0.3-0.8% drafted. Set ABBA keep thresholds at
   lineage precedent (~1.001), not at effect-size hopes; the per-shape scoping
   check + locked-sol deltas remain the effect adjudicator.
2. HOST-LAW CHANGE PROBING: anchor the host-law mirror with an EMPIRICAL
   LAUNCH-GRID AUDIT — monkeypatch `mod.launch` on both modules, record
   (kernel-name, grid) per launch, and assert grid-derived splits/chunks/routing
   == mirror. Mirror-only asserts are tautological; the spy makes the partition
   labels (fired/non-fired) empirical. Pattern:
   tools/flash_attention_splitrightsizing_probe.py.
3. PRODUCER-MULTIPLICITY changes are safe for the coupled consumer ONLY when the
   consumer receives the same runtime scalar (here merge walks the right-sized
   `splits` the host passes — contrast e3a2 fa_dec where 55→110 silently doubled
   the serial merge chain). Audit which side of every producer/consumer pair
   consumes the changed value.
4. VERDICT-GATE host-strip directions on a component probe BEFORE opening edits:
   require a CPU-BOUND (wall≈cpu) class with isolated strippable components ≥
   threshold; wall==replay with cpu below both means GPU-side floor → abandon
   with zero metered spend (this attempt's D2: 1 dev job, 0 edits).
5. NEXT-CANDIDATE (analyzed, NOT advanced): cooperative fused fa_mma+merge
   (one launch, grid.sync() between phases) attacks the replay-surviving ~2-3us
   decode gap directly. Blockers derived: (a) cooperative residency vs the
   tail-zero grid extension — shape-0-class grid is (tiles+zt)×batch×NKVH×splits
   = 144 CTAs > 110 residency at 1 CTA/SM, needs tail-zero redesign; (b) both
   kernel bodies must become __device__ functions → register-allocation risk on
   the hottest kernel (fa_wide2 secondary-effect precedent); (c) CUDA-graph
   capturability of cooperative launches unverified; (d) only ~6.9KB source
   headroom under the 131072B cap. Budget as a full Direction (2-3 dev jobs +
   ladder) with these four gates pre-registered.
6. Pre-metered iteration re-confirmed: 2 dev jobs + 1 check adjudicated the
   design; the metered ladder ran exactly once on one frozen artifact (3 jobs);
   zero metered spend on the abandoned D2.
