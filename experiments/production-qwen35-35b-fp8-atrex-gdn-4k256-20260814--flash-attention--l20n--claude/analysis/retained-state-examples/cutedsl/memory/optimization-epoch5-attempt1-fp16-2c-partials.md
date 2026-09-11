# Epoch 5 / Attempt 1 — fp16 raw split-KV partials for the 2C decode engine (KEPT, +0.869% trusted ABBA)

Operator `flash_attention`, DSL `cutedsl`, hardware `sm_120`.
Direction `direction_f0fb3c7a96da46c9beaa703b4f20e19f` (**completed**).
Direction `direction_83a1b6ce09ef45f882a3f353dfde9304` (pipeline depth, **abandoned**
under its own pre-registered stop condition).
Experiments: `experiment_fb201c684d35460ca119bc7573be0db5` (D1 falsifier, no edit),
`experiment_2799856f0ecf4d99a2b9111ca8e39d64` (D2 keep, `action: keep_after`).

## Decision and shipped change

Candidate **C5a**, `work/kernel/kernel.py`, 130,955 B (cap 131,072, headroom 117 B;
parent 131,067 B), file sha256 `4b912b74f4c873bd59cfe46527e239ef1866e2d9b5532101c612d81739ec2707`,
artifact `sha256:18c8aecf0959e1112b8732ff6d2fddb50977fdc6347a8f6ce920888903590f3d`,
trial `gtrial_074741e35283ead47929bb0877a69330`. Parent = the unchanged incumbent
(artifact `sha256:980ea918…`, trial `gtrial_039ccfbd22700cb877e78e2e62ea18fb`,
sealed geomean 237.615 us).

Mechanism: the **2C decode engine only** stores its raw (unnormalised) split-KV
`o_part` accumulator as **fp16** instead of fp32 (1032 → 520 B per row-split);
`ml_part` stays fp32. The old engine, the big prefill engine, the fused
cooperative split-merge tail and **every host cost-model constant** are untouched,
so dispatch is bit-identical. Six code sites; prose paid for the bytes by moving
three historical blocks verbatim into
`knowledge/cutedsl-sm120-decode-roofline.md`'s prose vault.

## Measured facts (trusted)

* Compile gate `check` job `cp_7d109aa1c59b`: succeeded, `ok: true`, `diagnostics: []`.
* `correctness_only` job `ev_1ed1cf28c355`, result `sha256:e6a16f0c…`: correct,
  PASS, `failures: []`, `max_abs_err 0.001953125` (= 2⁻⁹, one bf16 quantum),
  `max_rel_err 0.03204345703125` — **identical to the incumbent's floor**.
* **Promotion statistic**, trusted paired ABBA (A-B-B-A, repeats 2, 90 contract
  shapes, geometric_mean), result `sha256:2584eb59…`:
  **speedup 1.0087647 / improvement 0.8689%**; baseline geomean 235.39875 us →
  candidate 233.35347 us. Per-observation geomeans A 235.0812, B 233.3984,
  B 233.3085, A 235.7167 — both B's below both A's; the two B's agree to 0.039%.
* Band attribution, each shape against its **own** within-side repeatability
  (`|obs1/obs2 − 1|`: median 0.054%, mean 0.445%, p90 0.894%). The mix's largest
  latency gap is 269.82 → 893.44 us, which is the decode/prefill boundary:

  | band | n | geomean speedup | faster beyond noise | slower | tie |
  |---|---|---|---|---|---|
  | decode <300 us | 68 | **1.01165** | 29 | **0** | 39 |
  | prefill ≥300 us (untouched code) | 22 | 1.00039 | 1 | 1 | 20 |

  Sub-bands fall monotonically: 0–60 us 1.01444, 60–150 us 1.01286,
  150–300 us 1.00990. 12 shapes are bit-identical across all four observations.
  Largest single win: shape 46, 1.1704 (+17.0%).
* Sealing full `evaluate` of the nominated tree, result `sha256:fff7a0d1…`:
  90/90 PASS, `failures: []`, geomean **236.768 us** (incumbent's recovered sealed
  237.615 us). Cross-job geomeans are a consistency check only — cases re-draw per
  job, so per-shape values are not comparable across jobs.

## Reusable lessons

1. **Use the mix's own latency gap as the band boundary.** Splitting at a round
   250 us silently put 12 top-decode shapes into the "prefill" control group and
   understated the acted-on band (1.01298 → 1.01165 after correction) while making
   the control group look noisier than it is. Sort the 90 baseline latencies and
   look for the gap; here it was 623.62 us wide.
2. **Judge per-shape effects against each shape's own repeatability, not a global
   threshold.** With A-B-B-A and repeats 2 the within-side spread is 0.054% median
   but 0.894% at p90, so a fixed ±0.5% bar both over-claims on quiet shapes and
   under-claims on noisy ones. Doing this turned "11 shapes slower" (side-aggregate
   view) into "1 slower beyond noise, in the untouched band".
3. **A uniform census over the declared domain over-predicts byte-mechanism gains.**
   `tools/byte_share_census.py` predicts a median 9.13% traffic saving (speedup
   1.1004) over the 309 acted-on grid points, but the contract mix measured
   +1.17%: the grid's median acted point has an 18.4% partial round-trip share
   while the two profiled contract shapes had 3.6%. The contract's decode cases sit
   at the **large-kv, low-share** end. Size byte-removal candidates from profiled
   contract shapes and use the grid only for *ordering* (which it got right: saving
   13.61% at 8–32 MB KV → 7.17% at ≥32 MB; measured gain falls with latency).
4. **Freezing the cost model is what makes a dtype change provably safe.** With
   every constant untouched, dispatch parity is checkable offline and for free
   (`tools/diff_code_tokens.py`: 163 changed code tokens, forbidden-token list of
   all cost constants → PASS), so the ABBA delta can only come from the mechanism.
   Lineage precedent: dtype + refit lost (+0.235%) where dtype with a frozen basis
   won (+0.057%); this attempt's frozen-basis win is +0.869% because the band it
   acts on is at the DRAM roofline.
5. **fp16 raw partials are precision-free at this output quantisation.** The error
   floor did not move at all (still exactly one bf16 quantum), so the
   range-safety fallback (store normalised `Ohat_s = O_s/l_s`) is unnecessary here —
   but keep it registered: raw fp16 overflow needs `max|v| > 28.6`, which is an
   input-distribution assumption, not a proof.
6. **A promoted candidate's follow-ups must be priced against the byte cap.** At
   117 B of headroom, extending fp16 partials to the old engine (which serves the
   sub-60 us shapes where partials really are L2-warm: +0.057% measured previously)
   or reworking `fa_merge` parallelism (coupled to `merge_rpc`/`merge_grid`, hence
   forces a refit) both cost a full ABBA + sealing cycle for sub-noise expected
   value. Stopping was the correct call, not a missed opportunity.

## Open / unverified

* Register pressure of the added fp16 fragment is **unverified** (2C engine ran
  235 regs at a 256 ceiling for 2 CTA/SM; no candidate profile was taken). Zero
  decode regressions beyond noise is indirect evidence of free disjoint liveness,
  not a register count. A `profile level=sol` on the nominated tree would settle it.
* Shape 81's −0.57% sits in the untouched prefill band (20 of 22 ties) and is read
  as noise, but `fa_merge` is shared by every engine, so a small prefill-side effect
  from the merge's new `p32` fragment cannot be formally excluded at repeats 2.
* 39 of 68 decode shapes were within noise. Opaque shape ids cannot say which had
  `S == 1` (genuinely unchanged) versus `S > 1` with a low partial share, so that
  tie count mixes two populations.
* The `tps >= 2` flip-family residue and the arithmetic-progression argmin remain
  open levers inherited from epoch 4 (see
  `knowledge/cutedsl-sm120-flash-attention.md`); neither was touched here.
