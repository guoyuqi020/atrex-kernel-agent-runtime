"""Host-dispatch code-sim of this lineage's flash_attention forward().

Replicates the pure-host dispatch arithmetic of work/kernel/kernel.py forward()
(no GPU, no tensors): engine pick, fold geometry, S models, fuse gate, grid.
Constants mirror the epoch4-a3 NOMINATED tree (BN16 2C engine, refit 2C
constants 1.70/2.34, tps>=1 clamp via CLAMP_TPS1_2C).  main() still runs the
epoch4-a2 small-prefill->2C gate audit (that gate is FALSIFIED -- retained as
the worked example); the importable dispatch() is what later tools build on
(census_refit_delta.py overrides CTA_C0_2C_US/TILE_US_2C to diff constant sets).
Audit questions for any candidate gate:
  1. decode domain (msq <= 32) delta count must be ZERO unless measured;
  2. big-grid / deep-sweep prefill delta count must be ZERO;
  3. deltas only where the sweeps measured 2C wins.
Usage: python3 gate_sim.py R_MAX T_MAX   (e.g. 1.0 16)
"""
import json
import math
import sys

# ---- module constants (epoch4-a3 nominated tree: BN16xSTAGES2 2C refit) ----
# 2C arm constants refit on the BN16 engine (dv_aa53b5f08fc0); were 4.14/1.78
# for v10's STAGES=1 engine. NSTAGES_2C/BLOCK_N are dispatch-irrelevant here
# (the sim prices decisions only) but kept in sync as documentation.
BLOCK_M, BLOCK_N, NSTAGES, NTHREADS = 64, 32, 2, 128
BLOCK_M_BIG, NTHREADS_BIG, R_BIG, FOLD_BIG, NSTAGES_BIG = 128, 256, 16, 8, 2
NSTAGES_2C, CTAS_PER_SM_2C = 2, 2
S_CANDIDATES = (2, 3, 4, 6, 8, 12, 16, 24, 32, 48)
SCRATCH_CAP_BYTES = 256 << 20
CTA_C0_US, TILE_US = 2.60, 1.720
CTA_C0_2C_US, TILE_US_2C = 1.70, 2.34
MERGE_FIX_US, MERGE_LAT_US = 1.40, 0.0700
MERGE_L2_BYTES_PER_US, MERGE_DRAM_BYTES_PER_US = 3.2e6, 1.3e6
MERGE_L2_FIT_BYTES = 224.0e6
SCRATCH_WRITE_BYTES_PER_US = 1.8e6
CLAMP_TPS1_2C = True   # shipped e4a3: 2C arm skips tps==1 candidates
MERGE_ROWS_PER_CTA = 4
TAIL_CTAS_MAX, TAIL_ROWS_PER_CTA = 256, 8
FUSE_CNT_CELLS = 512          # only gates fuse eligibility, value irrelevant
N_SMS = 110


def dispatch(msq, msk, batch, q_slots, n_sms=N_SMS,
             num_q_heads=16, heads_per_kv=8, head_dim=256,
             gate=None):
    """Returns dict(engine, R, fold, S, nmb, main_ctas, grid_x, fuse)."""
    use_big = (msq > 32 and FOLD_BIG <= heads_per_kv
               and heads_per_kv % FOLD_BIG == 0
               and num_q_heads % FOLD_BIG == 0)
    force_2c = False
    if use_big and gate:
        # EXACT ship form: staircase, integer arithmetic, runtime features only.
        # tier 1: te <= 8 and ctas_big <= 0.6*n_sms   (5c <= 3n)
        # tier 2: te <= 4 and ctas_big <= 1.5*n_sms   (2c <= 3n)
        # te tightened 16->8: te>=12 has a mid-ctas KV-re-stream loss pocket
        # (dv_696c1b34d9e8: 1.03-1.11 at ctas 24-32); te<=8 is clean/monotone.
        nmb_big = max(1, (msq + R_BIG - 1) // R_BIG)
        ctas_big = batch * nmb_big * (num_q_heads // FOLD_BIG)
        te = (msk + BLOCK_N - 1) // BLOCK_N
        if ((te <= 8 and 5 * ctas_big <= 3 * n_sms)
                or (te <= 4 and 2 * ctas_big <= 3 * n_sms)):
            force_2c = True
    if use_big and not force_2c:
        cfg_bm, R = BLOCK_M_BIG, R_BIG
    else:
        cfg_bm, R = BLOCK_M, BLOCK_M
        if not (use_big and force_2c):
            for r_cand in (8, 16, 32):
                f = BLOCK_M // r_cand
                if (msq <= r_cand and f <= heads_per_kv
                        and heads_per_kv % f == 0
                        and num_q_heads % f == 0):
                    R = r_cand
                    break
        else:
            R = 8
    fold = cfg_bm // R
    hg = num_q_heads // fold
    nmb = max(1, (msq + R - 1) // R)
    main_ctas = batch * nmb * hg
    tiles_est = (msk + BLOCK_N - 1) // BLOCK_N
    rows_bound = batch * msq * num_q_heads
    part_bytes = rows_bound * (head_dim * 4 + 8)
    kv_bytes = batch * msk * (num_q_heads // heads_per_kv) * head_dim * 2 * 2
    rpc = MERGE_ROWS_PER_CTA
    while rpc > 1 and (rows_bound + rpc - 1) // rpc < n_sms:
        rpc //= 2
    merge_grid = max(1, (rows_bound + rpc - 1) // rpc)

    def merge_t(s_c):
        m_bytes = s_c * part_bytes
        return (MERGE_FIX_US + MERGE_LAT_US * s_c
                * min(1.0, n_sms / merge_grid)
                + m_bytes / (MERGE_L2_BYTES_PER_US
                             if m_bytes + kv_bytes <= MERGE_L2_FIT_BYTES
                             else MERGE_DRAM_BYTES_PER_US))

    S = 1
    use_2c = not use_big
    if use_big and not force_2c:
        best = None
        if tiles_est >= 2 and rows_bound > 0:
            for s_c in (1,) + S_CANDIDATES:
                if s_c > tiles_est or s_c * part_bytes > SCRATCH_CAP_BYTES:
                    continue
                tps = (tiles_est + s_c - 1) // s_c
                active = (tiles_est + tps - 1) // tps
                waves = (main_ctas * active + n_sms - 1) // n_sms
                t = waves * (CTA_C0_US + tps * TILE_US)
                if s_c > 1:
                    t += merge_t(s_c)
                if best is None or t < best:
                    best, S = t, s_c
        engine = "big"
    else:
        best_t2 = best_to = None
        S2 = So = 1
        cap2 = n_sms * CTAS_PER_SM_2C
        if tiles_est >= 1 and rows_bound > 0:
            for s_c in (1,) + S_CANDIDATES:
                if s_c > tiles_est or s_c * part_bytes > SCRATCH_CAP_BYTES:
                    continue
                tps = (tiles_est + s_c - 1) // s_c
                active = (tiles_est + tps - 1) // tps
                ctas = main_ctas * active
                t2 = ((ctas + cap2 - 1) // cap2
                      * (CTA_C0_2C_US + tps * TILE_US_2C))
                t_o = ((ctas + n_sms - 1) // n_sms
                       * (CTA_C0_US + tps * TILE_US))
                if s_c > 1:
                    t2 += (s_c * part_bytes / SCRATCH_WRITE_BYTES_PER_US
                           + merge_t(s_c))
                    t_o += merge_t(s_c)
                if (not CLAMP_TPS1_2C or tps > 1) and (
                        best_t2 is None or t2 < best_t2):
                    best_t2, S2 = t2, s_c
                if best_to is None or t_o < best_to:
                    best_to, So = t_o, s_c
        use_2c = best_t2 is not None and (
            force_2c or best_t2 < best_to)
        S = S2 if use_2c else So
        engine = "2c" if use_2c else "old"
    tail = min(TAIL_CTAS_MAX,
               max(1, (q_slots + TAIL_ROWS_PER_CTA - 1) // TAIL_ROWS_PER_CTA))
    grid_x = tail + main_ctas * S
    fuse = (S > 1 and engine == "old" and grid_x <= n_sms and nmb == 1
            and main_ctas <= FUSE_CNT_CELLS)
    return dict(engine=engine, R=R, fold=fold, S=S, nmb=nmb,
                main_ctas=main_ctas, grid_x=grid_x, fuse=fuse,
                use_2c=use_2c, tiles_est=tiles_est)


def main():
    gate = True   # staircase form is self-contained in dispatch()
    # ---- measured dev batteries (name, q_lens, kv_lens, slots) ----
    batteries = json.load(open("scratch/sim_shapes.json"))
    print("# gate: (te<=16 & 5c<=3n) | (te<=4 & 2c<=3n)")
    ndelta = ndec = 0
    for (name, q_lens, kv_lens, slots) in batteries:
        msq, msk = max(q_lens), max(kv_lens)
        batch = len(q_lens)
        slots = slots if slots else sum(q_lens)
        a = dispatch(msq, msk, batch, slots)
        b = dispatch(msq, msk, batch, slots, gate=gate)
        if a != b:
            ndelta += 1
            if msq <= 32:
                ndec += 1
            print(f"DELTA {name:22s} msq={msq:5d} msk={msk:5d} b={batch:2d} "
                  f"| v10 {a['engine']:3s} R{a['R']:<2d} S{a['S']:<2d} "
                  f"grid {a['grid_x']:6d} -> cand {b['engine']:3s} "
                  f"R{b['R']:<2d} S{b['S']:<2d} grid {b['grid_x']:6d}")
    print(f"# battery deltas: {ndelta} (decode-domain: {ndec})")
    # ---- wide runtime grid ----
    g_dec = g_pf = 0
    flips = []
    for batch in (1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 32):
        for msq in (33, 48, 64, 96, 128, 192, 256, 384, 512, 768,
                    1024, 1536, 2048, 3072, 4096, 4319):
            for msk in (msq, msq + 64, msq + 256, 128, 256, 512,
                        1024, 2048, 4352, 4578):
                if msk < msq or msk > 4578:
                    continue
                slots = batch * msq
                if slots > 8192:
                    continue
                a = dispatch(msq, msk, batch, slots)
                b = dispatch(msq, msk, batch, slots, gate=gate)
                if a != b:
                    if msq <= 32:
                        g_dec += 1
                    else:
                        g_pf += 1
                        flips.append((batch, msq, msk,
                                      a["engine"], a["S"],
                                      b["engine"], b["S"]))
    print(f"# grid deltas: prefill={g_pf} decode={g_dec} (decode MUST be 0)")
    for f in flips[:60]:
        print("  flip b%-2d msq%-5d msk%-5d %s S%-2d -> %s S%-2d" % f)
    if len(flips) > 60:
        print(f"  ... {len(flips) - 60} more")


if __name__ == "__main__":
    main()
