"""Tensor-core CUDA implementation of the paged-GQA flash-attention operator.

Operator (public contract in ``agent_problem.json`` / immutable ``reference.py``):
causal variable-length paged GQA attention with bottom-right causal alignment.

  q          : [q_slots, num_q_heads, head_dim]      bfloat16 (CUDA-graph slots)
  k_cache    : [num_pages, page_size, num_kv_heads, head_dim] bfloat16
  v_cache    : same layout as k_cache
  cu_seqlens_q : int32 [batch+1] inclusive prefix sums of query lengths
  seqused_k  : int32 [batch] kv lengths
  block_table: int32 [batch, bt_stride] page ids of each request's kv pages
  q/k/v_descale, scheduler_metadata: ignored (reference ignores them; descales
    are float32 ones per contract)
  output     : [q_slots, num_q_heads, head_dim] bfloat16, inactive slots zero.

Everything below the launch plumbing is a self-authored CUDA kernel compiled at
runtime with nvrtc through ``cuda.core`` (``Program.compile("cubin")``, which
drives the ``cuda.bindings`` nvrtc/driver APIs) and launched on torch's current
(default) stream through cuda.core's driver-API ``launch``.  The >48 KB dynamic
shared-memory opt-in for the main kernel is set once per process through
``cuda.bindings.driver.cuKernelSetAttribute``.  torch is used only for output
allocation, device/stream plumbing and reading tensor shapes/pointers -- no
torch compute operators are used.

cuda.core marshals Python ``int`` arguments as 64-bit integers and Python
``float`` arguments as C doubles, so the kernel's scalar parameters use a
``long long`` / ``double`` ABI and are narrowed to ``int`` / ``float`` inside
the kernel (verified empirically on the target machine).

Kernel structure (FlashAttention-2 style, tensor cores):
  * GQA folding: the num_q_heads/num_kv_heads = 8 query heads of one KV group
    are folded into the M dimension, so one CTA covers 8 query positions x 8
    heads = 64 effective rows and reads each K/V tile exactly once for all 8
    heads.
  * grid = (ceil(min(max_seqlen_q, slots)/8), num_kv_heads, batch * splits),
    128 threads/CTA = 4 warps; each warp owns 16 folded rows (m16) x the full
    head_dim (256).
  * Per KV tile of 64 rows (= one page): S = Q.K^T via
    mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 with ldmatrix operands
    from swizzled shared memory, bottom-right causal masking applied only on
    diagonal-band tiles, fp32 online softmax in the log2 domain (running max
    m2 and sum l), P converted to an f16 A-fragment in registers, then
    O = O*alpha + P.V via mma.sync...f32.f16.f16.f32 over ldmatrix.trans V
    fragments (V bf16->f16 widens exactly; the 10-bit f16 mantissa keeps the
    softmax-probability rounding 4x tighter than bf16, matching the fp32-P
    reference's error class).
  * K/V tiles are staged with cp.async 16B copies in a single-buffered
    pipeline: Q + K0 + V0 committed up front; per iteration wait for K, run
    QK+softmax, prefetch K(t+1), wait for V, run PV, prefetch V(t+1).  Rows
    past the sequence end are zero-filled at cp.async issue time (src_size=0)
    so a masked p=0 can never multiply garbage into NaN.
  * Split-KV flash decoding: when tiles * num_kv_heads * batch leaves the GPU
    underfilled (decode-like shapes), the host splits each request's KV pages
    across `splits` CTAs (grid.z = batch * splits; split sp covers pages
    [sp*pps, (sp+1)*pps) with pps = ceil(pages/splits)).  Each split writes
    fp32 partials -- unnormalized O, running max m, denominator l, laid out
    [splits, rows_bound, H, HD|1|1] in one buffer; m = -INF marks empty or
    fully-masked splits -- and a small fa_combine kernel (one warp per
    (row, head)) merges them with an online log2-domain merge and writes the
    bf16 output rows.  splits == 1 keeps the direct bf16 store path in
    fa_main: no partial buffers, no combine launch.  The split count comes
    from a calibrated makespan model (wave quantization x tiles-per-CTA plus
    partial traffic); splits are forced to 1 when the direct grid already
    spans two waves or the KV footprint far exceeds L2 (DRAM-bound batches
    gain nothing from splitting).
  * Output rows are written exactly once: torch.empty_like, then active rows
    by fa_main directly (splits=1) or by fa_combine (splits>1).  CUDA-graph
    padding rows (slot >= total active queries) are zeroed by a tiny
    zero_inactive kernel on the splits=1 path, or by fa_combine's padding
    branch on split paths (one fewer ~3 us driver launch per call).
  * Softmax math is fp32 (matching the reference's float32 accumulation
    semantics); the two GEMMs run on tensor cores with fp32 accumulators,
    identical numerics to a canonical FA2 bf16 implementation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from cuda.core import LaunchConfig, Program, ProgramOptions, Stream, launch
from cuda.bindings import driver as _cud

_HEAD_DIM = 256
_PAGE_ROWS = 64          # folded rows per CTA / kv rows per tile (contract-fixed)
_POS_PER_CTA = 8         # query positions per CTA (64 folded rows / GQA group 8)
_THREADS_MAIN = 128
_SMEM_BYTES = 98304      # 3 x 32 KB (Q, K, V tiles of 64x256 bf16)
_LOG2E = 1.4426950408889634
_KERNEL_ATTR_MAX_DYN_SMEM = 8  # CU_KERNEL_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES

_CUDA_SOURCE = r"""
/* Self-contained: no external CUDA headers. bfloat16 <-> fp32 bit tricks are
 * exact; packed f32 -> bf16x2 conversion uses the hardware cvt (round-to-
 * nearest-even), matching torch's .to(bfloat16) for finite values. */

#define HD 256            /* head_dim, fixed by the public contract */
#define NTILE 64          /* kv rows per tile (== page size, contract) */
#define MROWS 64          /* folded q rows per CTA (8 positions x 8 heads) */
#define TILEB 32768       /* bytes per swizzled tile (64 rows x 512 B) */

__device__ __forceinline__ float u32_to_f32(unsigned int u) {
    union { unsigned int u; float f; } x;
    x.u = u;
    return x.f;
}

#define NEG_INF u32_to_f32(0xff800000u)

__device__ __forceinline__ float ex2f(float x) {
    float y;
    asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

/* f32 pair -> packed bf16x2 (lo in bits 0..15, hi in bits 16..31), RNE. */
__device__ __forceinline__ unsigned int pack_bf16(float lo, float hi) {
    unsigned int r;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}

/* f32 pair -> packed f16x2, RNE with saturation (P is in [0,1]). */
__device__ __forceinline__ unsigned int pack_f16(float lo, float hi) {
    unsigned int r;
    asm("cvt.rn.satfinite.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}

/* packed bf16x2 -> packed f16x2. bf16 widens to f32 exactly (16-bit shift);
 * f32 -> f16 is exact for the bf16 8-bit mantissa whenever the exponent is
 * in f16 range, and satfinite clamps pathological cache magnitudes instead
 * of producing inf. */
__device__ __forceinline__ unsigned int bf16x2_to_f16x2(unsigned int r) {
    union { unsigned int u; float f; } lo, hi;
    lo.u = r << 16;
    hi.u = r & 0xffff0000u;
    unsigned int h;
    asm("cvt.rn.satfinite.f16x2.f32 %0, %1, %2;"
        : "=r"(h) : "f"(hi.f), "f"(lo.f));
    return h;
}

__device__ __forceinline__ unsigned int smem_u32(const void *p) {
    unsigned int a;
    asm("{ .reg .u64 u; cvta.to.shared.u64 u, %1; cvt.u32.u64 %0, u; }"
        : "=r"(a) : "l"(p));
    return a;
}

__device__ __forceinline__ void cp16(unsigned int dst, const void *src,
                                     int src_size) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
                 :: "r"(dst), "l"(src), "r"(src_size));
}
__device__ __forceinline__ void cp_commit() {
    asm volatile("cp.async.commit_group;");
}
__device__ __forceinline__ void cp_wait1() {
    asm volatile("cp.async.wait_group 1;");
}

__device__ __forceinline__ void ldmx4(unsigned int &r0, unsigned int &r1,
                                      unsigned int &r2, unsigned int &r3,
                                      unsigned int a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                 "{%0,%1,%2,%3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ void ldmx4t(unsigned int &r0, unsigned int &r1,
                                       unsigned int &r2, unsigned int &r3,
                                       unsigned int a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
                 "{%0,%1,%2,%3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}

/* D[16x8] += A[16x16] * B[16x8], bf16 in, fp32 accumulate. */
__device__ __forceinline__ void mma16816(float *d, const unsigned int *a,
                                         const unsigned int *b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
                   "r"(b[0]), "r"(b[1]));
}

/* Same shape with f16 operands (used for P.V: the 10-bit f16 mantissa keeps
 * softmax-probability rounding 4x tighter than bf16, matching the fp32-P
 * reference's error class). */
__device__ __forceinline__ void mma16816_f16(float *d, const unsigned int *a,
                                             const unsigned int *b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
                   "r"(b[0]), "r"(b[1]));
}

/* Swizzled byte offset of 16 B chunk `c` in row `r` of a 64x256-bf16 tile
 * (512 B rows = 32 chunks): the chunk's 8-chunk (128 B) group is kept, its
 * low 3 bits are XORed with the row.  This makes cp.async 16 B stores and
 * ldmatrix reads bank-conflict free. */
__device__ __forceinline__ unsigned int swz_off(int r, int c) {
    return (unsigned int)(r * 512 + (((c & ~7) | ((c ^ r) & 7)) << 4));
}

/* Issue one K or V tile (kv rows [n0, n0+64) of one page) into shared
 * memory.  Rows at/after c_end are zero-filled via cp.async src_size=0; the
 * global address is clamped to a valid page row so nothing is dereferenced
 * out of bounds.  Every one of the 128 threads issues 16 chunks. */
__device__ __forceinline__ void issue_kv_tile(
    unsigned char *dst, const unsigned short *__restrict__ cache,
    const int *__restrict__ btab, long long b, int bt_stride, int n0,
    int kvg, int kvh_stride, int c_end, int tid)
{
    const int page = btab[b * bt_stride + (n0 >> 6)];
    const int rows_here = min(NTILE, c_end - n0);
    const int c = tid & 31;                 /* 16 B chunk within the row */
    const int rsub = tid >> 5;              /* 0..3 */
    const unsigned int dbase = smem_u32(dst);
    const long long pagebase = ((long long)page << 6) * kvh_stride;
#pragma unroll
    for (int p = 0; p < 16; ++p) {
        const int row = (p << 2) | rsub;
        const int valid = row < rows_here;
        const int rowc = valid ? row : 0;
        const unsigned short *src =
            cache + pagebase + (long long)rowc * kvh_stride +
            (long long)kvg * HD + (c << 3);
        cp16(dbase + swz_off(row, c), src, valid ? 16 : 0);
    }
    cp_commit();
}

/* Scalar parameters use the long long / double ABI that cuda.core's launcher
 * produces for Python int / float arguments; they are narrowed at entry. */
extern "C" __global__ __launch_bounds__(128) void fa_main(
    const unsigned short *__restrict__ q,       /* bf16 [slots, H, HD] */
    const unsigned short *__restrict__ kcache,  /* bf16 [pages, PS, KVH, HD] */
    const unsigned short *__restrict__ vcache,  /* bf16 [pages, PS, KVH, HD] */
    unsigned short *__restrict__ out,           /* bf16 [slots, H, HD] */
    float *__restrict__ po,                     /* fp32 partials (splits>1) */
    float *__restrict__ pm,                     /* fp32 partial maxes */
    float *__restrict__ pl,                     /* fp32 partial sums */
    const int *__restrict__ cu_q,               /* [B+1] */
    const int *__restrict__ seqk,               /* [B] */
    const int *__restrict__ btab,               /* [B, bt_stride] */
    long long num_heads_,
    long long num_kv_heads_,
    long long bt_stride_,
    long long splits_,
    long long rows_bound_,
    double qscale_)                             /* softmax_scale * log2(e) */
{
    const int H = (int)num_heads_;
    const int KVH = (int)num_kv_heads_;
    const int G = H / KVH;                      /* heads per kv group (8) */
    const int bt_stride = (int)bt_stride_;
    const int kvh_stride = KVH * HD;            /* elements per cache row */
    const float qscl = (float)qscale_;
    const int splits = (int)splits_;            /* 1 = direct bf16 path */
    const int rows_bound = (int)rows_bound_;    /* partials row stride */

    const int t = blockIdx.x;                   /* folded-row tile */
    const int kvg = blockIdx.y;                 /* kv head / q-head group */
    const int z = blockIdx.z;                   /* request * splits + split */
    const int b = z / splits;
    const int sp = z - b * splits;

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    const int qs = cu_q[b];
    const int q_len = cu_q[b + 1] - qs;
    const int i0 = t * (MROWS / 8);             /* first query position */
    if (i0 >= q_len) {
        return;                                 /* empty tile (padding) */
    }

    const int kv_len = seqk[b];
    /* This split's KV range: whole pages, [sp_start, sp_end). */
    const int pages_total = (kv_len + NTILE - 1) >> 6;
    const int pps = (pages_total + splits - 1) / splits;
    const int sp_start = sp * pps * NTILE;
    const int sp_end = min(kv_len, sp_start + pps * NTILE);
    /* Bottom-right causal: position i sees kv cols <= kv_len - q_len + i.
     * Folded rows for positions beyond q_len-1 are clamped (their results
     * are never stored), so lim_min below is the tightest limit in the CTA
     * and lim_max the loosest stored/clamped one.  Columns at/after sp_end
     * belong to the next split (zero-filled here, masked below). */
    const int lim_min = kv_len - q_len + i0;
    const int lim_max = min(kv_len - 1, lim_min + 7);
    const int c_end = min(sp_end, lim_max + 1);
    const int c_unmasked = min(lim_min, sp_end - 1);
    const int n_full = (c_unmasked + 1 > sp_start)
                           ? ((c_unmasked + 1 - sp_start) >> 6) : 0;
    const int n_tiles =
        (c_end > sp_start) ? ((c_end - sp_start + NTILE - 1) >> 6) : 0;
    if (n_tiles == 0) {
        /* Empty or fully masked split range: mark this split's partial rows
         * m = -INF so fa_combine skips them without reading po/pl (which
         * stay uninitialized). */
        if (splits > 1) {
            const int rl = (warp << 4) + (lane >> 2);
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int r = rl + (half << 3);
                const int i = i0 + (r >> 3);
                if (i < q_len && (lane & 3) == 0) {
                    pm[((long long)sp * rows_bound + (qs + i)) * H +
                       kvg * G + (r & 7)] = NEG_INF;
                }
            }
        }
        return;
    }

    extern __shared__ unsigned char smem[];
    unsigned char *const sQ = smem;             /* [64][256] bf16 swizzled */
    unsigned char *const sK = smem + TILEB;
    unsigned char *const sV = smem + 2 * TILEB;

    /* ---- prologue: Q tile, then K0 and V0 (3 cp.async groups) ---- */
    {
        const int c = tid & 31;
        const int rsub = tid >> 5;
        const unsigned int qbase = smem_u32(sQ);
#pragma unroll
        for (int p = 0; p < 16; ++p) {
            const int row = (p << 2) | rsub;
            const int i = min(i0 + (row >> 3), q_len - 1);
            const int head = kvg * G + (row & 7);
            const unsigned short *src =
                q + (((long long)(qs + i) * H + head) << 8) + (c << 3);
            cp16(qbase + swz_off(row, c), src, 16);
        }
        cp_commit();
    }
    issue_kv_tile(sK, kcache, btab, b, bt_stride, sp_start, kvg, kvh_stride,
                  c_end, tid);
    issue_kv_tile(sV, vcache, btab, b, bt_stride, sp_start, kvg, kvh_stride,
                  c_end, tid);
    cp_wait1();                                 /* Q + K0 done, V0 in flight */
    __syncthreads();

    /* ---- per-lane fragment geometry ---- */
    const int r_lo = (warp << 4) + (lane >> 2); /* folded row owned (low) */
    const int c_col = (lane & 3) << 1;          /* col base within each n8 */
    const int i_lo = min(i0 + (r_lo >> 3), q_len - 1);
    const int i_hi = min(i0 + ((r_lo + 8) >> 3), q_len - 1);
    const int lim_lo = kv_len - q_len + i_lo;   /* causal limit, row r_lo */
    const int lim_hi = kv_len - q_len + i_hi;   /* causal limit, row r_lo+8 */
    const int lt = lane >> 3;                   /* ldmatrix sub-matrix id */
    const int lr = lane & 7;
    /* A(Q) ldmatrix row/chunk pieces: row fixed per lane, chunk = 2*ks + off */
    const int arow = (warp << 4) + ((lt & 1) << 3) + lr;
    const int achk = (lt >> 1);
    /* B(K) ldmatrix: row = jp*16 + brow0, chunk = 2*ks + bchk */
    const int brow0 = ((lt >> 1) << 3) + lr;
    const int bchk = (lt & 1);
    /* B(V) trans ldmatrix: row = ks4*16 + vrow0, chunk = 2*djp + vchk */
    const int vrow0 = ((lt & 1) << 3) + lr;
    const int vchk = (lt >> 1);

    const unsigned int qbase = smem_u32(sQ);
    const unsigned int kbase = smem_u32(sK);
    const unsigned int vbase = smem_u32(sV);

    float o[32][4];                             /* fp32 O accumulator */
#pragma unroll
    for (int j = 0; j < 32; ++j) {
        o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.0f;
    }
    float m_lo = NEG_INF, m_hi = NEG_INF;       /* running max (log2 domain) */
    float l_lo = 0.0f, l_hi = 0.0f;             /* running denominators */

    for (int tile = 0; tile < n_tiles; ++tile) {
        const int n0 = sp_start + (tile << 6);
        const bool masked = (tile >= n_full);

        if (tile > 0) {
            cp_wait1();                         /* K(tile) done, V(tile) flies */
            __syncthreads();
        }

        /* ---- S = Q . K^T  (m16 x n64 per warp, fp32) ---- */
        float s[8][4];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.0f;
        }
#pragma unroll
        for (int ks = 0; ks < 16; ++ks) {
            unsigned int a[4];
            ldmx4(a[0], a[1], a[2], a[3],
                  qbase + swz_off(arow, (ks << 1) + achk));
#pragma unroll
            for (int jp = 0; jp < 4; ++jp) {
                unsigned int bb[4];
                ldmx4(bb[0], bb[1], bb[2], bb[3],
                      kbase + swz_off((jp << 4) + brow0, (ks << 1) + bchk));
                unsigned int b0[2] = {bb[0], bb[1]};
                unsigned int b1[2] = {bb[2], bb[3]};
                mma16816(s[2 * jp], a, b0);
                mma16816(s[2 * jp + 1], a, b1);
            }
        }

        /* ---- scale + bottom-right causal mask (edge tiles only) ----
         * Beyond the causal limit, also mask columns at/after sp_end: those
         * belong to the next split and are zero-filled here, which would
         * otherwise leak weight exp(0 - m) into unmasked rows.  For
         * splits == 1, sp_end == kv_len makes the second test redundant
         * (lim_* <= kv_len - 1). */
        if (masked) {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int col = n0 + (j << 3) + c_col;
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    if (col + e > lim_lo || col + e >= sp_end)
                        s[j][e] = NEG_INF;
                    if (col + e > lim_hi || col + e >= sp_end)
                        s[j][e + 2] = NEG_INF;
                }
            }
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            s[j][0] *= qscl; s[j][1] *= qscl;
            s[j][2] *= qscl; s[j][3] *= qscl;
        }

        /* ---- online softmax: row max over the 4 lanes of each row ---- */
        float mx_lo = NEG_INF, mx_hi = NEG_INF;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            mx_lo = fmaxf(mx_lo, fmaxf(s[j][0], s[j][1]));
            mx_hi = fmaxf(mx_hi, fmaxf(s[j][2], s[j][3]));
        }
        mx_lo = fmaxf(mx_lo, __shfl_xor_sync(0xffffffffu, mx_lo, 1));
        mx_lo = fmaxf(mx_lo, __shfl_xor_sync(0xffffffffu, mx_lo, 2));
        mx_hi = fmaxf(mx_hi, __shfl_xor_sync(0xffffffffu, mx_hi, 1));
        mx_hi = fmaxf(mx_hi, __shfl_xor_sync(0xffffffffu, mx_hi, 2));
        /* Clamp the running max away from -INF: a row whose whole split range
         * is causally masked would otherwise hit -INF - -INF = NaN in the
         * alpha / P exponents.  With m = -1e30 those become exact zeros
         * (ex2(-inf) = 0), the partial stays neutral (l = 0, O = 0), and
         * fa_combine merges it harmlessly.  No-op for splits == 1: every row
         * sees column 0 there, so mx is always finite. */
        const float mn_lo = fmaxf(fmaxf(m_lo, mx_lo), -1.0e30f);
        const float mn_hi = fmaxf(fmaxf(m_hi, mx_hi), -1.0e30f);
        const float al_lo = ex2f(m_lo - mn_lo); /* ex2(-inf) = 0 on first tile */
        const float al_hi = ex2f(m_hi - mn_hi);
        m_lo = mn_lo;
        m_hi = mn_hi;

        /* ---- P = exp2(S - m), row sums, pack into a bf16 A-fragment ---- */
        float ps_lo = 0.0f, ps_hi = 0.0f;
        unsigned int pf[4][4];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float p0 = ex2f(s[j][0] - m_lo);
            const float p1 = ex2f(s[j][1] - m_lo);
            const float p2 = ex2f(s[j][2] - m_hi);
            const float p3 = ex2f(s[j][3] - m_hi);
            ps_lo += p0 + p1;
            ps_hi += p2 + p3;
            s[j][0] = p0; s[j][1] = p1; s[j][2] = p2; s[j][3] = p3;
        }
#pragma unroll
        for (int k4 = 0; k4 < 4; ++k4) {
            pf[k4][0] = pack_f16(s[2 * k4][0], s[2 * k4][1]);
            pf[k4][1] = pack_f16(s[2 * k4][2], s[2 * k4][3]);
            pf[k4][2] = pack_f16(s[2 * k4 + 1][0], s[2 * k4 + 1][1]);
            pf[k4][3] = pack_f16(s[2 * k4 + 1][2], s[2 * k4 + 1][3]);
        }
        ps_lo += __shfl_xor_sync(0xffffffffu, ps_lo, 1);
        ps_lo += __shfl_xor_sync(0xffffffffu, ps_lo, 2);
        ps_hi += __shfl_xor_sync(0xffffffffu, ps_hi, 1);
        ps_hi += __shfl_xor_sync(0xffffffffu, ps_hi, 2);
        l_lo = l_lo * al_lo + ps_lo;
        l_hi = l_hi * al_hi + ps_hi;

        /* ---- K(tile+1) prefetch (empty commit keeps group counts aligned) */
        __syncthreads();                        /* all warps done reading sK */
        if (n0 + NTILE < c_end) {
            issue_kv_tile(sK, kcache, btab, b, bt_stride, n0 + NTILE, kvg,
                          kvh_stride, c_end, tid);
        } else {
            cp_commit();
        }

        cp_wait1();                             /* V(tile) done */
        __syncthreads();

        /* ---- O *= alpha ---- */
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            o[j][0] *= al_lo; o[j][1] *= al_lo;
            o[j][2] *= al_hi; o[j][3] *= al_hi;
        }

        /* ---- O += P . V  (k64 x n256, 4 k-slices x 32 d-tiles) ---- */
#pragma unroll
        for (int k4 = 0; k4 < 4; ++k4) {
#pragma unroll
            for (int djp = 0; djp < 16; ++djp) {
                unsigned int bb[4];
                ldmx4t(bb[0], bb[1], bb[2], bb[3],
                       vbase + swz_off((k4 << 4) + vrow0,
                                       (djp << 1) + vchk));
                unsigned int b0[2] = {bf16x2_to_f16x2(bb[0]),
                                      bf16x2_to_f16x2(bb[1])};
                unsigned int b1[2] = {bf16x2_to_f16x2(bb[2]),
                                      bf16x2_to_f16x2(bb[3])};
                mma16816_f16(o[2 * djp], pf[k4], b0);
                mma16816_f16(o[2 * djp + 1], pf[k4], b1);
            }
        }

        __syncthreads();                        /* all warps done reading sV */
        if (n0 + NTILE < c_end) {
            issue_kv_tile(sV, vcache, btab, b, bt_stride, n0 + NTILE, kvg,
                          kvh_stride, c_end, tid);
        } else {
            cp_commit();
        }
    }

    /* ---- epilogue ---- */
    if (splits == 1) {
        /* Direct path: normalize and store active rows as bf16. */
        const float inv_lo = (l_lo > 0.0f) ? (1.0f / l_lo) : 0.0f;
        const float inv_hi = (l_hi > 0.0f) ? (1.0f / l_hi) : 0.0f;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int r = r_lo + (half << 3);
            const int i = i0 + (r >> 3);
            if (i >= q_len) {
                continue;                       /* padding row: never stored */
            }
            const int head = kvg * G + (r & 7);
            unsigned short *op =
                out + (((long long)(qs + i) * H + head) << 8);
            const float inv = half ? inv_hi : inv_lo;
#pragma unroll
            for (int j = 0; j < 32; ++j) {
                const unsigned int pk =
                    pack_bf16(o[j][2 * half] * inv, o[j][2 * half + 1] * inv);
                *reinterpret_cast<unsigned int *>(op + (j << 3) + c_col) = pk;
            }
        }
    } else {
        /* Split-KV path: store raw fp32 partials (O unnormalized, running
         * max m, denominator l) for fa_combine to merge.  po rows are 1 KB
         * and 8 B-aligned float2 stores land at (j*8 + c_col)*4 bytes. */
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int r = r_lo + (half << 3);
            const int i = i0 + (r >> 3);
            if (i >= q_len) {
                continue;
            }
            const int head = kvg * G + (r & 7);
            const long long rh =
                ((long long)sp * rows_bound + (qs + i)) * H + head;
            float *pop = po + rh * HD;
#pragma unroll
            for (int j = 0; j < 32; ++j) {
                float2 v2;
                v2.x = o[j][2 * half];
                v2.y = o[j][2 * half + 1];
                *reinterpret_cast<float2 *>(pop + (j << 3) + c_col) = v2;
            }
            if ((lane & 3) == 0) {
                pm[rh] = half ? m_hi : m_lo;
                pl[rh] = half ? l_hi : l_lo;
            }
        }
    }
}

/* Merge split-KV partials into the bf16 output.  grid = ceil(out_rows*H /
 * warps_per_block); the block size is a host-side grid-spread knob (see
 * _comb_warps), and the device derives warps_per_block from blockDim.x so the
 * host can retune it without a source change.  Each warp owns one (slot row,
 * head) pair and lane `l` owns head_dim d = l*8 .. l*8+7, running the same
 * sequential log2-domain online merge as attempt 1 (bitwise-identical output;
 * only the warp->SM distribution changes).  Spreading a small merge warp total
 * (decode: 16 warps) over more, smaller blocks occupies more SMs and hides the
 * partial-load latency that made fa_combine 51% of decode device time at 0.08%
 * compute SOL.  Empty splits (pm == -INF) are skipped before their
 * (uninitialized) po/pl are touched.  Rows >= active were already zeroed by
 * zero_inactive; a row whose every split is skipped (kv_len = 0) leaves
 * l_run = 0 -> inv = 0 -> exact zeros, matching the reference's empty-softmax
 * result. */
extern "C" __global__ void fa_combine(
    unsigned short *__restrict__ out,           /* bf16 [slots, H, HD] */
    const float *__restrict__ po,               /* fp32 [splits, rows_bound, H, HD] */
    const float *__restrict__ pm,               /* fp32 [splits, rows_bound, H] */
    const float *__restrict__ pl,               /* fp32 [splits, rows_bound, H] */
    const int *__restrict__ cu_q,               /* [B+1] */
    long long splits_,
    long long rows_bound_,
    long long num_heads_,
    long long batch_,
    long long out_rows_)                        /* == slots: full output bound */
{
    const int splits = (int)splits_;
    const int rows_bound = (int)rows_bound_;
    const int H = (int)num_heads_;
    const int lane = threadIdx.x & 31;
    /* Warps per block derived from blockDim so the host can retune the
     * block size (grid = ceil(out_rows*H / warps_per_block)) without a
     * device-source change. */
    const int g = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (g >= out_rows_ * H) {
        return;
    }
    const int row = g / H;
    const int h = g - row * H;
    if (row >= cu_q[batch_]) {
        /* Inactive row (CUDA-graph padding, or any slot >= active count):
         * zero it here so the host can skip the dedicated zero_inactive
         * launch whenever this kernel runs (splits > 1).  The warp owns one
         * (row, head) segment: HD bf16 = 32 lanes x 16 B, one uint4 each.
         * Rows >= rows_bound are always inactive (active <= rows_bound by
         * contract), so they land here and never touch the partials. */
        uint4 z;
        z.x = 0u; z.y = 0u; z.z = 0u; z.w = 0u;
        *reinterpret_cast<uint4 *>(out + (((long long)row * H + h) << 8)
                                   + (lane << 3)) = z;
        return;
    }
    const int d0 = lane << 3;
    const long long rh = (long long)row * H + h;
    /* Sequential online-softmax merge across splits (log2 domain), with
     * arithmetic IDENTICAL to the validated attempt-1 artifact: the adaptive
     * block size (_comb_warps) only changes which SM each warp runs on, never
     * the per-(row,head) computation, so the merged output stays bitwise
     * identical.  Skip pm == -INF splits (fa_main's n_tiles == 0 early-out)
     * before touching their uninitialized po/pl.  A row whose every split is
     * skipped leaves l_run = 0 -> inv = 0 -> exact zeros. */
    float m_run = NEG_INF, l_run = 0.0f;
    float acc[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        acc[e] = 0.0f;
    }
    for (int sp = 0; sp < splits; ++sp) {
        const long long p = ((long long)sp * rows_bound) * H + rh;
        const float m = pm[p];
        if (m == NEG_INF) {
            continue;                           /* skipped split: po/pl dead */
        }
        const float l = pl[p];
        const float4 a0 =
            *reinterpret_cast<const float4 *>(po + p * HD + d0);
        const float4 a1 =
            *reinterpret_cast<const float4 *>(po + p * HD + d0 + 4);
        const float m_new = fmaxf(m_run, m);
        const float so = ex2f(m_run - m_new);   /* ex2(-inf) = 0 first pass */
        const float sn = ex2f(m - m_new);
        l_run = l_run * so + l * sn;
        acc[0] = acc[0] * so + a0.x * sn;
        acc[1] = acc[1] * so + a0.y * sn;
        acc[2] = acc[2] * so + a0.z * sn;
        acc[3] = acc[3] * so + a0.w * sn;
        acc[4] = acc[4] * so + a1.x * sn;
        acc[5] = acc[5] * so + a1.y * sn;
        acc[6] = acc[6] * so + a1.z * sn;
        acc[7] = acc[7] * so + a1.w * sn;
        m_run = m_new;
    }
    const float inv = (l_run > 0.0f) ? (1.0f / l_run) : 0.0f;
    uint4 z;
    z.x = pack_bf16(acc[0] * inv, acc[1] * inv);
    z.y = pack_bf16(acc[2] * inv, acc[3] * inv);
    z.z = pack_bf16(acc[4] * inv, acc[5] * inv);
    z.w = pack_bf16(acc[6] * inv, acc[7] * inv);
    *reinterpret_cast<uint4 *>(out + (rh << 8) + d0) = z;
}

/* Zero the CUDA-graph padding rows (slot >= total active query count) of the
 * output buffer.  grid = ceil(slots/4), 256 threads: each CTA owns 4 slot
 * rows of H*HD bf16 and skips rows below the active count.  Launched only on
 * the splits == 1 path; when splits > 1, fa_combine zeroes padding rows
 * itself (saves one ~3 us driver launch per call). */
extern "C" __global__ void zero_inactive(
    unsigned short *__restrict__ out,
    const int *__restrict__ cu_q,
    long long slots_,
    long long batch_,
    long long hd_elems_)                        /* H * HD per slot row */
{
    const int active = cu_q[batch_];
    const int row0 = blockIdx.x << 2;
    const int rowe = min(row0 + 4, (int)slots_);
    const int chunks = (int)(hd_elems_ >> 3);   /* 16 B chunks per row */
    for (int r = row0; r < rowe; ++r) {
        if (r < active) continue;
        uint4 *p = reinterpret_cast<uint4 *>(out + (long long)r * hd_elems_);
        uint4 z;
        z.x = 0u; z.y = 0u; z.z = 0u; z.w = 0u;
        for (int c = threadIdx.x; c < chunks; c += blockDim.x) {
            p[c] = z;
        }
    }
}
"""


class _KernelAttr:
    """Shim exposing ``.value`` for cuda.bindings kernel-attribute enums
    (this bindings build has no CUkernel_attribute enum)."""

    def __init__(self, value: int) -> None:
        self.value = value


# Process-wide cache: (device, cc_major, cc_minor) -> (object_code, kernels).
# Keyed only on stable device/architecture invariants (never on input data or
# pointers). The ObjectCode owns the loaded library and must stay alive while
# the Kernels are used.
_MODULE_CACHE: dict[tuple[int, int, int], tuple[object, dict]] = {}

# device index -> SM count (for the adaptive split-KV heuristic).
_SM_COUNT_CACHE: dict[int, int] = {}


def _sm_count(dev: int) -> int:
    c = _SM_COUNT_CACHE.get(dev)
    if c is None:
        c = int(torch.cuda.get_device_properties(dev).multi_processor_count)
        _SM_COUNT_CACHE[dev] = c
    return c


def _comb_warps(warps_tot: int, sm: int) -> int:
    """fa_combine warps per block: the device derives warps-per-block from
    blockDim.x, so this is a pure host-side grid-spread knob built from
    runtime geometry only (total merge warps, SM count).  Returns the largest
    power-of-two in [1, 8] whose grid still covers ~1 block per SM: tiny warp
    totals (decode: 16 merge warps on 110 SMs) spread 1-warp blocks over many
    SMs to hide the merge latency, large totals (padded slots) keep 256-thread
    blocks to amortize block-scheduling overhead.  Module-level so dev latency
    sweeps can override it without touching the device source."""
    return min(8, 1 << (max(1, warps_tot // max(1, sm)).bit_length() - 1))


# device index -> default-stream raw handle, and (device, handle) -> cuda.core
# Stream wrapper.  Only the default stream is ever cached: torch owns it for
# the process lifetime, so a cached wrapper can never dangle.  User/capture
# streams fall through to a fresh (uncached) wrapper -- their handles are
# pointer values that a destroyed stream could recycle.
_DEFAULT_STREAM_HANDLE: dict[int, int] = {}
_STREAM_CACHE: dict[tuple[int, int], object] = {}


def _fast_stream(dev: int):
    """cuda.core Stream wrapper for the current stream, ~5us cheaper on the
    default stream (torch's current_stream() + Stream.from_handle per call)."""
    try:
        handle = int(torch._C._cuda_getCurrentRawStream(dev))
    except AttributeError:
        handle = int(torch.cuda.current_stream(dev).cuda_stream)
    dh = _DEFAULT_STREAM_HANDLE.get(dev)
    if dh is None:
        dh = int(torch.cuda.default_stream(dev).cuda_stream)
        _DEFAULT_STREAM_HANDLE[dev] = dh
    if handle != dh:
        # User or capture stream: wrap fresh, never cache.
        return Stream.from_handle(handle)
    key = (dev, handle)
    s = _STREAM_CACHE.get(key)
    if s is None:
        s = Stream.from_handle(handle)
        _STREAM_CACHE[key] = s
    return s


def _load_kernels():
    """Compile the embedded CUDA source with NVRTC via cuda.core (cached) and
    opt the main kernel into its >48 KB dynamic shared-memory launch."""
    major, minor = torch.cuda.get_device_capability()
    key = (torch.cuda.current_device(), int(major), int(minor))
    cached = _MODULE_CACHE.get(key)
    if cached is not None:
        return cached[1]

    # Make sure torch's primary context is initialized and current on this
    # thread before cuda.core loads the compiled object into it (the raw
    # driver calls below also require a current context).
    torch.cuda.init()
    _cud.cuInit(0)
    _err, dev = _cud.cuDeviceGet(key[0])
    _err, ctx = _cud.cuDevicePrimaryCtxRetain(dev)
    _cud.cuCtxSetCurrent(ctx)

    prog = Program(
        _CUDA_SOURCE,
        "c++",
        ProgramOptions(
            name="fa_paged_gqa_tc",
            arch=f"sm_{major}{minor}",
            std="c++17",
        ),
    )
    obj_code = prog.compile("cubin")
    kernels = {
        "fa_main": obj_code.get_kernel("fa_main"),
        "fa_combine": obj_code.get_kernel("fa_combine"),
        "zero_inactive": obj_code.get_kernel("zero_inactive"),
    }
    ret = _cud.cuKernelSetAttribute(
        _KernelAttr(_KERNEL_ATTR_MAX_DYN_SMEM),
        _SMEM_BYTES,
        kernels["fa_main"].handle,
        dev,
    )
    # cuda.bindings returns (CUresult,) tuples; accept either shape.
    err = ret[0] if isinstance(ret, (tuple, list)) else ret
    if not isinstance(err, int):
        err = err.value  # CUresult enum
    if int(err) != 0:
        raise RuntimeError(f"cuKernelSetAttribute(smem opt-in) failed: {err}")
    _MODULE_CACHE[key] = (obj_code, kernels)
    return kernels


class Model(nn.Module):
    # Every constructor parameter has a default so a bare Model() construction
    # (used by the check probe) succeeds. Defaults are the public shape-domain
    # maxima: max_seqlen_q=8192 (cuda_graph_query_slots bound; the launch grid
    # is additionally clamped by the physical q.shape[0], and every per-request
    # query length is <= the slot count, so coverage can never be short) and
    # max_seqlen_k=4578 (max logical context tokens per request; bounds the
    # split-KV split cap, not used for the per-request kv loop which reads
    # seqused_k). Explicit init_kwargs override both.
    def __init__(
        self,
        max_seqlen_q: int = 8192,
        max_seqlen_k: int = 4578,
        softmax_scale: float = 0.0625,
        fa_version: int = 3,
    ) -> None:
        super().__init__()
        del fa_version
        self.max_seqlen_q = int(max_seqlen_q)
        self.max_seqlen_k = int(max_seqlen_k)
        self.softmax_scale = float(softmax_scale)
        self._qscale = float(softmax_scale) * _LOG2E
        # (slots, batch) -> (splits, rows_bound, cached LaunchConfigs).
        # Derived purely from runtime geometry + fixed model constants
        # (max_seqlen_q/k, SM count, contract H/KVH/page_size), never from
        # input data. LaunchConfig objects are immutable descriptors, safe
        # to reuse across calls and CUDA-graph captures.
        self._heur_cache: dict = {}
        # Grow-only pool of fp32 partial buffers for the split-KV path.
        # Buffers are NEVER freed: a CUDA graph captured around forward()
        # bakes in the buffer pointer, so releasing an old (smaller) buffer
        # could leave a replayed graph writing into recycled memory.  Internal
        # scratch only -- rewritten by fa_main before fa_combine reads it on
        # every launch, single-stream ordered.
        self._partials_pool: list = []

    def forward(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        seqused_k: torch.Tensor,
        block_table: torch.Tensor,
        q_descale: torch.Tensor,
        k_descale: torch.Tensor,
        v_descale: torch.Tensor,
        scheduler_metadata: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # q/k/v_descale are float32 ones per contract and ignored by the
        # reference; scheduler_metadata is an opaque scheduling hint also
        # ignored by the reference. Both are dropped here as well.
        del q_descale, k_descale, v_descale, scheduler_metadata

        # Guarded: inputs are contiguous by contract; is_contiguous() is a
        # cheaper dispatch than an unconditional contiguous() no-op.
        if not q.is_contiguous():
            q = q.contiguous()
        if not k_cache.is_contiguous():
            k_cache = k_cache.contiguous()
        if not v_cache.is_contiguous():
            v_cache = v_cache.contiguous()
        if not cu_seqlens_q.is_contiguous():
            cu_seqlens_q = cu_seqlens_q.contiguous()
        if not seqused_k.is_contiguous():
            seqused_k = seqused_k.contiguous()
        if not block_table.is_contiguous():
            block_table = block_table.contiguous()

        slots = int(q.shape[0])
        num_heads = int(q.shape[1])
        head_dim = int(q.shape[2])
        page_size = int(k_cache.shape[1])
        num_kv_heads = int(k_cache.shape[2])
        batch = int(seqused_k.shape[0])
        bt_stride = int(block_table.shape[1])

        if head_dim != _HEAD_DIM or int(k_cache.shape[3]) != _HEAD_DIM:
            raise NotImplementedError(
                f"kernel is authored for head_dim={_HEAD_DIM}, got {head_dim}"
            )
        if q.dtype != torch.bfloat16 or k_cache.dtype != torch.bfloat16 \
                or v_cache.dtype != torch.bfloat16:
            raise NotImplementedError("kernel is authored for bfloat16 q/k/v")
        if cu_seqlens_q.dtype != torch.int32 or seqused_k.dtype != torch.int32 \
                or block_table.dtype != torch.int32:
            raise NotImplementedError("metadata tensors must be int32")
        # The tensor-core tiling (folded M=64 = 8 positions x 8 heads, N=64 =
        # one page) is authored for the contract-fixed geometry: 16 q-heads,
        # 2 kv-heads (GQA group 8), 64-row pages.
        if num_heads != 16 or num_kv_heads != 2 or page_size != _PAGE_ROWS:
            raise NotImplementedError(
                "kernel is authored for the contract geometry "
                f"(H=16, KVH=2, page_size=64), got H={num_heads}, "
                f"KVH={num_kv_heads}, page_size={page_size}"
            )
        if batch < 1:
            raise ValueError("degenerate batch dimension")

        out = torch.empty_like(q)
        kernels = _load_kernels()

        # Contract invariant: max_seqlen_q == max(query_lengths); additionally
        # clamp by the physical slot count so the grid can never overshoot.
        # Everything through the LaunchConfigs below is a pure function of
        # (slots, batch) plus fixed model/device constants (max_seqlen_q/k,
        # SM count, contract H/KVH/head_dim validated above) -- cached per
        # geometry (probe16 measured ~9.5us of Python body + ~1us of
        # LaunchConfig construction inside the ~20us forward host time).
        hkey = (slots, batch)
        hit = self._heur_cache.get(hkey)
        if hit is None:
            tiles = max(
                1,
                min(
                    (self.max_seqlen_q + _POS_PER_CTA - 1) // _POS_PER_CTA,
                    (slots + _POS_PER_CTA - 1) // _POS_PER_CTA,
                ),
            )

            # Adaptive split-KV, calibrated by a forced-splits latency sweep
            # over the shape classes (scratch/probe11.py).  Splitting only
            # pays when it shrinks the makespan; three regimes:
            #   * direct grid already spans >= 2 waves (prefill) -> splits=1;
            #   * KV footprint far exceeds the 96 MB L2 (large-batch decode):
            #     extra CTAs add no DRAM bandwidth, only wave quantization
            #     and partial traffic -> splits = 1;
            #   * otherwise pick splits minimizing a makespan estimate
            #       waves(base*s) * ceil(pages/s) * T_TILE + s*rows*CP_ROW
            #     (waves = ceil(base*s / SMs): 96 KB smem pins occupancy at
            #     one CTA/SM; T_TILE ~ 3 us; CP_ROW ~ fp32 partial buffer
            #     traffic at DRAM speed).  For grids far below one wave this
            #     reduces to filling toward ~SM CTAs, capped by a
            #     pages-per-split floor: tiny-pps CTAs are prologue-latency
            #     bound, and batch > 4 wants a higher floor (measured on
            #     graph-padded batches).
            base_ctas = tiles * num_kv_heads * batch
            sm = _sm_count(torch.cuda.current_device())
            pages_est = max(
                1, (self.max_seqlen_k + _PAGE_ROWS - 1) // _PAGE_ROWS
            )
            kv_bytes = batch * self.max_seqlen_k * (num_kv_heads * head_dim * 4)
            if base_ctas >= 2 * sm or kv_bytes >= 256_000_000:
                splits = 1
            elif base_ctas * 2 > sm:
                # One-to-two-wave grid (mid-size prefill): explicit argmin
                # over a small split set; strict improvement keeps ties at
                # fewer splits.
                rows_tot = min(slots, batch * self.max_seqlen_q) * num_heads
                splits = 1
                best = None
                for x in (1, 2, 3, 4, 6):
                    if x > pages_est:
                        break
                    c = -(-(base_ctas * x) // sm) * (-(-pages_est // x)) * 3.0
                    if x > 1:
                        c += x * rows_tot * 1.14e-3
                    if best is None or c < best - 1e-9:
                        best, splits = c, x
            else:
                # Sub-wave grid, batch-tiered fill (probe11/probe12 sweeps):
                #   batch <= 4: fill toward ~2 waves with a pages-per-split
                #     >= 3 floor (pps <= 2 CTAs are prologue-latency bound:
                #     b1 decode measured fastest at pps=3, s=24-32);
                #   batch 5..7: two-wave fill, no floor -- small batches are
                #     ragged-prone and straggler-dominated (one long
                #     request); under-splitting the straggler costs far more
                #     than the prologue/combine overhead of extra splits;
                #   batch >= 8: one-wave fill with pps >= 16 floor -- larger
                #     batches are usually balanced/graph-padded, where wave
                #     quantization and tiny-pps prologues dominate (measured
                #     on b8/b16/b24 classes).
                if batch <= 7:
                    s_fill = -(-(2 * sm) // base_ctas)
                    s_pps = max(1, pages_est // 3) if batch <= 4 else pages_est
                else:
                    s_fill = max(1, sm // base_ctas)
                    s_pps = max(1, pages_est // 16)
                splits = min(pages_est, s_fill, s_pps)

            rows_bound = slots
            if splits > 1:
                # Slot rows that can hold partials: below the active count,
                # which is <= slots and (contract invariant max_seqlen_q >=
                # every request's q_len) <= batch * max_seqlen_q.
                rows_bound = min(slots, batch * self.max_seqlen_q)

            # Immutable launch descriptors, reused across calls and graph
            # captures of the same geometry.  fa_combine's grid spans ALL
            # slots: its padding branch zeroes inactive rows (the partials
            # merge itself only touches rows < rows_bound).
            cfg_zero = LaunchConfig(
                grid=(max(1, (slots + 3) // 4), 1, 1), block=256
            )
            cfg_main = LaunchConfig(
                grid=(tiles, num_kv_heads, batch * splits),
                block=_THREADS_MAIN,
                shmem_size=_SMEM_BYTES,
            )
            warps_tot = max(1, slots * num_heads)
            comb_warps = _comb_warps(warps_tot, sm)
            cfg_comb = LaunchConfig(
                grid=(max(1, (warps_tot + comb_warps - 1) // comb_warps),
                      1, 1),
                block=comb_warps * 32,
            )
            hit = (splits, rows_bound, cfg_zero, cfg_main, cfg_comb)
            self._heur_cache[hkey] = hit
        splits, rows_bound, cfg_zero, cfg_main, cfg_comb = hit
        qscale = self._qscale

        po_ptr = pm_ptr = pl_ptr = 0
        if splits > 1:
            n_pr = splits * rows_bound * num_heads
            # One buffer: [po: n_pr*head_dim][pm: n_pr][pl: n_pr] fp32, taken
            # from the never-freed grow-only pool (see __init__).
            need = n_pr * (head_dim + 2)
            partials = None
            for buf in self._partials_pool:
                if buf.numel() >= need and buf.device == q.device:
                    partials = buf
                    break
            if partials is None:
                partials = torch.empty(
                    need, dtype=torch.float32, device=q.device
                )
                self._partials_pool.append(partials)
            base_p = partials.data_ptr()
            po_ptr = base_p
            pm_ptr = base_p + n_pr * head_dim * 4
            pl_ptr = pm_ptr + n_pr * 4

        # cuda.core Stream wrapper for the current stream (cached for the
        # default stream; from_handle is non-owning). All work stays on this
        # single stream.
        stream = _fast_stream(int(q.device.index or 0))

        if splits == 1:
            # fa_main writes active rows only; padding rows are zeroed by
            # the dedicated kernel.  On the split path fa_combine's padding
            # branch zeroes them itself, saving one driver launch (probe16:
            # the ~3 us cuLaunchKernel floor dominates any launch mechanism,
            # so launch COUNT is the lever, not the binding).
            launch(
                stream,
                cfg_zero,
                kernels["zero_inactive"],
                out.data_ptr(),
                cu_seqlens_q.data_ptr(),
                slots,
                batch,
                num_heads * head_dim,
            )

        launch(
            stream,
            cfg_main,
            kernels["fa_main"],
            q.data_ptr(),
            k_cache.data_ptr(),
            v_cache.data_ptr(),
            out.data_ptr(),
            po_ptr,
            pm_ptr,
            pl_ptr,
            cu_seqlens_q.data_ptr(),
            seqused_k.data_ptr(),
            block_table.data_ptr(),
            num_heads,
            num_kv_heads,
            bt_stride,
            splits,
            rows_bound,
            qscale,
        )

        if splits > 1:
            launch(
                stream,
                cfg_comb,
                kernels["fa_combine"],
                out.data_ptr(),
                po_ptr,
                pm_ptr,
                pl_ptr,
                cu_seqlens_q.data_ptr(),
                splits,
                rows_bound,
                num_heads,
                batch,
                slots,
            )
        return out
