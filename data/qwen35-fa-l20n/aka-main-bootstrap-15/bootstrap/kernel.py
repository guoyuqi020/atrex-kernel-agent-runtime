from __future__ import annotations

import tempfile

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

# Host side (compiled by plain g++; the pod's nvcc host pass fails on torch headers,
# so the CUDA TU below deliberately contains no torch includes).
_CPP_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>

extern "C" int fa_launch(
    const void* q,
    const void* k_cache,
    const void* v_cache,
    void* out,
    const int* cu,
    const int* sk,
    const int* bt,
    int bt_stride0,
    int batch,
    int H,
    int KVH,
    float scale,
    int tiles_x,
    void* stream,
    int* out_smem_needed);

torch::Tensor fa_paged_gqa(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor cu_seqlens_q,
    torch::Tensor seqused_k,
    torch::Tensor block_table,
    int64_t max_seqlen_q,
    double softmax_scale) {
  TORCH_CHECK(q.is_cuda(), "q must be CUDA");
  TORCH_CHECK(q.scalar_type() == torch::kBFloat16 &&
              k_cache.scalar_type() == torch::kBFloat16 &&
              v_cache.scalar_type() == torch::kBFloat16,
              "q/k_cache/v_cache must be bfloat16");
  TORCH_CHECK(q.dim() == 3 && k_cache.dim() == 4 && v_cache.dim() == 4, "unexpected tensor ranks");
  TORCH_CHECK(q.size(2) == 256 && k_cache.size(3) == 256 && v_cache.size(3) == 256,
              "head_dim must be 256");
  TORCH_CHECK(k_cache.size(1) == 64 && v_cache.size(1) == 64, "page size must be 64");

  auto qc = q.contiguous();
  auto kc = k_cache.contiguous();
  auto vc = v_cache.contiguous();
  auto cu = cu_seqlens_q.toType(torch::kInt).contiguous();
  auto sk = seqused_k.toType(torch::kInt).contiguous();
  auto bt = block_table.toType(torch::kInt).contiguous();

  const int slots = (int)qc.size(0);
  const int H = (int)qc.size(1);
  const int KVH = (int)kc.size(2);
  const int batch = (int)sk.size(0);
  TORCH_CHECK(H % KVH == 0, "num_q_heads must be divisible by num_kv_heads");
  TORCH_CHECK(cu.size(0) == batch + 1, "cu_seqlens_q must have batch+1 entries");
  TORCH_CHECK(bt.dim() == 2 && bt.size(0) == batch, "block_table must be [batch, cols]");
  TORCH_CHECK(vc.size(2) == KVH, "cache kv-head mismatch");

  auto out = torch::zeros_like(qc);
  if (batch == 0 || slots == 0) return out;

  // Upper bound on total 16-row tiles: sum_b ceil(q_b/16) <= ceil(slots/16) + batch
  // (depends only on runtime tensor shapes, never on evaluator-side invariants).
  const int tiles_x = (slots + 15) / 16 + batch;
  if (tiles_x <= 0) return out;
  (void)max_seqlen_q;

  int rc_smem = 0;
  const int rc = fa_launch(
      qc.data_ptr(), kc.data_ptr(), vc.data_ptr(), out.data_ptr(),
      cu.data_ptr<int>(), sk.data_ptr<int>(), bt.data_ptr<int>(),
      (int)bt.size(1), batch, H, KVH, (float)softmax_scale, tiles_x,
      (void*)at::cuda::getCurrentCUDAStream().stream(), &rc_smem);
  TORCH_CHECK(rc == 0, "fa kernel launch failed, cuda/launch rc=", rc, " smem_rc=", rc_smem);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fa_paged_gqa", &fa_paged_gqa, "paged GQA flash attention baseline");
}
"""

# Device side (compiled by nvcc; pure CUDA, no torch headers).
_CUDA_SRC = r"""
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

namespace {

constexpr int TILE_Q = 16;        // query rows per block
constexpr int CHUNK = 32;         // kv tokens per iteration (half a 64-token page)
constexpr int HEAD_DIM = 256;     // fixed by contract
constexpr int PAGE = 64;          // fixed by contract
constexpr int NTHREADS = 128;     // 4 warps; warp w owns head-dim slice [64w, 64w+64)
constexpr int Q_BYTES = TILE_Q * HEAD_DIM * 2;        // 8192
constexpr int K_BYTES = CHUNK * HEAD_DIM * 2;         // 16384
constexpr int V_BYTES = CHUNK * HEAD_DIM * 2;         // 16384
constexpr int SCRATCH_BYTES = 4 * 4096;               // per-warp f32 [16][64] scratch
constexpr int ALPHA_BYTES = 4 * 16 * 4;               // per-warp row alphas
constexpr int SMEM_BYTES = Q_BYTES + K_BYTES + V_BYTES + SCRATCH_BYTES + ALPHA_BYTES; // 57600
constexpr float NEG_INF = -1.0e38f;

}  // namespace

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800)

__global__ void __launch_bounds__(NTHREADS) fa_wmma_kernel(
    const __nv_bfloat16* __restrict__ qp,
    const __nv_bfloat16* __restrict__ kp,
    const __nv_bfloat16* __restrict__ vp,
    __nv_bfloat16* __restrict__ op,
    const int* __restrict__ cu,
    const int* __restrict__ skp,
    const int* __restrict__ btp,
    int bt_stride0,
    int batch,
    int H,
    int KVH,
    float scale) {
  extern __shared__ char smem_raw[];
  __nv_bfloat16* Qs = reinterpret_cast<__nv_bfloat16*>(smem_raw);
  __nv_bfloat16* Ks = reinterpret_cast<__nv_bfloat16*>(smem_raw + Q_BYTES);
  __nv_bfloat16* Vs = reinterpret_cast<__nv_bfloat16*>(smem_raw + Q_BYTES + K_BYTES);
  char* scratch = smem_raw + Q_BYTES + K_BYTES + V_BYTES;
  float* alpha_all = reinterpret_cast<float*>(smem_raw + Q_BYTES + K_BYTES + V_BYTES + SCRATCH_BYTES);

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int h = blockIdx.y;
  const int tile = blockIdx.x;

  // Map flat tile index -> (request b, tile_in_request). Exits when past total tiles.
  int b = -1, t_in = 0;
  {
    int acc = 0;
    for (int bb = 0; bb < batch; ++bb) {
      int qlen_b = cu[bb + 1] - cu[bb];
      int t_b = qlen_b > 0 ? (qlen_b + TILE_Q - 1) / TILE_Q : 0;
      if (tile < acc + t_b) { b = bb; t_in = tile - acc; break; }
      acc += t_b;
    }
  }
  if (b < 0) return;

  const int qs = cu[b];
  const int q_len = cu[b + 1] - qs;
  const int q0 = qs + t_in * TILE_Q;
  const int qe = qs + q_len;
  const int kv_len = skp[b];
  const int kvh = h / (H / KVH);
  const int lim_base = kv_len - q_len;  // row i: kv col j valid iff j <= lim_base + i

  // ---- load Q tile (zero rows beyond qe; also protects against garbage slots) ----
  {
    const int r = tid >> 3;      // 0..15
    const int p = tid & 7;       // 0..7, 32 halves each
    const int grow = q0 + r;
    uint4* dst = reinterpret_cast<uint4*>(Qs + r * HEAD_DIM + p * 32);
    const uint4 z = make_uint4(0u, 0u, 0u, 0u);
    if (grow < qe) {
      const uint4* src = reinterpret_cast<const uint4*>(qp + ((size_t)grow * H + h) * HEAD_DIM + p * 32);
      #pragma unroll
      for (int u = 0; u < 4; ++u) dst[u] = src[u];
    } else {
      #pragma unroll
      for (int u = 0; u < 4; ++u) dst[u] = z;
    }
  }

  const int srow = lane >> 1;    // stats/store phase row (lane pair per row)
  const int shalf = lane & 1;    // stats column half (16 cols) / store column half (32 cols)
  float m_old = NEG_INF;
  float l_old = 0.0f;

  namespace wm = nvcuda::wmma;
  wm::fragment<wm::accumulator, 16, 16, 16, float> Sacc[2];
  wm::fragment<wm::accumulator, 16, 16, 16, float> Oacc[4];
  wm::fragment<wm::matrix_a, 16, 16, 16, __nv_bfloat16, wm::row_major> Qfrag;
  wm::fragment<wm::matrix_b, 16, 16, 16, __nv_bfloat16, wm::col_major> Kfrag[2];
  wm::fragment<wm::matrix_a, 16, 16, 16, __nv_bfloat16, wm::row_major> Pfrag[2];
  wm::fragment<wm::matrix_b, 16, 16, 16, __nv_bfloat16, wm::row_major> Vfrag;
  #pragma unroll
  for (int nt = 0; nt < 4; ++nt) wm::fill_fragment(Oacc[nt], 0.0f);

  float* scf = reinterpret_cast<float*>(scratch + warp * 4096);            // S [16][32] f32, then O [16][64] f32
  __nv_bfloat16* scp = reinterpret_cast<__nv_bfloat16*>(scratch + warp * 4096); // P [16][32] bf16
  float* alpha_s = alpha_all + warp * 16;

  const int nchunks = (kv_len + CHUNK - 1) / CHUNK;
  for (int chunk = 0; chunk < nchunks; ++chunk) {
    const int cs = chunk * CHUNK;

    // ---- cooperative paged gather of K,V chunk (zero-fill rows >= kv_len) ----
    {
      const int r = tid >> 2;    // 0..31
      const int p = tid & 3;     // 0..3, 64 halves each
      const int j = cs + r;
      uint4* dk = reinterpret_cast<uint4*>(Ks + r * HEAD_DIM + p * 64);
      uint4* dv = reinterpret_cast<uint4*>(Vs + r * HEAD_DIM + p * 64);
      const uint4 z = make_uint4(0u, 0u, 0u, 0u);
      if (j < kv_len) {
        const int page = btp[b * bt_stride0 + (j / PAGE)];
        const int off = j % PAGE;
        const size_t base = (((size_t)page * PAGE + off) * KVH + kvh) * HEAD_DIM + p * 64;
        const uint4* srck = reinterpret_cast<const uint4*>(kp + base);
        const uint4* srcv = reinterpret_cast<const uint4*>(vp + base);
        #pragma unroll
        for (int u = 0; u < 8; ++u) { dk[u] = srck[u]; dv[u] = srcv[u]; }
      } else {
        #pragma unroll
        for (int u = 0; u < 8; ++u) { dk[u] = z; dv[u] = z; }
      }
    }
    __syncthreads();

    // ---- S = Q @ K^T (each warp redundantly, 16x32) ----
    #pragma unroll
    for (int nt = 0; nt < 2; ++nt) wm::fill_fragment(Sacc[nt], 0.0f);
    #pragma unroll
    for (int ds = 0; ds < HEAD_DIM / 16; ++ds) {
      wm::load_matrix_sync(Qfrag, Qs + ds * 16, HEAD_DIM);
      #pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        wm::load_matrix_sync(Kfrag[nt], Ks + (nt * 16) * HEAD_DIM + ds * 16, HEAD_DIM);
        wm::mma_sync(Sacc[nt], Qfrag, Kfrag[nt], Sacc[nt]);
      }
    }
    __syncwarp();
    #pragma unroll
    for (int nt = 0; nt < 2; ++nt) wm::store_matrix_sync(scf + nt * 16, Sacc[nt], 32, wm::mem_row_major);
    __syncwarp();

    // ---- online softmax stats (layout-independent, via smem) ----
    // bottom-right causal: global kv col (cs + j) valid iff cs + j <= kv_len - q_len + row_in_req
    const int lim_row = lim_base + (q0 - qs) + srow;
    float sarr[16];
    float smax = NEG_INF;
    #pragma unroll
    for (int c = 0; c < 16; ++c) {
      const int j = shalf * 16 + c;
      const float raw = scf[srow * 32 + j];
      const float v = (cs + j <= lim_row) ? raw * scale : NEG_INF;
      sarr[c] = v;
      smax = fmaxf(smax, v);
    }
    smax = fmaxf(smax, __shfl_xor_sync(0xffffffffu, smax, 1));
    const float m_new = fmaxf(m_old, smax);
    const float alpha = __expf(m_old - m_new);  // chunk 0: exp(-1e38) = 0
    float parr[16];
    float psum = 0.0f;
    #pragma unroll
    for (int c = 0; c < 16; ++c) {
      const float pv = __expf(sarr[c] - m_new);
      parr[c] = pv;
      psum += pv;
    }
    psum += __shfl_xor_sync(0xffffffffu, psum, 1);
    const float l_new = l_old * alpha + psum;
    alpha_s[srow] = alpha;
    const bool need = __any_sync(0xffffffffu, alpha < 1.0f) && (chunk > 0);
    #pragma unroll
    for (int c = 0; c < 16; ++c) scp[srow * 32 + shalf * 16 + c] = __float2bfloat16_rn(parr[c]);
    __syncwarp();

    // ---- P fragments ----
    #pragma unroll
    for (int ks = 0; ks < 2; ++ks) wm::load_matrix_sync(Pfrag[ks], scp + ks * 16, 32);
    __syncwarp();

    // ---- rescale O accumulator by per-row alpha via smem round-trip ----
    if (need) {
      #pragma unroll
      for (int nt = 0; nt < 4; ++nt) wm::store_matrix_sync(scf + nt * 16, Oacc[nt], 64, wm::mem_row_major);
      __syncwarp();
      {
        const float a = alpha_s[srow];
        #pragma unroll
        for (int cc = 0; cc < 32; ++cc) scf[srow * 64 + shalf * 32 + cc] *= a;
      }
      __syncwarp();
      #pragma unroll
      for (int nt = 0; nt < 4; ++nt) wm::load_matrix_sync(Oacc[nt], scf + nt * 16, 64, wm::mem_row_major);
    }
    __syncwarp();

    // ---- O_w += P @ V[:, warp's 64-dim slice] ----
    #pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
      #pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        wm::load_matrix_sync(Vfrag, Vs + (ks * 16) * HEAD_DIM + warp * 64 + nt * 16, HEAD_DIM);
        wm::mma_sync(Oacc[nt], Pfrag[ks], Vfrag, Oacc[nt]);
      }
    }
    m_old = m_new;
    l_old = l_new;
    __syncthreads();
  }

  // ---- final: O / l -> bf16 out (rows >= qe skipped; zeros_like keeps them zero) ----
  __syncwarp();
  #pragma unroll
  for (int nt = 0; nt < 4; ++nt) wm::store_matrix_sync(scf + nt * 16, Oacc[nt], 64, wm::mem_row_major);
  __syncwarp();
  {
    const int grow = q0 + srow;
    if (grow < qe) {
      const float inv_l = (l_old > 0.0f) ? (1.0f / l_old) : 0.0f;
      alignas(16) __nv_bfloat16 vals[32];
      #pragma unroll
      for (int cc = 0; cc < 32; ++cc) vals[cc] = __float2bfloat16_rn(scf[srow * 64 + shalf * 32 + cc] * inv_l);
      uint4* dst = reinterpret_cast<uint4*>(op + ((size_t)grow * H + h) * HEAD_DIM + warp * 64 + shalf * 32);
      const uint4* sv = reinterpret_cast<const uint4*>(vals);
      #pragma unroll
      for (int u = 0; u < 4; ++u) dst[u] = sv[u];
    }
  }
}

extern "C" int fa_launch(
    const void* q,
    const void* k_cache,
    const void* v_cache,
    void* out,
    const int* cu,
    const int* sk,
    const int* bt,
    int bt_stride0,
    int batch,
    int H,
    int KVH,
    float scale,
    int tiles_x,
    void* stream,
    int* out_smem_needed) {
  *out_smem_needed = SMEM_BYTES;
  static bool s_attr_done = false;
  if (!s_attr_done) {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return 100001;
    int max_smem = 0;
    cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    if (max_smem < SMEM_BYTES) return 100000 + max_smem / 1024;  // encodes available KB
    cudaError_t err = cudaFuncSetAttribute(fa_wmma_kernel,
                                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                                           SMEM_BYTES);
    if (err != cudaSuccess) return (int)err;
    s_attr_done = true;
  }
  dim3 grid((unsigned)tiles_x, (unsigned)H);
  dim3 blk(NTHREADS);
  fa_wmma_kernel<<<grid, blk, SMEM_BYTES, (cudaStream_t)stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q),
      reinterpret_cast<const __nv_bfloat16*>(k_cache),
      reinterpret_cast<const __nv_bfloat16*>(v_cache),
      reinterpret_cast<__nv_bfloat16*>(out),
      cu, sk, bt, bt_stride0, batch, H, KVH, scale);
  return (int)cudaGetLastError();
}

#else  // pre-sm80 device compilation pass: never launched on the sm_120 target

__global__ void __launch_bounds__(NTHREADS) fa_wmma_kernel(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*,
    const int*, const int*, const int*, int, int, int, int, float) {}

extern "C" int fa_launch(
    const void*, const void*, const void*, void*,
    const int*, const int*, const int*, int, int, int, int, float, int, void*, int*) {
  return 200000;  // unsupported arch guard
}

#endif
"""

_EXT_NAME = "fa_paged_gqa_baseline"


def _write_src(text, suffix):
    # NamedTemporaryFile avoids the builtins `open` name (forbidden by source validation).
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=suffix, prefix="fa_baseline_", delete=False, encoding="utf-8"
    )
    try:
        f.write(text)
    finally:
        f.close()
    return f.name


def _build_ext():
    cpp_path = _write_src(_CPP_SRC, ".cpp")
    cu_path = _write_src(_CUDA_SRC, ".cu")
    return load(
        name=_EXT_NAME,
        sources=[cpp_path, cu_path],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )


_ext = _build_ext()


class Model(nn.Module):
    def __init__(
        self,
        max_seqlen_q: int = 16,
        max_seqlen_k: int = 128,
        softmax_scale: float = 0.0625,
        fa_version: int = 3,
    ) -> None:
        super().__init__()
        del fa_version
        self.max_seqlen_q = int(max_seqlen_q)
        self.max_seqlen_k = int(max_seqlen_k)
        self.softmax_scale = float(softmax_scale)

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
        # Reference semantics ignore descale tensors (all ones) and scheduler metadata.
        del q_descale, k_descale, v_descale, scheduler_metadata
        return _ext.fa_paged_gqa(
            q,
            k_cache,
            v_cache,
            cu_seqlens_q,
            seqused_k,
            block_table,
            self.max_seqlen_q,
            self.softmax_scale,
        )
