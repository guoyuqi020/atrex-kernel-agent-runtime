"""Self-contained CUDA implementation of block-scaled FP8 fused MoE.

The GPU computation is implemented directly in CUDA C++ (embedded below),
compiled in-process with NVRTC and launched through the CUDA driver API via
``cuda.bindings``.  ``torch`` is used only for plumbing: tensor allocation,
dtype/layout inspection, device/context bootstrap, and stream lookup.  No
torch compute operator participates in the result.

Pipeline per ``forward`` (all on the current/default stream):
  1. expert histogram over flattened top-k ids
  2. single-block scan -> per-expert row offsets + per-expert M-tile descriptors
  3. scatter of expanded rows into per-expert sorted order
  4. per-128-group FP8 (e4m3) quantization of the expanded hidden states
  5. tiled block-scaled GEMM1: hidden_q @ w1[e]^T  (fp32 accumulation)
  6. SiLU(gate)*up with bf16 rounding + per-128-group FP8 re-quantization
  7. tiled block-scaled GEMM2: inter_q @ w2[e]^T    (fp32 accumulation)
  8. top-k weighted reduction back to [token_count, hidden_size]
"""

from __future__ import annotations

import struct

import torch
import torch.nn as nn

_FP8_MAX = 448.0
_GROUP = 128

_CUDA_SOURCE = r"""
#define TM 32
#define TN 32
#define TK 128
#define FP8_MAX 448.0f
#define SCALE_EPS 1e-12f

struct Params {
  const unsigned char* a_q;
  const float* a_scale;
  const unsigned char* w;
  const float* w_scale;
  int* sorted_rows;
  int* desc;
  int* total_tiles;
  int* offsets;
  const int* topk_ids;
  const float* topk_weights;
  const unsigned short* x_hidden;
  unsigned short* y16;
  unsigned short* out;
  unsigned char* q_out;
  float* scale_out;
  int* hist;
  int* fill;
  long long w_stride;
  int R;
  int N;
  int K;
  int n_groups_k;
  int w_scale_stride;
  int tile_m;
  int experts;
  int top_k;
};

static __device__ __forceinline__ float bf16_to_f32(unsigned short h) {
  return __uint_as_float(((unsigned int)h) << 16);
}

static __device__ __forceinline__ unsigned short f32_to_bf16_rn(float x) {
  unsigned int b = __float_as_uint(x);
  if ((b & 0x7F800000u) == 0x7F800000u && (b & 0x007FFFFFu) != 0u) {
    return (unsigned short)((b >> 16) | 0x40u);  // NaN
  }
  unsigned int rounding_bias = ((b >> 16) & 1u) + 0x7FFFu;
  return (unsigned short)((b + rounding_bias) >> 16);
}

static __device__ __forceinline__ float e4m3_to_f32(unsigned char v) {
  unsigned int sign = ((unsigned int)(v & 0x80u)) << 24;
  unsigned int e = (v >> 3) & 0xFu;
  unsigned int m = v & 0x7u;
  float mag;
  if (e == 0xFu && m == 7u) {
    mag = __uint_as_float(0x7FC00000u);  // NaN (never expected in weights)
  } else if (e == 0u) {
    mag = (float)m * (1.0f / 512.0f);  // subnormal: m * 2^-9
  } else {
    mag = ldexpf((float)(8u + m), (int)e - 10);
  }
  return __int_as_float(sign | __float_as_uint(mag));
}

// float32 -> e4m3fn, round-to-nearest-even.  Callers clamp |x| <= 448 first.
static __device__ __forceinline__ unsigned char f32_to_e4m3_rn(float x) {
  unsigned int b = __float_as_uint(x);
  unsigned int sign = (b >> 24) & 0x80u;
  b &= 0x7FFFFFFFu;
  if (b >= 0x7F800000u) {
    return (unsigned char)(sign | 0x7Fu);  // inf/NaN -> NaN
  }
  float ax = __uint_as_float(b);
  if (ax > FP8_MAX) {
    return (unsigned char)(sign | 0x7Eu);  // saturate to 448
  }
  int e = (int)((b >> 23) & 0xFFu) - 127;
  if (e >= -6) {
    unsigned int mant = b & 0x007FFFFFu;
    unsigned int m3 = mant >> 20;
    unsigned int round_bit = (mant >> 19) & 1u;
    unsigned int sticky = (mant & 0x0007FFFFu) ? 1u : 0u;
    if (round_bit && (sticky || (m3 & 1u))) {
      m3++;
    }
    unsigned int ex = (unsigned int)(e + 7);
    unsigned int val = (m3 == 8u) ? ((ex + 1u) << 3) : ((ex << 3) | m3);
    if (val > 0x7Eu) {
      val = 0x7Eu;  // rounding at e=8 can reach 512; clamp to 448
    }
    return (unsigned char)(sign | val);
  }
  // Subnormal target: value = m * 2^-9 (m in 0..7), carry m==8 -> 0x08.
  unsigned int m = (unsigned int)__float2uint_rn(ax * 512.0f);
  if (m > 8u) {
    m = 8u;
  }
  return (unsigned char)(sign | (m & 0xFFu));
}

extern "C" {

// 1. Histogram of expert assignments over flattened [T, top_k] ids.
__global__ void k_hist(const Params* P) {
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= P->R) return;
  int e = P->topk_ids[r];
  if (e >= 0 && e < P->experts) {
    atomicAdd(&P->hist[e], 1);
  }
}

// 2. Single-block scan: row offsets, M-tile descriptor list, total tile count.
__global__ void k_scan_desc(const Params* P) {
  __shared__ int s_off[257];
  __shared__ int s_tile_off[257];
  int tid = threadIdx.x;
  if (tid == 0) {
    int run = 0;
    int trun = 0;
    for (int e = 0; e < P->experts; e++) {
      int c = P->hist[e];
      s_off[e] = run;
      run += c;
      int tc = (c + P->tile_m - 1) / P->tile_m;
      s_tile_off[e] = trun;
      trun += tc;
    }
    s_off[P->experts] = run;
    s_tile_off[P->experts] = trun;
    *P->total_tiles = trun;
  }
  __syncthreads();
  for (int i = tid; i <= P->experts; i += blockDim.x) {
    P->offsets[i] = s_off[i];
  }
  if (tid < P->experts) {
    int e = tid;
    int c = P->hist[e];
    int tc = (c + P->tile_m - 1) / P->tile_m;
    for (int j = 0; j < tc; j++) {
      int slot = s_tile_off[e] + j;
      P->desc[2 * slot] = e;
      P->desc[2 * slot + 1] = s_off[e] + j * P->tile_m;
    }
  }
}

// 3. Scatter expanded rows into per-expert sorted order.
__global__ void k_scatter(const Params* P) {
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= P->R) return;
  int e = P->topk_ids[r];
  if (e < 0 || e >= P->experts) return;
  int pos = P->offsets[e] + atomicAdd(&P->fill[e], 1);
  P->sorted_rows[pos] = r;
}

// 4. Per-128-group FP8 quantization of the expanded hidden states.
__global__ void k_quant_act(const Params* P) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = P->R * P->n_groups_k;
  if (idx >= total) return;
  int g = idx % P->n_groups_k;
  int r = idx / P->n_groups_k;
  int t = r / P->top_k;
  const unsigned short* src =
      P->x_hidden + (long long)t * P->K + g * TK;
  float amax = 0.0f;
#pragma unroll 8
  for (int i = 0; i < TK; i++) {
    float v = bf16_to_f32(src[i]);
    amax = fmaxf(amax, fabsf(v));
  }
  float scale = fmaxf(amax, SCALE_EPS) * (1.0f / FP8_MAX);
  P->scale_out[(long long)r * P->n_groups_k + g] = scale;
  unsigned char* dst = P->q_out + (long long)r * P->K + g * TK;
#pragma unroll 8
  for (int i = 0; i < TK; i++) {
    float v = bf16_to_f32(src[i]) / scale;
    v = fminf(fmaxf(v, -FP8_MAX), FP8_MAX);
    dst[i] = f32_to_e4m3_rn(v);
  }
}

// 5/7. Tiled block-scaled FP8 GEMM with fp32 accumulation.
// Rows come from the per-expert sorted order (desc[tile]); weights belong to
// the tile's expert.  Activation scales are applied when loading the A tile
// (one K-group per chunk), weight scales when accumulating each chunk.
__global__ void k_gemm(const Params* P) {
  int tile = blockIdx.x;
  if (tile >= *P->total_tiles) return;
  int expert = P->desc[2 * tile];
  int row_start = P->desc[2 * tile + 1];
  int tile_rows = min(TM, P->offsets[expert + 1] - row_start);
  int n0 = blockIdx.y * TN;
  int tid = threadIdx.y * blockDim.x + threadIdx.x;

  const unsigned char* W = P->w + P->w_stride * expert;
  const float* WS = P->w_scale + (long long)P->w_scale_stride * expert;

  __shared__ float As[TM][TK + 1];
  __shared__ float Bs[TN][TK + 1];

  int m_base = (tid >> 4) << 2;
  int n_base = (tid & 15) << 1;
  float acc[4][2];
#pragma unroll
  for (int i = 0; i < 4; i++) {
    acc[i][0] = 0.0f;
    acc[i][1] = 0.0f;
  }

  int k_chunks = P->K / TK;
  for (int kc = 0; kc < k_chunks; kc++) {
    int k0 = kc * TK;
    {
      int m = tid >> 2;
      int kk0 = (tid & 3) << 5;
      if (m < tile_rows) {
        int row = P->sorted_rows[row_start + m];
        const unsigned char* a =
            P->a_q + (long long)row * P->K + k0 + kk0;
        float sa = P->a_scale[(long long)row * P->n_groups_k + kc];
#pragma unroll
        for (int j = 0; j < 32; j++) {
          As[m][kk0 + j] = e4m3_to_f32(a[j]) * sa;
        }
      } else {
#pragma unroll
        for (int j = 0; j < 32; j++) {
          As[m][kk0 + j] = 0.0f;
        }
      }
    }
    {
      int n = tid >> 2;
      int kk0 = (tid & 3) << 5;
      const unsigned char* wrow =
          W + (long long)(n0 + n) * P->K + k0 + kk0;
#pragma unroll
      for (int j = 0; j < 32; j++) {
        Bs[n][kk0 + j] = e4m3_to_f32(wrow[j]);
      }
    }
    __syncthreads();
    float d[4][2];
#pragma unroll
    for (int i = 0; i < 4; i++) {
      d[i][0] = 0.0f;
      d[i][1] = 0.0f;
    }
#pragma unroll 4
    for (int kk = 0; kk < TK; kk++) {
      float b0 = Bs[n_base][kk];
      float b1 = Bs[n_base + 1][kk];
#pragma unroll
      for (int i = 0; i < 4; i++) {
        float a = As[m_base + i][kk];
        d[i][0] += a * b0;
        d[i][1] += a * b1;
      }
    }
#pragma unroll
    for (int j = 0; j < 2; j++) {
      int ncol = n0 + n_base + j;
      float sw = WS[(ncol >> 7) * P->n_groups_k + kc];
#pragma unroll
      for (int i = 0; i < 4; i++) {
        acc[i][j] += sw * d[i][j];
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < 4; i++) {
    int m = m_base + i;
    if (m < tile_rows) {
      int row = P->sorted_rows[row_start + m];
      unsigned short* orow =
          P->y16 + (long long)row * P->N + n0 + n_base;
      orow[0] = f32_to_bf16_rn(acc[i][0]);
      orow[1] = f32_to_bf16_rn(acc[i][1]);
    }
  }
}

// 6. SiLU(gate)*up in bf16 rounding semantics + per-128-group quantization.
__global__ void k_silu_quant(const Params* P) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int groups = P->K / TK;
  int total = P->R * groups;
  if (idx >= total) return;
  int g = idx % groups;
  int r = idx / groups;
  const unsigned short* gate = P->y16 + (long long)r * P->N + g * TK;
  const unsigned short* up = gate + P->K;
  float amax = 0.0f;
#pragma unroll 8
  for (int i = 0; i < TK; i++) {
    float gv = bf16_to_f32(gate[i]);
    float uv = bf16_to_f32(up[i]);
    float sil = gv * (1.0f / (1.0f + expf(-gv)));
    float sil_b = bf16_to_f32(f32_to_bf16_rn(sil));
    float pb = bf16_to_f32(f32_to_bf16_rn(sil_b * uv));
    amax = fmaxf(amax, fabsf(pb));
  }
  float scale = fmaxf(amax, SCALE_EPS) * (1.0f / FP8_MAX);
  P->scale_out[(long long)r * groups + g] = scale;
  unsigned char* dst = P->q_out + (long long)r * P->K + g * TK;
#pragma unroll 8
  for (int i = 0; i < TK; i++) {
    float gv = bf16_to_f32(gate[i]);
    float uv = bf16_to_f32(up[i]);
    float sil = gv * (1.0f / (1.0f + expf(-gv)));
    float sil_b = bf16_to_f32(f32_to_bf16_rn(sil));
    float pb = bf16_to_f32(f32_to_bf16_rn(sil_b * uv));
    float v = pb / scale;
    v = fminf(fmaxf(v, -FP8_MAX), FP8_MAX);
    dst[i] = f32_to_e4m3_rn(v);
  }
}

// 8. Weighted top-k reduction back to [T, hidden].
__global__ void k_reduce(const Params* P) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int T = P->R / P->top_k;
  if (idx >= T * P->N) return;
  int t = idx / P->N;
  int h = idx % P->N;
  float acc = 0.0f;
  for (int k = 0; k < P->top_k; k++) {
    float w = P->topk_weights[t * P->top_k + k];
    float v = bf16_to_f32(P->y16[(long long)(t * P->top_k + k) * P->N + h]);
    acc += w * v;
  }
  P->out[idx] = f32_to_bf16_rn(acc);
}

}  // extern "C"
"""

_KERNEL_NAMES = (
    "k_hist",
    "k_scan_desc",
    "k_scatter",
    "k_quant_act",
    "k_gemm",
    "k_silu_quant",
    "k_reduce",
)

# struct layout must match ``struct Params`` above (little-endian, packed).
_PARAMS_FORMAT = "<17Qq8i"
_PARAMS_SIZE = struct.calcsize(_PARAMS_FORMAT)
_PTR_FIELDS = (
    "a_q", "a_scale", "w", "w_scale", "sorted_rows", "desc", "total_tiles",
    "offsets", "topk_ids", "topk_weights", "x_hidden", "y16", "out",
    "q_out", "scale_out", "hist", "fill",
)
_INT_FIELDS = ("w_stride", "R", "N", "K", "n_groups_k", "w_scale_stride",
               "tile_m", "experts", "top_k")


def _pack_params(**fields) -> bytes:
    values = []
    for name in _PTR_FIELDS:
        values.append(int(fields.get(name, 0)))
    for name in _INT_FIELDS:
        values.append(int(fields.get(name, 0)))
    return struct.pack(_PARAMS_FORMAT, *values)


class _CudaRuntime:
    """Lazy, per-device NVRTC/driver runtime. Keyed by device index only."""

    def __init__(self, device_index: int) -> None:
        from cuda.bindings import nvrtc as nvrtc_mod
        from cuda.bindings import driver as cu_mod

        self.nvrtc = nvrtc_mod
        self.cu = cu_mod

        def drv(result):
            err = result[0]
            if err != cu_mod.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"CUDA driver error: {err!r}")
            return result[1:]

        def nv(result):
            err = result[0]
            if err != nvrtc_mod.nvrtcResult.NVRTC_SUCCESS:
                raise RuntimeError(f"NVRTC error: {err!r}")
            return result[1:]

        self._drv = drv
        self._nv = nv

        drv(cu_mod.cuInit(0))

        # Bootstrap the CUDA context for this device (a torch allocation on
        # the device initializes the primary context), then make sure the
        # driver API sees a current context.
        torch.cuda.init()
        torch.empty(1, device=torch.device("cuda", device_index))

        err, ctx = cu_mod.cuCtxGetCurrent()
        if err != cu_mod.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"CUDA driver error: {err!r}")
        if ctx is None:
            err, device = cu_mod.cuDeviceGet(device_index)
            if err != cu_mod.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"CUDA driver error: {err!r}")
            err, ctx = cu_mod.cuDevicePrimaryCtxRetain(device)
            if err != cu_mod.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"CUDA driver error: {err!r}")
            drv(cu_mod.cuCtxSetCurrent(ctx))

        err, device = cu_mod.cuDeviceGet(device_index)
        if err != cu_mod.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"CUDA driver error: {err!r}")
        attr = cu_mod.CUdevice_attribute
        _, major = cu_mod.cuDeviceGetAttribute(
            attr.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device)
        _, minor = cu_mod.cuDeviceGetAttribute(
            attr.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device)
        arch = int(major) * 10 + int(minor)

        ptx = self._compile(arch)
        module, = drv(cu_mod.cuModuleLoadData(ptx))
        self.module = module
        self.funcs = {}
        for name in _KERNEL_NAMES:
            func, = drv(cu_mod.cuModuleGetFunction(module, name.encode()))
            self.funcs[name] = func

        self.params_ptr, = drv(cu_mod.cuMemAlloc(_PARAMS_SIZE))
        # Two-level marshalling for cuLaunchKernel.  kernelParams is a
        # ``void**``: the host address of an array whose i-th entry points
        # at the storage holding the value of kernel argument i.  Every
        # kernel here takes one argument (the device address of the params
        # struct), so we need:
        #   level-1 array  A[0] = address of the level-2 cell H
        #   level-2 cell   H    = device address of the params struct
        # cuda.bindings accepts a plain int for kernelParams (the void**
        # host pointer), which is the unambiguous marshalling form.
        cell = torch.zeros(1, dtype=torch.int64)
        cell.fill_(int(self.params_ptr))
        arr = torch.zeros(1, dtype=torch.int64)
        arr.fill_(cell.data_ptr())
        self._param_cell = cell
        self._param_array = arr
        self._kernel_params_arg = int(arr.data_ptr())
        self._stream_variant = None

    def _compile(self, arch: int) -> bytearray:
        nvrtc_mod = self.nvrtc
        nv = self._nv
        candidates = [arch, 120, 90, 80]
        seen: set[int] = set()
        last_log = ""
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            program, = nv(nvrtc_mod.nvrtcCreateProgram(
                _CUDA_SOURCE.encode(), b"fused_moe.cu", 0, [], []))
            options = [f"--gpu-architecture=compute_{candidate}".encode()]
            err, = nvrtc_mod.nvrtcCompileProgram(program, len(options), options)
            if err != nvrtc_mod.nvrtcResult.NVRTC_SUCCESS:
                try:
                    log_size, = nv(nvrtc_mod.nvrtcGetProgramLogSize(program))
                    log_buf = bytearray(log_size)
                    nv(nvrtc_mod.nvrtcGetProgramLog(program, log_buf))
                    last_log = log_buf.decode(errors="replace")
                except Exception:
                    pass
                continue
            ptx_size, = nv(nvrtc_mod.nvrtcGetPTXSize(program))
            ptx = bytearray(ptx_size)
            nv(nvrtc_mod.nvrtcGetPTX(program, ptx))
            nvrtc_mod.nvrtcDestroyProgram(program)
            return ptx
        raise RuntimeError(
            "NVRTC compilation failed for all candidate architectures; "
            f"last log: {last_log[:4000]}")

    def _h2d(self, data: bytes) -> None:
        # Host-buffer marshalling for cuMemcpyHtoD using only the declared
        # dependency set: a stdlib bytearray first, then a torch CPU view
        # over the same bytes.  No third-party buffer library is needed.
        cu = self.cu
        buf = bytearray(data)
        try:
            err, = cu.cuMemcpyHtoD(self.params_ptr, buf, len(buf))
        except TypeError:
            host = torch.frombuffer(buf, dtype=torch.uint8)
            err, = cu.cuMemcpyHtoD(self.params_ptr, host, len(buf))
        if err != cu.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuMemcpyHtoD failed: {err!r}")

    def launch(self, name: str, grid: tuple, block: tuple,
               stream: int, params: bytes) -> None:
        self._h2d(params)
        cu = self.cu
        func = self.funcs[name]
        gx, gy = grid
        bx, by = block
        stream_int = int(stream)

        def try_launch(make_s):
            err, = cu.cuLaunchKernel(
                func, gx, gy, 1, bx, by, 1, 0,
                make_s(stream_int), self._kernel_params_arg, 0)
            if err != cu.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"cuLaunchKernel({name}) failed: {err!r}")

        if self._stream_variant is None:
            def stream_as_int(s):
                return s

            def stream_as_handle(s):
                return cu.CUstream(s)

            last_exc: Exception | None = None
            for make_s in (stream_as_int, stream_as_handle):
                try:
                    try_launch(make_s)
                    self._stream_variant = make_s
                    return
                except TypeError as exc:
                    last_exc = exc
            raise RuntimeError(
                "cuLaunchKernel parameter marshalling failed for "
                f"{name}: {last_exc}")
        try_launch(self._stream_variant)


_RUNTIME_CACHE: dict[int, _CudaRuntime] = {}

_TILE_M = 32
_TILE_N = 32


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


class Model(nn.Module):
    def __init__(
        self,
        num_experts: int = 256,
        intermediate_size: int = 512,
        top_k: int = 8,
        block_shape: list[int] | tuple[int, int] = (128, 128),
    ) -> None:
        super().__init__()
        self.num_experts = int(num_experts)
        self.intermediate_size = int(intermediate_size)
        self.top_k = int(top_k)
        shape = tuple(int(value) for value in block_shape)
        if shape != (_GROUP, _GROUP):
            raise NotImplementedError(
                f"block_shape {shape} is not supported; only (128, 128)")
        self.block_shape = shape

    def forward(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        w1_scale: torch.Tensor,
        w2_scale: torch.Tensor,
    ) -> torch.Tensor:
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()
        if not w1.is_contiguous():
            w1 = w1.contiguous()
        if not w2.is_contiguous():
            w2 = w2.contiguous()
        if not topk_weights.is_contiguous():
            topk_weights = topk_weights.contiguous()
        if not topk_ids.is_contiguous():
            topk_ids = topk_ids.contiguous()
        if not w1_scale.is_contiguous():
            w1_scale = w1_scale.contiguous()
        if not w2_scale.is_contiguous():
            w2_scale = w2_scale.contiguous()

        token_count, hidden_size = hidden_states.shape
        experts, n1, k1 = w1.shape
        n2, k2 = w2.shape[1], w2.shape[2]
        top_k = topk_ids.shape[1]
        if experts != self.num_experts:
            raise ValueError("w1 expert dimension mismatch")
        if n1 != 2 * self.intermediate_size or k1 != hidden_size:
            raise ValueError("w1 shape mismatch")
        if n2 != hidden_size or k2 != self.intermediate_size:
            raise ValueError("w2 shape mismatch")
        if hidden_size % _GROUP or k1 % _GROUP or k2 % _GROUP:
            raise NotImplementedError("dimensions must be multiples of 128")
        if n1 % _TILE_N or n2 % _TILE_N:
            raise NotImplementedError("output dims must be multiples of 32")
        if experts > 256:
            raise NotImplementedError("num_experts above 256 is not supported")

        device = hidden_states.device
        runtime = _RUNTIME_CACHE.get(device.index)
        if runtime is None:
            runtime = _CudaRuntime(device.index)
            _RUNTIME_CACHE[device.index] = runtime
        stream = torch.cuda.current_stream(device).cuda_stream

        rows = token_count * top_k
        max_tiles = _cdiv(rows, _TILE_M) + experts + 2
        dev = device

        sorted_rows = torch.empty(rows, dtype=torch.int32, device=dev)
        desc = torch.empty(2 * max_tiles, dtype=torch.int32, device=dev)
        total_tiles = torch.empty(1, dtype=torch.int32, device=dev)
        offsets = torch.empty(experts + 1, dtype=torch.int32, device=dev)
        hist = torch.zeros(experts, dtype=torch.int32, device=dev)
        fill = torch.zeros(experts, dtype=torch.int32, device=dev)
        act_q = torch.empty(rows * k1, dtype=torch.uint8, device=dev)
        act_scale = torch.empty(rows * (k1 // _GROUP), dtype=torch.float32,
                                device=dev)
        # Holds gate_up (rows*n1) first, then out2 (rows*n2); n2 >= n1.
        # Zero-initialized so unrouted rows (reference leaves them zero) stay
        # zero even for out-of-domain routing ids.
        y16 = torch.zeros(rows * max(n1, n2), dtype=torch.bfloat16, device=dev)
        inter_q = torch.empty(rows * k2, dtype=torch.uint8, device=dev)
        inter_scale = torch.empty(rows * (k2 // _GROUP), dtype=torch.float32,
                                  device=dev)
        out = torch.empty((token_count, hidden_size),
                          dtype=hidden_states.dtype, device=dev)

        launch = runtime.launch
        base = dict(
            topk_ids=topk_ids.data_ptr(),
            topk_weights=topk_weights.data_ptr(),
            R=rows,
            experts=experts,
            top_k=top_k,
            tile_m=_TILE_M,
        )

        # 1. histogram
        launch("k_hist", (_cdiv(rows, 256), 1), (256, 1), stream,
               _pack_params(hist=hist.data_ptr(), **base))
        # 2. scan + descriptors
        launch("k_scan_desc", (1, 1), (256, 1), stream,
               _pack_params(hist=hist.data_ptr(), offsets=offsets.data_ptr(),
                            desc=desc.data_ptr(),
                            total_tiles=total_tiles.data_ptr(), **base))
        # 3. scatter
        launch("k_scatter", (_cdiv(rows, 256), 1), (256, 1), stream,
               _pack_params(sorted_rows=sorted_rows.data_ptr(),
                            offsets=offsets.data_ptr(), fill=fill.data_ptr(),
                            **base))
        # 4. activation quantization
        launch("k_quant_act", (_cdiv(rows * (k1 // _GROUP), 256), 1),
               (256, 1), stream,
               _pack_params(x_hidden=hidden_states.data_ptr(),
                            q_out=act_q.data_ptr(),
                            scale_out=act_scale.data_ptr(),
                            K=k1, n_groups_k=k1 // _GROUP,
                            R=rows, top_k=top_k))
        # 5. GEMM1: [rows, k1] x [experts, n1, k1]^T -> [rows, n1] (bf16)
        launch("k_gemm", (max_tiles, n1 // _TILE_N), (16, 8), stream,
               _pack_params(a_q=act_q.data_ptr(), a_scale=act_scale.data_ptr(),
                            w=w1.data_ptr(), w_scale=w1_scale.data_ptr(),
                            sorted_rows=sorted_rows.data_ptr(),
                            desc=desc.data_ptr(),
                            total_tiles=total_tiles.data_ptr(),
                            offsets=offsets.data_ptr(), y16=y16.data_ptr(),
                            N=n1, K=k1, n_groups_k=k1 // _GROUP,
                            w_stride=n1 * k1,
                            w_scale_stride=(n1 // _GROUP) * (k1 // _GROUP),
                            R=rows, experts=experts, tile_m=_TILE_M))
        # 6. SiLU*up + quantization
        launch("k_silu_quant", (_cdiv(rows * (k2 // _GROUP), 256), 1),
               (256, 1), stream,
               _pack_params(y16=y16.data_ptr(), q_out=inter_q.data_ptr(),
                            scale_out=inter_scale.data_ptr(),
                            N=n1, K=k2, n_groups_k=k2 // _GROUP,
                            R=rows))
        # 7. GEMM2: [rows, k2] x [experts, n2, k2]^T -> [rows, n2] (bf16)
        launch("k_gemm", (max_tiles, n2 // _TILE_N), (16, 8), stream,
               _pack_params(a_q=inter_q.data_ptr(),
                            a_scale=inter_scale.data_ptr(),
                            w=w2.data_ptr(), w_scale=w2_scale.data_ptr(),
                            sorted_rows=sorted_rows.data_ptr(),
                            desc=desc.data_ptr(),
                            total_tiles=total_tiles.data_ptr(),
                            offsets=offsets.data_ptr(), y16=y16.data_ptr(),
                            N=n2, K=k2, n_groups_k=k2 // _GROUP,
                            w_stride=n2 * k2,
                            w_scale_stride=(n2 // _GROUP) * (k2 // _GROUP),
                            R=rows, experts=experts, tile_m=_TILE_M))
        # 8. weighted top-k reduction
        launch("k_reduce", (_cdiv(token_count * hidden_size, 256), 1),
               (256, 1), stream,
               _pack_params(y16=y16.data_ptr(), out=out.data_ptr(),
                            topk_weights=topk_weights.data_ptr(),
                            N=hidden_size, R=rows, top_k=top_k))
        return out
