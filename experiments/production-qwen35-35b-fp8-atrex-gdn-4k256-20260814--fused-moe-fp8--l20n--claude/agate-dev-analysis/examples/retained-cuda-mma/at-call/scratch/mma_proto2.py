"""MMA layout probe + candidate fix.

Kernel A: original mapping (a0=(r0-7,k0-15), a1=(r0-7,k16-31),
a2=(r8-15,k0-15), a3=(r8-15,k16-31)).
Kernel B: swapped mapping (a0=(r0-7,k0-15), a1=(r8-15,k0-15),
a2=(r0-7,k16-31), a3=(r8-15,k16-31)), matching the f16-m16n8k16 analogy.

First runs exact random-integer tests on both. Then, if either failed,
runs one-hot mapping probes to dump the hardware's actual row/k reading of
each placement slot, so the correct permutation is visible in the output.
"""
import sys

import torch

CUDA_SRC = r"""
extern "C" {

// Original mapping
__global__ void mma_orig(const unsigned char* A, const unsigned char* B,
                         float* C) {
  int lane = threadIdx.x;
  unsigned int a0 = *(const unsigned int*)(A + (lane >> 2) * 32 + (lane & 3) * 4);
  unsigned int a1 = *(const unsigned int*)(A + (lane >> 2) * 32 + 16 + (lane & 3) * 4);
  unsigned int a2 = *(const unsigned int*)(A + ((lane >> 2) + 8) * 32 + (lane & 3) * 4);
  unsigned int a3 = *(const unsigned int*)(A + ((lane >> 2) + 8) * 32 + 16 + (lane & 3) * 4);
  unsigned int b0 = *(const unsigned int*)(B + (lane >> 2) * 32 + (lane & 3) * 4);
  unsigned int b1 = *(const unsigned int*)(B + (lane >> 2) * 32 + 16 + (lane & 3) * 4);
  float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  C[(lane >> 2) * 8 + (lane & 3) * 2 + 0] = c0;
  C[(lane >> 2) * 8 + (lane & 3) * 2 + 1] = c1;
  C[((lane >> 2) + 8) * 8 + (lane & 3) * 2 + 0] = c2;
  C[((lane >> 2) + 8) * 8 + (lane & 3) * 2 + 1] = c3;
}

// Swapped mapping (f16-analogy)
__global__ void mma_swap(const unsigned char* A, const unsigned char* B,
                         float* C) {
  int lane = threadIdx.x;
  unsigned int a0 = *(const unsigned int*)(A + (lane >> 2) * 32 + (lane & 3) * 4);
  unsigned int a1 = *(const unsigned int*)(A + ((lane >> 2) + 8) * 32 + (lane & 3) * 4);
  unsigned int a2 = *(const unsigned int*)(A + (lane >> 2) * 32 + 16 + (lane & 3) * 4);
  unsigned int a3 = *(const unsigned int*)(A + ((lane >> 2) + 8) * 32 + 16 + (lane & 3) * 4);
  unsigned int b0 = *(const unsigned int*)(B + (lane >> 2) * 32 + (lane & 3) * 4);
  unsigned int b1 = *(const unsigned int*)(B + (lane >> 2) * 32 + 16 + (lane & 3) * 4);
  float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  C[(lane >> 2) * 8 + (lane & 3) * 2 + 0] = c0;
  C[(lane >> 2) * 8 + (lane & 3) * 2 + 1] = c1;
  C[((lane >> 2) + 8) * 8 + (lane & 3) * 2 + 0] = c2;
  C[((lane >> 2) + 8) * 8 + (lane & 3) * 2 + 1] = c3;
}

// Batched probe: block p does one MMA with A[p], B[p] -> C[p].
__global__ void mma_batch(const unsigned char* A, const unsigned char* B,
                          float* C, int nprobes) {
  int p = blockIdx.x;
  if (p >= nprobes) return;
  int lane = threadIdx.x;
  const unsigned char* Ap = A + (long long)p * 16 * 32;
  const unsigned char* Bp = B + (long long)p * 8 * 32;
  unsigned int a0 = *(const unsigned int*)(Ap + (lane >> 2) * 32 + (lane & 3) * 4);
  unsigned int a1 = *(const unsigned int*)(Ap + ((lane >> 2) + 8) * 32 + (lane & 3) * 4);
  unsigned int a2 = *(const unsigned int*)(Ap + (lane >> 2) * 32 + 16 + (lane & 3) * 4);
  unsigned int a3 = *(const unsigned int*)(Ap + ((lane >> 2) + 8) * 32 + 16 + (lane & 3) * 4);
  unsigned int b0 = *(const unsigned int*)(Bp + (lane >> 2) * 32 + (lane & 3) * 4);
  unsigned int b1 = *(const unsigned int*)(Bp + (lane >> 2) * 32 + 16 + (lane & 3) * 4);
  float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  float* Cp = C + (long long)p * 16 * 8;
  Cp[(lane >> 2) * 8 + (lane & 3) * 2 + 0] = c0;
  Cp[(lane >> 2) * 8 + (lane & 3) * 2 + 1] = c1;
  Cp[((lane >> 2) + 8) * 8 + (lane & 3) * 2 + 0] = c2;
  Cp[((lane >> 2) + 8) * 8 + (lane & 3) * 2 + 1] = c3;
}

}  // extern "C"
"""


def main():
    from cuda.bindings import driver as cu
    from cuda.bindings import nvrtc

    def drv(res):
        if res[0] != cu.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"driver error {res[0]!r}")
        return res[1:]

    def nv(res):
        if res[0] != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError(f"nvrtc error {res[0]!r}")
        return res[1:]

    torch.cuda.init()
    torch.empty(1, device="cuda")
    drv(cu.cuInit(0))
    err, ctx = cu.cuCtxGetCurrent()
    if ctx is None:
        err, dev = cu.cuDeviceGet(0)
        err, ctx = cu.cuDevicePrimaryCtxRetain(dev)
        drv(cu.cuCtxSetCurrent(ctx))

    prog, = nv(nvrtc.nvrtcCreateProgram(CUDA_SRC.encode(), b"proto2.cu", 0, [], []))
    opts = [b"--gpu-architecture=compute_120"]
    err, = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        sz, = nv(nvrtc.nvrtcGetProgramLogSize(prog))
        buf = bytearray(sz)
        nv(nvrtc.nvrtcGetProgramLog(prog, buf))
        print("PROTO2 COMPILE FAIL")
        print(buf.decode(errors="replace")[-2000:])
        sys.exit(1)
    sz, = nv(nvrtc.nvrtcGetPTXSize(prog))
    ptx = bytearray(sz)
    nv(nvrtc.nvrtcGetPTX(prog, ptx))
    module, = drv(cu.cuModuleLoadData(ptx))
    f_orig, = drv(cu.cuModuleGetFunction(module, b"mma_orig"))
    f_swap, = drv(cu.cuModuleGetFunction(module, b"mma_swap"))
    f_batch, = drv(cu.cuModuleGetFunction(module, b"mma_batch"))

    stream = torch.cuda.current_stream().cuda_stream
    keep = []

    def launch(func, grid, block, ptrs):
        cell = torch.tensor(ptrs, dtype=torch.int64)
        arr = torch.tensor([cell.data_ptr() + 8 * i for i in range(len(ptrs))],
                           dtype=torch.int64)
        keep.append((cell, arr))
        err, = cu.cuLaunchKernel(func, grid, 1, 1, block, 1, 1, 0, stream,
                                 int(arr.data_ptr()), 0)
        if err != cu.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"launch failed {err!r}")

    ONE = 0x38  # e4m3 encoding of 1.0
    ok = True
    results = {}

    def tile_test(func, name):
        torch.manual_seed(0)
        Af = torch.randint(-6, 7, (16, 32), device="cuda").float()
        Bf = torch.randint(-6, 7, (8, 32), device="cuda").float()
        Aq = Af.to(torch.float8_e4m3fn)
        Bq = Bf.to(torch.float8_e4m3fn)
        C = torch.zeros(16, 8, device="cuda")
        launch(func, 1, 32, [Aq.data_ptr(), Bq.data_ptr(), C.data_ptr()])
        torch.cuda.synchronize()
        ref = Af @ Bf.t()
        err = (C - ref).abs().max().item()
        print(f"PROTO2 {name} max_err={err}")
        return err == 0.0

    ok_orig = tile_test(f_orig, "mma_orig")
    ok_swap = tile_test(f_swap, "mma_swap")
    ok = ok_orig or ok_swap
    results["mapping_ok"] = "orig" if ok_orig else ("swap" if ok_swap else "none")
    print(f"PROTO2 mapping={results['mapping_ok']}")

    if not ok:
        # Probe the swap-mapping kernel's hardware interpretation.
        # A probe: one-hot A at every (r,k) under the *placement* below,
        # B all ones -> fired row reveals HW row of that slot.
        nA = 16 * 32
        Ap = torch.zeros(nA, 16, 32, dtype=torch.uint8, device="cuda")
        Bp = torch.full((nA, 8, 32), ONE, dtype=torch.uint8, device="cuda")
        Cp = torch.zeros(nA, 16, 8, device="cuda")
        # placement under swap mapping: find (row_expr, col_expr) for (r,k)
        for r in range(16):
            for k in range(32):
                # inverse of swap kernel's load: which (lane, reg, q) loads
                # position (r,k)? reg a0: rows0-7/cols0-15; a1: rows8-15/cols0-15
                # a2: rows0-7/cols16-31; a3: rows8-15/cols16-31
                p = r * 32 + k
                grp = r & 7
                tig = (k & 15) >> 2
                q = k & 3
                Ap[p, r, tig * 4 + q if k < 16 else 16 + (k & 15) // 4 * 4 + ((k & 15) % 4)] = ONE
        launch(f_batch, nA, 32, [Ap.data_ptr(), Bp.data_ptr(), Cp.data_ptr(), nA])
        torch.cuda.synchronize()
        obs = []
        for p in range(nA):
            rowsum = Cp[p].sum(dim=1)  # [16]
            fired = torch.nonzero(rowsum > 0.5).flatten().tolist()
            obs.append(fired)
        bad = 0
        for r in range(16):
            line = []
            for k in range(32):
                f = obs[r * 32 + k]
                exp = r
                got = f[0] if len(f) == 1 else -1
                if got != exp:
                    bad += 1
                line.append(got)
            print(f"PROBE_A r={r:2d} rows_seen={line}")
        print(f"PROBE_A mismatches={bad}/512")

        # B probe: A all ones, B one-hot per (k,n) with value v(k)=1+k%7
        nB = 32 * 8
        Ap2 = torch.full((nB, 16, 32), ONE, dtype=torch.uint8, device="cuda")
        Bp2 = torch.zeros(nB, 8, 32, dtype=torch.uint8, device="cuda")
        Cp2 = torch.zeros(nB, 16, 8, device="cuda")
        vals = [(1 + (k % 7)) for k in range(32)]
        qv = torch.tensor(vals, dtype=torch.float32).to(torch.float8_e4m3fn)
        qv = qv.view(torch.uint8).cuda()
        for k in range(32):
            for n in range(8):
                p = k * 8 + n
                Bp2[p, n, k] = qv[k]
        launch(f_batch, nB, 32, [Ap2.data_ptr(), Bp2.data_ptr(), Cp2.data_ptr(), nB])
        torch.cuda.synchronize()
        badb = 0
        for k in range(32):
            line = []
            for n in range(8):
                p = k * 8 + n
                col = Cp2[p, 0]  # [8], value v at fired col
                nz = torch.nonzero(col.abs() > 0.5).flatten().tolist()
                if len(nz) == 1:
                    got_n = nz[0]
                    got_v = col[nz[0]].item()
                    line.append((got_n, round(got_v, 1)))
                    if got_n != n or abs(got_v - vals[k]) > 1e-3:
                        badb += 1
                else:
                    line.append(None)
                    badb += 1
            print(f"PROBE_B k={k:2d} (col,val)={line}")
        print(f"PROBE_B mismatches={badb}/256")

    print("PROTO2_VERDICT " + ("PASS" if ok else "PROBED"))
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
