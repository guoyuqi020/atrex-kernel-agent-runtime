"""D1 component probe: decompose the small-shape latency floor + prototype a raw-launch host path.

For each case class measures (us/call):
  eager_wall  : CUDA events around an unsynced N-iteration loop (probe/evaluator style)
  cpu_rate    : perf_counter around an unsynced loop (pure host enqueue rate)
  gpu_sum     : torch.profiler per-kernel device-time sums, split into
                memset / fa_mma / fa_merge / fa_wide
  graph_replay: events around N graph replays (GPU-only floor, Python stripped)
  verdict     : CPU-BOUND / GPU-BOUND / MIXED from the decomposition
Plus isolated host-component costs (zeros_like, empty_like, contiguous x6,
current_device+capability, _load_kernels cached path, Stream.from_handle,
LaunchConfig construction, one cuda.core launch enqueue for fa_mma and
fa_merge, workspace torch.empty, and a RAW cached-arg cuLaunchKernel enqueue).

Section 3 prototypes the D1(b) host fast path WITHOUT editing the candidate:
`kernel.launch` (the module-level cuda.core `launch` symbol) is monkeypatched
with a wrapper that reuses per-signature pre-packed ctypes argument buffers and
calls cuda.bindings.driver.cuLaunchKernel directly.  For every class it reports
the raw-path wall/cpu rates and whether the raw-path output is BITWISE equal to
the cuda.core-path output, plus CUDA-graph capture+replay under the raw path.
Timing + mechanism validation only; no candidate edits involved.
"""
import ctypes
import time

import torch
from torch.profiler import ProfilerActivity, profile

import kernel as cand
from cuda.core import LaunchConfig, Stream, launch
from cuda.bindings import driver as cu

ARGS_CACHE = {}
RAW_STATS = {"calls": 0, "errs": []}


def make_case(name, q_lens, kv_lens, slots=None, seed=0):
    torch.manual_seed(seed)
    dev = "cuda"
    B = len(q_lens)
    cu_ = [0]
    for ql in q_lens:
        cu_.append(cu_[-1] + ql)
    active = cu_[-1]
    slots = active if slots is None else slots
    pages_per = [(kl + 63) // 64 for kl in kv_lens]
    total_pages = sum(pages_per)
    q = torch.randn(slots, 16, 256, device=dev, dtype=torch.float32).to(torch.bfloat16)
    k = torch.randn(total_pages, 64, 2, 256, device=dev, dtype=torch.float32).to(torch.bfloat16)
    v = torch.randn(total_pages, 64, 2, 256, device=dev, dtype=torch.float32).to(torch.bfloat16)
    cu_q = torch.tensor(cu_, dtype=torch.int32, device=dev)
    seqk = torch.tensor(kv_lens, dtype=torch.int32, device=dev)
    btab = torch.zeros(B, 4097, dtype=torch.int32, device=dev)
    poff = 0
    for b in range(B):
        for i in range(pages_per[b]):
            btab[b, i] = poff + i
        poff += pages_per[b]
    maxq = max(q_lens)
    maxk = max(kv_lens)
    return dict(name=name, q=q, k=k, v=v, cu_q=cu_q, seqk=seqk, btab=btab,
                active=active, slots=slots, maxq=maxq, maxk=maxk)


def build_fn(case):
    m = cand.Model(max_seqlen_q=case["maxq"], max_seqlen_k=case["maxk"],
                   softmax_scale=0.0625, fa_version=3)
    ones = torch.ones(case["cu_q"].shape[0] - 1, 2, dtype=torch.float32, device="cuda")
    args = (case["q"], case["k"], case["v"], case["cu_q"], case["seqk"],
            case["btab"], ones, ones, ones, None)
    return lambda: m(*args)


def wall_us(fn, iters):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000.0 / iters


def cpu_us(fn, iters):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    torch.cuda.synchronize()
    return (t1 - t0) * 1e6 / iters


def gpu_breakdown(fn, iters):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    ms = mma = mrg = wide = other = 0.0
    for ev in prof.key_averages():
        t = float(ev.self_device_time_total) / iters
        if t <= 0:
            continue
        key = ev.key.lower()
        if "fa_mma" in key:
            mma += t
        elif "fa_merge" in key:
            mrg += t
        elif "fa_wide" in key:
            wide += t
        elif "memset" in key or "fill" in key or "elementwise" in key:
            ms += t
        else:
            other += t
    return ms, mma, mrg, wide, other


def graph_replay_us(fn, iters):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    ev0 = torch.cuda.Event(True); ev1 = torch.cuda.Event(True)
    ev0.record()
    for _ in range(iters):
        g.replay()
    ev1.record(); torch.cuda.synchronize()
    return ev0.elapsed_time(ev1) * 1000.0 / iters, g


# ---------------------------------------------------------------- raw launch
def _stream_handle(stream):
    h = getattr(stream, "handle", stream)
    return h if isinstance(h, int) else int(h)


def _cfg_parts(cfg):
    grid = getattr(cfg, "grid", None)
    if grid is None:
        grid = getattr(cfg, "grid_dim", None)
    block = getattr(cfg, "block", None)
    if block is None:
        block = getattr(cfg, "block_dim", None)
    smem = 0
    for attr in ("dynamic_smem_size", "shmem_size", "shared_mem_bytes"):
        val = getattr(cfg, attr, None)
        if val:
            smem = int(val)
            break
    gx, gy, gz = (grid + (1, 1, 1))[:3] if isinstance(grid, tuple) else (int(grid), 1, 1)
    bx, by, bz = (block + (1, 1, 1))[:3] if isinstance(block, tuple) else (int(block), 1, 1)
    return int(gx), int(gy), int(gz), int(bx), int(by), int(bz), smem


def raw_launch(stream, config, kernel, *args):
    """cuda.core `launch` drop-in: cached pre-packed ctypes args + cuLaunchKernel.

    Handle params are passed as raw ints (cuda.bindings accepts them; the
    incumbent already passes cuda.core's int handle to cuKernelSetAttribute).
    """
    gx, gy, gz, bx, by, bz, smem = _cfg_parts(config)
    key = id(kernel)
    entry = ARGS_CACHE.get(key)
    if entry is None or entry[0] != len(args):
        objs = [ctypes.c_double(a) if isinstance(a, float)
                else ctypes.c_longlong(int(a)) for a in args]
        arr = (ctypes.c_void_p * len(objs))(*[ctypes.addressof(o) for o in objs])
        entry = (len(args), objs, arr, kernel.handle)
        ARGS_CACHE[key] = entry
    _, objs, arr, fh = entry
    for i, a in enumerate(args):
        objs[i].value = a          # c_double accepts float, c_longlong accepts int
    RAW_STATS["calls"] += 1
    res = cu.cuLaunchKernel(fh, gx, gy, gz, bx, by, bz, smem,
                            _stream_handle(stream), arr, 0)
    err = res[0] if isinstance(res, tuple) else res
    if int(err) != 0:
        RAW_STATS["errs"].append(str(err))
        raise RuntimeError(f"cuLaunchKernel failed: {err}")


def host_components(case):
    q = case["q"]
    tensors = (case["q"], case["k"], case["v"], case["cu_q"], case["seqk"], case["btab"])
    N = 500
    res = {}

    def timed(name, f, n=N):
        for _ in range(5):
            f()
        t0 = time.perf_counter()
        for _ in range(n):
            f()
        t1 = time.perf_counter()
        res[name] = (t1 - t0) * 1e6 / n

    timed("zeros_like(q)", lambda: torch.zeros_like(q))
    timed("empty_like(q)", lambda: torch.empty_like(q))
    timed("contiguous()x6", lambda: [t.contiguous() for t in tensors])
    timed("cur_dev+capability", lambda: (torch.cuda.current_device(),
                                         torch.cuda.get_device_capability()))
    timed("_load_kernels(cached)", lambda: cand._load_kernels())
    timed("Stream.from_handle", lambda: Stream.from_handle(
        int(torch.cuda.current_stream().cuda_stream)))
    kernels = cand._load_kernels()
    fa_mma, fa_merge = kernels[0], kernels[1]
    splits = 55
    base = 2
    o = torch.empty(splits * base * 64 * 256, dtype=torch.float32, device="cuda")
    mp = torch.empty(splits * base * 64, dtype=torch.float32, device="cuda")
    lp = torch.empty(splits * base * 64, dtype=torch.float32, device="cuda")
    od = torch.zeros(8 * 16 * 256, dtype=torch.int16, device="cuda")
    stream = Stream.from_handle(int(torch.cuda.current_stream().cuda_stream))
    timed("torch.empty(ws 55-split)", lambda: torch.empty(
        splits * base * 64 * 256 + 2 * splits * base * 64,
        dtype=torch.float32, device="cuda"))
    cfg = LaunchConfig(grid=(1, 1, 2), block=256)
    timed("LaunchConfig(merge) ctor", lambda: LaunchConfig(grid=(1, 1, 2), block=256))
    timed("LaunchConfig(mma,99KB) ctor", lambda: LaunchConfig(
        grid=(2, 1, 110), block=128, shmem_size=101376))
    mcfg = LaunchConfig(grid=(1, 1, 32), block=256)
    timed("launch(fa_merge) enqueue", lambda: launch(
        stream, mcfg, fa_merge, od.data_ptr(), o.data_ptr(), mp.data_ptr(),
        lp.data_ptr(), case["cu_q"].data_ptr(), splits))
    timed("RAW launch(fa_merge) enqueue", lambda: raw_launch(
        stream, mcfg, fa_merge, od.data_ptr(), o.data_ptr(), mp.data_ptr(),
        lp.data_ptr(), case["cu_q"].data_ptr(), splits))
    # fa_mma direct-mode enqueue: 15 args incl. the 99KB dynamic smem.
    bigcfg = LaunchConfig(grid=(2, 1, 2), block=128, shmem_size=101376)
    mma_args = (case["q"].data_ptr(), case["k"].data_ptr(), case["v"].data_ptr(),
                od.data_ptr(), 0, 0, 0, case["cu_q"].data_ptr(),
                case["seqk"].data_ptr(), case["btab"].data_ptr(), 4097,
                0.0625 * 1.4426950408889634, 1, 1, case["slots"])
    timed("launch(fa_mma) enqueue", lambda: launch(
        stream, bigcfg, fa_mma, *mma_args))
    timed("RAW launch(fa_mma) enqueue", lambda: raw_launch(
        stream, bigcfg, fa_mma, *mma_args))
    timed("RAW launch(fa_mma) cached-only", lambda: raw_launch(
        stream, bigcfg, fa_mma, *mma_args))
    torch.cuda.synchronize()
    return res


def raw_path_validation(cases):
    """Bitwise-equality + graph-capture check of the raw launch path."""
    print("\n=== raw cuLaunchKernel path (monkeypatched kernel.launch) ===")
    print(f"{'case':20s} {'equal':>6s} {'graph':>6s} {'wall_cc':>8s} {'wall_raw':>9s} "
          f"{'cpu_cc':>7s} {'cpu_raw':>7s} {'replay_cc':>9s} {'replay_raw':>10s}")
    for case in cases:
        fn = build_fn(case)
        big = case["slots"] >= 512 or case["name"].startswith("prefill")
        n_wall = 60 if big else 150
        # reference outputs via the stock cuda.core path
        out_cc = fn()
        torch.cuda.synchronize()
        ref = out_cc.clone()
        rep_cc, _ = graph_replay_us(fn, 30 if big else 60)
        w_cc = wall_us(fn, n_wall)
        c_cc = cpu_us(fn, 60)
        # swap in the raw launcher
        orig = cand.launch
        cand.launch = raw_launch
        try:
            out_raw = fn()
            torch.cuda.synchronize()
            equal = bool(torch.equal(out_raw.view(torch.int16), ref.view(torch.int16)))
            try:
                rep_raw, _ = graph_replay_us(fn, 30 if big else 60)
                graph_ok = "ok"
            except Exception as exc:            # noqa: BLE001
                rep_raw = float("nan")
                graph_ok = f"FAIL:{type(exc).__name__}"
            w_raw = wall_us(fn, n_wall)
            c_raw = cpu_us(fn, 60)
        finally:
            cand.launch = orig
        print(f"{case['name']:20s} {str(equal):>6s} {graph_ok:>6s} {w_cc:8.1f} {w_raw:9.1f} "
              f"{c_cc:7.1f} {c_raw:7.1f} {rep_cc:9.1f} {rep_raw:10.1f}")
    print(f"raw launch calls: {RAW_STATS['calls']}, errors: {RAW_STATS['errs'][:3]}")


def main():
    torch.cuda.init()
    p = torch.cuda.get_device_properties(0)
    print(f"device: {p.name} sm_{p.major}{p.minor} SMs={p.multi_processor_count} "
          f"mem={p.total_memory >> 20}MB")
    try:
        from cuda.bindings import runtime as rt
        err, clo = rt.cudaDeviceGetAttribute(
            rt.cudaDeviceAttr.cudaDevAttrClockRate, 0)
        err2, memclo = rt.cudaDeviceGetAttribute(
            rt.cudaDeviceAttr.cudaDevAttrMemoryClockRate, 0)
        err3, busw = rt.cudaDeviceGetAttribute(
            rt.cudaDeviceAttr.cudaDevAttrGlobalMemoryBusWidth, 0)
        if int(err) == 0 and int(err2) == 0 and int(err3) == 0:
            peak = 2.0 * clo * busw / 8.0
            print(f"clock_khz={clo} memclock_khz={memclo} buswidth_bits={busw} "
                  f"theoretical_BW_GBs={peak:.1f}")
    except Exception as exc:                    # noqa: BLE001
        print(f"(device attrs unavailable: {exc})")

    probe_cfg = LaunchConfig(grid=(1, 1, 1), block=128, shmem_size=101376)
    fields = getattr(probe_cfg, "__dict__", None)
    print(f"LaunchConfig attrs: {fields if fields else probe_cfg}")
    print(f"_cfg_parts(probe_cfg) -> {_cfg_parts(probe_cfg)}")
    try:
        st = cand._current_stream(0)
        print(f"stream handle type={type(getattr(st, 'handle', None))} "
              f"raw={torch._C._cuda_getCurrentRawStream(0)}")
    except Exception as exc:                    # noqa: BLE001
        print(f"(stream introspection failed: {exc})")

    cases = [
        make_case("single_token", [1], [1], slots=8),
        make_case("decode1_kv4578", [1], [4578], slots=8),
        make_case("decode_b8_q1", [1] * 8, [4352] * 8, slots=8),
        make_case("small_decode_b4", [4, 4, 4, 4], [4096] * 4, slots=16),
        make_case("mid_decode_b16", [8] * 16, [4096] * 16, slots=128),
        make_case("mid_b16_q8_kv3072", [8] * 16, [3072] * 16, slots=128),
        make_case("wave_b32_q16", [16] * 32, [4096] * 32, slots=512),
        make_case("decode_b32_q1", [1] * 32, [4096] * 32, slots=32),
        make_case("padded_mid_b14", [16] * 14, [4096] * 14, slots=1024),
        make_case("prefill_2048", [2048], [4352], slots=2048),
    ]
    print(f"\n{'case':20s} {'wall':>8s} {'cpu':>8s} {'gpu_sum':>8s} {'memset':>7s} "
          f"{'fa_mma':>8s} {'fa_merge':>8s} {'fa_wide':>8s} {'other':>6s} {'replay':>8s}  verdict")
    for case in cases:
        fn = build_fn(case)
        big = case["slots"] >= 512 or case["name"].startswith("prefill")
        n_wall = 100 if big else 200
        n_prof = 20 if big else 50
        n_rep = 50 if big else 100
        w = wall_us(fn, n_wall)
        c = cpu_us(fn, 100)
        ms, mma, mrg, wide, other = gpu_breakdown(fn, n_prof)
        rep, _ = graph_replay_us(fn, n_rep)
        gsum = ms + mma + mrg + wide + other
        if c >= 0.8 * w and c > gsum:
            verdict = "CPU-BOUND"
        elif gsum >= 0.8 * w:
            verdict = "GPU-BOUND"
        else:
            verdict = "MIXED"
        print(f"{case['name']:20s} {w:8.1f} {c:8.1f} {gsum:8.1f} {ms:7.1f} "
              f"{mma:8.1f} {mrg:8.1f} {wide:8.1f} {other:6.1f} {rep:8.1f}  {verdict}")

    print("\nhost component costs (us, decode1-class tensors):")
    for k, v in host_components(cases[1]).items():
        print(f"  {k:32s} {v:8.2f}")
    print("\nhost component costs (us, slots=1024 padded-mid tensors):")
    for k, v in host_components(cases[8]).items():
        print(f"  {k:32s} {v:8.2f}")

    raw_path_validation(cases)
    print("COMPONENT PROBE DONE")


if __name__ == "__main__":
    main()
