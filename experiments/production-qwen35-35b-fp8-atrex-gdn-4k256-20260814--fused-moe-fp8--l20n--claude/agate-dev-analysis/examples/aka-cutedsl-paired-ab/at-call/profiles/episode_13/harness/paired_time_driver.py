"""Episode-13 fallback timing driver: same-pod paired A/B of candidate vs HEAD,
with instantiation-order bias cancellation.

The first revision of this driver measured a large slot bias in A/A mode
(incumbent vs itself: -32us at T=8192, +16us at T=512) because the model
instantiated first gets systematically different memory placement. This
revision cancels that bias with two measurement phases: phase 1 instantiates
the candidate first (slot bias +B on the candidate delta), phase 2
instantiates the incumbent first (slot bias -B on the candidate delta); the
mean of the two phase deltas is the bias-corrected estimate. Within each
phase the two models alternate per iteration so slow drift cancels in the
mean. Models hold only compiled executors (weights are call arguments), so
four instances fit easily.

Emits a JSON digest. No evaluator state is touched; this driver only calls
the candidates' forward() on synthetic in-domain inputs.

Environment (TORCH_ prefix required by the gateway env allowlist):
    TORCH_PROFILE_ITERS   timed iterations per phase (default 60)
    TORCH_PROFILE_WARMUP  warmup iterations per model (default 8)
    TORCH_PROFILE_TOKENS  packed token count of the synthetic case (default 8192)
    TORCH_PROFILE_MODE    ab (candidate vs HEAD) or aa (HEAD vs HEAD control)
"""

from __future__ import annotations

import importlib.util
import json
import os
import statistics
import sys
from pathlib import Path
from types import ModuleType

HARNESS_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = HARNESS_DIR.parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))


def _import_from(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    warmup = max(0, int(os.environ.get("TORCH_PROFILE_WARMUP", "8")))
    iters = max(1, int(os.environ.get("TORCH_PROFILE_ITERS", "60")))
    tokens = int(os.environ.get("TORCH_PROFILE_TOKENS", "8192"))
    mode = os.environ.get("TORCH_PROFILE_MODE", "ab")
    if not (450 <= tokens <= 8192):
        raise SystemExit("TORCH_PROFILE_TOKENS outside the public shape_domain [450, 8192]")

    device = "cuda"
    import torch

    kernel_v14 = _import_from(WORKSPACE_ROOT / "kernel.py", "kernel_candidate")
    kernel_v13 = _import_from(HARNESS_DIR / "incumbent_kernel.py", "kernel_incumbent")
    if mode == "aa":
        kernel_v14 = kernel_v13  # control: incumbent against itself
    inputs_module = _import_from(WORKSPACE_ROOT / "input.py", "workload_input")

    init_kwargs = {
        "num_experts": 256,
        "intermediate_size": 512,
        "top_k": 8,
        "block_shape": [128, 128],
    }
    input_kwargs = {
        "token_count": tokens,
        "hidden_size": 2048,
        "intermediate_size": 512,
        "num_experts": 256,
        "top_k": 8,
        "block_shape": [128, 128],
    }

    raw = inputs_module._make_inputs(**input_kwargs)
    inputs = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in raw.items()
    }

    stream = torch.cuda.current_stream(device)

    def run_phase(first, second, n):
        """Alternate first/second over n iterations; return per-iteration
        (first_us, second_us)."""
        ev_a = torch.cuda.Event(enable_timing=True)
        ev_b = torch.cuda.Event(enable_timing=True)
        ev_c = torch.cuda.Event(enable_timing=True)
        first_us: list[float] = []
        second_us: list[float] = []
        with torch.no_grad():
            for i in range(n):
                if i % 2 == 0:
                    ev_a.record(stream)
                    first(**inputs)
                    ev_b.record(stream)
                    second(**inputs)
                    ev_c.record(stream)
                else:
                    ev_a.record(stream)
                    second(**inputs)
                    ev_b.record(stream)
                    first(**inputs)
                    ev_c.record(stream)
                torch.cuda.synchronize(device)
                t_ab = ev_a.elapsed_time(ev_b) * 1000.0
                t_bc = ev_b.elapsed_time(ev_c) * 1000.0
                if i % 2 == 0:
                    first_us.append(t_ab)
                    second_us.append(t_bc)
                else:
                    second_us.append(t_ab)
                    first_us.append(t_bc)
        return first_us, second_us

    with torch.no_grad():
        # Phase 1: candidate instantiated first (slot bias +B on candidate).
        cand_1 = kernel_v14.Model(**init_kwargs).to(device).eval()
        inc_1 = kernel_v13.Model(**init_kwargs).to(device).eval()
        # Phase 2: incumbent instantiated first (slot bias -B on candidate).
        inc_2 = kernel_v13.Model(**init_kwargs).to(device).eval()
        cand_2 = kernel_v14.Model(**init_kwargs).to(device).eval()

        for _ in range(warmup):
            cand_1(**inputs)
            inc_1(**inputs)
            inc_2(**inputs)
            cand_2(**inputs)
        torch.cuda.synchronize(device)

        c1_us, i1_us = run_phase(cand_1, inc_1, iters)
        i2_us, c2_us = run_phase(inc_2, cand_2, iters)

    d1 = [c - i for c, i in zip(c1_us, i1_us)]   # candidate - incumbent, phase 1
    d2 = [c - i for c, i in zip(c2_us, i2_us)]   # candidate - incumbent, phase 2
    m1 = statistics.fmean(d1)
    m2 = statistics.fmean(d2)
    corrected = (m1 + m2) / 2.0
    out = {
        "tokens": tokens,
        "mode": mode,
        "iters_per_phase": iters,
        "phase1_candidate_first_delta_mean_us": round(m1, 2),
        "phase2_incumbent_first_delta_mean_us": round(m2, 2),
        "bias_cancelled_delta_mean_us": round(corrected, 2),
        "slot_bias_estimate_us": round((m1 - m2) / 2.0, 2),
        "phase1_delta_stdev_us": round(statistics.stdev(d1), 2),
        "phase2_delta_stdev_us": round(statistics.stdev(d2), 2),
        "corrected_se_us": round(
            ((statistics.stdev(d1) ** 2 + statistics.stdev(d2) ** 2) / iters) ** 0.5
            / 2.0,
            2,
        ),
    }
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
