#!/usr/bin/env python3
"""Agate smoke checks for the SM120 upstream kernel and target ABI."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ATOL = 0.06
RTOL = 0.04
for _path in (ROOT / "vendor_support", ROOT / "vendor/flash_attention"):
    sys.path.insert(0, str(_path))

import torch


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sm120_bf16_smoke() -> None:
    """Compile and execute the dense BF16 capability present in upstream SM120 FA4."""
    from flash_attn.cute import flash_attn_varlen_func

    torch.manual_seed(1)
    device = torch.device("cuda", 0)
    q = torch.randn(128, 16, 256, dtype=torch.bfloat16, device=device) * 0.1
    k = torch.randn(128, 1, 256, dtype=torch.bfloat16, device=device) * 0.1
    v = torch.randn_like(k) * 0.1
    cu_q = torch.tensor([0, 128], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, 128], dtype=torch.int32, device=device)
    with torch.inference_mode():
        out = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=128,
            max_seqlen_k=128,
            softmax_scale=256**-0.5,
            causal=True,
            num_splits=1,
            pack_gqa=True,
            tile_mn=(64, 64),
        )
        if isinstance(out, (tuple, list)):
            out = out[0]
        torch.cuda.synchronize()
    if tuple(out.shape) != tuple(q.shape) or not torch.isfinite(out.float()).all():
        raise RuntimeError(f"invalid SM120 smoke output: {out.shape=} {out.dtype=}")
    print(json.dumps({"status": "passed", "mode": "sm120-bf16", "shape": list(out.shape)}))


def target_smoke(shape_id: str) -> None:
    """Run one production-contract shape through the correctness-first SM120 R0."""
    input_module = _load("aka_r0_input", ROOT / "reference/input.py")
    reference_module = _load("aka_r0_reference", ROOT / "reference/reference.py")
    candidate_module = _load("aka_r0_candidate", ROOT / "kernel.py")
    payload = json.loads((ROOT / "reference/shapes.json").read_text())
    entry = next(item for item in payload["shapes"] if str(item["id"]) == shape_id)
    torch.manual_seed(1)
    candidate_inputs = input_module._make_inputs(**entry["input_kwargs"])
    torch.manual_seed(1)
    reference_inputs = input_module._make_inputs(**entry["input_kwargs"])
    with torch.inference_mode():
        actual = candidate_module.Model().cuda().eval()(**candidate_inputs)
        expected = reference_module.Model().cuda().eval()(**reference_inputs)
        torch.cuda.synchronize()
    actual_f32, expected_f32 = actual.float(), expected.float()
    passed = torch.isclose(actual_f32, expected_f32, atol=ATOL, rtol=RTOL).all()
    max_abs_err = float((actual_f32 - expected_f32).abs().max())
    if not bool(passed):
        raise RuntimeError(
            "target correctness failed: "
            f"atol={ATOL} rtol={RTOL} max_abs_err={max_abs_err}"
        )
    print(
        json.dumps(
            {
                "status": "passed",
                "mode": "target",
                "shape_id": shape_id,
                "atol": ATOL,
                "rtol": RTOL,
                "max_abs_err": max_abs_err,
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("sm120-bf16", "target"))
    parser.add_argument("--shape-id", default="0")
    args = parser.parse_args()
    if args.mode == "sm120-bf16":
        sm120_bf16_smoke()
    else:
        target_smoke(args.shape_id)


if __name__ == "__main__":
    main()
