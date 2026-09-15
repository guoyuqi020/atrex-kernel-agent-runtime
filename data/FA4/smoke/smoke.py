#!/usr/bin/env python3
"""Agate smoke checks for the pristine upstream kernel and target ABI."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
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


def upstream_p128_smoke() -> None:
    """Compile and execute exactly the capability present in upstream FA4."""
    from flash_attn.cute import flash_attn_varlen_func

    torch.manual_seed(1)
    device = torch.device("cuda", 0)
    q = (torch.randn(128, 16, 256, dtype=torch.bfloat16, device=device) * 0.1).to(
        torch.float8_e4m3fn
    )
    k = (torch.randn(1, 128, 1, 256, dtype=torch.bfloat16, device=device) * 0.1).to(
        torch.float8_e4m3fn
    )
    v = (torch.randn_like(k, dtype=torch.bfloat16) * 0.1).to(torch.float8_e4m3fn)
    cu_q = torch.tensor([0, 128], dtype=torch.int32, device=device)
    page_table = torch.tensor([[0]], dtype=torch.int32, device=device)
    with torch.inference_mode():
        out = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=None,
            max_seqlen_q=128,
            max_seqlen_k=128,
            page_table=page_table,
            softmax_scale=256**-0.5,
            causal=True,
            num_splits=1,
            pack_gqa=False,
        )
        if isinstance(out, (tuple, list)):
            out = out[0]
        torch.cuda.synchronize()
    if tuple(out.shape) != tuple(q.shape) or not torch.isfinite(out.float()).all():
        raise RuntimeError(f"invalid upstream smoke output: {out.shape=} {out.dtype=}")
    print(json.dumps({"status": "passed", "mode": "upstream-p128", "shape": list(out.shape)}))


def target_smoke(shape_id: str) -> None:
    """Run one current C05-contract shape; R0 is expected to expose its gaps."""
    input_module = _load("aka_r0_input", ROOT / "reference/input.py")
    reference_module = _load("aka_r0_reference", ROOT / "reference/reference.py")
    candidate_module = _load("aka_r0_candidate", ROOT / "kernel.py")
    payload = json.loads((ROOT / "reference/shapes.json").read_text())
    entry = next(item for item in payload["shapes"] if str(item["id"]) == shape_id)
    torch.manual_seed(1)
    inputs = input_module._make_inputs(**entry["input_kwargs"])
    with torch.inference_mode():
        actual = candidate_module.Model().cuda().eval()(**inputs)
        expected = reference_module.Model().cuda().eval()(**inputs)
        torch.cuda.synchronize()
    rel_l2 = float(
        torch.linalg.vector_norm((actual.float() - expected.float()).reshape(-1))
        / torch.linalg.vector_norm(expected.float().reshape(-1)).clamp_min(1e-12)
    )
    if rel_l2 > 0.05:
        raise RuntimeError(f"target correctness failed: relative_l2={rel_l2}")
    print(json.dumps({"status": "passed", "mode": "target", "shape_id": shape_id, "relative_l2": rel_l2}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("upstream-p128", "target"))
    parser.add_argument("--shape-id", default="0")
    args = parser.parse_args()
    if args.mode == "upstream-p128":
        upstream_p128_smoke()
    else:
        target_smoke(args.shape_id)


if __name__ == "__main__":
    main()
