#!/usr/bin/env python3
"""Build the byte-faithful agate eval request that the AKA runtime would have sent.

Mirrors src/atrex_runtime/gateway/agate.py::_build_request for GatewayOperation.EVALUATE:
  reference = {operator, reference_py, input_py, shapes, metadata, roofline(stripped)}
  options   = contract options with optimizer overrides (num_correctness_cases=5, bench_iters=100)
  spec      = {"languages": ["cuda"]}   (injected-DSL arm)
  contract  = requirements [], deps_mode freeze_installed, mode full, lock_clocks True,
              harness atrex_bench, env_vars {} -> omitted

Usage:
  python3 make_payload.py --reference-dir reference     --out eval-full90.json
  python3 make_payload.py --reference-dir reference-s58 --out eval-shape58.json
"""
import argparse
import json
from pathlib import Path

from atrex_gateway_client import build_eval_request_from_content

OPERATOR = "qwen35_35b_fp8_atrex_gdn_4319x256_flash_attention"
GPU = "L20N"  # -> sm_120, "Agate hardware resolved: L20N" (bootstrap.log)

ap = argparse.ArgumentParser()
ap.add_argument("--candidate", default=str(Path(__file__).parent / "kernel.py"))
ap.add_argument("--reference-dir", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--name", default=f"{OPERATOR}_repro-cuda-core")
ap.add_argument("--idempotency-key", default=None,
                help="unique per draw, e.g. repro-cudacore-<n> (runtime always sends one)")
args = ap.parse_args()

refd = Path(args.reference_dir)
reference = {
    "operator": OPERATOR,
    "reference_py": (refd / "reference.py").read_text(encoding="utf-8"),
    "input_py": (refd / "input.py").read_text(encoding="utf-8"),
    "shapes": json.loads((refd / "shapes.json").read_text(encoding="utf-8")),
}
meta = refd / "metadata.json"
if meta.exists():
    reference["metadata"] = json.loads(meta.read_text(encoding="utf-8"))
roof = refd / "roofline.json"
if roof.exists():
    reference["roofline"] = json.loads(roof.read_text(encoding="utf-8"))

payload = build_eval_request_from_content(
    Path(args.candidate).read_text(encoding="utf-8"),
    reference,
    GPU,
    name=args.name,
    spec_fields={"languages": ["cuda"]},
    # gate_policy.optimizer: correctness_cases=5, bench_iters=100; contract: atol/rtol/timeout_s
    options={
        "num_correctness_cases": 5,
        "bench_iters": 100,
        "atol": 0.01,
        "rtol": 0.05,
        "timeout_s": 600,
    },
    env_vars=None,               # contract env_vars == {}
    requirements=None,           # contract requirements == []
    deps_mode="freeze_installed",
    mode="full",
    lock_clocks=True,
    harness="atrex_bench",
    atrex_bench_version=None,    # contract carries none; gateway pins its own
    runner_overrides=None,       # contract runner_overrides == {}
    idempotency_key=args.idempotency_key,
)

out = Path(args.out)
out.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"wrote {out} ({out.stat().st_size} bytes); shapes={len(reference['shapes'])}")
