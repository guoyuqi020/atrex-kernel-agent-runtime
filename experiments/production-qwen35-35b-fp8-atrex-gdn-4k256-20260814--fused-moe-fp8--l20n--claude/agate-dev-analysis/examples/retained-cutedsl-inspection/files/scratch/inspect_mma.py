"""Dev probe 2: dump the SM120/FP8-relevant parts of cute.nvgpu.warp.mma and locate examples."""
import json
import os
import re

out = {}

mma_path = "/venv/lib/python3.12/site-packages/nvidia_cutlass_dsl/dsl_packages/cutlass/cute/nvgpu/warp/mma.py"
try:
    src = open(mma_path, encoding="utf-8").read()
    out["mma_py_lines"] = len(src.splitlines())
    # Class definitions with surrounding context
    class_info = []
    lines = src.splitlines()
    for i, ln in enumerate(lines):
        m = re.match(r"^class (\w+)", ln)
        if m:
            name = m.group(1)
            # docstring snippet
            snippet = "\n".join(lines[i:i + 14])
            class_info.append(f"line {i+1}: {snippet}")
    out["classes"] = class_info
except Exception as e:  # noqa
    out["mma_error"] = repr(e)

# blackwell_helpers: show SM120 + make_trivial_tiled_mma portions
try:
    bh = open("/venv/lib/python3.12/site-packages/nvidia_cutlass_dsl/dsl_packages/cutlass/utils/blackwell_helpers.py", encoding="utf-8").read()
    sel = []
    lines = bh.splitlines()
    keep = False
    for i, ln in enumerate(lines):
        if re.match(r"^(def|class) ", ln):
            keep = bool(re.search(r"SM120|sm120|make_trivial_tiled_mma|fp8|F8", ln))
        if keep:
            sel.append(ln)
        if len(sel) > 220:
            break
    out["blackwell_helpers_excerpt"] = "\n".join(sel)
except Exception as e:  # noqa
    out["bh_error"] = repr(e)

# /usr/local/cutlass listing
for base in ("/usr/local/cutlass",):
    if os.path.isdir(base):
        entries = []
        for root, dirs, files in os.walk(base):
            rel = os.path.relpath(root, base)
            depth = rel.count(os.sep)
            if "python" in rel.lower() or "CuTeDSL" in rel or depth <= 2:
                sub = sorted(dirs)[:60]
                pyfiles = sorted(f for f in files if f.endswith(".py"))[:60]
                entries.append(f"{rel}: dirs={sub} py={pyfiles}")
            if depth > 4:
                dirs[:] = []
            if len(entries) > 120:
                break
        out["cutlass_tree"] = entries

print(json.dumps(out, indent=2))
