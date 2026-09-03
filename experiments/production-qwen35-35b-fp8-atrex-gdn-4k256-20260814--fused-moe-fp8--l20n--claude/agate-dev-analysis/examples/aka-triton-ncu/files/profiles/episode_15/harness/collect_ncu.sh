#!/bin/bash
# Episode-15 ncu collection with stdout transport (the remote pod has the ncu
# CLI but not the ncu_report python module, and the dev artifact frame is
# elided in inline transport, so results are parsed remotely and printed as a
# compact table on stdout). Same pattern as episode-13/14 collect_ncu.sh.
#
# Usage (the trailing profile_driver.py token is load-bearing: the sandbox
# detects the profile command by argv basename and only then injects the
# private real shape and prefixes the PROFILE_* controls):
#   bash profiles/episode_15/harness/collect_ncu.sh <output-dir> time "" profile_driver.py
#   bash profiles/episode_15/harness/collect_ncu.sh <output-dir> full <kernel-regex> profile_driver.py
set -u
OUT=$1
MODE=${2:-time}
REGEX=${3:-}
DRIVER=${4:-profile_driver.py}
mkdir -p "$OUT"

export NCU_DEBUG=1
if [ "$MODE" = "full" ]; then
  ncu --set full -k "$REGEX" --launch-skip 4 --launch-count 4 \
      --csv python "$DRIVER" > "$OUT/ncu_full.csv" 2> "$OUT/ncu_full.err"
else
  ncu --metrics gpu__time_duration.sum,launch__registers_per_thread,launch__grid_size,launch__block_size,launch__waves_per_multiprocessor,launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem,launch__shared_mem_per_block_static,launch__shared_mem_per_block_dynamic \
      --launch-skip 16 --launch-count 48 \
      --csv python "$DRIVER" > "$OUT/ncu_time.csv" 2> "$OUT/ncu_time.err"
fi
echo "[collect_ncu] mode=$MODE done"
if [ "$MODE" = "full" ]; then
  echo "[collect_ncu] err-tail:"; tail -c 1200 "$OUT/ncu_full.err" 2>/dev/null
else
  echo "[collect_ncu] err-tail:"; tail -c 1200 "$OUT/ncu_time.err" 2>/dev/null
fi

python3 - "$OUT" "$MODE" <<'PYEOF'
import csv, sys, os, collections

outdir, mode = sys.argv[1], sys.argv[2]
path = os.path.join(outdir, "ncu_full.csv" if mode == "full" else "ncu_time.csv")
if not os.path.isfile(path) or os.path.getsize(path) == 0:
    print("[collect_ncu] PARSE-FAIL: no csv")
    sys.exit(0)

header, rows = None, []
with open(path, newline="") as fh:
    for row in csv.reader(fh):
        if not row:
            continue
        first = row[0].strip().strip('"')
        if header is None:
            if first == "ID":
                header = [c.strip().strip('"') for c in row]
            continue
        rows.append(row)

if header is None:
    print("[collect_ncu] PARSE-FAIL: no header")
    sys.exit(0)

if os.environ.get("NCU_DEBUG"):
    print("[collect_ncu] HEADER ncol=%d nrows=%d: %s" % (len(header), len(rows), "|".join(header[:60])))
    for r in rows[:3]:
        print("[collect_ncu] ROW: %s" % "|".join(x[:22] for x in r[:60]))

idx = {name: i for i, name in enumerate(header)}

def col(row, name):
    i = idx.get(name)
    if i is None or i >= len(row):
        return None
    return row[i]

def num(value):
    if value is None:
        return None
    v = value.strip().strip('"').replace(",", "")
    if v in ("", "n/a", "N/A"):
        return None
    try:
        return float(v)
    except ValueError:
        return None

agg = collections.OrderedDict()
SKIP = {"ID", "Kernel Name", "Kernel", "Block Size", "Grid Size", "Device",
        "Context", "Stream", "Section Name", "Metric Name", "Metric Unit",
        "CC", "Profile Duration", "Elapsed Time", "Timestamp", "Process ID",
        "Process Name", "Host Name", "CUDA Driver", "Location"}
LONG = "Metric Name" in idx and "Metric Value" in idx
for row in rows:
    name = (col(row, "Kernel Name") or "?").strip().strip('"')
    base = name.split("(")[0].strip()
    if base not in agg:
        agg[base] = collections.defaultdict(list)
    if LONG:
        metric = (col(row, "Metric Name") or "").strip().strip('"')
        v = num(col(row, "Metric Value"))
        if metric and v is not None:
            agg[base][metric].append(v)
        continue
    for metric in header:
        if metric in SKIP:
            continue
        v = num(col(row, metric))
        if v is not None:
            agg[base][metric].append(v)

WANT_SUBSTR = [
    "gpu__time_duration", "registers_per_thread", "grid_size", "block_size",
    "waves_per_multiprocessor", "occupancy_limit", "shared_mem_per_block",
    "pipe_tensor", "pipe_hmma", "pipe_fma", "pipe_alu", "pipe_lsu",
    "hmma_cycles_active", "sm__inst_executed_pipe",
    "dram__throughput", "dram__bytes", "lts__t_sector_hit_rate",
    "l1tex__throughput", "lts__throughput", "sm__throughput",
    "smsp__inst_executed.sum", "issue_stalled", "warps_issue_stalled",
    "warps_active", "occupancy", "local", "spill",
    "sass_thread_inst_executed_op_hmma", "tensor",
    "memory_throughput", "compute_throughput", "achieved_active",
]

def wanted(metric):
    m = metric.lower()
    return any(s in m for s in WANT_SUBSTR)

SECTIONS_FULL = {
    "GPU Speed Of Light Throughput", "Occupancy", "Warp State Statistics",
    "Scheduler Statistics", "Launch Statistics", "Memory Workload Analysis",
    "Compute Workload Analysis",
}
long_sections = collections.defaultdict(dict)
if LONG and mode == "full":
    for row in rows:
        name = (col(row, "Kernel Name") or "?").strip().strip('"')
        base = name.split("(")[0].strip()
        section = (col(row, "Section Name") or "").strip().strip('"')
        metric = (col(row, "Metric Name") or "").strip().strip('"')
        unit = (col(row, "Metric Unit") or "").strip().strip('"')
        v = num(col(row, "Metric Value"))
        if not metric or v is None:
            continue
        long_sections[(base, section)].setdefault(metric, []).append((v, unit))

if LONG and mode == "full":
    gridinfo = {}
    for row in rows:
        base = (col(row, "Kernel Name") or "?").strip().strip('"').split("(")[0].strip()
        if base not in gridinfo:
            gridinfo[base] = ((col(row, "Grid Size") or "?"), (col(row, "Block Size") or "?"))
    for (base, section), metrics in long_sections.items():
        if section not in SECTIONS_FULL and "stall" not in section.lower():
            continue
        print(f"== {base} :: {section} grid={gridinfo.get(base, ('?','?'))}")
        for metric, values in sorted(metrics.items()):
            vals = [v for v, _ in values]
            unit = values[0][1]
            mean = sum(vals) / len(vals)
            print(f"  {metric}: mean={mean:.5g} max={max(vals):.5g} n={len(vals)} {unit}")
else:
    for base, metrics in agg.items():
        nlaunch = max((len(v) for v in metrics.values()), default=0)
        dur = metrics.get("gpu__time_duration.sum")
        dur_str = ""
        if dur:
            dur_str = f" total={sum(dur):.1f}us mean={sum(dur)/len(dur):.2f}us n={len(dur)}"
        print(f"KERNEL {base}{dur_str} launches={nlaunch}")
        keep = []
        for metric, values in metrics.items():
            if not wanted(metric):
                continue
            keep.append((metric, sum(values) / len(values), max(values), len(values)))
        keep.sort(key=lambda t: t[0])
        for metric, mean, mx, n in keep:
            print(f"  {metric}: mean={mean:.4g} max={mx:.4g} n={n}")
print("[collect_ncu] PARSE-OK")
PYEOF
