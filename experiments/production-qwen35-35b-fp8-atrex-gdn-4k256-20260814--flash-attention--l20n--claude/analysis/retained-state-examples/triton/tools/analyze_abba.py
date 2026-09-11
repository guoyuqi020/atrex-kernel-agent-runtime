"""Analyze an ABBA evaluate response (gateway-execute stdout JSON, local — no GPU).

ABBA responses carry `result.measurements`: one entry per run (side A/B x repeat),
each with a full `latency_us_by_shape` dict. This tool averages per side per shape,
prints overall speedup, correctness margins, per-shape ratios sorted, per-repeat
spread (to separate deterministic config effects from noise), tier geomeans by
baseline latency, and the B-side trial/artifact/result identities needed for
record-experiment / attempt-report.

Usage:
  python3 agent/optimizer/src/runtime_tools.py gateway-execute --request scratch/abba.json \
      > scratch/abba.out 2> scratch/abba.err   # keep stderr OUT of the JSON
  python3 tools/analyze_abba.py scratch/abba.out

Decision-gate helpers (Epoch-2 lesson: CUDA-graph timings are near-deterministic;
treat per-shape losses >2% with ~0% repeat spread as config defects, not noise):
  --gates KEEP_GEOMEAN KEEP_MAX_LOSS   e.g. --gates 1.01 0.02 prints PASS/FAIL
                                       for "geomean >= KEEP_GEOMEAN and no
                                       consistent loss worse than -KEEP_MAX_LOSS".
A loss is "consistent" only if every B repeat is slower than every A repeat by
more than the threshold (guards against A-side noise like shape 35's 3.1%).
"""
import json
import math
import sys


def load(path):
    raw = open(path).read()
    i = raw.rfind('EXIT:')
    return json.loads(raw[:i] if i >= 0 else raw)


def main():
    path = sys.argv[1]
    gates = None
    if '--gates' in sys.argv:
        gi = sys.argv.index('--gates')
        gates = (float(sys.argv[gi + 1]), float(sys.argv[gi + 2]))
    d = load(path)
    print('status:', d.get('status'))
    res = d.get('result') or {}
    comp = res.get('comparison') or {}
    print('comparison:', comp.get('method'), 'repeats', comp.get('repeats'))
    print('speedup (A/B):', res.get('speedup'), '| improvement_pct:', res.get('improvement_pct'))
    print('B kernel_trial_id:', d.get('kernel_trial_id'))
    print('B kernel_artifact_digest:', d.get('kernel_artifact_digest'))
    print('result_artifact_digest:', d.get('result_artifact_digest'))
    for side in ('baseline', 'candidate'):
        s = res.get(side) or {}
        c = s.get('correctness') or {}
        print(f"{side}: correct={s.get('correct')} {c.get('status')} "
              f"max_abs={c.get('max_abs_err')} max_rel={c.get('max_rel_err')} "
              f"geo={s.get('latency_us_geomean')}")

    meas = res.get('measurements') or []
    runs = {}   # sid -> {A0: lat, A1: lat, B0: lat, ...}
    for m in meas:
        key = f"{m.get('side')}{m.get('repeat')}"
        for sid, lat in (m.get('latency_us_by_shape') or {}).items():
            runs.setdefault(int(sid), {})[key] = lat
    if not runs:
        print('NO per-shape data found')
        return
    rows = []
    for sid, e in sorted(runs.items()):
        a = [v for k, v in e.items() if k.startswith('A')]
        b = [v for k, v in e.items() if k.startswith('B')]
        if not a or not b:
            continue
        A, B = sum(a) / len(a), sum(b) / len(b)
        asp = (max(a) - min(a)) / A * 100
        bsp = (max(b) - min(b)) / B * 100
        rows.append((sid, A, B, A / B, asp, bsp, a, b))
    ratios = [r[3] for r in rows]
    geo = math.exp(sum(map(math.log, ratios)) / len(ratios))
    print(f"\nshapes: {len(rows)} | per-shape geomean A/B: {geo:.4f} | "
          f"min {min(ratios):.4f} max {max(ratios):.4f}")
    wins = [r for r in rows if r[3] > 1.02]
    loss = [r for r in rows if r[3] < 0.98]
    print(f"wins>2%: {len(wins)} | losses>2%: {len(loss)}")
    rows.sort(key=lambda r: r[3])
    print('worst 8:')
    for r in rows[:8]:
        print(f"  sid={r[0]:>3} A={r[1]:8.1f} B={r[2]:8.1f} ratio={r[3]:.4f} "
              f"spread A{r[4]:.1f}%/B{r[5]:.1f}%")
    print('best 8:')
    for r in rows[-8:]:
        print(f"  sid={r[0]:>3} A={r[1]:8.1f} B={r[2]:8.1f} ratio={r[3]:.4f} "
              f"spread A{r[4]:.1f}%/B{r[5]:.1f}%")
    tiers = [(0, 30), (30, 100), (100, 300), (300, 1000), (1000, float('inf'))]
    print('tier geomeans (by baseline latency):')
    for lo, hi in tiers:
        t = [r[3] for r in rows if lo <= r[1] < hi]
        if t:
            g = math.exp(sum(map(math.log, t)) / len(t))
            print(f"  {lo:>5}-{hi if hi != float('inf') else 'inf':>5}: n={len(t):>2} "
                  f"geo={g:.4f} min={min(t):.4f}")
    if gates:
        keep_geo, keep_loss = gates
        consistent = [r for r in rows
                      if min(r[7]) > max(r[6]) * (1 + keep_loss)]
        ok = (res.get('speedup') or 0) >= keep_geo and not consistent
        print(f"\nGATES geomean>={keep_geo} & 0 consistent losses>{keep_loss*100:.0f}%: "
              f"{'PASS' if ok else 'FAIL'}")
        if consistent:
            print('  consistent losers:', [(r[0], round(r[3], 4)) for r in consistent])


if __name__ == '__main__':
    main()
