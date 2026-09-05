#!/usr/bin/env python3
"""Reconstruct and plot best-so-far Kernel latency by search depth.

Runtime Attempts at the same serial distance from Bootstrap are one point even
when multiple Trajectories or Agent Branches execute that layer in parallel.
AKA uses Episode number as its serial distance. Point zero is the archived
incumbent/baseline measurement. A curve only moves when the corresponding
system retains a lower-latency Kernel.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import sqlite3
from pathlib import Path
from typing import Any


EXPERIMENT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_ARCHIVE = Path(
    "/Users/guoyuqi/atrex-runs/"
    "production-qwen35-35b-fp8-atrex-gdn-4k256-20260814--"
    "fused-moe-fp8--l20n--claude"
)
SUMMARY_PATH = EXPERIMENT_DIR / "analysis/checkpoint-summary.json"
OUTPUT_DIR = Path(__file__).resolve().parent

DSLS = ("cuda", "triton", "cutedsl")
DISPLAY_DSL = {"cuda": "CUDA", "triton": "Triton", "cutedsl": "CuteDSL"}
CURVE_ORDER = (
    "AKA best of two",
    "Isolated best of two",
    "Pooled",
    "Retained",
    "Evolve",
)
STYLES = {
    "AKA best of two": {"color": "#111827", "dash": ""},
    "Isolated best of two": {"color": "#2563EB", "dash": "12 8"},
    "Pooled": {"color": "#F59E0B", "dash": "4 7"},
    "Retained": {"color": "#059669", "dash": ""},
    "Evolve": {"color": "#C026D3", "dash": "16 7 4 7"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def runtime_registry(archive: Path) -> Path:
    return (
        archive
        / "runtime/workspace-full-20260902.unpacked/production/control-l20n"
        / "state/registry.sqlite"
    )


def runtime_lineages() -> dict[tuple[str, str], str]:
    summary = json.loads(SUMMARY_PATH.read_text())
    entries = summary["full_budget"]["runtime"]
    return {
        (entry["dsl"], entry["arm"]): entry["lineage_id"]
        for entry in entries.values()
    }


def runtime_curve(db: sqlite3.Connection, lineage_id: str) -> list[dict[str, Any]]:
    baseline = db.execute(
        """
        SELECT k.latency_us
        FROM epochs AS e
        JOIN kernel_revisions AS k ON k.id = e.starting_kernel_revision_id
        WHERE e.lineage_id = ?
        ORDER BY e.number
        LIMIT 1
        """,
        (lineage_id,),
    ).fetchone()
    if baseline is None or baseline[0] is None:
        raise ValueError(f"missing Runtime baseline for {lineage_id}")

    rows = db.execute(
        """
        SELECT
            e.number,
            a.branch,
            a.challenger_ordinal,
            a.trajectory_ordinal,
            a.iteration_ordinal,
            a.id,
            a.completed_at,
            a.accepted_as_branch_best,
            k.latency_us
        FROM attempts AS a
        JOIN epochs AS e ON e.id = a.epoch_id
        LEFT JOIN kernel_revisions AS k ON k.id = a.output_kernel_revision_id
        WHERE e.lineage_id = ? AND a.status = 'completed'
        ORDER BY e.number, a.iteration_ordinal, a.branch,
                 a.challenger_ordinal, a.trajectory_ordinal, a.id
        """,
        (lineage_id,),
    ).fetchall()

    best = float(baseline[0])
    curve: list[dict[str, Any]] = [
        {
            "step": 0,
            "latency_us": best,
            "source": "bootstrap",
            "lineage_id": lineage_id,
        }
    ]
    layers: dict[tuple[int, int], list[tuple[Any, ...]]] = {}
    for row in rows:
        layers.setdefault((int(row[0]), int(row[4])), []).append(row)

    for step, ((epoch, iteration), layer) in enumerate(sorted(layers.items()), start=1):
        retained_latencies = [
            float(row[8]) for row in layer if row[7] and row[8] is not None
        ]
        layer_best = min(retained_latencies, default=None)
        improved = bool(layer_best is not None and layer_best < best)
        if improved and layer_best is not None:
            best = layer_best
        curve.append(
            {
                "step": step,
                "latency_us": best,
                "source": "runtime_parallel_attempt_layer",
                "lineage_id": lineage_id,
                "epoch": epoch,
                "attempt_in_trajectory": iteration,
                "parallel_width": len(layer),
                "attempts": [
                    {
                        "attempt_id": row[5],
                        "branch": row[1],
                        "challenger_ordinal": row[2],
                        "trajectory": row[3],
                        "completed_at": row[6],
                        "retained": bool(row[7]),
                        "candidate_latency_us": row[8],
                    }
                    for row in layer
                ],
                "improved_best_so_far": improved,
            }
        )
    return curve


def aka_curve(root: Path, dsl: str) -> list[dict[str, Any]]:
    paths = sorted(
        (
            root
            / f"kernel_opt_fused_moe_fp8_{dsl}_l20n_production"
            / ".atrex_long_horizon/episodes"
        ).glob("e*/attempt.json")
    )
    attempts = [json.loads(path.read_text()) for path in paths]
    attempts.sort(key=lambda item: int(item["episode"]))
    if not attempts:
        raise ValueError(f"missing AKA Episodes for {root.name}/{dsl}")

    baseline = next(
        (
            float(attempt["verification"]["incumbent_latency_us"])
            for attempt in attempts
            if (attempt.get("verification") or {}).get("incumbent_latency_us") is not None
        ),
        None,
    )
    if baseline is None:
        raise ValueError(f"missing AKA incumbent for {root.name}/{dsl}")

    best = baseline
    curve: list[dict[str, Any]] = [
        {"step": 0, "latency_us": best, "source": "incumbent", "run": root.name}
    ]
    for attempt in attempts:
        verification = attempt.get("verification") or {}
        candidate = verification.get("candidate_latency_us")
        improved = bool(attempt.get("accepted") and candidate is not None and candidate < best)
        if improved:
            best = float(candidate)
        curve.append(
            {
                "step": int(attempt["episode"]),
                "latency_us": best,
                "source": "aka_episode",
                "run": root.name,
                "status": attempt.get("status"),
                "accepted": bool(attempt.get("accepted")),
                "improved_best_so_far": improved,
            }
        )
    return curve


def best_of_two(
    left: list[dict[str, Any]], right: list[dict[str, Any]], left_name: str, right_name: str
) -> list[dict[str, Any]]:
    if len(left) != len(right):
        raise ValueError(f"unaligned best-of-two curves: {len(left)} != {len(right)}")
    result = []
    for lhs, rhs in zip(left, right, strict=True):
        if lhs["step"] != rhs["step"]:
            raise ValueError("best-of-two steps are not aligned")
        winner_name, winner = (left_name, lhs) if lhs["latency_us"] <= rhs["latency_us"] else (right_name, rhs)
        result.append(
            {
                "step": lhs["step"],
                "latency_us": winner["latency_us"],
                "selected_run": winner_name,
            }
        )
    return result


def fmt_latency(value: float) -> str:
    if value >= 100_000:
        return f"{value / 1000:.0f}k"
    if value >= 10_000:
        return f"{value / 1000:.1f}k"
    if value >= 1_000:
        return f"{value / 1000:.2f}k"
    return f"{value:.0f}"


def linear_ticks(low: float, high: float, count: int = 6) -> list[float]:
    span = high - low
    rough = span / max(1, count - 1)
    magnitude = 10 ** math.floor(math.log10(rough))
    normalized = rough / magnitude
    unit = (1 if normalized <= 1 else 2 if normalized <= 2 else 5 if normalized <= 5 else 10) * magnitude
    first = math.floor(low / unit) * unit
    ticks = []
    value = first
    while value <= high + unit:
        if value >= low - unit * 0.01:
            ticks.append(value)
        value += unit
    return ticks


def log_ticks(low: float, high: float) -> list[float]:
    ticks = []
    start = math.floor(math.log10(low)) - 1
    end = math.ceil(math.log10(high)) + 1
    for exponent in range(start, end + 1):
        for multiplier in (1, 2, 5):
            value = multiplier * (10**exponent)
            if low <= value <= high:
                ticks.append(float(value))
    return ticks


def points_attribute(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def svg_plot(dsl: str, curves: dict[str, list[dict[str, Any]]], output: Path) -> None:
    width, height = 1600, 940
    left, right, top, bottom = 155, 105, 185, 120
    plot_width, plot_height = width - left - right, height - top - bottom
    max_step = max(point["step"] for curve in curves.values() for point in curve)
    values = [float(point["latency_us"]) for curve in curves.values() for point in curve]
    raw_min, raw_max = min(values), max(values)
    use_log = raw_max / raw_min >= 3
    if use_log:
        low, high = raw_min * 0.88, raw_max * 1.15
        y0, y1 = math.log10(low), math.log10(high)
        ticks = log_ticks(low, high)
        transform_y = lambda value: top + (y1 - math.log10(value)) / (y1 - y0) * plot_height
    else:
        padding = (raw_max - raw_min) * 0.12
        low, high = raw_min - padding, raw_max + padding
        ticks = linear_ticks(low, high)
        transform_y = lambda value: top + (high - value) / (high - low) * plot_height

    transform_x = lambda step: left + step / max_step * plot_width
    out: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif;fill:#111827}</style>',
        f'<text x="{left}" y="68" font-size="38" font-weight="700">{DISPLAY_DSL[dsl]} — Best-so-far Kernel Latency</text>',
        f'<text x="{left}" y="108" font-size="21" fill="#4B5563">Bootstrap / incumbent = 0 · Parallel Attempts at the same depth count once</text>',
    ]

    for tick in ticks:
        y = transform_y(tick)
        out.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#E5E7EB" stroke-width="1"/>')
        out.append(f'<text x="{left - 18}" y="{y + 7:.2f}" text-anchor="end" font-size="19" fill="#4B5563">{fmt_latency(tick)}</text>')
    for tick in range(0, max_step + 1, 5):
        x = transform_x(tick)
        out.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}" stroke="#F3F4F6" stroke-width="1"/>')
        out.append(f'<text x="{x:.2f}" y="{top + plot_height + 36}" text-anchor="middle" font-size="19" fill="#4B5563">{tick}</text>')

    out.extend(
        [
            f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#6B7280" stroke-width="2"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#6B7280" stroke-width="2"/>',
            f'<text x="{left + plot_width / 2:.2f}" y="{height - 32}" text-anchor="middle" font-size="22" font-weight="600">Optimization Distance from Bootstrap</text>',
            f'<text x="42" y="{top + plot_height / 2:.2f}" text-anchor="middle" font-size="22" font-weight="600" transform="rotate(-90 42 {top + plot_height / 2:.2f})">Latency (µs){" · log scale" if use_log else ""}</text>',
        ]
    )

    legend_x, legend_y = width - 555, 40
    for index, name in enumerate(CURVE_ORDER):
        style = STYLES[name]
        y = legend_y + index * 27
        dash = f' stroke-dasharray="{style["dash"]}"' if style["dash"] else ""
        out.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 52}" y2="{y}" stroke="{style["color"]}" stroke-width="5"{dash}/>')
        final = curves[name][-1]["latency_us"]
        label = html.escape(f"{name}  ·  {final:,.1f} µs")
        out.append(f'<text x="{legend_x + 67}" y="{y + 7}" font-size="19" font-weight="600">{label}</text>')

    for name in CURVE_ORDER:
        curve = curves[name]
        style = STYLES[name]
        step_points: list[tuple[float, float]] = []
        previous = None
        improved_points: list[tuple[float, float]] = []
        for point in curve:
            x, y = transform_x(point["step"]), transform_y(point["latency_us"])
            if previous is not None:
                step_points.append((x, previous[1]))
            step_points.append((x, y))
            if previous is None or point["latency_us"] < previous[2] - 1e-12:
                improved_points.append((x, y))
            previous = (x, y, point["latency_us"])
        dash = f' stroke-dasharray="{style["dash"]}"' if style["dash"] else ""
        out.append(
            f'<polyline points="{points_attribute(step_points)}" fill="none" stroke="{style["color"]}" '
            f'stroke-width="4" stroke-linejoin="round" stroke-linecap="round"{dash}/>'
        )
        for x, y in improved_points:
            out.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" fill="#FFFFFF" stroke="{style["color"]}" stroke-width="3"/>')

    out.append("</svg>\n")
    output.write_text("\n".join(out))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    registry = runtime_registry(args.archive_root)
    if not registry.exists():
        raise FileNotFoundError(registry)

    lineages = runtime_lineages()
    aka_roots = {
        "AKA-1": args.archive_root / "AKA/atrex-runs.unpacked/atrex-runs",
        "AKA-2": args.archive_root / "AKA/atrex-runs2.unpacked/atrex-runs2",
    }
    payload: dict[str, Any] = {
        "metric": "best_so_far_retained_kernel_latency_us",
        "x_axis": "serial optimization distance from Bootstrap",
        "parallel_attempt_policy": "Attempts with the same Epoch and iteration ordinal count once",
        "bootstrap_step": 0,
        "runtime_registry": str(registry),
        "dsl": {},
    }

    with sqlite3.connect(f"file:{registry}?mode=ro", uri=True) as db:
        for dsl in DSLS:
            isolated_one = runtime_curve(db, lineages[dsl, "ablation-isolated-01"])
            isolated_two = runtime_curve(db, lineages[dsl, "ablation-isolated-02"])
            aka_one = aka_curve(aka_roots["AKA-1"], dsl)
            aka_two = aka_curve(aka_roots["AKA-2"], dsl)
            curves = {
                "AKA best of two": best_of_two(aka_one, aka_two, "AKA-1", "AKA-2"),
                "Isolated best of two": best_of_two(
                    isolated_one, isolated_two, "isolated-01", "isolated-02"
                ),
                "Pooled": runtime_curve(db, lineages[dsl, "ablation-pooled"]),
                "Retained": runtime_curve(db, lineages[dsl, "ablation-retained"]),
                "Evolve": runtime_curve(db, lineages[dsl, "evolve"]),
            }
            payload["dsl"][dsl] = {"curves": curves}
            svg_plot(dsl, curves, args.output_dir / f"{dsl}.svg")

    (args.output_dir / "curves.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    main()
