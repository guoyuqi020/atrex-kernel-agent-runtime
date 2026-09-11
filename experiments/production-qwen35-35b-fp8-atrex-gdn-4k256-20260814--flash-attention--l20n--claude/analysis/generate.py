#!/usr/bin/env python3
"""Reconstruct the Flash Attention ablation report evidence from the archive."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import html
import json
import math
import re
import shutil
import sqlite3
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTED_ROOT = Path("/tmp/atrex-runs-inspect.E7Gttv/production")
RUN_NAME = (
    "production-qwen35-35b-fp8-atrex-gdn-4k256-20260814--"
    "flash-attention--l20n--claude"
)
DSLS = ("cuda", "triton", "cutedsl")
DISPLAY_DSL = {"cuda": "CUDA", "triton": "Triton", "cutedsl": "CuteDSL"}
COMPARISON_ORDER = (
    "AKA best of two",
    "Isolated best of two",
    "Pool-3",
    "Pool-retained-3",
    "Retained best of two",
    "Evolve-3",
)
STYLES = {
    "AKA best of two": ("#DC2626", "2 6"),
    "Isolated best of two": ("#2563EB", "12 8"),
    "Pool-3": ("#F59E0B", "4 7"),
    "Pool-retained-3": ("#059669", ""),
    "Retained best of two": ("#0F766E", "12 5 3 5"),
    "Evolve-3": ("#C026D3", "16 7 4 7"),
}
TOKEN_COLUMNS = (
    "uncached_input_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "output_tokens",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extracted-root", type=Path, default=DEFAULT_EXTRACTED_ROOT)
    parser.add_argument("--archive", type=Path, default=Path.home() / "atrex-runs/workspace-full-20260909.tar.zst")
    parser.add_argument("--aka-root", type=Path, default=Path.home() / "atrex-runs")
    parser.add_argument("--aka-extracted-root", type=Path, default=Path("/tmp/atrex-flash-aka"))
    parser.add_argument(
        "--runtime-baseline-artifacts",
        type=Path,
        default=Path("/tmp/atrex-flash-runtime-baselines/production/control-l20n/state/artifacts/sha256"),
    )
    return parser.parse_args()


def _aka_archive_dir(root: Path, run_number: int) -> Path:
    matches = sorted(root.glob(f"*flash-attention*standalone-run{run_number}"))
    if len(matches) != 1:
        raise ValueError(
            f"expected one Flash Attention standalone run{run_number} archive under {root}, "
            f"found {len(matches)}"
        )
    return matches[0]


def prepare_aka_inputs(root: Path, extracted_root: Path) -> dict[int, dict[str, Path]]:
    """Extract the two small AKA archives when their cached trees are absent."""

    result: dict[int, dict[str, Path]] = {}
    for run_number in (7, 8):
        archive_dir = _aka_archive_dir(root, run_number)
        workspace = extracted_root / f"run{run_number}" / f"atrex-runs{run_number}"
        traces = extracted_root / f"run{run_number}-traces" / "claude-traces"
        if not workspace.is_dir():
            workspace.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    "tar",
                    "--use-compress-program=unzstd",
                    "-xf",
                    str(archive_dir / "workspace-full-20260909.tar.zst"),
                    "-C",
                    str(workspace.parent),
                ],
                check=True,
            )
        if not traces.is_dir():
            traces.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    "tar",
                    "--use-compress-program=unzstd",
                    "-xf",
                    str(archive_dir / "claude-traces-redacted-20260909.tar.zst"),
                    "-C",
                    str(traces.parent),
                ],
                check=True,
            )
        result[run_number] = {
            "archive": archive_dir,
            "workspace": workspace,
            "traces": traces,
            "manifest": archive_dir / "MANIFEST.json",
        }
    return result


def arm_catalog(campaign_results: dict[str, Any]) -> list[dict[str, Any]]:
    catalog: list[dict[str, Any]] = []
    for dsl, item in campaign_results["dsls"].items():
        entries = [
            {
                "arm": item["arm"],
                "campaign_id": item["campaign_id"],
                "result": item["result"],
                "attempts_per_trajectory": 3,
                "trajectories_per_branch": 1,
                "ephemeral_agent_state": False,
                "challenger_count": 1,
            }
        ] + list(item["ablation"])
        for entry in entries:
            lineage = entry["result"]["lineages"][0]
            catalog.append(
                {
                    "dsl": dsl,
                    "arm": entry["arm"],
                    "campaign_id": entry["campaign_id"],
                    "lineage_id": lineage["lineage_id"],
                    "attempts_per_trajectory": entry.get("attempts_per_trajectory", 3),
                    "trajectories_per_branch": entry.get("trajectories_per_branch", 1),
                    "ephemeral_agent_state": entry.get("ephemeral_agent_state", False),
                    "challenger_count": entry.get("challenger_count", 0),
                }
            )
    return catalog


def scalar(db: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> Any:
    row = db.execute(sql, params).fetchone()
    return None if row is None else row[0]


def usage(
    db: sqlite3.Connection,
    campaign_ids: Iterable[str],
    *,
    roles: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    ids = tuple(campaign_ids)
    placeholders = ",".join("?" for _ in ids)
    role_filter = ""
    params: tuple[Any, ...] = ids
    if roles:
        role_filter = f" AND role IN ({','.join('?' for _ in roles)})"
        params += roles
    rows = db.execute(
        f"""
        SELECT role, status,
               COALESCE(uncached_input_tokens, 0),
               COALESCE(cache_read_tokens, 0),
               COALESCE(cache_write_tokens, 0),
               COALESCE(output_tokens, 0)
        FROM worker_sessions WHERE campaign_id IN ({placeholders}){role_filter}
        """,
        params,
    ).fetchall()
    totals = {name: 0 for name in TOKEN_COLUMNS}
    by_role: dict[str, dict[str, Any]] = {}
    statuses: Counter[str] = Counter()
    for role, status, *values in rows:
        statuses[status] += 1
        item = by_role.setdefault(
            role,
            {"sessions": 0, **{name: 0 for name in TOKEN_COLUMNS}},
        )
        item["sessions"] += 1
        for name, value in zip(TOKEN_COLUMNS, values):
            item[name] += int(value)
            totals[name] += int(value)
    total = sum(totals.values())
    return {
        "sessions": len(rows),
        "session_statuses": dict(sorted(statuses.items())),
        **totals,
        "total_tokens": total,
        "by_role": by_role,
    }


def lineage_summary(db: sqlite3.Connection, entry: dict[str, Any]) -> dict[str, Any]:
    lineage_id = entry["lineage_id"]
    row = db.execute(
        """
        SELECT k.latency_us, k.id, k.artifact_digest, k.gateway_result_digest,
               k.produced_by_attempt_id
        FROM lineages l JOIN kernel_revisions k ON k.id=l.best_kernel_revision_id
        WHERE l.id=?
        """,
        (lineage_id,),
    ).fetchone()
    baseline = float(
        scalar(
            db,
            """
            SELECT k.latency_us FROM lineage_kernel_versions v
            JOIN kernel_revisions k ON k.id=v.kernel_revision_id
            WHERE v.lineage_id=? AND v.revision_number=0
            """,
            (lineage_id,),
        )
    )
    attempts = db.execute(
        """
        SELECT a.status, a.accepted_as_branch_best, k.correct
        FROM attempts a JOIN epochs e ON e.id=a.epoch_id
        LEFT JOIN kernel_revisions k ON k.id=a.output_kernel_revision_id
        WHERE e.lineage_id=?
        """,
        (lineage_id,),
    ).fetchall()
    counts = Counter(status for status, _, _ in attempts)
    best_attempt = None
    if row[4] is not None:
        best_attempt_row = db.execute(
            """
            SELECT e.number,a.branch,a.challenger_ordinal,a.trajectory_ordinal,
                   a.iteration_ordinal,a.attempt_report_digest,a.runtime_state_digest
            FROM attempts a JOIN epochs e ON e.id=a.epoch_id WHERE a.id=?
            """,
            (row[4],),
        ).fetchone()
        if best_attempt_row:
            best_attempt = {
                "attempt_id": row[4],
                "epoch": best_attempt_row[0],
                "branch": best_attempt_row[1],
                "challenger_ordinal": best_attempt_row[2],
                "trajectory": best_attempt_row[3],
                "iteration": best_attempt_row[4],
                "attempt_report_digest": best_attempt_row[5],
                "runtime_state_digest": best_attempt_row[6],
            }
    result = {
        **entry,
        "baseline_latency_us": baseline,
        "final_latency_us": float(row[0]),
        "speedup_over_baseline": baseline / float(row[0]),
        "best_kernel_revision_id": row[1],
        "best_kernel_artifact_digest": row[2],
        "best_result_artifact_digest": row[3],
        "best_attempt": best_attempt,
        "attempts": len(attempts),
        "attempt_statuses": dict(sorted(counts.items())),
        "correct_outputs": sum(correct == 1 for _, _, correct in attempts),
        "incorrect_outputs": sum(correct == 0 for _, _, correct in attempts),
        "no_output": sum(correct is None for _, _, correct in attempts),
        "accepted_outputs": sum(bool(accepted) for _, accepted, _ in attempts),
    }
    result["usage"] = usage(
        db, [entry["campaign_id"]], roles=("optimizer", "evolver")
    )
    return result


def runtime_curve(db: sqlite3.Connection, lineage_id: str) -> list[dict[str, Any]]:
    baseline = float(
        scalar(
            db,
            """
            SELECT k.latency_us FROM lineage_kernel_versions v
            JOIN kernel_revisions k ON k.id=v.kernel_revision_id
            WHERE v.lineage_id=? AND v.revision_number=0
            """,
            (lineage_id,),
        )
    )
    rows = db.execute(
        """
        SELECT e.number,a.branch,a.challenger_ordinal,a.trajectory_ordinal,
               a.iteration_ordinal,a.id,a.accepted_as_branch_best,k.latency_us
        FROM attempts a JOIN epochs e ON e.id=a.epoch_id
        LEFT JOIN kernel_revisions k ON k.id=a.output_kernel_revision_id
        WHERE e.lineage_id=? AND a.status='completed'
        ORDER BY e.number,a.iteration_ordinal,a.branch,a.challenger_ordinal,
                 a.trajectory_ordinal,a.id
        """,
        (lineage_id,),
    ).fetchall()
    layers: dict[tuple[int, int], list[tuple[Any, ...]]] = {}
    for row in rows:
        layers.setdefault((int(row[0]), int(row[4])), []).append(row)
    best = baseline
    curve = [{"step": 0, "latency_us": best, "source": "bootstrap"}]
    for step, ((epoch, iteration), layer) in enumerate(sorted(layers.items()), 1):
        retained = [float(row[7]) for row in layer if row[6] and row[7] is not None]
        layer_best = min(retained, default=None)
        if layer_best is not None:
            best = min(best, layer_best)
        curve.append(
            {
                "step": step,
                "latency_us": best,
                "epoch": epoch,
                "iteration": iteration,
                "parallel_width": len(layer),
                "attempts": [
                    {
                        "attempt_id": row[5],
                        "branch": row[1],
                        "challenger_ordinal": row[2],
                        "trajectory": row[3],
                        "accepted": bool(row[6]),
                        "latency_us": row[7],
                    }
                    for row in layer
                ],
            }
        )
    return curve


def best_of_two(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> list[dict[str, Any]]:
    assert len(left) == len(right)
    return [
        {"step": a["step"], "latency_us": min(a["latency_us"], b["latency_us"])}
        for a, b in zip(left, right)
    ]


def _claude_trace(path: Path) -> dict[str, Any]:
    """Read one Claude JSONL using the last usage for each streamed message ID."""

    messages: dict[str, dict[str, Any]] = {}
    tool_uses: dict[str, str] = {}
    tool_results: set[str] = set()
    for line in path.read_text().splitlines():
        value = json.loads(line)
        message = value.get("message")
        if not isinstance(message, dict):
            continue
        if value.get("type") == "assistant":
            message_id = message.get("id")
            if message_id and message.get("usage"):
                messages[message_id] = message["usage"]
            content = message.get("content")
            if isinstance(content, list):
                for block in content:
                    if block.get("type") == "tool_use" and block.get("id"):
                        tool_uses.setdefault(block["id"], str(block.get("name", "unknown")))
        elif value.get("type") == "user":
            content = message.get("content")
            if isinstance(content, list):
                for block in content:
                    if block.get("type") == "tool_result" and block.get("tool_use_id"):
                        tool_results.add(block["tool_use_id"])
    totals = {name: 0 for name in TOKEN_COLUMNS}
    for value in messages.values():
        totals["uncached_input_tokens"] += int(value.get("input_tokens", 0))
        totals["cache_read_tokens"] += int(value.get("cache_read_input_tokens", 0))
        totals["cache_write_tokens"] += int(value.get("cache_creation_input_tokens", 0))
        totals["output_tokens"] += int(value.get("output_tokens", 0))
    return {
        **totals,
        "total_tokens": sum(totals.values()),
        "responses": len(messages),
        "tool_uses": len(tool_uses),
        "tool_results": len(tool_results),
        "tools_by_name": dict(sorted(Counter(tool_uses.values()).items())),
    }


def _sum_usage(items: Iterable[dict[str, Any]]) -> dict[str, int]:
    totals = {name: 0 for name in TOKEN_COLUMNS}
    for item in items:
        for name in TOKEN_COLUMNS:
            totals[name] += int(item.get(name, 0))
    totals["total_tokens"] = sum(totals.values())
    return totals


def _aka_curve(episode_paths: list[Path]) -> tuple[float, list[dict[str, Any]], Counter[str]]:
    attempts = [json.loads(path.read_text()) for path in episode_paths]
    baseline = next(
        float(verification["incumbent_latency_us"])
        for attempt in attempts
        if (verification := attempt.get("verification") or {}).get("incumbent_latency_us")
        is not None
    )
    current = baseline
    curve: list[dict[str, Any]] = [
        {"step": 0, "latency_us": current, "source": "framework_baseline"}
    ]
    outcomes: Counter[str] = Counter()
    for expected_episode, attempt in enumerate(attempts, 1):
        episode = int(attempt["episode"])
        if episode != expected_episode:
            raise ValueError(f"non-contiguous AKA Episodes: expected {expected_episode}, got {episode}")
        verification = attempt.get("verification") or {}
        candidate = verification.get("candidate_latency_us")
        if attempt.get("accepted") and candidate is not None:
            current = float(candidate)
            outcomes["accepted"] += 1
        else:
            outcomes[str(attempt.get("status", "unknown"))] += 1
        curve.append(
            {
                "step": episode,
                "latency_us": current,
                "status": attempt.get("status"),
                "accepted": bool(attempt.get("accepted")),
                "candidate_latency_us": candidate,
            }
        )
    return baseline, curve, outcomes


def aka_run_summary(run_number: int, inputs: dict[str, Path]) -> dict[str, Any]:
    manifest = json.loads(inputs["manifest"].read_text())
    trace_root = inputs["traces"]
    episode_directories = [
        path
        for path in trace_root.iterdir()
        if path.is_dir() and re.search(r"-e\d{4}-", path.name)
    ]
    main_traces = sorted(
        path for directory in episode_directories for path in directory.glob("*.jsonl")
    )
    subagent_traces = sorted(
        path
        for directory in episode_directories
        for path in directory.glob("*/subagents/*.jsonl")
    )
    if len(main_traces) != 42:
        raise ValueError(f"AKA run{run_number}: expected 42 main Episode traces, found {len(main_traces)}")
    main_trace_values = [_claude_trace(path) for path in main_traces]
    subagent_trace_values = [_claude_trace(path) for path in subagent_traces]
    main_usage = _sum_usage(main_trace_values)
    subagent_usage = _sum_usage(subagent_trace_values)
    trace_usage = _sum_usage([main_usage, subagent_usage])
    behavior = {
        "optimizer_sessions": len(main_traces),
        "subagent_sessions": len(subagent_traces),
        "responses": sum(item["responses"] for item in main_trace_values),
        "tool_uses": sum(item["tool_uses"] for item in main_trace_values),
        "tool_results": sum(item["tool_results"] for item in main_trace_values),
        "tools_by_name": dict(
            sorted(
                sum(
                    (Counter(item["tools_by_name"]) for item in main_trace_values),
                    Counter(),
                ).items()
            )
        ),
    }
    by_dsl: dict[str, Any] = {}
    manifest_names = {"cuda": "Cuda", "triton": "Triton", "cutedsl": "CuteDSL"}
    for dsl in DSLS:
        episode_paths = sorted(
            (
                inputs["workspace"]
                / f"kernel_opt_flash_attention_{dsl}_l20n_production"
                / ".atrex_long_horizon/episodes"
            ).glob("e*/attempt.json")
        )
        baseline, curve, outcomes = _aka_curve(episode_paths)
        branch = manifest["branches"][manifest_names[dsl]]
        final_latency = float(branch["best_pass_candidate"]["latency_us_geomean"])
        if not math.isclose(curve[-1]["latency_us"], final_latency, rel_tol=1e-12):
            raise ValueError(
                f"AKA run{run_number}/{dsl}: Episode reconstruction {curve[-1]['latency_us']} "
                f"does not match Manifest {final_latency}"
            )
        by_dsl[dsl] = {
            "episodes": len(episode_paths),
            "baseline_latency_us": baseline,
            "final_latency_us": final_latency,
            "speedup_over_baseline": baseline / final_latency,
            "best_version": branch["best_pass_candidate"]["version"],
            "v1_baseline_commit": branch["v1_baseline_commit"],
            "exit_code": branch["exit_code"],
            "outcomes": dict(sorted(outcomes.items())),
            "curve": curve,
        }
    policy_tokens = int(manifest["token_accounting"]["policy_reviewer_tokens"])
    return {
        "run": f"AKA-{run_number - 6}",
        "run_label": manifest["run_label"],
        "configured_max_iters": manifest["max_iters"],
        "episodes": sum(item["episodes"] for item in by_dsl.values()),
        "by_dsl": by_dsl,
        "usage": {
            "trace_resolved": trace_usage,
            "policy_reviewer_tokens_unpartitioned": policy_tokens,
            "total_tokens": trace_usage["total_tokens"] + policy_tokens,
            "main_episode_trace": main_usage,
            "subagents": subagent_usage,
        },
        "policy_review_calls": manifest["token_accounting"]["policy_review_calls"],
        "policy_review_timeout_kills": manifest["token_accounting"][
            "policy_review_timeout_kills"
        ],
        "behavior": behavior,
        "source_archive": str(inputs["archive"]),
    }


def seed_comparison(
    aka_runs: list[dict[str, Any]], aka_inputs: dict[int, dict[str, Path]], runtime_artifacts: Path
) -> dict[str, Any]:
    runtime_digests = {
        "cuda": "d413f1ef188d86c05bb68cc21c2a5644ce1fd810b3034e7a39c946ff0317c055",
        "triton": "4f64a37510e6efdab321260005638b3dfc20f0c4d86c796adf58ba4cfae73fcf",
        "cutedsl": "de3b33492f395864c8c0838335c43b07b822f14e21fdf14cbd28a19e35d34113",
    }
    descriptions = {
        "cuda": "Runtime adds default Model constructor values and updates compiler-plumbing prose; GPU source is unchanged.",
        "triton": "Runtime adds default Model constructor values and a bare-Model grid fallback; exact init_kwargs preserve the original launch geometry.",
        "cutedsl": "Runtime adds default Model constructor values; CuteDSL GPU source is unchanged.",
    }
    items: dict[str, Any] = {}
    for dsl in DSLS:
        left_commit = aka_runs[0]["by_dsl"][dsl]["v1_baseline_commit"]
        right_commit = aka_runs[1]["by_dsl"][dsl]["v1_baseline_commit"]
        if left_commit != right_commit:
            raise ValueError(f"AKA v1 seed mismatch for {dsl}")
        runtime_path = runtime_artifacts / runtime_digests[dsl] / "payload/kernel.py"
        if not runtime_path.is_file():
            raise FileNotFoundError(
                f"Runtime baseline source is not extracted: {runtime_path}. "
                "Extract the three baseline Artifact directories or pass --runtime-baseline-artifacts."
            )
        aka_repo = (
            aka_inputs[7]["workspace"]
            / f"kernel_opt_flash_attention_{dsl}_l20n_production"
        )
        aka_source = subprocess.check_output(
            ["git", "-C", str(aka_repo), "show", f"{left_commit}:kernel.py"]
        )
        runtime_source = runtime_path.read_bytes()
        opcodes = difflib.SequenceMatcher(
            None,
            aka_source.decode().splitlines(),
            runtime_source.decode().splitlines(),
        ).get_opcodes()
        changed = [opcode for opcode in opcodes if opcode[0] != "equal"]
        items[dsl] = {
            "aka_run7_run8_exact_commit_match": True,
            "aka_v1_commit": left_commit,
            "aka_source_sha256": hashlib.sha256(aka_source).hexdigest(),
            "runtime_artifact_digest": f"sha256:{runtime_digests[dsl]}",
            "runtime_source_sha256": hashlib.sha256(runtime_source).hexdigest(),
            "byte_identical_to_runtime": aka_source == runtime_source,
            "diff_hunks": len(changed),
            "difference_scope": descriptions[dsl],
        }
    return items


def digest_payload(artifact_root: Path, digest: str) -> Path:
    return artifact_root / digest.removeprefix("sha256:") / "payload"


def file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def state_files(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): file_hash(path)
        for path in root.rglob("*")
        if path.is_file()
    }


def state_diff(artifact_root: Path, before_digest: str, after_digest: str) -> dict[str, Any]:
    before = state_files(digest_payload(artifact_root, before_digest))
    after = state_files(digest_payload(artifact_root, after_digest))
    added = sorted(after.keys() - before.keys())
    removed = sorted(before.keys() - after.keys())
    modified = sorted(path for path in before.keys() & after.keys() if before[path] != after[path])
    category = lambda path: path.split("/", 1)[0]
    return {
        "added": added,
        "removed": removed,
        "modified": modified,
        "counts": {
            "added": len(added),
            "removed": len(removed),
            "modified": len(modified),
        },
        "changed_by_top_level": dict(sorted(Counter(map(category, added + removed + modified)).items())),
    }


def evolver_summary(db: sqlite3.Connection, artifact_root: Path) -> dict[str, Any]:
    rows = db.execute(
        """
        SELECT l.dsl,e.number,ec.proposal_type,ec.evolution_trace_digest,
               CASE WHEN e.winner_kernel_agent_revision_id=ec.kernel_agent_revision_id
                    THEN 1 ELSE 0 END AS won
        FROM epoch_challengers ec JOIN epochs e ON e.id=ec.epoch_id
        JOIN lineages l ON l.id=e.lineage_id
        WHERE ec.proposal_type != 'replica'
        ORDER BY l.dsl,e.number
        """
    ).fetchall()
    items = []
    changed_paths: Counter[str] = Counter()
    top_levels: Counter[str] = Counter()
    unimplemented = 0
    for dsl, epoch, proposal_type, digest, won in rows:
        value = json.loads((digest_payload(artifact_root, digest) / "value.json").read_text())
        output = value["output"]
        paths = output.get("changed_paths", [])
        changed_paths.update(paths)
        top_levels.update(path.split("/", 1)[0] for path in paths)
        unimplemented += len(output.get("unimplemented_capabilities", []))
        items.append(
            {
                "dsl": dsl,
                "epoch": epoch,
                "proposal_type": proposal_type,
                "won": bool(won),
                "trace_digest": digest,
                "hypothesis": output.get("hypothesis"),
                "expected_effect": output.get("expected_effect"),
                "changed_paths": paths,
                "contributing_paths": output.get("contributing_paths", []),
                "unimplemented_capabilities": output.get("unimplemented_capabilities", []),
            }
        )
    return {
        "proposals": len(items),
        "wins": sum(item["won"] for item in items),
        "wins_by_dsl": {
            dsl: sum(item["won"] for item in items if item["dsl"] == dsl) for dsl in DSLS
        },
        "changed_path_occurrences": sum(changed_paths.values()),
        "distinct_changed_paths": len(changed_paths),
        "changed_top_levels": dict(sorted(top_levels.items())),
        "unimplemented_capabilities": unimplemented,
        "items": items,
    }


def plot_svg(dsl: str, curves: dict[str, list[dict[str, Any]]], output: Path) -> None:
    width, height = 1500, 860
    left, right, top, bottom = 145, 90, 175, 105
    pw, ph = width - left - right, height - top - bottom
    values = [float(p["latency_us"]) for curve in curves.values() for p in curve]
    low, high = min(values) * 0.88, max(values) * 1.15
    use_log = high / low >= 3
    max_step = max(p["step"] for curve in curves.values() for p in curve)
    tx = lambda x: left + x / max_step * pw
    if use_log:
        y0, y1 = math.log10(low), math.log10(high)
        ty = lambda y: top + (y1 - math.log10(y)) / (y1 - y0) * ph
        ticks = [v for p in range(math.floor(y0) - 1, math.ceil(y1) + 1) for v in (10**p, 2 * 10**p, 5 * 10**p) if low <= v <= high]
    else:
        span = high - low
        unit = 10 ** math.floor(math.log10(span / 5))
        unit *= 1 if span / 5 / unit <= 1 else 2 if span / 5 / unit <= 2 else 5
        start = math.floor(low / unit) * unit
        ticks = []
        value = start
        while value <= high + unit:
            if value >= low:
                ticks.append(value)
            value += unit
        ty = lambda y: top + (high - y) / (high - low) * ph
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif;fill:#111827}</style>',
        f'<text x="{left}" y="62" font-size="34" font-weight="700">{DISPLAY_DSL[dsl]} — Best-so-far Kernel Latency</text>',
        f'<text x="{left}" y="102" font-size="20" fill="#4B5563">Bootstrap = 0 · Parallel Attempts at one serial depth count once</text>',
    ]
    for tick in ticks:
        y = ty(tick)
        label = f"{tick / 1000:.1f}k" if tick >= 1000 else f"{tick:.0f}"
        out += [
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + pw}" y2="{y:.2f}" stroke="#E5E7EB"/>',
            f'<text x="{left - 16}" y="{y + 7:.2f}" text-anchor="end" font-size="18" fill="#4B5563">{label}</text>',
        ]
    for tick in range(0, max_step + 1, 3):
        x = tx(tick)
        out += [
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + ph}" stroke="#F3F4F6"/>',
            f'<text x="{x:.2f}" y="{top + ph + 34}" text-anchor="middle" font-size="18" fill="#4B5563">{tick}</text>',
        ]
    out += [
        f'<line x1="{left}" y1="{top + ph}" x2="{left + pw}" y2="{top + ph}" stroke="#6B7280" stroke-width="2"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + ph}" stroke="#6B7280" stroke-width="2"/>',
        f'<text x="{left + pw / 2:.2f}" y="{height - 24}" text-anchor="middle" font-size="21" font-weight="600">Serial optimization distance from Bootstrap</text>',
        f'<text x="38" y="{top + ph / 2:.2f}" text-anchor="middle" font-size="21" font-weight="600" transform="rotate(-90 38 {top + ph / 2:.2f})">Latency (µs){" · log scale" if use_log else ""}</text>',
    ]
    legend_x, legend_y = width - 530, 34
    for index, name in enumerate(COMPARISON_ORDER):
        color, dash = STYLES[name]
        y = legend_y + index * 25
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        out.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 48}" y2="{y}" stroke="{color}" stroke-width="5"{dash_attr}/>')
        out.append(f'<text x="{legend_x + 62}" y="{y + 6}" font-size="17" font-weight="600">{html.escape(name)} · {curves[name][-1]["latency_us"]:,.1f} µs</text>')
    for name in COMPARISON_ORDER:
        color, dash = STYLES[name]
        points: list[tuple[float, float]] = []
        previous_y = None
        for point in curves[name]:
            x, y = tx(point["step"]), ty(point["latency_us"])
            if previous_y is not None:
                points.append((x, previous_y))
            points.append((x, y))
            previous_y = y
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        coords = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        out.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="4" stroke-linejoin="round"{dash_attr}/>')
    out.append("</svg>\n")
    output.write_text("\n".join(out))


def main() -> None:
    args = parse_args()
    aka_inputs = prepare_aka_inputs(args.aka_root, args.aka_extracted_root)
    aka_runs = [aka_run_summary(run_number, aka_inputs[run_number]) for run_number in (7, 8)]
    registry = args.extracted_root / "control-l20n/state/registry.sqlite"
    campaign_results_path = args.extracted_root / RUN_NAME / "campaign-results.json"
    artifact_root = args.extracted_root / "control-l20n/state/artifacts/sha256"
    campaign_results = json.loads(campaign_results_path.read_text())
    catalog = arm_catalog(campaign_results)
    output_dir = Path(__file__).resolve().parent
    curves_dir = output_dir / "latency-curves"
    kernels_dir = output_dir / "best-kernels"
    state_examples_dir = output_dir / "retained-state-examples"
    curves_dir.mkdir(parents=True, exist_ok=True)
    kernels_dir.mkdir(parents=True, exist_ok=True)
    state_examples_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "source_archive": str(args.archive),
        "source_archive_size_bytes": args.archive.stat().st_size,
        "source_registry_member": "production/control-l20n/state/registry.sqlite",
        "source_campaign_results_member": f"production/{RUN_NAME}/campaign-results.json",
        "comparison_policy": (
            "AKA: two independent runs of 14 non-Bootstrap Episodes per DSL. Runtime: "
            "30 optimizer attempts per DSL; parallel attempts at the same epoch/iteration count "
            "as one serial step. Bootstrap is excluded from Token comparisons."
        ),
        "aka": {
            "runs": aka_runs,
            "usage": {
                "trace_resolved": _sum_usage(
                    run["usage"]["trace_resolved"] for run in aka_runs
                ),
                "policy_reviewer_tokens_unpartitioned": sum(
                    run["usage"]["policy_reviewer_tokens_unpartitioned"]
                    for run in aka_runs
                ),
                "total_tokens": sum(run["usage"]["total_tokens"] for run in aka_runs),
            },
            "behavior": {
                "optimizer_sessions": sum(
                    run["behavior"]["optimizer_sessions"] for run in aka_runs
                ),
                "subagent_sessions": sum(
                    run["behavior"]["subagent_sessions"] for run in aka_runs
                ),
                "responses": sum(run["behavior"]["responses"] for run in aka_runs),
                "tool_uses": sum(run["behavior"]["tool_uses"] for run in aka_runs),
                "tool_results": sum(
                    run["behavior"]["tool_results"] for run in aka_runs
                ),
                "tools_by_name": dict(
                    sorted(
                        sum(
                            (
                                Counter(run["behavior"]["tools_by_name"])
                                for run in aka_runs
                            ),
                            Counter(),
                        ).items()
                    )
                ),
            },
        },
        "seed_comparison": seed_comparison(
            aka_runs, aka_inputs, args.runtime_baseline_artifacts
        ),
    }
    with sqlite3.connect(f"file:{registry}?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        lineages = [lineage_summary(db, entry) for entry in catalog]
        by_key = {(item["dsl"], item["arm"]): item for item in lineages}
        payload["lineages"] = lineages
        all_campaign_ids = [item["campaign_id"] for item in catalog]
        payload["session_totals"] = usage(db, all_campaign_ids)
        payload["optimization_session_totals"] = usage(
            db, all_campaign_ids, roles=("optimizer", "evolver")
        )
        comparisons: dict[str, Any] = {}
        groups = {
            "Isolated best of two": ["ablation-isolated-01", "ablation-isolated-02"],
            "Pool-3": ["ablation-pool-3"],
            "Pool-retained-3": ["ablation-pool-retained-3"],
            "Retained best of two": ["ablation-retained-01", "ablation-retained-02"],
            "Evolve-3": ["evolve-3"],
        }
        for name, arms in groups.items():
            items = [by_key[dsl, arm] for dsl in DSLS for arm in arms]
            comparisons[name] = {
                "arms": arms,
                "attempts": sum(item["attempts"] for item in items),
                "latency_us_by_dsl": {
                    dsl: min(by_key[dsl, arm]["final_latency_us"] for arm in arms)
                    for dsl in DSLS
                },
                "usage": usage(
                    db,
                    [item["campaign_id"] for item in items],
                    roles=("optimizer", "evolver"),
                ),
            }
        payload["comparisons"] = comparisons
        payload["aka"]["best_of_two_latency_us_by_dsl"] = {
            dsl: min(run["by_dsl"][dsl]["final_latency_us"] for run in aka_runs)
            for dsl in DSLS
        }
        curves_payload: dict[str, Any] = {}
        for dsl in DSLS:
            isolated = best_of_two(
                runtime_curve(db, by_key[dsl, "ablation-isolated-01"]["lineage_id"]),
                runtime_curve(db, by_key[dsl, "ablation-isolated-02"]["lineage_id"]),
            )
            retained = best_of_two(
                runtime_curve(db, by_key[dsl, "ablation-retained-01"]["lineage_id"]),
                runtime_curve(db, by_key[dsl, "ablation-retained-02"]["lineage_id"]),
            )
            curves = {
                "AKA best of two": best_of_two(
                    aka_runs[0]["by_dsl"][dsl]["curve"],
                    aka_runs[1]["by_dsl"][dsl]["curve"],
                ),
                "Isolated best of two": isolated,
                "Pool-3": runtime_curve(db, by_key[dsl, "ablation-pool-3"]["lineage_id"]),
                "Pool-retained-3": runtime_curve(db, by_key[dsl, "ablation-pool-retained-3"]["lineage_id"]),
                "Retained best of two": retained,
                "Evolve-3": runtime_curve(db, by_key[dsl, "evolve-3"]["lineage_id"]),
            }
            curves_payload[dsl] = curves
            plot_svg(dsl, curves, curves_dir / f"{dsl}.svg")
        (curves_dir / "curves.json").write_text(json.dumps(curves_payload, indent=2) + "\n")
        payload["evolver"] = evolver_summary(db, artifact_root)

        initial_states = {}
        for dsl in DSLS:
            initial_states[dsl] = scalar(
                db,
                """
                SELECT a.input_runtime_state_digest FROM attempts a
                JOIN epochs e ON e.id=a.epoch_id JOIN lineages l ON l.id=e.lineage_id
                WHERE l.dsl=? AND l.challenger_count=1 AND e.number=1
                ORDER BY a.branch,a.ordinal LIMIT 1
                """,
                (dsl,),
            )
        state_diffs = {}
        representative_state_files = {
            "cuda": (
                "tools/flash_attention_component_probe.py",
                "memory/epoch4-attempt2-split-rightsizing-keep-record.md",
            ),
            "triton": (
                "tools/analyze_abba.py",
                "memory/epoch4-attempt2-tma-lessons.md",
            ),
            "cutedsl": (
                "tools/gate_sim.py",
                "memory/optimization-epoch5-attempt1-fp16-2c-partials.md",
            ),
        }
        for dsl in DSLS:
            for arm in ("ablation-pool-retained-3", "ablation-retained-01", "ablation-retained-02"):
                item = by_key[dsl, arm]
                final_state = item["best_attempt"]["runtime_state_digest"]
                state_diffs[f"{dsl}/{arm}"] = state_diff(
                    artifact_root, initial_states[dsl], final_state
                )
            pool_state_digest = by_key[dsl, "ablation-pool-retained-3"]["best_attempt"]["runtime_state_digest"]
            pool_state = digest_payload(artifact_root, pool_state_digest)
            target_root = state_examples_dir / dsl
            for relative_path in representative_state_files[dsl]:
                target = target_root / relative_path
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(pool_state / relative_path, target)
        payload["runtime_state_diffs"] = state_diffs

        best_by_dsl = {}
        for dsl in DSLS:
            best = min((item for item in lineages if item["dsl"] == dsl), key=lambda item: item["final_latency_us"])
            result = json.loads(
                (digest_payload(artifact_root, best["best_result_artifact_digest"]) / "value.json").read_text()
            )
            candidate = result.get("candidate", result.get("result", {}))
            latencies = candidate.get("latency_us_by_shape", {})
            best["shape_count"] = len(latencies)
            best["shape_latency_min_us"] = min(latencies.values())
            best["shape_latency_max_us"] = max(latencies.values())
            best["correctness"] = candidate.get("correctness")
            best_by_dsl[dsl] = best
            source = digest_payload(artifact_root, best["best_kernel_artifact_digest"]) / "kernel.py"
            target_dir = kernels_dir / dsl
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target_dir / "kernel.py")
            report_digest = best["best_attempt"]["attempt_report_digest"]
            shutil.copyfile(
                digest_payload(artifact_root, report_digest) / "value.json",
                target_dir / "attempt-report.json",
            )
            result_summary = {
                key: value
                for key, value in result.items()
                if key not in {"jobs", "payloads"}
            }
            result_summary["source_result_artifact_digest"] = best[
                "best_result_artifact_digest"
            ]
            (target_dir / "gateway-result.json").write_text(
                json.dumps(result_summary, indent=2, ensure_ascii=False) + "\n"
            )
        payload["best_by_dsl"] = best_by_dsl

    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
