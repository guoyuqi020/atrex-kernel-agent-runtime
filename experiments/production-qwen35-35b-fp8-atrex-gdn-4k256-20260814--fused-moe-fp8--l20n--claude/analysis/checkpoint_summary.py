"""Build matched-checkpoint and full-budget summaries from read-only archives."""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


DSLS = ("cuda", "triton", "cutedsl")
ARMS = {
    "evolve",
    "ablation-retained",
    "ablation-pooled",
    "ablation-isolated-01",
    "ablation-isolated-02",
}
USAGE_KEYS = ("input", "cache_read", "cache_write", "output")
PRICES = {"input": 12, "cache_read": 1, "cache_write": 15, "output": 36}


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def terminal_usage(trace: Path) -> Counter:
    usage = Counter()
    for line in trace.read_text().splitlines():
        value = json.loads(line)
        event = value.get("event", value)
        if event.get("type") != "result" or not event.get("modelUsage"):
            continue
        usage.clear()
        for model in event["modelUsage"].values():
            usage.update(
                {
                    "input": int(model.get("inputTokens", 0)),
                    "cache_read": int(model.get("cacheReadInputTokens", 0)),
                    "cache_write": int(model.get("cacheCreationInputTokens", 0)),
                    "output": int(model.get("outputTokens", 0)),
                }
            )
    if not sum(usage.values()):
        raise ValueError(f"no terminal modelUsage in {trace}")
    return usage


def finish_usage(usage: Counter) -> dict:
    result = {key: int(usage[key]) for key in USAGE_KEYS}
    result["total"] = sum(result.values())
    result["cost_cny"] = sum(result[key] * PRICES[key] for key in USAGE_KEYS) / 1_000_000
    return result


def runtime_summary(
    production: Path, database: sqlite3.Connection, *, through_epoch: int
) -> dict:
    artifacts = production / "control-l20n/state/artifacts/sha256"
    result = {}
    lineage_specs = []
    for seed in sorted(production.glob("*/dsls/*/*/seed-result.json")):
        arm = seed.parent.name
        dsl = seed.parent.parent.name
        if arm not in ARMS:
            continue
        lineage_id = json.loads(seed.read_text())["lineage"]["lineage_id"]
        lineage_specs.append((dsl, arm, lineage_id))
    for lineage in database.execute(
        "select id,dsl from lineages where challenger_count=1 order by dsl"
    ):
        lineage_specs.append((lineage["dsl"], "evolve", lineage["id"]))
    for dsl, arm, lineage_id in lineage_specs:
        epoch_one = database.execute(
            "select * from epochs where lineage_id=? and number=1", (lineage_id,)
        ).fetchone()
        target_epoch = database.execute(
            "select * from epochs where lineage_id=? and number=?", (lineage_id, through_epoch)
        ).fetchone()
        next_epoch = database.execute(
            "select starting_kernel_revision_id from epochs where lineage_id=? and number=?",
            (lineage_id, through_epoch + 1),
        ).fetchone()
        if next_epoch is not None:
            best_kernel_revision_id = next_epoch["starting_kernel_revision_id"]
        else:
            best_kernel_revision_id = database.execute(
                "select best_kernel_revision_id from lineages where id=?", (lineage_id,)
            ).fetchone()["best_kernel_revision_id"]
        kernel = database.execute(
            "select * from kernel_revisions where id=?", (best_kernel_revision_id,)
        ).fetchone()
        baseline = database.execute(
            "select * from kernel_revisions where id=?", (epoch_one["starting_kernel_revision_id"],)
        ).fetchone()
        sessions = database.execute(
            """select s.* from worker_sessions s join epochs e on e.id=s.epoch_id
               where s.lineage_id=? and e.number<=? and s.role in ('optimizer','evolver')
               and s.status='completed' order by s.started_at""",
            (lineage_id, through_epoch),
        ).fetchall()
        usage = Counter()
        roles = Counter()
        agent_seconds = 0.0
        for session in sessions:
            roles[session["role"]] += 1
            agent_seconds += (
                parse_time(session["completed_at"]) - parse_time(session["started_at"])
            ).total_seconds()
            digest = session["trace_digest"].split(":", 1)[1]
            usage.update(terminal_usage(artifacts / digest / "payload/conversation.jsonl"))
        result[f"{dsl}/{arm}"] = {
            "dsl": dsl,
            "arm": arm,
            "lineage_id": lineage_id,
            "baseline_latency_us": baseline["latency_us"],
            "latency_us": kernel["latency_us"],
            "epoch_wall_hours": (
                parse_time(target_epoch["completed_at"]) - parse_time(epoch_one["created_at"])
            ).total_seconds()
            / 3600,
            "agent_hours": agent_seconds / 3600,
            "optimizer_sessions": roles["optimizer"],
            "evolver_sessions": roles["evolver"],
            "usage": finish_usage(usage),
        }
    if len(result) != 15:
        raise ValueError(f"expected 15 Runtime arms, found {len(result)}")
    return result


def aka_roots(archive: Path) -> dict[str, Path]:
    return {
        "AKA-1": archive / "AKA/atrex-runs.unpacked/atrex-runs",
        "AKA-2": archive / "AKA/atrex-runs2.unpacked/atrex-runs2",
    }


def aka_episode_for_review(
    archive: Path, trace: dict, journals: dict[tuple[str, str], list[tuple[int, datetime]]]
) -> int:
    first = json.loads((archive / trace["trace"]).open().readline())
    timestamp = parse_time(first["timestamp"])
    starts = journals[trace["system"], trace["dsl"]]
    preceding = [entry for entry in starts if entry[1] <= timestamp]
    # A few first-Episode reviews started immediately before Journal creation.
    return max(preceding, key=lambda entry: entry[1])[0] if preceding else starts[0][0]


def aka_summary(
    archive: Path, provider_audit: Path, *, through_episode: int
) -> dict:
    roots = aka_roots(archive)
    result = {}
    journals: dict[tuple[str, str], list[tuple[int, datetime]]] = {}
    for system, root in roots.items():
        for dsl in DSLS:
            episode_paths = sorted(
                root.glob(
                    f"kernel_opt_fused_moe_fp8_{dsl}_l20n_production/"
                    ".atrex_long_horizon/episodes/e*/attempt.json"
                )
            )
            all_journal_starts = []
            for path in episode_paths:
                journal = json.loads((path.parent / "episode_runtime/journal.json").read_text())
                all_journal_starts.append(
                    (int(json.loads(path.read_text())["episode"]), parse_time(journal["created_at"]))
                )
            episodes = []
            current_latency = None
            status = Counter()
            for path in episode_paths:
                attempt = json.loads(path.read_text())
                if int(attempt["episode"]) > through_episode:
                    continue
                verification = attempt.get("verification") or {}
                if current_latency is None and verification.get("incumbent_latency_us") is not None:
                    current_latency = verification["incumbent_latency_us"]
                if attempt.get("accepted") and verification.get("candidate_latency_us") is not None:
                    current_latency = verification["candidate_latency_us"]
                    status["promoted"] += 1
                elif attempt.get("status") == "pivot":
                    status["pivot"] += 1
                elif attempt.get("status") == "invalid_handoff":
                    status["protocol_failure"] += 1
                else:
                    status["rejected"] += 1
                journal = json.loads((path.parent / "episode_runtime/journal.json").read_text())
                episodes.append((attempt, journal))
            if len(episodes) != through_episode or current_latency is None:
                raise ValueError(f"incomplete AKA checkpoint for {system}/{dsl}")
            journals[system, dsl] = sorted(all_journal_starts, key=lambda entry: entry[1])
            result[f"{system}/{dsl}"] = {
                "system": system,
                "dsl": dsl,
                "latency_us": current_latency,
                "episode_wall_hours": (
                    parse_time(episodes[-1][1]["finalized_at"])
                    - parse_time(episodes[0][1]["created_at"])
                ).total_seconds()
                / 3600,
                "episodes": through_episode,
                "outcomes": dict(status),
                "usage": Counter(),
            }

    audit = json.loads(provider_audit.read_text())
    included = Counter()
    role_usage: dict[str, Counter] = defaultdict(Counter)
    for trace in audit["traces"]:
        if trace["role"] == "policy_review":
            episode = aka_episode_for_review(archive, trace, journals)
        else:
            match = re.search(r"-e(\d{4})-", trace["trace"])
            if not match:
                raise ValueError(f"cannot identify Episode for {trace['trace']}")
            episode = int(match.group(1))
        if episode > through_episode:
            continue
        target = result[f"{trace['system']}/{trace['dsl']}"]["usage"]
        source = trace["usage"]
        target.update(
            {
                "input": int(source["input_tokens"]),
                "cache_read": int(source["cache_read_input_tokens"]),
                "cache_write": int(source["cache_creation_input_tokens"]),
                "output": int(source["output_tokens"]),
            }
        )
        included[trace["role"]] += 1
        role_usage[trace["role"]].update(
            {
                "input": int(source["input_tokens"]),
                "cache_read": int(source["cache_read_input_tokens"]),
                "cache_write": int(source["cache_creation_input_tokens"]),
                "output": int(source["output_tokens"]),
            }
        )
    for item in result.values():
        item["usage"] = finish_usage(item["usage"])
    unknown_usage = audit["method"].get("unknown_usage")
    if through_episode >= 19 and unknown_usage:
        usage = result[f"{unknown_usage['system']}/{unknown_usage['dsl']}"]["usage"]
        unknown_tokens = int(unknown_usage["total_tokens"])
        usage["unknown"] = unknown_tokens
        usage["total"] += unknown_tokens
        usage["cost_cny_min"] = usage["cost_cny"] + unknown_tokens / 1_000_000
        usage["cost_cny_max"] = usage["cost_cny"] + unknown_tokens * 36 / 1_000_000
        unassigned_usage = None
    else:
        # The total-only review has no Trace timestamp, so an Episode-10
        # checkpoint cannot determine which side of the boundary contains it.
        unassigned_usage = unknown_usage
    return {
        "runs": result,
        "included_traces": dict(included),
        "included_usage_by_role": {
            role: finish_usage(usage) for role, usage in role_usage.items()
        },
        "unassigned_usage": unassigned_usage,
    }


def tree_files(path: Path) -> dict[str, bytes]:
    result = {}
    for item in path.rglob("*"):
        if item.is_file() and not {"__pycache__", ".pytest_cache"}.intersection(item.parts):
            result[str(item.relative_to(path))] = item.read_bytes()
    return result


def evolution_category(scope: str, path: str) -> str:
    if scope == "source":
        if path.startswith("prompts/"):
            return "prompt"
        if path.startswith("tests/"):
            return "tests"
        if path.startswith("src/"):
            return "src"
        return "other_source"
    if path.startswith("skills/"):
        return "skills"
    if path.startswith("tools/"):
        return "tools"
    return "other_state"


def evolution_summary(
    production: Path, database: sqlite3.Connection, *, through_epoch: int
) -> dict:
    rows = database.execute(
        """select s.workspace_path,l.dsl,e.number from worker_sessions s
           join lineages l on l.id=s.lineage_id join epochs e on e.id=s.epoch_id
           where s.role='evolver' and e.number<=? and s.status='completed'
           order by l.dsl,e.number""",
        (through_epoch,),
    ).fetchall()
    file_changes = Counter()
    line_changes = Counter()
    size_direction = Counter()
    proposals = Counter()
    for row in rows:
        workspace_suffix = Path(row["workspace_path"]).parts[-2:]
        matches = list(
            production.glob(
                f"*/state/evolution-workspaces/{workspace_suffix[0]}/{workspace_suffix[1]}"
            )
        )
        if len(matches) != 1:
            raise ValueError(f"cannot resolve Evolution workspace {row['workspace_path']}")
        workspace = matches[0]
        report = json.loads((workspace / "scratch/evolution-report.json").read_text())
        proposals[report["proposal_type"]] += 1
        state_inputs = list(
            (workspace / "input/agents/active/runtime-state/trajectories").glob("*")
        )
        if len(state_inputs) != 1:
            raise ValueError(f"expected one Active runtime state in {workspace}")
        comparisons = (
            ("source", workspace / "input/agents/active/source", workspace / "candidate/source"),
            ("state", state_inputs[0], workspace / "candidate/runtime-state"),
        )
        for scope, before_path, after_path in comparisons:
            before = tree_files(before_path)
            after = tree_files(after_path)
            for path in sorted(set(before) | set(after)):
                category = evolution_category(scope, path)
                if path not in before:
                    file_changes[category, "added"] += 1
                elif path not in after:
                    file_changes[category, "deleted"] += 1
                elif before[path] != after[path]:
                    file_changes[category, "modified"] += 1
                    try:
                        old_lines = before[path].decode().splitlines()
                        new_lines = after[path].decode().splitlines()
                    except UnicodeDecodeError:
                        continue
                    added = deleted = 0
                    for opcode, i1, i2, j1, j2 in difflib.SequenceMatcher(
                        None, old_lines, new_lines
                    ).get_opcodes():
                        if opcode in {"insert", "replace"}:
                            added += j2 - j1
                        if opcode in {"delete", "replace"}:
                            deleted += i2 - i1
                    line_changes[category, "added"] += added
                    line_changes[category, "deleted"] += deleted
                    direction = (
                        "longer"
                        if len(new_lines) > len(old_lines)
                        else "shorter"
                        if len(new_lines) < len(old_lines)
                        else "same"
                    )
                    size_direction[category, direction] += 1
    expected_sessions = 3 * (through_epoch - 1)
    if len(rows) != expected_sessions:
        raise ValueError(
            f"expected {expected_sessions} Evolution sessions, found {len(rows)}"
        )
    return {
        "sessions": len(rows),
        "proposals": dict(proposals),
        "file_changes": {f"{a}/{b}": value for (a, b), value in file_changes.items()},
        "line_changes": {f"{a}/{b}": value for (a, b), value in line_changes.items()},
        "modified_file_size_direction": {
            f"{a}/{b}": value for (a, b), value in size_direction.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--provider-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    production = next(args.archive_root.glob("runtime/*.unpacked/production"))
    database = sqlite3.connect(
        f"file:{production}/control-l20n/state/registry.sqlite?mode=ro", uri=True
    )
    database.row_factory = sqlite3.Row
    value = {
        "matched_checkpoint": {
            "scope": {
                "runtime_through_epoch": 5,
                "aka_through_episode": 10,
                "bootstrap_included": False,
            },
            "runtime": runtime_summary(production, database, through_epoch=5),
            "aka": aka_summary(
                args.archive_root, args.provider_audit, through_episode=10
            ),
            "evolution": evolution_summary(production, database, through_epoch=5),
        },
        "full_budget": {
            "scope": {
                "runtime_through_epoch": 10,
                "aka_configured_max_iters": 20,
                "aka_archived_optimizer_episodes": 19,
                "bootstrap_included": False,
            },
            "runtime": runtime_summary(production, database, through_epoch=10),
            "aka": aka_summary(
                args.archive_root, args.provider_audit, through_episode=19
            ),
            "evolution": evolution_summary(production, database, through_epoch=10),
        },
    }
    args.output.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
