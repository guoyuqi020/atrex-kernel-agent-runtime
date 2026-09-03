"""Estimate Bash text exposure across observed requests and compaction boundaries.

Offline analysis, not a Provider bill or a counterfactual cost attribution. Never
execute trace commands. See README.md for exclusions and sensitivity scenarios.
"""

from __future__ import annotations

import argparse
import gzip
import importlib.metadata
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import tiktoken
from bash_text_tokens import CATEGORIES, ENCODINGS, content_text, digest, extract_bash

SCENARIOS = ("preserved_messages", "reset_at_compaction", "ignore_compaction")


def events_from(payload: bytes, runtime: bool) -> list[tuple[int, dict[str, Any]]]:
    events = []
    for line_number, line in enumerate(payload.decode().splitlines(), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        event = raw.get("event", {}) if runtime else raw
        if isinstance(event, dict):
            events.append((line_number, event))
    return events


def chain_of(event: dict[str, Any]) -> str:
    return event.get("parent_tool_use_id") or "main"


def prepare_fragments(
    events: list[tuple[int, dict[str, Any]]],
    annotations: dict[str, dict[str, Any]],
    encodings: dict[str, Any],
) -> list[dict[str, Any]]:
    """Keep literal text identity, visibility line and owning chain separately."""
    fragments = []
    seen_tools: set[str] = set()
    seen_results: set[tuple[str, str]] = set()
    chains = {}
    for _, event in events:
        message = event.get("message", {})
        blocks = message.get("content", [])
        if isinstance(blocks, list):
            for block in blocks:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    chains[block.get("id")] = chain_of(event)
    for line, event in events:
        blocks = event.get("message", {}).get("content", [])
        if not isinstance(blocks, list):
            continue
        for block in blocks:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                tool_id = block.get("id")
                if tool_id not in annotations or tool_id in seen_tools:
                    continue
                seen_tools.add(tool_id)
                kind = "command"
                text = block.get("input", {}).get("command", "")
            elif block.get("type") == "tool_result":
                tool_id = block.get("tool_use_id")
                signature = (tool_id, json.dumps(block, sort_keys=True))
                if tool_id not in annotations or signature in seen_results:
                    continue
                seen_results.add(signature)
                kind = "result"
                text = content_text(block.get("content", ""))
            else:
                continue
            if not event.get("uuid"):
                raise ValueError(f"Cannot track a Bash fragment without UUID at line {line}")
            fragments.append(
                {
                    "line": line,
                    "uuid": event["uuid"],
                    "tool_id": tool_id,
                    "chain": chains[tool_id],
                    "kind": kind,
                    "chars": len(text),
                    "flags": annotations[tool_id]["flags"],
                    "tokens": {n: len(e.encode_ordinary(text)) for n, e in encodings.items()},
                    "reads": dict.fromkeys(SCENARIOS, 0),
                    "compaction_passes": 0,
                }
            )
    return fragments


def replay(
    events: list[tuple[int, dict[str, Any]]], fragments: list[dict[str, Any]]
) -> tuple[Counter[str], list[dict[str, Any]]]:
    """Read the *prior* context once per response ID, then append its content.

    Full request payloads are unavailable. Between recorded compactions, assume
    observed Bash fragments remain in their chain's context. At compaction keep
    only fragments whose event UUID appears in the explicit preserved list.
    """
    by_line: dict[int, list[int]] = defaultdict(list)
    for i, fragment in enumerate(fragments):
        by_line[fragment["line"]].append(i)
    active = {scenario: defaultdict(set) for scenario in SCENARIOS}
    known_uuids: set[str] = set()
    requests: dict[tuple[str, str], dict[str, Any]] = {}
    diagnostics: Counter[str] = Counter()
    for line, event in events:
        chain = chain_of(event)
        if event.get("subtype") == "compact_boundary":
            metadata = event.get("compact_metadata", event.get("compactMetadata", {}))
            preserved = metadata.get("preserved_messages", metadata.get("preservedMessages"))
            if not isinstance(preserved, dict):
                raise ValueError(f"Missing preserved-message metadata at line {line}")
            keep = preserved.get("all_uuids", preserved.get("allUuids", preserved.get("uuids")))
            if not isinstance(keep, list):
                raise ValueError(f"Missing explicit preserved UUID list at line {line}")
            keep = set(keep)
            diagnostics["compactions"] += 1
            diagnostics["preserved_uuids"] += len(keep)
            diagnostics["unobserved_preserved_uuids"] += len(keep - known_uuids)
            before = active["preserved_messages"][chain]
            for i in before:
                fragments[i]["compaction_passes"] += 1
            after = {i for i in before if fragments[i]["uuid"] in keep}
            diagnostics["bash_fragments_preserved_at_boundaries"] += len(after)
            active["preserved_messages"][chain] = after
            active["reset_at_compaction"][chain].clear()
        message = event.get("message", {})
        if event.get("type") == "assistant" and message.get("id"):
            key = (chain, message["id"])
            if key not in requests:
                diagnostics["observed_requests"] += 1
                diagnostics[f"{'main' if chain == 'main' else 'child'}_requests"] += 1
                footprint = Counter()
                for scenario in SCENARIOS:
                    for i in active[scenario][chain]:
                        fragment = fragments[i]
                        fragment["reads"][scenario] += 1
                        if scenario == "preserved_messages":
                            footprint.update(fragment["tokens"])
                requests[key] = {
                    "line": line,
                    "message_id": message["id"],
                    "chain": chain,
                    "bash_input_tokens": dict(footprint),
                }
            if message.get("usage"):
                # Replace, never sum intermediate stream usage snapshots.
                requests[key]["usage_last_observed"] = message["usage"]
        for i in by_line[line]:
            for scenario in SCENARIOS:
                active[scenario][fragments[i]["chain"]].add(i)
        if event.get("uuid"):
            known_uuids.add(event["uuid"])
    for request in requests.values():
        usage = request.get("usage_last_observed", {})
        if all(
            k in usage
            for k in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
        ):
            input_tokens = sum(
                usage[k]
                for k in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
            )
            diagnostics["requests_with_full_input_usage_fields"] += 1
            diagnostics["observed_full_input_tokens"] += input_tokens
            footprint = request["bash_input_tokens"].get("o200k_base", 0)
            if footprint > input_tokens:
                diagnostics["requests_bash_estimate_exceeds_full_input"] += 1
                diagnostics["bash_estimate_excess_tokens"] += footprint - input_tokens
    return diagnostics, list(requests.values())


def fragment_counts(fragment: dict[str, Any]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for encoding, tokens in fragment["tokens"].items():
        kind = fragment["kind"]
        counts[f"{encoding}_{kind}_once"] += tokens
        counts[f"{encoding}_visible_once"] += tokens
        for scenario in SCENARIOS:
            prefix = f"{encoding}_{scenario}"
            reads = fragment["reads"][scenario]
            counts[f"{prefix}_{kind}_input"] += tokens * reads
            counts[f"{prefix}_input"] += tokens * reads
            generated = tokens if kind == "command" else 0
            counts[f"{prefix}_generated"] += generated
            counts[f"{prefix}_total"] += tokens * reads + generated
        counts[f"{encoding}_optional_single_compaction_pass_input"] += (
            tokens * fragment["compaction_passes"]
        )
    return counts


def measure(index: dict[str, Any], archive: Path) -> dict[str, Any]:
    encodings = {name: tiktoken.get_encoding(name) for name in ENCODINGS}
    groups: dict[str, Any] = {}
    sessions = []
    for number, session in enumerate(index["sessions"], 1):
        path = (archive / session["trace"]).resolve()
        if not path.is_relative_to(archive.resolve()):
            raise ValueError("Trace path escaped archive root")
        payload = path.read_bytes()
        if digest(payload) != session["trace_sha256"]:
            raise ValueError(f"Trace changed: {session['trace']}")
        runtime = session["group"] == "retained"
        tools = extract_bash(payload, runtime)
        if sum(map(len, tools.values())) != len(session["actions"]):
            raise ValueError("Bash count differs from annotations")
        annotations = {}
        for action in session["actions"]:
            tool = tools[(action["line"], action["command_sha256"])].pop(0)
            if (
                len(tool["command"]) != action["command_chars"]
                or sum(map(len, tool["results"])) != action["output_chars"]
            ):
                raise ValueError("Bash text differs from character audit")
            annotations[tool["id"]] = action
        events = events_from(payload, runtime)
        fragments = prepare_fragments(events, annotations, encodings)
        diagnostics, _ = replay(events, fragments)
        total: Counter[str] = Counter(calls=len(annotations))
        categories = {category: Counter() for category in CATEGORIES}
        child_total: Counter[str] = Counter()
        for fragment in fragments:
            counts = fragment_counts(fragment)
            total.update(counts)
            if fragment["chain"] != "main":
                child_total.update(counts)
                if fragment["kind"] == "command":
                    diagnostics["child_bash_calls"] += 1
            for category in fragment["flags"]:
                if category in categories:
                    categories[category].update(counts)
        for action in annotations.values():
            for category in action["flags"]:
                if category in categories:
                    categories[category]["calls"] += 1
        # The primary and sensitivity retention rules should nest, never invert.
        for encoding in ENCODINGS:
            assert (
                total[f"{encoding}_reset_at_compaction_total"]
                <= total[f"{encoding}_preserved_messages_total"]
                <= total[f"{encoding}_ignore_compaction_total"]
            )
        group = groups.setdefault(
            session["group"],
            {
                "total": Counter(),
                "categories": {c: Counter() for c in CATEGORIES},
                "dsl": {},
                "diagnostics": Counter(),
                "child_total": Counter(),
            },
        )
        group["total"].update(total)
        group["diagnostics"].update(diagnostics)
        group["child_total"].update(child_total)
        group["dsl"].setdefault(session["dsl"], Counter()).update(total)
        for category, counts in categories.items():
            group["categories"][category].update(counts)
        sessions.append(
            {
                "group": session["group"],
                "dsl": session["dsl"],
                "session": session["session"],
                "trace": session["trace"],
                "total": total,
                "diagnostics": diagnostics,
            }
        )
        if number % 25 == 0:
            print(f"Replayed {number}/{len(index['sessions'])} sessions", flush=True)
    return {
        "method": {
            "tokenizer": f"tiktoken=={importlib.metadata.version('tiktoken')}",
            "encodings": ENCODINGS,
            "primary": "preserved_messages",
            "request": (
                "First appearance of a distinct (chain, assistant message.id); "
                "stream blocks do not add requests"
            ),
            "formula": (
                "command generation once + command/context-result token count times "
                "subsequent observed in-chain requests retaining the original fragment"
            ),
            "compaction": (
                "Keep only explicit preserved_messages.all_uuids / preservedMessages.allUuids; "
                "no attribution of summary text to old Bash categories"
            ),
            "scenarios": {
                "preserved_messages": "Observed compactions, including retained original messages",
                "reset_at_compaction": "Sensitivity: discard all old Bash text at each compaction",
                "ignore_compaction": "Naive comparator: keep original text until chain ends",
            },
            "optional_compaction_input": (
                "Additional one pass over pre-compaction Bash context per boundary; "
                "hypothesis, not an observed model request; excluded from primary total"
            ),
            "between_boundaries_assumption": (
                "Observed fragments remain intact until the next logged compaction "
                "unless their chain ends; unlogged context edits cannot be reconstructed"
            ),
            "child_chains": (
                "Separate parent_tool_use_id chains; only locally observed content "
                "is replayed; unknown inherited parent context excluded"
            ),
            "categories_overlap": True,
            "excluded": [
                "Initial/system prompt, thinking, prose, non-Bash tools and schema overhead",
                "Hidden requests, retries without distinct response ID, summary generation",
                "Separate Reviewer/subagent archives",
                "Unobserved inherited child context",
                "TaskOutput/non-Bash reads not back-attributed to Bash",
            ],
            "not_inferred": [
                "Unlogged context truncation, clearing or replacement",
                "Per-fragment cache hit status or input/cache-read/cache-write split",
                "Provider billing or currency cost",
                "Counterfactual savings from removing an action",
            ],
            "same_bootstrap_dsl_seed": True,
        },
        "groups": groups,
        "sessions": sessions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    index = json.loads(gzip.decompress(args.index.read_bytes()))
    result = measure(index, args.archive_root)
    result["method"]["index_sha256"] = digest(args.index.read_bytes())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                g: {"total": v["total"], "diagnostics": v["diagnostics"]}
                for g, v in result["groups"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
