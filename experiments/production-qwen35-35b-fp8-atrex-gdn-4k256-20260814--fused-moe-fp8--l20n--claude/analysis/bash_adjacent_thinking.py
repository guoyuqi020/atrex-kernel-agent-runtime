"""Estimate visible thinking adjacent to Bash; never execute archived commands.

Generation is observable text. Replayed input is a retention scenario, not proof
that the Provider actually resubmitted thinking on subsequent requests.
"""

from __future__ import annotations

import argparse
import gzip
import importlib.metadata
import json
from collections import Counter, defaultdict
from pathlib import Path

import tiktoken
from bash_context_tokens import SCENARIOS, chain_of, events_from, replay
from bash_text_tokens import digest, extract_bash
from shell_read_audit import classify


def thinking_fragments(events, annotations, encoder):
    """Use adjacent action batches in the same chain, never search distant turns.

    Ordinary assistant prose and non-conversation metadata are transparent.
    Human messages, compaction and other content are barriers. Before-thinking
    belongs only to calls in its own response; after-thinking follows the nearest
    completed tool batch. Non-Bash neighbors are retained as ambiguity metadata.
    """
    tools = {}
    for _, event in events:
        message = event.get("message", {})
        blocks = message.get("content", [])
        if isinstance(blocks, list):
            for block in blocks:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    tools[block["id"]] = (chain_of(event), message.get("id"))
    chains = defaultdict(list)
    seen = set()
    diagnostics = Counter()
    for line, event in events:
        chain = chain_of(event)
        message = event.get("message", {})
        mid = message.get("id")
        if event.get("subtype") == "compact_boundary" or event.get("isCompactSummary"):
            chains[chain].append({"type": "barrier"})
            continue
        if event.get("type") not in ("assistant", "user"):
            continue
        blocks = message.get("content", [])
        if not isinstance(blocks, list):
            chains[chain].append({"type": "barrier"})
            continue
        for block in blocks:
            if not isinstance(block, dict):
                chains[chain].append({"type": "barrier"})
                continue
            kind = block.get("type")
            if event.get("type") == "user" and kind != "tool_result":
                chains[chain].append({"type": "barrier"})
                continue
            tid = block.get("id") if kind == "tool_use" else block.get("tool_use_id")
            owner, producer = tools.get(tid, (chain, mid))
            if kind == "tool_use":
                signature = (kind, tid)
            elif kind == "tool_result":
                signature = (kind, tid, json.dumps(block, sort_keys=True))
            else:
                signature = (chain, mid, kind, json.dumps(block, sort_keys=True))
            if signature in seen:
                diagnostics[f"duplicate_{kind}_blocks"] += 1
                continue
            seen.add(signature)
            node = {
                "type": kind,
                "line": line,
                "uuid": event.get("uuid"),
                "mid": producer if kind == "tool_result" else mid,
                "tool_id": tid,
            }
            if kind == "thinking" and event.get("type") == "assistant":
                if not mid or not event.get("uuid"):
                    raise ValueError(f"Thinking lacks response ID/UUID at line {line}")
                node["text"] = block.get("thinking", "")
                if not isinstance(node["text"], str):
                    raise ValueError(f"Thinking is not text at line {line}")
                diagnostics["visible_thinking_blocks"] += 1
            elif kind == "text" and event.get("type") == "assistant":
                pass
            elif kind not in ("tool_use", "tool_result"):
                node["type"] = "barrier"
            chains[owner if kind == "tool_result" else chain].append(node)

    fragments = []
    for chain, nodes in chains.items():
        for pos, node in enumerate(nodes):
            if node["type"] != "thinking":
                continue
            before, after = set(), set()
            # All calls in the adjacent response batch share the preceding thought.
            # Native logs may interleave results between calls from ONE message ID.
            for following in nodes[pos + 1 :]:
                if following.get("mid") != node["mid"]:
                    break
                if following["type"] == "tool_use":
                    before.add(following["tool_id"])
                elif following["type"] not in ("text", "thinking", "tool_result"):
                    break
            preceding = pos - 1
            while preceding >= 0:
                previous = nodes[preceding]
                if previous["type"] in ("text", "thinking") and previous.get("mid") == node["mid"]:
                    preceding -= 1
                else:
                    break
            if preceding >= 0 and nodes[preceding]["type"] == "tool_result":
                batch = nodes[preceding]["mid"]
                while preceding >= 0:
                    previous = nodes[preceding]
                    if previous.get("mid") != batch or previous["type"] not in (
                        "tool_use",
                        "tool_result",
                        "text",
                    ):
                        break
                    if previous["type"] == "tool_result":
                        after.add(previous["tool_id"])
                    preceding -= 1
            before_bash = before & annotations.keys()
            after_bash = after & annotations.keys()
            neighbors = before_bash | after_bash
            if not neighbors:
                continue
            relation = (
                "both" if before_bash and after_bash else "before" if before_bash else "after"
            )
            fragment = {
                "line": node["line"],
                "uuid": node["uuid"],
                "message_id": node["mid"],
                "chain": chain,
                "kind": "thinking",
                "chars": len(node["text"]),
                "tokens": {"o200k_base": len(encoder.encode_ordinary(node["text"]))},
                "reads": dict.fromkeys(SCENARIOS, 0),
                "compaction_passes": 0,
                "before_tool_ids": sorted(before_bash),
                "after_tool_ids": sorted(after_bash),
                "neighbor_tool_ids": sorted(neighbors),
                "relation": relation,
                "non_bash_neighbors": sorted((before | after) - annotations.keys()),
            }
            fragments.append(fragment)
    return fragments, diagnostics


def counts(fragment):
    tokens = fragment["tokens"]["o200k_base"]
    result = Counter(blocks=1, generated=tokens)
    for scenario in SCENARIOS:
        result[f"{scenario}_input"] = tokens * fragment["reads"][scenario]
        result[f"{scenario}_total"] = tokens + result[f"{scenario}_input"]
    return result


def measure(index, archive, raw_output=None):
    encoder = tiktoken.get_encoding("o200k_base")
    groups, sessions = {}, []
    raw_sessions = []
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
            raise ValueError("Bash count differs from frozen index")
        annotations, buckets = {}, {}
        for action in session["actions"]:
            tool = tools[(action["line"], action["command_sha256"])].pop(0)
            annotations[tool["id"]] = action
            buckets[tool["id"]] = (
                classify(tool["command"], action["flags"])[0]
                if "shell_read_or_filter" in action["flags"]
                else "other_bash"
            )
        events = events_from(payload, runtime)
        fragments, diagnostics = thinking_fragments(events, annotations, encoder)
        replay(events, fragments)
        total = Counter(calls=len(annotations))
        by_bucket, by_relation = defaultdict(Counter), defaultdict(Counter)
        mixed = Counter()
        covered = set()
        records = []
        for fragment in fragments:
            value = counts(fragment)
            total.update(value)
            by_relation[fragment["relation"]].update(value)
            covered.update(fragment["neighbor_tool_ids"])
            if fragment["non_bash_neighbors"]:
                mixed.update(value)
            # This is explicit association, not causal attribution. Shared thought
            # is divided equally across distinct adjacent Bash calls, even across buckets.
            weight = len(fragment["neighbor_tool_ids"])
            for tid in fragment["neighbor_tool_ids"]:
                by_bucket[buckets[tid]].update({key: n / weight for key, n in value.items()})
            records.append(
                {
                    **{
                        key: fragment[key]
                        for key in (
                            "line",
                            "uuid",
                            "message_id",
                            "chain",
                            "before_tool_ids",
                            "after_tool_ids",
                            "non_bash_neighbors",
                            "relation",
                            "reads",
                        )
                    },
                    "generated_tokens": fragment["tokens"]["o200k_base"],
                }
            )
        total["calls_with_adjacent_thinking"] = len(covered)
        group = groups.setdefault(
            session["group"],
            {
                "total": Counter(),
                "buckets": defaultdict(Counter),
                "relations": defaultdict(Counter),
                "mixed_non_bash": Counter(),
                "dsl": defaultdict(Counter),
                "diagnostics": Counter(),
            },
        )
        group["total"].update(total)
        group["dsl"][session["dsl"]].update(total)
        group["diagnostics"].update(diagnostics)
        group["mixed_non_bash"].update(mixed)
        for key, value in by_bucket.items():
            group["buckets"][key].update(value)
        for key, value in by_relation.items():
            group["relations"][key].update(value)
        identity = {key: session[key] for key in ("group", "dsl", "session", "trace")}
        sessions.append(
            {
                **identity,
                "total": total,
                "top_fragments": sorted(
                    records,
                    key=lambda r: r["generated_tokens"] * (1 + r["reads"]["preserved_messages"]),
                    reverse=True,
                )[:3],
            }
        )
        if raw_output is not None:
            raw_sessions.append({**identity, "fragments": records})
        if number % 25 == 0:
            print(f"Inspected thinking {number}/{len(index['sessions'])} sessions", flush=True)
    if raw_output is not None:
        raw_output.write_text(json.dumps(raw_sessions, ensure_ascii=False, indent=2) + "\n")
    return {
        "method": {
            "tokenizer": f"tiktoken=={importlib.metadata.version('tiktoken')}",
            "encoding": "o200k_base",
            "same_bootstrap_dsl_seed": True,
            "adjacency": (
                "Thinking before calls in its same response batch, or immediately after the "
                "preceding result batch in its own chain. Assistant prose and metadata are "
                "transparent; human messages, compaction and other content are barriers."
            ),
            "dedup": (
                "One block per (chain, message.id, canonical content); one union of Bash neighbors"
            ),
            "allocation": "Equal shares across distinct adjacent Bash calls; not causal ownership",
            "thinking_input_scenario": (
                "Generation once plus hypothetical replay until observed compaction. Explicit "
                "preserved UUIDs survive. Request payloads are unavailable: automatic thinking "
                "stripping cannot be ruled out. Generation-only is also reported."
            ),
            "excluded": (
                "Hidden/redacted thinking, signatures, prose, summaries, unobserved requests"
            ),
            "fragment_references": (
                "Three largest cumulative fragments per session; no thinking text"
            ),
        },
        "groups": groups,
        "sessions": sessions,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--raw-output", type=Path, help="Optional full local fragment-to-call index"
    )
    args = parser.parse_args()
    index = json.loads(gzip.decompress(args.index.read_bytes()))
    result = measure(index, args.archive_root, args.raw_output)
    result["method"]["index_sha256"] = digest(args.index.read_bytes())
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
