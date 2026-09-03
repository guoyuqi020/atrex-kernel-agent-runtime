"""Offline tokenization of annotated Bash calls. Never execute trace commands.

Install tiktoken==0.12.0 in a separate environment. Reproduce using --index,
--archive-root, and --output. --actions is only for sealing the original action
annotations into the compressed index; subsequent runs need no temporary files.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import tiktoken

CATEGORIES = (
    "gpu_entry",
    "aka_control_union",
    "journal_report_union",
    "shell_read_or_filter",
    "shell_discovery",
    "inline_python",
    "python_json",
    "file_content_write_signal",
)
ENCODINGS = ("o200k_base", "cl100k_base")


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def seal_index(actions_path: Path, archive: Path, destination: Path) -> None:
    sessions: dict[str, dict[str, Any]] = {}
    for row in json.loads(actions_path.read_text()):
        relative = Path(row["trace"]).resolve().relative_to(archive.resolve()).as_posix()
        if relative not in sessions:
            sessions[relative] = {
                "trace": relative,
                "trace_sha256": digest((archive / relative).read_bytes()),
                "group": row["group"],
                "dsl": row["dsl"],
                "session": row["session"],
                "actions": [],
            }
        sessions[relative]["actions"].append(
            {
                "line": row["line"],
                "flags": row["flags"],
                "command_sha256": digest(row["command"].encode()),
                "command_chars": row["command_chars"],
                "output_chars": row["output_chars"],
            }
        )
    index = {
        "annotation_method": "Existing report section 4.1 conservative shell/heredoc/AST labels",
        "categories_overlap": True,
        "actions_sha256": digest(actions_path.read_bytes()),
        "sessions": list(sessions.values()),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(gzip.compress(json.dumps(index, ensure_ascii=False).encode(), mtime=0))


def content_text(value: Any) -> str:
    """Use the same visible-text boundary as the report's existing character audit."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(item.get("text", "") for item in value if isinstance(item, dict))
    return json.dumps(value, ensure_ascii=False)


def extract_bash(payload: bytes, runtime: bool) -> dict[tuple[int, str], list[dict[str, Any]]]:
    tools: dict[str, dict[str, Any]] = {}
    results: dict[str, list[str]] = defaultdict(list)
    seen_results: set[tuple[str, str]] = set()
    for line_number, line in enumerate(payload.decode().splitlines(), 1):
        if not line.strip():
            continue
        event = json.loads(line)
        if runtime:
            event = event.get("event", {})
        if not isinstance(event, dict):
            continue
        message = event.get("message", {})
        if not isinstance(message, dict) or not isinstance(message.get("content"), list):
            continue
        for block in message["content"]:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use" and block.get("id") not in tools:
                tools[block["id"]] = {
                    "id": block["id"],
                    "line": line_number,
                    "name": block.get("name"),
                    "command": block.get("input", {}).get("command", ""),
                }
            if block.get("type") == "tool_result":
                tool_id = block.get("tool_use_id")
                signature = (tool_id, json.dumps(block, sort_keys=True))
                if signature not in seen_results:
                    seen_results.add(signature)
                    results[tool_id].append(content_text(block.get("content", "")))
    found: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for tool_id, tool in tools.items():
        if tool["name"] == "Bash":
            key = (tool["line"], digest(tool["command"].encode()))
            found[key].append({**tool, "results": results.get(tool_id, [])})
    return found


def add(target: Counter[str], counts: dict[str, int]) -> None:
    target.update(counts)


def measure(index: dict[str, Any], archive: Path) -> dict[str, Any]:
    encodings = {name: tiktoken.get_encoding(name) for name in ENCODINGS}
    groups: dict[str, dict[str, Any]] = {}
    per_session: list[dict[str, Any]] = []
    for number, session in enumerate(index["sessions"], 1):
        path = (archive / session["trace"]).resolve()
        if not path.is_relative_to(archive.resolve()):
            raise ValueError("Trace path escaped archive root")
        payload = path.read_bytes()
        if digest(payload) != session["trace_sha256"]:
            raise ValueError(f"Trace content changed: {session['trace']}")
        tools = extract_bash(payload, runtime=session["group"] == "retained")
        if sum(map(len, tools.values())) != len(session["actions"]):
            raise ValueError(f"Bash count differs from action annotations: {session['trace']}")
        group = groups.setdefault(
            session["group"],
            {
                "sessions": 0,
                "total": Counter(),
                "categories": {category: Counter() for category in CATEGORIES},
                "dsl": {},
            },
        )
        group["sessions"] += 1
        dsl = group["dsl"].setdefault(session["dsl"], Counter())
        total: Counter[str] = Counter()
        categories = {category: Counter() for category in CATEGORIES}
        for action in session["actions"]:
            tool = tools[(action["line"], action["command_sha256"])].pop(0)
            command = tool["command"]
            outputs = tool["results"]
            if digest(command.encode()) != action["command_sha256"]:
                raise ValueError(f"Command does not match annotation: {session['trace']}")
            counts = {
                "calls": 1,
                "calls_with_result": int(bool(outputs)),
                "result_messages": len(outputs),
                "command_chars": len(command),
                "output_chars": sum(map(len, outputs)),
            }
            for key in ("command_chars", "output_chars"):
                if counts[key] != action[key]:
                    raise ValueError(
                        f"Character audit mismatch: {session['trace']}:{action['line']} {key}"
                    )
            for name, encoding in encodings.items():
                counts[f"{name}_command_tokens"] = len(encoding.encode_ordinary(command))
                counts[f"{name}_output_tokens"] = sum(
                    len(encoding.encode_ordinary(text)) for text in outputs
                )
                counts[f"{name}_visible_tokens"] = (
                    counts[f"{name}_command_tokens"] + counts[f"{name}_output_tokens"]
                )
            add(total, counts)
            for category in CATEGORIES:
                if category in action["flags"]:
                    add(categories[category], counts)
        add(group["total"], dict(total))
        add(dsl, dict(total))
        for category, counts in categories.items():
            add(group["categories"][category], dict(counts))
        per_session.append(
            {
                "group": session["group"],
                "dsl": session["dsl"],
                "session": session["session"],
                "trace": session["trace"],
                "total": dict(total),
            }
        )
        if number % 25 == 0:
            print(f"Tokenized {number}/{len(index['sessions'])} sessions", flush=True)
    assert groups["AKA"]["total"]["calls"] == 13745
    assert groups["retained"]["total"]["calls"] == 6922
    return {
        "method": {
            "package": "tiktoken",
            "package_version": importlib.metadata.version("tiktoken"),
            "primary_encoding": "o200k_base",
            "sensitivity_encoding": "cl100k_base",
            "source": "https://github.com/openai/tiktoken",
            "command": "Exact Bash input.command, including heredocs, excluding other input fields",
            "output": "Visible text from distinct tool_result blocks joined by tool_use_id",
            "dedup": "Tool calls by tool ID; results by tool ID plus canonical result block",
            "category_allocation": (
                "Full call text assigned to every matching label; labels overlap"
            ),
            "excluded": [
                "Initial/system prompt",
                "thinking",
                "surrounding response prose",
                "historical context replay/cache accounting",
                "message/tool schema overhead",
                "non-text/image tokens",
                "TaskOutput/non-Bash reads; later Bash reads belong only to their own call",
            ],
            "interpretation": (
                "Visible text token estimates, not Claude billing or causal action cost"
            ),
            "same_bootstrap_dsl_seed": True,
        },
        "groups": groups,
        "sessions": per_session,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument(
        "--actions", type=Path, help="Seal original action annotations before measuring"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.actions:
        seal_index(args.actions, args.archive_root, args.index)
    index = json.loads(gzip.decompress(args.index.read_bytes()))
    result = measure(index, args.archive_root)
    result["method"]["index_sha256"] = digest(args.index.read_bytes())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["groups"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
