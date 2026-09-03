"""Drill into the broad shell_read_or_filter label without executing commands.

Exclusive buckets describe whole calls, not token allocation to shell segments.
Rules inspect executable shell text after stripping heredoc payloads; uncertain
mixed commands remain explicit. --raw-output is optional local audit material.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import shlex
from collections import Counter, defaultdict
from pathlib import Path

import tiktoken
from bash_context_tokens import events_from, fragment_counts, prepare_fragments, replay
from bash_text_tokens import digest, extract_bash

TEXT_PROGRAMS = {"cat", "head", "tail", "rg", "grep", "sed", "awk", "cut", "wc"}


def shell_commands(command):
    """Same conservative shell/heredoc boundary as the original label audit."""
    lines = command.splitlines()
    shell = []
    i = 0
    while i < len(lines):
        line = lines[i]
        shell.append(line)
        ends = re.findall(r"<<-?\s*(['\"]?)([A-Za-z_][\w.-]*)\1", line)
        i += 1
        for _, end in ends:
            while i < len(lines) and lines[i].strip() != end:
                i += 1
            i += 1
    residual = "\n".join(shell)
    try:
        lexer = shlex.shlex(residual, posix=True, punctuation_chars="();<>|&\n")
        lexer.whitespace = " \t\r"
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return [], residual, False
    segments, current = [], []
    for token in tokens:
        if token and all(c in "();|&\n" for c in token):
            if current:
                segments.append(current)
            current = []
        else:
            current.append(token)
    if current:
        segments.append(current)
    commands = []
    for segment in segments:
        body = list(segment)
        while body and (
            body[0]
            in (
                "do",
                "then",
                "else",
                "if",
                "while",
                "until",
                "!",
                "{",
                "time",
                "command",
                "exec",
                "nohup",
            )
            or re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", body[0])
        ):
            body.pop(0)
        if body and body[0] == "env":
            body.pop(0)
            while body and (
                body[0].startswith("-") or re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", body[0])
            ):
                body.pop(0)
        if body:
            commands.append((body[0].rsplit("/", 1)[-1], body[1:]))
    return commands, residual, True


def is_writer(program, args):
    if program == "sed":
        return any(arg.startswith("-i") for arg in args)
    if program != "cat":
        return False
    # A heredoc sent to a file is construction, not a file-content read. A cat
    # without an output redirection is still a reader even if its input is stdin.
    return any(arg.startswith("<<") for arg in args) and any(arg in (">", ">>") for arg in args)


def classify(command, flags):
    commands, text, ok = shell_commands(command)
    text = text.lower()
    matches = [(p, a) for p, a in commands if p in TEXT_PROGRAMS]
    readers = [(p, a) for p, a in matches if not is_writer(p, a)]
    writers = [(p, a) for p, a in matches if is_writer(p, a)]
    detail = {
        "text_programs": sorted({p for p, _ in matches}),
        "writer_matches": len(writers),
        "reader_matches": len(readers),
        "shell_parsed": ok,
    }
    if not ok or not matches:
        return "unresolved_shell", detail
    if not readers:
        return "write_only_name_match", detail
    if re.search(r"(?:^|\s)--help(?:\s|$)", text):
        return "cli_help", detail
    if "gpu_entry" in flags or re.search(r"(?:tools/)?sandbox\.py.*--kind", text):
        return "gpu_execution_and_filter", detail
    if "aka_wiki" in flags or "runtime_wiki-query" in flags or "query_nl.py" in text:
        return "wiki_query_and_filter", detail
    if "journal_report_union" in flags:
        return "journal_api_and_filter", detail
    if "aka_control_union" in flags:
        return "control_command_and_filter", detail
    if re.search(
        r"(?:\.claude/projects|conversation\.jsonl|session.*\.jsonl|journal_cli|journal_path)", text
    ):
        return "session_or_protocol_recovery", detail
    if re.search(
        r"(?:\bmemory\b|/evidence/|direction[^\s]*\.json|experiment[^\s]*\.json|report\.json|\bplans/|\bplan/|solution\.json|summarize_reports|journal\.json|telemetry\.jsonl|attempt\.json)",
        text,
    ):
        return "history_plan_report_files", detail
    if re.search(
        r"(?:\bprofiles?/|\.csv\b|\.log\b|\.output\b|tasks/|summary\.txt|eval[^\s]*\.json|profile[^\s]*\.json|/tmp/[^\s]*\.txt)",
        text,
    ):
        return "measurement_and_task_logs", detail
    if re.search(r"(?:reference-projects/|site-packages/)", text):
        return "reference_library_code", detail
    if "agent_problem.json" in text:
        return "operator_contract", detail
    if re.search(r"(?:\bskills/|/skills\b|\.md\b|\bgpu-wiki/)", text):
        return "skills_docs_files", detail
    if re.search(
        r"(?:kernel\.py|solution\.py|test_kernel\.py|model\.py|reference\.py|\.cu\b|\bkernels/)",
        text,
    ):
        return "kernel_implementation_files", detail
    if re.search(
        r"(?:tools/(?:sandbox\.py|profile_nvidia\.sh|analyze_reports\.py|iteration_trace\.py)|runtime_tools\.py|long_horizon|third_party/atrex-kernel-agent/|agent/optimizer/src/)",
        text,
    ):
        return "harness_framework_code", detail
    if re.search(r"(?:tools/|/src/|\.py\b)", text):
        return "experiment_helper_code", detail
    if "shell_discovery" in flags:
        return "directory_environment_listing", detail
    return "other_mixed_reads", detail


def analyze(index, archive):
    encoding = {"o200k_base": tiktoken.get_encoding("o200k_base")}
    rows = []
    for number, session in enumerate(index["sessions"], 1):
        path = (archive / session["trace"]).resolve()
        if not path.is_relative_to(archive.resolve()):
            raise ValueError("Trace escaped archive root")
        payload = path.read_bytes()
        if digest(payload) != session["trace_sha256"]:
            raise ValueError("Trace hash mismatch")
        tools = extract_bash(payload, session["group"] == "retained")
        annotations, selected = {}, {}
        for action in session["actions"]:
            tool = tools[(action["line"], action["command_sha256"])].pop(0)
            if "shell_read_or_filter" in action["flags"]:
                annotations[tool["id"]] = action
                selected[tool["id"]] = tool
        events = events_from(payload, session["group"] == "retained")
        fragments = prepare_fragments(events, annotations, encoding)
        replay(events, fragments)
        counts = defaultdict(Counter)
        for fragment in fragments:
            counts[fragment["tool_id"]].update(fragment_counts(fragment))
        for tid, action in annotations.items():
            tool = selected[tid]
            bucket, detail = classify(tool["command"], action["flags"])
            count = counts[tid]
            rows.append(
                {
                    "group": session["group"],
                    "dsl": session["dsl"],
                    "session": session["session"],
                    "trace": session["trace"],
                    "line": action["line"],
                    "tool_id": tid,
                    "command_sha256": action["command_sha256"],
                    "result_sha256": digest(
                        json.dumps(tool["results"], ensure_ascii=False).encode()
                    ),
                    "flags": action["flags"],
                    "bucket": bucket,
                    **detail,
                    "command": tool["command"],
                    "result_excerpt": "\n".join(tool["results"])[:1500],
                    "visible_tokens": count["o200k_base_visible_once"],
                    "command_total_tokens": count["o200k_base_preserved_messages_command_input"]
                    + count["o200k_base_preserved_messages_generated"],
                    "result_total_tokens": count["o200k_base_preserved_messages_result_input"],
                    "total_tokens": count["o200k_base_preserved_messages_total"],
                }
            )
        if number % 25 == 0:
            print(f"Inspected {number}/{len(index['sessions'])} sessions", flush=True)
    return rows


def summarize(rows):
    groups = {}
    for group in ("AKA", "retained"):
        selected = [row for row in rows if row["group"] == group]
        total, buckets = Counter(), defaultdict(Counter)
        for row in selected:
            value = {
                "calls": 1,
                **{
                    k: row[k]
                    for k in (
                        "visible_tokens",
                        "command_total_tokens",
                        "result_total_tokens",
                        "total_tokens",
                    )
                },
            }
            total.update(value)
            buckets[row["bucket"]].update(value)
        repeats = Counter()
        seen_commands, seen_results = set(), set()
        for row in selected:
            command_key = (row["session"], row["command_sha256"])
            result_key = (*command_key, row["result_sha256"])
            if command_key in seen_commands:
                repeats["same_command_extra_calls"] += 1
                repeats["same_command_extra_token_exposure"] += row["total_tokens"]
            if result_key in seen_results:
                repeats["same_command_and_result_extra_calls"] += 1
                repeats["same_command_and_result_extra_token_exposure"] += row["total_tokens"]
            seen_commands.add(command_key)
            seen_results.add(result_key)
        groups[group] = {
            "total": total,
            "buckets": buckets,
            "exact_repeats_within_session": repeats,
        }
    compact = []
    for group, values in groups.items():
        for bucket in values["buckets"]:
            selected = sorted(
                (r for r in rows if r["group"] == group and r["bucket"] == bucket),
                key=lambda r: r["total_tokens"],
                reverse=True,
            )[:5]
            compact.extend(
                {k: v for k, v in r.items() if k not in ("command", "result_excerpt")}
                for r in selected
            )
    return {
        "method": (
            "Heuristic whole-call exclusive buckets, prioritized as written in classify(); "
            "includes writer-only false positives from original utility-name label. "
            "Commands are never executed. Token exposure reuses the compaction-aware replay "
            "estimator. Top-five call references per bucket are included, not raw text."
        ),
        "groups": groups,
        "top_calls": compact,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path)
    args = parser.parse_args()
    rows = analyze(json.loads(gzip.decompress(args.index.read_bytes())), args.archive_root)
    result = summarize(rows)
    result["index_sha256"] = digest(args.index.read_bytes())
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    if args.raw_output:
        args.raw_output.write_text(json.dumps(rows, ensure_ascii=False) + "\n")
    print(json.dumps(result["groups"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
