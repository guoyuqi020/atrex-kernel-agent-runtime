"""Reproduce the full-budget AKA/isolated/pooled/retained report from archived traces.

Reuse the original AKA call annotations and the same fragment replay/tokenizer.
Compare AKA's archived 19-Episode runs with isolated, pooled, and retained Epochs 1-10. Runtime
call labels use executable shell segments, excluding heredoc bodies.
Never execute archived commands. Output contains derived counts and references.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import sqlite3
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import tiktoken
from bash_adjacent_thinking import counts, thinking_fragments
from bash_context_tokens import chain_of, events_from, fragment_counts, prepare_fragments, replay
from bash_text_tokens import content_text, digest, extract_bash
from shell_read_audit import TEXT_PROGRAMS, classify, shell_commands


def call_flags(command):
    commands, _, _ = shell_commands(command)
    flags = set()
    journals = {"update-direction", "list-directions", "load-direction", "record-experiment",
                "list-experiments", "load-experiment", "attempt-report"}
    for program, args in commands:
        if program in TEXT_PROGRAMS:
            flags.add("shell_read_or_filter")
        if program in {"ls", "find", "pwd", "which", "command", "env", "stat", "du", "ps"}:
            flags.add("shell_discovery")
        scripts = []
        if re.fullmatch(r"python[\d.]*|bash|sh|zsh", program) and args:
            if args[0] == "-m" and len(args) > 1:
                scripts.append((args[1], args[2:]))
            elif not args[0].startswith("-"):
                scripts.append((args[0].rsplit("/", 1)[-1], args[1:]))
        elif program.endswith((".py", ".sh")):
            scripts.append((program, args))
        for name, rest in scripts:
            if name == "runtime_tools.py" and rest:
                if rest[0] == "gateway-execute": flags.add("gpu_entry")
                if rest[0] == "wiki-query": flags.add("runtime_wiki-query")
                if rest[0] in journals: flags.add("journal_report_union")
            if name == "sandbox.py": flags.add("gpu_entry")
            if name == "query_nl.py": flags.add("aka_wiki")
            if name == "long_horizon.journal": flags.add("journal_report_union")
            if name in {"validate-gen-plan-io.sh", "ask-reviewers.sh"} or (
                name == "iteration_trace.py" and rest and rest[0] in {"phase-start", "phase-end"}
            ): flags.add("aka_control_union")
        if program == "git" and args and args[0] in {
            "status", "log", "diff", "show", "commit", "add", "rev-parse", "restore", "checkout", "reset"
        }: flags.add("aka_control_union")
    return sorted(flags)


def build_index(archive, frozen):
    index = json.loads(gzip.decompress(frozen.read_bytes()))
    sessions = []
    for session in index["sessions"]:
        if session["group"] != "AKA":
            continue
        match = re.search(r"-e(\d{4})-", session["trace"])
        if match and int(match.group(1)) <= 19:
            sessions.append(session)
    production = next(archive.glob("runtime/*.unpacked/production"))
    with sqlite3.connect(f"file:{production}/control-l20n/state/registry.sqlite?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        runtime_groups = (
            ("pooled", "ablation-pooled", 40),
            ("isolated", "ablation-isolated-*", 20),
            ("retained", "ablation-retained", 40),
        )
        for group, arm_pattern, expected_sessions in runtime_groups:
            for seed in sorted(
                production.glob(f"*/dsls/*/{arm_pattern}/seed-result.json")
            ):
                lineage = json.loads(seed.read_text())["lineage"]
                rows = db.execute("""select s.*,l.dsl,e.number,a.trajectory_ordinal,a.iteration_ordinal
                    from worker_sessions s join lineages l on l.id=s.lineage_id
                    join epochs e on e.id=s.epoch_id join attempts a on a.id=s.attempt_id
                    where s.lineage_id=? and s.role='optimizer' and e.number<=10
                    order by e.number,a.ordinal""",
                    (lineage["lineage_id"],)).fetchall()
                assert len(rows) == expected_sessions
                first_kernel_digests = set()
                for row in rows:
                    assert row["status"] == "completed"
                    trace = production / "control-l20n/state/artifacts/sha256" / row["trace_digest"].split(":")[1] / "payload/conversation.jsonl"
                    payload = trace.read_bytes()
                    tools = [t for ts in extract_bash(payload, True).values() for t in ts]
                    sessions.append({"group": group, "dsl": row["dsl"], "session": row["id"],
                        "trace": str(trace.relative_to(archive)), "trace_sha256": digest(payload),
                        "epoch": row["number"], "trajectory": row["trajectory_ordinal"],
                        "attempt": row["iteration_ordinal"], "attempt_id": row["attempt_id"],
                        "attempt_report_digest": db.execute("select attempt_report_digest from attempts where id=?", (row["attempt_id"],)).fetchone()[0],
                        "actions": [{"line": t["line"], "command_sha256": digest(t["command"].encode()),
                            "command_chars": len(t["command"]), "output_chars": sum(map(len,t["results"])),
                            "flags": call_flags(t["command"])} for t in tools]})
                    if row["number"] == row["iteration_ordinal"] == 1:
                        first_kernel_digests.add(db.execute("select k.artifact_digest from attempts a join kernel_revisions k on k.id=a.input_kernel_revision_id where a.id=?", (row["attempt_id"],)).fetchone()[0])
                assert len(first_kernel_digests) == 1
                print(group.title(), rows[0]["dsl"], "seed", next(iter(first_kernel_digests)), flush=True)
    return {"method": "Frozen AKA annotations; Runtime shell_commands/call_flags", "sessions": sessions}


def measure(index, archive):
    encoder = tiktoken.get_encoding("o200k_base")
    groups = defaultdict(lambda: {"behavior": Counter(), "usage": Counter(), "bash": Counter(),
        "thinking": Counter(), "buckets": defaultdict(Counter), "dsl_usage": defaultdict(Counter)})
    sessions, examples = [], defaultdict(list)
    for number, session in enumerate(index["sessions"], 1):
        payload = (archive/session["trace"]).read_bytes()
        assert digest(payload) == session["trace_sha256"]
        runtime = session["group"] != "AKA"
        events = events_from(payload, runtime)
        tools = extract_bash(payload, runtime)
        annotations, buckets, commands = {}, {}, {}
        for a in session["actions"]:
            t = tools[a["line"], a["command_sha256"]].pop(0)
            annotations[t["id"]] = a; commands[t["id"]] = t
            buckets[t["id"]] = classify(t["command"],a["flags"])[0] if "shell_read_or_filter" in a["flags"] else "other_bash"
        group = groups[session["group"]]
        fragments = prepare_fragments(events, annotations, {"o200k_base": encoder})
        replay(events, fragments)
        per_call = defaultdict(Counter)
        for f in fragments:
            v = fragment_counts(f); group["bash"].update(v);per_call[f["tool_id"]].update(v)
        for tid,a in annotations.items():
            v=per_call[tid];bucket=buckets[tid]
            value={"calls":1,"command":v["o200k_base_preserved_messages_command_input"]+v["o200k_base_preserved_messages_generated"],
                "result":v["o200k_base_preserved_messages_result_input"],"bash":v["o200k_base_preserved_messages_total"]}
            group["buckets"][bucket].update(value)
            examples[session["group"],bucket].append({"trace":session["trace"],"line":a["line"],"dsl":session["dsl"],
                "session":session["session"],"tokens":value["bash"],"command":commands[tid]["command"],
                "result_excerpt":"\n".join(commands[tid]["results"])[:1500]})
        thoughts,_ = thinking_fragments(events, annotations, encoder);replay(events,thoughts)
        for f in thoughts:
            v=counts(f);group["thinking"].update(v)
            for tid in f["neighbor_tool_ids"]:
                group["buckets"][buckets[tid]]["thinking"]+=v["preserved_messages_total"]/len(f["neighbor_tool_ids"])
        ids,uses,results,usage = set(),{},set(),Counter();behavior=Counter(sessions=1)
        for _,e in events:
            if e.get("type")=="system" and e.get("subtype")=="compact_boundary":behavior["compactions"]+=1
            m=e.get("message") or {}
            if e.get("type")=="assistant" and m.get("id"):ids.add((chain_of(e),m["id"]))
            if e.get("type")=="result" and runtime:
                usage=Counter()
                for model,u in e.get("modelUsage",{}).items():
                    usage.update({k:int(u.get(v,0)) for k,v in {"input":"inputTokens","cache_read":"cacheReadInputTokens","cache_write":"cacheCreationInputTokens","output":"outputTokens"}.items()})
            for b in m.get("content",[]) if isinstance(m.get("content"),list) else []:
                if b.get("type")=="tool_use":uses.setdefault(b["id"],b["name"])
                if b.get("type")=="tool_result":
                    sig=(b.get("tool_use_id"),json.dumps(b,sort_keys=True))
                    if sig not in results:behavior["tool_result_chars"]+=len(content_text(b.get("content","")));results.add(sig)
        behavior.update(Counter(uses.values()));behavior["responses"]=len(ids);behavior["tools"]=len(uses)
        if runtime:assert sum(usage.values())>0
        group["behavior"].update(behavior);group["usage"].update(usage);group["dsl_usage"][session["dsl"]].update(usage)
        sessions.append({k:v for k,v in session.items() if k!="actions"}|{"behavior":behavior,"usage":usage,"total_tokens":sum(usage.values())})
        if number%25==0:print(f"Analyzed {number}/{len(index['sessions'])}",flush=True)
    for name,g in groups.items():
        ss=[s for s in sessions if s["group"]==name]
        g["response_median"]=statistics.median(s["behavior"]["responses"] for s in ss)
        g["token_median"]=statistics.median(s["total_tokens"] for s in ss) if name!="AKA" else None
    return {"method":{"tokenizer":"tiktoken=="+tiktoken.__version__,"encoding":"o200k_base","replay":"preserved_messages",
        "counts":"Per-chain message.id; tool-use ID and identical result dedup; provider terminal modelUsage for Runtime",
        "comparison":"AKA archived Episodes 1-19 vs isolated, pooled, and retained Epochs 1-10; excludes Bootstrap; raw archives read-only"},
        "groups":groups,"sessions":sessions,"examples":{f"{g}/{b}":sorted(v,key=lambda r:-r["tokens"])[:5] for (g,b),v in examples.items()}}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--archive-root",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();index=build_index(a.archive_root,Path(__file__).with_name("bash-action-index.json.gz"))
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/"bash-action-index.json.gz").write_bytes(gzip.compress(json.dumps(index).encode(),mtime=0))
    result=measure(index,a.archive_root)
    (a.output/"comparison.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")


if __name__=="__main__":main()
