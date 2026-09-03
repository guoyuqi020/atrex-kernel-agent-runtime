"""Offline Dev audit: read archived files/SQLite; never execute Agent commands.

Only writes derived JSON to --output. The frozen sibling Bash index defines the
114 AKA / 120 retained sessions; no dependence on temporary analysis caches.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import shlex
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def text_content(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(v.get("text", "") for v in value if isinstance(v, dict))
    return json.dumps(value, ensure_ascii=False)


def extract_tools(payload, runtime):
    tools, results, seen = {}, defaultdict(list), set()
    for line_no, line in enumerate(payload.decode().splitlines(), 1):
        event = json.loads(line)
        if runtime:
            event = event.get("event", {})
        message = event.get("message", {})
        if not isinstance(message, dict) or not isinstance(message.get("content"), list):
            continue
        for block in message["content"]:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                tools.setdefault(block["id"], {"id": block["id"], "line": line_no,
                    "name": block.get("name"), "input": block.get("input", {})})
            elif block.get("type") == "tool_result":
                signature = (block.get("tool_use_id"), json.dumps(block, sort_keys=True))
                if signature not in seen:
                    seen.add(signature)
                    results[block.get("tool_use_id")].append({"line": line_no,
                        "text": text_content(block.get("content", "")),
                        "is_error": block.get("is_error", False)})
    for tid, tool in tools.items():
        tool["results"] = results[tid]
    return list(tools.values())


def linked_results(tool, tools):
    initial = "\n".join(r["text"] for r in tool["results"])
    task_ids = set(re.findall(r"(?:background with ID: |task_id[^\w]+)([a-zA-Z0-9_-]+)", initial))
    output_paths = set(re.findall(r"(/tmp/[^\s\"]+/tasks/[a-zA-Z0-9_-]+\.output)", initial))
    linked = []
    for other in tools:
        inp = other["input"]
        if other["line"] <= tool["line"]:
            continue
        exact_task = inp.get("task_id") in task_ids and bool(task_ids)
        exact_file = inp.get("file_path") in output_paths and bool(output_paths)
        shell_read = other["name"] == "Bash" and any(p in inp.get("command", "") for p in output_paths)
        if exact_task or exact_file or shell_read:
            linked.extend(other["results"])
    return tool["results"] + linked, bool(task_ids or output_paths)


def shell_commands(command):
    """Conservative lexical simple commands, excluding heredoc bodies.

    Does not expand loops, substitute variables, run Python, or open wrappers.
    """
    lines, shell, i = command.splitlines(), [], 0
    while i < len(lines):
        line = lines[i]
        shell.append(line)
        i += 1
        for _, end in re.findall(r"<<-?\s*(['\"]?)([A-Za-z_][\w.-]*)\1", line):
            while i < len(lines) and lines[i].strip() != end:
                i += 1
            i += 1
    try:
        lexer = shlex.shlex("\n".join(shell), posix=True, punctuation_chars="();<>|&\n")
        lexer.whitespace = " \t\r"
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return []
    parts, current = [], []
    for token in tokens + [";"]:
        if token and all(c in "();|&\n" for c in token):
            if current:
                parts.append(current)
            current = []
        else:
            current.append(token)
    found = []
    for body in parts:
        while body and (body[0] in ("do", "then", "else", "if", "while", "until", "!", "{", "time", "command", "exec", "nohup") or re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", body[0])):
            body.pop(0)
        if body and body[0] == "env":
            body.pop(0)
            while body and (body[0].startswith("-") or re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", body[0])):
                body.pop(0)
        if not body:
            continue
        program, args = body[0].rsplit("/", 1)[-1], body[1:]
        if re.fullmatch(r"python[\d.]*|bash|sh|zsh", program) and args and not args[0].startswith("-"):
            program, args = args[0].rsplit("/", 1)[-1], args[1:]
        found.append((program, args))
    return found


def option(args, name):
    if name in args and args.index(name) + 1 < len(args):
        return args[args.index(name) + 1]
    return next((x[len(name)+1:] for x in args if x.startswith(name + "=")), None)


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError, UnicodeError):
        return None


def workspace_file(workspace, name):
    name = name.removeprefix("/home/agent/workspace/").removeprefix("./")
    p = (workspace / name).resolve()
    if not p.is_relative_to(workspace.resolve()):
        return None
    return p if p.is_file() else None


def failure_tag(value):
    patterns = [
        ("missing_module", r"ModuleNotFoundError|No module named"),
        ("missing_file", r"FileNotFoundError|can't open file|No such file or directory"),
        ("api_mismatch", r"AttributeError|TypeError|too many values to unpack|not enough values to unpack"),
        ("assertion", r"AssertionError"),
        ("illegal_memory_access", r"illegal memory access|misaligned address|device-side assert"),
        ("compilation", r"CompilationError|NVRTC_ERROR|compilation failed|ptxas.*error|error:.*(?:asm|identifier|type)"),
        ("timeout", r"[Tt]imeout|[Tt]imed out"),
        ("out_of_memory", r"out of memory|OutOfMemoryError"),
        ("schema_validation", r"source validation failed|validation_error|protected and cannot be set"),
        ("other_exception", r"Traceback|RuntimeError|ValueError|SyntaxError"),
    ]
    return next((name for name, pattern in patterns if re.search(pattern, value)), "other_or_not_visible")


def audit(archive, index):
    state = archive / "runtime/workspace-full-20260902.unpacked/production/control-l20n/state"
    def connection(name):
        db = sqlite3.connect((state / name).as_uri() + "?mode=ro", uri=True)
        db.row_factory = sqlite3.Row
        return db
    registry, gateway = connection("registry.sqlite"), connection("gateway.sqlite")
    workers = {r["id"]: dict(r) for r in registry.execute("select * from worker_sessions")}
    attempts_table = {r["id"]:dict(r) for r in registry.execute("select * from attempts")}
    epochs = {r["id"]:dict(r) for r in registry.execute("select * from epochs")}
    operations = defaultdict(list)
    for row in gateway.execute("select * from gateway_operations where operation='dev' order by created_at"):
        operations[row["attempt_id"]].append(dict(row))
    def artifact(value):
        if not value:
            return None
        return read_json(state / "artifacts/sha256" / value.split(":")[1] / "payload/value.json")

    runtime_rows, aka_rows, file_inventory, trace_inventory = [], [], [], []
    diagnostics = Counter()
    for session in index["sessions"]:
        trace = archive / session["trace"]
        payload = trace.read_bytes()
        if digest(payload) != session["trace_sha256"]:
            raise ValueError(f"Trace changed: {trace}")
        tools = extract_tools(payload, session["group"] == "retained")
        bash = [t for t in tools if t["name"] == "Bash"]
        diagnostics[session["group"]+"_sessions"] += 1
        trace_inventory.append({k:session[k] for k in ("trace","trace_sha256","session","dsl","group")})
        if session["group"] == "retained":
            worker = workers[session["session"]]
            attempt = attempts_table[worker["attempt_id"]]
            workspace = archive / "runtime/workspace-full-20260902.unpacked" / worker["workspace_path"].split("/workspaces/", 1)[1]
            if not workspace.exists():
                raise ValueError(f"Missing retained workspace: {workspace}")
            local_requests = {}
            for path in (workspace / "scratch").rglob("*.json"):
                obj = read_json(path)
                if isinstance(obj, dict) and obj.get("operation") == "dev":
                    local_requests[path.relative_to(workspace).as_posix()] = obj
                    file_inventory.append({"attempt_id":worker["attempt_id"],"dsl":session["dsl"],
                        "path":path.relative_to(archive).as_posix(),"sha256":digest(path.read_bytes()),
                        "command":obj.get("command"),"file_paths":obj.get("file_paths",[]),
                        "intent":obj.get("intent")})
            # Request arguments + direct result or a redirected result file link to job IDs.
            calls = []
            for tool in bash:
                command = tool["input"].get("command", "")
                for program, args in shell_commands(command):
                    if program != "runtime_tools.py" or not args or args[0] != "gateway-execute" or "--help" in args:
                        continue
                    request_arg = option(args, "--request")
                    if not request_arg:
                        continue
                    request_path = workspace_file(workspace, request_arg)
                    request = read_json(request_path) if request_path else None
                    if not isinstance(request,dict) or request.get("operation") != "dev":
                        continue
                    linked, _ = linked_results(tool, tools)
                    results = "\n".join(r["text"] for r in linked)
                    job_ids = set(re.findall(r"\bdv_[a-zA-Z0-9]+\b",results))
                    link_method = "tool_result_or_exact_background_task"
                    for i, value in enumerate(args[:-1]):
                        if value not in (">", ">>") or (i and args[i-1] in ("2","1")):
                            continue
                        result_path = workspace_file(workspace,args[i+1])
                        later_uses_same_output = any(t["line"] > tool["line"] and
                            args[i+1] in t["input"].get("command", "") and
                            "gateway-execute" in t["input"].get("command", "") for t in bash)
                        if result_path and not later_uses_same_output:
                            saved = result_path.read_text(errors="replace")
                            for jid in re.findall(r"\bdv_[a-zA-Z0-9]+\b",saved):
                                if jid not in job_ids:
                                    job_ids.add(jid)
                                    link_method = "direct_or_final_redirect_file"
                    calls.append({"line":tool["line"],"tool_id":tool["id"],"command":command,
                        "request_path":request_path.relative_to(archive).as_posix(),
                        "request":request,"job_ids":sorted(job_ids),"link_method":link_method,
                        "direct_result_lines":[r["line"] for r in linked]})
            for row in operations[worker["attempt_id"]]:
                result = artifact(row["gateway_result_digest"]) or {}
                inner = result.get("result") or {}
                if not isinstance(inner,dict):
                    inner = {}
                job_id = result.get("job_id")
                matches = [c for c in calls if job_id and job_id in c["job_ids"]]
                sources = []
                for call in matches:
                    for name in call["request"].get("file_paths",[]):
                        path = workspace_file(workspace,name)
                        if path:
                            sources.append({"requested_path":name,"archive_path":path.relative_to(archive).as_posix(),
                                "sha256":digest(path.read_bytes()),"bytes":path.stat().st_size})
                stdout = str(inner.get("stdout") or "")
                runtime_rows.append({**row,"dsl":session["dsl"],"session":session["session"],
                    "epoch":epochs[attempt["epoch_id"]]["number"],
                    "trajectory":attempt.get("trajectory_ordinal"),"iteration":attempt.get("iteration_ordinal"),
                    "trace":session["trace"],"workspace":workspace.relative_to(archive).as_posix(),
                    "job_id":job_id,"job_status":result.get("status"),"command_ok":result.get("command_ok"),
                    "exit_code":inner.get("exit_code"),"stdout":stdout,
                    "error":result.get("error"),"failure_tag":failure_tag(stdout),
                    "calls":matches,"files":sources})
            diagnostics["retained_dev_request_invocations_with_final_json"] += len(calls)
            diagnostics["retained_final_dev_request_files"] += len(local_requests)
        else:
            episode_match = re.search(r"production-e(\d+)-",session["trace"])
            episode = int(episode_match[1]) if episode_match else None
            instance = "AKA-2" if "/atrex-runs2.with-traces/" in session["trace"] else "AKA-1"
            for tool in bash:
                command = tool["input"].get("command", "")
                for program, args in shell_commands(command):
                    if program != "sandbox.py":
                        continue
                    prefix = args[:args.index("--")] if "--" in args else args
                    if "--help" in prefix or "-h" in prefix:
                        continue
                    kind = option(prefix, "--kind") or "auto"
                    if kind not in ("dev","profile","auto"):
                        continue
                    linked, background = linked_results(tool, tools)
                    output = "\n".join(r["text"] for r in linked)
                    routes = sorted(set(re.findall(r"\[sandbox\] gateway_kind=(\w+)",output)))
                    remote_command = args[args.index("--")+1:] if "--" in args else []
                    aka_rows.append({"instance":instance,"dsl":session["dsl"],"episode":episode,
                        "session":session["session"],"trace":session["trace"],"line":tool["line"],
                        "tool_id":tool["id"],"requested_kind":kind,"command":command,
                        "remote_argv_lexical":remote_command,"observed_routes":routes,
                        "fallback_to_dev":"profile interface unsupported" in output and "using dev" in output,
                        "dev_job_ids":sorted(set(re.findall(r"\bdv_[a-zA-Z0-9]+\b",output))),
                        "background":background,
                        "output_lines":sorted(set(r["line"] for r in linked)),
                        "output":output,"failure_tag":failure_tag(output)})
    diagnostics["retained_dev_records"] = len(runtime_rows)
    diagnostics["retained_dev_records_linked_to_request"] = sum(bool(r["calls"]) for r in runtime_rows)
    diagnostics["retained_dev_records_multiple_call_matches"] = sum(len(r["calls"])>1 for r in runtime_rows)
    summary = {
        "scope": {"archive_root":str(archive),"index_sha256":digest(json.dumps(index,sort_keys=True).encode()),"diagnostics":dict(diagnostics)},
        "retained": {
            "by_dsl":dict(Counter(r["dsl"] for r in runtime_rows)),
            "exit_codes":dict(Counter(str(r["exit_code"]) for r in runtime_rows)),
            "job_status":dict(Counter(str(r["job_status"]) for r in runtime_rows)),
            "unique_jobs":len({r["job_id"] for r in runtime_rows if r["job_id"]}),
            "nonzero_exit_failure_tags":dict(Counter(r["failure_tag"] for r in runtime_rows if r["exit_code"] not in (0,None))),
        },
        "aka": {
            "requested_kinds":dict(Counter(r["requested_kind"] for r in aka_rows)),
            "requested_kind_by_dsl":dict(Counter(r["dsl"]+"/"+r["requested_kind"] for r in aka_rows)),
            "observed_fallback_calls":sum(r["fallback_to_dev"] for r in aka_rows),
            "unique_observed_dev_jobs":len({jid for r in aka_rows for jid in r["dev_job_ids"]}),
            "calls_with_background":sum(r["background"] for r in aka_rows),
        },
    }
    registry.close(); gateway.close()
    return summary,runtime_rows,aka_rows,file_inventory,trace_inventory


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root",type=Path,required=True)
    parser.add_argument("--index",type=Path,default=Path(__file__).parent.parent/'analysis/bash-action-index.json.gz')
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    archive=args.archive_root.resolve()
    output=args.output.resolve()
    if output.is_relative_to(archive):
        raise ValueError("Do not write derived data inside the input archive")
    index=json.loads(gzip.decompress(args.index.read_bytes()))
    summary,retained,aka,files,traces=audit(archive,index)
    output.mkdir(parents=True,exist_ok=True)
    for name, value in (("summary.json",summary),("retained-dev.json",retained),("aka-dev-profile.json",aka),("scratch-dev-requests.json",files),("traces.json",traces)):
        encoded=json.dumps(value,ensure_ascii=False,indent=2).encode()
        if name in ("retained-dev.json","aka-dev-profile.json"):
            (output/(name+'.gz')).write_bytes(gzip.compress(encoded,mtime=0))
        else:
            (output/name).write_bytes(encoded+b'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
