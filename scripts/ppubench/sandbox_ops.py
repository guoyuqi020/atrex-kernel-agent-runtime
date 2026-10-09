#!/usr/bin/env python3
"""Synchronize exact commits and launch a prepared PPU task in an existing sandbox."""

from __future__ import annotations

import argparse
import fcntl
import importlib.metadata
import json
import os
import re
import secrets
import shlex
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

DEFAULT_TASK = Path("data/ppubench/gated-residual-combine-cuda")


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.chmod(0o600)
    temporary.replace(path)


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def active_runtime_pids():
    result = []
    for process in Path("/proc").iterdir():
        if not process.name.isdecimal():
            continue
        try:
            if process.joinpath("stat").read_text().rsplit(")", 1)[1].split()[0] == "Z":
                continue
            args = process.joinpath("cmdline").read_bytes().decode(errors="replace").split("\0")
            if any("atrex-kernel-agent-runtime" in arg for arg in args) and any(
                arg in {"serve", "run-campaign", "bootstrap"} for arg in args
            ):
                result.append(int(process.name))
        except (OSError, IndexError):
            continue
    return result


def inspect(args):
    repos = {}
    for name, relative in (
        ("runtime", "."),
        ("core", "src/atrex-kernel-agent-core"),
        ("kda", "src/kernel-design-agents"),
        ("evolver", "src/atrex-kernel-agent-evolver"),
        ("bench", "third_party/atrex-bench"),
    ):
        path = args.root / relative
        repos[name] = {
            "head": git(path, "rev-parse", "HEAD"),
            "changes": git(path, "status", "--porcelain", "--untracked-files=no"),
        }
    result = {
        "repos": repos,
        "active_runtime_pids": active_runtime_pids(),
        "workspace_exists": args.workspace.exists(),
        "agate_client": importlib.metadata.version("atrex-gateway-client"),
    }
    print(json.dumps(result, indent=2))


def sync(args):
    if active_runtime_pids():
        raise RuntimeError(
            "Runtime/Campaign processes remain active; refusing in-place source update"
        )
    manifest = read(args.record / "source-sync.json")
    for name in ("core", "kda", "runtime"):
        item = manifest[name]
        repo = args.root / item["path"]
        dirty = git(
            repo, "status", "--porcelain", "--untracked-files=no", "--ignore-submodules=all"
        )
        if dirty:
            raise RuntimeError(f"{name} has tracked modifications; refusing overwrite")
        current = git(repo, "rev-parse", "HEAD")
        if current != item["head"]:
            if current != item["base"]:
                raise RuntimeError(f"{name} unexpected HEAD {current}; inspect before syncing")
            subprocess.run(
                ["git", "-C", str(repo), "fetch", str(args.record / item["bundle"]), "HEAD"],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(repo), "merge", "--ff-only", item["head"]],
                check=True,
                stdout=subprocess.DEVNULL,
            )
        if git(repo, "rev-parse", "HEAD") != item["head"]:
            raise RuntimeError(f"{name} sync verification failed")
    write(args.record / "source-synced.json", manifest)
    inspect(args)


def frozen_launch_settings(args):
    """Use this run's frozen task, never newer task templates, for launch decisions."""
    frozen = args.workspace / "task-definition"
    definition = read(frozen / "task.json")
    template = read(frozen / "campaign.template.json")
    campaign = read(args.workspace / "campaign.json")
    runtime = read(args.workspace / "runtime.json")
    dsl = definition["dsl"]
    if (
        campaign["operator"] != definition["operator"]
        or campaign["hardware_target"] != template["hardware_target"]
        or list(campaign["lineages"]) != [dsl]
        or campaign["lineages"][dsl]["models"] != template["lineages"][dsl]["models"]
    ):
        raise RuntimeError("Prepared Campaign differs from its frozen task")
    agate = definition["runtime"]["agate"]
    for key in ("base_url", "auth_mode", "access_key_env", "secret_key_env"):
        if runtime["agate"].get(key) != agate[key]:
            raise RuntimeError(f"Runtime Agate {key} differs from its frozen task")
    if agate["auth_mode"] != "ak_sk":
        raise RuntimeError("Sandbox launcher requires a task configured for Agate AK/SK")
    timeout = definition["outer_timeout_seconds"]
    if type(timeout) is not int or timeout <= 0:
        raise RuntimeError("Frozen outer_timeout_seconds must be a positive integer")
    port = runtime["server"]["port"]
    if args.port is not None and args.port != port:
        raise RuntimeError("Requested port differs from the prepared Runtime port")
    return definition, campaign, runtime, timeout


def load_env(args, definition, campaign):
    env = os.environ.copy()
    env["PATH"] = f"/opt/node/bin:{args.root}/.venv/bin:/usr/local/bin:/usr/bin:/bin"
    task_runtime = definition["runtime"]
    home = Path(task_runtime["host_home"])
    agate = task_runtime["agate"]
    credential_names = (agate["access_key_env"], agate["secret_key_env"])
    # Existing, previously provisioned credentials are used only in child environments.
    for line in (home / ".config/atrex-runtime/agate.env").read_text().splitlines():
        match = re.match(r"(?:export\s+)?([A-Za-z_][A-Za-z_0-9]*)\s*=\s*(.*)", line.strip())
        if match and match.group(1) in credential_names:
            values = shlex.split(match.group(2), comments=True)
            if len(values) == 1:
                env[match.group(1)] = values[0]
    if not all(env.get(key) for key in credential_names):
        raise RuntimeError("Provisioned Agate credentials are missing")
    settings = read(home / ".claude/settings.json")
    model_env = settings.get("env", {})
    models = set(campaign["lineages"][definition["dsl"]]["models"].values())
    if models != {model_env.get("ANTHROPIC_MODEL")}:
        raise RuntimeError("Provisioned Claude model differs from the frozen Campaign")
    if not model_env.get("ANTHROPIC_BASE_URL") or not any(
        model_env.get(key) for key in ("ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY")
    ):
        raise RuntimeError("Provisioned Claude endpoint/credential is missing")
    env.update({key: str(value) for key, value in model_env.items()})
    env.update(AGATE_GPU=campaign["hardware_target"], AGATE_URL=agate["base_url"])
    env.pop("AGATE_TOKEN", None)
    auth_path = args.workspace / "runtime-secrets.json"
    if not auth_path.exists():
        write(
            auth_path,
            {
                key: secrets.token_hex(32)
                for key in ("ATREX_CAPABILITY_SIGNING_KEY", "ATREX_ADMIN_BEARER_TOKEN")
            },
        )
    env.update(read(auth_path))
    return env


def launch(args):
    repo_script = args.root / "scripts/ppubench/sandbox_ops.py"
    if Path(__file__).resolve() != repo_script.resolve():
        raise RuntimeError(
            "After syncing, invoke launch using the script in the Runtime repository"
        )
    args.record.mkdir(parents=True, exist_ok=True)
    python = args.root / ".venv/bin/python"
    cli = args.root / ".venv/bin/atrex-kernel-agent-runtime"
    campaign_script = args.root / "scripts/ppubench/campaign.py"
    with (args.record / ".launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (args.record / "launched.json").exists():
            raise RuntimeError("Launch already recorded; inspect instead of starting twice")
        if not (args.workspace / "prepared.json").exists():
            port = args.port
            if port is None:
                port = read(args.task_dir / "policy.json")["runtime"]["port"]
            command = [
                str(python),
                str(campaign_script),
                "prepare",
                "--root",
                str(args.root),
                "--workspace",
                str(args.workspace),
                "--task-dir",
                str(args.task_dir),
                "--port",
                str(port),
            ]
            if args.creation_key is not None:
                command.extend(["--creation-key", args.creation_key])
            subprocess.run(command, check=True)
        definition, campaign, runtime_config, timeout = frozen_launch_settings(args)
        port = runtime_config["server"]["port"]
        host = runtime_config["server"]["host"]
        with socket.socket() as probe:
            if probe.connect_ex((host, port)) == 0:
                raise RuntimeError("Requested Runtime port is already in use")
        env = load_env(args, definition, campaign)

        def start(argv, logfile):
            with logfile.open("ab", buffering=0) as output:
                return subprocess.Popen(
                    ["timeout", "--signal=TERM", "--kill-after=30s", str(timeout), *argv],
                    cwd=args.root,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )

        runtime = start(
            [str(cli), "serve", "--config", str(args.workspace / "runtime.json")],
            args.record / "runtime.log",
        )
        for _ in range(30):
            if runtime.poll() is not None:
                raise RuntimeError("Runtime exited; inspect runtime.log")
            try:
                with urllib.request.urlopen(f"http://{host}:{port}/healthz", timeout=1) as response:
                    if response.status == 200:
                        break
            except OSError:
                time.sleep(1)
        else:
            raise RuntimeError("Runtime health check failed")
        runner = start(
            [
                str(python),
                str(campaign_script),
                "run",
                "--root",
                str(args.root),
                "--workspace",
                str(args.workspace),
            ],
            args.record / "campaign.log",
        )
        time.sleep(3)
        if runner.poll() is not None:
            raise RuntimeError("Campaign runner exited; inspect campaign.log")
        record = {
            **read(args.workspace / "prepared.json"),
            "workspace": str(args.workspace),
            "runtime_pid": runtime.pid,
            "runner_pid": runner.pid,
            "outer_timeout_seconds": timeout,
            "started_at": time.time(),
            "port": port,
        }
        write(args.record / "launched.json", record)
        print(json.dumps(record, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inspect", "sync", "launch"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--task-dir", type=Path, default=DEFAULT_TASK)
    parser.add_argument("--port", type=int)
    parser.add_argument("--creation-key")
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.record = args.record.resolve()
    args.workspace = args.workspace.resolve()
    args.task_dir = (args.root / args.task_dir).resolve()
    os.umask(0o077)
    {"inspect": inspect, "sync": sync, "launch": launch}[args.action](args)
