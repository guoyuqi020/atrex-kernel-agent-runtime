"""Materialize one example-owned Runtime config and Campaign definition."""

from __future__ import annotations

import argparse
import json
import os
import pwd
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

from atrex_runtime.bootstrap import CampaignSpecV3
from atrex_runtime.config import RuntimeSettings


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--runtime-template", type=Path, required=True)
    parser.add_argument("--campaign-template", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    return parser.parse_args()


def _required_environment(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"{name} must be exported before preparing Bootstrap")
    return value


def _integer_environment(name: str, default: int) -> int:
    raw = os.environ.get(name)
    value = default if raw is None else int(raw)
    if value <= 0:
        raise SystemExit(f"{name} must be positive")
    return value


def _nonnegative_integer_environment(name: str, default: int) -> int:
    raw = os.environ.get(name)
    value = default if raw is None else int(raw)
    if value < 0:
        raise SystemExit(f"{name} must be non-negative")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(path)


def _git_commit(repository: Path) -> str:
    result = subprocess.run(
        ("/usr/bin/git", "-C", str(repository), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise SystemExit(f"Agent repository did not resolve to a full commit: {repository}")
    return commit


def _runtime_config(
    root: Path,
    state: Path,
    agate_url: str,
    template_path: Path,
) -> dict[str, Any]:
    template = cast(
        dict[str, Any],
        json.loads(template_path.read_text(encoding="utf-8")),
    )
    server_template = cast(dict[str, Any], template["server"])
    host = os.environ.get("ATREX_RUNTIME_HOST", str(server_template["host"]))
    port = _integer_environment("ATREX_RUNTIME_PORT", int(server_template["port"]))
    runtime_state = state / "runtime-state"

    template["server"] = {"host": host, "port": port}
    template["storage"] = {
        "registry_database": str(runtime_state / "registry.sqlite"),
        "gateway_database": str(runtime_state / "gateway.sqlite"),
        "agate_jobs_database": str(runtime_state / "agate-jobs.sqlite"),
        "artifacts_root": str(runtime_state / "artifacts"),
    }
    template["agate"] = {
        "base_url": agate_url,
        "auth_mode": "ak_sk",
        "access_key_env": "AGATE_AK",
        "secret_key_env": "AGATE_SK",
        "http_timeout_s": _integer_environment("AGATE_HTTP_TIMEOUT", 1800),
        "wait_timeout_s": _integer_environment("AGATE_WAIT_TIMEOUT", 3900),
        "health_check_interval_s": _integer_environment("AGATE_HEALTH_CHECK_INTERVAL", 30),
    }
    template["kernel_agent"]["base_source"]["repository"] = str(
        root / "src/kernel-design-agents"
    )
    template["kernel_agent"]["base_source"]["git_executable"] = "/usr/bin/git"

    campaign = template["campaign"]
    campaign["attempt_workspaces_root"] = str(runtime_state / "attempt-workspaces")
    campaign["evolution_workspaces_root"] = str(runtime_state / "evolution-workspaces")
    campaign["problem_generalization_workspaces_root"] = str(
        runtime_state / "problem-generalization-workspaces"
    )
    campaign["lineage_bootstrap_workspaces_root"] = str(
        runtime_state / "lineage-bootstrap-workspaces"
    )
    campaign["gateway_proxy_url"] = f"http://{host}:{port}"
    launcher = cast(dict[str, Any], campaign["launcher"])
    sandbox = launcher.get("sandbox")
    if isinstance(sandbox, dict):
        sandbox["worker_user"] = os.environ.get(
            "ATREX_SANDBOX_WORKER_USER",
            os.environ.get("SUDO_USER") or pwd.getpwuid(os.getuid()).pw_name,
        )
    worker_python = str(Path(sys.executable).resolve())
    campaign["optimizer"]["command_prefix"] = [worker_python]
    optimizer_backend = os.environ.get("ATREX_OPTIMIZER_AGENT_BACKEND", "").strip()
    if optimizer_backend:
        if optimizer_backend not in {"claude", "codex", "qodercli", "pi"}:
            raise SystemExit(f"unsupported ATREX_OPTIMIZER_AGENT_BACKEND: {optimizer_backend}")
        campaign["optimizer"]["agent_backend"] = optimizer_backend
    evolver_backend = os.environ.get("ATREX_EVOLVER_AGENT_BACKEND", "").strip()
    if evolver_backend:
        if evolver_backend not in {"claude", "codex", "qodercli", "pi"}:
            raise SystemExit(f"unsupported ATREX_EVOLVER_AGENT_BACKEND: {evolver_backend}")
        campaign["evolver"]["agent_backend"] = evolver_backend
    campaign["optimizer"]["max_session_tokens"] = _integer_environment(
        "ATREX_OPTIMIZER_MAX_SESSION_TOKENS", 20_000_000
    )
    campaign["optimizer"]["max_session_credits"] = _integer_environment(
        "ATREX_OPTIMIZER_MAX_SESSION_CREDITS", 1_000_000
    )
    campaign["evolver"]["repository"] = str(root / "src/atrex-kernel-agent-evolver")
    campaign["evolver"]["git_executable"] = "/usr/bin/git"
    campaign["evolver"]["command_prefix"] = [worker_python]

    wiki_url = os.environ.get("ATREX_WIKI_URL", "").strip()
    if wiki_url:
        template["gpu_wiki"]["base_url"] = wiki_url
        wiki_token_env = os.environ.get("ATREX_WIKI_TOKEN_ENV", "").strip()
        if wiki_token_env:
            template["gpu_wiki"]["bearer_token_env"] = wiki_token_env
    else:
        template.pop("gpu_wiki", None)
    return template


def _resolve_template_path(template: Path, raw: object, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise SystemExit(f"Campaign template requires {label}")
    path = Path(raw)
    return path if path.is_absolute() else (template.parent / path).resolve()


def _campaign_inputs(
    root: Path,
    state: Path,
    gpu: str,
    template_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    spec = cast(
        dict[str, Any],
        json.loads(template_path.read_text(encoding="utf-8")),
    )
    contract_source = _resolve_template_path(
        template_path,
        spec.get("evaluation_contract"),
        "evaluation_contract",
    )
    contract = cast(
        dict[str, Any],
        json.loads(contract_source.read_text(encoding="utf-8")),
    )
    contract_path = state / "evaluation-contract.json"
    _write_json(contract_path, contract)

    core_commit = _git_commit(root / "src/kernel-design-agents")
    gpu_key = re.sub(r"[^a-z0-9]+", "-", gpu.lower()).strip("-")
    template_key = re.sub(
        r"[^a-z0-9]+",
        "-",
        str(spec["creation_key"]).lower(),
    ).strip("-")
    default_key = f"{template_key}-{gpu_key}-{core_commit[:12]}"
    spec["creation_key"] = os.environ.get("ATREX_BOOTSTRAP_CREATION_KEY", default_key)
    spec["hardware_target"] = gpu
    spec["evaluation_contract"] = str(contract_path)
    if spec.get("agent_problem") is not None:
        spec["agent_problem"] = str(
            _resolve_template_path(template_path, spec["agent_problem"], "agent_problem")
        )
    spec["base_revision"]["commit"] = core_commit
    spec["challenger_count"] = _nonnegative_integer_environment(
        "ATREX_CHALLENGER_COUNT",
        cast(int, spec["challenger_count"]),
    )
    spec["challenger_start_epoch"] = _integer_environment(
        "ATREX_CHALLENGER_START_EPOCH",
        cast(int, spec["challenger_start_epoch"]),
    )
    spec["trajectories_per_branch"] = _integer_environment(
        "ATREX_TRAJECTORIES_PER_BRANCH",
        cast(int, spec["trajectories_per_branch"]),
    )
    spec["attempts_per_trajectory"] = _integer_environment(
        "ATREX_ATTEMPTS_PER_TRAJECTORY",
        cast(int, spec["attempts_per_trajectory"]),
    )
    lineages = cast(dict[str, dict[str, Any]], spec["lineages"])
    for dsl, lineage in lineages.items():
        lineage["baseline_kernel"] = str(
            _resolve_template_path(
                template_path,
                lineage.get("baseline_kernel"),
                f"lineages.{dsl}.baseline_kernel",
            )
        )
        lineage["initial_evidence"] = str(
            _resolve_template_path(
                template_path,
                lineage.get("initial_evidence"),
                f"lineages.{dsl}.initial_evidence",
            )
        )
    return spec, contract


def main() -> None:
    arguments = _arguments()
    root = Path(__file__).resolve().parents[2]
    state = arguments.state_dir.resolve()
    agate_url = _required_environment("AGATE_URL")
    gpu = _required_environment("AGATE_GPU")
    config_path = arguments.config.resolve()
    campaign_path = arguments.campaign.resolve()
    runtime_template = arguments.runtime_template.resolve()
    campaign_template = arguments.campaign_template.resolve()
    pinned_evolver_commit: str | None = None
    pinned_kernel_agent: dict[str, Any] | None = None
    if config_path.is_file():
        existing_settings = RuntimeSettings.from_file(config_path)
        if campaign_path.is_file():
            pinned_kernel_agent = existing_settings.kernel_agent.model_dump(mode="json")
        if existing_settings.campaign is not None:
            pinned_evolver_commit = existing_settings.campaign.evolver.commit
    config = _runtime_config(root, state, agate_url, runtime_template)
    if pinned_kernel_agent is not None:
        # A frozen commit must keep its repository/Skill allowlist, too.
        config["kernel_agent"] = pinned_kernel_agent
    if pinned_evolver_commit is not None:
        config["campaign"]["evolver"]["commit"] = pinned_evolver_commit
    _write_json(config_path, config)
    if campaign_path.is_file():
        # A generated Campaign definition is the durable identity for this
        # workspace. In particular, do not silently switch Lineages when the
        # checked-out Core HEAD changes between resumptions.
        spec = cast(
            dict[str, Any],
            json.loads(campaign_path.read_text(encoding="utf-8")),
        )
        spec_source = "pinned existing workspace definition"
    else:
        spec, _contract = _campaign_inputs(
            root,
            state,
            gpu,
            campaign_template,
        )
        _write_json(campaign_path, spec)
        spec_source = "new workspace definition"
    RuntimeSettings.from_file(config_path)
    CampaignSpecV3.from_file(campaign_path)
    print(f"Runtime config: {config_path}")
    print(f"Campaign definition: {campaign_path}")
    print(f"Campaign identity: {spec['creation_key']} ({spec_source})")
    print(f"Optimizer commit: {spec['base_revision']['commit']}")
    print(
        f"Evolver commit: {config['campaign']['evolver']['commit']}"
        + (" (pinned existing workspace config)" if pinned_evolver_commit else "")
    )
    print(f"Remote Agate: {agate_url} ({gpu})")
    optimizer = config["campaign"]["optimizer"]
    if optimizer["agent_backend"] == "qodercli":
        print(f"Optimizer quota per Core session: {optimizer['max_session_credits']} Qoder credits")
    else:
        print(
            f"Optimizer quota per Core session: {optimizer['max_session_tokens']} provider tokens"
        )
    print(
        "Agent backends: "
        f"optimizer={config['campaign']['optimizer']['agent_backend']}, "
        f"evolver={config['campaign']['evolver']['agent_backend']}"
    )


if __name__ == "__main__":
    main()
