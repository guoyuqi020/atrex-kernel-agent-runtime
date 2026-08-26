"""Campaign worker configuration and production composition tests."""

from __future__ import annotations

import base64
import json
import os
import subprocess
from pathlib import Path

import pytest

from atrex_runtime.composition.campaign import build_campaign_runtime
from atrex_runtime.config import RuntimeSettings, WorkerEnvironmentSettings
from atrex_runtime.domain.models import KernelMeasurementPurpose
from atrex_runtime.ports import KernelMeasurementRun, KernelPairMeasurementResult
from atrex_runtime.workers.evolution import (
    EvolutionSessionResult,
    PreparedEvolution,
)
from atrex_runtime.workers.optimizer import OptimizerSessionConfig, OptimizerSessionResult
from atrex_runtime.workers.workspace import PreparedAttempt


class UnusedOptimizerSessionDriver:
    """Fail if a composition-only test unexpectedly starts the Core Optimizer."""

    async def run(
        self,
        prepared: PreparedAttempt,
        config: OptimizerSessionConfig,
    ) -> OptimizerSessionResult:
        del prepared, config
        raise AssertionError("composition must not start an Optimizer")


class UnusedEvolutionSessionDriver:
    """Fail if a composition-only test unexpectedly starts an Evolver."""

    async def run(self, prepared: PreparedEvolution) -> EvolutionSessionResult:
        del prepared
        raise AssertionError("composition must not start an Evolver")


class UnusedKernelMeasurementRunner:
    """Fail if composition invokes repeated Evaluate before an epoch runs."""

    async def run(
        self,
        revision: object,
        repeat: int,
        purpose: KernelMeasurementPurpose,
    ) -> KernelMeasurementRun:
        del revision, repeat, purpose
        raise AssertionError("composition must not run repeated Evaluate")


class UnusedKernelPairMeasurementRunner:
    """Fail if composition invokes ABBA before an epoch runs."""

    async def run_pair(self, *args: object, **kwargs: object) -> KernelPairMeasurementResult:
        del args, kwargs
        raise AssertionError("composition must not run same-allocation ABBA")


def test_optional_worker_environment_forwards_only_present_values() -> None:
    settings = WorkerEnvironmentSettings(
        values={"STATIC": "value"},
        inherit=("REQUIRED",),
        inherit_optional=("CLAUDE_AUTH", "CODEX_HOME"),
    )

    assert settings.resolve(
        {
            "REQUIRED": "required",
            "CODEX_HOME": "/credentials/codex",
            "UNDECLARED": "hidden",
        }
    ) == (
        ("CODEX_HOME", "/credentials/codex"),
        ("REQUIRED", "required"),
        ("STATIC", "value"),
    )


def _commit_repository(root: Path) -> str:
    subprocess.run(("git", "init", "-q", str(root)), check=True)
    subprocess.run(("git", "-C", str(root), "add", "."), check=True)
    environment = {
        **os.environ,
        "GIT_AUTHOR_NAME": "ATREX Test",
        "GIT_AUTHOR_EMAIL": "atrex@example.invalid",
        "GIT_COMMITTER_NAME": "ATREX Test",
        "GIT_COMMITTER_EMAIL": "atrex@example.invalid",
    }
    subprocess.run(
        ("git", "-C", str(root), "commit", "-q", "-m", "test bundle"),
        check=True,
        env=environment,
    )
    return subprocess.run(
        ("git", "-C", str(root), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_config(tmp_path: Path) -> Path:
    (tmp_path / "bin").mkdir()
    executable = tmp_path / "bin/python"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o700)
    evolver_root = tmp_path / "evolver-bundle"
    (evolver_root / "src").mkdir(parents=True)
    (evolver_root / "src/main.py").write_text("raise SystemExit(0)\n", encoding="utf-8")
    (evolver_root / "atrex-evolver-bundle.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "bundle_format": "atrex-kernel-agent-evolver-bundle-v1",
                "entrypoint": {"command": "src/main.py"},
            }
        ),
        encoding="utf-8",
    )
    evolver_commit = _commit_repository(evolver_root)
    path = tmp_path / "runtime.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "server": {"host": "127.0.0.1", "port": 8765},
                "storage": {
                    "registry_database": "state/registry.sqlite",
                    "gateway_database": "state/gateway.sqlite",
                    "agate_jobs_database": "state/agate-jobs.sqlite",
                    "artifacts_root": "state/artifacts",
                },
                "gateway_proxy": {
                    "max_request_bytes": 65536,
                    "max_candidate_files": 16,
                    "max_candidate_bytes": 32768,
                    "capability_signing_key_env": "TEST_SIGNING_KEY",
                    "candidate_diff_allowed_paths": {
                        "cuda": ["*.cu"],
                        "triton": ["*.py"],
                        "cutedsl": ["*.py"],
                    },
                    "candidate_diff_require_change": True,
                },
                "agate": {
                    "base_url": "https://gateway.example.test",
                    "auth_mode": "none",
                    "http_timeout_s": 60,
                    "wait_timeout_s": 900,
                },
                "kernel_agent": {
                    "max_bundle_files": 1024,
                    "max_bundle_bytes": 8388608,
                    "max_entrypoint_bytes": 524288,
                    "max_agent_problem_bytes": 262144,
                },
                "campaign": {
                    "attempt_workspaces_root": "state/attempts",
                    "evolution_workspaces_root": "state/evolution",
                    "problem_generalization_workspaces_root": "state/generalization",
                    "lineage_bootstrap_workspaces_root": "state/bootstrap",
                    "fencing_lease_seconds": 120,
                    "fencing_heartbeat_seconds": 30,
                    "gateway_proxy_url": "http://127.0.0.1:8765",
                    "gateway_operations": ["evaluate", "profile"],
                    "gateway_max_calls": 8,
                    "gateway_capability_lifetime_seconds": 7200,
                    "gate_policy": {
                        "optimizer": {
                            "correctness_cases": 1,
                            "bench_iters": 100,
                            "evaluate_repeats": 1,
                        },
                        "bootstrap": {
                            "stages": [
                                {"correctness_cases": 1, "evaluate_repeats": 1},
                                {"correctness_cases": 5, "evaluate_repeats": 1},
                            ],
                            "bench_iters": 5,
                        },
                        "retention": {"correctness_cases": 1, "bench_iters": 100},
                        "warmup_iters": 10,
                        "atol": 0.01,
                        "rtol": 0.05,
                        "evaluation_timeout_seconds": 600,
                        "candidate_timeout_seconds": 120,
                        "performance_timeout_seconds": 120,
                        "lock_clocks": False,
                        "evaluator": {
                            "repository": "../atrex-bench",
                            "commit": "c" * 40,
                            "git_executable": "bin/git",
                            "fetch_timeout_seconds": 30,
                            "max_archive_bytes": 8388608,
                            "max_bundle_files": 64,
                            "max_bundle_bytes": 2097152,
                        },
                    },
                    "max_infrastructure_retries": 2,
                    "kernel_retention_comparison": {
                        "method": "evaluate",
                        "repeats": 1,
                        "measurement_uncertainty_us": 0.2,
                    },
                    "agent_promotion_comparison": {
                        "method": "evaluate",
                        "repeats": 1,
                        "measurement_uncertainty_us": 0.3,
                    },
                    "evidence": {
                        "max_trace_files": 8,
                        "max_trace_bytes": 1048576,
                        "max_trace_events": 10000,
                        "max_projection_text_bytes": 262144,
                        "max_diff_files": 64,
                        "max_diff_bytes": 1048576,
                        "redaction_patterns": [],
                    },
                    "optimizer": {
                        "command_prefix": ["bin/python"],
                        "environment": {
                            "values": {"DEPLOYMENT_LABEL": "test"},
                            "inherit": ["MODEL_API_KEY"],
                        },
                        "isolated_home_environment_keys": ["HOME"],
                        "session_trace_relative_path": "sessions/core",
                        "token_usage_report_relative_path": "scratch/token-usage.json",
                        "max_attempt_report_bytes": 65536,
                        "timeout_seconds": 1800,
                        "terminate_grace_seconds": 5,
                        "max_diagnostic_bytes": 65536,
                        "max_session_tokens": 100000,
                    },
                    "evolver": {
                        "repository": "./evolver-bundle",
                        "commit": evolver_commit,
                        "git_executable": "/usr/bin/git",
                        "fetch_timeout_seconds": 10,
                        "max_archive_bytes": 1048576,
                        "command_prefix": ["bin/python"],
                        "isolated_home_environment_keys": ["HOME"],
                        "session_trace_relative_path": None,
                        "token_usage_report_relative_path": "scratch/token-usage.json",
                        "environment": {"inherit": ["EVOLVER_API_KEY"]},
                        "timeout_seconds": 600,
                        "terminate_grace_seconds": 5,
                        "max_diagnostic_bytes": 65536,
                        "max_output_manifest_bytes": 16384,
                    },
                    "launcher": {
                        "mode": "development",
                        "env_executable": "/usr/bin/env",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _environment() -> dict[str, str]:
    return {
        "TEST_SIGNING_KEY": base64.urlsafe_b64encode(b"s" * 32).decode(),
        "MODEL_API_KEY": "model-secret",
        "EVOLVER_API_KEY": "evolver-secret",
        "UNDECLARED_AMBIENT_VALUE": "must-not-be-forwarded",
    }


def test_campaign_config_resolves_worker_paths_and_explicit_environment(tmp_path: Path) -> None:
    settings = RuntimeSettings.from_file(_write_config(tmp_path))
    campaign = settings.campaign
    assert campaign is not None

    assert campaign.attempt_workspaces_root == tmp_path / "state/attempts"
    assert campaign.evolution_workspaces_root == tmp_path / "state/evolution"
    assert campaign.problem_generalization_workspaces_root == tmp_path / "state/generalization"
    assert campaign.lineage_bootstrap_workspaces_root == tmp_path / "state/bootstrap"
    assert campaign.optimizer.command_prefix == (str(tmp_path / "bin/python"),)
    assert campaign.evolver.repository == str(tmp_path / "evolver-bundle")
    assert len(campaign.evolver.commit) == 40
    assert campaign.evolver.command_prefix == (str(tmp_path / "bin/python"),)
    assert campaign.optimizer.environment.resolve(_environment()) == (
        ("DEPLOYMENT_LABEL", "test"),
        ("MODEL_API_KEY", "model-secret"),
    )
    assert campaign.kernel_retention_comparison.method == "evaluate"
    assert campaign.kernel_retention_comparison.repeats == 1
    assert campaign.agent_promotion_comparison.method == "evaluate"
    assert campaign.agent_promotion_comparison.repeats == 1
    assert campaign.bootstrap_max_parallel_lineages == 1
    assert campaign.max_parallel_branches == 4


def test_campaign_config_requires_positive_parallel_branch_limit(tmp_path: Path) -> None:
    path = _write_config(tmp_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["campaign"]["max_parallel_branches"] = 0
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="max_parallel_branches"):
        RuntimeSettings.from_file(path)


def test_campaign_config_requires_positive_parallel_bootstrap_limit(tmp_path: Path) -> None:
    path = _write_config(tmp_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["campaign"]["bootstrap_max_parallel_lineages"] = 0
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="bootstrap_max_parallel_lineages"):
        RuntimeSettings.from_file(path)


def test_campaign_config_resolves_commit_pinned_roofline_builder(tmp_path: Path) -> None:
    path = _write_config(tmp_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["campaign"]["roofline_builder"] = {
        "repository": "../atrex-bench",
        "commit": "b" * 40,
        "git_executable": "bin/git",
        "python_executable": "bin/python-roofline",
        "fetch_timeout_seconds": 20,
        "execution_timeout_seconds": 30,
        "max_archive_bytes": 1000000,
        "max_output_bytes": 100000,
        "sku_by_hardware_target": {"L20N": "Test L20N"},
    }
    path.write_text(json.dumps(value), encoding="utf-8")

    campaign = RuntimeSettings.from_file(path).campaign

    assert campaign is not None and campaign.roofline_builder is not None
    builder = campaign.roofline_builder
    assert builder.repository == str((tmp_path / "../atrex-bench").resolve())
    assert builder.git_executable == tmp_path / "bin/git"
    assert builder.python_executable == tmp_path / "bin/python-roofline"
    assert builder.commit == "b" * 40
    assert builder.sku_by_hardware_target == {"L20N": "Test L20N"}


def test_config_resolves_local_core_repository_against_config_directory(tmp_path: Path) -> None:
    path = _write_config(tmp_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["kernel_agent"]["base_source"] = {
        "repository": "../core",
        "git_executable": "/usr/bin/git",
        "fetch_timeout_seconds": 10,
        "max_archive_bytes": 1024,
        "allowed_submodules": {},
    }
    path.write_text(json.dumps(value), encoding="utf-8")

    source = RuntimeSettings.from_file(path).kernel_agent.base_source

    assert source is not None
    assert source.repository == str((tmp_path / "../core").resolve())


def test_campaign_config_rejects_removed_comparison_methods(tmp_path: Path) -> None:
    path = _write_config(tmp_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["campaign"]["kernel_retention_comparison"] = {
        "method": "paired",
        "repeats": 2,
        "minimum_improvement_percent": 0.5,
    }
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="kernel_retention_comparison"):
        RuntimeSettings.from_file(path)


def test_campaign_config_accepts_commit_pinned_same_allocation_abba(tmp_path: Path) -> None:
    path = _write_config(tmp_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["campaign"]["gate_policy"]["evaluator"] = {
        "repository": "../atrex-bench",
        "commit": "c" * 40,
        "git_executable": "bin/git",
        "fetch_timeout_seconds": 30,
        "max_archive_bytes": 8388608,
        "max_bundle_files": 64,
        "max_bundle_bytes": 2097152,
    }
    value["campaign"]["kernel_retention_comparison"] = {
        "method": "same_allocation_abba",
        "repeats": 2,
        "minimum_improvement_percent": 0.5,
        "allocation_timeout_seconds": 600,
        "shape_batch_size": 4,
        "max_parallel_shape_batches": 3,
    }
    path.write_text(json.dumps(value), encoding="utf-8")

    campaign = RuntimeSettings.from_file(path).campaign

    assert campaign is not None
    assert campaign.gate_policy.evaluator.repository == str((tmp_path / "../atrex-bench").resolve())
    assert campaign.gate_policy.evaluator.git_executable == tmp_path / "bin/git"
    comparison = campaign.kernel_retention_comparison
    assert comparison.method == "same_allocation_abba"
    assert comparison.minimum_improvement_percent == 0.5


def test_campaign_config_controls_independent_evaluate_repetitions(tmp_path: Path) -> None:
    path = _write_config(tmp_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["campaign"]["gate_policy"]["optimizer"]["evaluate_repeats"] = 3
    value["campaign"]["gate_policy"]["bootstrap"]["stages"] = [
        {"correctness_cases": 2, "evaluate_repeats": 2}
    ]
    value["campaign"]["gate_policy"]["bootstrap"]["bench_iters"] = 7
    path.write_text(json.dumps(value), encoding="utf-8")

    campaign = RuntimeSettings.from_file(path).campaign

    assert campaign is not None
    assert campaign.gate_policy.optimizer.evaluate_repeats == 3
    assert campaign.gate_policy.bootstrap.stages[0].evaluate_repeats == 2
    assert campaign.gate_policy.bootstrap.bench_iters == 7


def test_campaign_config_uses_atrex_kernel_agent_gate_policy(tmp_path: Path) -> None:
    campaign = RuntimeSettings.from_file(_write_config(tmp_path)).campaign

    assert campaign is not None
    gate = campaign.gate_policy
    assert gate.optimizer.correctness_cases == 1
    assert gate.optimizer.bench_iters == 100
    assert [
        (stage.correctness_cases, stage.evaluate_repeats) for stage in gate.bootstrap.stages
    ] == [(1, 1), (5, 1)]
    assert gate.bootstrap.bench_iters == 5
    assert gate.retention.correctness_cases == 1
    assert gate.retention.bench_iters == 100
    assert gate.production_gate is False
    assert gate.candidate_timeout_seconds == 120
    assert gate.performance_timeout_seconds == 120
    assert gate.atol == 0.01
    assert gate.rtol == 0.05


def test_campaign_gate_policy_locks_clocks_by_default_and_allows_opt_out(
    tmp_path: Path,
) -> None:
    path = _write_config(tmp_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["campaign"]["gate_policy"].pop("lock_clocks")
    path.write_text(json.dumps(value), encoding="utf-8")

    campaign = RuntimeSettings.from_file(path).campaign

    assert campaign is not None
    assert campaign.gate_policy.lock_clocks is True

    value["campaign"]["gate_policy"]["lock_clocks"] = False
    path.write_text(json.dumps(value), encoding="utf-8")
    campaign = RuntimeSettings.from_file(path).campaign

    assert campaign is not None
    assert campaign.gate_policy.lock_clocks is False


def test_campaign_config_can_enable_production_gate(tmp_path: Path) -> None:
    path = _write_config(tmp_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["campaign"]["gate_policy"]["production_gate"] = True
    path.write_text(json.dumps(value), encoding="utf-8")

    campaign = RuntimeSettings.from_file(path).campaign

    assert campaign is not None
    assert campaign.gate_policy.production_gate is True


@pytest.mark.parametrize("name", ("atol", "rtol"))
def test_campaign_config_rejects_negative_evaluation_tolerance(
    tmp_path: Path,
    name: str,
) -> None:
    path = _write_config(tmp_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["campaign"]["gate_policy"][name] = -0.1
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match=name):
        RuntimeSettings.from_file(path)


def test_campaign_config_rejects_nonpositive_optimizer_evaluate_repetitions(
    tmp_path: Path,
) -> None:
    path = _write_config(tmp_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["campaign"]["gate_policy"]["optimizer"]["evaluate_repeats"] = 0
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="evaluate_repeats"):
        RuntimeSettings.from_file(path)


def test_campaign_config_rejects_nonpositive_bootstrap_evaluate_repetitions(
    tmp_path: Path,
) -> None:
    path = _write_config(tmp_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["campaign"]["gate_policy"]["bootstrap"]["stages"][0]["evaluate_repeats"] = 0
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="evaluate_repeats"):
        RuntimeSettings.from_file(path)


def test_campaign_config_rejects_nonpositive_bootstrap_bench_iterations(
    tmp_path: Path,
) -> None:
    path = _write_config(tmp_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["campaign"]["gate_policy"]["bootstrap"]["bench_iters"] = 0
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="bench_iters"):
        RuntimeSettings.from_file(path)


def test_campaign_config_rejects_removed_combined_evaluate_repetitions(
    tmp_path: Path,
) -> None:
    path = _write_config(tmp_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["campaign"]["ordinary_evaluate_repeats"] = 3
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="ordinary_evaluate_repeats"):
        RuntimeSettings.from_file(path)


def test_campaign_config_rejects_removed_runtime_final_evaluate_repetitions(
    tmp_path: Path,
) -> None:
    path = _write_config(tmp_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["campaign"]["runtime_final_evaluate_repeats"] = 3
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="runtime_final_evaluate_repeats"):
        RuntimeSettings.from_file(path)


def test_campaign_config_requires_evaluator_for_same_allocation_abba(tmp_path: Path) -> None:
    path = _write_config(tmp_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["campaign"]["kernel_retention_comparison"] = {
        "method": "same_allocation_abba",
        "repeats": 2,
        "minimum_improvement_percent": 0,
        "allocation_timeout_seconds": 600,
        "shape_batch_size": 4,
        "max_parallel_shape_batches": 2,
    }
    del value["campaign"]["gate_policy"]["evaluator"]
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="evaluator"):
        RuntimeSettings.from_file(path)


def test_campaign_config_keeps_agent_framework_inside_core(tmp_path: Path) -> None:
    path = _write_config(tmp_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["campaign"]["optimizer"] = {
        "command_prefix": ["bin/python"],
        "environment": {
            "values": {"ATREX_CORE_AGENT_CLI": "codex"},
            "inherit": ["MODEL_API_KEY"],
        },
        "isolated_home_environment_keys": ["HOME", "CODEX_HOME"],
        "session_trace_relative_path": "sessions/core",
        "token_usage_report_relative_path": "scratch/token-usage.json",
        "max_attempt_report_bytes": 65536,
        "timeout_seconds": 1800,
        "terminate_grace_seconds": 5,
        "max_diagnostic_bytes": 65536,
        "max_session_tokens": 100000,
    }
    path.write_text(json.dumps(value), encoding="utf-8")

    settings = RuntimeSettings.from_file(path)
    campaign = settings.campaign
    assert campaign is not None
    assert campaign.optimizer.command_prefix == (str(tmp_path / "bin/python"),)
    assert campaign.optimizer.environment.resolve(_environment()) == (
        ("ATREX_CORE_AGENT_CLI", "codex"),
        ("MODEL_API_KEY", "model-secret"),
    )

    runtime = build_campaign_runtime(
        settings,
        _environment(),
        optimizer_session_driver=UnusedOptimizerSessionDriver(),
        evolution_session_driver=UnusedEvolutionSessionDriver(),
    )
    runtime.close()


def test_campaign_config_rejects_removed_evolver_token_quota(tmp_path: Path) -> None:
    path = _write_config(tmp_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["campaign"]["evolver"]["max_session_tokens"] = 100_000
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="max_session_tokens"):
        RuntimeSettings.from_file(path)


def test_campaign_composition_builds_repeated_evaluate_runner(tmp_path: Path) -> None:
    path = _write_config(tmp_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["campaign"]["kernel_retention_comparison"]["repeats"] = 3
    path.write_text(json.dumps(value), encoding="utf-8")
    settings = RuntimeSettings.from_file(path)

    automatic = build_campaign_runtime(
        settings,
        _environment(),
        optimizer_session_driver=UnusedOptimizerSessionDriver(),
        evolution_session_driver=UnusedEvolutionSessionDriver(),
    )
    automatic.close()

    runtime = build_campaign_runtime(
        settings,
        _environment(),
        optimizer_session_driver=UnusedOptimizerSessionDriver(),
        evolution_session_driver=UnusedEvolutionSessionDriver(),
        measurement_runner=UnusedKernelMeasurementRunner(),
    )
    runtime.close()


def test_campaign_composition_builds_same_allocation_abba_runner(tmp_path: Path) -> None:
    path = _write_config(tmp_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["campaign"]["gate_policy"]["evaluator"] = {
        "repository": "../atrex-bench",
        "commit": "d" * 40,
        "git_executable": "bin/git",
        "fetch_timeout_seconds": 30,
        "max_archive_bytes": 8388608,
        "max_bundle_files": 128,
        "max_bundle_bytes": 4194304,
    }
    value["campaign"]["kernel_retention_comparison"] = {
        "method": "same_allocation_abba",
        "repeats": 2,
        "minimum_improvement_percent": 0,
        "allocation_timeout_seconds": 600,
        "shape_batch_size": 4,
        "max_parallel_shape_batches": 2,
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    settings = RuntimeSettings.from_file(path)

    automatic = build_campaign_runtime(
        settings,
        _environment(),
        optimizer_session_driver=UnusedOptimizerSessionDriver(),
        evolution_session_driver=UnusedEvolutionSessionDriver(),
    )
    automatic.close()

    injected = build_campaign_runtime(
        settings,
        _environment(),
        optimizer_session_driver=UnusedOptimizerSessionDriver(),
        evolution_session_driver=UnusedEvolutionSessionDriver(),
        pair_measurement_runner=UnusedKernelPairMeasurementRunner(),
    )
    injected.close()


def test_campaign_composition_owns_and_closes_durable_connections(tmp_path: Path) -> None:
    settings = RuntimeSettings.from_file(_write_config(tmp_path))

    runtime = build_campaign_runtime(
        settings,
        _environment(),
        optimizer_session_driver=UnusedOptimizerSessionDriver(),
        evolution_session_driver=UnusedEvolutionSessionDriver(),
    )
    runtime.close()
    runtime.close()

    replacement = build_campaign_runtime(
        settings,
        _environment(),
        optimizer_session_driver=UnusedOptimizerSessionDriver(),
        evolution_session_driver=UnusedEvolutionSessionDriver(),
    )
    replacement.close()


def test_campaign_composition_rejects_missing_explicit_inherited_environment(
    tmp_path: Path,
) -> None:
    settings = RuntimeSettings.from_file(_write_config(tmp_path))
    environment = _environment()
    del environment["MODEL_API_KEY"]

    with pytest.raises(ValueError, match="MODEL_API_KEY"):
        build_campaign_runtime(
            settings,
            environment,
            optimizer_session_driver=UnusedOptimizerSessionDriver(),
            evolution_session_driver=UnusedEvolutionSessionDriver(),
        )

    replacement = build_campaign_runtime(
        settings,
        _environment(),
        optimizer_session_driver=UnusedOptimizerSessionDriver(),
        evolution_session_driver=UnusedEvolutionSessionDriver(),
    )
    replacement.close()


def test_campaign_composition_defers_evolver_environment_until_challenger(
    tmp_path: Path,
) -> None:
    settings = RuntimeSettings.from_file(_write_config(tmp_path))
    environment = _environment()
    del environment["EVOLVER_API_KEY"]

    runtime = build_campaign_runtime(
        settings,
        environment,
        optimizer_session_driver=UnusedOptimizerSessionDriver(),
        evolution_session_driver=UnusedEvolutionSessionDriver(),
    )
    runtime.close()


def test_campaign_config_rejects_gateway_policy_without_evaluate(tmp_path: Path) -> None:
    path = _write_config(tmp_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["campaign"]["gateway_operations"] = ["profile"]
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="must include evaluate"):
        RuntimeSettings.from_file(path)


def test_campaign_composition_does_not_receive_gpu_wiki_service_secret(tmp_path: Path) -> None:
    path = _write_config(tmp_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["gpu_wiki"] = {
        "base_url": "https://wiki.example.test",
        "bearer_token_env": "GPU_WIKI_TOKEN",
        "timeout_seconds": 10,
        "max_proxy_request_bytes": 65536,
        "max_query_bytes": 32768,
        "max_response_bytes": 65536,
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    settings = RuntimeSettings.from_file(path)

    runtime = build_campaign_runtime(
        settings,
        _environment(),
        optimizer_session_driver=UnusedOptimizerSessionDriver(),
        evolution_session_driver=UnusedEvolutionSessionDriver(),
    )
    runtime.close()
