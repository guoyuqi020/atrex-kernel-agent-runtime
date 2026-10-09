"""The Wiki adapter changes only its own model/effort and never reads real credentials."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
URL = "https://example.invalid/anthropic"
MODEL = "qwen3.8-flash"
ERROR = "Wiki Claude bridge configuration is unavailable or invalid; refusing launch.\n"


@pytest.fixture
def bridge(tmp_path: Path) -> SimpleNamespace:
    wrapper = tmp_path / "claude"
    shutil.copyfile(ROOT / "scripts/claude", wrapper)
    wrapper.chmod(0o755)
    executable = tmp_path / "real-claude"
    marker = tmp_path / "executed"
    executable.write_text(
        f"#!{sys.executable}\nimport os,json,sys\nfrom pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('called')\n"
        "print(json.dumps({'argv':sys.argv[1:], 'env':dict(os.environ)}))\n"
    )
    executable.chmod(0o755)
    settings = tmp_path / "settings.json"
    provider = {
        "ANTHROPIC_BASE_URL": URL,
        "ANTHROPIC_MODEL": "qwen3.8-max",
        "ANTHROPIC_AUTH_TOKEN": "fake-configured-token-only",
        "ANTHROPIC_DEFAULT_SONNET_MODEL": "qwen3.8-max",
        "ANTHROPIC_DEFAULT_OPUS_MODEL": "some-other-source-alias",
        "UNRELATED_SETTING_SECRET": "fake-do-not-forward",
    }
    settings.write_text(json.dumps({"env": provider, "effortLevel": "low"}))
    config = tmp_path / "claude-config.json"
    configuration = {
        "real_claude": str(executable),
        "settings_file": str(settings),
        "base_url": URL,
        "model": MODEL,
        "effort": "high",
    }
    config.write_text(json.dumps(configuration))
    environment = {
        "PATH": f"{tmp_path}:{Path(sys.executable).parent}:/usr/bin:/bin",
        "HOME": str(tmp_path),
        "ANTHROPIC_BASE_URL": "https://wrong.example.invalid",
        "ANTHROPIC_MODEL": "wrong-model",
        "ANTHROPIC_API_KEY": "fake-ambient-api-key",
        "ANTHROPIC_AUTH_TOKEN": "fake-ambient-token",
        "ANTHROPIC_EXTRA_SECRET": "fake-ambient-secret",
        "CLAUDE_CODE_OAUTH_TOKEN": "fake-ambient-oauth",
        "AGATE_AK": "fake-ambient-agate",
        "ARBITRARY_SECRET": "fake-ambient-unrelated",
        "HTTP_PROXY": "http://proxy.example.invalid:8080",
        "CI": "0",
    }
    arguments = [
        "--bare",
        "--print",
        "--output-format",
        "json",
        "--tools",
        "",
        "--no-session-persistence",
        "--session-id",
        "fake-session",
        "--effort",
        "low",
        "--prompt-suggestions",
        "false",
        "literal 'quotes' $(never-execute) ;\n知识查询",
    ]

    def run() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(wrapper), *arguments],
            env=environment,
            text=True,
            capture_output=True,
            timeout=10,
        )

    return SimpleNamespace(
        root=tmp_path,
        wrapper=wrapper,
        executable=executable,
        marker=marker,
        settings=settings,
        provider=provider,
        config=config,
        configuration=configuration,
        environment=environment,
        arguments=arguments,
        run=run,
    )


def refused(bridge: SimpleNamespace) -> None:
    result = bridge.run()
    assert result.returncode == 78
    assert result.stdout == ""
    assert result.stderr == ERROR
    assert not bridge.marker.exists()


def test_flash_high_override_isolated_from_shared_settings(bridge: SimpleNamespace) -> None:
    before = bridge.settings.read_bytes()
    result = bridge.run()
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    expected = bridge.arguments[:9] + bridge.arguments[11:]
    assert output["argv"] == ["--model", MODEL, "--effort", "high", *expected]
    assert bridge.settings.read_bytes() == before
    environment = output["env"]
    assert environment["ANTHROPIC_BASE_URL"] == URL
    for name in (
        "ANTHROPIC_MODEL",
        "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        "ANTHROPIC_DEFAULT_OPUS_MODEL",
        "ANTHROPIC_SMALL_FAST_MODEL",
    ):
        assert environment[name] == MODEL
    assert environment["CLAUDE_CODE_EFFORT_LEVEL"] == "high"
    assert environment["ANTHROPIC_AUTH_TOKEN"] == "fake-configured-token-only"
    assert environment["HTTP_PROXY"] == bridge.environment["HTTP_PROXY"]
    assert environment["CI"] == "1"
    for name in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_EXTRA_SECRET",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "AGATE_AK",
        "ARBITRARY_SECRET",
        "UNRELATED_SETTING_SECRET",
    ):
        assert name not in environment


@pytest.mark.parametrize(
    "override",
    [
        ["--model", "other", "--effort", "low"],
        ["--model=other", "--effort=low"],
        ["--model", "other", "--model=third", "--effort=low", "--effort", "medium"],
    ],
)
def test_overrides_explicit_duplicate_options(bridge: SimpleNamespace, override: list[str]) -> None:
    bridge.arguments[:] = [*override, "--bare", "--tools", "", "prompt"]
    result = bridge.run()
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["argv"] == [
        "--model",
        MODEL,
        "--effort",
        "high",
        "--bare",
        "--tools",
        "",
        "prompt",
    ]


@pytest.mark.parametrize(
    "suffix",
    [
        ["prompt", "--model", "literal", "--effort=literal"],
        ["--", "--effort", "low", "--model=literal"],
        ["explain --model=foo and --effort low\n$(do not execute)"],
        [""],
    ],
)
def test_prompt_and_post_prompt_arguments_unchanged(
    bridge: SimpleNamespace,
    suffix: list[str],
) -> None:
    bridge.arguments[:] = ["--bare", "--effort", "low", *suffix]
    result = bridge.run()
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["argv"] == [
        "--model",
        MODEL,
        "--effort",
        "high",
        "--bare",
        *suffix,
    ]


def test_other_option_values_are_not_rewritten(bridge: SimpleNamespace) -> None:
    bridge.arguments[:] = ["--mcp-config", "--model", "--tools=--effort", "prompt"]
    result = bridge.run()
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["argv"] == [
        "--model",
        MODEL,
        "--effort",
        "high",
        *bridge.arguments,
    ]


@pytest.mark.parametrize("arguments", [["--effort"], ["--output-format"], ["--unknown", "foo"]])
def test_invalid_or_unknown_option_refused(bridge: SimpleNamespace, arguments: list[str]) -> None:
    bridge.arguments[:] = arguments
    refused(bridge)


@pytest.mark.parametrize("name", ["config", "settings"])
def test_missing_file_refused(bridge: SimpleNamespace, name: str) -> None:
    getattr(bridge, name).unlink()
    refused(bridge)


@pytest.mark.parametrize(
    "change",
    [
        {"base_url": "https://wrong.example.invalid"},
        {"base_url": "http://example.invalid"},
        {"base_url": "https://user:password@example.invalid"},
        {"base_url": None},
        {"model": ""},
        {"model": ["wrong"]},
        {"effort": "unsupported"},
        {"extra": True},
        {"real_claude": "relative-claude"},
        {"settings_file": "settings.json"},
    ],
)
def test_invalid_configuration_refused(bridge: SimpleNamespace, change: dict[str, object]) -> None:
    bridge.config.write_text(json.dumps({**bridge.configuration, **change}))
    refused(bridge)


@pytest.mark.parametrize(
    "change",
    [
        {"ANTHROPIC_BASE_URL": "https://wrong.example.invalid"},
        {"ANTHROPIC_BASE_URL": None},
        {"ANTHROPIC_AUTH_TOKEN": ""},
    ],
)
def test_missing_auth_or_different_provider_refused(
    bridge: SimpleNamespace,
    change: dict[str, object],
) -> None:
    bridge.settings.write_text(json.dumps({"env": {**bridge.provider, **change}}))
    refused(bridge)


def test_malformed_settings_never_echoed(bridge: SimpleNamespace) -> None:
    bridge.settings.write_text('{"fake-secret-must-not-appear":')
    refused(bridge)


@pytest.mark.parametrize("kind", ["direct", "symlink", "hardlink", "copy"])
def test_recursion_refused(bridge: SimpleNamespace, kind: str) -> None:
    target = bridge.wrapper
    if kind != "direct":
        target = bridge.root / "wrapper-alias"
        if kind == "symlink":
            target.symlink_to(bridge.wrapper)
        elif kind == "hardlink":
            os.link(bridge.wrapper, target)
        else:
            shutil.copyfile(bridge.wrapper, target)
            target.chmod(0o755)
    bridge.config.write_text(json.dumps({**bridge.configuration, "real_claude": str(target)}))
    refused(bridge)


def test_non_executable_refused(bridge: SimpleNamespace) -> None:
    bridge.executable.chmod(0o600)
    refused(bridge)


def test_api_key_supported_without_inherited_token(bridge: SimpleNamespace) -> None:
    provider = dict(bridge.provider)
    provider.pop("ANTHROPIC_AUTH_TOKEN")
    provider["ANTHROPIC_API_KEY"] = "fake-configured-api-key"
    bridge.settings.write_text(json.dumps({"env": provider}))
    result = bridge.run()
    assert result.returncode == 0, result.stderr
    environment = json.loads(result.stdout)["env"]
    assert environment["ANTHROPIC_API_KEY"] == "fake-configured-api-key"
    assert "ANTHROPIC_AUTH_TOKEN" not in environment


def test_native_launcher_environment_filter_and_low_effort(bridge: SimpleNamespace) -> None:
    source = ROOT / "corpus/internal_gpu_wiki/tools/agent_launch.py"
    if not source.is_file():
        pytest.skip("optional internal Wiki corpus is not installed")
    spec = importlib.util.spec_from_file_location("wiki_native_launch", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    filtered = module.bridge_environment("claude", bridge.environment)
    assert "ANTHROPIC_MODEL" not in filtered
    assert "ANTHROPIC_BASE_URL" not in filtered
    stdout, stderr, code, timed_out = module.run_json(
        "claude",
        "测试查询 --effort low",
        bridge.root,
        10,
        bridge.environment,
    )
    assert code == 0, stderr
    assert not timed_out
    output = json.loads(stdout)
    arguments = output["argv"]
    assert arguments[:4] == ["--model", MODEL, "--effort", "high"]
    assert arguments.count("--effort") == 1
    assert "--bare" in arguments
    assert arguments[arguments.index("--tools") + 1] == ""
    assert arguments[-1] == "测试查询 --effort low"
    assert output["env"]["ANTHROPIC_MODEL"] == MODEL
