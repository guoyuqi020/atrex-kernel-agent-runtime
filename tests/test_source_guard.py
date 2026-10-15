"""Source edit boundaries survive every tool mode and final candidate rejection."""

import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from test_kernel_sources import source_seed  # noqa: F401

# The imported pytest fixture is requested by name below.
# ruff: noqa: F811
from atrex_runtime.config import BwrapContainerSettings
from atrex_runtime.kernel_sources import fixed_source_mounts, project_source_guard
from atrex_runtime.workers.launcher import BwrapContainerLauncher


def project(seed, tmp_path):
    root = tmp_path / "worker"
    shutil.copytree(seed.working, root / "work/kernel")
    project_source_guard(root, seed.contract)
    return root


def test_fixed_files_are_read_only_and_mounted_over_writable_workspace(source_seed, tmp_path):
    root = project(source_seed, tmp_path)
    fixed = fixed_source_mounts(root)
    assert root / "work/kernel/kernel.py" in fixed
    assert all(p.stat().st_mode & 0o222 == 0 for p in fixed)
    editable = root / "work/kernel/example_kernel/impl/value.py"
    editable.write_text("VALUE = 9\n")
    launcher = BwrapContainerLauncher(
        Path("/usr/bin/env"),
        BwrapContainerSettings(bwrap_executable=Path("/usr/bin/bwrap")),
        (tmp_path,),
    )
    argv = launcher.wrap(("/bin/true",), workspace=root, environment={})
    for path in fixed:
        index = argv.index(str(path))
        assert argv[index - 1] == "--ro-bind"
        assert argv[index + 1] == str(Path("/home/agent/workspace") / path.relative_to(root))
    assert str(editable) not in argv


@pytest.mark.parametrize("change", ["parent_symlink", "fixed_symlink", "changed_bytes"])
def test_source_mounts_refuse_unsafe_inputs(source_seed, tmp_path, change):
    root = project(source_seed, tmp_path)
    fixed = root / "work/kernel/kernel.py"
    if change == "parent_symlink":
        (root / "work/kernel").rename(root / "relocated")
        (root / "work/kernel").symlink_to(root / "relocated")
    elif change == "fixed_symlink":
        fixed.unlink()
        fixed.symlink_to(source_seed.working / "kernel.py")
    else:
        fixed.chmod(0o600)
        fixed.write_text("unauthorized change")
    with pytest.raises(ValueError):
        fixed_source_mounts(root)


@pytest.mark.parametrize("bundle", ["atrex-kernel-agent-core", "kernel-design-agents"])
def test_source_repair_precedes_report_check_without_runtime_tool_dependency(
    source_seed, tmp_path, bundle
):
    root = project(source_seed, tmp_path)
    script = """
import sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from sessions.source_completion import SourceCompletionCheck, source_boundary_prompt
root=Path(sys.argv[2]); calls=[]
check=SourceCompletionCheck(root,lambda remaining: calls.append(remaining))
bad=root/'work/kernel/change-notes.md';bad.write_text('notes')
repair=check(30)
assert not calls and check.exhausted_exit_status==128
assert 'change-notes.md' in repair and 'terminal attempt report' in repair
assert 'kernel.py' in source_boundary_prompt(root)
bad.unlink()
assert check(20) is None and calls==[20] and check.exhausted_exit_status==127
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(Path(__file__).resolve().parents[1] / "src" / bundle / "src"),
            str(root),
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.anyio
@pytest.mark.parametrize("raw", [True, False])
@pytest.mark.parametrize(
    "reason", ["candidate-validation-failed", "candidate-validation-exhausted"]
)
async def test_invalid_source_settles_attempt_without_retrying_or_aborting_campaign(
    tmp_path, raw, reason
):
    from conftest import digest
    from test_optimizer_runner import (
        FakeFinalizer,
        FakeGatewayAuthorities,
        FakeRuntimeEventRecorder,
        FakeSessionDriver,
        FakeSessionTraceRecorder,
        FakeWorkspaceAssembler,
        SequencedOutcomes,
        _config,
        _request,
        _session_result,
    )

    from atrex_runtime.ports import AttemptCandidateResult
    from atrex_runtime.workers.optimizer import SessionOptimizerRunner
    from atrex_runtime.workers.workspace import PreparedAttempt

    finalizer = FakeFinalizer(
        AttemptCandidateResult(digest("unused"), digest("result"), True, 1), []
    )
    session = replace(
        _session_result(reason, "invalid source"),
        attempt_report_error="fixed source file was modified: PROVENANCE.json",
    )
    events = FakeRuntimeEventRecorder([])
    runner = SessionOptimizerRunner(
        FakeWorkspaceAssembler(
            PreparedAttempt(tmp_path, tmp_path / "manifest", tmp_path / "sessions", "test")
        ),
        FakeSessionDriver(session),
        SequencedOutcomes([None]),
        finalizer,
        FakeGatewayAuthorities(),
        FakeSessionTraceRecorder([]),
        events,
        _config(tmp_path),
    )
    request = replace(_request(), raw=raw, tool_modules=(), result_artifact_read_enabled=False)
    result = await runner.run_attempt(request)
    assert result.candidate is None and "PROVENANCE.json" in result.failure_reason
    assert not finalizer.calls
    assert events.records[-1][0] == "attempt.report_rejected"


@pytest.mark.skipif(sys.platform != "linux", reason="requires Linux namespaces")
def test_real_bwrap_prevents_fixed_file_writes_chmod_unlink_and_replace(source_seed, tmp_path):
    if not shutil.which("bwrap") or not shutil.which("cc"):
        pytest.skip("requires bwrap and C compiler")
    root = project(source_seed, tmp_path)
    settings = BwrapContainerSettings(
        bwrap_executable=Path(shutil.which("bwrap")),
        read_only_bind_paths=tuple(
            dict.fromkeys(
                (
                    Path(sys.prefix),
                    Path(sys.base_prefix),
                    Path(str(getattr(sys, "_base_executable", sys.executable))).parent.parent,
                )
            )
        ),
    )
    launcher = BwrapContainerLauncher(Path("/usr/bin/env"), settings, (tmp_path,))
    code = """
from pathlib import Path
p=Path('work/kernel/kernel.py'); original=p.read_bytes()
temp=p.with_name('replacement.py');temp.write_text('replacement')
for action in (lambda:p.write_text('bad'),lambda:p.chmod(0o600),p.unlink,lambda:temp.replace(p)):
    try:action()
    except OSError:pass
    else:raise AssertionError('fixed file mutation succeeded')
assert p.read_bytes()==original
Path('work/kernel/example_kernel/impl/value.py').write_text('VALUE=99')
print('SOURCE_GUARD_OK')
"""
    argv = launcher.wrap(
        (sys.executable, "-c", code),
        workspace=root,
        environment={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
    )
    result = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    if "Operation not permitted" in result.stderr and "bwrap:" in result.stderr:
        pytest.skip("host disables user namespaces")
    assert result.returncode == 0, result.stderr
    assert "SOURCE_GUARD_OK" in result.stdout
