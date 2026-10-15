"""Real Linux boundary regression; macOS runs the policy/unit tests instead."""

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from atrex_runtime.config import BwrapContainerSettings
from atrex_runtime.workers.launcher import BwrapContainerLauncher


@pytest.mark.skipif(platform.system() != "Linux", reason="requires Linux user/PID namespaces")
def test_empty_root_blocks_archive_and_proc_escape_with_long_workspace(tmp_path):
    if not shutil.which("bwrap") or not shutil.which("cc"):
        pytest.skip("requires bubblewrap and C compiler")
    root = tmp_path / ("attempt-" + "a" * 100)
    workspace = root / "worker"
    workspace.mkdir(parents=True)
    canary = tmp_path / "archives/previous-kernel.py"
    canary.parent.mkdir()
    canary.write_text("PRIVATE_OLD_KERNEL")
    (workspace / "escape").symlink_to(canary)
    settings = BwrapContainerSettings(
        bwrap_executable=Path(shutil.which("bwrap")),
        read_only_bind_paths=tuple(
            dict.fromkeys(
                (
                    Path(sys.prefix),
                    Path(sys.base_prefix),
                    Path(str(getattr(sys, "_base_executable", sys.executable))).parent.parent,
                    Path(__file__).resolve().parents[1] / "src/atrex_runtime",
                )
            )
        ),
    )
    launcher = BwrapContainerLauncher(Path("/usr/bin/env"), settings, (root,))
    code = f"""
import os
from pathlib import Path
for name in {str(canary), "/home/agent/workspace/escape", "/proc/1/root", "/proc/self/environ"}:
    try: Path(name).read_bytes()
    except (OSError, ValueError): pass
    else: raise AssertionError('escaped isolation: ' + name)
# metadata is available, while opening proc descriptors/root links remains denied.
assert Path('/proc/self/stat').is_file()
assert '[stack]' in Path('/proc/self/maps').read_text()
assert not Path('/proc/self/root').exists()
assert not Path('/proc/self/fd/0').exists()
print('ISOLATED')
"""
    argv = launcher.wrap(
        (sys.executable, "-c", code),
        workspace=workspace,
        environment={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
    )
    result = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    if "Operation not permitted" in result.stderr and "bwrap:" in result.stderr:
        pytest.skip("host disables user namespaces")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ISOLATED"
