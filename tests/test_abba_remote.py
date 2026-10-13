"""Shared native ABBA and diagnostic helper tests."""

from __future__ import annotations

from atrex_runtime.gateway import abba_remote


def test_nvidia_clock_lock_uses_max_supported_clock_and_restores(
    monkeypatch: object,
) -> None:
    commands: list[list[str]] = []

    def fake_command(argv: list[str]) -> str:
        commands.append(argv)
        if "--query-gpu=index,uuid" in argv:
            return "0, GPU-aabb\n"
        if "--query-supported-clocks=gr" in argv:
            return "1200\n1500\n"
        return ""

    monkeypatch.setattr(abba_remote.shutil, "which", lambda name: f"/bin/{name}")  # type: ignore[attr-defined]
    monkeypatch.setattr(abba_remote, "_run_clock_command", fake_command)  # type: ignore[attr-defined]
    for name in (
        "ATREX_BENCH_CLOCK_DEVICE",
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "ATREX_BENCH_CLOCKS_LOCKED",
        "SOL_EXECBENCH_CLOCKS_LOCKED",
        "ATREX_BENCH_CLOCK_LOCK_SOURCE",
    ):
        monkeypatch.delenv(name, raising=False)  # type: ignore[attr-defined]

    with abba_remote._clock_lock(True) as report:
        assert report["graphics_mhz"] == 1500
        assert abba_remote.os.environ["ATREX_BENCH_CLOCKS_LOCKED"] == "1"
        assert abba_remote.os.environ["SOL_EXECBENCH_CLOCKS_LOCKED"] == "1"
        assert abba_remote.os.environ["ATREX_BENCH_CLOCK_LOCK_SOURCE"] == ("atrex-runtime-abba")

    assert report["restored"] is True
    assert "ATREX_BENCH_CLOCKS_LOCKED" not in abba_remote.os.environ
    assert commands[-1] == ["/bin/nvidia-smi", "-i", "GPU-aabb", "--reset-gpu-clocks"]
