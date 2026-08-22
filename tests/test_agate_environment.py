"""Agate environment metadata resolution tests."""

from __future__ import annotations

import pytest

from atrex_runtime.gateway.environment import AgateHardwareTargetResolver


class Client:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.calls: list[tuple[str, bool]] = []

    def get_env(self, gpu: str, force: bool = False) -> dict[str, object]:
        self.calls.append((gpu, force))
        return self.response


def test_resolver_uses_canonical_gpu_and_arch() -> None:
    client = Client({"gpu": "L20N", "gpu_model": "NVIDIA L20N", "arch": "sm_120"})

    resolved = AgateHardwareTargetResolver(client).resolve("RTX-PRO-5000")

    assert resolved.gpu == "L20N"
    assert resolved.arch == "sm_120"
    assert client.calls == [("RTX-PRO-5000", False)]


@pytest.mark.parametrize(
    "response, missing",
    [
        ({"arch": "sm_120"}, "canonical gpu"),
        ({"gpu": "L20N"}, "arch"),
    ],
)
def test_resolver_rejects_incomplete_environment_metadata(
    response: dict[str, object], missing: str
) -> None:
    with pytest.raises(ValueError, match=missing):
        AgateHardwareTargetResolver(Client(response)).resolve("L20N")
