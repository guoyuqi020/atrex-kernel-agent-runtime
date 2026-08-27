from __future__ import annotations

from atrex_runtime.gateway.environment import (
    AgateHardwareTargetResolver,
    infer_accelerator_backend,
)


def test_infer_accelerator_backend_from_architecture_and_device_names() -> None:
    assert infer_accelerator_backend("L20N", "sm_120") == "cuda"
    assert infer_accelerator_backend("MI308X", "gfx942") == "rocm"
    assert infer_accelerator_backend("PPU-ZW810E", "PPU-ZW810E") == "ppu"
    assert infer_accelerator_backend("ZW-M890P", "ZW-M890P") == "ppu"
    assert infer_accelerator_backend("custom", "custom") is None


def test_resolver_normalizes_explicit_ppu_environment_metadata() -> None:
    class Client:
        def get_env(self, gpu: str, force: bool = False) -> dict[str, object]:
            assert gpu == "ppu-prod"
            assert force is False
            return {
                "gpu": "ZW-M890P",
                "arch": "ZW-M890P",
                "accelerator_backend": "ppu",
                "device_slug": "zw-m890p",
            }

    resolved = AgateHardwareTargetResolver(Client()).resolve("ppu-prod")
    assert resolved.gpu == "ZW-M890P"
    assert resolved.arch == "ZW-M890P"
    assert resolved.accelerator_backend == "ppu"
    assert resolved.device_slug == "zw-m890p"
    assert resolved.supports_clock_lock is False


def test_resolver_infers_ppu_metadata_for_older_agate_response() -> None:
    class Client:
        def get_env(self, gpu: str, force: bool = False) -> dict[str, object]:
            return {"gpu": "ppu-prod", "arch": "PPU-ZW810E"}

    resolved = AgateHardwareTargetResolver(Client()).resolve("ppu-prod")
    assert resolved.accelerator_backend == "ppu"
    assert resolved.device_slug == "ppu-zw810e"
    assert resolved.supports_clock_lock is False
