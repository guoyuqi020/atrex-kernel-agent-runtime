"""Trusted content-level Production Gate tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from atrex_runtime.artifacts.local import ArtifactKind, LocalArtifactStore
from atrex_runtime.domain.ids import new_attempt_id
from atrex_runtime.domain.models import Dsl
from atrex_runtime.gateway.contract import (
    AgateEvaluationContext,
    AgateEvaluationContractV1,
    AgateEvaluationOptionsV1,
)
from atrex_runtime.gateway.production_policy import (
    ProductionKernelPolicy,
    RegistryProductionKernelValidator,
)


def _write(root: Path, source: str, solution: object | None = None) -> None:
    root.mkdir()
    root.joinpath("kernel.py").write_text(source, encoding="utf-8")
    if solution is not None:
        root.joinpath("solution.json").write_text(json.dumps(solution), encoding="utf-8")


@pytest.mark.parametrize(
    ("dsl", "source"),
    [
        (
            Dsl.TRITON,
            "import torch\nimport triton\nimport triton.language as tl\n"
            "@triton.jit\ndef kernel(x):\n    return\n",
        ),
        (
            Dsl.CUTEDSL,
            "import torch\nfrom cutlass import cute\n@cute.kernel\ndef kernel(x):\n    return\n",
        ),
        (
            Dsl.CUDA,
            "import torch\nfrom torch.utils.cpp_extension import load_inline\n"
            "CUDA_SRC = r'''__global__ void kernel(float* x) {}'''\n",
        ),
    ],
)
def test_production_gate_accepts_self_contained_fixed_dsl(
    tmp_path: Path,
    dsl: Dsl,
    source: str,
) -> None:
    _write(tmp_path / "candidate", source)

    ProductionKernelPolicy().validate(tmp_path / "candidate", "kernel.py", dsl)


def test_production_gate_explains_missing_cuda_marker_components(tmp_path: Path) -> None:
    root = tmp_path / "candidate"
    _write(
        root,
        "import torch\nfrom torch.utils.cpp_extension import load\n",
    )

    violations = ProductionKernelPolicy().violations(root, "kernel.py", Dsl.CUDA)

    assert violations == (
        "missing self-authored cuda implementation marker: kernel.py must contain "
        "'__global__' CUDA source and reference at least one approved CUDA loader in the "
        "same file (load_inline, cpp_extension, CUDAExtension, nvrtc, cuda.bindings); "
        "detected __global__=no, approved_loader=cpp_extension",
    )


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (
            "import torch\nimport triton\n"
            "@triton.jit\ndef kernel(x, y):\n    return torch.matmul(x, y)\n",
            "PyTorch compute call is forbidden",
        ),
        (
            "import torch\nimport triton\nfrom cutlass import cute\n"
            "@triton.jit\ndef kernel(x):\n    return\n"
            "@cute.kernel\ndef alternate(x):\n    return\n",
            "mixed/alternate framework marker is forbidden",
        ),
        (
            "import torch\nimport triton\nimport flashinfer\n"
            "@triton.jit\ndef kernel(x):\n    return\n",
            "third-party dependency is not approved",
        ),
        (
            "import torch\nimport triton\nimport importlib\n"
            "@triton.jit\ndef kernel(x):\n    return\n",
            "dynamic external-code loading is forbidden",
        ),
    ],
)
def test_production_gate_rejects_fallbacks_and_alternate_implementations(
    tmp_path: Path,
    source: str,
    message: str,
) -> None:
    _write(tmp_path / "candidate", source)

    with pytest.raises(ValueError, match=message):
        ProductionKernelPolicy().validate(tmp_path / "candidate", "kernel.py", Dsl.TRITON)


def test_production_gate_locates_every_forbidden_library_token(tmp_path: Path) -> None:
    _write(
        tmp_path / "candidate",
        "import torch\n"
        "from torch.utils.cpp_extension import load_inline\n"
        'CUDA_SRC = r"""\n'
        "#include <cublas_v2.h>\n"
        "__global__ void kernel(float* x) {}\n"
        "void gemm() { cublasSgemm(h, CUBLAS_OP_T, 1, 1, 1); }\n"
        '"""\n',
    )

    violations = ProductionKernelPolicy().violations(tmp_path / "candidate", "kernel.py", Dsl.CUDA)

    assert violations == (
        "prebuilt CUDA math/operator library reference is forbidden: "
        "cublas_v2 (line 4), cublasSgemm (line 6), CUBLAS_OP_T (line 6)",
    )


def test_production_gate_locates_a_forbidden_cutlass_include(tmp_path: Path) -> None:
    _write(
        tmp_path / "candidate",
        "import torch\nimport triton\n@triton.jit\ndef kernel(x):\n    return\n"
        'SRC = r"""\n#include <cutlass/gemm/device/gemm.h>\n"""\n',
    )

    violations = ProductionKernelPolicy().violations(
        tmp_path / "candidate", "kernel.py", Dsl.TRITON
    )

    assert violations == (
        "CUTLASS implementation is forbidden outside the CuteDSL lineage: "
        "#include <cutlass/gemm/device/gemm.h> (line 7)",
    )


def test_production_gate_ignores_a_forbidden_library_named_only_in_prose(tmp_path: Path) -> None:
    _write(
        tmp_path / "candidate",
        "import torch\n"
        "from torch.utils.cpp_extension import load_inline\n"
        'CUDA_SRC = r"""__global__ void kernel(float* x) {}"""\n'
        "\n"
        "\n"
        "class Model(torch.nn.Module):\n"
        '    """Accumulation matches cuBLAS fp32 results bitwise."""\n'
        "\n"
        "    def forward(self, x):  # cuBLAS parity verified\n"
        "        return x\n",
    )

    violations = ProductionKernelPolicy().violations(tmp_path / "candidate", "kernel.py", Dsl.CUDA)

    assert violations == ()


def test_production_gate_validates_solution_dependencies_and_languages(tmp_path: Path) -> None:
    root = tmp_path / "candidate"
    _write(
        root,
        "import torch\nimport triton\n@triton.jit\ndef kernel(x):\n    return\n",
        {
            "spec": {
                "languages": ["python", "cuda"],
                "dependencies": ["torch", "flashinfer>=1"],
            }
        },
    )

    violations = ProductionKernelPolicy().violations(root, "kernel.py", Dsl.TRITON)

    assert any("solution dependency is not approved" in value for value in violations)
    assert any("languages omit fixed DSL" in value for value in violations)
    assert any("languages contain alternate DSLs" in value for value in violations)


def test_registry_validator_uses_production_gate_state_sealed_in_contract(
    tmp_path: Path,
) -> None:
    root = tmp_path / "candidate"
    _write(root, "import torch\ndef kernel(x, y):\n    return torch.matmul(x, y)\n")
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    digest = artifacts.put_directory(root, ArtifactKind.KERNEL)

    def context(enabled: bool) -> AgateEvaluationContext:
        contract = AgateEvaluationContractV1(
            candidate_path="kernel.py",
            reference_py="reference",
            input_py="inputs",
            shapes={"0": {}},
            options=AgateEvaluationOptionsV1(
                num_correctness_cases=1,
                bench_iters=1,
                atol=0,
                rtol=0,
                timeout_s=60,
            ),
            lock_clocks=False,
            production_gate=enabled,
        )
        return AgateEvaluationContext("op", "gpu", Dsl.TRITON, contract)

    class Contexts:
        enabled = False

        def resolve(self, _attempt_id: object) -> AgateEvaluationContext:
            return context(self.enabled)

    contexts = Contexts()
    validator = RegistryProductionKernelValidator(
        contexts,  # type: ignore[arg-type]
        artifacts,
        ProductionKernelPolicy(),
    )
    attempt_id = new_attempt_id()

    validator.validate(attempt_id, digest)
    contexts.enabled = True
    with pytest.raises(ValueError, match="production gate rejected"):
        validator.validate(attempt_id, digest)
