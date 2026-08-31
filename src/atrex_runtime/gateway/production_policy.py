"""Trusted content-level Production Gate for generated Kernel candidates."""

from __future__ import annotations

import ast
import io
import json
import re
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..artifacts.local import LocalArtifactStore
from ..domain.ids import ArtifactDigest, AttemptId
from ..domain.models import Dsl
from .candidate import resolve_kernel_candidate
from .contract import AgateEvaluationContextResolver

_STDLIB_IMPORTS = (
    frozenset(getattr(sys, "stdlib_module_names", ()))
    | frozenset(sys.builtin_module_names)
    | {"__future__"}
)
_ALLOWED_IMPORTS = {
    Dsl.TRITON: {"torch", "triton", "sol_execbench"},
    Dsl.CUTEDSL: {"torch", "cutlass", "cuda", "sol_execbench"},
    Dsl.CUDA: {"torch", "cuda", "sol_execbench"},
}
_ALLOWED_DEPENDENCIES = {
    Dsl.TRITON: {"torch", "triton"},
    Dsl.CUTEDSL: {"torch", "cutlass", "nvidiacutlassdsl", "cuda", "cudapython"},
    Dsl.CUDA: {"torch", "cuda", "cudapython"},
}
_DYNAMIC_LOADING_IMPORTS = frozenset({"ctypes", "importlib", "pkgutil", "runpy", "subprocess"})
_BANNED_TORCH_COMPUTE = frozenset(
    {
        "addmm",
        "amax",
        "amin",
        "bmm",
        "conv1d",
        "conv2d",
        "conv3d",
        "cumprod",
        "cumsum",
        "einsum",
        "exp",
        "gelu",
        "layer_norm",
        "log",
        "log_softmax",
        "matmul",
        "max",
        "mean",
        "min",
        "mm",
        "rms_norm",
        "scaled_dot_product_attention",
        "sigmoid",
        "silu",
        "softmax",
        "sort",
        "sum",
        "topk",
    }
)
_CUDA_LOADER_MARKERS = (
    "load_inline",
    "cpp_extension",
    "CUDAExtension",
    "nvrtc",
    "cuda.bindings",
)


class CandidateProductionValidator(Protocol):
    """Validate one sealed Attempt candidate against its fixed DSL."""

    def validate(self, attempt_id: AttemptId, candidate_digest: ArtifactDigest) -> None:
        """Raise when the candidate violates Production policy."""
        ...

    def violations(
        self, attempt_id: AttemptId, candidate_digest: ArtifactDigest
    ) -> tuple[str, ...]:
        """Report Production policy violations without rejecting the request."""
        ...


@dataclass(frozen=True, slots=True)
class ProductionKernelPolicy:
    """Mechanically enforce self-contained, single-DSL production Kernels."""

    def validate(self, root: Path, candidate_path: str, dsl: Dsl) -> None:
        """Fail closed with every mechanically detected policy violation."""
        violations = self.violations(root, candidate_path, dsl)
        if violations:
            raise ValueError("production gate rejected candidate: " + "; ".join(violations))

    def violations(self, root: Path, candidate_path: str, dsl: Dsl) -> tuple[str, ...]:
        kernel_path = root.joinpath(*candidate_path.split("/"))
        if kernel_path.is_symlink() or not kernel_path.is_file():
            return (f"candidate source is missing: {candidate_path}",)
        source = kernel_path.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(source, filename=candidate_path)
        except SyntaxError as error:
            return (f"candidate source is not valid Python: {error.msg} (line {error.lineno})",)

        errors: list[str] = []
        roots, relative_import = _import_roots(tree)
        if relative_import:
            errors.append("relative/local-module imports are not self-contained")
        for name in sorted(roots & _DYNAMIC_LOADING_IMPORTS):
            errors.append(f"dynamic external-code loading is forbidden: {name}")
        allowed = _STDLIB_IMPORTS | _ALLOWED_IMPORTS[dsl]
        for name in sorted(roots - allowed):
            errors.append(f"third-party dependency is not approved for {dsl.value}: {name}")

        errors.extend(_torch_compute_violations(tree))
        policy_source = _code_without_prose(source)
        markers = _framework_markers(policy_source, source)
        if not markers[dsl]:
            if dsl is Dsl.CUDA:
                detected_loaders = tuple(
                    marker for marker in _CUDA_LOADER_MARKERS if marker in source
                )
                errors.append(
                    "missing self-authored cuda implementation marker: "
                    f"{candidate_path} must contain '__global__' CUDA source and reference "
                    "at least one approved CUDA loader in the same file "
                    f"({', '.join(_CUDA_LOADER_MARKERS)}); detected __global__="
                    f"{'yes' if '__global__' in policy_source else 'no'}, approved_loader="
                    f"{', '.join(detected_loaders) if detected_loaders else 'none'}"
                )
            else:
                errors.append(f"missing self-authored {dsl.value} implementation marker")
        for other, present in markers.items():
            if other is not dsl and present:
                errors.append(f"mixed/alternate framework marker is forbidden: {other.value}")

        for pattern, message in (
            (r"\btorch\.ops\b", "torch.ops dispatch is a prebuilt/custom operator call"),
            (
                r"\btorch\.nn\.functional\b",
                "torch.nn.functional is not the selected kernel framework",
            ),
            (
                r"\btorch\.(?:linalg|_scaled_mm)\b",
                "PyTorch compute fallback is not the selected kernel framework",
            ),
            (
                r"\b(?:flashinfer|flash_attn|xformers|vllm|sglang|bitsandbytes)\b",
                "prebuilt third-party kernel/operator library reference is forbidden",
            ),
            (
                r"\b(?:cublas|cudnn)[A-Za-z0-9_]*\b",
                "prebuilt CUDA math/operator library reference is forbidden",
            ),
        ):
            matched = _matched_tokens(pattern, policy_source)
            if matched:
                errors.append(f"{message}: {', '.join(matched)}")

        if dsl is not Dsl.CUTEDSL:
            cutlass_includes = _matched_tokens(
                r"#\s*include\s*[<\"]cutlass/[^>\"]*[>\"]?", policy_source
            )
            if cutlass_includes:
                errors.append(
                    "CUTLASS implementation is forbidden outside the CuteDSL lineage: "
                    + ", ".join(cutlass_includes)
                )
        errors.extend(_solution_violations(root / "solution.json", dsl))
        return tuple(dict.fromkeys(errors))


class RegistryProductionKernelValidator:
    """Resolve Attempt DSL and apply the shared Production Gate to its sealed Artifact."""

    def __init__(
        self,
        contexts: AgateEvaluationContextResolver,
        artifacts: LocalArtifactStore,
        policy: ProductionKernelPolicy,
    ) -> None:
        self._contexts = contexts
        self._artifacts = artifacts
        self._policy = policy

    def validate(self, attempt_id: AttemptId, candidate_digest: ArtifactDigest) -> None:
        violations = self.violations(attempt_id, candidate_digest)
        if violations:
            raise ValueError("production gate rejected candidate: " + "; ".join(violations))

    def violations(
        self, attempt_id: AttemptId, candidate_digest: ArtifactDigest
    ) -> tuple[str, ...]:
        context = self._contexts.resolve(attempt_id)
        if not context.contract.production_gate:
            return ()
        resolved = resolve_kernel_candidate(
            self._artifacts,
            candidate_digest,
            context.contract.candidate_path,
            error_type=ValueError,
            kind_error="production gate requires a Kernel Artifact",
            missing_error="production gate requires the contract candidate path",
        )
        return self._policy.violations(
            resolved.root,
            context.contract.candidate_path,
            context.dsl,
        )


def _code_without_prose(source: str) -> str:
    """Remove comments and bare string expressions before marker scanning."""
    lines = source.splitlines()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        tree = None
    if tree is not None:
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
                and node.end_lineno is not None
            ):
                for index in range(node.lineno, node.end_lineno + 1):
                    lines[index - 1] = ""
    blanked = "\n".join(lines)
    try:
        tokens = tokenize.generate_tokens(io.StringIO(blanked).readline)
        comments = [token for token in tokens if token.type == tokenize.COMMENT]
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return blanked
    output = blanked.splitlines()
    for token in comments:
        row, column = token.start
        output[row - 1] = output[row - 1][:column]
    return "\n".join(output)


def _matched_tokens(pattern: str, source: str, *, limit: int = 5) -> tuple[str, ...]:
    """Locate each distinct match of a forbidden pattern as "token (line N)"."""
    located: dict[str, int] = {}
    for match in re.finditer(pattern, source, flags=re.IGNORECASE):
        token = match.group(0).strip()
        if token not in located:
            located[token] = source.count("\n", 0, match.start()) + 1
    descriptors = tuple(f"{token} (line {line})" for token, line in located.items())
    if len(descriptors) <= limit:
        return descriptors
    return (*descriptors[:limit], f"and {len(descriptors) - limit} more")


def _import_roots(tree: ast.AST) -> tuple[set[str], bool]:
    roots: set[str] = set()
    relative = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            relative = relative or bool(node.level)
            if node.module:
                roots.add(node.module.split(".", 1)[0])
    return roots, relative


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def _torch_compute_violations(tree: ast.AST) -> list[str]:
    torch_aliases = {"torch"}
    functional_aliases: set[str] = set()
    direct_functional_calls: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "torch":
                    torch_aliases.add(alias.asname or "torch")
                elif alias.name == "torch.nn.functional":
                    functional_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module == "torch.nn.functional":
            direct_functional_calls.update(alias.asname or alias.name for alias in node.names)

    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            violations.append("Python/PyTorch matrix multiplication is forbidden")
            continue
        if not isinstance(node, ast.Call):
            continue
        dotted = _dotted_name(node.func)
        parts = dotted.split(".")
        if not dotted:
            continue
        if parts[0] in direct_functional_calls or any(
            dotted == alias or dotted.startswith(alias + ".") for alias in functional_aliases
        ):
            violations.append(f"PyTorch functional call is forbidden: {dotted}")
        elif parts[0] in torch_aliases and len(parts) >= 2:
            suffix = parts[1:]
            if suffix[0] == "ops":
                violations.append(f"torch.ops dispatch is forbidden: {dotted}")
            elif suffix[:2] == ["nn", "functional"]:
                violations.append(f"PyTorch functional call is forbidden: {dotted}")
            elif suffix[-1] in _BANNED_TORCH_COMPUTE:
                violations.append(f"PyTorch compute call is forbidden: {dotted}")
    return list(dict.fromkeys(violations))


def _framework_markers(source: str, raw_source: str) -> dict[Dsl, bool]:
    return {
        Dsl.TRITON: bool(re.search(r"(?:^|\n)\s*(?:import|from)\s+triton\b", source)),
        Dsl.CUTEDSL: "cutlass.cute" in source or "@cute.kernel" in source,
        Dsl.CUDA: "__global__" in source
        and any(marker in raw_source for marker in _CUDA_LOADER_MARKERS),
    }


def _normalized_dependency(value: object) -> str:
    text = re.split(r"[<>=!~\[; ]", str(value).strip(), maxsplit=1)[0]
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _solution_violations(path: Path, dsl: Dsl) -> list[str]:
    if not path.is_file():
        return []
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return [f"solution.json is invalid: {type(error).__name__}: {error}"]
    if not isinstance(value, dict):
        return ["solution.json must be an object"]
    spec = value.get("spec")
    if spec is None:
        return []
    if not isinstance(spec, dict):
        return ["solution.json spec must be an object"]
    errors: list[str] = []
    dependencies = spec.get("dependencies", [])
    if not isinstance(dependencies, list):
        errors.append("solution.json spec.dependencies must be a list")
    else:
        for dependency in dependencies:
            token = _normalized_dependency(dependency)
            if token and token not in _ALLOWED_DEPENDENCIES[dsl]:
                errors.append(f"solution dependency is not approved for {dsl.value}: {dependency}")
    languages = spec.get("languages")
    if isinstance(languages, list):
        normalized = {_normalized_dependency(language) for language in languages}
        expected = {
            Dsl.TRITON: "triton",
            Dsl.CUTEDSL: "cutedsl",
            Dsl.CUDA: "cuda",
        }[dsl]
        allowed_languages = {"python", "pytorch", "torch", expected}
        unexpected = sorted(
            language for language in normalized if language not in allowed_languages
        )
        if expected not in normalized:
            errors.append(f"solution.json languages omit fixed DSL: {dsl.value}")
        if unexpected:
            errors.append(f"solution.json languages contain alternate DSLs: {unexpected}")
    return errors


__all__ = [
    "CandidateProductionValidator",
    "ProductionKernelPolicy",
    "RegistryProductionKernelValidator",
]
