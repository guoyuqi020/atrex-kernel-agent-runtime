"""Import and enforce immutable, task-owned multi-file Kernel source contracts."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .artifacts.local import ArtifactKind, LocalArtifactStore
from .domain.ids import ArtifactDigest
from .domain.models import Dsl
from .git_import import SafeGitImporter


def source_path(value: str, *, dot: bool = False) -> str:
    """Validate portable paths before Git, filesystem, or remote use."""
    if dot and value == ".":
        return value
    if not value or "\\" in value or "\x00" in value or value.startswith(("/", "-")):
        raise ValueError(f"unsafe source path: {value!r}")
    if any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError(f"unsafe source path: {value!r}")
    return value


def ignored_source_path(path: PurePosixPath | Path, directory: bool = False) -> bool:
    """Only generated caches are omitted; source files never disappear silently."""
    return any(
        part in {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
        for part in path.parts
    ) or path.suffix in {".pyc", ".pyo"}


class SourceRevision(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    name: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
    archive_paths: tuple[str, ...] = Field(min_length=1)
    package_root: str = "."

    @field_validator("archive_paths")
    @classmethod
    def _paths(cls, paths: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(source_path(path) for path in paths)

    @field_validator("package_root")
    @classmethod
    def _package(cls, path: str) -> str:
        return source_path(path, dot=True)


class SourceManifest(BaseModel):
    """The GDN source/adapter/editable_roots declaration, without its controller."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: Literal[1] = 1
    name: str
    adapter: str
    source: SourceRevision
    editable_roots: tuple[str, ...] = Field(min_length=1)
    runtime_requirements: tuple[dict[str, str], ...] = ()
    # These describe the original Repository Horizon controller, not Runtime policy.
    repository_search: dict[str, object] = Field(default_factory=dict)
    bringup: dict[str, object] = Field(default_factory=dict)
    measurement: dict[str, object] = Field(default_factory=dict)
    runtime_support: tuple[dict[str, object], ...] = ()

    @field_validator("adapter")
    @classmethod
    def _adapter(cls, path: str) -> str:
        return source_path(path)

    @field_validator("editable_roots")
    @classmethod
    def _editable(cls, paths: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(source_path(path) for path in paths)

    @model_validator(mode="after")
    def _supported(self) -> SourceManifest:
        if self.runtime_support:
            raise ValueError(
                "runtime_support uploads are not supported; provision the GPU environment"
            )
        if self.repository_search.get("mode", "snapshot") != "snapshot":
            raise ValueError("only snapshot source manifests are supported")
        for requirement in self.runtime_requirements:
            if set(requirement) - {"distribution", "import", "version"}:
                raise ValueError("unknown runtime requirement field")
            if not requirement.get("distribution") or not requirement.get("import"):
                raise ValueError("runtime requirement needs distribution and import")
        return self


class KernelSourceContract(BaseModel):
    """Runtime-owned lock, sealed inside the Evaluation Contract, never in Candidate."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    source_revision: str = Field(pattern=r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
    seed_digest: ArtifactDigest = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    package_root: str
    editable_roots: tuple[str, ...]
    immutable_files: dict[str, str]
    runtime_requirements: tuple[dict[str, str], ...] = ()

    @field_validator("package_root")
    @classmethod
    def _package(cls, value: str) -> str:
        return source_path(value, dot=True)

    @field_validator("editable_roots")
    @classmethod
    def _editable(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("editable_roots cannot be empty")
        return tuple(source_path(item) for item in value)

    @field_validator("immutable_files")
    @classmethod
    def _immutable(cls, value: dict[str, str]) -> dict[str, str]:
        for path, digest in value.items():
            source_path(path)
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError("immutable source hashes must be SHA256 hex")
        return value

    def editable(self, path: str) -> bool:
        return any(path == root or path.startswith(root + "/") for root in self.editable_roots)

    def validate_tree(self, root: Path) -> dict[str, str]:
        """Validate every file, including fixed deletions and unauthorized additions."""
        if root.is_symlink() or not root.is_dir():
            raise ValueError("Kernel source root must be a real directory")
        files: dict[str, str] = {}
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root)
            if path.is_symlink():
                raise ValueError(f"source tree cannot contain symlinks: {relative}")
            if ignored_source_path(relative) or path.is_dir():
                continue
            name = source_path(relative.as_posix())
            package_path = PurePosixPath(name)
            if self.package_root != "." and package_path.is_relative_to(self.package_root):
                package_path = package_path.relative_to(self.package_root)
            if package_path.parts[0].removesuffix(".py") in {
                "atrex_bench",
                "torch",
                "cutlass",
                "cuda",
                "triton",
                "site",
                "sitecustomize",
                "usercustomize",
                "importlib",
                "packaging",
                "json",
                "subprocess",
                "os",
                "sys",
            }:
                raise ValueError(f"source tree shadows a protected import: {name}")
            if not path.is_file():
                raise ValueError(f"source tree contains a special file: {name}")
            content = path.read_bytes()
            try:
                files[name] = content.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError(
                    f"source tree must contain UTF-8 source, not build products: {name}"
                ) from error
            expected = self.immutable_files.get(name)
            if expected is not None:
                if hashlib.sha256(content).hexdigest() != expected:
                    raise ValueError(f"fixed source file was modified: {name}")
            elif not self.editable(name):
                raise ValueError(f"source file is outside editable_roots: {name}")
            elif path.suffix in {".so", ".a", ".o", ".cubin", ".whl", ".pth"}:
                raise ValueError(f"prebuilt/loading artifact is not permitted: {name}")
        missing = self.immutable_files.keys() - files.keys()
        if missing:
            raise ValueError(f"fixed source files were removed: {', '.join(sorted(missing))}")
        return files

    def seal(self, root: Path, artifacts: LocalArtifactStore) -> ArtifactDigest:
        self.validate_tree(root)
        return artifacts.put_directory(
            root,
            ArtifactKind.KERNEL,
            exclude=lambda path, directory: directory or ignored_source_path(path),
        )


def import_source_tree(
    manifest_path: Path,
    repository: Path,
    artifacts: LocalArtifactStore,
    *,
    candidate_path: str = "kernel.py",
) -> KernelSourceContract:
    """Archive exactly one commit; never mount or modify the input checkout."""
    manifest = SourceManifest.model_validate_json(manifest_path.read_bytes())
    source_path(candidate_path)
    adapter = manifest_path.parent / manifest.adapter
    if adapter.is_symlink() or not adapter.is_file():
        raise ValueError("source adapter must be a regular file beside its manifest")
    git = shutil.which("git")
    if git is None:
        raise ValueError("source import requires git in PATH")
    importer = SafeGitImporter(
        git, timeout_seconds=120, max_archive_bytes=64 * 1024 * 1024, label="Kernel source seed"
    )
    with tempfile.TemporaryDirectory(prefix="atrex-kernel-source-") as directory:
        temporary = Path(directory)
        archive, tree = temporary / "source.tar", temporary / "tree"
        importer.archive(
            repository, manifest.source.revision, archive, paths=manifest.source.archive_paths
        )
        tree.mkdir()
        importer.extract(archive, tree)
        target = tree / candidate_path
        if target.exists():
            raise ValueError("adapter path collides with archived source")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(adapter, target)
        for editable in manifest.editable_roots:
            if not (tree / editable).is_dir():
                raise ValueError(f"editable_root must be an imported directory: {editable}")
            if candidate_path == editable or candidate_path.startswith(editable + "/"):
                raise ValueError("adapter cannot be editable")
        if not (tree / manifest.source.package_root).is_dir():
            raise ValueError("source package_root is not an imported directory")
        immutable = {
            path.relative_to(tree).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in tree.rglob("*")
            if path.is_file()
            and not any(
                path.relative_to(tree).as_posix().startswith(root + "/")
                for root in manifest.editable_roots
            )
        }
        digest = artifacts.put_directory(
            tree, ArtifactKind.KERNEL, exclude=lambda path, directory: directory
        )
        contract = KernelSourceContract(
            source_revision=manifest.source.revision,
            seed_digest=digest,
            package_root=manifest.source.package_root,
            editable_roots=manifest.editable_roots,
            immutable_files=immutable,
            runtime_requirements=manifest.runtime_requirements,
        )
        contract.validate_tree(tree)
        return contract


def source_instructions(contract: KernelSourceContract) -> str:
    return (
        "\n# Kernel source tree\n"
        "work/kernel/ contains the entire Candidate source tree, with no extra source/ layer. "
        "The adapter and all paths outside editable_roots are fixed. You may edit, add, or "
        "delete source files only within these roots. Keep generated builds/caches in scratch/. "
        "Evaluate, profile, comparison, nomination and historical adoption use the entire tree; "
        "pass a directory (not an individual Python file) for comparison candidates. "
        "An unchanged historical tree can be adopted; a source change is not mandatory.\n"
        "Source-tree evaluation supports full, correctness_only, custom input/shapes and ABBA. "
        "Profile, check and disassemble also submit the entire tree through Runtime-owned "
        "Agate Dev drivers; you do not need to construct a Dev command. On NVIDIA GPUs use "
        'profile {"level":"survey|sol|deep"}; deep requires kernel_name or kernel_regex. '
        "Optional source=true returns source-correlated SASS, launch_skip/launch_count select "
        "launches within one post-warmup Model.forward, and shape_id selects an opaque case. "
        'check {"sanitize":"memcheck|racecheck|initcheck|synccheck"} runs a GPU sanitizer; '
        "without sanitize it triggers JIT compilation and one launch, not a correctness gate. "
        'disassemble {"fmt":"auto|sass|ptx"} extracts assembly; PTX availability depends on '
        "the installed NCU/toolchain. Check and disassemble use the first opaque case. "
        "arch on check must match the allocated GPU; it is not cross-compilation. "
        "Diagnostic passed=false is a failure even if the transport completed. "
        "These diagnostic results cannot replace Evaluate evidence. "
        "The fixed adapter may import the bundled package; this is not a prebuilt fallback.\n"
        + json.dumps(
            {"editable_roots": contract.editable_roots, "package_root": contract.package_root},
            ensure_ascii=False,
        )
        + "\n"
    )


def inject_source_instructions(root: Path, contract: KernelSourceContract) -> None:
    """Extend the Runtime-owned fragment and its integrity digest together."""
    prompt = root / ".runtime/evidence-instructions.md"
    manifest = root / ".runtime/evidence-manifest.json"
    text = prompt.read_text(encoding="utf-8") + source_instructions(contract)
    value = json.loads(manifest.read_bytes())
    value["prompt_fragment_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    for path, content in ((prompt, text), (manifest, json.dumps(value))):
        path.chmod(0o600)
        path.write_text(content, encoding="utf-8")
        path.chmod(0o400)
    for relative in contract.immutable_files:
        (root / "work/kernel" / relative).chmod(0o400)


def inject_bootstrap_source_instructions(root: Path, contract: KernelSourceContract) -> None:
    """Write source rules for the launcher's session-only Bootstrap prompt projection."""
    text = source_instructions(contract) + (
        "\n## Source-tree Bootstrap\n"
        "The complete pinned implementation is already supplied in input/kernel/ and "
        "work/kernel/. Establish the first correct baseline from this tree. Evaluate the "
        "unchanged seed first; repair only within editable_roots if necessary. Do not "
        "rewrite the fixed adapter, replace the package with a single-file implementation, "
        "or import an optimization result from another run. This supplied source package "
        "is the task's seed, not a prebuilt fallback or a wholesale reference copy. "
        "Follow the normal Bootstrap Direction/Experiment Journal and terminal-report "
        "workflow, including exactly one baseline Experiment. A correct unchanged seed "
        "is a valid nomination. Runtime independently evaluates the entire nominated "
        "tree before registering v0.\n"
    )
    fragment = root / ".runtime/source-instructions.md"
    fragment.write_text(text, encoding="utf-8")
    fragment.chmod(0o400)
    for relative in contract.immutable_files:
        (root / "work/kernel" / relative).chmod(0o400)


def load_kernel_source_contract(
    artifacts: LocalArtifactStore, digest: ArtifactDigest, dsl: Dsl
) -> KernelSourceContract | None:
    """Project only source rules; worker assembly does not interpret private test inputs."""
    stored = artifacts.verify(digest)
    if stored.kind is not ArtifactKind.EVALUATION_CONTRACT:
        raise ValueError("source rules require an Evaluation Contract Artifact")
    value = json.loads((stored.payload_path / "value.json").read_bytes())
    sources = value.get("kernel_sources", {})
    return (
        None
        if dsl.value not in sources
        else KernelSourceContract.model_validate(sources[dsl.value])
    )


@dataclass(frozen=True)
class KernelSourceBundle:
    """Validated sources passed internally to trusted evaluators, not an Agent API."""

    files: dict[str, str]
    contract: KernelSourceContract
    entrypoint: str


def read_kernel_source(
    root: Path, entrypoint: str, contract: KernelSourceContract | None
) -> str | KernelSourceBundle:
    if contract is None:
        return (root / entrypoint).read_text(encoding="utf-8")
    return KernelSourceBundle(contract.validate_tree(root), contract, entrypoint)
