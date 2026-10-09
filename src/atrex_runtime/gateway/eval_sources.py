"""JSON-safe, sealed source archives for Agate's native Eval transport."""

from __future__ import annotations

from collections.abc import Sequence

from ..kernel_sources import KernelSourceBundle
from .oss_remote import validate_path

EVAL_ARCHIVES_KEY = "__atrex_eval_archives"


def prepare_eval_sources(
    candidate: str | KernelSourceBundle,
    baseline: str | KernelSourceBundle | None = None,
    *,
    requirements: Sequence[str] = (),
) -> tuple[str | dict[str, object], str | dict[str, object] | None, dict[str, object]]:
    """Describe exact trees without staging files or invoking a remote service.

    The OSS adapter uploads both archives in one Eval reservation. Keeping source
    text here makes retries and durable request identities independent of upload
    URLs. No import shims, source rewriting or Dev fallback are involved.
    """
    archives: dict[str, dict[str, str]] = {}
    dependencies = list(requirements)

    def prepare(source: str | KernelSourceBundle, label: str) -> str | dict[str, object]:
        if isinstance(source, str):
            return source
        if source.contract.package_root != ".":
            raise ValueError(
                "native Agate Eval source archives require package_root='.'; "
                "a separate package import root is not supported (no Dev fallback)"
            )
        validate_path(source.entrypoint)
        if source.entrypoint not in source.files:
            raise ValueError("native Eval source entry_point is missing from its archive")
        for path in source.files:
            validate_path(path)
        archive = f"archives/{label}.tar.gz"
        archives[archive] = dict(source.files)
        for requirement in source.contract.runtime_requirements:
            dependency = requirement["distribution"] + requirement.get("version", "")
            if dependency not in dependencies:
                dependencies.append(dependency)
        return {"archive": archive, "entry_point": source.entrypoint}

    candidate_wire = prepare(candidate, "candidate")
    baseline_wire = None if baseline is None else prepare(baseline, "baseline")
    attachments: dict[str, object] = {}
    if archives:
        attachments[EVAL_ARCHIVES_KEY] = archives
        if dependencies:
            attachments["requirements"] = dependencies
    return candidate_wire, baseline_wire, attachments
