"""Transport multi-file Evaluate through the trusted Dev evaluator, not Agent commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from ..domain.errors import InfrastructureError
from ..kernel_sources import KernelSourceBundle, KernelSourceContract
from .abba import CommitPinnedAtrexBenchEvaluator, build_abba_source_request
from .abba_remote import RESULT_PREFIX
from .contract import AgateEvaluationContractV1
from .protocol import AGATE_MAX_JOB_TIMEOUT_S
from .source_diagnostics import DIAGNOSTIC_PREFIX

SOURCE_REQUEST_KEY = "__atrex_kernel_source_tree"


def attach_source_tree(
    payload: dict[str, object],
    source: KernelSourceBundle,
    contract: AgateEvaluationContractV1,
    hardware_target: str,
) -> dict[str, object]:
    """Keep a JSON-safe descriptor so copies/retries retain the exact sealed sources."""
    payload[SOURCE_REQUEST_KEY] = {
        "files": source.files,
        "source_contract": source.contract.model_dump(mode="json"),
        "contract": contract.model_dump(mode="json"),
        "hardware_target": hardware_target,
    }
    return payload


class SourceTreeAgateClient:
    """Keep native jobs unchanged; adapt explicitly Runtime-created tree evaluations."""

    def __init__(self, client: object, evaluator: CommitPinnedAtrexBenchEvaluator | None):
        self._client = client
        self._evaluator = evaluator

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    def submit_job(self, kind: str, payload: dict[str, object]) -> dict[str, object]:
        raw = payload.get(SOURCE_REQUEST_KEY)
        if raw is None:
            return self._client.submit_job(kind, payload)  # type: ignore[attr-defined,no-any-return]
        if kind not in {"eval", "profile", "compile", "disassemble"}:
            raise ValueError(f"unsupported source-tree operation: {kind}")
        if not isinstance(raw, dict):
            raise ValueError("invalid Runtime source-tree descriptor")
        contract = AgateEvaluationContractV1.model_validate(raw["contract"])
        source = KernelSourceBundle(
            raw["files"],
            KernelSourceContract.model_validate(raw["source_contract"]),
            contract.candidate_path,
        )
        if kind != "eval":
            request = build_diagnostic_request(
                kind, payload, source, contract, raw["hardware_target"]
            )
            return self._client.submit_job("dev", request)  # type: ignore[attr-defined,no-any-return]
        if self._evaluator is None:
            raise ValueError("source-tree Evaluate requires gate_policy.evaluator")
        timeout = float(contract.options.timeout_s)
        request = build_abba_source_request(
            hardware_target=raw["hardware_target"],
            contract=contract,
            shape_ids=sorted(contract.shapes),
            schedule=[{"revision": "candidate", "repeat": 0}],
            incumbent_source=source,
            candidate_source=source,
            evaluator_files=self._evaluator.files(),
            per_run_timeout_seconds=timeout,
            allocation_timeout_seconds=timeout + 120,
        )
        files = cast(dict[str, str], request["files"])
        driver = json.loads(files["request.json"])
        driver["raw_result"] = True
        # A single Evaluate has no incumbent; do not upload a second identical tree.
        for path in tuple(files):
            if path.startswith("snapshots/incumbent/"):
                del files[path]
        driver["sources"]["incumbent"] = driver["sources"]["candidate"]
        files["request.json"] = json.dumps(driver)
        files.setdefault("reference/metadata.json", "{}")
        request["idempotency_key"] = payload.get("idempotency_key")
        request["dev_note"] = "trusted source-tree Evaluate"
        return self._client.submit_job("dev", request)  # type: ignore[attr-defined,no-any-return]

    def get_job(self, job_id: str, **kwargs: Any) -> dict[str, object]:
        job = self._client.get_job(job_id, **kwargs)  # type: ignore[attr-defined]
        if not isinstance(job, dict) or job.get("status") != "succeeded":
            return job  # type: ignore[no-any-return]
        result = job.get("result")
        stdout = result.get("stdout") if isinstance(result, dict) else None
        if not isinstance(stdout, str):
            return job
        for line in reversed(stdout.splitlines()):
            if line.startswith(DIAGNOSTIC_PREFIX):
                diagnostic = json.loads(line[len(DIAGNOSTIC_PREFIX) :])
                if diagnostic.get("operation") not in {"profile", "check", "disassemble"}:
                    raise InfrastructureError("invalid source-tree diagnostic result")
                return {**job, "result": diagnostic, "source_tree_execution": result}
            if not line.startswith(RESULT_PREFIX):
                continue
            raw = json.loads(line[len(RESULT_PREFIX) :])
            if not raw.get("raw_result"):
                return job
            runs = raw.get("runs", [])
            if raw.get("error") or len(runs) != 1:
                raise InfrastructureError(f"source-tree evaluator failed: {raw.get('error')}")
            output = runs[0].get("result")
            if not isinstance(output, dict) or not isinstance(output.get("raw_result"), dict):
                raise InfrastructureError(f"source-tree evaluator produced no result: {runs[0]}")
            # Logical operation ownership remains in the durable Agate job binding;
            # a Dev tool result never becomes trusted Evaluate evidence by normalization.
            return {**job, "result": output["raw_result"], "source_tree_execution": result}
        return job


def build_diagnostic_request(
    kind: str,
    payload: dict[str, object],
    source: KernelSourceBundle,
    contract: AgateEvaluationContractV1,
    hardware_target: str,
) -> dict[str, object]:
    """Stage one sealed tree and a Runtime-owned diagnostic, never Agent shell commands."""
    parameters = dict(cast(dict[str, object], payload.get("parameters", {})))
    if parameters.get("profiler") not in (None, "ncu"):
        raise ValueError("source-tree diagnostics currently require NVIDIA NCU, not rocprofv3")
    if parameters.get("fmt") == "isa":
        raise ValueError(
            "source-tree disassemble on NVIDIA supports fmt=auto, sass or ptx, not isa"
        )
    allocation_timeout = min(AGATE_MAX_JOB_TIMEOUT_S, int(contract.options.timeout_s) + 30)
    files = {f"candidate/{path}": text for path, text in source.files.items()}
    files.update(
        {
            "__atrex_diagnostic.py": Path(__file__).with_name("source_diagnostics.py").read_text(),
            "__atrex_abba.py": Path(__file__).with_name("abba_remote.py").read_text(),
            "input.py": contract.input_py,
            "request.json": json.dumps(
                {
                    "operation": "check" if kind == "compile" else kind,
                    "entrypoint": source.entrypoint,
                    "package_root": source.contract.package_root,
                    "runtime_requirements": source.contract.runtime_requirements,
                    "requirements": parameters.get("requirements") or list(contract.requirements),
                    "shapes": contract.shapes,
                    "parameters": parameters,
                    "lock_clocks": contract.lock_clocks,
                    "timeout_s": min(contract.options.timeout_s, allocation_timeout - 30),
                }
            ),
        }
    )
    return {
        "spec": {"target_hardware": [hardware_target]},
        "command": "python3 __atrex_diagnostic.py request.json",
        "files": files,
        "env_vars": {**contract.env_vars, **cast(dict[str, str], parameters.get("env_vars", {}))},
        "timeout_s": allocation_timeout,
        "recycle": True,
        "dev_intent": "sanitize"
        if parameters.get("sanitize")
        else ("compile" if kind == "compile" else "profile_adhoc"),
        "dev_note": f"Runtime source-tree {kind}",
        "idempotency_key": payload.get("idempotency_key"),
    }
