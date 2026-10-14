"""Transport sealed trees through native Eval and Runtime-owned diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from ..kernel_sources import KernelSourceBundle, KernelSourceContract
from .abba import CommitPinnedAtrexBenchEvaluator
from .async_sdk import call_method, prepare_call
from .contract import AgateEvaluationContractV1
from .eval_sources import prepare_eval_sources
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
        kind, request = self._prepare_submission(kind, payload)
        return self._client.submit_job(kind, request)  # type: ignore[attr-defined,no-any-return]

    async def call_async(self, method: str, *args: object, **kwargs: object) -> Any:
        if method == "submit_job":
            kind, payload = await prepare_call(self._prepare_submission, *args)
            return await call_method(self._client, method, kind, payload)
        value = await call_method(self._client, method, *args, **kwargs)
        return self._decode_job(value) if method == "get_job" else value

    def _prepare_submission(
        self,
        kind: str,
        payload: dict[str, object],
    ) -> tuple[str, dict[str, object]]:
        raw = payload.get(SOURCE_REQUEST_KEY)
        if raw is None:
            return kind, payload
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
            return "dev", request
        candidate, _, attachments = prepare_eval_sources(source, requirements=contract.requirements)
        request = {key: value for key, value in payload.items() if key != SOURCE_REQUEST_KEY}
        request.update(attachments)
        request["candidate"] = candidate
        return "eval", request

    def get_job(self, job_id: str, **kwargs: Any) -> dict[str, object]:
        job = self._client.get_job(job_id, **kwargs)  # type: ignore[attr-defined]
        return self._decode_job(job)

    @staticmethod
    def _decode_job(job: Any) -> dict[str, object]:
        if (
            not isinstance(job, dict)
            or job.get("status") != "succeeded"
            or job.get("kind") != "dev"
        ):
            return job  # type: ignore[no-any-return]
        result = job.get("result")
        stdout = result.get("stdout") if isinstance(result, dict) else None
        if not isinstance(stdout, str):
            return job
        for line in reversed(stdout.splitlines()):
            if line.startswith(DIAGNOSTIC_PREFIX):
                try:
                    diagnostic = json.loads(line[len(DIAGNOSTIC_PREFIX) :])
                except ValueError:
                    # Preserve the transport's truncation evidence. The adapter
                    # knows whether this is a diagnostic or an ordinary Dev job.
                    return job
                if (
                    not isinstance(diagnostic, dict)
                    or diagnostic.get("operation") not in {"profile", "check", "disassemble"}
                    or not isinstance(diagnostic.get("passed"), bool)
                ):
                    return job
                return {**job, "result": diagnostic, "source_tree_execution": result}

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
