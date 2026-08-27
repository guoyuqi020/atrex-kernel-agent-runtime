"""Shared trusted execution primitives for Agate SDK-backed use cases."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

import anyio
from pydantic import TypeAdapter

from ..artifacts.local import ArtifactKind, JsonValue, LocalArtifactStore
from ..domain.errors import InfrastructureError
from ..domain.ids import ArtifactDigest
from ..domain.models import Dsl
from ..roofline import strip_roofline_hardware_suffix
from .agate import AgateCandidateRejection, AgateClient, AgateRequestBuilder
from .contract import AgateEvaluationContractV1

_JSON_ADAPTER: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)


def build_evaluation_request(
    request_builder: AgateRequestBuilder,
    *,
    candidate_source: str,
    operator: str,
    contract: AgateEvaluationContractV1,
    hardware_target: str,
    dsl: Dsl,
    name: str,
    idempotency_key: str,
) -> dict[str, object]:
    """Build the canonical Agate Evaluate payload from one sealed contract."""
    reference: dict[str, object] = {
        "operator": operator,
        "reference_py": contract.reference_py,
        "input_py": contract.input_py,
        "shapes": contract.shapes,
    }
    if contract.metadata is not None:
        reference["metadata"] = contract.metadata
    if contract.roofline is not None:
        reference["roofline"] = strip_roofline_hardware_suffix(contract.roofline)
    return request_builder(
        candidate_source,
        reference,
        hardware_target,
        name=name,
        spec_fields={"languages": [dsl.value]},
        options=cast(Mapping[str, object], contract.options.model_dump(mode="python")),
        env_vars=contract.env_vars or None,
        requirements=contract.requirements or None,
        deps_mode=contract.deps_mode,
        mode=contract.mode,
        lock_clocks=contract.lock_clocks,
        harness=contract.harness,
        atrex_bench_version=contract.atrex_bench_version,
        runner_overrides=cast(Mapping[str, object], contract.runner_overrides) or None,
        idempotency_key=idempotency_key,
    )


async def call_agate_json(
    operation: Callable[[], object],
    *,
    request_error: str,
    invalid_response: str,
    non_object_response: str,
) -> dict[str, JsonValue]:
    """Run one blocking SDK call and validate a JSON-object response."""
    try:
        value = await anyio.to_thread.run_sync(operation)
    except Exception as error:
        raise InfrastructureError(f"{request_error}: {type(error).__name__}: {error}") from error
    return require_json_object(
        value,
        invalid_response=invalid_response,
        non_object_response=non_object_response,
    )


async def submit_agate_job(
    client: AgateClient,
    kind: str,
    payload: dict[str, object],
    *,
    request_error: str,
    invalid_response: str,
    non_object_response: str,
) -> dict[str, JsonValue]:
    """Submit one job while preserving structured candidate-validation rejection."""
    try:
        value = await anyio.to_thread.run_sync(lambda: client.submit_job(kind, payload))
    except Exception as error:
        fields = vars(error)
        if (
            kind == "eval"
            and fields.get("status") in {400, 422}
            and fields.get("error_class") == "validation"
        ):
            try:
                rejection = _JSON_ADAPTER.validate_python(fields.get("payload"))
            except ValueError:
                pass
            else:
                raise AgateCandidateRejection(rejection) from error
        raise InfrastructureError(f"{request_error}: {type(error).__name__}: {error}") from error
    return require_json_object(
        value,
        invalid_response=invalid_response,
        non_object_response=non_object_response,
    )


def require_json_object(
    value: object,
    *,
    invalid_response: str,
    non_object_response: str,
) -> dict[str, JsonValue]:
    """Normalize an SDK value into the Runtime's bounded JSON object type."""
    try:
        normalized = _JSON_ADAPTER.validate_python(value)
    except ValueError as error:
        raise InfrastructureError(invalid_response) from error
    if not isinstance(normalized, dict):
        raise InfrastructureError(non_object_response)
    return normalized


def store_gateway_result(
    artifacts: LocalArtifactStore,
    job: JsonValue,
    profile: JsonValue | None,
    *,
    temporary_prefix: str,
) -> ArtifactDigest:
    """Seal an Eval alone or an Eval/Profile pair as one Gateway Result Artifact."""
    if profile is None:
        return artifacts.put_json(job, ArtifactKind.GATEWAY_RESULT)
    with tempfile.TemporaryDirectory(prefix=temporary_prefix) as directory:
        root = Path(directory)
        for name, value in (("value.json", job), ("profile.json", profile)):
            root.joinpath(name).write_text(
                json.dumps(
                    value,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                encoding="utf-8",
            )
        return artifacts.put_directory(root, ArtifactKind.GATEWAY_RESULT)
