"""Result identities replace Agent-authored Trial identities without weakening provenance."""

import json
from pathlib import Path

import pytest
from test_gateway_proxy import _request, _service

from atrex_runtime.gateway.journals import RuntimeJournalService
from atrex_runtime.gateway.protocol import gateway_agent_request_schema


@pytest.mark.anyio
async def test_result_identity_is_per_invocation_and_reread_returns_kernel(tmp_path: Path) -> None:
    registry, control, attempt, capability, service, _ = _service(tmp_path)
    try:
        payload = json.loads(_request(attempt))
        payload.update(operation="profile", level="sol", idempotency_key="profile-one")
        first = await service.execute(capability.token, json.dumps(payload).encode())
        replay = await service.execute(capability.token, json.dumps(payload).encode())
        payload["idempotency_key"] = "profile-two"
        second = await service.execute(capability.token, json.dumps(payload).encode())
        assert first == replay
        assert first.result == second.result
        assert first.kernel_artifact_digest == second.kernel_artifact_digest
        assert first.result_artifact_digest != second.result_artifact_digest
        assert "kernel_trial_id" not in first.model_dump()
        read = await service.execute(
            capability.token,
            json.dumps(
                {
                    "schema_version": 2,
                    "attempt_id": attempt.id,
                    "idempotency_key": "read-one",
                    "operation": "result_artifact_read",
                    "result_artifact_digest": first.result_artifact_digest,
                }
            ).encode(),
            operation_scope="runtime",
        )
        assert read.result["kernel_artifact_digest"] == first.kernel_artifact_digest
        assert read.result["result_artifact_digest"] == first.result_artifact_digest
        assert read.result["result"] == first.result
        trials = {t.id: t for t in control.list_kernel_trials((attempt.id,))}
        subject = RuntimeJournalService._materialize_experiment_subject(
            attempt.id,
            "after",
            {"result_artifact_digest": first.result_artifact_digest},
            trials,
        )
        assert subject == {
            "kernel_artifact_digest": first.kernel_artifact_digest,
            "result_artifact_digests": [first.result_artifact_digest],
        }
        with pytest.raises(ValueError, match="exactly"):
            RuntimeJournalService._materialize_experiment_subject(
                attempt.id,
                "after",
                {"kernel_trial_id": next(iter(trials))},
                trials,
            )
        with pytest.raises(ValueError, match="visible history"):
            RuntimeJournalService._materialize_experiment_subject(
                attempt.id,
                "after",
                {"result_artifact_digest": "sha256:" + "0" * 64},
                trials,
            )
    finally:
        control.close()
        registry.close()


def test_trial_show_is_not_an_agent_operation() -> None:
    with pytest.raises(ValueError):
        gateway_agent_request_schema("kernel_trial_show")
