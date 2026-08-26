"""Tests for the trusted Runtime-to-worker manifest."""

from __future__ import annotations

import json

import pytest
from conftest import digest
from pydantic import ValidationError

from atrex_runtime.domain.ids import (
    new_attempt_id,
    new_campaign_id,
    new_epoch_id,
    new_kernel_agent_revision_id,
    new_kernel_revision_id,
    new_lineage_id,
)
from atrex_runtime.domain.models import Dsl
from atrex_runtime.workers.manifest import AttemptInputManifestV9, AttemptTaskContextV5


def test_attempt_manifest_round_trips_and_excludes_evolver() -> None:
    manifest = AttemptInputManifestV9(
        attempt_id=new_attempt_id(),
        kernel_agent_revision_id=new_kernel_agent_revision_id(),
        input_kernel_revision_id=new_kernel_revision_id(),
        input_kernel_digest=digest("kernel"),
        epoch_evidence_checkpoint=digest("evidence"),
        attempt_evidence_digest=digest("attempt-evidence"),
        optimizer_digest=digest("optimizer"),
        dsl=Dsl.TRITON,
        context=AttemptTaskContextV5(
            campaign_id=new_campaign_id(),
            lineage_id=new_lineage_id(),
            epoch_id=new_epoch_id(),
            epoch_number=2,
            attempt_ordinal=3,
            operator="vector_add",
            hardware_target="h100",
            evaluation_contract_digest=digest("contract"),
            agent_problem_digest=digest("problem"),
        ),
    )

    payload = manifest.canonical_json_bytes()
    parsed = AttemptInputManifestV9.from_json_bytes(payload)

    assert parsed == manifest
    assert "evolver" not in payload.decode()
    assert json.loads(payload)["schema_version"] == 9
    assert "paths" not in json.loads(payload)
    assert json.loads(payload)["context"]["operator"] == "vector_add"


def test_attempt_manifest_rejects_unknown_fields_and_invalid_ids() -> None:
    payload = {
        "schema_version": 9,
        "attempt_id": "not-an-attempt",
        "kernel_agent_revision_id": str(new_kernel_agent_revision_id()),
        "input_kernel_revision_id": str(new_kernel_revision_id()),
        "input_kernel_digest": str(digest("kernel")),
        "epoch_evidence_checkpoint": str(digest("evidence")),
        "attempt_evidence_digest": str(digest("attempt-evidence")),
        "optimizer_digest": str(digest("optimizer")),
        "dsl": "triton",
        "context": {
            "campaign_id": str(new_campaign_id()),
            "lineage_id": str(new_lineage_id()),
            "epoch_id": str(new_epoch_id()),
            "epoch_number": 1,
            "attempt_ordinal": 1,
            "operator": "vector_add",
            "hardware_target": "h100",
            "evaluation_contract_digest": str(digest("contract")),
            "agent_problem_digest": str(digest("problem")),
        },
        "unexpected": True,
    }

    with pytest.raises(ValidationError):
        AttemptInputManifestV9.model_validate(payload)
