"""Opaque durable identifiers and artifact digests."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import NewType
from uuid import uuid4

CampaignId = NewType("CampaignId", str)
LineageId = NewType("LineageId", str)
EpochId = NewType("EpochId", str)
AttemptId = NewType("AttemptId", str)
KernelAgentRevisionId = NewType("KernelAgentRevisionId", str)
KernelRevisionId = NewType("KernelRevisionId", str)
CampaignTaskId = NewType("CampaignTaskId", str)
WorkerSessionId = NewType("WorkerSessionId", str)
ArtifactDigest = NewType("ArtifactDigest", str)

_ID_SUFFIX = re.compile(r"^[0-9a-f]{32}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


def _new_id[Identifier](prefix: str, constructor: Callable[[str], Identifier]) -> Identifier:
    return constructor(f"{prefix}_{uuid4().hex}")


def _parse_id[Identifier](
    value: str,
    prefix: str,
    constructor: Callable[[str], Identifier],
) -> Identifier:
    expected = f"{prefix}_"
    if not value.startswith(expected) or _ID_SUFFIX.fullmatch(value[len(expected) :]) is None:
        raise ValueError(f"invalid {prefix} identifier: {value!r}")
    return constructor(value)


def new_campaign_id() -> CampaignId:
    """Return a new Campaign identifier."""
    return _new_id("campaign", CampaignId)


def new_lineage_id() -> LineageId:
    """Return a new lineage identifier."""
    return _new_id("lineage", LineageId)


def new_epoch_id() -> EpochId:
    """Return a new epoch identifier."""
    return _new_id("epoch", EpochId)


def new_attempt_id() -> AttemptId:
    """Return a new Attempt identifier."""
    return _new_id("attempt", AttemptId)


def new_kernel_agent_revision_id() -> KernelAgentRevisionId:
    """Return a new Kernel Agent revision identifier."""
    return _new_id("agentrev", KernelAgentRevisionId)


def new_kernel_revision_id() -> KernelRevisionId:
    """Return a new Kernel revision identifier."""
    return _new_id("kernelrev", KernelRevisionId)


def new_campaign_task_id() -> CampaignTaskId:
    """Return a new Campaign task identifier."""
    return _new_id("task", CampaignTaskId)


def new_worker_session_id() -> WorkerSessionId:
    """Return a new Worker session identifier."""
    return _new_id("workersession", WorkerSessionId)


def parse_campaign_id(value: str) -> CampaignId:
    """Validate a Campaign identifier from persistence or the wire."""
    return _parse_id(value, "campaign", CampaignId)


def parse_lineage_id(value: str) -> LineageId:
    """Validate a lineage identifier from persistence or the wire."""
    return _parse_id(value, "lineage", LineageId)


def parse_epoch_id(value: str) -> EpochId:
    """Validate an epoch identifier from persistence or the wire."""
    return _parse_id(value, "epoch", EpochId)


def parse_attempt_id(value: str) -> AttemptId:
    """Validate an Attempt identifier from persistence or the wire."""
    return _parse_id(value, "attempt", AttemptId)


def parse_kernel_agent_revision_id(value: str) -> KernelAgentRevisionId:
    """Validate a Kernel Agent revision identifier from persistence or the wire."""
    return _parse_id(value, "agentrev", KernelAgentRevisionId)


def parse_kernel_revision_id(value: str) -> KernelRevisionId:
    """Validate a Kernel revision identifier from persistence or the wire."""
    return _parse_id(value, "kernelrev", KernelRevisionId)


def parse_campaign_task_id(value: str) -> CampaignTaskId:
    """Validate a Campaign task identifier."""
    return _parse_id(value, "task", CampaignTaskId)


def parse_worker_session_id(value: str) -> WorkerSessionId:
    """Validate a Worker session identifier."""
    return _parse_id(value, "workersession", WorkerSessionId)


def parse_artifact_digest(value: str) -> ArtifactDigest:
    """Validate a SHA-256 artifact digest from persistence or the wire."""
    if _DIGEST.fullmatch(value) is None:
        raise ValueError(f"invalid artifact digest: {value!r}")
    return ArtifactDigest(value)
