# Decision 0003: Make Evidence publication a durable lineage handoff

English | [中文](0003-evidence-handoff.zh.md)

## Status

Accepted.

## Context

Epoch selection atomically promotes the winning Kernel Agent and best Kernel, but the next epoch also needs a new common Evidence checkpoint. Building that artifact copies and hashes immutable data outside the SQLite transaction. If a completed epoch immediately returned its lineage to `ready`, a crash could start the next epoch with stale Evidence or require guessing whether checkpoint assembly had finished.

Scheduling by “run N more epochs” has a related retry problem: after a crash, repeating the same request could run an extra epoch even when the original target had already completed.

## Decision

Each lineage durably owns its current Evidence checkpoint and per-branch Attempt budget. Completing an epoch atomically updates Agent and Kernel promotion, increments `next_epoch_number`, and changes the lineage to `awaiting_evidence`. No new epoch may start in that state.

`LocalEvidenceAssembler` deterministically copies the previous flat cumulative bundle, appends the completed epoch's authoritative Attempt, Kernel, evaluation, and selection facts, and seals a new artifact. A compare-and-swap operation then replaces the expected checkpoint digest and changes the lineage to `ready`. Repeating assembly before that operation yields the same digest; restarting after it observes `ready` and does no duplicate work.

`CampaignScheduler` accepts an absolute target epoch rather than a relative count. Repeating the same scheduling request resumes incomplete state and stops once every requested lineage has advanced beyond the target. Distinct DSL lineages in one Campaign may run concurrently, while Registry state serializes each lineage.

## Consequences

Epoch promotion and Evidence publication are two recoverable commits with an explicit intermediate state. The next epoch never observes stale history. Renewable Registry fencing serializes multiple scheduler processes sharing this SQLite deployment.
