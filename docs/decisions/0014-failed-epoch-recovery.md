# Decision 0014: Recover Failed Epochs by generation

English | [中文](0014-failed-epoch-recovery.zh.md)

## Status

Accepted and implemented.

## Context

An infrastructure-failed Attempt may exhaust its automatic retry budget after a host loss, expired Gateway capability, or unavailable sandbox backend. Treating the Failed Epoch as permanently terminal forces a new Campaign and loses the stable relationship among its Attempt identity, model-visible Evidence, Gateway operations, and audit history. Simply changing the status back to running would be unsafe because an old scheduler fence or bearer capability could still exist, and replaying old Gateway reservations would mix two operator-authorized executions.

## Decision

The Registry stores an idempotent recovery record keyed by `(epoch_id, recovery_key)`. A trusted operator supplies the key and a non-empty reason through `recover-epoch`. One transaction requires a Failed Epoch with at least one infrastructure-failed Attempt, reopens the same Epoch and Attempt identities, clears their exhausted infrastructure counts, increments their recovery generations, records a new authority start time, supersedes the lineage fence, reopens the lineage and eligible Campaign, and emits correlated recovery events. Exact replay returns the original record; key reuse with a different reason fails.

Gateway Control stores the Attempt recovery generation. Every operation authorization and outcome commit checks it against the Registry, so advancing Registry generation immediately rejects the old bearer and any in-flight authorization. The next issuance derives a new token and expiry and resets quota and revocation state. Generation-scoped idempotency reservations remain as audit history but cannot authorize the new generation. The original epoch Evidence and branch-local Attempt Evidence remain unchanged. An existing authoritative Gateway outcome is recovered before Worker launch and is never replaced by capability rotation.

Registry Schema 8 and Gateway Control Schema 2 intentionally reject earlier pre-release files.

## Consequences

Operators can resume an exhausted infrastructure failure without creating a replacement Attempt or contaminating competition accounting. Recovery is explicit, auditable, idempotent, and safe against stale schedulers and credentials. The command does not recover Agent-output, policy, or control-plane defects; those require their owning procedure. Target-image crash testing at every lifecycle edge remains required before production acceptance.
