# Decision 0032: Keep CAS identity separate from lineage-local Kernel versions

English | [中文](0032-lineage-kernel-version-labels.zh.md)

## Status

Accepted and implemented.

## Context

A content-addressed Artifact is identified by `sha256:<digest>`. The Digest is excellent for
verification and deduplication but does not show when a Kernel was referenced or how it relates to
earlier Kernels. Adding a timestamp or display version to the hashed manifest would make identical
content produce different Digests. A timestamp is also not intrinsic to a deduplicated object: the
same Artifact may be referenced by several durable records at different times.

Opaque Kernel Revision IDs and parent IDs already preserve identity and ancestry, but operators
cannot quickly read a `v0`, `v1`, `v2` history. Attempt evaluations additionally form exploratory,
retryable sequences and must not be confused with durable Kernel Revisions.

## Decision

CAS identity remains unchanged. Registry schema 16 adds `lineage_kernel_versions`, which assigns
every Kernel Revision one immutable, zero-based `revision_number` inside exactly one Lineage.
Bootstrap links the baseline as `v0`; each terminal Attempt Kernel receives the next number in the
same transaction that registers it. The mapping stores `linked_at`, enforces uniqueness by Kernel
and by `(lineage, revision_number)`, and requires a child parent to be present in the same Lineage.
Migration from schema 15 reconstructs stable numbers in Epoch/Attempt domain order.

Catalog projections expose `version`, `revision_number`, `parent_version`, disposition, semantic
creation time, and performance change relative to the parent. They retain all opaque IDs and
Digests. Artifact references are also projected as objects containing `digest`, `kind`, and
`referenced_at`; legacy Digest fields remain available. `list-kernels --format table` renders a
human-readable history.

Exploratory Gateway evaluations use a separate `g<recovery_generation>-e<ordinal>` label. They do
not consume Kernel `vN` numbers. A Runtime-final candidate only receives a Kernel version when the
trusted controller registers the terminal Kernel Revision.

## Consequences

Kernel versions remain stable across restarts, later insertions, and query-order changes. Siblings
may have consecutive version numbers while sharing the same `parent_version`, so the parent link—not
numeric adjacency—defines evolution. CAS deduplication and verification remain intact. Operators
must interpret `referenced_at` as the time of a semantic reference, not an intrinsic blob creation
time.
