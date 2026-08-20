# Architecture decision records

English | [中文](README.zh.md)

These records explain decisions that still constrain the released implementation. Superseded
designs were removed from the release tree. Current behavior is defined by code, the
[architecture](../architecture.md), [configuration reference](../configuration.md), and
[interfaces](../interfaces.md).

## Trust, storage, and control

- [0001](0001-agate-sdk-adapter.md): Agate SDK adapter boundary
- [0003](0003-evidence-handoff.md): immutable Evidence handoff
- [0007](0007-immutable-agent-traces.md): immutable raw Agent traces
- [0012](0012-documentation-status-and-release-gates.md): documentation authority and release gates
- [0013](0013-branch-local-attempt-evidence.md): branch-local Attempt Evidence
- [0014](0014-failed-epoch-recovery.md): failed-Epoch recovery
- [0015](0015-durable-campaign-task-control-plane.md): durable Campaign Tasks
- [0016](0016-durable-correlated-lifecycle-events.md): correlated lifecycle Events
- [0017](0017-administration-lifecycle-and-cooperative-cancellation.md): administration lifecycle
- [0018](0018-readiness-and-offline-artifact-retention.md): readiness and offline retention
- [0019](0019-isolated-wheel-smoke.md): isolated-wheel release smoke
- [0020](0020-failed-evolution-evidence.md): failed-Evolution Evidence

## Agents, Wiki, and evolution

- [0021](0021-live-gpu-wiki-capability.md): live Wiki query and post-Epoch feedback
- [0022](0022-local-wiki-test-double.md): wire-compatible local Wiki
- [0027](0027-unified-epoch-evidence-view.md): Epoch-organized Evidence
- [0028](0028-commit-only-runtime-boundary.md): commit-only Runtime/Agent boundary
- [0031](0031-unbounded-evolver-token-accounting.md): Evolver token accounting without a cutoff
- [0035](0035-role-scoped-evolver-evidence.md): role-scoped Evolver Evidence
- [0036](0036-frozen-evolver-runtime-tools.md): frozen Evolver inspection tools
- [0037](0037-runtime-bound-agent-backends.md): Runtime-bound Backend policy
- [0039](0039-three-form-challenger-proposals.md): evolved/reuse/evolve-from-history proposals
- [0041](0041-lineage-bound-model-selection.md): Lineage-bound model selection
- [0043](0043-campaign-frozen-evolver-commit.md): Campaign-frozen Evolver commit
- [0045](0045-provider-native-usage-accounting.md): provider-native credits and token accounting

## Evaluation, versioning, and scheduling

- [0029](0029-append-only-bootstrap-generations.md): append-only Bootstrap generations
- [0030](0030-exploratory-and-authoritative-evaluations.md): exploratory and authoritative evaluation
- [0032](0032-lineage-kernel-version-labels.md): Kernel `vN` labels
- [0033](0033-lineage-agent-version-labels.md): Agent `agent-vN` labels
- [0034](0034-configurable-epoch-topology.md): configurable Epoch topology
- [0040](0040-bounded-parallel-branch-execution.md): bounded parallel Branch execution
- [0042](0042-artifact-seeded-lineages.md): Artifact-seeded Lineages
- [0046](0046-worker-host-network.md): shared host networking for sandboxed Workers
- [0047](0047-worker-owned-workspace-roots.md): Worker-created Sandbox roots
