# Atrex Kernel Agent Runtime documentation

English | [中文](README.zh.md)

## Use and operate

- [User Guide](user-guide.md): install, configure, Bootstrap, run, inspect, and recover.
- [Configuration Reference](configuration.md): Runtime schema v1 and Campaign schema v3.
- [Interface Reference](interfaces.md): CLI, HTTP, Optimizer Runtime Tools, and Evolver filesystem.
- [Deployment and Operations](operations.md): production topology, isolation, backup, and incidents.
- [Runnable Examples](../examples/README.md) and
  [Production Runner](../scripts/production/README.md).

## Understand and verify

- [Design Principles](design-principles.md): why the Runtime separates Agent evolution from trusted
  control, uses Lineage-local competition, and keeps history append-only.
- [Design and Implementation](../DESIGN.md): AKA limitations, missing evolution capabilities,
  Runtime responsibilities, end-to-end execution, implementation mapping, and remaining risks.
- [Architecture](architecture.md): terminology, ownership, lifecycle, storage, isolation, and source
  organization.
- [Evaluation and Promotion](evaluation.md): private contracts, correctness, Production Gate,
  Evaluate/ABBA, Roofline, NCU, and selection.
- [Protocols](protocols.md): durable identities, Artifacts, Evidence, Session, and visibility rules.
- [Architecture Decisions](decisions/README.md): rationale that still constrains the implementation.
- [Testing and Production Acceptance](testing-and-acceptance.md): repository checks and deployment
  evidence.
- [Release Checklist](release-checklist.md) and [Changelog](../CHANGELOG.md).

## Documentation authority

Current code and strict schemas are executable authority. Configuration and Interface references
define the supported public surface. Architecture and Protocols describe current semantics;
Architecture Decisions only explain why constraints exist. Production readiness is established by
the Testing/Acceptance document and Release Checklist, not by a manually maintained feature-status
table.
