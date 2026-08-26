# Atrex Kernel Agent Runtime documentation

English | [中文](README.zh.md)

## Start here

- [User Guide](user-guide.md): install, configure, bootstrap, run, inspect, recover, and maintain.
- [Interface Reference](interfaces.md): CLI, HTTP API, Optimizer Runtime Tools, Evolver filesystem
  inputs, and external service contracts.
- [Configuration Reference](configuration.md): Runtime schema v1 and Campaign schema v3.
- [Runnable examples](../examples/README.md): remote Agate, Bootstrap, Lineage, Evolution, dev
  shells, and local Wiki.
- [Production runner](../scripts/production/README.md): persistent control plane, managed Campaign
  tasks, per-DSL inspection, and recovery.

## Design

- [Architecture](architecture.md): trust boundary, lifecycle, isolation, and data flow.
- [Module design](module-design.md): implementation modules and ownership.
- [Runtime code organization](code-organization.md): dependency direction and source layout.
- [Full-repository Agent revision](full-repository-optimizer-revision.md): Git import, sealing,
  evolution, and provenance.
- [Performance gates](performance-gates.md): correctness, Production Gate, Evaluate, and ABBA.
- [Trusted Roofline construction](roofline-builder.md): optional commit-pinned Roofline generation.
- [Architecture decisions](decisions/README.md): current decisions only.

## Operations and release

- [Deployment and operations](operations.md): production topology, sandbox setup, backup, recovery,
  retention, and incidents.
- [Protocols](protocols.md): durable schemas and Artifact semantics.
- [Implementation status](implementation-status.md): implemented scope and remaining acceptance work.
- [Testing and production acceptance](testing-and-acceptance.md): automated and target-environment
  verification.
- [Release checklist](release-checklist.md): packaging and publication gate.
- [Changelog](../CHANGELOG.md): released behavior by version.

## Documentation authority

Current code and strict schemas are the executable authority. The interface and configuration
references describe the supported public surface. Architecture decisions explain why current
constraints exist; they do not add compatibility behavior. If target-environment acceptance is not
listed as complete in implementation status, the feature is implemented but not production-proven.
