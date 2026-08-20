# Decision 0012: Separate target design, implementation status, and release acceptance

English | [中文](0012-documentation-status-and-release-gates.zh.md)

## Status

Accepted.

## Context

The Runtime is being designed and implemented incrementally. Architecture prose necessarily describes end-state responsibilities, while code, automated checks, and deployment validation advance at different rates. A design document that uses unqualified completion language can make a partial mechanism look production-ready and can hide missing product semantics such as within-epoch branch memory or failed-epoch recovery.

## Decision

Documentation has three separate authorities:

1. `architecture.md` and `module-design.md` describe the target and annotate current implementation where useful.
2. `implementation-status.md` is the authoritative requirement-by-requirement statement of Implemented, Partial, Planned, Deferred, and Deployment-verification-required work.
3. `testing-and-acceptance.md` defines repository evidence and mandatory target-environment production gates.

Configuration, protocol, and operations references describe only supported interfaces and explicitly name operational gaps. Every English project document has a maintained Chinese peer. Completion claims must link to current evidence, and a design decision that completes named mechanisms must not imply that unrelated gates pass.

## Consequences

Feature work updates the implementation matrix, affected design/reference documents, and acceptance evidence together. Missing capabilities remain visible rather than being replaced with speculative interfaces. The repository can truthfully call its current single-node foundation substantial while prohibiting a production-ready declaration until all mandatory gates pass.
