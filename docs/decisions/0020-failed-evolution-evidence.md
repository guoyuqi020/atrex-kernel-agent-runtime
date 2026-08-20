# Decision 0020: Retain bounded evidence for failed Evolution runs

English | [中文](0020-failed-evolution-evidence.zh.md)

## Status

Accepted and implemented.

## Context

Successful Evolver runs already produced immutable provenance, but timeouts, process failures, and rejected Candidate manifests had only lifecycle Events and mutable execution workspaces. That made failed self-evolution harder to audit and prevented a uniform Artifact-retention policy. Sealing an entire untrusted workspace would be unsafe and unbounded.

## Decision

Every failed Evolver invocation attempts to seal an `EVOLUTION` failure artifact. The strict version-1 record contains the immutable input manifest, failure phase, exception type, and, when the Coding Agent returned a structured result, its non-secret Agent descriptor, return code, bounded stdout/stderr, validated token report, and optional separately sealed Session Trace. It does not copy the Candidate tree, exception message, prompt, argv, environment values, or secrets.

The Worker timeout/failure or `evolution.candidate_rejected` Event carries `failure_artifact_digest`. If failure-evidence sealing itself fails, the original exception remains authoritative and the Event records only `failure_retention_error_type`; retention must never mask or reclassify the primary failure. A failed trace never creates or promotes a Kernel Agent Revision.

## Consequences

Successful and unsuccessful self-evolution are both auditable through immutable bounded Artifacts, while promotion semantics remain unchanged. Raw Session retention is explicit and content-addressed when configured; the execution workspace remains diagnostic deployment state until bounded offline workspace GC. Failure artifacts participate in Event-rooted transitive Artifact GC retention.
