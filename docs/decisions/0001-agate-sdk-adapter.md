# Decision 0001: Use the published Agate Python SDK

English | [中文](0001-agate-sdk-adapter.zh.md)

## Status

Accepted on 2026-08-15.

## Context

The trusted Gateway Proxy must expose Agate's remote command surface without exposing Gateway credentials to a Worker. The installed `atrex-gateway-client` 0.12.1 package provides a zero-runtime-dependency synchronous `Client`, pluggable authentication, a stable eval request builder, typed job submission, bounded long polling, cancellation, job and environment queries, liveness, and structured `GatewayError` fields.

## Decision

The Runtime directly uses `build_eval_request_from_content` and the applicable `Client` methods. It runs synchronous SDK calls in an AnyIO worker thread. It does not spawn the `agate` CLI and does not copy Agate's HTTP or AK/SK implementation. Agent-facing protocol v2 exposes evaluate/profile/dev/check/disassemble/env; Runtime retains Job lookup, polling, cancellation, health, and connection inspection for recovery and administration. Package update remains deployment-owned because it is not a Gateway request and mutates the trusted Python installation.

The SDK remains the upstream wire authority. Runtime-owned code validates deployment configuration, resolves sealed Campaign evaluation contracts, seals candidates, validates response JSON and Atrex-Bench result fields, classifies failures, persists external job ownership, and commits authoritative Attempt outcomes. Only `evaluate` may commit an outcome; raw EvalRequest submit and SOL results are diagnostic. Job listing, polling, and cancellation are Runtime-internal.

## Consequences

`atrex-gateway-client` is a pinned production dependency from its internal package index. SDK upgrades require Adapter contract tests against the new published package. Agate validation rejection during authoritative evaluation becomes a failed candidate outcome; transport errors, unknown statuses, and malformed responses remain infrastructure failures. Non-authoritative failed jobs return structured failed results. Internal Job lookup, polling, and cancellation require a durable `(Attempt, job id)` binding. OSS attachment streaming remains a separate future Artifact capability.
