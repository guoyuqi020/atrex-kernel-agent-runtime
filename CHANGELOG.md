# Changelog

English | [中文](CHANGELOG.zh.md)

All notable changes to Atrex Kernel Agent Runtime are documented here.

## Unreleased

- Added persistent production control-plane and managed multi-DSL Campaign task scripts with
  per-DSL inspection.
- Made Sandbox host preparation concurrency-safe and Lima-virtiofs compatible by creating Worker
  roots/probes directly as the configured non-root Worker.
- Documented shared-host networking as the explicit Worker network boundary.
- Removed high-frequency Claude `system/thinking_tokens` estimate telemetry at every authoritative
  Session sealing, Agent Evidence, and Wiki feedback boundary while retaining final usage records.
- Made new production Campaign preparation reject dirty Core or Evolver worktrees so commit pins
  always identify the exact Agent Bundle source.

## 0.1.0 - 2026-08-20

- First release candidate of the single-node trusted Runtime.
- Commit-pinned Core/Evolver import, Campaign Bootstrap, Artifact-seeded Lineages, configurable
  Epoch topology, Agent/Kernel version histories, and resumable scheduling.
- Exploratory Gateway operations, authoritative ordinary-Evaluate or same-allocation ABBA gates,
  Production Gate, hidden Evaluation Contracts, Roofline construction, and NCU SOL fallback.
- Live GPU Wiki query with freeze-before-return and durable post-Epoch feedback.
- Claude, Codex, QoderCLI, and Pi Backend bindings with raw Session and provider-token accounting.
- Development launcher plus Linux bubblewrap/cgroup-v2 sandbox with shared host networking.
- Authenticated administration API, CLI inspection, recovery, Events, Tasks, and offline retention.
