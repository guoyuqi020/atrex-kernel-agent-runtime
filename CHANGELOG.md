# Changelog

English | [中文](CHANGELOG.zh.md)

All notable changes to Atrex Kernel Agent Runtime are documented here.

## Unreleased

- Added persistent production control-plane and managed multi-DSL Campaign task scripts with
  per-DSL inspection.
- Made Sandbox host preparation concurrency-safe and Lima-virtiofs compatible by creating Worker
  roots/probes directly as the configured non-root Worker.
- Documented shared-host networking as the explicit Worker network boundary.
- Removed high-frequency Claude `system/thinking_tokens` estimate telemetry from authoritative
  Session sealing and Agent Evidence while retaining final usage records.
- Removed GPU Wiki feedback generation, persistence, delivery, and ingestion; GPU Wiki is now a
  query-only external knowledge service.
- Made new production Campaign preparation reject dirty Core or Evolver worktrees so commit pins
  always identify the exact Agent Bundle source.
- Added pinned upstream GPU kernel projects as a `reference/` tree in every Attempt and
  framework-baseline workspace, bound read-only from `reference_projects_root` in both bubblewrap
  launcher modes.
- Raised the Attempt manifest to schema 8 and stopped publishing the workspace layout in it. The
  layout is fixed at both ends and stated in the Agent Prompt, so serializing it only compared one
  hardcoded table against another while making every layout change a breaking protocol bump. A
  Kernel Agent revision registered against an earlier schema no longer starts, so existing Lineages
  must be re-bootstrapped.

## 0.1.0 - 2026-08-20

- First release candidate of the single-node trusted Runtime.
- Commit-pinned Core/Evolver import, Campaign Bootstrap, Artifact-seeded Lineages, configurable
  Epoch topology, Agent/Kernel version histories, and resumable scheduling.
- Exploratory Gateway operations, authoritative ordinary-Evaluate or same-allocation ABBA gates,
  Production Gate, hidden Evaluation Contracts, Roofline construction, and NCU SOL fallback.
- Live GPU Wiki query with freeze-before-return.
- Claude, Codex, QoderCLI, and Pi Backend bindings with raw Session and provider-token accounting.
- Development launcher plus Linux bubblewrap/cgroup-v2 sandbox with shared host networking.
- Authenticated administration API, CLI inspection, recovery, Events, Tasks, and offline retention.
