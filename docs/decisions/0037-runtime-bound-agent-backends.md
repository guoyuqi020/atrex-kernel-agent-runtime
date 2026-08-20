# Decision 0037: Runtime-bound Agent backends

English | [中文](0037-runtime-bound-agent-backends.zh.md)

## Context

Core already implemented Claude, Codex, QoderCLI, and Pi Adapters, but selected its Backend from the
evolvable repository. Evolver implemented only Claude. This allowed an Agent revision to change the
Provider while it was being compared with another revision, confounding Agent-design promotion with
a Provider/model-policy change. It also made deployment configuration and example prerequisites
unable to state the actual executable and credential contract.

## Decision

Runtime configuration independently binds `optimizer` and `evolver` to one of `claude`, `codex`,
`qodercli`, or `pi`, together with `reasoning_effort` and an opaque Backend-specific
`session_settings` string. Runtime injects the three values as an all-or-nothing reserved environment
triplet. Worker environment allowlists cannot override it.

Core and Evolver keep repository defaults for standalone development, but managed Sessions must use
the Runtime binding. Core applies it to problem generalization, framework baseline, and every
Optimizer Attempt. Evolver implements the same four Adapter and token-accounting contracts without
a token cutoff. Both record the actual Backend, effort, settings digest, raw Provider streams, and
normalized usage in the Session trace. Missing or incomplete provider usage remains fail-closed.

Credentials are never configuration values. They are forwarded only through each Worker's explicit
environment allowlist, and the selected CLI must be available through that Worker's `PATH`.

## Consequences

- Active and Challenger Agent revisions in one Runtime deployment use a comparable Provider policy.
- Optimizer and Evolver may use different Backends.
- Changing the Runtime binding changes execution policy but does not rewrite a Core or Evolver Git
  revision.
- Prompts, tools, workflows, memory policy, and Adapter implementation remain Bundle-owned.
- The single Core entrypoint and Bundle-owned framework implementation remain unchanged.
- Target-provider acceptance still requires real credentials and CLI testing for every selected
  Backend.
