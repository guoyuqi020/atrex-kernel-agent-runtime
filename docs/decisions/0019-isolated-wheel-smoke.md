# Decision 0019: Gate reference-checkout independence with an isolated wheel smoke

English | [中文](0019-isolated-wheel-smoke.zh.md)

## Status

Accepted and implemented.

## Context

The Runtime must consume the published Agate SDK without importing its source tree, and the wheel must not import adjacent Core, Evolver, or Atrex Bench checkouts as Python packages. A successful editable-source test or wheel build alone does not prove this isolation.

## Decision

`npm run smoke:wheel` builds the wheel with the local build backend, installs it without dependencies or an index into a fresh temporary target, removes repository root/source entries from the child interpreter search path, imports the complete Runtime, and resolves every packaged CUDA, Triton, and CuteDSL Base Revision. It rejects any ATREX module or distribution metadata loaded outside that target and any declared dependency containing `deepseek-harness` or `atrex-kernel-agent`.

## Consequences

Repository verification now includes an automated negative source-resolution gate. The smoke reuses already installed third-party dependencies, so production acceptance still requires installation and execution on a genuinely clean deployment host with pinned dependency artifacts.
