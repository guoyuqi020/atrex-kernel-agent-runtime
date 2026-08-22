# Release Checklist

English | [中文](release-checklist.zh.md)

## Source and packaging

- [ ] Release version in `pyproject.toml` and tag agree; working tree contains no generated state.
- [ ] Core, Evolver, and Atrex Bench submodule commits are pushed, reachable, and recorded.
- [ ] `python -m build` succeeds from a clean checkout.
- [ ] `scripts/smoke-wheel-independence.py` proves the sdist/wheel exclude adjacent repositories and
  the installed wheel does not import adjacent source trees.
- [ ] Wheel/sdist contents contain Runtime code, documentation/license metadata, and no secrets,
  databases, workspaces, Session traces, or local credentials.

## Quality gates

- [ ] Runtime Ruff, strict mypy, and full pytest suite pass.
- [ ] Core and Evolver Ruff, strict mypy, and pytest suites pass at the pinned commits.
- [ ] Local Wiki suite passes against the same query wire contract.
- [ ] Every JSON example parses and every Markdown relative link resolves.
- [ ] CLI `--help`, Bootstrap, one no-Challenger Epoch, and one Evolution Epoch pass from the built
  wheel in a clean environment.

## Target-environment acceptance

- [ ] Exact production Linux image passes `scripts/validate-linux-sandbox.py` with filesystem,
  namespace, cgroup, shared-host-network, DNS, and cleanup checks.
- [ ] All four Worker roots and concurrent host probes are created as `worker_user`; the exact
  deployment filesystem (including virtiofs when used) passes Bootstrap without ownership repair.
- [ ] Selected Claude/Codex/QoderCLI/Pi Backend completes a real Session and records raw Trace plus
  provider token usage.
- [ ] Every enabled Gateway operation works through Runtime against production Agate and target GPU.
- [ ] Ordinary Evaluate/ABBA repeatability, clock-lock behavior, correctness tolerances, Production
  Gate, and optional Roofline/NCU SOL fallback are validated on representative operators.
- [ ] Production GPU Wiki query is rehearsed.

## Operations

- [ ] Capability signing key and admin token are in a secret manager and survive planned restarts.
- [ ] Storage ownership, disk budget, backup, restore, Event export/prune, Wiki retry, and offline GC
  are rehearsed.
- [ ] `services.sh`, managed `campaign.sh`, and per-DSL `inspect.sh` are rehearsed with separate
  service/task workspaces.
- [ ] Forced termination is tested during Bootstrap, Attempt evaluation, Epoch commit, Evolution,
  Wiki delivery, and Task leasing; restart produces no duplicate authority or lost durable outcome.
- [ ] Metrics/log collection, external alerting, incident owner, rollback commit, and release notes
  are ready.

Do not call the release production-ready while any applicable target-environment item is open. See
[Implementation Status](implementation-status.md) for known remaining evidence.
