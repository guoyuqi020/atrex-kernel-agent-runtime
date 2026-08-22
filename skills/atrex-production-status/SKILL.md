---
name: atrex-production-status
description: Inspect live ATREX Kernel Agent Runtime production campaigns in Lima and report service health, per-DSL Bootstrap/Optimizer/Evolver progress, kernel performance, session activity, token or credit usage, and recent errors. Use for requests such as “现在跑得怎么样”, “查看运行状态”, or “按 DSL 查看 token 消耗”.
---

# ATREX Production Status

Run the bundled read-only inspector instead of reconstructing Registry queries manually:

```bash
python3 scripts/status.py
```

The command runs from the host and enters the `ubuntu` Lima instance. Defaults can be overridden when needed:

```bash
python3 scripts/status.py \
  --instance ubuntu \
  --repo /home/guoyuqi.guest/atrex-runtime \
  --service-workspace workspaces/production/control-l20n
```

The script returns structured JSON. Summarize it in Chinese unless the user uses another language. Lead with whether the system is healthy and whether work is progressing, then report:

- the number of registered and still-bootstrapping routes;
- each operator/backend/DSL route's current phase, Epoch/Attempt, and best latency;
- accepted performance improvements and pivots/failures;
- active Evolver sessions and whether their traces are fresh;
- recent Runtime errors and genuinely stale sessions;
- usage by operator/backend/DSL/stage when requested.

Treat `settled` usage as authoritative. Label `running_partial` as provisional: Claude may reclassify cache buckets at Session completion, while Qoder reports credits rather than tokens. Do not add tokens and credits together.

Use `observed_best_latency_us` for in-progress branch progress and `registered_best_latency_us` for the Registry's promoted best. If they differ, explain that an accepted attempt has not yet been promoted at the Epoch boundary.

This workflow is read-only. Do not restart services, kill workers, retry jobs, alter SQLite state, or expose capability tokens and environment secrets unless the user separately requests an authorized action.
