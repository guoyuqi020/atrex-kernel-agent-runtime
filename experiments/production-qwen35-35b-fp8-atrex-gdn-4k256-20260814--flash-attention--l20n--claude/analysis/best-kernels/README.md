# Best Kernel artifacts

These files are extracted from `/Users/guoyuqi/atrex-runs/workspace-full-20260909.tar.zst` by `../generate.py`.

| DSL | Arm | Kernel Artifact | Result Artifact | Latency |
|---|---|---|---|---:|
| CUDA | `ablation-pool-3` | `sha256:42e5c7e0a8b2252a963e5cda8459cb053541d326e69fe414afad1f14c0153f9a` | `sha256:29ee9a1504a3ddfa9492a96c78c4e549bf3bf7dba35708f8efa9ec86941c064b` | 222.558 µs |
| Triton | `ablation-retained-02` | `sha256:6ee169890663d57809912d099196d96e28691f66cc34ada0ed413b907653b07a` | `sha256:b6aca26dc318a91143b9e0d5bbabbb345a3310229da71ae156c4678fa507e13f` | 247.344 µs |
| CuteDSL | `ablation-pool-retained-3` | `sha256:18c8aecf0959e1112b8732ff6d2fddb50977fdc6347a8f6ce920888903590f3d` | `sha256:15bdff674bd77c2e2beddbb5b1a7b3883a0cf611a5eff5a5b5db55c4f610995f` | 232.952 µs |

Each DSL directory contains the exact `kernel.py`, the Agent-authored terminal `attempt-report.json`, and a normalized `gateway-result.json`. The normalized result removes raw Agate job logs and submitted payloads but preserves both comparison arms, per-shape measurements, the schedule, and the source Result Artifact Digest.
