# FA4 SM120 题目对齐核对

[English](ALIGNMENT.md) | 中文

SM120 题目复用原 FA4 题目的算子语义，只更改与架构相关的内容：

| 资产 | 与 `data/FA4` 的关系 |
| --- | --- |
| `adapter.py`、`reference.py`、`input.py` | 逐字相同 |
| `shape_train.json`、`shape_valid.json` | 逐字相同，保留全部 30 个 Shape |
| Evaluator Bundle | 逐字相同并固定到同一 Commit |
| Metadata | 保留同一捕获来源；题目 ID 与目标硬件标明 L20N/SM120 |
| Roofline | 语义 FLOPs/Bytes 不变；按 L20N SM120 峰值重算 SOL |
| Source Bundle | 同一 FA4/Quack 上游闭包，加一个已提交的 SM120 Vendor Overlay |
| Initial Evidence | 针对 SM120 重写，不再给出 SM100 双 CTA 优化处方 |

Overlay 位于 `interface.py`，固定 Adapter 保持逐字不变。它将原来不支持的
`SM120 + FP8 + paged KV` 组合变为 correctness-first 路径：先物化 BF16 Dense K/V，再调用
现有 SM120 FA4 Kernel。所有临时分配、转换和 Host 同步都留在被计时的 Candidate 边界内，
因此可以被后续优化，而不是被 Adapter 隐藏。

本题不声称历史 L20D Latency 适用于 L20N；这些记录只作为 Shape 来源保留。L20N SOL 按
`max(FP8 FLOPs / 549e12, bytes / 1.344e12)` 计算。

可选 target smoke 与权威 Evaluator 使用相同的逐元素 `atol=0.06`、`rtol=0.04` 判定；它只
诊断一个 Shape，不能替代 Runtime 验证。

`asset-integrity.json` 记录完整任务与 Bundle 哈希。准备阶段在发布 Workspace 前验证每个文件、
两个 Git Commit、公开/私有 Shape 一致性以及固定的 Optimizer/Evolver。
