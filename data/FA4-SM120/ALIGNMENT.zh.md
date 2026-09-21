# FA4 SM120 题目对齐核对

[English](ALIGNMENT.md) | 中文

SM120 题目复用原始 Workload，但替换 Candidate 边界：

| 资产 | 与 `data/FA4` 的关系 |
| --- | --- |
| `reference.py`、`input.py` | 逐字相同 |
| `shape_train.json`、`shape_valid.json` | 逐字相同，保留全部 30 个 Shape |
| Evaluator Bundle | 逐字相同并固定到同一 Commit |
| Metadata | 保留同一捕获来源；题目 ID 与目标标明 L20N/SM120 |
| Roofline | 语义 FLOPs/Bytes 不变；按 L20N/SM120 峰值重算 SOL |
| 固定 Adapter | 保持同一生产 ABI，但只调度新的 SM120 实现 |
| Source Bundle | 只读 SM103-family Reference，加一个空的可编辑 SM120 实现 |

题目不再提供 Vendor 树或 SM120 Fallback。只读 Reference 保留原题固定的 SM103-family 路径，
并剔除 SM120 源码与混合架构调度入口，供 Agent 分析其设计；Runtime 只允许写入
`implementation/`。因此 Candidate 必须真正实现 SM120，
而不是调优或修补一个已提供的 SM120 Bridge。

历史 L20D 测量只作为 Shape 来源，不代表 L20N 性能。L20N SOL 为
`max(FP8 FLOPs / 549e12, bytes / 1.344e12)`。`asset-integrity.json` 记录全部题目与 Bundle
哈希，准备阶段会在发布 Workspace 前验证 Source 和 Evaluator Commit。
