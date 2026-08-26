# 决策 0039：三种 Challenger 提案形态

## 决策

每次 Evolver 调用准确输出一个统一的 `EvolutionOutput` 提案。所有模式都使用
`kernel_agent_revision_id` 与 `changed_paths`；后者只包含相对于所选 Agent Source 根目录的排序路径。
`reuse` 要求空数组；如果只修改 Runtime State，新 Revision 模式也可以使用空数组：

- `evolved` 创建新 Agent Revision，其 Parent 是本 Epoch 的 Active Revision；
- `reuse` 原样让一个可见历史 Revision 参赛，不创建 Revision，也不增加版本号；
- `evolve_from_history` 创建新 Agent Revision，其 Parent 是一个可见历史 Revision。

报告的 Revision 只标识 Agent Source；Runtime State 身份与 Checkpoint 仍是 Runtime 私有控制数据。
Runtime 根据本次调用冻结的可见集合和同 DSL Lineage 校验所有 Source 引用。创建新 Revision 时，Runtime
独立比较 Candidate 的 `source/`、`runtime-state/` 和所选 Source Base/初始 Active State Checkpoint，
校验报告的 Source-only 修改集合，并要求 Source 或 State 至少一项真实变化。Runtime State 路径不进入
Agent 报告；复用时 Source 与公共种子都必须保持不变。对于 `evolve_from_history`，Evolver
用所选 `input/historical/agent-vN/` 的 Source 替换 Candidate Source，并可从可见历史状态整理公共种子。
Runtime 校验 Base 并跨 Source 与状态种子比较最终 Diff。

每个提案都可以携带有界的 `unimplemented_capabilities`。每一项说明一种 Agent 能力、预期的
Kernel 优化收益，以及 Evolver 无法实现它的原因。Runtime 将这些内容作为不可信 Evolution
Evidence 保存，供后续 Evolver 查看；它们不会授予能力，也不影响胜负选择。

提案类型、Source Reference Revision、实际参赛 Revision 和 Evolution Trace Digest 属于 Epoch Challenger
参赛记录。Revision 祖先关系仍是单 Parent 树；Epoch 竞争、复用与晋升是另一条时间线，不增加
祖先边。

## 影响

历史设计可以重新参赛，而无需制造无意义副本或膨胀版本号；也可以从有潜力的旧设计派生新
Revision，而无需假装它来自当前 Active。只要最终 Bundle Contract 有效，Evolver 可以重新设计
任意 Optimizer Candidate 内容。Evidence 与管理视图同时展示祖先关系和参赛来源。
