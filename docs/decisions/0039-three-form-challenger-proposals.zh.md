# 决策 0039：三种 Challenger 提案形态

## 决策

每次 Evolver 调用准确输出一个带判别字段的 `EvolutionOutputV3` 提案：

- `evolved` 创建新 Agent Revision，其 Parent 是本 Epoch 的 Active Revision；
- `reuse` 原样让一个可见历史 Revision 参赛，不创建 Revision，也不增加版本号；
- `evolve_from_history` 创建新 Agent Revision，其 Parent 是一个可见历史 Revision。

Runtime 根据本次调用冻结的可见集合和同 DSL Lineage 校验所有引用。创建新 Revision 时，Runtime
独立比较完整 Candidate 仓库和所选 Base，并要求申报的修改文件集合完全准确；复用时，初始化的
Candidate 必须保持不变。`evolve_from_history` 还必须使用受约束的 Runtime
`candidate-reset` 操作，并产生匹配的 Candidate Base 记录。

每个提案都可以携带有界的 `unimplemented_capabilities`。每一项说明一种 Agent 能力、预期的
Kernel 优化收益，以及 Evolver 无法实现它的原因。Runtime 将这些内容作为不可信 Evolution
Evidence 保存，供后续 Evolver 查看；它们不会授予能力，也不影响胜负选择。

提案类型、Base Revision、实际参赛 Revision 和 Evolution Trace Digest 属于 Epoch Challenger
参赛记录。Revision 祖先关系仍是单 Parent 树；Epoch 竞争、复用与晋升是另一条时间线，不增加
祖先边。

## 影响

历史设计可以重新参赛，而无需制造无意义副本或膨胀版本号；也可以从有潜力的旧设计派生新
Revision，而无需假装它来自当前 Active。只要最终 Bundle Contract 有效，Evolver 可以重新设计
任意 Optimizer Candidate 内容。Evidence 与管理视图同时展示祖先关系和参赛来源。
