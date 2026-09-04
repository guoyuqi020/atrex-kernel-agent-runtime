# 决策 0039：三种 Challenger 提案形态

## 决策

每次 Evolver 调用准确输出一个统一的 `EvolutionOutput` 提案。所有模式都使用
`kernel_agent_revision_id`、`changed_paths` 与 `contributing_paths`；`changed_paths` 只包含相对于
所选 Agent Bundle 根目录的排序路径，包括六个自适应目录的改动。`reuse` 要求空数组，新版本必须有真实改动：

- `evolved` 创建新 Agent Revision，其 Parent 是本 Epoch 的 Active Revision；
- `reuse` 原样让一个可见历史 Revision 参赛，不创建 Revision，也不增加版本号；
- `evolve_from_history` 创建新 Agent Revision，其 Parent 是一个可见历史 Revision。

报告的 Revision 选择可见 Bundle 和唯一 Parent，具体 State 身份仍由 Runtime 管理。
`evolve_from_history` 从 `input/agents/agent-vN/` 复制完整 Bundle 到 `candidate/`，然后修改。
Runtime 校验同 DSL、可见范围、提案资格及整个 Bundle 的准确 Diff。六个自适应目录的修改也计入
`changed_paths`；复用时 Candidate 必须不变。

每个提案都可以携带有界的 `unimplemented_capabilities`。每一项说明一种 Agent 能力、预期的
Kernel 优化收益，以及 Evolver 无法实现它的原因。Runtime 将这些内容作为不可信 Evolution
Evidence 保存，供后续 Evolver 查看；它们不会授予能力，也不影响胜负选择。

`contributing_paths` 记录实际吸收内容的、排序且去重的 Workspace 相对文件或目录路径，允许
`input/agents/agent-vN/` 和 `input/evidence/agent-vN/resources/`，包括 Parent 其他 Trajectory 的资源。
仅阅读和自动继承 Parent 不算贡献。路径必须存在、无链接或越界，且属于合格已评估历史或 Parent，
不能引用同 Epoch 尚未评估的 Challenger。`reuse` 要求 `[]`。Runtime 在 Evolution Trace 中保存归属
和准确内容快照；该字段不改变 Bundle Base 或 Revision 祖先关系。

提案类型、Source Reference Revision、实际参赛 Revision 和 Evolution Trace Digest 属于 Epoch Challenger
参赛记录。Revision 祖先关系仍是单 Parent 树；Epoch 竞争、复用与晋升是另一条时间线，不增加
祖先边。

## 影响

历史设计可以重新参赛，而无需制造无意义副本或膨胀版本号；也可以从有潜力的旧设计派生新
Revision，而无需假装它来自当前 Active。只要最终 Bundle Contract 有效，Evolver 可以重新设计
任意 Optimizer Candidate 内容。Evidence 与管理视图同时展示祖先关系和参赛来源。
