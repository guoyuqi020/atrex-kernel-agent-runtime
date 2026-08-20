# ADR 0035：按 Role 区分已完成 Epoch Evidence

[中文](0035-role-scoped-evolver-evidence.zh.md) | [English](0035-role-scoped-evolver-evidence.md)

## 状态

已接受并实现。

## 背景

Optimizer 需要分支隔离，避免全新 Attempt 复制竞争搜索路径。Evolver 的任务不同：它需要在
观察 Active 与 Challenger 设计是否真正产生更优 Kernel 后改进 Agent 设计。只展示晋升分支会
隐藏负面结果、落败设计、已评测 Kernel 源码，也无法区分有用 Agent 修改与偶然的快 Kernel。

## 决策

`EvidenceViewManifestV1.visibility.completed_epochs` 按 Role 区分：

- Optimizer 使用 `promoted_lineage`；已完成 Epoch 继续移除 Branch 控制身份，当前 Epoch
  只暴露同一 Trajectory 中较早的 Attempt。
- Evolver 使用 `all_completed_branches`；它不接收进行中 Epoch，但可以看到每个已完成
  Active 与 Challenger 分支。

每个 Evolver Epoch 保留 Active、Challenger、胜出 Agent、起始 Kernel 和最佳 Kernel 身份。
`branches/` 包含每个 Attempt Summary、Report、Diff 和已保留 Session Artifact 的未脱敏派生副本；
权威 Session 保留策略仅省略高频 Claude `system/thinking_tokens` 估算遥测，派生副本会对旧
Artifact 应用同一规则；
`evolution/` 包含所有可用 Challenger Evolver Session；`kernels/index.json` 记录 Role 与权威
Outcome，每个被引用的精确 Kernel Artifact 只在 `kernels/<kernel-revision-id>/` 下物化一次。

累计来源 Evidence 除 ID 外，还保存结构化起始/最佳 Kernel 事实。只要 Attempt Output 带
Artifact Digest，对应 Kernel 就会被投影。

## 后果

- Evolver 可以用精确 Kernel/评测历史比较成功与失败 Agent 设计。
- Optimizer 分支隔离和全新 Session 语义不变。
- Evolver Evidence 更大，相邻 Epoch View 可能重复物化同一个保留 Kernel；精确内容寻址身份仍明确。
- Evolver Bundle 在使用投影历史前校验 `all_completed_branches`。
