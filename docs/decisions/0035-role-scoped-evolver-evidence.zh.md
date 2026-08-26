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

持久 Evidence Store 保留每个已完成分支的 Active、Challenger、胜出 Agent、Attempt Outcome、
Session Artifact 和精确 Kernel 身份。Evolution Workspace 只做更小的投影：对当前参赛的每个
Agent，Runtime 物化一份权威 Optimization Summary，并保留该 Agent 最近已完成 Epoch 中每个 Attempt
的一份未脱敏 Conversation。已完成且非当前的 Agent 版本也会在源码旁携带同样的精简汇总。
更早的详细 Epoch Tree 仅保留在 Runtime Registry 和 Artifact Store 中。

## 后果

- Evolver 可以根据精简的权威 Outcome 和最近相关 Conversation 比较成功与失败 Agent 设计。
- Optimizer 分支隔离和全新 Session 语义不变。
- Runtime 保留完整审计历史，但不把它重复复制到每个 Evolution Workspace。
- 冻结文件系统输入对当前竞争仍保持分支完整，同时尺寸受控。
