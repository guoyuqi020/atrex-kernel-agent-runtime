# 决策 0003：把 Evidence 发布建模为持久 lineage 交接

[English](0003-evidence-handoff.md) | 中文

## 状态

已接受。

## 背景

Epoch 选择会原子晋升获胜 Kernel Agent 和最佳 Kernel，但下一 epoch 还需要新的公共 Evidence checkpoint。构建该 Artifact 需要在 SQLite 事务之外复制和哈希不可变数据。如果已完成 epoch 立即让 lineage 回到 `ready`，崩溃后可能使用过期 Evidence 启动下一 epoch，也可能无法判断 Checkpoint 组装是否已经完成。

“再运行 N 个 epoch”也存在相关重试问题：崩溃后重复同一请求，即使原目标已经完成，也可能额外执行一个 epoch。

## 决策

每条 lineage 持久拥有当前 Evidence checkpoint 和每分支 Attempt 预算。完成 epoch 时，在同一事务中更新 Agent 与 Kernel 晋升、递增 `next_epoch_number`，并把 lineage 转换为 `awaiting_evidence`；该状态不能启动新 epoch。

`LocalEvidenceAssembler` 确定性复制上一个扁平累计 Bundle，追加已完成 epoch 的权威 Attempt、Kernel、Evaluation 和 Selection 事实，然后封存新 Artifact。Compare-and-Swap 操作随后替换预期 Checkpoint digest，并把 lineage 转换为 `ready`。在该操作前重复组装会得到相同 digest；在操作后重启只会看到 `ready`，不会重复工作。

`CampaignScheduler` 接受绝对目标 epoch，而不是相对数量。重复同一调度请求会恢复未完成状态，并在所有指定 lineage 都越过目标后停止。同一 Campaign 的不同 DSL lineage 可以并发执行，Registry 状态负责串行化每条 lineage。

## 影响

Epoch 晋升和 Evidence 发布成为带显式中间状态的两个可恢复提交，下一 Epoch 不会看到过期历史。可续租 Registry Fencing 会串行化共享当前 SQLite 部署的多个 Scheduler 进程。
