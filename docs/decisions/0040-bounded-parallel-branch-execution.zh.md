# 决策 0040：有界并发 Branch 执行

## 决策

Evolver 调用保持串行，直到配置的 Challenger Pool 完整。随后 Runtime 并发运行 Active Branch 和
全部 Challenger Branch，并受部署配置 `max_parallel_branches` 限制（正数，默认 `4`）。每个获准
运行的 Branch 内仍并发执行配置的 Trajectory，而每条 Trajectory 的 Attempt 保持串行。

所有 Branch 使用同一个冻结的 Epoch 起始 Kernel 和 Evidence，不能消费兄弟 Branch 的中间结果。
只有全部 Branch 成功结束后，Runtime 才开始 Agent 选择。

Runtime 在 Branch 任务内部捕获异常，避免 Task Group 默认取消兄弟 Branch。兄弟 Branch 可以完成并
持久化 Attempt，之后 Runtime 再确定性地抛出失败。基础设施重试耗尽时，在兄弟任务清理后让 Epoch
失败；意外进程中断仍保留 Running Epoch，供正常续跑。

## 影响

Optimizer Session 最大并发数为
`min(1 + K, max_parallel_branches) × Y`。该上限属于 Runtime 部署策略，不属于不可变 Campaign
拓扑，因此运维可以按模型、Gateway 与 GPU 容量调整，而不改变 Lineage 身份。
