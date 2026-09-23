# 决策 0040：有界并发 Attempt 执行

## 决策

可执行的 Agent Workflow 代码决定有哪些 Branch 和 Trajectory、一个逻辑轮次包含哪些 Attempt，
以及下一轮使用哪个 Kernel 和 Agent State。Runtime 不再根据 Campaign 参数推断或重建这些拓扑。

Workflow 通过一次 `run_attempts_parallel` 提交并行批次时，Runtime 同时最多准入
`max_parallel_attempts` 个 Optimizer Session（正数，默认 `4`）；其余启动项在同一个可信操作内
等待。该上限只控制部署压力，不改变已冻结的 Workflow 计划、Trajectory 内顺序约束或 Lineage
身份。

Runtime 会分别捕获已准入 Attempt 的失败，避免 Task Group 取消兄弟任务并丢弃结果。兄弟 Attempt
可以先完成并持久化，Runtime 再确定性传播失败；基础设施恢复仍由 Runtime 负责。

## 影响

Workflow 代码可以表达串行搜索、并行 Pool、Active/Challenger 竞争、Kernel 广播和显式 Agent State
路由，而无需在 Runtime 配置中新增拓扑开关。运维仍可通过 `max_parallel_attempts` 独立限制
Provider、Gateway 和 GPU 压力。
