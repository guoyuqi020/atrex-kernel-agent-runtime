# 决策 0015：通过持久任务执行 Campaign 请求

[English](0015-durable-campaign-task-control-plane.md) | 中文

## 状态

已接受并实现；管理生命周期与协作式 Cancellation 由[决策 0017](0017-administration-lifecycle-and-cooperative-cancellation.zh.md)扩展。

## 背景

一个 Campaign 可能运行数小时并启动多个隔离 Agent 进程。如果在 HTTP 请求内直接执行这些工作，Scheduler 所有权会与 ASGI 连接耦合，进程丢失后的状态也会变得模糊，并且没有持久的幂等或接管位置。Operator 和上游系统还需要一种带认证的方式来提交工作、消费有序控制平面 Event，而不应直接访问 Registry。

## 决策

可选管理平面提供带版本和 Bearer 认证的 API。`POST /v1/admin/tasks` 把 Campaign 的绝对目标记录到 Registry 后立即返回。调用方提供的 Creation Key 只有在 Campaign ID、目标 Epoch 和 Finalization Flag 完全相同时才是幂等的。状态查询、Queued Task 取消和基于 Sequence Cursor 的 Event 读取使用同一 API。Bearer 值由环境持有，必须至少包含 32 字节，并使用常量时间比较。

独立的 `run-task-worker` 进程通过可续租 Lease 领取最早的可执行 Task，调用已有 Scheduler，并记录终态结果。过期的 Running Task 可以被接管，因此 Task 执行语义为至少一次。安全性不只依赖 Task 所有权：Scheduler 写入仍然需要 lineage Lease 与 Fence，Gateway 操作仍然需要 Attempt Capability 与 Recovery Generation。取消操作只能迁移 Queued Task，绝不声称可以中断正在运行的 Agent 进程树。

Registry Schema 9 会有意拒绝此前的 Pre-release 文件。

## 影响

HTTP 请求生命周期与 Campaign 执行被分离；重复提交得到稳定结果；丢失的 Task Worker 可以被替换；上游观察者可以保存有序 Event Sequence 的检查点。被接管的 Task 可能重复调用 Scheduler，因此 Handler 必须继续保留已有的幂等绝对目标语义。直接 Campaign Bootstrap 与聚合状态 API、Running Task 中断、Failed Task 重新入队、过滤 Event 导出、Payload Version 演进和多节点协调仍属于后续工作。
