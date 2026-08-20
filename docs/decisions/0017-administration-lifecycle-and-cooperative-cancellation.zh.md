# 决策 0017：补齐单节点管理生命周期

[English](0017-administration-lifecycle-and-cooperative-cancellation.md) | 中文

## 状态

已接受并实现。本文扩展决策 0015。

## 背景

首版持久 Task API 可以入队和检查工作，但 Bootstrap 与 Failed Epoch 恢复仍要求 CLI，不能把已检查的失败重新入队，也只能取消 Queued Task。Event Cursor 没有过滤、导出、保留操作或受限聚合视图。由 HTTP Handler 直接杀死 Active Task Worker 并不安全，因为同步 Subprocess Thread 必须保持所有权，直到完整进程树被回收。

## 决策

带认证管理平面提供：使用可信主机绝对路径的幂等 Bootstrap、Campaign 聚合状态与静止取消、Failed Epoch 恢复、Failed Task 重入队、带关联的 Event 过滤、受限 NDJSON 导出、受限且确认过的 Prefix 清理，以及当前 Event/Task 计数。所有 Mutation 都复用已有 Registry 与 Bootstrap Service。ASGI 进程仍然绝不启动 Agent。

取消 Running Task 时，状态会变为 `cancelling`，同时保留 Worker Lease。Task Worker 在 Heartbeat 中观察请求并取消 Scheduler Scope。同步 Worker Subprocess 会继续受到 Cancellation Shield 保护，直到其有界进程 Owner 已终止并回收全部后代；之后不会再启动新的 Attempt 或 Epoch 工作。Completion 或 Failure 随后原子记录 `cancelled`。如果 Task Worker 丢失，另一 Worker 会在 Lease 过期后直接完成取消，不会重新启动 Campaign。Queued 取消仍然立即且幂等。

Event Filter 接受准确 Kind 与权威 Correlation ID。Export 使用受限 NDJSON。Prune 只按配置批大小删除调用方已确认的 Sequence Prefix，并在被删除区间之后追加新的审计 Event。Registry Schema 10 会有意拒绝此前的 Pre-release 文件。

## 影响

上游 Controller 无需直接访问 SQLite，即可管理当前实现的完整单节点生命周期。Running Cancellation 是协作式的，其时延受当前 Worker Operation 的上限约束；它不是不安全的立即杀进程。Operator 必须在 Prune 前保存已导出 Event Sequence 检查点。决策 0018 后续补充了本地依赖就绪；长期遥测存储、分布式协调和目标镜像中断/崩溃验收仍属于独立要求。
