# 决策 0016：在 Registry 中持久化带关联的生命周期 Event

[English](0016-durable-correlated-lifecycle-events.md) | 中文

## 状态

已接受并实现。

## 背景

Session Log 与封存 Trace 可以解释 Agent 行为，但不能为事故响应提供小型、有序的控制平面历史。已有 Registry 状态迁移 Event 没有覆盖 Worker 进程、Gateway 操作、GPU Wiki 选择或显式回滚，Payload 也没有版本和一致的聚合关联。仅使用进程本地日志还会丢失它与持久生命周期变更之间的顺序关系。

## 决策

可信 Runtime 组件通过 Registry 追加 Event。每个 Payload 都带有 `schema_version: 1` 和根据权威关系生成的 `correlation` 对象，其中包含适用的 Campaign、lineage、Epoch、Attempt、Task、Kernel Revision 或 Wiki Feedback 身份。Payload 可以包含受限元数据、状态、Token 数量和 Artifact Digest，但绝不包含 Secret、Prompt、模型响应或原始 Session 内容。

Optimizer 与 Evolver Runner 在其持有的进程被回收后记录 Worker 启动、退出、基础设施失败或超时，以及清理。Evolver 校验记录已封存或拒绝的 Candidate。Gateway Proxy 记录已授权提交、终态结果和失败。Wiki Proxy 记录实时 Query 提交、交互冻结后的完成和失败。Registry 选择事务在已有生命周期 Event 旁记录 Kernel 和 Kernel Agent 晋升或回滚。

Event 是同步的持久事实：必需 Event 写入失败时，关联的可信操作会失败，而不会静默产生不可观测的状态变更。带认证 Sequence Cursor API 继续作为读取接口。

## 影响

Operator 无需读取 Worker 文件或复制模型可见内容，就可以关联主要单节点生命周期。重放外部调用时，即使底层操作是幂等的，也可能产生多条调用 Event；Operation 与 Idempotency 身份可以区分它们。保留策略、服务端过滤、导出格式、直接 Metrics 聚合和目标镜像强杀测试仍属于独立要求。
