# 决策 0014：按 Generation 恢复 Failed Epoch

[English](0014-failed-epoch-recovery.md) | 中文

## 状态

已接受并实现。

## 背景

宿主丢失、Gateway Capability 过期或 Sandbox Backend 不可用可能让 Infrastructure Failed Attempt 耗尽自动重试额度。如果永久终止 Failed Epoch，Operator 只能创建新 Campaign，也会丢失 Attempt 身份、模型可见 Evidence、Gateway 操作和审计历史之间的稳定关系。只把状态改回 Running 同样不安全：旧 Scheduler Fence 或 Bearer Capability 可能仍然存在，重放旧 Gateway Reservation 还会混合两次 Operator 授权的执行。

## 决策

Registry 持久化以 `(epoch_id, recovery_key)` 为键的幂等 Recovery Record。可信 Operator 通过 `recover-epoch` 提交该 Key 和非空理由。一个事务要求 Epoch 已 Failed 且至少存在一个 Infrastructure Failed Attempt，重新打开相同 Epoch 与 Attempt 身份，清零已耗尽的基础设施计数，递增 Recovery Generation，记录新的 Authority 起始时间，取代 lineage Fence，重新打开 lineage 和符合条件的 Campaign，并生成可关联恢复事件。完全相同的重放返回原记录；同一 Key 携带不同理由会失败。

Gateway Control 保存 Attempt Recovery Generation。每次操作授权与 Outcome 提交都会将其和 Registry 对比，因此推进 Registry Generation 会立即拒绝旧 Bearer 与任何在途 Authorization。下一次签发会派生新 Token 与过期时间，并重置额度与撤销状态。按 Generation 隔离的幂等 Reservation 会作为审计历史保留，但不能授权新 Generation。原 Epoch Evidence 和分支本地 Attempt Evidence 保持不变。已有权威 Gateway Outcome 会在 Worker 启动前恢复，绝不会被 Capability 轮换替换。

Registry Schema 8 与 Gateway Control Schema 2 会有意拒绝此前的 Pre-release 文件。

## 影响

Operator 可以恢复耗尽额度的基础设施故障，无需创建替代 Attempt，也不会污染竞争计量。恢复是显式、可审计、幂等的，并且能够抵御陈旧 Scheduler 与凭据。该命令不恢复 Agent 输出、策略或控制平面缺陷；这些故障必须使用各自流程。生产验收前仍需在目标镜像上对每个生命周期边缘执行崩溃测试。
