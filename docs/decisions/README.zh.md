# 架构决策记录

[English](README.md) | 中文

这里只保留仍约束当前发布实现的决策。已被取代的设计已从发布树删除；当前行为以代码、
[架构设计](../architecture.zh.md)、[配置说明](../configuration.zh.md)和
[接口说明](../interfaces.zh.md)为准。

## 信任、存储与控制

- [0001](0001-agate-sdk-adapter.zh.md)：Agate SDK 适配边界
- [0003](0003-evidence-handoff.zh.md)：不可变 Evidence 交接
- [0007](0007-immutable-agent-traces.zh.md)：不可变原始 Agent Trace
- [0012](0012-documentation-status-and-release-gates.zh.md)：文档权威与发布门禁
- [0013](0013-branch-local-attempt-evidence.zh.md)：分支本地 Attempt Evidence
- [0014](0014-failed-epoch-recovery.zh.md)：失败 Epoch 恢复
- [0015](0015-durable-campaign-task-control-plane.zh.md)：持久 Campaign Task
- [0016](0016-durable-correlated-lifecycle-events.zh.md)：关联生命周期事件
- [0017](0017-administration-lifecycle-and-cooperative-cancellation.zh.md)：管理生命周期
- [0018](0018-readiness-and-offline-artifact-retention.zh.md)：就绪检查与离线保留
- [0019](0019-isolated-wheel-smoke.zh.md)：隔离 Wheel 发布冒烟
- [0020](0020-failed-evolution-evidence.zh.md)：失败 Evolution Evidence

## Agent、Wiki 与进化

- [0021](0021-live-gpu-wiki-capability.zh.md)：实时 Wiki 查询和 Epoch 后反馈
- [0022](0022-local-wiki-test-double.zh.md)：接口兼容的本地 Wiki
- [0027](0027-unified-epoch-evidence-view.zh.md)：按 Epoch 组织的 Evidence
- [0028](0028-commit-only-runtime-boundary.zh.md)：仅 Commit 的 Runtime/Agent 边界
- [0031](0031-unbounded-evolver-token-accounting.zh.md)：无截止 Evolver Token 统计
- [0035](0035-role-scoped-evolver-evidence.zh.md)：按角色隔离 Evolver Evidence
- [0036](0036-frozen-evolver-runtime-tools.zh.md)：Evolver 仅使用文件系统 Evidence
- [0037](0037-runtime-bound-agent-backends.zh.md)：Runtime 绑定 Backend 策略
- [0039](0039-three-form-challenger-proposals.zh.md)：三种 Challenger 提案
- [0041](0041-lineage-bound-model-selection.zh.md)：Lineage 绑定模型
- [0043](0043-campaign-frozen-evolver-commit.zh.md)：Campaign 冻结 Evolver Commit
- [0045](0045-provider-native-usage-accounting.zh.md)：Provider 原生 Credit 与 Token 记账

## 评测、版本与调度

- [0029](0029-append-only-bootstrap-generations.zh.md)：只追加 Bootstrap Generation
- [0030](0030-exploratory-and-authoritative-evaluations.zh.md)：探索性和权威评测
- [0032](0032-lineage-kernel-version-labels.zh.md)：Kernel `vN` 标签
- [0033](0033-lineage-agent-version-labels.zh.md)：Agent `agent-vN` 标签
- [0034](0034-configurable-epoch-topology.zh.md)：可配置 Epoch 拓扑
- [0040](0040-bounded-parallel-branch-execution.zh.md)：有界并行 Branch 执行
- [0042](0042-artifact-seeded-lineages.zh.md)：Artifact Seed Lineage
- [0046](0046-worker-host-network.zh.md)：Sandbox Worker 共享宿主网络
- [0047](0047-worker-owned-workspace-roots.zh.md)：由 Worker 创建 Sandbox Root
- [0048](0048-outer-container-worker-launcher.zh.md)：外层容器 Worker 边界
