# 决策 0028：只接受 Commit 的 Agent Bundle 与最小 Runtime Launcher

[English](0028-commit-only-runtime-boundary.md) | 中文

## 状态

2026-08-18 接受。本决策是当前 Runtime/Agent 源码边界。

## 决策

Runtime 只接受 Campaign schema v3 和完整 Core Git Commit。Campaign 定义的 `lineages` Key 是完整 DSL 集合，部署配置不再重复该拓扑。Core/Evolver 都是独立版本化的完整仓库，通过同一个安全 Git 边界导入并封存为不可变 Artifact。Core 与 Evolver 持有各自 Agent 框架、Prompt 和 Backend Adapter；Runtime 部署配置持有 Backend，Campaign/Lineage 状态持有具体 Model 选择。Runtime 只记录 Evolver Commit、Tree、Artifact 和 Launch Fingerprint，并发送固定 stdin 指令。

Runtime 提供统一 Launcher Contract，并显式区分 `development` 与 Linux `sandbox`。生产模式把
bubblewrap 和 cgroup v2 应用到完整 Core/Evolver 进程树；Runtime 绝不会把无隔离
Development 模式作为降级路径。

Evidence 只持久化规范化 Projection 和 Session 来源 Digest，不复制原始字节。Agent View 按 Digest 物化原始、未脱敏 Session Artifact。Wiki Feedback 在 Epoch 结束后独立构造准确、有界的原始 Projection。

## 结果

配置与 Provenance 只有一个事实来源，废弃兼容路径被删除。完整 Worker 边界由当前沙箱与宿主网络决策定义；任何框架专用的历史设计都不能作为当前行为证据。
