# 实现状态

[English](implementation-status.md) | 中文

历史决策或目标设计与当前代码不一致时，以本文为准。

| 区域 | 状态 | 当前实现 | 剩余工作 |
| --- | --- | --- | --- |
| Campaign/Epoch 控制 | 已实现 | SQLite 持久状态、幂等 Transition、可续租 Generation Fencing、串行构造 Challenger 后有界并发运行 Active/Challenger Branch、每 Branch 并发 Trajectory、每 Trajectory 串行 Attempt、兄弟分支失败隔离、持久完成后的 CLI 进度、恢复、取消/完成 | 多节点 Registry 延后 |
| Campaign Bootstrap | 已实现 | 独立 Campaign schema v3 定义，DSL 拓扑与 Model 选择由 `lineages` 持有，支持可选 Campaign 问题泛化 Model、Campaign 冻结完整 Evolver Commit 并在恢复时拒绝漂移、Commit 固定可信 Atrex Bench Roofline 构建及准确校验与封存恢复、完整 Core Commit、共享 Agent Problem、每 DSL 强制 Core Baseline、带 Session/Token/Report/Outcome 审计的 Append-only 重试 Generation | 真实 Provider/Gateway 验收与代表性生产 Roofline Cost Model 覆盖 |
| Artifact Seed Lineage | 已实现 | 管理 API/CLI 接受已封存 Agent+Kernel Artifact 或注册 Revision ID，校验同 DSL Bundle 内容，执行目标 Campaign 权威 Agate 评测，创建独立 `agent-v0`/`v0`，保留来源 Provenance 并支持幂等恢复 | 真实跨 Campaign GPU 验收与 Crash Window 演练 |
| Core Optimizer | 已实现 | 唯一 Core 入口；Problem、Baseline、Attempt 共用 Runtime 绑定的 Claude/Codex/QoderCLI/Pi 选择；运行中 Session 非封存投影，结束后重建并封存终态 Trace；Lima 已验证 Claude/Codex/QoderCLI 真实 Sandbox 连通性 | 各 Backend 完整流程验收与 Pi 连通性 |
| Evolver | 已实现 | 独立 Runtime 绑定的四 Backend 选择、延迟解析凭据/Bundle、每次调用一个带判别字段的 `evolved`/`reuse`/`evolve_from_history` Challenger 提案、冻结 Agent/Kernel 检索及历史 Candidate 原子 Reset、Commit/Tree/Artifact 身份、Candidate Base/Diff、同 DSL 与冻结历史校验、逐 Epoch 提案来源、失败证据 | 递归 Evolver 自进化延后 |
| Kernel/Agent 选择 | 已实现 | 普通 Evaluate 或 Commit-pinned 同 Allocation ABBA；逐轮持久证据 | 真实 GPU 重复性研究 |
| Production Gate | 已实现 | Campaign 冻结的可选内容门禁，在 Agate 与权威发布前执行；检查固定 DSL/自包含标记、Python AST Fallback、动态/预构建依赖及 `solution.json` 语言与依赖 | 为来源模糊的第三方实现接入受控独立 Reviewer |
| Evidence | 已实现 | 分支本地 Optimizer Evidence、晋升 Lineage Optimizer 时间线、包含 Agent 胜者/精确 Kernel Artifact/Outcome/原始 Session 的 Evolver 全完成分支时间线 | 长 Campaign 保留容量评估 |
| Gateway/Agate | SDK Adapter 已实现 | Capability、封存私有 Evaluation Contract、严格公开 Agent Problem 校验、单 Attempt 多条不可变 Agent 评测、准确 Kernel/原始 Result 保留与独立不透明 ID Worker 投影、单私有 Case Profile、权威完整集合留存 Comparator Outcome、Attempt Job 归属 | 真实 Agate/GPU 全操作验收与 Crash Window 演练 |
| GPU Wiki | 部分实现 | 实时冻结查询、Epoch 后 Outbox、有界原始 Session 上传、本地 Wire 兼容替身 | 生产 Wiki 合约/保留演练 |
| Token 记账 | 已实现 | Optimizer 使用正数单 Session 配额；Evolver 不限 Token，但 Provider Usage Report 必须 Fail-closed | 真实 Provider 记账验收 |
| 管理/运维 | 本地已实现 | 认证 Campaign/Task/恢复/Event/Lineage Seed API；覆盖 Optimizer、Baseline、Generalization、Evolver 且从进程启动前持续到终态的统一 Worker Session Catalog，保留原始 Trace/Workspace/Token/Error 身份；持久 Kernel/Evaluation/Bootstrap Run Catalog；有界准确 Source/Result 导出；Readiness；不启动 Agent 的真实 Attempt 调试 Shell；SQLite 备份路径；Artifact/Workspace GC | TLS/前置代理、外部告警、Crash 演练 |
| 打包/来源隔离 | 本地已实现 | Runtime Wheel 排除 Core/Evolver；精确 Git 导入和封存 Provenance | 干净主机发布演练 |
| 完整 Worker 隔离 | 文件系统/进程/资源隔离已实现；等待准确目标环境验收 | 两种模式都从 Worker 协议排除私有评测路径/数据；生产 bwrap 额外提供只读根、私有 `~/workspace`、Runtime Storage 与相邻 Root 遮蔽、Capability/Namespace 隔离；system-manager transient service 以非 root Worker 创建/探测 Root、执行 bwrap，并应用 cgroup v2 内存/Swap/CPU/PID 限制；跨进程 Host Check 锁和 Worker 原生创建覆盖 Lima virtiofs；Worker 有意共享宿主 DNS/路由/服务和公网出站；可重复 Lima 边界测试及 Claude/Codex/QoderCLI 真实请求通过 | 准确生产 Linux 逃逸/资源/超时/Soak 证据 |

## 自动化基线

仓库维护 Runtime、独立版本化 Core/Evolver、本地 Wiki Suite，以及 Ruff、strict mypy、Wheel
独立性和 Linux Sandbox 检查。本文不固定数量；发布证据保存准确命令输出。Lima 参考环境已经覆盖
Sandbox、virtiofs Worker Root 准备与 Claude/Codex/QoderCLI 真实连通性。准确命令见
[测试与生产验收](testing-and-acceptance.zh.md)。这些结果不能证明生产安全，因为仍缺少准确部署
镜像恶意代码验证和目标 Agent/Agate/Wiki/GPU Crash/Soak 证据。

## 下一步顺序

1. 在准确部署 Kernel 与 systemd/bwrap 版本上运行并归档完整 Worker 文件系统/网络/Namespace/cgroup/宿主服务逃逸负测。
2. 让所选 Core Backend 经 Runtime Gateway Proxy 对真实 Agate 和目标 GPU 跑通全部授权操作。
3. 在每个持久 Transition 强制终止，演练备份、恢复、GC、凭据轮换和 Wiki Outbox 恢复。
4. 运行代表性多 DSL Soak，并用证据设定存储与资源限制。
