# 协议参考

[English](protocols.md) | 中文

所有 JSON Schema 都拒绝未知字段。ID 使用类型前缀；Artifact Digest 使用 `sha256:<64 位小写十六进制>`。需要参与 Digest 的持久 Payload 使用规范 UTF-8 JSON。

## HTTP 接口

| 路由 | 权限 | 用途 |
| --- | --- | --- |
| `GET /healthz` | 无 | 进程存活。 |
| `GET /readyz` | 无 | 本地 Registry、Gateway Control、Agate Job Store、Artifact Store 探针。 |
| `POST /v1/operations` | Attempt Capability | 结构化 Gateway Operation；Agent 的每次 `evaluate` 都形成不可变探索记录。 |
| `POST /v1/wiki/query` | Attempt Capability | Runtime 补全上下文的实时 Wiki 查询，返回前冻结。 |
| `POST /v1/admin/campaigns/bootstrap` | Admin Bearer | 只接受 Campaign schema v3。 |
| `POST /v1/admin/campaigns/{id}/lineages` | Admin Bearer | 从已封存 Agent 与 Kernel 内容增加独立版本化的 Lineage。 |
| `GET /v1/admin/bootstrap-attempts/{id}/runs` | Admin Bearer | 查询有序的物理 Bootstrap 执行 Generation。 |
| `GET /v1/admin/bootstrap-attempts/{id}/runs/{generation}` | Admin Bearer | 查询准确状态、失败、Workspace、Token、Trace、Report 与结果身份。 |
| `GET /v1/admin/campaigns/{id}/epochs` | Admin Bearer | 跨 Campaign Lineage 查询 Active/Challenger 竞争与胜者历史。 |
| `GET /v1/admin/lineages/{id}/epochs` | Admin Bearer | 查询带 `agent-vN` 和 `vN` 标签的有序 Epoch 胜者历史。 |
| `GET /v1/admin/campaigns/{id}/attempts` | Admin Bearer | 跨 Campaign Lineage 查询全部 Attempt，包括无 Candidate 的结果。 |
| `GET /v1/admin/lineages/{id}/attempts` | Admin Bearer | 查询单个 Lineage 完整、有序的 Attempt 历史。 |
| `GET /v1/admin/attempts/{id}` | Admin Bearer | 查询一个 Attempt 的状态、Report 处置和输入/输出版本关系。 |
| `GET /v1/admin/attempts/{id}/evaluations` | Admin Bearer | 查询有序的 Agent 探索评测和 Runtime 终评。 |
| `GET /v1/admin/attempts/{id}/evaluations/{evaluation_id}` | Admin Bearer | 查询一对不可变的被评测 Kernel/Result 身份。 |
| `GET .../evaluations/{evaluation_id}/source` | Admin Bearer | 返回该次评测对应的准确、有界 Kernel 文件。 |
| `GET .../evaluations/{evaluation_id}/result` | Admin Bearer | 返回该次评测的完整原始 Gateway Result。 |
| `GET /v1/admin/attempts/{id}/kernel-trials` | Admin Bearer | 查询 Attempt 内观察到的精确未版本化 Candidate。 |
| `GET /v1/admin/attempts/{id}/kernel-trials/{trial_id}` | Admin Bearer | 查询一个 Trial 的 Gateway Observation 与 Agent 决策。 |
| `GET .../kernel-trials/{trial_id}/source` | Admin Bearer | 返回被测量、探测或回退 Trial 的精确文件。 |
| `GET .../kernel-trials/{trial_id}/results` | Admin Bearer | 返回每个 Trial Observation 保留的结构化结果。 |
| `GET /v1/admin/campaigns/{id}/kernels` | Admin Bearer | 跨 DSL Lineage 查询全部 Baseline 与终止 Attempt Kernel。 |
| `GET /v1/admin/lineages/{id}/kernels` | Admin Bearer | 查询单个 DSL Lineage 的有序 Kernel Catalog。 |
| `GET /v1/admin/campaigns/{id}/agent-revisions` | Admin Bearer | 查询 Campaign 各 Lineage 的 Agent 版本历史。 |
| `GET /v1/admin/lineages/{id}/agent-revisions` | Admin Bearer | 查询单个 DSL Lineage 的有序 `agent-vN` 历史。 |
| `GET /v1/admin/agent-revisions/{id}` | Admin Bearer | 查询一个 Agent Revision 的 Parent、处置结果与 Artifact 来源。 |
| `GET /v1/admin/kernels/{id}` | Admin Bearer | 查询 Kernel、生产 Agent/Attempt 上下文、主评测和持久 Repeat Measurement。 |
| `GET /v1/admin/kernels/{id}/source` | Admin Bearer | 以 UTF-8 或 Base64 返回有界、精确的 Kernel Artifact 文件。 |
| `GET /v1/admin/kernels/{id}/measurements` | Admin Bearer | 查询主 Gateway Evaluation 与持久 Retention/Promotion Repeat。 |
| `/v1/admin/campaigns/...` | Admin Bearer | 查询持久化的 Campaign/Lineage Model、取消及调度 Campaign/Task。 |
| `/v1/admin/epochs/.../recover` | Admin Bearer | 显式恢复 Failed Epoch。 |
| `/v1/admin/events`、`/events/export`、`/events/prune`、`/metrics` | Admin Bearer | 有界可观测性与维护。 |

不存在 `/v1/admin/bootstrap` 兼容别名。

Agent 的每次不同评测都会封存准确 Candidate 与原始 Result，但不会提交 Attempt Outcome。终止
Candidate 由 Runtime 封存，并通过配置的权威留存 Comparator 评测；只有 `runtime_final` 记录会关联权威 Outcome。CLI 的
`list-evaluations` 与 `show-evaluation [--source] [--result]` 暴露同一份历史。
每个 Candidate 型 Operation 都会在外部执行前绑定封存后的 Candidate Artifact；即使 Operation
失败、Event 被裁剪或源码被回退，该绑定仍是持久 Artifact Root。Schema-v3 Attempt Report 可以
把 `continue`、`revert`、`pivot` 注解绑定到当前 Attempt 已观察的 Digest。这些未版本化
`Kernel Trial` 不占用 `vN` Revision 编号。
Optimizer 可用 `operation: "kernel_trials"` 列出这些 Trial，再用
`operation: "kernel_trial_read"` 获取已验证的精确文件索引或源码。Runtime 会包含当前 Attempt，
因此同一 Session 内回退的 Candidate 仍可恢复；更早 Attempt 则继续遵循已晋升分支与同轨迹可见性
规则。这两个 Runtime 本地读操作不计配额，也不会访问 Agate。
Worker 返回与管理面可见的原始 Result 有意不同：私有输入、Request、逐 Case 失败详情和 Evaluator
Spec 都会被隐藏。`evaluate` 只返回聚合正确性/延迟，以及可选的按不透明 `shape_id` 标识的延迟；
`profile` 可选一个不透明 `shape_id`，省略时由评测器选择一个私有 Case，并返回脱敏后的 Profiler
视图。`check` 不暴露 Shape 选择参数：Runtime 确定性选择 Contract 中排序后的第一个不透明
Shape，并把其私有 `init_kwargs` 传给 Agate Compile，使参数化 `Model` 构造器得到正确检查。
该协议不依赖 `launcher.mode`。

Runtime 把成功的 `evaluate` 与 `profile` 响应规范化为只追加的 `gateway_measurements` 记录。
`operation: "measurements"` 通过同一个 Attempt 范围 Gateway Endpoint 查询这些记录，不消耗
评测调用配额，也不会访问 Agate。可见性由 Runtime 推导，而非调用方选择：对 Optimizer，已完成
Epoch 只暴露胜出分支 Attempt；当前 Epoch 只暴露相同 Branch、Challenger Slot 与 Trajectory 中
更早完成的 Attempt。过滤器可选择 Kernel Revision/Artifact、Operation Kind、不透明 Shape、
Profiler Kernel、Metric 名称和有界数量。每条记录保留原始 Gateway Result Digest 以便审计。

Attempt 历史独立于 Kernel 历史。`list-attempts [--format table]` 与 `show-attempt` 会保留
`pivot`、`blocked`、基础设施失败及其他无 Candidate 的 Session；由于没有注册 Kernel
Revision，这些 Attempt 不会获得输出 `vN`。

Epoch 历史暴露赛前 Active、有序 Challenger Pool、最终胜者、
`active_retained`/`challenger_promoted` 决策、起始 Kernel 和全局最佳 Kernel。
`list-epochs [--format table]` 同时支持 Campaign 与 Lineage Scope。

Kernel Catalog Entry 暴露不可变 Kernel/Agent Revision ID、Lineage 内 `vN` 与 Parent Version
标签、Campaign、Lineage、DSL、可选 Epoch/Attempt/Branch 上下文、保留决策、源码与 Gateway
Result Artifact 引用、正确性、延迟、相对 Parent 的性能变化、Gateway 报告的 SOL 百分比和创建
时间。单 Shape 使用准确的 `sol.pct`；多 Shape 使用全部 Shape 的几何平均。如果任一 Shape 没有
SOL 值，JSON 投影为 `null`，表格显示 `-`。同 Allocation ABBA Result 携带封存的 Evaluation
Contract Digest，以及权威的 Candidate 延迟/SOL 聚合，因此 ABBA Artifact 替换探索 Eval Result
后，Kernel Catalog 不会丢失 Roofline 证据。Artifact 引用对象包含
`digest`、`kind`、`referenced_at`，并继续保留兼容的纯 Digest 字段。探索评测使用稳定的
`g<generation>-e<ordinal>` 标签，不占用 Kernel 版本号。Framework Baseline 和 Attempt 主评测来自 Kernel Revision；每次 Retention/Promotion Repeat
Evaluate 独立写入 `kernel_measurements`，即使生命周期 Event 被清理也仍可查询。

Kernel Agent Catalog 使用独立的 Lineage 本地 `agent-vN` 序列，暴露 Parent Agent Version、
Bootstrap/Lineage Seed/Evolver 来源、引入 Epoch、Active 标记、晋升处置结果、创建时间与准确 Optimizer
Artifact 身份。每条 Kernel 记录也包含生产它的 Agent Version。CLI 的
`list-agent-revisions [--format table]` 与 `show-agent-revision` 暴露相同投影。

## Bootstrap 与 Bundle

Campaign schema v3 必须提供 `creation_key`、算子、硬件目标、Evaluation Contract
路径、完整 `base_revision.commit`、`challenger_count`、`challenger_start_epoch`、
`trajectories_per_branch`、`attempts_per_trajectory` 和各 DSL 的
`baseline_kernel`/`initial_evidence`。`lineages` 的 Key 是权威且完整的初始 Bootstrap DSL 集合；可选
`agent_problem` 跳过 Core 问题泛化。可选的 `lineages.<dsl>.models.optimizer` 与
`.evolver` 分别绑定该 Lineage 的两类 Session；缺省或 `null` 表示使用 Backend CLI 默认
Model。顶层可选 `problem_generalization_model` 只用于生成 Agent Problem。Runtime 持久化这些
Model 身份，并以 `ATREX_AGENT_MODEL` 注入新 Session。Runtime 只导入一次 Core，并从相同
Optimizer Digest 为各 DSL 创建 Revision。一个稳定 Bootstrap Attempt 可以有多个 Append-only
Recovery Generation；每个新物理 Session 获得新的 Capability Generation，旧 Gateway Operation
按原 Generation 保留，只有正确且已提交的 Outcome 才能创建 Baseline。

Lineage seed schema v1 必须提供 `creation_key`、固定 `dsl`、Epoch 拓扑和带判别字段的
`seed`。`source_type: "artifacts"` 指定 `agent_artifact_digest` 与
`kernel_artifact_digest`；`source_type: "revisions"` 指定 `agent_revision_id` 与
`kernel_revision_id`。可选的 `models.optimizer`、`models.evolver` 和 `initial_evidence`
配置新 Lineage。Runtime 会重新校验完整 Agent Bundle，在目标 Campaign Contract 下独立评测
准确 Kernel，并发布新的 `agent-v0`/`v0` 身份。等价 CLI 是
`seed-lineage --config ... --campaign ... --spec ...`。标准 Bootstrap 仍由
`base_revision.commit` 锚定；Seed 操作不导入 Git 源码，只接受已封存 CAS 内容。
Campaign Bootstrap 还会冻结部署选择的完整 Evolver Commit。Bootstrap 与 Campaign 查询响应会
返回 `evolver_commit`；所有调度路径都会在执行 Evolver 前拒绝不同的配置 Commit。Seed Lineage
继承这个 Campaign 级值。

Evaluation Contract 内的 `roofline` 仍是可选项；显式值具有权威性。其为 null 且部署配置了
`campaign.roofline_builder` 时，Runtime 执行 Commit 固定的 Atrex Bench Converter，要求准确
Shape 覆盖以及有限非负的 W/Q/SOL 字段，并在 Problem Generalization 前封存生成结果。已有
Campaign 重试时，只要输入的非 Roofline 字段未变，就复用已封存的生成版 Contract。两种来源都
没有或构建失败时，Runtime 保留不带 Roofline 的 Contract，并在每个正确的 Agent Eval 或权威
Eval 后执行一次 NCU SOL Profile。两个原始响应封存在同一个 Gateway Result Artifact 中；Profile
失败不会推翻
Eval 的正确性或延迟。

每条 Lineage 的结果会保留兼容字段，并额外返回 `bootstrap_attempt_id` 与结构化
`baseline_kernel` 对象；后者包含 Revision ID、`v0` 标签、创建时间和 Bootstrap Producer
身份，返回还包含初始 `agent-v0` 身份与 Optimizer Artifact。该 Producer 与 Kernel Catalog
中普通 Epoch `attempt_id` 明确分离。

Core `atrex-bundle.json` schema v1 声明唯一仓库相对入口。Runtime 把 `ATREX_CORE_PHASE` 设置为 `problem_generalization`、`framework_baseline` 或 `optimization_attempt`，并提供阶段 Manifest/Report、Token Budget/Report、可选 Session Trace、隔离 Home 与受限 Gateway/Wiki Authority。退出 0 表示完成，125 表示 Token Budget Exhausted，其他退出码均显式保留。Token Report 必须有效。

Sandbox 模式下 Runtime 还会设置 `HOME=/home/agent`、
`ATREX_WORKSPACE=/home/agent/workspace` 与 `ATREX_SANDBOX=bwrap-cgroup-v1`。Worker 直接共享
宿主网络与 DNS；Runtime 不注入代理变量，也不限制可访问的宿主 Port。文件系统和 cgroup 隔离不受
影响。任何位于 Session Root 下的宿主绝对路径都会
在启动前映射到对应 `~/workspace` 路径。

Evolver `atrex-evolver-bundle.json` schema v1 声明唯一入口。Runtime 固定通过 stdin 发送
`Run the versioned Evolver Bundle once.`。Evolution Input schema v4 固定 Parent、Evidence
Checkpoint、DSL、Optimizer Digest、Workspace Path 和只读 `visible_agents` Catalog；其中包含
Active Parent、已保留的 Lineage Agent 历史，以及同一 Epoch 中此前创建的 Challenger。每个条目
包含仓库路径、Parent Link、创建者、关系类型，以及适用时的当前 Epoch Challenger Ordinal。
带判别字段的 Output schema v3 声明提案形态、所选 Revision/Base、假设、预期效果，以及适用时的
准确 Changed Paths；Trace schema v7 记录所选 Model、
Bundle Commit/Tree/Artifact 与进程证据。Evolver 没有 Token 截止；必需 Report 使用空 Budget
记录完整 Provider Usage。
Schema v4 还绑定 `runtime-tools/`。Runtime 冻结 `catalog.json`、全部 Lineage Kernel Artifact
和冻结的 `evolver_tools.py` Client。其有界 JSON 检索命令包括 `history`、`branches`、`attempts`、
`kernels`、`kernel-read`、`agents`、`agent-diff` 和 `trace-paths`。`candidate-reset` 是唯一写操作：
它只把已完成 Lineage 历史原子加载到 `candidate/` 并记录 Base。Client 不携带 HTTP Credential
或可变 Runtime 权限。

## Evidence 与 Wiki

Attempt Evidence schema v2 不可变，只包含相同 Branch Slot 和 Trajectory 中较早的 Attempt。
已完成 Epoch Evidence 保留 Challenger 与 Trajectory 身份、Summary、Report、Diff、规范化
Session Projection、规范化 Evaluate/Profile Measurement、Lesson 和来源 Digest。Agent View
schema v1 按 Role 限制：Optimizer 只看到
已晋升的完成 Lineage 和有界的当前同 Trajectory Attempt，并移除 Branch 控制身份；
Evolver 看到所有已完成分支、明确 Agent 选择事实、每个 Attempt Outcome、所有 Challenger
Evolution Trace 与被引用的精确 Kernel Artifact。两种 View 都按 Digest 把 Session Artifact
物化为派生只读副本。Optimizer 投影只包含已晋升分支的完成 Epoch Measurement；Evolver
投影冻结全部已完成 Active/Challenger Measurement，不需要实时 Gateway Authority。权威 Session
保留策略省略 Claude `system/thinking_tokens` 估算遥测，
派生副本也会对旧 Session Artifact 防御性地应用同一规则。

每个新 Session Artifact 都包含 `conversation.jsonl`：它以版本化 Schema 未脱敏记录准确 Runtime
输入、每条保留的 Provider stdout Event、适用时的 Codex Rollout Event，以及 Runtime 捕获终态。
当 CLI 不导出 Provider 管理的 System Prompt 时，记录会明确声明不可获取。
高频 Claude `system/thinking_tokens` 估算事件不会写入 `provider/stdout.stream-json` 和对话，
且由 `session.json.provider_event_filters` 明确声明。`events.jsonl` 仍保留规范化的权威
Usage/Projection Ledger。

Wiki Query Service API 为版本 1。Query Content 使用 GPU Wiki 准确的
`records`/`notes` 投影，稳定 Record ID 是 `records` Mapping Key，每个 Value 都是完整安全 Record。
响应会在 Core 收到仅知识 Content 前冻结为 Interaction Artifact。Runtime 不向 Wiki 上传 Query
消费记录、Agent 历史或 Session Trace。

Gateway Capability 有签名、绑定 Attempt、限制 Operation/调用数/过期时间，可撤销，并由 Registry Authority 重建。Worker 编写的 Report/Annotation 和 Agent 可见探索评测是不可信 Evidence；Runtime-final Gateway Outcome 与 Registry Transition 才是权威来源。
