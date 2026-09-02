# 协议参考

[English](protocols.md) | 中文

所有 JSON Schema 都拒绝未知字段。ID 使用类型前缀；Artifact Digest 使用 `sha256:<64 位小写十六进制>`。需要参与 Digest 的持久 Payload 使用规范 UTF-8 JSON。

## HTTP 接口

| 路由 | 权限 | 用途 |
| --- | --- | --- |
| `GET /healthz` | 无 | 进程存活。 |
| `GET /readyz` | 无 | 本地 Registry、Gateway Control、Agate Job Store、Artifact Store 探针。 |
| `POST /v1/operations` | Attempt Capability | 结构化 GPU/Agate Operation；Agent 的每次 `evaluate` 都形成不可变探索记录。 |
| `POST /v1/runtime/queries` | Attempt Capability | 按已知 Trial、Kernel 源码或 Gateway Result 执行不计配额的读取。 |
| `POST /v1/runtime/journals` | Attempt Capability | 立即持久化 Direction/Experiment Mutation，并执行授权的 list/load 读取。 |
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
| `GET /v1/admin/attempts/{id}/report` | Admin Bearer | 查询包含 parent/Candidate 权威 Gateway 性能的 Runtime 最终 Attempt Report。 |
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
失败、Event 被裁剪或源码被回退，该绑定仍是持久 Artifact Root。Schema-v12 Agent Attempt Handoff 为
每个 Direction 和 Experiment 分配持久 ID。Direction 定义不可变，状态变化以事件追加；每个 Experiment
绑定其 Direction。普通比较 Action 会把 `before`、`after` 两侧分别绑定到准确 Kernel Artifact、
Kernel Trial 与非空 Gateway Result Digest 列表。Bootstrap 专用的 `baseline` Action 则要求
`before=null` 和完整 `after`，只建立首个测量锚点，不注册 `v0`。`evidence` 保存观测事实，
`analysis` 保存解释与假设判定；其他 `action` 记录 Agent 保留 After Kernel、恢复 Before Kernel 或
放弃方向，并由 Runtime 映射为已观察 `after` Trial 的内部处置状态。Report 的终态 `status` 只是
Optimizer 交接状态，不是保留决策；协议不接受顶层 `decision`。这些未版本化 `Kernel Trial`
不占用 `vN` Revision 编号。随后 Runtime 把 Handoff 与 Registry 中的输入/输出 Kernel 及其权威
Gateway Result Artifact 合并，派生 schema-v1 最终 Attempt Report。该投影展示准确 Kernel
Artifact 身份与整体/逐 Shape 延迟，但不暴露内部 Kernel Revision ID；同时把可信内容级
Production Gate 投影为 `NOT_ENABLED`、`PASS`、`FAIL` 或 `NOT_RUN`，该字段完全由 Runtime 生成。
Direction 与 Experiment Mutation 统一使用 `/v1/runtime/journals`。Runtime 会在响应前完成校验与
持久提交，并把幂等键绑定到逻辑 Attempt，因此 Journal 可以立即查询，且不因物理 Session 失败或
Recovery Generation 变化而丢失。`attempt-report` 从 Runtime 获取快照，并把同一份权威的本 Attempt
Journal 嵌入终态 Handoff；终态发布不是 Journal 第一次持久化的时点。
已完成的选中和未选中搜索路径都会向后续 Optimizer Session 提供冻结的 Direction/Experiment
Journal，但不暴露调度来源。
Optimizer 可用 Gateway 响应或已保留 Experiment 记录中的已知 Trial ID 调用
`kernel-trial-show` 查看溯源信息，按
`kernel_artifact_digest` 调用 `kernel-artifact-read` 读取准确源码，并按
`gateway_result_digest` 调用 `gateway-result-read` 读取规范化 Agent 可见测量。Runtime 会包含当前 Attempt，
因此同一 Session 内回退的 Candidate 仍可恢复；更早 Attempt 则继续遵循已晋升分支与同轨迹可见性
规则。这些 Runtime 本地读操作不计配额，也不会访问 Agate。
Worker 返回与管理面可见的原始 Result 有意不同：私有输入、Request、逐 Case 失败详情和 Evaluator
Spec 都会被隐藏。`evaluate` 只返回聚合正确性/延迟，以及可选的按不透明数字 `shape_id` 标识的延迟；
`profile` 可选一个数字 `shape_id`，省略时由评测器选择一个私有 Case，并只用该编号标记脱敏后的
Profiler 视图。`check` 与 `disassemble` 都不暴露 Shape 选择参数：Runtime 确定性选择 Contract 中排序
后第一个声明了 `init_kwargs` 的不透明 Shape，并把其私有 `init_kwargs` 传给 Agate 编译作业，使参数化
`Model` 构造器得到正确构建。
该协议不依赖 `launcher.mode`。

Runtime 把成功的 `evaluate` 与 `profile` 响应规范化为只追加的内部
`gateway_measurements` 记录。它们继续供 Gate、冻结 Evidence 和管理接口使用，但不再作为独立的
Agent 查询面。`kernel-trial-show` 接受已知 `kernel_trial_id`，只返回 Kernel Artifact Digest 和规范化
`gateway_results`，Result Entry 不再重复已经解析过的 Gateway Result Digest。可见性由 Runtime
推导，而非调用方选择：对 Optimizer，已完成 Epoch 只暴露
胜出分支 Attempt；当前 Epoch 只暴露相同 Branch、Challenger Slot 与 Trajectory 中更早完成的
Attempt，以及自身正在进行的 Trial。

Attempt 历史独立于 Kernel 历史。`list-attempts [--format table]` 与 `show-attempt` 会保留
`pivot`、`blocked`、基础设施失败及其他无 Candidate 的 Session；由于没有注册 Kernel
Revision，这些 Attempt 不会获得输出 `vN`。完成的 Agent Session 还会封存准确终态 Runtime
State；生产它的 Attempt 以 `runtime_state_digest` 记录 Artifact，已有 `attempt_id` 就是生产者引用，
不存在额外的 State Checkpoint ID。串行恢复使用该不可变摘要，而不信任可变 Workspace 缓存。第一次
物理 Session 前，Runtime 还会把逻辑输入封存为 `input_runtime_state_digest`；物理重试可以推进
`runtime_state_digest`，但不会改写逻辑输入。正常情况下，Runtime 使用获胜分支最佳 Kernel
Trajectory 最后一个 Attempt 的终态 State，作为下一 Epoch Active Branch 与 Evolver Candidate 的
共同种子；第一个 Attempt Input 只作为缺失终态 State 时的兼容性回退。

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

Campaign schema v3 必须提供 `creation_key`、算子、写在 `hardware_target` 中的 Agate GPU
环境选择器、Evaluation Contract
路径、完整 `base_revision.commit`、`challenger_count`、`challenger_start_epoch`、
`trajectories_per_branch`、`attempts_per_trajectory` 和各 DSL 的
`baseline_kernel`/`initial_evidence`。`lineages` 的 Key 是权威且完整的初始 Bootstrap DSL 集合；
优先使用可选的 `shape_train` 作为公开训练域 Contract；它与旧版 `agent_problem` 互斥，二者都可
跳过 Core 问题泛化。精确的 `shape_valid.json` Case、`metadata.json` 与 `roofline.json` 会封存在
Evaluation Contract，不会复制进 Agent 工作区。持久化的公开 Contract Artifact 保持完整；Core
只向 Agent Prompt 投影精简的执行视图：隐藏生成器与范围证据来源，并以 `shape_domain` 作为唯一的
参数域来源。固定参数直接表示为 JSON 值；可变参数才使用范围或多值 Domain 对象。算子名称与类别由
objective 表达；只有包含构造参数、输入
副作用、布局或返回行为等非 Shape ABI 语义时才保留 `operator_contract`。可直接推导的 invariant
不再重复展示；跨字段和语义 invariant 仍然保留。对于迁移期的 `atrex.agent_problem.v1`，旧版扁平
`operator_contract` 会全部归一化进 `shape_domain`；私有 `shapes.json` 仍只作为评测 Case 回退来源。
可选的 `lineages.<dsl>.models.optimizer` 与
`.evolver` 分别绑定该 Lineage 的两类 Session；缺省或 `null` 表示使用 Backend CLI 默认
Model。顶层可选 `problem_generalization_model` 只用于生成 Agent Problem。Runtime 持久化这些
Model 身份，并以 `ATREX_AGENT_MODEL` 注入新 Session。Runtime 只导入一次 Core，并从相同
Optimizer Digest 为各 DSL 创建 Revision。一个稳定 Bootstrap Attempt 可以有多个 Append-only
Recovery Generation；每个新物理 Session 获得新的 Capability Generation，旧 Gateway Operation
按原 Generation 保留，只有正确且已提交的 Outcome 才能创建 Baseline。

创建新 Campaign 前，Runtime 会调用 Agate `get_env(hardware_target)`。远端返回的 `arch`（例如
`sm_120`）成为 Campaign/Lineage 中保存并传给所有 Agent 的硬件目标；远端返回的规范 `gpu`
（例如 `L20N`）则作为 `agate_gpu` 单独封存在 Evaluation Contract 中，仅用于 Agate 调度。
若响应缺少任一字段，Bootstrap 会直接失败，不会再从环境别名猜测 Agent 可见硬件信息。
Agate 显式返回 Accelerator Backend 时 Runtime 会保留它；否则根据规范 GPU 与 Arch 推断 CUDA、
ROCm 或 PPU；Backend 与可选 Device Slug 会封存在 Evaluation Contract 中。`PPU-*` 与 `ZW-*`
设备会自动关闭托管锁频，因为 PPU-SMI 不支持该操作。

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

两种生产模式下 Runtime 都会设置 `HOME=/home/agent` 与
`ATREX_WORKSPACE=/home/agent/workspace`。`sandbox` 的 `ATREX_SANDBOX=bwrap-cgroup-v2`，
`container` 则为 `bwrap-container`。Worker 直接共享所在环境的网络与 DNS；Runtime 不注入代理
变量，也不限制可访问端口。两种模式都保留 bwrap 文件系统边界，只有 `sandbox` 增加逐 Session
cgroup。任何位于 Session Root 下的宿主绝对路径都会
在启动前映射到对应 `~/workspace` 路径。

Evolver `atrex-evolver-bundle.json` schema v1 声明唯一入口。Runtime 固定通过 stdin 发送
`Run the versioned Evolver Bundle once.`。Evolution Input schema v10 固定 Parent、Evidence
Checkpoint、DSL、Optimizer Digest、Workspace Path 和只读 `visible_agents` Catalog；其中包含
Active Parent、已保留的 Lineage Agent 历史，以及同一 Epoch 中此前创建的 Challenger。每个条目
包含仓库路径、Parent Link、创建者、关系类型，以及适用时的当前 Epoch Challenger Ordinal。
无版本号的 Evolution Output 声明提案形态、所选 Agent Revision、假设、预期效果、相对于 Source
根目录的准确 Changed Paths，以及 Candidate 取用过内容的其他可见 Revision；Trace schema v9 记录所选 Model、
Bundle Commit/Tree/Artifact 与进程证据。Evolver 没有 Token 截止；必需 Report 使用空 Budget
记录完整 Provider Usage。
Schema v10 不包含 Runtime 查询权限。Evolver 直接从冻结文件读取当前 Agent、历史
Agent、优化汇总、最近 Epoch Conversation 和 Runtime State。从历史派生时，它把所选历史 Source 复制到
Candidate Source，并可从可见历史状态整理扁平公共种子；Runtime 无需 Candidate Base 旁路记录，直接验证
Agent Revision、报告的 Source Diff 与私有 Runtime State Diff。
Agent 通过只读 Bundle 内的本地 `evolution-report` 命令提交该输出。无效 Draft 返回结构化
`issues`、`request_schema` 与 `recovery` 且不发布；第一个有效 Draft 原子写入
`scratch/evolution-report.json`。Agent 退出后 Runtime 再独立校验报告和 Candidate。
按 Trajectory 划分的 `runtime-state/trajectories/<N>/{skills,tools}/` 是唯一的自适应 Skill/Tool 存储，
属于非版本化 Lineage 状态。根级 `skills/` 和 `tools/` 在版本化 Agent Source 中无效。Evolver 可直接
整理扁平的 `candidate/runtime-state/{skills,tools}/` 公共种子，也可修改控制未来状态使用方式的版本化
机制。Runtime 始终独立封存 Candidate Source 与 State，并把两者组合为同一个不可变 Agent Bundle；
所有新 Trajectory 都从该 Bundle 的 State 初始化。State 是否相对输入发生修改不影响封存。Agent
Revision 直接记录 `optimizer_digest` 与 `runtime_state_digest`；Evolution Trace 是来源证据，而不是
State 身份的唯一位置。缺少直接字段的旧 Revision 仍可通过 Trace 兼容读取。

## Evidence 与 Wiki

Attempt Evidence schema v2 不可变，只包含相同 Branch Slot 和 Trajectory 中较早的 Attempt。
已完成 Epoch Evidence 保留 Challenger 与 Trajectory 身份、Summary、Report、Diff、规范化
Session Projection、规范化 Evaluate/Profile Measurement、Lesson 和来源 Digest。Agent View schema v1 按 Role
限制：Optimizer 按分支看到每个已完成分支的 Attempt Report 与 Conversation，每个 Epoch Summary 都标明
被选中的分支；当前 Epoch 仍只有有界的同 Trajectory Attempt，并移除 Branch
控制身份。Runtime 从完整持久 Evidence 派生精简 Evolver 文件系统 View：每个当前参赛者获得一份
权威的最近 Epoch 优化汇总和每个 Attempt 的一份 Conversation；已完成且非当前的 Agent 版本在源码旁获得
一份汇总。更早的详细分支 Tree 和精确 Kernel Artifact 保留在 Runtime Registry 与 Artifact Store，不重复进
Evolution Workspace。Evolver 不需要实时 Gateway Authority。权威 Session
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
