# 持久协议

[English](protocols.md) | 中文

本文定义持久身份与可见性规则。可调用命令和 Route 见[接口参考](interfaces.zh.md)，部署字段见
[配置参考](configuration.zh.md)，测量策略见[评测与晋升](evaluation.zh.md)。Pydantic Model 与数据库
Migration 是可执行 Schema 权威。

## 身份与版本

Typed ID 不透明且带 Prefix：`campaign_`、`lineage_`、`epoch_`、`attempt_`、
`workersession_`、`kernelrev_`、`agentrev_` 和 `sha256:`。调用方不能解析后缀语义。

内容身份与展示身份分离：

- Artifact Digest 标识精确字节，用于校验与去重。
- Kernel Revision ID 与 Lineage 内 `vN` 标识一次被保留的历史引用。
- Agent Revision ID 与 Lineage 内 `agent-vN` 标识一个 Agent Bundle 引用。
- Kernel Trial ID 标识 Attempt 可见历史中的未版本化测量 Candidate。

Rejected、Reverted、Blocked、Pivot 或 Infrastructure Failed Attempt 都会持久化；只有真正发布
Kernel Revision 时才消耗 `vN`。

## Campaign 与 Lineage

Campaign 冻结算子、解析后的硬件架构、Agate GPU Selector、Evaluation Contract、公开 Agent
Problem、Core 来源、Evolver Commit 与策略。只有所有不可变输入一致时，Creation Key 才能幂等复用。

每条 Lineage 持有唯一 DSL、独立 Agent/Kernel 版本树、Model、Epoch 拓扑、Active Agent、Best
Kernel、公共 Evidence Checkpoint 与 Runtime State 历史。Seed Lineage 会创建新的
`agent-v0`/`v0` 根；源 Revision ID 只是 Provenance，不共享版本祖先。

Ablation Arm 是从另一 Lineage 的封存 Bootstrap Baseline 创建的独立 Campaign/Lineage，没有
Challenger；Ephemeral State 行为属于其 Lineage 身份。

## Epoch、Branch、Trajectory 与 Attempt

一个 Epoch 记录赛前 Active Agent、有序 Challenger、冻结起始 Kernel/Evidence/State、Branch 与
Trajectory Outcome、最终 Best Kernel、Agent Winner 和发布 Checkpoint。

Branch Name 表示竞争角色，不表示祖先关系。所有 Branch 从相同 Epoch 输入开始。Trajectory 是
Branch 内独立的串行 Attempt 链；Attempt 是一次 Optimizer 逻辑步骤，自动基础设施重试不会创建新
Attempt。

Runtime Fencing 为每个 Scheduler Owner 关联可续租 Generation。过期 Owner 可以完成本地工作，但
不能提交 Transition。

## Bootstrap Generation 与 Worker Session

一个稳定 Bootstrap Attempt 可以包含多个 Append-only 物理 Generation。每个 Generation 使用新的
Capability Generation、Workspace、Worker Session、Usage Report、Trace、Report、Operation、Failure
与 Outcome；旧 Generation 永不覆盖。

每个模型进程都是 Worker Session，记录 Subject、Role、Backend/Model、起止状态、Workspace、终态
诊断、Provider Usage 和可选不可变 Session Artifact。Session 是物理执行身份；Attempt、Bootstrap
Attempt、Generalization Subject 或 Evolution Subject 是逻辑 Owner。

权威 Session Artifact 保留 Backend-neutral `conversation.jsonl` 与归一化 Event Ledger。Claude
高频 `system/thinking_tokens` 估算 Telemetry 被丢弃，终态 Provider Usage 仍然必需。

## Evolver 与 Agent State

Evolver `atrex-evolver-bundle.json` schema v1 声明唯一入口。Runtime 固定通过 stdin 发送
`Run the versioned Evolver Bundle once.`。Evolution Input schema v11 固定 Parent、Evidence
Checkpoint、DSL、Optimizer Digest、Workspace Path 和只读 `visible_agents` Catalog；其中包含
Active Parent、已保留的 Lineage Agent 历史，以及同一 Epoch 中此前创建的 Challenger。每个条目
包含仓库路径、Parent Link、创建者、关系类型，以及适用时的当前 Epoch Challenger Ordinal。
无版本号的 Evolution Output 声明提案形态、所选 Agent Revision、假设、预期效果、相对于 Source
根目录的准确 Changed Paths，以及 Candidate 取用过内容的其他可见 Revision；Trace schema v9 记录所选 Model、
Bundle Commit/Tree/Artifact 与进程证据。Evolver 没有 Token 截止；必需 Report 使用空 Budget
记录完整 Provider Usage。
Schema v11 不包含 Runtime 查询权限。每个可见 Agent 都按稳定 Lineage 版本位于
`input/agents/agent-vN/`，Runtime 派生 Evidence 位于 `input/evidence/agent-vN/`。每个版本都有生涯汇总；
最近完成 Epoch 的参赛者还额外暴露该 Epoch 的 Conversation 与 Attempt Report。Evolver 直接从冻结文件
读取 Source、Runtime State、汇总与可用 Session Evidence。从历史派生时，它把所选历史 Source 复制到
Candidate Source，并可从可见历史状态整理扁平公共种子；Runtime 无需 Candidate Base 旁路记录，直接验证
Agent Revision、报告的 Source Diff 与私有 Runtime State Diff。
新 Revision 可以融合多个合格可见 Agent 的内容。`contributing_revision_ids` 记录除唯一 Source Base 外
的全部贡献者；它只表示 Provenance，不改变 Diff Base，也不增加祖先边。当前 Epoch 中尚未参赛的
Challenger 虽会出现在 Catalog 中，但不能作为贡献者。
Agent 通过只读 Bundle 内的本地 `evolution-report` 命令提交该输出。无效 Draft 返回结构化
`issues`、`request_schema` 与 `recovery` 且不发布；第一个有效 Draft 原子写入
`scratch/evolution-report.json`。Agent 退出后 Runtime 再独立校验报告和 Candidate。
按 Trajectory 划分的 `runtime-state/trajectories/<N>/{memory,docs,skills,tools}/` 保存自适应 State，
属于非版本化 Lineage 状态。根级 `skills/` 和 `tools/` 在版本化 Agent Source 中无效。Evolver 可直接
整理扁平的 `candidate/runtime-state/{memory,docs,skills,tools}/` 公共种子，也可修改控制未来状态使用方式的版本化
机制。Runtime 始终独立封存 Candidate Source 与 State，并把两者组合为同一个不可变 Agent Bundle；
所有新 Trajectory 都从该 Bundle 的 State 初始化。State 是否相对输入发生修改不影响封存。Agent
Revision 直接记录 `optimizer_digest` 与 `runtime_state_digest`；Evolution Trace 是来源证据，而不是
State 身份的唯一位置。缺少直接字段的旧 Revision 仍可通过 Trace 兼容读取。

## Artifact 与 Measurement

本地 CAS 保存 Agent Source、Kernel Source、Runtime State、Evaluation Contract、Agent Problem、
Gateway Result、Report、Trace 与 Evidence 的 Canonical Manifest/Payload。Artifact 不可变；Registry
Row 与 Gateway Control Record 赋予 Artifact 领域语义。

每个携带 Candidate 的 Gateway 操作都会先封存 Kernel Source。`evaluate` 或 `profile` 创建不可变
Operation/Result 与归一化 Measurement；即使 Agent 后续恢复另一 Kernel，这些事实仍有效。

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

Kernel Trial 把一份精确 Kernel Artifact 与可见 Gateway Observation 组合。Agent 可以通过已记录
Experiment 恢复 Trial、按 Artifact Digest 读取源码，并在不调用 Agate 的情况下检查结果。归一化
Measurement 是内部持久事实，不是无限制 Agent 查询接口。

Kernel Revision 的主评测是 Runtime 选择的 Bootstrap 或 Retention Result；Comparator Repeat 是
独立持久 Measurement。精确 Raw Job 对管理端可见，Worker 只收到安全投影。

## Direction 与 Experiment Journal

Direction 表示研究/探索假设，不只表示代码修改。定义不可变，状态变化 Append-only。同一时刻最多
一个 Direction 为 In Progress；一个 Attempt 最多推进三个不同 Direction，但可提出更多未来方向。

Experiment 把一个 Direction 绑定到精确测量的 `before`/`after` Kernel、Trial、Gateway Result
身份，以及事实 Evidence、分析和 Action。Bootstrap 首次锚点使用 `action="baseline"`、
`before=null` 和完整 `after`。

Journal 调用会在响应前同步校验并持久化。Authority 作用于逻辑 Attempt，因此物理 Session 失败或
Retry Generation 不会丢记录。终态 `attempt-report` 快照同一 Journal，要求没有 Direction 仍在
进行；校验失败后可以修正重试，直至首次成功的 Write-once 发布。保留决策属于 Runtime，而不是
Agent Report。

Runtime 把 Agent Handoff 与权威 Parent/Candidate Outcome、Comparison、Production Gate、Correctness、
按不透明 Shape ID 的 Latency、Profile Evidence 和 Direction-bound Finding 合并为 Final Attempt
Report。

## Runtime State

版本化 Agent Source 不能包含顶层 `skills/` 或 `tools/`。Adaptive State 是独立 Artifact：

```text
runtime-state/
  trajectories/<N>/
    memory/README.md
    docs/README.md
    skills/README.md
    tools/README.md
```

Optimizer Workspace 把一条 Trajectory 的 State 展示为根级可写 `memory/`、`docs/`、`skills/` 和 `tools/`。
各目录 README 必须随内容的新增、修改、重命名和删除同步更新。四目录整体封存，遵循相同的继承与隔离
规则；清空状态的消融臂每次只保留四份 README 模板。自适应 Docs 与 Source 内的实现文档相互独立。Runtime
封存终态内容并为下一个串行 Attempt 恢复。Evolver 获得冻结 Participant/Historical State，并在
`candidate/runtime-state/{memory,docs,skills,tools}/` 编写一份扁平 Candidate Seed。新 Agent Revision 同时
记录 Source 与 State Digest，作为一个逻辑 Bundle；每条新 Trajectory 得到独立副本。

启用 Ephemeral Agent State 的 Ablation Lineage 会让每个 Attempt 从空 Adaptive State 开始。

## Evidence 可见性

可见性防止并发搜索间的信息泄漏：

- Optimizer 看到已晋升完成历史，以及当前 Trajectory 中更早的 Attempt。
- Epoch 运行中，Optimizer 看不到竞争 Branch。
- Epoch Barrier 后，胜负 Branch 的完成 Journal 可进入后续 Direction/Experiment 历史，但
  Branch/Selection Provenance 对 Optimizer Tool 隐藏。
- Evolver 看到冻结的最近完成 Epoch Active/Challenger Summary 与 Conversation、历史 Agent
  Source/State Summary，以及旧 Evolution Report。
- Evolver 没有 Gateway、Wiki、Journal 或 Runtime Query Authority；它只读冻结文件系统，并仅通过
  Bundle 本地 `evolution-report` 提交。

详细 Registry/Evidence Tree 不会复制进 Agent Workspace；Runtime 只物化当前 Session 角色所需文件。

## Agent Bundle 与 Evolution

Core/Evolver Entry Manifest 各声明一个仓库相对命令。Runtime 导入精确完整 Commit，移除 Git
Metadata，拒绝不安全 Tree 内容并封存完整 Source。Runtime 持有 Backend/Model 策略，通过 Launch
Contract 提供 Phase、Path、Usage 与受限 Authority。

Evolution Input 冻结当前参赛者、可见历史 Agent、Evidence、旧 Report、DSL 与 Candidate Seed。
Output 有三种：

- `evolved`：Parent 为 Active 的新 Revision；
- `reuse`：原样复用可见历史 Revision；
- `evolve_from_history`：Parent 为可见历史 Revision 的新 Revision。

Runtime 在封存前校验所选 Source、Source-relative Changed Path、私有 State Diff、同 DSL 身份、
File Policy 与 Manifest。进化内容只属于该 Lineage；Runtime 不推送回 Core 仓库。

## Capability 与外部服务

Gateway/Wiki Capability 带签名、Attempt Scope、Operation 限制、过期、撤销和 Generation Scope。
Idempotency Key 对同请求重放已提交响应，并拒绝内容不同的请求。Runtime 本地 Journal/History 查询
不调用 Agate，也不消耗 Gateway 配额。

Agate Credential 与私有 Request Construction 保留在 Runtime。GPU Wiki 只有实时 Query Contract；
Runtime 在只向 Core 返回知识前冻结完整交互，不存在 Wiki Feedback/Upload Protocol。

Provider Usage Fail Closed：QoderCLI 报告 Credit；Claude、Codex 与 Pi 报告 Provider Token Bucket。
Optimizer 有逐 Session 配额。Evolver 没有用量截止，但除非 Process Failure/Timeout 是主结果，仍
必须生成完整终态 Usage Report。
