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

固定 Commit 的 Evolver Bundle 用 `atrex-evolver-bundle.json` schema 1 声明入口。
Runtime 通过 stdin 固定发送 `Run the versioned Evolver Bundle once.`。Evolution Input schema 11
包含 Parent、DSL、Evidence Checkpoint、工作区路径和冻结 Catalog，不授予 Gateway/Wiki 或 Runtime 查询权限。
Trace schema 9 保存进程、Usage、Report、Candidate 身份和贡献内容快照；Provider Usage 必需，不设置 Evolver Token 截止。

每个 `input/agents/agent-vN/` 是完整只读 Bundle，可写 `candidate/` 使用相同布局。
`input/evidence/agent-vN/` 保存汇总及逐 Trajectory 补充资源；仅上一个完成 Epoch 的参赛者暴露
该 Epoch 的 Conversation 和 Attempt Report。历史创建报告位于 `input/evolution-reports/`。

Evolution Report 声明提案模式、所选 `kernel_agent_revision_id`、假设、预期效果、准确的 Bundle
相对 `changed_paths`、`contributing_paths` 和未实现能力。从历史派生时先复制完整 Bundle，再修改。
本地 `evolution-report` 校验 Draft，错误返回 `issues`、`request_schema`、`recovery`，不发布；
首次有效调用原子生成 `scratch/evolution-report.json`。Session 结束后 Runtime 再独立校验。

每个新 Revision 封存完整 Bundle 和六目录 State Checkpoint，记录 `optimizer_digest`、
`runtime_state_digest`；每个新 Trajectory 使用独立副本。Optimizer 的实现权限和 State 继承规则不变。

`contributing_paths` 记录实际吸收内容的、排序且去重的 Workspace 相对文件或目录路径，允许
`input/agents/agent-vN/` 和 `input/evidence/agent-vN/resources/`，包括 Parent 其他 Trajectory 的资源。
仅阅读和自动继承 Parent 不算贡献。路径必须存在、无链接或越界，且属于合格已评估历史或 Parent，
不能引用同 Epoch 尚未评估的 Challenger。`reuse` 要求 `[]`。Runtime 在 Evolution Trace 中保存归属
和准确内容快照；该字段不改变 Bundle Base 或 Revision 祖先关系。

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

版本化 Core Source 包含 `prompts/`、`memory/`、`knowledge/`、`skills/`、`tools/`、`hooks/` 初始种子，没有继承 State 时由 Runtime 复制。运行中积累的 Adaptive State 仍是独立 Artifact：

```text
runtime-state/
  trajectories/<N>/
    prompts/README.md
    memory/README.md
    knowledge/README.md
    skills/README.md
    tools/README.md
    hooks/README.md
```

Optimizer Workspace 把一条 Trajectory 的 State 展示为根级可写 `prompts/`、`memory/`、`knowledge/`、`skills/`、`tools/` 和 `hooks/`。
各目录 README 必须随内容的新增、修改、重命名和删除同步更新。六目录整体封存，遵循相同的继承与隔离
规则；没有继承 State 时，从固定 Core Source 加载六目录初始内容，重置状态的消融臂每次回到该种子。自适应 Knowledge 与 Source 内的实现文档相互独立。Runtime
封存终态内容并为下一个串行 Attempt 恢复。Evolver 获得冻结 Participant/Historical State，并在
`candidate/{prompts,memory,knowledge,skills,tools,hooks}/` 编写一份扁平 Candidate Seed。新 Agent Revision 同时
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
