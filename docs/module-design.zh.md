# 模块设计

[English](module-design.md) | 中文

## 仓库结构

```text
src/atrex_runtime/
  api/                     认证管理 API 与 ASGI Application
  cli/                     公开入口、Parser、命令族与进度输出
  composition/             共用 Gateway、Bootstrap、Campaign 与 Wiki 装配
  asgi.py                  共用受限 Body、Bearer 与 JSON Response 原语
  filesystem.py            共用私有目录树权限切换
  serialization.py         Canonical JSON 字节、Digest 与文件写入
  bootstrap.py             只接受 Commit 的 Campaign Bootstrap
  presentation.py          CLI/HTTP 共用 JSON Read Model 投影
  lineage_seed.py          以 Artifact/Revision 为种子的独立 Lineage 根
  config.py                严格部署配置
  dev_shell.py             不启动 Agent 的 Optimizer/Evolver 调试 Shell
  git_import.py            共用安全 Git 命令与 Archive 边界
  roofline.py              Commit 固定的可信 Atrex Bench Roofline Builder
  artifacts/               不可变本地 CAS
  controller/              Epoch、Campaign、Evidence、Fencing、持久 Task Worker
  gateway/                 Capability Model/Control、Agate Adapter/配置、Proxy
  kernel_agents/           完整仓库校验与 Core Git 导入
  knowledge/               Wiki 实时 Query 与 Epoch 后 Outbox
  registry/                SQLite 状态与 Event
  workers/                 Launcher、Workspace、Core/Evolver Adapter
src/atrex-kernel-agent-core/       独立 Git Submodule
src/atrex-kernel-agent-evolver/    独立 Git Submodule
third_party/atrex-bench/            固定的可信 Evaluator Git Submodule
local-wiki/                         仓库内的 GPU Wiki 服务与语料
```

## 关键模块

`kernel_agents/revision.py` 校验并封存完整 Core 仓库，只保留总文件数、总字节数和入口文件字节数三类真实生效的限制。`kernel_agents/git.py` 在不执行仓库内容的前提下导入精确 Commit 和显式批准的 Submodule。`git_import.py` 统一非交互 Git、受限 Tar、路径安全解包以及链接/特殊文件拒绝逻辑。

`bootstrap.py` 只支持 Campaign schema v3。Campaign 定义与 Runtime 部署配置分离，非空 `lineages` Map 是初始 DSL 选择的唯一来源。在启动任何 Agent Session 前，它会保留显式 Roofline，或让可选的 `roofline.py` Provider 从部署固定的 Atrex Bench Git Commit 执行标准 Converter，校验准确 Shape 覆盖，并把结果封存进共用 Evaluation Contract。一次操作只导入一次 Core，并让选定 DSL Revision 共享 Optimizer Digest 与来源 Provenance。Agent Problem 必须由输入提供或由 Core 问题泛化阶段生成；Campaign 级问题泛化 Model、部署选择的完整 Evolver Commit 和 Lineage 级 Optimizer/Evolver Model 会作为不可变恢复输入持久化。`CampaignScheduler` 会在持有 Lineage Fence 时校验冻结的 Evolver Commit。Core Baseline Generator 必须存在，不支持预生成 Gateway Result 或本地仓库路径。`gateway/control.py` 在 `bootstrap_runs` 中保留每次 Bootstrap 执行，并按 Attempt 与 Recovery Generation 联合隔离 Gateway Operation；`composition/bootstrap.py` 为每个被捕获的 Session 退出提交终态成功或失败证据。

Baseline Session 成功后，Runtime 会用不可变的 Epoch-0 Evidence Checkpoint 替换初始化输入。
Agent 可见的 `bootstrap/` 目录只含终态 `report.json` 与最新封存的后端无关
`conversation.jsonl`，与普通 Attempt 的精简历史表示一致；Runtime Metadata 保持内部可见。
除此之外，Bootstrap 是一种特殊 Attempt：它使用相同的 Direction/Experiment Journal、Runtime
查询工具及 schema-v12 `attempt-report`，但采用 Bootstrap 专用方法论且没有更早的 Lineage 历史。
其终态 Journal 与 Gateway 记录会成为后续 Optimizer Attempt 可见的根历史。

Bootstrap 会在 Roofline 解析或任何 Core 阶段之前，把 Runtime `gate_policy` 应用到封存 Contract；
Sampling、容差、Timeout、完整 Validation Mode、时钟 Policy、Evaluator Commit 与 Gate-owned
Runner 控制因而只有一个 Campaign 冻结的事实来源。同一 Contract 还会冻结
`production_gate`；`gateway/production_policy.py` 会在 Worker Eval、权威终评和 Artifact Seed
Lineage 发布前执行无状态内容检查。
`composition/gateway.py` 是该权威 Evaluator 的唯一构造入口，避免 CLI、Scheduler 与 HTTP
Bootstrap 的阶段设置或 Production Gate 接线发生漂移。

`gateway/proxy.py` 把 Worker 的每次 `evaluate` 视为探索评测，并封存准确 Candidate 与原始 Result。`gateway/private_results.py` 独立构造 Worker 返回：Evaluate 只给出不透明逐 Case 延迟与聚合状态，Profile 则递归移除私有字段；手工 Profile 只解析一个已封存私有 Shape。`gateway/control.py` 只追加这些记录，不占用 Attempt Outcome。`gateway/finalization.py` 只接受具有相同字节正确探索记录的被提名 Tree。Bootstrap 执行新的 Runtime-owned 终评；优化 Attempt 只用提名记录临时注册 Kernel，再在完成前由所选留存 Comparator 以 B 的普通 Evaluate 算术平均或 ABBA 几何平均替换其 Evaluation。管理 API 与 CLI 可查看准确原始 Artifact。
`gateway/candidate.py` 统一 Kernel Artifact 类型与路径校验；`asgi.py`、`filesystem.py` 和
`serialization.py` 则保证各个独立入口具有一致的传输、权限和持久 JSON 行为。

`lineage_seed.py` 直接或通过历史 Revision ID 解析一对已封存 Agent/Kernel Artifact，校验同
DSL 的完整 Agent Bundle，并创建稳定且独立的根身份。`gateway/lineage_seed.py` 在发布
`agent-v0`/`v0` 前执行目标 Campaign 下必需的 Agate 评测。来源 Revision 关联只属于
Provenance；新 Lineage 持有独立版本树并从 Epoch 1 开始。

`workers/core_phase.py` 是 Optimization Attempt、Problem Generalization 和 Framework Baseline 共用的命令解析、Sandbox 启动、受限进程、Token Report 与 Session Trace 获取路径。`workers/launcher.py` 构造 systemd cgroup 与 bwrap 挂载/进程边界，把所有宿主 Workspace 路径映射到 `~/workspace`，并有意保留宿主网络；它还串行化并发 Host Check，并要求 systemd 直接以非 root Worker 创建 Workspace Root/Probe，从而在 virtiofs 上维持正确 Owner。`workers/core.py`、`problem_generalization.py` 与 `lineage_bootstrap.py` 只持有各阶段环境和输出 Schema。

`workers/workspace.py` 还会在物理 Attempt Run 目录之外维护可写的 `skills/` 与 `tools/`
快照。它们按 Lineage、Agent Revision 和 Trajectory 建立作用域，每次新建 Optimizer Workspace
时恢复，并在 Session 退出时原子发布。子 Agent Revision 首次使用时继承父 Revision 的对应快照；
`tools/README.md` 为必需文件，用于说明每个持久工具的使用约定。
Framework Bootstrap 会发布 Revision 级初始快照；`agent-v0` 的每条新 Trajectory 从该快照开始，
随后独立演进。

`registry.worker_sessions` 是 Optimizer Attempt、Framework Baseline、Problem Generalization 与 Evolver 的统一物理进程目录。Workspace 准备完成后、获取 Authority 或启动进程之前，Runtime 先提交 Running 记录；随后只允许一次终态更新，记录 Completed、Failed 或 Timed Out，以及可用时未经修改的封存 Session Trace、Provider Token 用量、进程状态、诊断和稳定的 Workspace/Run 身份。由于 Generalization 发生在 Campaign 创建之前，上下文字段允许为空；已有 Attempt/Epoch 身份会自动展开为 Campaign/Lineage 上下文。该目录补充 `attempt_session_traces`、`bootstrap_runs` 等角色专用协议记录，不替代它们承载的领域 Evidence。

`dev_shell.py` 复用 `workers/core.py` 的同一 Launch Preparation，为现有或新建的首个 Active
Attempt 物化真实 Workspace、签发同范围 Capability 并启动交互式 `zsh/bash`，但不执行 Core
入口。该调试入口持有 Lineage Fence，退出后保留 Workspace 与 `running` Attempt，不生成假的
Agent Trace、Token Report 或结果。

同一模块还可以根据必填的 Lineage ID 和已有绝对 Epoch 编号重建 Evolver Workspace。该快照使用
Epoch 记录的 Parent Agent 与 Evidence Checkpoint，包含此前 Kernel 历史以及已经挂到目标 Epoch
的 Challenger，并排除目标及未来 Epoch 产生的 Kernel。它准备与正式 Evolver 相同的环境，但不
创建 Challenger、不执行 Agent、不晋升，也不生成 Token Report。

`workers/evolution.py` 创建固定的
input/agents/evidence/historical/candidate/scratch Workspace 与 Runtime-private 控制文件，并只接受
相同 DSL 的完整仓库变更。它冻结精确的 Lineage 内 Agent/Kernel 版本 Catalog、全部历史 Kernel
Artifact，以及按 Agent Revision/Trajectory 划分的只读 `skills/`、`tools/` 快照，供 Evolver
比较。`input/agents/` 只包含当前 Active 与本 Epoch Challenger；`input/evidence/<role>/` 保存实测
优化效果，并按 Trajectory 保存该 Agent 最近完成 Epoch 中每个 Attempt 的一份 Conversation；Bootstrap
与更早 Epoch 的 Conversation 只保留在 Runtime 私有历史中。`input/historical/agent-vN/` 将每个非当前版本的源码、效果汇总和
运行时状态放在一起。有序的 `input/evolution-reports/evo-N.json` Wrapper 暴露可用的 Agent 创建报告及
Source Base/产出 Agent 路径，不暴露完整 Evolution Trace。当前运行时状态位于 `input/agents/` 对应角色源码旁边，详细 Epoch Tree 保留在已有
Runtime Evidence Store 中，不再复制进 Evolution Workspace。每份效果汇总分别呈现最近完成 Epoch 的
Attempt 正确性与最佳 Kernel 逐 Shape Gateway 结果，以及累计 Epoch 参赛、获胜和失败次数。运行时状态与
版本化 Source 分离；Candidate 具有可写 `source/` 和唯一一份 Revision 级
`runtime-state/{skills,tools}/` 公共种子。根级 `skills/` 和 `tools/` 在 Source 中仍会被拒绝。Evolver
可直接整理公共种子，也可修改消费它的版本化机制；Runtime 独立封存完整 Source 与 State，把两个
Digest 组成逻辑 Bundle，并把准确 State 复制给所有新 Trajectory。Evolver 不获得 Runtime Tool 或 HTTP Capability。
从历史派生时，它复制所选历史 Source，并可从可见历史状态整理公共种子；Runtime 验证声明的 Agent
Revision、报告的 Source Diff 和私有 State Diff。
Runtime 用固定 stdin 指令启动 Commit 固定的 Evolver。
Provenance 记录 Commit、Tree、封存 Artifact Digest、argv Digest、环境变量名、进程结果、Token、
Session Trace、输出注释和 Candidate Digest。Provider、Model 与 Prompt 属于 Evolver 仓库，
不属于 Runtime 配置。

`controller/epoch.py` 实现可配置的 Epoch 拓扑：串行创建零个或多个 Challenger，把逐步增长的
Lineage Agent Catalog 暴露给每次 Evolver 调用，在每个 Branch 内并发执行独立 Trajectory，并
在每条 Trajectory 内串行执行 Attempt。`controller/attempt_evidence.py` 按 Trajectory 隔离增量
记忆；`controller/evidence.py` 只在完成选择后发布包含全部测量结果的 Epoch。

`controller/projection.py` 只输出包含 Session 来源 Digest 的有界规范化摘要，并明确排除未脱敏的
`conversation.jsonl`。`workers/session_trace.py` 在每个 Core 或 Evolver Session Artifact 封存前
应用权威保留策略：从 Provider stdout 和对话中移除高频 Claude `system/thinking_tokens` 估算遥测，
权威 Usage 继续保存在 `events.jsonl`。对于 Optimizer 可见的每个历史 Attempt，
`workers/evidence_view.py` 只按最新封存 Session Digest 投影一个与 Runtime Final `report.json`
平级的 `conversation.jsonl`；全部重试 Artifact 仍保留在 Runtime 存储中。Evolver Evidence 继续保留
更丰富的全分支诊断视图。`knowledge/ingest.py` 在 Epoch 结束后独立构造
有界的保留 Session 上传 Projection，并应用同一兼容过滤。

## 稳定接口

可替换边界使用 Python Protocol：Artifact Store、Registry、Measurement Runner、Comparator、Optimizer/Evolver Runner、Worker Launcher、Lineage Lease Manager 与 Wiki Query Client。生产实现是 `BwrapSandboxLauncher`。`CleanEnvironmentLauncher` 只存在于显式 `launcher.mode=development` 下，不声明隔离能力。普通 Optimizer/Evolver Session、Bootstrap/Generalization 阶段和两个 dev-shell 入口全部使用同一个已配置 Launcher。

所有持久 ID/Digest 都经过校验；Registry Transition 幂等、生命周期 Event 追加写，Scheduler 写入使用可续租 Generation Fencing。Kernel 保留与 Agent 晋升可以独立选择普通 Evaluate 或同 Allocation ABBA。后者为每个 Shape Batch 在一个受信任 Dev Job 中上传 Commit-pinned Evaluator 与两个 Kernel Snapshot，校验完整交错 Schedule，并把每轮结果写成 Kernel Measurement。
