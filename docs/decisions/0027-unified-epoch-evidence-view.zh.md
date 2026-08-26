# ADR 0027：统一、按 Epoch 组织的 Evidence View

[English](0027-unified-epoch-evidence-view.md) | 中文

## 状态

已接受并实现。

## 背景

Runtime 持久化生命周期和 Digest 不同的累计 Epoch 起点 `EVIDENCE` Checkpoint 与同
Trajectory 的 `ATTEMPT_EVIDENCE` Snapshot。若把这些存储对象作为多棵目录树直接暴露，会把
控制面机制泄漏给 Agent，也会迫使 Prompt 自行合并重叠历史。

## 决策

Runtime 保持来源 Artifact 独立，但把获授权数据投影成根位于 `input/evidence` 的单一只读 Tree。
内部 `.runtime/evidence-manifest.json` 绑定 Agent Role、Lineage Checkpoint、最后完成 Epoch、可选
进行中 Snapshot 和可见性策略；`.runtime/evidence-instructions.md` 经校验后注入 Prompt。二者都不
属于 Agent-facing Evidence Tree。`bootstrap/` 与 `epochs/NNNNNNNN/` 构成一条时间线。

对 Optimizer，已完成 Epoch 只包含晋升 Agent Revision 的 Attempts，当前 Epoch 只包含所选
Trajectory 中较小 Ordinal 的 Attempt。对 Evolver，已完成 Epoch 按 ADR 0035 包含全部已完成
Active 与 Challenger Branch。

Optimizer 的所有 Epoch 统一使用 `trajectories/<ordinal>/attempts/<ordinal>/` 层级，不存在 Agent
可见的 `branches/` 目录或 Active/Challenger 任务字段。同一 Trajectory 内 Attempt 串行执行，并从
最近一次保留的 Kernel 继续；Sibling Trajectory 则从同一个 Epoch 起点 Kernel 独立并行搜索。
Epoch 自身也构成串行链：Bootstrap 为 Epoch 1 提供初始状态；每个已完成 Epoch 会彼此独立地选择
下一 Epoch 的 Active Agent Revision 与最佳保留正确 Kernel。因此二者不一定来自同一个生产分支；
若没有 Candidate 改善 Kernel，则原起点 Kernel 原样传递到下一 Epoch。每个
可见 Attempt 目录只包含 Runtime Final `report.json` 和最新封存的 `conversation.jsonl`。Runtime
存储仍保留全部 Session、Kernel、Trial、Gateway Result、Summary 与 Diff Artifact；精确数据通过
Runtime Tool 恢复，不再复制进 Optimizer 文件系统。Agent 与 Evolver Annotation 继续明确标记为
不可信。Snapshot 选择是 Manifest 元数据，绝不成为独立 Evidence Root 或来源 Digest 的替代品。
Evolver Evidence 继续保留更丰富的全分支诊断视图。list/load 工具使用的跨分支 Direction 与
Experiment Journal 会立即追加到 Runtime 自管的 Attempt 表；终态 Report Artifact 冻结 Handoff
快照并作为旧数据兼容回退。Journal 由 Attempt-scoped Runtime Query
按需解析；Workspace 不生成 Journal 历史投影。Evolver Trace 绝不投影到 Optimizer Evidence。

Optimizer 的 Epoch 目录只包含 `trajectories/`：Runtime 不投影 Epoch `summary.json`、
`lessons.json` 或 `measurements.json`。调度状态留在 Registry，当前 Kernel 以 `input/kernel/` 为
权威，精确测量继续绑定 Gateway Result Artifact。Evolver Evidence 为诊断保留更丰富的聚合文件。

Runtime 在 View Manifest 旁写入唯一的 `instructions.md` Prompt Fragment。该 Fragment 描述准确
结构与读取规则，但不给 Layout 单独命名版本。Manifest 绑定其 SHA-256，每个进程获得固定路径。
Core 与 Evolver 仓库保留通用校验和拼接 Hook，因此结构文案只有一个 Runtime 持有的来源。

Optimizer 与 Evolution Input Manifest 都绑定 `input/evidence`；Runtime 保留独立来源 Digest
用于恢复和来源追踪，并按接收 Role 校验投影后的 View。

## 后果

- Optimizer 与 Evolver 使用同一 Evidence Contract 和同一时间线 Root；
- Registry 恢复、幂等、Artifact 保留和 Trajectory 隔离继续使用独立来源身份；
- Active/Challenger 仍是可信调度和晋升概念，但不会出现在 Optimizer 文件系统、Manifest Task
  Context 或 Prompt 中；
- Agent 可见 Trace 可能包含 Backend 捕获的 Prompt、Reasoning、工具输入输出、命令输出、凭据和
  其他敏感内容；
- View 是确定性 Projection，可从持久来源 Artifact 重建。
