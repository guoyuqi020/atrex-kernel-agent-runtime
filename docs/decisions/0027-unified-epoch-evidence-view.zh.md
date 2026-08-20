# ADR 0027：统一、按 Epoch 组织的 Evidence View

[English](0027-unified-epoch-evidence-view.md) | 中文

## 状态

已接受并实现。

## 背景

Runtime 持久化生命周期和 Digest 不同的累计 Epoch 起点 `EVIDENCE` Checkpoint 与同
Trajectory 的 `ATTEMPT_EVIDENCE` Snapshot。若把这些存储对象作为多棵目录树直接暴露，会把
控制面机制泄漏给 Agent，也会迫使 Prompt 自行合并重叠历史。

## 决策

Runtime 保持来源 Artifact 独立，但把它们投影成根位于 `input/evidence` 的单一只读
`EvidenceViewManifestV1` Tree。`manifest.json` 绑定 Agent Role、Lineage Checkpoint、最后完成
Epoch、可选进行中 Snapshot 和可见性策略。`bootstrap/` 与 `epochs/NNNNNNNN/` 构成一条时间线。

对 Optimizer，已完成 Epoch 只包含晋升 Agent Revision 的 Attempts，当前 Epoch 只包含所选
Trajectory 中较小 Ordinal 的 Attempt。对 Evolver，已完成 Epoch 按 ADR 0035 包含全部已完成
Active 与 Challenger Branch。

Optimizer 的每个 Epoch 直接包含 `attempts/`，不存在 Agent 可见的 `branches/` 目录或
Active/Challenger 任务字段。每个 Attempt 目录可包含可信 Summary、受限 Kernel Diff、结构化
Report 和完整原始 Session Artifact 目录。Runtime 会解析被选中的来源 Digest，不脱敏、不过滤、
不改写地物化 Trace 目录树。Agent 与 Evolver Annotation 继续明确标记为不可信。Snapshot 选择是
Manifest 元数据，绝不成为独立 Evidence Root 或来源 Digest 的替代品。

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
