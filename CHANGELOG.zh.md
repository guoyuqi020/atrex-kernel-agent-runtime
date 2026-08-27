# 变更记录

[English](CHANGELOG.md) | 中文

Atrex Kernel Agent Runtime 的重要变化记录在这里。

## 未发布

- 增加常驻生产控制面、受管多 DSL Campaign Task 与逐 DSL 检查脚本。
- Sandbox Host 准备支持并发，并通过以非 root Worker 直接创建 Root/Probe 兼容 Lima virtiofs。
- 明确记录 Worker 共享宿主网络的边界。
- 在权威 Session 封存和 Agent Evidence 中移除高频 Claude `system/thinking_tokens`
  估算遥测，同时保留最终 Usage 记录。
- 移除 GPU Wiki Feedback 的生成、持久化、投递和接收；GPU Wiki 现在仅提供知识查询。
- 新生产 Campaign 准备会拒绝不干净的 Core/Evolver Worktree，确保固定 Commit 准确标识
  Agent Bundle 源码。
- 固定版本的上游 GPU Kernel 项目作为每个 Attempt 与 Framework Baseline Workspace 的
  `reference/` 目录提供，在两种 bubblewrap 模式下从 `reference_projects_root` 只读挂载。
- Attempt Manifest 升到 schema 8，并不再在其中发布 Workspace 布局。布局在两端都是固定的，
  且已由 Agent Prompt 说明，序列化它只是让一张写死的表和另一张写死的表互相比较，同时把每次
  布局调整都变成破坏性协议升级。按更早 schema 注册的 Kernel Agent Revision 不再能启动，
  已有 Lineage 需要重新 Bootstrap。
- 修复 Artifact 封存会静默丢弃空目录的问题。Runtime State 封存时在本地校验 `skills/` 与
  `tools/` 均存在，但 Manifest 只记录文件，因此没有保存任何 Skill 的 Agent 会产出缺少
  `skills/` 的 Artifact，导致下一个 Epoch 的 Evolver 拒绝胜出 Trajectory 的状态。Manifest
  现在记录无子目录，且在不存在时完全省略该键，以保证已封存 Artifact 的 Digest 不变。缺少两个
  目录之一的 Seed 现在被接受而非拒绝，因为 Payload 不可变，且所有消费方本就会重建这两个目录。

- 移除无法到达的 Gateway `submit` 与 `sol` 操作。两者既未注册进 Agent 请求分发表，也不在部署
  操作白名单中，属于死协议面。SOL Profile 不受影响，仍通过 `profile` 的 `level="sol"` 使用。

## 0.1.0 - 2026-08-20

- 单节点可信 Runtime 的首个发布候选版本。
- 支持固定 Commit 的 Core/Evolver 导入、Campaign Bootstrap、Artifact Seed Lineage、可配置 Epoch
  拓扑、Agent/Kernel 版本历史和可恢复调度。
- 支持探索性 Gateway Operation、权威普通 Evaluate/同 Allocation ABBA Gate、Production Gate、
  私有 Evaluation Contract、Roofline 构建和 NCU SOL Fallback。
- 支持返回前冻结的实时 GPU Wiki Query。
- 支持 Claude、Codex、QoderCLI、Pi Backend，保留原始 Session 和 Provider Token 统计。
- 提供 Development Launcher 与 Linux bubblewrap/cgroup-v2 沙箱；Worker 共享宿主网络。
- 提供认证 Administration API、CLI Inspect、恢复、Event、Task 和离线保留。
