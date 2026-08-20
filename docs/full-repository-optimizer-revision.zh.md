# 整仓 Optimizer Revision

[English](full-repository-optimizer-revision.md) | 中文

## 决策

`git@github.com:guoyuqi020/atrex-kernel-agent-core.git` 的一个固定 Commit 是完整的 Optimizer Base Revision。Runtime 不再把该仓库重新封装成独立版本化的 Prompt、Workflow、记忆策略、Tool 或 DSL Overlay 组件。

仓库 Commit 和不可变 Artifact Digest 承担不同职责。Git 标识上游源码，并支持同步与评审；Artifact Digest 标识移除 Git Metadata、完成文件策略校验后，某条 Lineage 实际使用的精确源码快照。

## Runtime 装载

Campaign 启动前，可信控制面执行以下操作：

1. 拉取部署允许的仓库；
2. 解析并校验明确指定的完整 Commit SHA；
3. 导出不含 `.git` Metadata 的完整 Tracked Tree；
4. 拒绝未解析 Submodule、符号链接、特殊文件、超限内容和无效 Runtime 入口 Manifest；
5. 把完整目录树封存成一个不可变 Optimizer Artifact；
6. 把仓库 URL、Commit SHA、Source Tree Identity 和 Artifact Digest 注册为 Base Revision 来源。

Branch 和 Tag 只用于同步，不能作为生产版本身份。上游 Branch 前进时，已有 Lineage 不会发生变化。

## Runtime 入口 Manifest

Core 是上游仓库的维护型 Fork，而不是路径完全兼容的 Overlay。根目录的小型 Manifest 声明框架
无关的仓库命令；Manifest 不枚举全部 Payload，因为完整 Tracked Repository 就是 Revision。

该命令负责 Core 的 Agent 框架选择以及 Prompt/Workflow Assembly。Manifest 不包含框架专用 Adapter 区块，也不能定义凭据、Gateway/Wiki 权限、挂载、网络权限、评测规则或晋升策略。Runtime 校验命令路径，并单独注入可信任务上下文与范围受限的 Capability。

## Optimizer 执行

Runtime 在每个全新 Attempt 中只读物化完整 Optimizer Revision，通过部署持有的命令前缀执行 `entrypoint.command`，并通过版本化 Attempt Manifest 与环境协议传递可信 Campaign、Lineage、Epoch、DSL、硬件、Evaluation Contract、Evidence、Gateway、Wiki、Report 和 Token Budget 上下文；Core 自己选择并调用 Agent 框架，Runtime 不存在另一条框架专用启动路径。

Runtime 入口必须只执行一个 Attempt。旧的外层 Campaign CLI 不会自动满足该协议，也不得再启动另一层 Campaign、创建 Git Worktree、启动本地 Gateway、管理 Canonical Memory 或执行晋升。即使迁移期间 Optimizer Revision 中暂时还保留相关源码，这些职责仍属于 Runtime。

## 自进化

独立版本化的固定 Evolver Bundle 获得只读 Parent Optimizer 仓库、已完成 Epoch Evidence，以及一个初始由 Active 填充的可写整仓 Candidate。选择 `evolve_from_history` 时，受约束的 Runtime Tool 可以把 Candidate 原子切换到已完成 Lineage 历史；最终封存会校验记录的 Base 与真实 Diff。Evolver 可以修改任何通过校验的 Optimizer 自有文件，包括 Agent 框架选择、Prompt、Skill、Workflow 和 Helper Tool。Runtime、Evolver 代码和策略不在 Candidate 中，因此无法被修改。递归 Evolver 自进化仍然推迟，并需要独立评测与晋升设计。

只有完整仓库快照通过文件策略、Manifest 校验、容量限制及独立 Active/Challenger 评估后，Candidate 才能被接受。成功的 Lineage Revision 仍只属于该 Lineage；Runtime 不会向上游 Core 仓库 Push、Merge、Rebase 或创建 Ref。

把成功的进化 Revision 贡献回上游属于独立、受评审的导出流程。上游同步同样只会产生新的候选 Base Commit，不会修改已有 Lineage Revision。

## Core 简化策略

Core 通过 Git 历史记录上游 Base Commit；后续同步是把上游改动经评审 Merge 或 Patch-port 到当前
`src/`、`prompts/` 与 Manifest 布局。Runtime 不自动执行同步。删除 Runtime 已持有控制面代码所
造成的结构差异是明确接受的；每次上游更新必须重新执行 Core 单元/静态检查与 Runtime 协议测试，
才能形成新的 Base Commit。

| Core 现有职责 | 处理方式 |
| --- | --- |
| Episode Prompt 与 Kernel 优化 Workflow | 合并到根部 `prompts/`；Runtime 操作使用真实的连字符 Core CLI 命令 |
| 有用的结果分析知识 | 仅在兼容 Runtime 结构化结果和固定可写目录时保留 |
| Agent 实现与 Backend Adapter | 维护在 `src/`，每个 Backend 都属于可进化仓库 |
| `long_horizon/` Episode、Journal、Handoff 与比较代码 | 持久语义迁移到 Runtime Evidence/Registry 后删除 |
| Git Worktree、Campaign 调度、Process/Session 管理 | 从 Core 删除，由 Runtime 持有 |
| 本地 Gateway 与旧 Sandbox Transport | 替换为框架无关的 Core `gateway-execute` 协议客户端；所有 Backend 使用相同 Runtime 契约 |
| 本地 GPU Wiki 与直接 Query Script | 替换为 Core `wiki-query` 协议客户端；Wiki 内容属于外部服务 |
| Canonical Memory 修改和 Iteration Marker Script | 替换为 Runtime Evidence、结构化 Experiment/Report 协议与所选 Agent 框架的 Trace |
| 评测 Adapter 与隐藏输入处理 | 替换为 Runtime Evaluation Contract 和 Gateway |
| Reference Project Checkout 与嵌套依赖 | 除非某个 Helper 确实需要且 Runtime 策略允许，否则删除 |

只有 Core Entry Command 决定实际 Optimizer 行为；不属于维护 Bundle 的历史上游代码通过 Git
History 查阅，不再作为失活源码保留。
