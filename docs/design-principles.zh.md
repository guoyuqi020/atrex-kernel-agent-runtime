# 设计理念

[English](design-principles.md) | 中文

Atrex Kernel Agent Runtime 的目标，是允许 GPU Kernel Agent 修改自身，同时不让正在变化的组件掌握
评测、晋升、隔离或历史记录的最终权力。因此，系统最核心的设计原则是：

> Agent 负责提出并解释修改；Runtime 负责身份、执行、测量、比较、持久化与晋升。

本文解释系统为什么采用当前形态。[架构](architecture.zh.md)描述由此形成的组件与生命周期，
[协议](protocols.zh.md)定义对应的持久契约。

## 即使不考虑 Agent 进化，为什么仍需要 Runtime 控制面

Agent 进化并不是将 Runtime 从 Kernel Agent 中独立出来的最初理由。即使 Optimizer 永远固定，当一个
长期优化 Harness 主要通过 Prompt 与工作区文件表达控制、记忆、测量和可观测性时，也需要可信控制面。

固定在 `third_party/atrex-kernel-agent/` 的上游 Atrex Kernel Agent，相比早期 clean-session
Workflow 已经有了明显改进。当前 Long Horizon Engine 要求 Agent 在决定性实验发生后立即追加 Episode
Journal，刷新非权威 `memory/live.json`，由 Supervisor 构造 canonical `memory/vN.json`，并对终态
Candidate 进行独立验证。这些改进降低了 Session 结束时集中回忆造成的信息损失，也避免直接使用 Agent
声明决定最终晋升。但它们没有让 Journal 自动成为实际执行过程的完整记录。

### Prompt 驱动 Harness 的局限

1. **Workflow 由强制指令表达。** Fast 和 Full Episode 都预先规定 Planning、Review、Profile、
   Implementation、Evaluation、Recording 与 Handoff 协议。即使某个阶段对当前算子、DSL 或已有 Evidence
   的价值有限，Agent 仍需承担对应的指令理解与执行成本。继续增加恢复和证据规则，也会扩大 Agent 必须
   正确理解的协议表面。
2. **跨 Session 认知仍以摘要为中心。** Canonical Memory、Plan、Profile 和归档 Journal 虽然都存在于
   磁盘，但下一个 Episode 仍主要通过压缩后的 `memory/vN.json` 重建历史。详细 Evidence 被保存下来，
   却没有自然形成统一、按需查询的 Direction、Kernel Trial、测量与失败历史。
3. **Journal 完整性仍依赖 Agent。** 实验结束后，Agent 必须主动判断它是否值得记录、解析 Evaluator
   输出、构造结构化 `evaluation` 字段并调用 Journal CLI。Agent 一旦忘记、崩溃、超时、在上下文压缩中
   丢失早期事件或错误转述结果，Journal 与它的 `memory/live.json` 镜像都不会包含该事实。
4. **Recovery 修复协议，不能恢复从未记录的 Evidence。** Same-session Handoff Recovery 和中断
   Worktree Recovery 都有价值，但无法还原从未被捕获的 Kernel 修改、测量或失败。执行虽然可以恢复，
   系统仍可能只保留一份不完整的到达路径说明。
5. **中间事实与 Agent 解释没有完全分离。** 终态晋升可以依赖 Supervisor Verification，但 Episode
   Journal 中间的正确性、延迟、瓶颈和 Decision 字段，仍是 Agent 对工具输出的转述。格式合法的报告并
   不能证明它忠实表达了原始结果。
6. **运行中观测仍由文件更新驱动。** Active State、Live Memory、Journal 与 Telemetry 改善了可见性，
   但只有对应生产者更新时才会反映进度。它们自身不能提供一个将完整 Session Trace、每次 Runtime Tool
   调用、精确 Kernel、Gateway Result、Experiment 与最终选择关联起来的持久视图。

这里的架构结论不是禁止 Agent 编写 Plan 或 Analysis，而是：Agent 编写的 Report 不能成为某项执行事实
唯一存在的位置。

### 在执行边界自动捕获事实

Runtime 在授权操作发生时、将控制权交还 Agent 之前持久化事实：

| 事件 | Runtime 持有的持久事实 | Agent 提供的解释 |
| --- | --- | --- |
| 提交 Kernel | 精确 Kernel Artifact、Digest、Attempt、Session 与时间戳 | 修改意图与 Hypothesis |
| Evaluate 或 Profile | Operation、请求绑定、Gateway Result Artifact、归一化正确性与逐 Shape 测量 | 瓶颈分析与结果意义 |
| Tool 失败 | Operation、稳定错误分类、响应与重试历史 | 故障诊断与修复方案 |
| Experiment | 不可变的修改前后 Kernel 与 Gateway Result 引用 | Analysis、Lesson 与下一步 Decision |
| Attempt 完成 | 保留的 Session Trace、Journal Snapshot 与被提名的已测 Candidate | 终态叙述与 Open Direction |
| 晋升 | Runtime Comparison、Gate Outcome、祖先关系与被选 Revision | Agent 无权单方面决定 |

Agent 仍然可能遗漏或修改 Analysis，但它不能让已经执行的测量消失、替换原始数值，也不能仅凭一份有说服力
的 Report 产生晋升。因此，终态 Report 是对已经持久化事实的校验视图，而不是这些事实的来源。

这层分离在启用任何自进化能力之前就有价值；后续 Agent 进化建立在同一边界之上，而不是重新发明边界。

## 1. 控制面保持稳定，Worker 可以进化

Runtime 是普通的可信 Python 控制面，不是 Agent。它执行确定性调度并持有策略。Core 与 Evolver 是
Commit 固定、以不可信 Worker 运行的 Agent Bundle。

Optimizer 可以修改 Kernel 源码以及自适应 `prompts/`、`memory/`、`knowledge/`、`skills/`、`tools/`、`hooks/`；Evolver 可以修改 Optimizer 的
源码、Workflow 与自适应 Runtime State。但两者都不能修改 Registry、读取私有验证输入、为自己签发
Capability、决定自身晋升结果或重写历史。

这条边界既允许 Agent 大范围进化，又保证故障可恢复、结果可比较。

## 2. 分离提案、测量与晋升

Kernel 生成、GPU 测量和版本晋升属于不同权力：

1. Optimizer 提出 Kernel 和分析；
2. Agate 执行 Runtime 构造的请求；
3. Runtime 记录精确 Kernel Trial 与 Gateway Result；
4. Runtime 应用配置的比较策略，决定是否保留 Kernel Revision；
5. Runtime 独立决定哪个 Agent Revision 赢得 Epoch。

Gateway 测量是持久事实；Agent 的解释、计划与提名只是 Evidence，不是 Authority。Runtime 可以保留
Kernel 而不晋升其 Agent；Agent 比较也可以采用 Branch 产出的最佳 Kernel，而不是最后一次 Attempt。

## 3. 局部进化，而不是全局晋升

每个算子 Campaign 中的每种 DSL 都拥有独立 Lineage。CUDA、Triton 与 CuteDSL 可能需要不同的
Prompt、Tool、搜索策略和累积 State；一个 DSL 上的改进并不意味着可以安全升级全局 Agent。

局部进化也使归因更清楚：每个 Agent 与 Kernel 版本都绑定一个不可变的算子、硬件目标、
Evaluation Contract、DSL 与祖先关系。跨 Lineage 复用必须显式创建新 Seed，而不是发生不可见的
全局晋升。

## 4. 使用时间尺度不同的两层循环

内层循环优化 Kernel。每个 Attempt 都是全新的 Optimizer Session，可以探索并测量多个 Kernel
Trial，最后提交一份终态报告。

外层循环优化 Agent。在 Epoch 边界，Evolver 分析已完成的 Conversation、Outcome、Agent Source
和可复用 State，然后生成 Challenger。Active 与 Challenger 在下一个受控竞争中从相同的 Kernel
和 State 起点开始运行。

因此，Agent 进化是基于完整 Evidence、按计划触发的实验，而不是在一个失败 Session 内临时修改
自身。

## 5. 使用全新上下文和持久状态

Attempt 重试会创建新的物理 Session。Runtime 不把模型上下文当作 Lineage 的持久记忆，而是在
Session 外保存真正需要继承的数据：

- 不可变 Agent 与 Kernel Artifact；
- 精确 Kernel Trial 与 Gateway Result；
- Direction、Experiment、Attempt 与 Evolution Report；
- 保留的 Conversation 与 Provider Usage；
- `prompts/`、`memory/`、`knowledge/`、`skills/`、`tools/`、`hooks/` 中的自适应 Runtime State，各自维护 README 索引。

这样可以避免上下文污染，使每次重试可观察，并避免把无限增长的历史对话重新塞进每个 Prompt。

## 6. 状态只追加，恢复必须显式

内容寻址 Artifact 永不修改。逻辑版本号 `vN`、`agent-vN` 只是 Lineage 内不可变记录的标签。
Creation Key 保证重复命令幂等；Lease 与 Fence 防止两个 Scheduler 提交同一 Transition。

失败的 Session、Bootstrap Generation 或 Epoch 都保留在历史中。恢复通过增加新的 Generation 或
Session 推进，而不是修改失败记录。Active 会一直保持可用，直到经过测量的 Challenger 胜出，
因此自进化始终存在回滚点。

## 7. 暴露训练域，隐藏验证集

Agent 可以看到公开算子 Contract 与 `shape_train` 训练域，但看不到精确验证 Shape、Reference/Input
实现、私有 Metadata 或 Roofline 输入。Agent 可见的 Shape ID 是不透明编号。

这既提供了进行泛化优化所需的结构，又避免直接记忆验收 Case。Runtime 从封存的私有 Evaluation
Contract 构造正确性、性能、Production Gate 与晋升请求。

## 8. 提供受限工具，而不是环境级权力

Worker 只获得 Attempt 级 Capability 和一个 Workspace。Runtime Tool 会把请求绑定到当前 Attempt，
并只暴露被授权的历史。Agate 与 GPU Wiki 都是由 Runtime Client 和投影控制的外部服务。

GPU Wiki 只提供外源知识。Agent 产生的 Experiment、Conversation 与自适应 State 属于各自 Lineage，
不会被静默聚合进 Wiki。

## 9. 没有证据时不引入可进化 Coordinator

Campaign 调度、并发、重试与晋升属于确定性的控制面工作，应留在 Runtime。Evolver 已经持有高价值
的元层决策：Optimizer 应当如何改变。再增加 Coordinator Agent 会增加成本、可变 Authority 和
归因难度，却尚未对应一个独立、可测量的优化问题。

只有当 Coordinator 拥有明确可测目标，且 Runtime 能独立评测其有限 Authority 时，才应引入。

## 设计取舍

| 选择 | 收益 | 成本 |
| --- | --- | --- |
| Commit 固定的 Agent Bundle | 源码可复现，进化受控 | 更新 Agent 必须产生新 Revision |
| 每个 Attempt 使用全新 Session | 上下文干净、故障隔离 | 增加进程启动与显式 State 管理 |
| Lineage 内局部晋升 | Agent 专业化、归因清楚 | 不会自动进行全局知识迁移 |
| Active/Challenger 竞争 | Agent 进化可回滚 | 增加评测成本 |
| 私有验证 Contract | 降低过拟合风险 | Agent 无法直接调试精确隐藏 Case |
| 不可变 Artifact 与只追加历史 | 可审计、可恢复 | 需要额外存储和生命周期机制 |
| Runtime 持有比较权 | 选择结果可信 | Agent 不能单方面宣布成功 |

## 明确不做的事情

当前 Runtime 不负责多节点调度、跨 Campaign Memory 聚合、全局 Agent 晋升、Evolver 递归自进化，
也不会把信任边界交给 Agent。这些能力只有在 Authority、Evidence 与恢复语义都明确后才应加入。
