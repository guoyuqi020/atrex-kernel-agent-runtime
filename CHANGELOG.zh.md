# 变更记录

[English](CHANGELOG.md) | 中文

Atrex Kernel Agent Runtime 的重要变化记录在这里。

## 未发布

- Agent Revision 现在可携带可执行的 `workflow/main.py` Epoch 编排程序。Runtime 通过 Worker 隔离
  边界运行 Active Revision 的程序。Agent 代码只实现一个 `run_epoch(epoch)` 函数，以 Pool 和同步轮次
  组织工作；SDK 隐藏 Attempt 序号与底层协议，同时保留根据结果路由 Kernel/State 的能力。Runtime 强制
  精确 Attempt 预算、保证轮次幂等恢复，并继续独占跨 Epoch 调度、评测、Gate、晋升、回滚与 Registry
  权限。

- 生产与消融 Lineage 现在使用 Runtime 持有的臂模板构造初始 Agent Revision。只有被选择的程序会被
  封存为 `workflow/main.py`，无关臂程序不会暴露给 Optimizer 或 Evolver。`evolve-3`、Isolated、
  Retained、Pool-3 与 Pool-Retained-3 仍分别执行独立版本化程序，同时共享受控的 Optimizer Source
  与完全相同的 Bootstrap Kernel。

- 默认通过 HTTP 接入官方 Agate localhost 后端（`127.0.0.1:8000`、GPU `local`）。配置生成和
  CLI 示例保留显式地址覆盖与服务端 AK/SK 鉴权；无鉴权 loopback 部署可不设置凭据。
  Runtime/Wiki 脚本不管理 Agate 服务。

- Agent 完整 Evaluate 与权威 ABBA 改为每个 Shape 执行一次测量，不再额外执行三次完整调用并
  逐 Shape 取中位数。单 Job 内 GPU Benchmark 采样及配置的 ABBA Schedule 不变；结果标记为
  `single_measurement`、`repetitions=1`。

- 新 Campaign 用固定种子 `42` 随机封存 50/50 Valid/Test Shape 划分：Agent 操作及普通评测只使用 Valid，
  Runtime 权威 ABBA 使用 Valid + Test。Agent Evidence 隐藏 Test 明细、全量聚合延迟及 Test
  误差指标；奇数多出的一个归 Valid，单 Shape 拒绝启动。两个集合各随机抽取最多 15 个 Shape，多出的
  Shape 不参与评测，Metadata/Roofline 同步裁剪。私有 `shape_split` 留档种子、算法、原始全集与选中 ID。
  VecAdd 示例新增第二个 Shape。

- Evolver 按 Lineage/Backend 持续 resume 原生会话，跨进化、串行 Challenger 构建、基础设施重试和
  控制器重启保留历史；每次仍加载新的输入和 Candidate，Trace 与用量不重复计入历史内容。

- 明确 Evolver 不限于修复上轮改动，也可分析已完成 Optimizer Trajectory，依据具体运行行为
  判断是否新增能力或修改 Candidate 代码以帮助 Kernel 优化。

- Evolver 再次修改前先复核上一轮已评估改动，跟踪新工具的发现、运行和实际使用，对照预期效果，
  不把分支获胜直接当作改动有效，也不把尚未评估的提案视为失败。

- 为 Evolver 注入下一轮 Optimizer 的 Runtime 服务目录，明确可通过 Candidate 代码组合现有服务，
  再判断真实能力缺口；不新增 Runtime 权限，也不允许在 Evolution 中调用这些服务。

- 移除 Bootstrap/Evolver 主动建议 Direction 的机制：当前 `suggest` 动作和 Evolution 报告的
  `suggested_directions` 字段均被拒绝并提供修正提示，历史 Journal 仍可读取。Evolver 专注跨分支
  证据整合、归因纠偏与 Agent 改进，Optimizer 自主选择研究方向。

- FA4 源码树任务与生产七臂消融对齐：Epoch 1 使用同 Agent 的 Active/Challenger 副本；准备阶段
  冻结六个对照臂；任务入口可启动全部七个 Campaign，每条 Trajectory 固定 15 个 Attempt。

- 超限的 Dev 文件映射自动走 Agate OSS，覆盖源码树 Agent/权威 ABBA。执行前校验归档校验和并
  还原精确文件；上传各阶段独立重试，不改变逻辑请求身份和测量策略。

- 新增 Agent `evaluate.comparison`（`method="abba"`）：Core 与 KDA 从工作区文件或目录上传
  A/B 两份 Kernel。Runtime 封存源码，并通过 Agate `dev` 使用固定版本评测器及相同输入运行
  测试；记录逐侧测量与相对
  加速，不改变 Kernel 保留与 Agent 晋升的权威决策。不再提供独立的 `abba` 操作。
- Agent `evaluate` 支持自定义输入生成器与 Shapes，并新增不测性能、不自动 Profile 的
  `mode="correctness_only"`。Core 可从工作区文件上传输入；Runtime 封存探索性请求与结果，
  但不允许其代替 Candidate 提交所需的可信 Contract 完整评测。未完成的检查可以重试。
- 新增 `ablation-evolve-1` 和 `ablation-evolve-5`，分别运行 15 x 1 和 3 x 5；现有主臂标为
  `evolve-3`（5 x 3）。三臂均为 30 次 Optimizer Attempt，分别进化 14/4/2 次。首轮同一 Agent
  在两个独立分支运行，不调用 Evolver 或创建新 Agent Revision；从 Epoch 2 开始正常进化。
  各臂复用同一 Bootstrap Baseline，继承模型和 Evolver
  Commit，独立维护历史并保留 Skills/Tools。
- 新增 `ablation-pool-1` 和 `ablation-pool-5`，默认 Pool 更名为 `ablation-pool-3`。
  每条 Trajectory 固定运行 15 个 Bootstrap 之后的 Optimizer Attempt。三个 Pool 均为两条并行
  Trajectory，每臂合计 30 次；Retained 默认改为与 Isolated 一一对应的 `ablation-retained-01/02`
  独立 Campaign，各一条 Trajectory、15 次 Attempt，仅保留自己的 Skills/Tools。各 Arm 按串行 Attempt 数派生
  目标 Epoch，Bootstrap 始终不计入。
- 为 Optimizer 的 Attempt Report 新增必填的 `contributing_kernel_trial_ids`，列出本次 Attempt 取用过
  其代码或思路的历史 Kernel Trial。之所以用 Kernel Trial ID，是因为 Optimizer 本来就没有、也不应该
  获得 Kernel Revision 词汇。Core 与 Runtime 都只校验形状，都不去解析它是否在可见历史内，与该 Report
  中其他 Kernel Trial 引用的处理方式一致。Runtime 会把它带入派生的 Final Report，因此后续 Attempt 与
  Evolver 无需额外改动即可读到。
- 明确告知 Evolver 可以研究、汇总并融合多个可见 Agent 的 Source、Skill 与 Tool 到同一个 Candidate，
  并新增必填的 `contributing_revision_ids` 提案字段，声明除 Source Base 以外所有被取用过内容的
  Revision。Runtime 会按冻结可见范围、Lineage DSL 和已完成历史重新校验每一项，随后记录进已封存的
  Evolution Trace、Sealed Proposal 事件与 Epoch Lesson，并以 Source 路径形式投影到
  `input/evolution-reports/evo-N.json`。Source Base、Source Diff 目标与 Revision 祖先关系仍然唯一。
- 在 `input/evolution-reports/evo-N.json` 中补充每次历史 Evolution 的准确 Source 修改集合，
  Evolver 不再需要 Diff 两棵 Source 树才能知道那次演化改了哪些文件。
- 向 Optimizer 开放每个已完成 Epoch 的全部分支（包括未被选中的），位于
  `epochs/N/branches/<label>/`，每个 Epoch 的 `summary.json` 标明被选中的分支。当前 Epoch 仍只显示本
  Attempt 自己的 Trajectory，绝不暴露并发运行的兄弟分支。
- 围绕架构、配置、接口、评测、运维与持久协议重新整合发布文档；删除已被取代的设计/状态文档，
  并将术语与当前 Campaign、Lineage、Epoch、Branch、Trajectory、Attempt、Kernel Trial、
  Kernel Revision 和 Agent Revision 模型同步。
- 新增精简的设计理念文档，解释可进化 Agent 与可信 Runtime Authority 之间的分离。
- 删除无调用方的完整 Agent State 快照校验，并统一 Gateway Result 投影、Candidate 路径解析、
  Artifact 文件索引与 SQLite 事务处理。
- 增加常驻生产控制面、受管多 DSL Campaign Task 与逐 DSL 检查脚本。
- Sandbox Host 准备支持并发，并通过以非 root Worker 直接创建 Root/Probe 兼容 Lima virtiofs。
- 明确记录 Worker 共享宿主网络的边界。
- 在权威 Session 封存和 Agent Evidence 中移除高频 Claude `system/thinking_tokens`
  估算遥测，同时保留最终 Usage 记录。
- 移除 GPU Wiki Feedback 的生成、持久化、投递和接收；GPU Wiki 现在仅提供知识查询。
- 新生产 Campaign 准备会拒绝不干净的 Core/Evolver Worktree，确保固定 Commit 准确标识
  Agent Bundle 源码。
- 固定版本的上游 GPU Kernel 项目作为 Framework Baseline Workspace 的 `reference/` 目录提供，
  在两种 bubblewrap 模式下从 `reference_projects_root` 只读挂载。Attempt 不再挂载这棵树：
  通读上游项目属于建立首个实现的工作，而 Attempt 应当依据自己已测得的历史推进。
- Attempt Manifest 升到 schema 9，并不再在其中发布 Workspace 布局。布局在两端都是固定的，
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
