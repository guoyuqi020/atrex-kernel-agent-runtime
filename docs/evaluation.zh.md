# 评测与晋升

[English](evaluation.md) | 中文

Runtime 持有评测策略与晋升权。Core 可以请求探索性操作，Agate 持有 GPU 执行，而是否保留 Kernel
或 Agent Revision 只能由 Runtime 决定。

## 评测输入与隐私

每个 Campaign 封存一份私有 Evaluation Contract，其中包含 Reference 实现、Input Generator、
Validation Shapes、Metadata、可选 Roofline、容差、采样策略、锁频策略与 Production Gate 开关。
Runtime 会在封存前用部署策略覆盖所有 Gate 持有字段。

Agent 不会看到精确 Validation Shapes、`reference.py`、`input.py`、Metadata 或 Roofline，只会得到
描述合法参数域和非 Shape ABI 约束的公开 `shape_train` Contract。Gateway 响应只暴露聚合正确性、
聚合延迟、按不透明数字 Shape ID 的延迟和清理后的 Profile 数据。

## 探索性操作

Optimizer Runtime Tools 通过 `gateway-execute` 暴露 `check`、`dev`、`evaluate`、`profile`、
`disassemble` 和 `env`。携带 Candidate 的操作会在
调用 Agate 前封存精确源码，每个结果都不可变，并可通过返回给 Agent 的身份查询。

探索性 `evaluate` 是可信测量证据，但不会直接创建 `vN` Kernel Revision。Agent 可以在一个
Attempt 中评测多个 Candidate、将其写入 Experiment Journal，并在 `attempt-report` 中提名一个已
评测 Candidate。

## 普通 Evaluate 的 Shape 分批

每轮普通 Evaluate 为每个验证 Shape 提交一个 Agate Eval Job，最多 16 批并发。默认与 ABBA 的
单 Shape、16 批并发一致，覆盖 Optimizer 请求、Bootstrap 各阶段、Lineage Seed 和普通 Evaluate
Comparator。Agent 仍只发起一个逻辑请求；Runtime 按批裁剪私有 Contract、对应 metadata 和
Roofline，并在聚合 Artifact 中保留每批的 Job 与结果。

全部 Shapes 必须通过正确性检查；跨 Shape 延迟取几何平均。配置的独立 Evaluate repeats
仍对各轮聚合延迟取算术平均。各 Repeat 独立运行，16 批限制按每轮计算，不是全局 GPU 并发限制。
ABBA 的比较配置不会改变普通 Evaluate 的上述默认分批设置。

## Correctness 与 Production Gate

`gate_policy` 定义容差、Correctness Case 数、Warmup/Benchmark 预算、超时、锁频和固定的 Atrex
Bench Evaluator。Bootstrap、Optimizer 探索、Kernel Retention、Agent Promotion 与 Lineage Seed
使用同一份封存策略，仅采样角色不同。

启用 `production_gate` 后，Runtime 会在 GPU 执行前和发布前再次运行可信源码检查。它强制指定
DSL，拒绝 PyTorch 计算回退和动态/预构建实现加载，并在存在 `solution.json` 时校验其内容。
探索性操作会返回安全的 Production Gate 警告，便于 Agent 修正；发布阶段仍然 Fail Closed。

## Bootstrap 与 Kernel Retention

Bootstrap 按 `gate_policy.bootstrap` 运行有序正确性阶段。成功的终态 Candidate 成为 Lineage 内
Kernel `v0`，此时没有 Incumbent 比较。

随仓库提供的策略将 `bootstrap.bench_iters` 设为 100，与普通 Optimizer Evaluate 一致。默认两个
阶段（先 1 case，再 5 cases）都使用该性能采样预算，并共用单 Shape、最多 16 批并发的执行器和
Agate 重试策略。网络错误重试原请求；`logs_unavailable` 且后端成功时，只重提失败批次并获取新
Job ID，按 5/10/20/40 秒退避，之后每 60 秒持续重试。Candidate 校验和正确性失败不按基础设施
错误重试。

普通 Attempt 使用 `kernel_retention_comparison`：

- `evaluate`：按配置重复次数分别测量 Incumbent A 与 Candidate B；Candidate 必须正确并超过配置
  的不确定性阈值。
- `same_allocation_abba`：在每个 Shape Batch 的同一 Agate Allocation 内交错测量 A/B；每个
  Repeat 各测一次 A、B，并在 `A, B` 与 `B, A` 之间交替，因此两个 Repeat 形成
  `A, B, B, A`。Runtime 校验 Schedule 并持久化每一轮结果。

所选 Comparator 的 B 聚合值就是 Candidate 的权威延迟，比较后不会再执行第二次独立 Attempt
终评。没有有效提名的 Attempt 仍进入 Attempt 历史，但不消耗 Kernel 版本号。

Agent Promotion 独立使用 `agent_promotion_comparison`。每个参赛 Agent 产出的最佳 Kernel 参与
比较；Runtime 可以保留 Kernel 而不晋升其 Agent，也可以在配置的比较通过后晋升 Challenger。

## Roofline 与 SOL

解析顺序如下：

1. 保留 Evaluation Contract 中显式提供的 Roofline；
2. 恢复 Campaign 时复用已封存 Roofline；
3. 若已配置，执行 Commit 固定的 Atrex Bench Roofline Builder，并校验精确 Shape 覆盖；
4. 当封存 Contract 没有 Roofline 时，在每次正确 Evaluate 后执行 NCU SOL Profile。

Builder 在受限输入输出下运行一个完整 Atrex Bench Commit 的可信代码，不获得 Agent Authority。
生成结果会在 Agent 启动前封存进 Campaign Contract。Profile 失败不会使正确性或延迟失效，SOL
保持不可用。

只有封存 Contract 的 `roofline` 字段为 null 时才会自动回退 NCU。结构合法但不包含实际 Agate
设备 Key 的显式 Roofline 可能无法产生 SOL，同时也不会触发自动回退。运维方应生成与设备兼容
的 Roofline，或不提供该字段。

当每个 Shape 都有 SOL 时，Kernel Catalog 以全 Shape 几何平均值展示；否则 JSON 为 `null`，
Table 显示 `-`。

## 持久证据

Runtime 保留精确 Candidate 源码、原始 Agate 结果、归一化 Measurement、Comparator 每轮结果、
Kernel Trial、Attempt Report 和版本化 Kernel/Agent 结果。Worker 投影保持隐私边界；管理接口可
读取有界的精确源码与原始结果。命令见[接口参考](interfaces.zh.md)，身份与可见性语义见
[协议](protocols.zh.md)。
