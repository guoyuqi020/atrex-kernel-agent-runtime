# 性能门禁

Kernel 留存与 Agent 晋升是两项相互独立的 Campaign Policy。任一 Policy 都可以选择普通
Evaluate 或 `same_allocation_abba`，一项的选择不会改变另一项。
所有路径都使用由 Runtime `gate_policy` 封存的 Campaign Contract；任务输入不能覆盖 Gate-owned
Sampling、容差、Timeout、时钟 Policy 或 Evaluator Identity。

`gate_policy.production_gate` 为 true 时，内容级 Production Gate 位于 Correctness 与 Performance
之前，强制固定 DSL、自包含 Source、批准依赖，并禁止 PyTorch/预构建 Kernel Fallback。Policy
拒绝的 Candidate 不会提交 Agate；即使存在旧 Evaluation，也不能被留存或晋升。

## 普通 Evaluate

`gate_policy.optimizer` 控制探索 Evaluate Sampling。与 Atrex 对齐的 Policy 使用 1 个
Correctness Case、1 次 Eval 和 100 个 eager bench iters。Bootstrap 不复用这组 Sampling，而是
顺序执行 `gate_policy.bootstrap.stages`：先 1 个 Case，再 5 个 Case，每阶段 5 个 eager bench
iters。两个阶段都必须通过，只有第二阶段提供权威 `v0` 延迟。Attempt 终结完全归所选留存 Policy。

Runtime 为 A、B 分别并发执行准确 `repeats` 次独立逻辑 Eval Measurement，要求所有 Measurement
正确，比较
算术平均，并且绝对提升必须严格大于 `measurement_uncertainty_us`。Attempt 完成前，B 的算术
平均和引用全部 B 原始 Result 的新聚合 Artifact 会成为 Candidate Kernel 的权威 Evaluation。
即使 `repeats: 1` 也执行这一流程，不会晋升此前的探索性延迟。

每个逻辑普通 Evaluate 都经过同一个可信分批执行器；Optimizer 探索、Bootstrap Stage、Lineage
Seed 校验和普通 Evaluate Comparator 共用该实现。执行器按确定顺序每 4 个 Shape 分为一批，最多
同时运行 4 个 Shape Batch。所有 Shape 都必须通过；Runtime 按每批 Shape 数量加权，对 Batch
延迟恢复完整 Workload 的几何平均，并在聚合结果中保留每个物理 Agate Job。Contract 不超过
4 个 Shape 时仍保持为一个未封装的 Agate Job。

## 同 Allocation ABBA

`repeats: 2` 时，精确 Schedule 是 Incumbent、Candidate、Candidate、Incumbent；更多 Repeat
继续交替每一对的顺序。Runtime 从配置指定的 Atrex Bench Commit 导出 evaluator-only 子集，
与不可变的两个 Kernel Source Snapshot、封存的私有 Evaluation Contract 一起上传。

每个 Shape Batch 只提交一个 Agate `dev` Job，因此只占用一个 GPU Allocation。Runtime 持有的
远端 Driver 只切换 Candidate Source，并为每个 Schedule Entry 启动一次全新的标准 Atrex Bench
Evaluation。不同 Shape Batch 可以在不同 Allocation 上并发。Runtime 不接受残缺或顺序不一致的
Schedule，也不会把失败 Allocation 的部分结果与重试结果拼接。
`gate_policy.lock_clocks` 默认为 `true`。当它为 true 时，远端 Driver 会在第一个 A/B Entry 前对整个 Allocation
加一次锁，将 Atrex Bench 的外部锁频标记传给所有 Evaluator 子进程，并在完整 Schedule 后恢复
时钟。无法应用或恢复请求的锁频都会让该 Batch 失败关闭；`lock_clocks: false` 不修改时钟。

全部 Batch 完成后，Runtime 根据逐 Shape 延迟重建每次完整 Workload Run，再分别对两个 Kernel
的各 Repeat 取几何平均。Incumbent 与 Candidate 的所有 Run 都必须通过 Compile、Correctness 与
Performance。只有满足以下严格条件时 Candidate 才获胜：

```text
100 * (incumbent_geomean - candidate_geomean) / incumbent_geomean
    > minimum_improvement_percent
```

严格的 `>` 意味着刚好等于阈值或完全相同时保留 Incumbent。每个重建出的 A/B Run 都写入
`kernel_measurements`；所有 Run 引用同一个聚合 Gateway Result Artifact，其中包含原始 Batch
Job、请求 Schedule、解析后的远端 Payload、封存的 Evaluation Contract Digest，以及权威的
Incumbent/Candidate 延迟与 Roofline SOL 聚合。每次远端标准 Evaluation 返回逐 Shape `sol.pct`；
Runtime 保留这些值，并对 Candidate 的各 Repeat 和 Shape 做几何聚合。Runtime Event 记录 Batch
Job ID、Atrex Bench Commit、标准 Evaluator Bundle Digest 和比较完成状态。

对于 Kernel 留存，Runtime 会在 Attempt 完成前，以 Candidate 的 ABBA 几何平均延迟及同一个聚合
Artifact 替换新 Kernel Revision 上的临时探索 Evaluation，不再提交单独的 Runtime-final 普通
Eval。Bootstrap 的 `v0` 没有 Incumbent 可做 A/B 比较，因此执行两个有序独立阶段。

同 Allocation 保证依赖 Agate 的 Job Contract：一个运行中的 `dev` Job 固定拥有一个 Allocation；
Runtime 不另外校验物理 GPU UUID。如果隐藏 Shape 被拆成多个 Batch，该保证分别作用于每个
Batch，这与旧 Atrex Kernel Agent Verifier 的行为一致。
