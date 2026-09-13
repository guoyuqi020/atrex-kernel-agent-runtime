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

探索性 `evaluate` 会记录测量证据，但不会直接创建 `vN` Kernel Revision。Agent 可以在一个
Attempt 中评测多个 Candidate，并写入 Experiment Journal。通过 `candidate_ready` 提名时，
该精确 Candidate 仍须成功完成基于可信 Evaluation Contract 的完整评测。
这份预检证据可以来自本 Attempt，也可以通过 `adopt` Experiment 显式采纳可见历史中的兼容成功
完整 Evaluate。Runtime 核验原始 Trial 和精确 Kernel/Result 绑定；采纳只记录当前决策，不新建
测量，也不改变原测量归属。配置的独立 Retention 比较保持不变。不兼容的历史证据需要重新做完整
Evaluate，无需通过修改注释来制造不同 Digest。

`evaluate` 接受可选的 `mode`、`input_py` 和 `shapes`。`mode` 只能为 `full`（默认）或
`correctness_only`；后者仅检查正确性，不测量性能，也不自动执行 SOL Profile。`input_py` 提供兼容
Agate 的 Python `_make_inputs` 生成器；`shapes` 提供以整数字符串为键的非空 JSON Object，每条
Shape Record 都是与生成器兼容的 Object。两个组件可以独立覆盖；未指定的组件继续使用封存
Contract 中的值。可信 Reference、容差和 Gate Policy 继续生效，Agent 不会因此获得私有输入。

Core 与 Kernel Design Agent 还支持 `input_path`、`shapes_path`，读取 Workspace 相对路径的 UTF-8 文件并将内容作为
`input_py`、`shapes` 上传。输入源码上限为 128 KiB，Shape 文件上限为 256 KiB；文件必须位于真实
Workspace 目录下且为普通文件，绝对路径、路径穿越、符号链接和 `.runtime` 控制路径都会被拒绝。
同一组件的内联形式与路径形式互斥。请求幂等键按文件内容计算，不依赖本地文件名。

```json
{"operation": "evaluate"}
```

```json
{"operation": "evaluate", "mode": "correctness_only"}
```

```json
{"operation": "evaluate", "mode": "correctness_only", "input_path": "scratch/custom-input.py", "shapes_path": "scratch/custom-shapes.json"}
```

自定义输入、自定义 Shape 或仅正确性评测的结果仍有 Kernel Artifact 与 Result Artifact 身份，
并记录实际 `mode` 和 `input_scope`（任一组件被覆盖时为 `custom`，否则为 `contract`）。
仅正确性结果不包含性能测量。这些调用不能替代 `candidate_ready`、Kernel Retention 或 Agent
Promotion 所要求的可信 Contract 完整评测。需要完整评测时应省略输入覆盖，并设置
`mode: "full"` 或省略 `mode`；原有默认请求与结果格式保持不变。

### 自定义输入文件示例

以下配套示例针对公开的向量加法 ABI：`Model.forward(left, right)`，Model 无构造参数，输入为
两个 CUDA float32 向量。实际使用时，应根据任务公开 ABI 调整参数名、dtype、device、构造参数
和合法尺寸；这里的演示 Case 不是私有验证 Shape。生成器沿用公共
[VecAdd 输入示例](../examples/shared/vecadd/reference/input.py)的格式。

保存为 `scratch/custom-input.py`：

```python
import torch


def _make_inputs(num_elements: int) -> dict[str, torch.Tensor]:
    left = torch.randn((num_elements,), device="cuda", dtype=torch.float32)
    return {"left": left, "right": torch.randn_like(left)}
```

保存为 `scratch/custom-shapes.json`：

```json
{
  "0": {"input_kwargs": {"num_elements": 1024}, "init_kwargs": null},
  "1": {"input_kwargs": {"num_elements": 4097}, "init_kwargs": null}
}
```

各字段的职责不同：

- 顶层数字字符串是自定义 Case ID，不是 Tensor 维度，也不是选择同编号隐藏测试的指令。
  多条记录表示多个 Case；`4097` 用于演示非对齐长度。
- `input_kwargs` 通过 `_make_inputs(**input_kwargs)` 传给生成器，键名必须匹配函数参数。
  不要把 `num_elements` 直接放在 `init_kwargs` 旁边，也不要在 JSON 里编码 Tensor。
- 生成器返回字典，键名匹配 `Model.forward` 参数。这里的 `left`、`right` 是 Tensor，
  不是 Shape 描述。直接返回该字典，不要返回 tuple 或额外包装成 `{"kwargs": ...}`。
- `init_kwargs` 用于构造 `Model(**init_kwargs)`；无构造参数时用 `null` 或 `{}`。
  它不传给输入生成器。
- 随机种子由评测器控制，不要在生成器里调用 `torch.manual_seed` 固定种子。

保存 `scratch/evaluate-custom.json`，再调用 Session 的 `gateway-execute` 工具：

```json
{
  "operation": "evaluate",
  "mode": "correctness_only",
  "input_path": "scratch/custom-input.py",
  "shapes_path": "scratch/custom-shapes.json"
}
```

```bash
python3 agent/optimizer/src/runtime_tools.py gateway-execute --request scratch/evaluate-custom.json
```

工具路径若不同，以 Session 提示的路径为准。省略 `mode` 即进行正确性与性能评测；
这两个文件也适用于下文的 ABBA 比较。通常建议成对提供：
只覆盖 Shapes 时，字段必须与保留的生成器兼容；只覆盖生成器时，函数必须接受保留的 Shapes
所传参数。任何一种方式都不会暴露未覆盖的私有组件。自定义输入仍须满足公开 ABI，
自定义测试通过也不能替代可信 Contract 的完整评测。

直接使用 HTTP 时，发送 Python 文件内容作为 `input_py`、解析后的 JSON Object 作为 `shapes`。
路径由 Core/KDA 工具展开，HTTP 端点不会根据路径去 Agent 容器中读取文件。

## 探索性 ABBA

`evaluate` 支持可选 `candidate_path`，普通单 Kernel 评测也适用；省略时使用当前 `work/kernel`
树。若需将 Candidate B 与基线 A 比较，应提供
`comparison: {method: "abba", baseline_path: "scratch/baseline.py"}`；基线路径在 `comparison`
内部必填。每个路径均为
Workspace 相对路径，指向普通 `.py` 文件或 Kernel Bundle 目录。单个 Python 文件上传为
`kernel.py`，目录保留内部相对文件名。绝对路径、路径穿越、符号链接、`.runtime` 控制路径、
特殊文件及空 Bundle 会被拒绝。两个 Bundle 都会封存，请求身份按上传内容生成；仅重命名文件而
不改变上传内容不会产生新的比较。

```json
{"operation": "evaluate", "comparison": {"method": "abba", "baseline_path": "scratch/baseline.py"}}
```

```json
{"operation": "evaluate", "candidate_path": "scratch/candidate-kernel", "comparison": {"method": "abba", "baseline_path": "scratch/baseline-kernel", "repeats": 2}, "input_path": "scratch/custom-input.py", "shapes_path": "scratch/custom-shapes.json"}
```

`comparison.repeats` 默认 2，表示每侧的观测次数，生成 A、B、B、A 顺序。取值范围为 2–20，但 Schedule
还须满足 Runtime 的 Allocation 预算。每个 Shape Batch 在同一个 Allocation 内测量两侧；不同
Shape Batch 可以使用不同 Allocation。ABBA 始终使用 `mode: "full"`（通常省略），拒绝
`correctness_only`。两侧共用选定的输入生成器和 Shapes，支持与 Evaluate 相同的独立
`input_py`/`shapes` 覆盖及 `input_path`/`shapes_path` 文件参数；省略的组件继续复用私有
Contract，不会将其暴露给 Agent。

即使使用可信 Contract，ABBA 仍仅用于探索，不会保留 Kernel 或晋升 Agent，也不能替代
`candidate_ready` 所需的成功可信 Contract 完整 Evaluate。Runtime 的权威 Retention 和
Promotion 比较仍独立执行。

响应保留 `operation: "evaluate"`；`result.comparison` 记录 `method: "abba"` 及实际 `repeats`
次数。不提供独立 ABBA Operation，也不接受顶层重复次数参数。Result Artifact 标识本次比较；响应中的
Kernel Artifact 身份属于 B。保留的 Result Artifact 包含 A 的
`baseline_kernel_artifact_digest`、`baseline`/`candidate` 正确性及延迟摘要、所有 `measurements`
及 `schedule`，以及 `mode`、`input_scope`。`speedup` 为 A/B 延迟比，
`improvement_pct` 为 (A−B)/A × 100，聚合延迟使用几何平均。通过 `result-artifact-read` 查询这些证据；探索性 ABBA 不生成普通 Evaluate Record。Runtime 会为 A、B
两侧保留归一化的逐 Shape 聚合结果，并将三次底层返回作为私有 Evidence 保存。

## 普通 Evaluate 的 Shape 分批

每轮普通 Evaluate 为每个验证 Shape 提交一个 Agate Eval Job，最多 16 批并发。默认与 ABBA 的
单 Shape、16 批并发一致，覆盖 Optimizer 请求、Bootstrap 各阶段、Lineage Seed 和普通 Evaluate
Comparator。Agent 仍只发起一个逻辑请求；Runtime 按批裁剪私有 Contract、对应 metadata 和
Roofline，并在聚合 Artifact 中保留每批的 Job 与结果。

全部 Shapes 必须通过正确性检查。Agent 发起完整 Evaluate 时，Runtime 固定执行三次完整的逻辑
Agate 调用，对每个 Shape 的三个值取中位数，再对这些中位数取几何平均；三次调用内部不会再叠加
配置的重复层。16 批限制分别作用于每次调用。Bootstrap、Lineage Seed 和可信 Comparator 仍使用
各自配置的采样策略。ABBA 的比较配置不会改变普通 Evaluate 的上述聚合方式。

## 单次提交与三次测量

在一条 Lineage 内，每个精确的完整普通 Evaluate 或探索性 ABBA 任务只允许 Agent 发起一次。任务
身份包含精确 Candidate Kernel、ABBA 的 Baseline Kernel、测量方法与参数，以及封存的输入域。
`correctness_only` Evaluate、Profile、Dev、Check 和 Disassemble 不属于这项规则，因为它们不共享
同一种逐 Shape 性能聚合语义。

- 第一次被接受的任务会执行三次独立、语义完全相同的逻辑 Agate 调用。
- Runtime 要求三次 Shape 覆盖一致，逐 Shape 取中位数，再机械计算聚合延迟和 ABBA Speedup。
  任一次明确的正确性失败都会被保留，不能被另一次成功结果掩盖。
- Runtime 只返回一个 Agent 可见 Result Artifact，并附带
  `measurement_aggregation: {"repetitions": 3, "method": "per_shape_median"}`；三次原始返回只作为
  私有 Evidence 保存。
- Agent 再次主动提交完全相同的任务时，Runtime 会在调用 Agate 前拒绝，并返回
  `previous_result_artifact_digest`，引导 Agent 使用 `result-artifact-read` 复用结果；同一次调用的
  网络重连仍保持幂等，并回放原响应。

该规则避免 Agent 通过重复提交未修改代码消耗评测资源或挑选有利样本。修改 Kernel、Baseline、
输入域或测量参数后会形成新的任务。

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
4. 当封存 Contract 没有 Roofline 时，在每次正确的完整 Evaluate 后执行 NCU SOL
   Profile；仅正确性评测不会触发自动 Profile。

Builder 在受限输入输出下运行一个完整 Atrex Bench Commit 的可信代码，不获得 Agent Authority。
生成结果会在 Agent 启动前封存进 Campaign Contract。Profile 失败不会使正确性或延迟失效，SOL
保持不可用。

使用可信 Contract 评测时，只有封存 Contract 的 `roofline` 字段为 null 才会自动回退 NCU。
结构合法但不包含实际 Agate
设备 Key 的显式 Roofline 可能无法产生 SOL，同时也不会触发自动回退。运维方应生成与设备兼容
的 Roofline，或不提供该字段。

自定义评测不携带 Contract 专属 Metadata 和 Roofline。因此，当自动 Profile 已启用时，
正确的自定义 `full` 评测也可触发同一回退流程。

当每个 Shape 都有 SOL 时，Kernel Catalog 以全 Shape 几何平均值展示；否则 JSON 为 `null`，
Table 显示 `-`。

## 持久证据

Runtime 保留精确 Candidate 源码、原始 Agate 结果、归一化 Measurement、Comparator 每轮结果、
Kernel Trial、Attempt Report 和版本化 Kernel/Agent 结果。Worker 投影保持隐私边界；管理接口可
读取有界的精确源码与原始结果。命令见[接口参考](interfaces.zh.md)，身份与可见性语义见
[协议](protocols.zh.md)。
