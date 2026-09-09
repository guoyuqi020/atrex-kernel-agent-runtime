# 多文件 Kernel 源码树

[English](source-trees.md) | 中文

## 声明与责任

Campaign 的每条 Lineage 二选一：原有 `baseline_kernel`，或新的 `source_manifest` +
`source_repository`。路径相对 Campaign 文件解析，两种形式都保留 `initial_evidence`。
Model、DSL、Epoch、分支竞争、自进化和 Registry 的版本体系不变。
[准备示例](../examples/source-tree/README.zh.md)直接读取 GDN 原始 Manifest，不把源码拼成单文件。

Manifest 使用 `source.revision`、`source.archive_paths`、`source.package_root`、`adapter`、
`editable_roots`，以及可选的 `runtime_requirements`。Runtime 严格从指定 Git commit 导出，
不导入未提交修改。固定适配器复制为 Evaluation Contract 的 `candidate_path`（通常为
`kernel.py`）；原仓库不作为共享可写目录挂载。整个 seed 封存到 CAS。

Runtime 在封存的 Evaluation Contract 中生成 `kernel_sources[DSL]`，包含 seed digest、
不可变文件哈希、源码 commit、Python 包根目录、可编辑范围和依赖。这份锁定信息不在
Agent 可编辑的 Candidate 内。不同 DSL 可以有不同源码树和编辑范围。

原 GDN Manifest 的 `measurement`、`bringup` 和旧生命周期设置只作为来源元数据接受，
**不覆盖 Runtime Gate**。正确性 cases、性能迭代、repeats、锁频和晋升仍由 Runtime 配置决定。
只支持 snapshot 导入；非空 `runtime_support` 和其他 repository-search 模式会被拒绝。
需要预先在 Agate GPU 环境提供声明的 distribution/version；评测时不会自动安装 Agent 选择的包。

## 工作区与可信校验

源码以只读 `input/kernel/` 作为输入，直接复制到 `work/kernel/`，没有额外 `source/` 层：

```text
work/kernel/
├── kernel.py                       固定适配器
├── LICENSE                         固定
├── UPSTREAM_PROVENANCE.json         固定
└── flashinfer/
    └── gdn_kernels/blackwell/        可编辑源码子树
```

Runtime 注入实际 editable roots 和调用说明，并同步更新 Prompt Fragment 的完整性哈希。
文件只读属性用于避免误改；真正的验收依据是 Runtime 侧的源码锁校验：固定文件修改或删除、
范围外新增文件、symlink、遮蔽受保护依赖、预编译加载产物都会被拒绝。允许在可编辑范围内
新增/删除源码，也允许不修改源码直接评测或复用历史版本，不要求为了生成新 digest 而修改注释。

Evaluate 和最终提交通道共用整树封存规则。Python 缓存、`.git` 元数据和空目录不改变源码身份，
其他文件不会被悄悄忽略，构建产物应写到 `scratch/`。每次 Trial 和 Kernel Revision 都指向
完整源码 Artifact，历史读取、回滚、adopt 都保留整棵树，不增加另一套版本层级。

Production Gate 校验固定支持文件，并把可编辑文件作为同一 DSL 实现进行扫描。允许对已
打包源码的包导入和相对导入；不等于允许调用外部预编译算子库、混用 DSL 或关闭 Gate。
它仍是静态策略校验，不是任意 Python 都无法干扰评测器的形式化保证。

## 评测与 Bootstrap

Agent 工具接口不变，Core/KDA 原有目录 Bundle 提交能力可直接使用。源码树普通 Evaluate
内部改走 Agate Dev：上传固定 Runtime driver、部署配置锁定 commit 的 Atrex Bench evaluator、
整棵源码、适配器、包根路径、私有输入/Shape 和 Gate 参数。Journal 中仍记为逻辑 Evaluate，
不是 Agent 自由运行 Dev 后自报的证据。原单文件任务继续使用原生 Agate Eval。

已支持 full、`correctness_only`、自定义输入/Shape、普通重复评测、Agent 探索 ABBA，以及
Runtime 权威 ABBA。ABBA 各步在同一 allocation 中使用不同的源码副本、独立进程和独立 JIT
缓存，沿用整段测量的锁频策略，防止 A/B 模块与缓存混用。

源码树 Bootstrap 使用配置的 Optimizer backend 启动完整 framework-baseline Session
（GDN 输入包使用 Claude）。Runtime 将源码范围追加到本次 Session 的 Bootstrap Prompt
副本，Core/KDA 在所有 backend 中加载它；固定 Bundle 和可复用 prompts 不变。
Agent 先评测提供的 seed，必要时仅修复允许修改的源码，记录正常 Direction/Experiment
Journal，并提交标准 Attempt Report；正确的原样 seed 可以直接提名。Runtime 按与探索
Evaluate 相同的规则封存整树、校验匹配的 Agent 证据，再独立执行 Bootstrap Gate 并注册 v0。
真实 Session trace 和标准报告进入 Lineage 历史，失败重试及终评恢复沿用正常 Bootstrap 机制。

已登记的 v0 不可变，重跑仍会复用。若旧任务使用了无模型 Bootstrap，要验证新流程应使用
新的 Campaign creation key，不改写历史报告，也不放宽 Journal 校验。

## Profile、Check 与 Disassemble

现有工具通过 Runtime 生成的 Agate Dev 驱动支持完整源码树，无需 Agent 自己拼命令：

```json
{"operation":"profile","level":"sol","shape_id":"0"}
{"operation":"profile","level":"deep","kernel_regex":".*GatedDelta.*","source":true,"launch_count":1}
{"operation":"check"}
{"operation":"check","sanitize":"memcheck"}
{"operation":"disassemble","fmt":"sass"}
```

- Profile 使用 NCU：`survey` 采集 launch 信息和 duration，`sol` 采集 SpeedOfLight，`deep`
  采集 full set 且必须有 Kernel filter。`counters` 添加指标；`kernel_name` 精确匹配 demangled
  名称，`kernel_regex` 使用正则；`source=true` 请求源码关联 SASS。`top_kernels` 限制结构化
  视图的条数，原始 CSV 保留实际采集的 launches。
- Runtime 按 opaque `shape_id` 选择一个输入（默认排序后的首个 ID），在采集区域外预热，
  重新生成输入后运行一次 forward。`launch_skip` / `launch_count` 选择这次 forward 内的
  launches，不是性能重复次数，默认 0 / 10；过滤后没有 Kernel 会明确失败。
- Check 调用固定适配器的 Model 和一次 forward，触发懒 JIT 编译。这是运行探针，
  **不是数值正确性测试**。可选 `sanitize` 支持 `memcheck/racecheck/initcheck/synccheck`，
  工具检查报错会产生非零退出码。`arch` 必须匹配已分配 GPU 的基础计算能力，例如 `sm_103`；
  不支持交叉编译或架构后缀。
- Disassemble 从 NCU report 导出 SASS（`auto/sass`）或 PTX（`ptx`）。PTX 依赖 NCU/工具链
  版本及 report 中是否包含 PTX，缺失会报错。Check 和 Disassemble 默认使用首个 opaque case。
  当前驱动面向 NVIDIA，明确拒绝 `rocprofv3` 和 AMD 的 `fmt=isa`。

每个作业有独立 JIT 缓存，并遵循 Contract 的 `lock_clocks`。只上传封存 Candidate 和选中的
input/case，不上传 evaluator 或 Reference model。声明依赖需在 GPU 镜像中预先提供；
`requirements` 只校验已安装版本，Dev 不安装软件，两种 `deps_mode` 都没有安装阶段。
Dev allocation 上限为 600 秒，预留 30 秒清理时间。

Runtime 仍按原逻辑操作记录归属、源码与结果，不把这些操作降格成任意 Dev 证据。
作业绑定与上游幂等键按 Attempt 的恢复代次隔离，避免中断后新 Session 与历史作业冲突。
同一代次中，重复请求会继续读取已登记的作业，不再额外创建 Dev 作业；历史归属和结果保留。
即使 transport completed，`passed=false/status=error` 仍代表诊断失败；缺少结构化输出
不能算成功。Result Artifact 保存结构化 Kernel 指标和 `exports` 文本（CSV/SASS/PTX）；
每份 export 带字节数、SHA-256，以及超过 512 KiB 时的显式截断标记。二进制 `.ncu-rep`
不由此驱动回传。工具 stdout/stderr 保存在可信原始证据内，不进入 Agent 的隐藏 case 视图。
诊断不能替代 Evaluate 正确性或晋升证据。

参数依据：[NVIDIA NCU CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)、
[Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)。

当前限制：没有 Roofline 时仍跳过自动 NCU fallback；已有 Roofline 仍可用于 SOL，
显式 Profile 可返回测得的 compute/memory SOL。临时 Optimizer dev-shell 仍要求
单文件 baseline；源码树可以在 Bootstrap 后用已有 Lineage 的 dev-shell。真实 GPU 环境的
兼容性还需部署验收，本地测试用 CPU evaluator double，不代表 GDN 已通过 GPU 正确性评测。

## 实现与验证

- `kernel_sources.py`：Git 导入、源码锁、编辑范围、整树身份、Prompt 投影。
- `gateway/source_tree.py`：逻辑 Evaluate 的 Dev 传输和可恢复结果解析。
- `gateway/source_diagnostics.py`：固定源码树 NCU、编译/运行、sanitizer 驱动。
- `gateway/abba.py`、`abba_remote.py`：固定驱动、完整 A/B 源码、进程及缓存隔离。
- `gateway/proxy.py`、`workers/core.py`：评测提交与最终提名的统一封存。
- `workers/lineage_bootstrap.py`、`composition/bootstrap.py`、`gateway/finalization.py`：Agent Bootstrap、整树提名与权威终评。
- `tests/test_kernel_sources.py`：commit 锚定、越界修改、源码身份、Prompt 哈希、真实子进程导入、
  恢复轮询、ABBA 缓存隔离、Bootstrap Gate。
- `tests/test_source_diagnostics.py`：真实驱动/子进程加 CPU GPU 工具替身，覆盖 NCU 参数、
  sanitizer 失败、导出和私有结果投影。
