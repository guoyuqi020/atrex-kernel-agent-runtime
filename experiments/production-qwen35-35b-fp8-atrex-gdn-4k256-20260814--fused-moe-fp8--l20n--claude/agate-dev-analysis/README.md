# AKA 与 retained 的 Agate Dev 使用研究

## 结论

**两边都把 Agate Dev 当作可编程的远端实验环境，而不只是编译或报错调试工具。** AKA 经常在 Dev 中运行 NCU、解析采集结果，也会自行编写配对计时与硬件探针。retained 则大量通过 `scratch/` 中的请求和脚本做候选筛选、随机输入检查、底层指令实验，再将稳定的辅助代码放进 `tools/` 跨 Attempt 使用。

因此，**retained 的普通 Evaluate 次数少，不等于实验数量或 Kernel 测量次数按同样比例减少**。一个 Dev Job 内可以运行多个候选、多个输入规模和几十轮计时。两边的 Dev 差异更接近“实验怎么组织、采集工作由谁承担”，不能简单归结为 AKA 在做无效工作、retained 只调用标准接口。

## 1. 范围与数量

沿用主报告的同起点对照：同一 DSL 的 AKA 与 retained 使用同一个 Bootstrap Kernel seed。覆盖两遍 AKA 的 114 个优化 Episode，以及 retained 双 Trajectory 的 120 个 Optimizer Attempt；排除 Bootstrap、其他消融臂及 Harness 自动提交的 ABBA Job。

| DSL | AKA 显式 `--kind dev` | AKA 请求 `--kind profile` | retained 登记的 Agent Dev 请求 | retained Dev 命令退出非零 |
|---|---:|---:|---:|---:|
| CUDA | 250 | 296 | 290 | 23 |
| Triton | 287 | 184 | 116 | 7 |
| CuteDSL | 333 | 246 | 210 | 58 |
| **合计** | **870** | **726** | **616** | **88** |

AKA 还出现 34 次未显式指定 kind 的 `auto` 调用。`profile` 和 `auto` 不能直接加到“已提交 Dev Job 数”：只有返回信息才能确认最终路由。此次在 218 条 Profile 调用、8 条 Auto 调用的关联输出中明确观察到回退 Dev；大量输出经过异步、重定向或截断，未看到回退提示不表示没有回退。870 条显式 Dev 命令中，392 条在远端命令参数里直接出现 `ncu`、`profile_driver` 或 NCU wrapper 名称；这只是可识别的采集相关子集。

retained 的 616 条请求对应 615 个有 Job ID 的远端 Job、1 次提交前校验拒绝。615 个 Job 的平台状态都是 `succeeded`，但其中 **88 个自定义命令退出非零**，另 527 个退出 0。平台成功只说明 Job 执行完成，不说明脚本中的正确性检查或优化假设通过。那次拒绝是 `STEP` 环境变量不符合 Gateway 的前缀白名单。

这些数字来自不同层：AKA 是 Trace 中的显式 CLI 调用，retained 是 Gateway 账本；都不是 HTTP 请求次数。AKA 一条 Shell 可以启动多次调用，循环和间接执行还可能产生更多 Job；本报告不据此估算 GPU 费用或宣称精确的调用节省比例。统计见 [summary.json](data/summary.json)。

## 2. retained：`scratch/` 是实际的 Dev 实验入口

### 从请求文件到远端执行

以 Triton 的实际调用为例，Agent 先准备请求 JSON，再调用 Runtime CLI：

```bash
python3 agent/optimizer/src/runtime_tools.py gateway-execute --request scratch/dev-bench-warps16.json > scratch/dev-bench-warps16-result.json
```

样例中的 [command.txt](examples/retained-triton-benchmark/command.txt) 和 [request.json](examples/retained-triton-benchmark/request.json) 保留实际原文。请求的核心字段如下：

```json
{
  "operation": "dev",
  "command": "python3 bench_moe_latency.py --iters 30 --cases 512,2048,4096,6144,8192",
  "file_paths": [
    "tools/bench_moe_latency.py",
    "tools/probe_moe_correctness.py",
    "scratch/variant_kernel.py"
  ],
  "intent": "custom_harness",
  "job_timeout_s": 600
}
```

`scratch/` 不是远端自动扫描的任务队列，也不会因为放了一个文件就执行。Agent 显式调用 `gateway-execute` 后，Core 工具读取 `file_paths`，将内容上传；这些路径在远端按文件 basename 命名，因而这里执行的是 `python3 bench_moe_latency.py`，而不是 `python3 tools/bench_moe_latency.py`。当前工作 Kernel 另外作为候选提供为 `kernel.py`。`scratch/variant_kernel.py` 则成为可同时导入的另一个候选。

### 工作区实际留下了什么

120 个 retained Attempt 的 `scratch/` 最终快照中，共找到 **484 份 `operation=dev` 的请求 JSON**。这不是 484 次调用：同一请求可以被重用、被覆盖，也可能写好后未执行。

结合 Trace 返回的 Job ID、对应异步任务及最终重定向结果，可将 616 条账本记录中的 520 条关联到请求文件。去掉 6 条命令归属有歧义的记录后，514 条对应请求快照中：

- **424 条**引用 `scratch/` 内的上传文件；
- **254 条**引用 `tools/` 内的上传文件；
- **14 条**不使用 `file_paths`，例如直接以 `python -c` 读取远端安装库，或查询 NCU 支持的 metric。

`scratch/` 与 `tools/` 可以同时出现，不能相加。请求关联和代码版本的限制见第 6 节；没有关联上的 96 条仍计入总请求和退出状态，不强行猜测其命令。完整文件目录见 [scratch-dev-requests.json](data/scratch-dev-requests.json)。

### 它们具体在做哪些实验

下面按已关联请求中的脚本入口统计。一条 Dev 可以依次调用多个脚本，因此各行可能重叠；这不是完整、互斥的用途分类。

| DSL | 脚本入口 | 对应 Dev 记录 | 实际用途 |
|---|---|---:|---|
| Triton | `bench_moe_latency.py` | 66 | 在同 Pod 比较当前候选与上传的 variant，遍历多种 Token 数，用 CUDA Event 重复计时 |
| Triton | `probe_moe_correctness.py` | 9 | 独立作为入口执行自构造输入的正确性检查；更多时候被其他脚本导入 |
| CUDA | `moe_check_probe.py` | 59 | 在指定 Token 数上运行候选与自建 Reference 检查 |
| CUDA | `moe_ood_probe.py` | 33 | 探索异常路由 ID 等边界行为 |
| CUDA | `moe_stress_probe.py` | 7 | 极端专家路由、部分 Tile、异常 ID 及计时 |
| CUDA | `moe_bitwise_ab_probe.py` | 27 | 候选与基线的输出一致性及 A/B 探针 |
| CUDA | `stage_ab_probe.py` | 20 | 拆分多个 Kernel launch 的阶段耗时 |
| CUDA | `fwd_latency_ab_probe.py` | 8 | 整体 forward 的 A/B 计时 |
| CuteDSL | `paired_ab_probe.py` | 31 | 候选/Incumbent 交错计时，输出一致性和中位延迟 |
| CuteDSL | `harness_compare.py` / `ab_harness.py` | 6 / 5 | 其他自建比较 harness |

这些计时和自构造输入属于探索证据，不等于隐藏测试集的权威 Evaluate。详见 [usage-patterns.json](data/usage-patterns.json)。

## 3. retained 的代表性样例

### 3.1 在 Dev 中筛选 Kernel，再决定是否 Evaluate

[Triton 样例](examples/retained-triton-benchmark/)上传 `bench_moe_latency.py`、输入生成辅助脚本和 `variant_kernel.py`。计时器在每个规模上 warmup 5 次、测量 30 次、取中位数，并在不同规模间交换候选与 variant 的执行顺序。它的说明直接写明用途是“before spending a Gateway evaluate, screen … variants”。

该次 [实际输出](examples/retained-triton-benchmark/stdout.txt)包含五组规模，所有输出差值为 0，候选/variant 延迟比的几何平均为 **0.9788**。这说明一次 Dev 已完成多组输出比较和反复计时，而不是一次简单的编译检查。这个比值是该 Dev 探针的结果，不替代权威性能结论。

这里还有一个工具质量问题：[调用前可恢复的脚本](examples/retained-triton-benchmark/at-call/tools/bench_moe_latency.py)声称会 assert 数值一致，但实际只打印 `max_abs`，没有阈值判断或失败退出；variant 导入失败还会退化为只测候选。因此，即使退出 0，也不能机械解释为“两份 Kernel 已通过正确性比较”。

### 3.2 CuteDSL 也自建配对计时，不只调用标准 Profile

[CuteDSL 样例](examples/retained-cutedsl-paired-ab/)中，`paired_ab_probe.py` 同时导入 `kernel` 和 `kernel_incumbent`，在四组自构造输入上运行 40 对交错计时，输出两侧中位数及加速比。

对应 [Job 输出](examples/retained-cutedsl-paired-ab/stdout.txt)中，四组 `equal` 都为 `false`，但脚本仍退出 0。例如 T=4096 的 `max_abs_diff=0.013671875`，同时输出 `speedup=1.051`。这不能单凭 `equal=false` 判定不符合算子容差；它说明的是：**该脚本只报告 bitwise 差异，并没有把它实现成阻断条件。**

### 3.3 CUDA：指令实验、压力测试和阶段归因

- [MMA 布局实验](examples/retained-cuda-mma/)：`scratch/mma_proto2.py` 自带两套 CUDA/PTX MMA fragment 映射。在这次 Dev 中，原映射 `max_err=199.0`，交换映射 `max_err=0.0`，得到 `mapping=swap`。这是验证底层实现假设的有效实验，不是 Gateway 故障恢复。
- [压力测试](examples/retained-cuda-stress/)：一次请求连续运行两个 `moe_check_probe.py` 和一个 `moe_stress_probe.py`。后者自行构建 Reference，检查极端路由、异常 ID，并执行 20 轮计时。它使用自己的容差表达式 `2*ATOL + 2*RTOL*abs(ref)`，并不等同于 Runtime 的完整 Correctness Gate。
- [逐阶段计时](examples/retained-cuda-stage-timing/)：归档脚本关闭模块的 Graph replay 开关，以 CUDA Event 包装 `cuLaunchKernel`，按 8 个阶段归因。实际输出中 T=8192 的总计时从 2896.1 µs 到 2868.9 µs，主要可见变化之一是 reduce 阶段从 344.1 µs 到 321.6 µs。它改变了被测执行路径，不能当作正常 Kernel 路径的权威延迟；该辅助脚本在 Attempt 内又有修改，本样例保留的是最终工作区快照。

### 3.4 Dev 也被用于远端源码和环境检查

[远端检查样例](examples/retained-cutedsl-inspection/)只上传 `scratch/inspect_mma.py`，读取实际安装的 CuTeDSL MMA 类、Blackwell helpers 和 CUTLASS 示例目录，并将结果打印回来。它主要解决本地环境与 GPU 镜像不一致的问题，不是在测量 Kernel。

## 4. AKA：Dev 同时承担采集管道和自建实验

### 4.1 Profile 请求背后，是 Dev 中的 NCU 与解析脚本

AKA 的输出多次明确说明：对于 generalized task，要通过 Dev profile route 注入一个私有真实 Shape，因此原生 Profile 入口不直接适用。随后 Agent 需要处理 NCU 命令、Kernel 过滤、CSV/Report 解析和结果传输。

[Triton NCU 样例](examples/aka-triton-ncu/)保留了实际 Shell 命令及 `collect_ncu.sh`。该脚本说明，远端有 NCU CLI，但缺少 `ncu_report` Python 模块，且 inline artifact frame 被省略，所以改为在远端解析 CSV、通过 stdout 返回摘要。它支持 `time` 与 `full` 两种采集方式，包含 launch skip/count、Kernel regex、CSV 表头识别、指标聚合和输出过滤。

这个样例展示了真实的采集维护工作，也暴露了可靠性风险：脚本的 `time` 分支直接给 `gpu__time_duration.sum` 打上 `us`，没有按 `Metric Unit` 转换；无 CSV 或无表头时打印 `PARSE-FAIL` 后却 `exit(0)`。**打印了数值或命令退出成功，不足以保证单位和测量语义正确。** 本次不把这些打印值重新解释为准确的 GPU 时间。

### 4.2 AKA 同样会设计有价值的配对实验

[CuteDSL 配对计时样例](examples/aka-cutedsl-paired-ab/)的 `paired_time_driver.py` 不只重复跑候选。它支持 A/A 对照、交换模型实例化顺序，并在两个阶段内交替执行两侧，以检查并抵消脚本所观察到的 slot bias。

Trace 中先有较简单的版本，之后 Agent 重写了该工具。样例的 `at-call/` 版本由第 899 行 Write 恢复，对应第 1001 行调用；其输出给出 T=512 的 `bias_cancelled_delta_mean_us=+5.57`、T=8192 的 `-35.8`，并带标准误。这里的差值是 candidate−incumbent，正负分别表示更慢/更快；仍只是自建探针上的比较。

这反驳了“AKA 的 Dev 全是流程性浪费”的解读：**它也在主动发现测量偏差、构建对照实验。** 区别在于这些操作常被组织在 `profiles/episode_N/harness/`，与 NCU 采集、Shell 过滤和交接文件混在同一段工作流中。

### 4.3 命令包装与上传识别也产生了额外尝试

[Triton 上传失败样例](examples/aka-triton-upload-failure/)中，Agent 用 `bash -c` 包装两条 Python 命令：先运行 MMA roofline 探针，再测带宽。sandbox 输出 `files=0`，Agate Job 已启动，但第一条命令找不到探针文件，退出 2。

Trace 能恢复调用前已经写出的本地探针，说明不只是“Agent 忘记创建文件”；该次包装命令没有把文件带到远端。之后改为直接调用 Python 后，又暴露了 JIT 全局常量问题。这类失败属于上传/工具协议和探针实现的修复，不能与 Kernel 优化本身的正确性失败混为一谈。

## 5. 跨 Attempt 复用：retained 的 `tools/` 确实被使用了

不能只看某个 `scratch/` 文件名重复出现就判断复用。这里同时比较请求引用、Attempt 身份及归档文件内容 SHA-256：

| 文件 | 关联 Attempt 数 | 关联 Dev 记录 | 覆盖 Epoch | 最终快照内容版本数 |
|---|---:|---:|---|---:|
| Triton `tools/bench_moe_latency.py` | 35 | 66 | 1–10 | 1 |
| Triton `tools/probe_moe_correctness.py` | 38 | 102 | 1–10 | 1 |
| CUDA `tools/moe_check_probe.py` | 25 | 59 | 1–7、10 | 1 |
| CUDA `tools/moe_bitwise_ab_probe.py` | 15 | 27 | 5–10 | 2 |
| CuteDSL `tools/paired_ab_probe.py` | 14 | 23 | 4–10 | 1 |

表中统计“作为上传依赖”的记录；例如 `probe_moe_correctness.py` 只被直接执行 9 次，却在 102 个已关联 Dev 中作为输入生成或检查依赖出现。`paired_ab_probe.py` 总入口计数为 31，其中以 `tools/` 路径上传的子集为 23。

这提供了具体的跨 Attempt 复用证据：稳定的输入生成与计时器放在 `tools/`，本次候选、一次性探针和请求文件放在 `scratch/`。它并不证明复用独立导致了 Token 节省，也不表示 AKA 没有复用；AKA 的 NCU wrapper 自身就注明沿用前几个 Episode 的模式。

## 6. 文件、证据与复核

### 样例怎么读

九个样例见 [examples/README.md](examples/README.md)。每个样例包含原始调用命令、可取得的辅助文件及结果；`provenance.json` 保存原归档相对路径、Trace 行号、Job/Attempt 身份和文件 SHA-256。

- `command.txt`：Trace 中的完整 Shell 调用原文，包括重定向和过滤；不自动执行。
- `request.json`：retained 工作区最终保存的请求文件。
- `files/`：Attempt/Episode 结束时的辅助文件快照；它们可能在会话中被覆盖，不能一律称为某次 Job 的精确上传副本。
- `at-call/`：能通过显式 Write/Edit 安全恢复的调用前辅助文件。只重放文本操作，不执行历史 Shell/Python；无法确认时不生成该目录。
- retained 的 `files/work/kernel/kernel.py`、`result.json`：来自该次操作的不可变 Kernel / Gateway Result Artifact，与最终辅助文件快照区分。
- `stdout.txt` / `output.txt`：对应 Job 或关联 Trace 的实际输出；AKA 复合命令的输出可能同时包含多次调用。

**这些是研究证据，不是可直接运行的生产示例。** 未补造缺失的框架文件、远端注入输入或历史文件版本，也未执行任何复制的 Agent 代码。

### 复现统计

在本目录执行，只需要 Python 标准库；原始归档只读打开，SQLite 使用 `mode=ro`。命令中的归档路径按实际解压位置修改。

```bash
ARCHIVE_ROOT="$HOME/atrex-runs/production-qwen35-35b-fp8-atrex-gdn-4k256-20260814--fused-moe-fp8--l20n--claude"
python3 audit_dev.py --archive-root "$ARCHIVE_ROOT" --output data
python3 summarize_usage.py --data data
python3 export_samples.py --archive-root "$ARCHIVE_ROOT" --data data --output examples
python3 -m unittest discover -p 'test_*.py'
```

统计以同级 `analysis/bash-action-index.json.gz` 的 234 份 Trace 为固定样本，校验原文摘要；按 Tool ID 去重，关联原调用的返回和明确的异步任务。Shell 只作静态解析，不展开循环或执行动态脚本。retained 通过 Job ID 关联账本与请求；请求/重定向文件可能被覆盖，多义关联在用途统计中排除。剩余未关联部分保留为未知，不按相邻位置强配。

`data/retained-dev.json.gz` 保存全部 616 条 Dev 账本记录与关联信息；`data/aka-dev-profile.json.gz` 保存 1630 条显式 Dev/Profile/Auto 调用及可见输出。它们用于追溯本报告，不替代原始 Archive。
