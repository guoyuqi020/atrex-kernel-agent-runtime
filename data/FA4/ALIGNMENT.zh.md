# FA4 资料对齐核对

[English](ALIGNMENT.md) | 中文

核对来源：`atrex-bench-new-qwen38-fa4-gdn` 与 `fa4-prefill-aka-r0-startpoint-20260915`。文件与正式算子评测定义保持一致；Runtime 的调度/测量策略不等同于原始冒烟检查。

## 逐字一致的资产

| 资产 | 核对结果 |
| --- | --- |
| `task/reference.py`、`task/input.py` | 与新版 Benchmark、R0 包内 Reference 都逐字相同 |
| `task/shape_train.json`、`task/shape_valid.json` | 与新版 Benchmark 逐字相同，全部 30 个 Shape 保留 |
| `task/metadata.json`、`task/roofline.json` | 与新版 Benchmark 逐字相同 |
| `task/adapter.py` | 与 R0 `kernel.py` 逐字相同，进入 Candidate 时仅改变文件名 |
| Source Bundle | 全部 70 个文件逐一与 R0 包比较，无增删或内容差异；运行时另外加入固定适配器，共 71 个文件 |
| Evaluator Bundle | 全部 32 个文件与提供的 Benchmark 比较，无内容差异；Runtime 导出其中 31 个评测代码文件，不提交 README |
| `smoke/run_agate_dev.sh`、`smoke/smoke.py`、`smoke/reference/shapes.json` | 与 R0 原始文件逐字相同 |
| `source-provenance.json`、`source-validation.json` | 与 R0 来源、历史验证记录逐字相同 |

`asset-integrity.json` 保存这些文件和 Bundle 的 SHA256。准备阶段检查哈希、完整源码文件集合和 Commit；发现变化即拒绝发布 Workspace。R0 的列表式旧 Shape 文档与新版字典式 `shape_valid.json` 逐 ID 归一化后完全相同，包括 `init_kwargs` 和全部 `input_kwargs`。

输入仍使用原始 `_make_inputs()`：Query 的 BF16 正态样本乘 1.22 后转换为 FP8 E4M3，KV 乘 0.96；Workspace 初始为零，页表、序列长度、缩放和输出分配均未改动。没有换用简化 Shape，也没有把私有 OSS 原始 Tensor 当成已经下载的评测输入。提供资料中的正式评测本身使用这一合成输入生成器。Benchmark 的 `solution.py` 不是本任务的初始 Candidate；起点保持为指定 FA4 R0，未引入 C05/Increment 优化版本。

## 正式评测与冒烟不是同一种检查

| 项目 | 原始 `smoke.py` | Runtime 正式评测 |
| --- | --- | --- |
| P128 冒烟 | 固定 P128 示例，检查输出形状及有限值 | 不作为目标正确性或 `v0` 注册依据 |
| 目标输入 | 指定一个 Shape，默认 0 | 完整 30 个 Shape；公开训练域不替代验证域 |
| 正确性 | seed=1，整体相对 L2≤0.05 | 使用原始 Benchmark 的逐元素检查；返回值和被修改的 `out` 均为 atol=0.06、rtol=0.04 |
| 输入副作用 | 不执行正式副作用 Gate；Candidate/Reference 依次使用同一输入对象 | 原始 Benchmark 检查输入副作用；允许修改 `out`、将 `workspace_buffer` 声明为 scratch |
| 随机种子 | `torch.manual_seed(1)` | 原始 Benchmark 的 `deterministic_input_seed(stage, shape_id, case_index)`，不改成 seed=1 |
| Latency | 不计时 | 原始 `run_eval.py` / `atrex_bench` 性能测量 |
| Agate | Dev，L20D，Job 超时 1800s、等待超时 2100s | Runtime 可信 Dev Driver 执行封存的评测器；超时、分批、ABBA 由 Runtime 配置 |

原始两份资料本来就区分冒烟与正式 Benchmark，因此不能要求两者所有参数和判断都相同。新增 `smoke` 入口只为了准确复现原始两种冒烟命令；它从固定 R0 物化临时目录，不提交终态报告、不注册 Kernel，不测试 Agent 已修改的 Candidate。

## Runtime 自己选择的测量策略

新版 Benchmark 明确规定输入和逐元素正确性策略，Warmup、超时、Case 数及 ABBA 排程由调用方控制。当前任务与不带额外配置的 `run_eval.py` 默认值的区别如下：

| 参数 | 提供的 `run_eval.py` 默认值 | 当前 Runtime 配置 |
| --- | --- | --- |
| 模式 | eager | eager |
| Warmup / Bench | 10ms / 100ms | 10ms / 100ms；字段名仍是 `warmup_iters` / `bench_iters` |
| 正确性 Case | 每 Shape 1 个 | Bootstrap 先 1 后 5；Optimizer 5；Retention 1 |
| Candidate / Performance 超时 | 60s / 600s | Candidate 120s；普通源码树 Evaluate 的实际 Performance 与外层单次运行预算均为 600s；ABBA 每个 A/B 运行预算为 120s |
| 锁频 | 默认 off，支持调用方管理 | 默认锁频；ABBA 外层锁定，评测器验证 external 标记 |
| 分批与重复 | 单次调用内执行给定 Shape | 每批 1 Shape，最多 16 批并发；普通 Evaluate 一次逻辑调用；ABBA 三次完整比较逐 Shape 取中位数，每次 repeats=2，即 A/B/B/A |

这不是逐字复现原始 CLI 的完整执行条件，但仍使用同一份正式评测实现及正确性定义。SHA256 核对不能证明不同测量策略或不同运行窗口产生相同 Latency。

当前 `performance_timeout_seconds=120` 配置不会覆盖普通源码树 Evaluate 的 Performance 预算：复用的 Driver Builder 用 `evaluation_timeout_seconds=600` 覆盖它。这里按实际提交参数记录，不把声明值当成生效值；该行为有传输层回归测试覆盖。

本环境的 Agate 资源 `L20D` 对应 B300。原始 Roofline 标签为 `NVIDIA B300 (SM100)`，数值不改动；Runtime 传输只移除硬件名括号后缀。是否成功取得 SOL/NCU 仍需远端运行验证，不能用文件一致性核对代替 GPU 验证。

本次核对未启动模型、服务或 GPU Job；`source-validation.json` 仍是提供的历史结果，不是重新测量的数据。静态 Production Gate 的源码树适配尚未在本次资产核对中实现，配置仍保持关闭。
