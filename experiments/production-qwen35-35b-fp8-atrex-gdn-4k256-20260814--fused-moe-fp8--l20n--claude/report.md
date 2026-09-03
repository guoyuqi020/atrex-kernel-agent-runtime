# Fused MoE FP8 生产消融实验与 Token 分析

## 摘要

本报告合并完整消融结果与 Session Trace 分析，覆盖 CUDA、Triton、CuteDSL。**在同一 DSL 内，AKA-1、AKA-2 与 retained 从同一个 Bootstrap Kernel seed 开始，成本不含 Bootstrap。**

AKA 两遍校正后共消耗 **1,911.906M Token**，retained 为 **1,113.496M**，少用 **41.8%**，最终性能与两遍 AKA 中较好结果相近。AKA 平均每轮有 157 个可见响应，retained 为 93 个；cache read 占净 Token 差额的 97.8%。Trace 中，AKA 更多承担采集/解析修复和显式流程维护，retained 更多处理 Runtime 接口及可复用知识，但两侧都有有效研究和工具错误，差额不能全视为浪费。

## 1. 实验设置与统计口径

- 算子与环境：`fused_moe_fp8`，L20N / `sm_120`，production 模式；Runtime 通过 Claude CLI 调用 `qwen3.8-max`。
- 实验配置：每个 DSL 包含 evolve、retained、pooled 和两个 isolated 重复臂，另运行两遍 AKA。
- 完成规模：Runtime 共 15 个臂、150 个 Epoch、501 个优化/演化 Session；两遍 AKA 共 114 个优化 Episode，均已完成。
- 对照起点：同一 DSL 的 AKA 与 retained 使用相同 Bootstrap seed，成本统计不含 Bootstrap；配置以运行归档为准。

### 性能、时间与轮次

“最终最佳 GeoMean latency”越低越好。Runtime 取 `lineages.best_kernel_revision_id` 对应 Kernel 的正确权威 `kernel_measurements.latency_us` 最小值，即一次隐藏 Shape 评测的 latency GeoMean；AKA 只取 `measurement_source=authoritative_verification`。不同运行窗口的历史最佳值不是同一 GPU allocation 下的配对复测。

- **Epoch wall**：Runtime 为 Epoch 1 创建至 Epoch 10 完成；AKA 为 E1 `created_at` 至 E19 `finalized_at`。包含评测、等待和恢复，排除 Bootstrap。
- **Agent-h**：Runtime Optimizer/Evolver 的 `completed_at - started_at` 之和；并发 Session 重复计时，不含 Session 外 Gateway 等待。AKA 无完全等价指标，记为 —。
- **O/E sessions**：每 DSL，evolve 为 38/9，retained/pooled 各为 40/0，每个 isolated 为 20/0；AKA 的 `19 Ep/0 Ev` 表示优化 Episode 数，不是 Worker Session 数。Episode/Attempt 内可有多次实验。

### Provider 用量与校正

Runtime 排除 Bootstrap 和 Problem Generalization，只统计 Optimizer/Evolver；逐个读取 501 份 Session Artifact 的终态 `modelUsage["qwen3.8-max"]`，0 个缺失。该值已含子 Agent/子请求，不再重复累计；相较 Registry 顶层 usage，多计入 **114,249,418 Token（+2.55%）**：uncached input +81,494,327、cache read +25,739,587、cache write +1,020,510、output +5,994,994。Provider 管理的子请求没有独立 Runtime Worker Session 清单。

AKA 排除 Bootstrap，汇总优化、嵌套子 Agent 和独立 production Policy Review Trace；同一 `message.id` 取最后观察到的 usage，不取未结算的首条。本次复核 237 份 Trace，无跨文件重复 message ID。相较初版，五个 Episode 账本少计 16.619M，子 Agent 流式 usage 少计 22.009M，共校正 **38.628M**。

M 表示一百万 Token，`Total = uncached input + cache read + cache write + output`。AKA-2 Triton 有一次超时 Policy Review 仅保留 **221,597 total tokens**，无四类分项：Total 包含该值，费用保留上下界。Token 总量不等同费用或计算量。

### 费用折算：实验时价格快照

沿用初版于 **2026-09-02** 记录的阿里云百炼中国区（华北 2，北京）Qwen3.8-Max 原价、显式缓存口径，不重新按当前价格估值：

| Token 类别 | 单价（元 / 1M tokens） |
|---|---:|
| Uncached input | 12 |
| Explicit cache hit / cache read | 1 |
| Explicit cache creation / cache write | 15 |
| Output | 36 |

`cost_cny = (uncached input × 12 + cache read × 1 + cache write × 15 + output × 36) / 1M`。

这是价格快照折算，不是 Provider 的 `total_cost_usd` 账单，不含免费额度、折扣、税费或汇率。快照中的隐式缓存价格为 1.5 元/M；若 Runtime 全部 cache read 按此计算，其总费用为 **¥11,658.51**，主表显式缓存口径为 **¥9,493.27**。缺失分项的 AKA Policy Review 按全部 cache read 至全部 output 估算范围。所有合计使用未舍入 Token 计算。

## 2. 各 DSL 完整结果

下表保留原性能、时间和完成状态；AKA Token 分项与费用已按完整 Trace 校正。

### CUDA

基线：26,694.1 µs。

| Arm | GeoMean µs | Epoch wall | Agent-h | O/E sessions | Uncached in M | Cache read M | Cache write M | Output M | Total M | Cost ¥ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| evolve (active+challenger) | 1,295.489 | 24h 58m | 38.12 | 38/9 | 7.176 | 351.988 | 11.803 | 5.126 | 376.093 | 799.67 |
| ablation-retained | 1,199.026 | 23h 21m | 40.45 | 40/0 | 7.181 | 374.746 | 10.299 | 5.417 | 397.642 | 810.39 |
| ablation-pooled | **1,086.193** | 24h 09m | 41.73 | 40/0 | 7.059 | 384.431 | 10.506 | 5.619 | 407.615 | 829.02 |
| ablation-isolated-01 | 1,655.317 | 20h 42m | 20.57 | 20/0 | 3.754 | 186.091 | 5.391 | 2.721 | 197.958 | 409.97 |
| ablation-isolated-02 | 1,517.628 | 22h 47m | 22.68 | 20/0 | 4.137 | 189.557 | 5.522 | 3.026 | 202.242 | 430.96 |
| AKA-1 | 1,275.351 | 26h 23m | — | 19 Ep/0 Ev | 0.017 | 266.002 | 8.526 | 3.166 | 277.711 | 508.07 |
| AKA-2 | 1,161.882 | 35h 09m | — | 19 Ep/0 Ev | 0.023 | 374.235 | 11.243 | 4.137 | 389.638 | 692.08 |

Runtime CUDA 合计：**1,581.549M tokens、163.55 Agent-hours、¥3,280.01**。AKA-1：19 Episode，12 晋升/7 Pivot/0 拒绝/0 协议失败；AKA-2：19 Episode，12 晋升/6 Pivot/0 拒绝/1 协议失败。

### Triton

基线：1,686.1 µs。

| Arm | GeoMean µs | Epoch wall | Agent-h | O/E sessions | Uncached in M | Cache read M | Cache write M | Output M | Total M | Cost ¥ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| evolve (active+challenger) | 1,080.145 | 24h 16m | 38.85 | 38/9 | 6.358 | 333.799 | 10.174 | 5.112 | 355.442 | 746.72 |
| ablation-retained | 1,138.087 | 17h 58m | 30.98 | 40/0 | 4.498 | 263.118 | 7.627 | 3.975 | 279.218 | 574.59 |
| ablation-pooled | 1,095.815 | 21h 10m | 38.55 | 40/0 | 5.801 | 293.995 | 9.384 | 5.188 | 314.368 | 691.13 |
| ablation-isolated-01 | **1,044.368** | 21h 00m | 20.91 | 20/0 | 3.235 | 193.464 | 4.951 | 2.746 | 204.396 | 405.39 |
| ablation-isolated-02 | 1,058.340 | 21h 58m | 21.87 | 20/0 | 3.808 | 177.817 | 5.348 | 2.906 | 189.879 | 408.35 |
| AKA-1 | 1,105.962 | 29h 02m | — | 19 Ep/0 Ev | 0.019 | 312.034 | 9.635 | 3.401 | 325.090 | 579.24 |
| AKA-2 | 1,293.979 | 22h 28m | — | 19 Ep/0 Ev | 0.014 | 218.159 | 6.999 | 2.631 | 228.025‡ | 418.26–426.02‡ |

Runtime Triton 合计：**1,343.303M tokens、151.15 Agent-hours、¥2,826.18**。AKA-1：19 Episode，7 晋升/10 Pivot/0 拒绝/2 协议失败；AKA-2：19 Episode，5 晋升/9 Pivot/3 拒绝/2 协议失败。

### CuteDSL

基线：303,764.9 µs。

| Arm | GeoMean µs | Epoch wall | Agent-h | O/E sessions | Uncached in M | Cache read M | Cache write M | Output M | Total M | Cost ¥ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| evolve (active+challenger) | 1,391.746 | 23h 29m | 36.37 | 38/9 | 6.073 | 324.754 | 11.093 | 4.797 | 346.718 | 736.72 |
| ablation-retained | **1,125.741** | 26h 30m | 42.85 | 40/0 | 8.059 | 411.715 | 11.118 | 5.744 | 436.636 | 881.98 |
| ablation-pooled | 1,364.712 | 26h 46m | 44.72 | 40/0 | 7.332 | 415.604 | 11.016 | 5.819 | 439.772 | 878.34 |
| ablation-isolated-01 | 1,713.009 | 21h 48m | 21.64 | 20/0 | 3.364 | 206.891 | 5.175 | 2.880 | 218.309 | 428.55 |
| ablation-isolated-02 | 1,309.801 | 23h 30m | 23.30 | 20/0 | 3.933 | 222.512 | 5.568 | 3.007 | 235.020 | 461.48 |
| AKA-1 | 1,626.643 | 33h 43m | — | 19 Ep/0 Ev | 0.021 | 334.863 | 10.433 | 4.097 | 349.414 | 639.11 |
| AKA-2 | 1,205.153 | 35h 54m | — | 19 Ep/0 Ev | 0.020 | 325.186 | 13.121 | 3.701 | 342.028 | 655.48 |

Runtime CuteDSL 合计：**1,676.455M tokens、168.87 Agent-hours、¥3,387.08**。AKA-1：19 Episode，10 晋升/7 Pivot/1 拒绝/1 协议失败；AKA-2：19 Episode，13 晋升/5 Pivot/0 拒绝/1 协议失败。

‡ AKA-2 Triton 的 Total 包含分项未知的 0.222M Token；费用上下界按上述规则计算。所有 AKA 行均含子 Agent 和独立 Policy Review，排除 Bootstrap；晋升、Pivot、拒绝、协议失败为 Episode 终态分类。

## 3. AKA 与 retained 的对照

| 项目 | AKA 两个实例 | Runtime retained |
|---|---|---|
| 初始 Kernel | 对应 DSL 的同一个 Bootstrap seed | 与 AKA 相同的对应 DSL Bootstrap seed |
| 每个 DSL 的搜索结构 | 两次独立运行，各 19 个优化 Episode | 一个 Lineage，10 个 Epoch；每 Epoch 两条 Trajectory，每条串行两个 Attempt |
| 每个 DSL 优化轮次合计 | 38 Episode | 40 Attempt，即每条 Trajectory 20 次 |
| 三个 DSL 合计 | 114 Episode | 120 Attempt |
| 独立 Evolver | 无 | 无，`challenger_count=0` |
| 运行态 | 跨 Episode 交接文件、知识和工具等 | `ephemeral_agent_state=false`，保留运行态；不是 pooled 的临时运行态配置 |
| Token 范围 | 主 Agent、嵌套子 Agent、独立 Policy Review | Optimizer 终态 Provider `modelUsage`，包含 Provider 管理的子请求 |
| Bootstrap | 不计入 | 不计入 |

Retained 配置由归档 `ablation.json` 和 Lineage 记录确认。

这不是严格等预算对照：Runtime 的两条 Trajectory 经过 Epoch 选择与状态交接，不是独立统计样本；Episode/Attempt 都可包含多次实验，AKA 前两个 fast Episode 还要求各做五个 Trial。两侧 Prompt、工具、历史知识组织和实际搜索量仍有差异。

| DSL | AKA-1 + AKA-2，M | Retained，M | AKA / retained | Retained 少用 Token | Retained 相对 AKA 较好值的延迟 |
|---|---:|---:|---:|---:|---:|
| CUDA | 667.349 | 397.642 | 1.68× | 40.4% | 延迟高 3.2% |
| Triton | 553.115 | 279.218 | 1.98× | 49.5% | 延迟高 2.9% |
| CuteDSL | 691.442 | 436.636 | 1.58× | 36.9% | 延迟低 6.6% |
| **合计** | **1,911.906** | **1,113.496** | **1.72×** | **41.8%** | — |

延迟对比取 AKA 两遍中较低的最终最佳值，而 Token 计入两遍全部成本；不同窗口下几个百分点的差异不足以证明稳定优劣。

## 4. Token 差额主要来自反复读取上下文

| Provider 用量类别 | AKA 两遍，M | Retained，M | AKA − retained，M |
|---|---:|---:|---:|
| Uncached input | 0.114 | 19.738 | −19.624 |
| Cache read | 1,830.480 | 1,049.579 | **+780.902** |
| Cache write | 59.956 | 29.043 | +30.913 |
| Output | 21.134 | 15.136 | +5.999 |
| 仅有 total、分项缺失的 Policy Review | 0.222 | 0 | +0.222 |
| **合计** | **1,911.906** | **1,113.496** | **+798.411** |

AKA 主 Agent 每次可见响应平均携带约 102.8k input tokens。更多的文件检查、命令调用和分析会让既有上下文反复参与请求：两侧 output 相差约 6.0M，cache read 却相差 780.9M。97.8% 是账本分解，不能直接归因于某个 Prompt 或视为无效开销。

### 4.1 交互链长度对比

AKA 此表仅统计优化主 Trace，不含独立子 Agent/Reviewer。Retained 用量取终态 `modelUsage`，包含子请求；行为计数取归档主 Trace 文件及其中可见子链，两侧的隐藏子请求覆盖不完全一致。

| 指标 | AKA 主 Agent | Retained |
|---|---:|---:|
| Episode / Attempt 数 | 114 | 120 |
| 平均每轮 Token，M | 16.308 | 9.279 |
| 每轮 Token 中位数，M | 15.600 | 8.249 |
| 可见模型响应总数 | 17,891 | 11,115 |
| 平均每轮响应数 | **156.9** | **92.6** |
| 每轮响应数中位数 | 150 | 83 |
| 工具调用总数 | 21,156 | 13,389 |
| 上下文压缩次数 | 230 | 142 |

三个 DSL 的平均响应数均为 AKA 更高，中位数也有明显差异，说明并非仅由少数长 Episode 拉高均值。响应按所属链的 assistant `message.id` 去重，不等同全部计费请求；Runtime 中间 usage 不完整，不用于推算逐请求上下文长度。

## 5. Bash 用途与 Session 案例

### 5.1 Bash 与相邻思考的 Token 分布

AKA 的 Bash 调用数约为 retained 的两倍，而直接 Read、Edit/Write 的差距较小：

| 工具行为 | AKA 主 Agent | Retained |
|---|---:|---:|
| Bash | **13,745** | **6,922** |
| Read | 2,373 | 2,289 |
| Edit + Write | 2,987 | 2,456 |
| 工具返回文本量，百万字符 | 47.52 | 35.43 |

Bash 也能修改文件，例如 retained 常用 heredoc 或内联 Python 生成请求 JSON，因此不能仅凭 Edit/Write 次数判断文件修改工作量。

#### 统一统计口径与总量

对 Bash 命令、工具返回和相邻 thinking 使用同一套 `tiktoken 0.12.0 / o200k_base` 文本估算，区分**首次生成**与**后续上下文读取**，单位均为 M Token：

- 命令与 thinking 首次生成计一次 output；工具返回不是模型生成，不计 output。
- 文本每保留到一次后续请求，再计一次 input；累计量为首次生成与后续读取之和，不是每段文本只计算一次。
- 请求按所属链的 `message.id` 去重；命令、返回和 thinking 分别追踪。两次已记录压缩之间假设原文保留，压缩后仅保留明确列出的消息，不将摘要反向分摊给旧动作。
- 相邻 thinking 取调用前同一响应及返回后紧接着的思考，不跨人工消息、压缩或其他工具动作。AKA / retained 共提取 **15,492 / 8,837 段**；共享思考去重后，按关联 Bash 调用等分归类。

**以下均为文本估算，不是 Provider 计费用量。** 特别是 Trace 没有完整请求 payload，无法确认旧 thinking 是否自动剥离；其后续读取与含 thinking 的合计仅代表“原文持续保留”的场景。

| 文本组成 | AKA 首次生成 | AKA 后续读取 | AKA 累计 | Retained 首次生成 | Retained 后续读取 | Retained 累计 |
|---|---:|---:|---:|---:|---:|---:|
| Bash 命令 | 1.788 | 45.983 | 47.771 | 0.960 | 18.906 | 19.865 |
| Bash 返回 | — | 209.897 | 209.897 | — | 107.920 | 107.920 |
| **Bash 小计** | **1.788** | **255.880** | **257.668** | **0.960** | **126.826** | **127.785** |
| 相邻 thinking | 10.394 | 309.760 | 320.155 | 7.142 | 171.825 | 178.966 |
| **Bash + 相邻 thinking** | **12.183** | **565.640** | **577.823** | **8.101** | **298.650** | **306.751** |

Bash 命令/返回的累计量为 retained 的 **2.02 倍**；加入相邻 thinking 的保留场景后为 **1.88 倍**。合计按未舍入值计算，小计与合计不重复相加。

#### 按调用用途拆分

细查包含 `cat/head/tail/rg/grep/sed/awk/cut/wc` 的 9,793 / 3,384 次调用，剔除 heredoc 正文对分类的干扰后，按整条命令的用途互斥归类，并抽查原始返回与相邻对话。其他 Bash 单列。

下表统一展示**累计量**：Bash 为命令生成与命令/返回重读之和；相邻思考为生成与假设重读之和。两列可相加，但思考列仍受上述保留场景限制。用途明细可加总，小计/总计不重复相加。

| 调用类型 | AKA 次数 | AKA Bash，M | AKA 相邻思考，M | Retained 次数 | Retained Bash，M | Retained 相邻思考，M |
|---|---:|---:|---:|---:|---:|---:|
| 阅读/搜索 harness 与控制层实现 | 1,549 | 28.624 | 23.420 | 232 | 3.209 | 4.397 |
| 查看 CLI 帮助、确认参数 | 567 | 7.483 | 11.638 | 76 | 0.602 | 2.627 |
| Memory、Journal、Plan、Report 等历史与状态文件 | 931 | 27.999 | 25.850 | 202 | 8.659 | 3.125 |
| 读取/整理测量、Profile、后台任务日志 | 1,274 | 23.849 | 34.294 | 137 | 1.485 | 4.407 |
| GPU 执行后截取/过滤输出 | 1,762 | 16.832 | 38.876 | 66 | 0.800 | 1.871 |
| 读取外部参考库实现与文档 | 671 | 13.222 | 22.980 | 0 | 0.000 | 0.000 |
| Wiki 查询后截取/过滤输出 | 150 | 10.625 | 1.711 | 28 | 0.994 | 0.463 |
| Git、phase、validator、reviewer 等控制动作附带输出处理 | 896 | 10.010 | 23.583 | 2 | 0.003 | 0.037 |
| Kernel 实现文件与入口代码 | 502 | 9.885 | 20.471 | 488 | 9.412 | 17.946 |
| Skills 与 Markdown 文档 | 410 | 5.959 | 4.941 | 418 | 17.516 | 11.139 |
| 搜索 Session 或找回工具调用协议 | 109 | 3.919 | 1.223 | 143 | 1.817 | 5.326 |
| Journal / Direction / Experiment / Report 接口附带输出处理 | 228 | 3.762 | 4.525 | 225 | 3.321 | 4.661 |
| 其他：实验辅助脚本、目录环境、算子说明及混合读取 | 533 | 7.461 | 9.403 | 443 | 8.217 | 8.200 |
| 仅由写入式语法命中原标签 | 211 | 7.430 | 3.615 | 924 | 11.634 | 23.677 |
| **Shell 文本命令小计** | **9,793** | **177.060** | **226.531** | **3,384** | **67.669** | **87.875** |
| 其他 Bash（不在本次 Shell 文本用途识别范围） | 3,952 | 80.608 | 93.624 | 3,538 | 60.117 | 91.091 |
| **全部 Bash 合计** | **13,745** | **257.668** | **320.155** | **6,922** | **127.785** | **178.966** |

AKA 的主要差异在 harness/CLI 理解、测量日志和历史状态恢复；retained 则有更多 Skills 阅读、请求文件写入及相关思考，Kernel 源码读取量接近。

#### 代表性 Shell 案例

下表 S1–S6 展示分类背后的具体工作。

| 案例 | 对话中的实际工作 |
|---|---|
| S1 · AKA-2 / CUDA / E19 | Profile 只解析第一个 NCU action；读取 wrapper 和解析代码，准备按 `NCU_KERNEL_NAME` 获取目标 GEMM 指标。 |
| S2 · AKA-1 / CUDA / E16 | `tail -40` 仍返回 29,622 字符，且采到的是输入生成 Kernel；继续调整 NCU 采集，而非单纯看日志。 |
| S3 · AKA-2 / Triton / E7 | 阅读多版 Memory，重建 commit、quality gate 结论与 Triton→Gluon 转换要求之间的关系。 |
| S4 · AKA-1 / Triton / E13 | 搜索自身 Session，找回成功的 Dev 命令、argv 和隐藏 Shape 注入约定。 |
| S5 · AKA-1 / CuteDSL / E1 | 阅读 CUTLASS 后区分硬件缩放约束，选择 dense FP8 MMA 并每 128 个 K 元素自行缩放，直接影响候选实现。 |
| S6 · Retained / CUDA / E9 | 阅读七份 Skill，重新判断旧知识对 NCU/Evaluate 延迟差和锁频的解释是否仍适用。 |

这些案例区分了证据获取维护、状态恢复与算法研究，也说明持久化的分析仍需复核。

### 5.2 AKA：工具修复与 Kernel 研究交织

**AKA-2 / CuteDSL / Episode 17：41.323M Token，387 个响应，6 次压缩，最终 pivot。**

Agent 经历 NCU 解析失败、census harness 修复、本地缺少分析工具后转远端解析，以及 Wiki JSON 解析错误；同时验证了 GEMM1/SiLU/quant 融合、shared-memory/occupancy 约束和不同 MMA 指令，形成负结果。该轮同时包含证据获取维护与有效研究，不能把全部用量归为浪费。

### 5.3 Retained：复用历史完成负结果交接

**CuteDSL / Epoch 7 / Trajectory 1 / Attempt 1：8.270M Token，90 个响应，最终 pivot**，用量接近 retained 中位数。

Agent 读取已有 skills/tools 和上一 Epoch 两条 Trajectory 的报告，检验 shared-memory bank-conflict 假设。Gateway 指标改善但整体延迟未改善后，它恢复 incumbent、登记 Experiment、关闭 Direction 并更新知识，终态报告一次提交成功；期间仍有脚本和权限错误。这展示了负结果的完整交接，不是与 AKA E17 相同难度的受控对照。

### 5.4 Retained：长探索也能产出最佳 Kernel

**CuteDSL / Epoch 10 / Trajectory 2 / Attempt 2：35.418M Token，318 个响应，5 次压缩**，是 retained 最长的一轮。

该轮研究 TMA 替换 cp.async，修复 tracing/dispatch、参数传递及 digest 校验问题，最终提交的候选被 Registry 接受，并成为 retained CuteDSL 最佳 Kernel。因此，长 Session 本身不能作为低效证据。
