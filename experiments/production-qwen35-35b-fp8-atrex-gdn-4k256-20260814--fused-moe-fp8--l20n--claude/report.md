# Fused MoE FP8 生产消融实验报告

## 实验概览

- 实验：`production-qwen35-35b-fp8-atrex-gdn-4k256-20260814--fused-moe-fp8--l20n--claude`
- Workload：`fused_moe_fp8`
- DSL：CUDA、Triton、CuteDSL
- Runtime 每个 DSL：1 个完整演化臂、1 个 retained 臂、1 个 pooled 臂、2 个 isolated 重复臂；另加入 `AKA-1`、`AKA-2` 两个独立消融臂
- Runtime 完成情况：15/15 个臂完成，150/150 个 Epoch 完成，474/474 个 Optimizer Session 完成，27/27 个 Evolver Session 完成；两个 AKA 实例各完成 3/3 个 DSL、57/57 个 Episode
- Runtime 统计明确排除 Bootstrap 和 Problem Generalization，只统计 `optimizer` 与 `evolver` Session；AKA 行使用独立的 Episode 口径
- Runtime 15 个臂并发运行的总墙钟跨度：2026-09-01 04:41:14 UTC 至 2026-09-02 07:28:59 UTC，共 **26h 48m**
- Runtime 非 Bootstrap Agent Session 累计时间：**483.57 Agent-hours**；该值对并发 Session 求和，不等于墙钟时间

## 摘要

| DSL | 最佳臂 | 最终最佳 GeoMean latency | 相对基线加速 |
|---|---|---:|---:|
| CUDA | `ablation-pooled` | **1,086.193 µs** | **24.58×** |
| Triton | `ablation-isolated-01` | **1,044.368 µs** | **1.61×** |
| CuteDSL | `ablation-retained` | **1,125.741 µs** | **269.84×** |

完整演化臂在 Runtime 的五个臂中，于 CUDA、Triton、CuteDSL 分别排名第 3、第 3、第 4。本轮没有观察到 Evolver 相对所有 Runtime 消融臂的一致优势，但每个配置只有一条 Lineage；两个同配置 isolated 重复臂在 CuteDSL 上相差 30.8%，说明单 Lineage 方差足以吞没多数臂间差异，因此不能据此得出 Evolver 无效的统计结论。

另外加入两个 AKA 消融臂：`AKA-1` 从 V0 开始独立运行，`AKA-2` 从已清理历史的同一 V1 pinned baseline 开始独立探索；两者均为 L20N/`sm_120`、production、Claude CLI、三个 DSL、`max-iters=20`，且关闭 Codex/Qoder 外部 plan reviewer。AKA 最佳值只取 `measurement_source=authoritative_verification`。AKA 的 **Epoch wall** 取 E1 `created_at` 至 E19 `finalized_at`；Token 从原始 Claude Session transcript 按 `message.id` 去重，包含 19 个 Episode 的全部 invocation/resume、嵌套子 Agent 和独立 production policy review，但排除 V0/V1 Bootstrap，并使用与 Runtime 相同的 Qwen3.8-Max 单价折算。AKA 没有与 Runtime Worker Session 完全等价的 Agent-h，因此该列记为 `—`。AKA-2 Triton 有一次超时 policy review 仅留下 221,597 total tokens、缺少四类分项，故其 Total 包含该值、费用以最小 cache-read 至最大 output 的区间表示。归档位于 `/oss/duoxing/atrex-runs/production-qwen35-35b-fp8-atrex-gdn-4k256-20260814--fused-moe-fp8--l20n--claude/AKA/`。

## 统计与计费口径

### 性能

“最终最佳 GeoMean latency”取 `lineages.best_kernel_revision_id` 对应 Kernel 的正确 `kernel_measurements.latency_us` 最小值。该字段是一次权威评测在隐藏 Shape 集合上的 latency GeoMean；取最小值与实验运行时使用的 “best latency” 监测口径一致。

### 时间

- **Epoch wall**：该 Lineage 的 Epoch 1 `created_at` 到 Epoch 10 `completed_at`。包含 Evolver、Optimizer、Gateway 评测、选择、并发等待和中断恢复，但排除 Bootstrap。
- **Agent-h**：该 Lineage 所有 `optimizer` 与 `evolver` Worker Session 的 `completed_at - started_at` 之和。并发 Session 会重复计时；不包括 Session 外部的 Gateway 等待。
- **O/E sessions**：Optimizer Session 数 / Evolver Session 数。

### Token（包含子 Agent）

Token 不使用 `worker_sessions` 中的顶层 `result.usage`，而是逐个读取 501 份原始 Session Artifact 的终态 Provider 事件，并汇总 `modelUsage["qwen3.8-max"]`：

- 501/501 个非 Bootstrap Session 都只有一个模型键 `qwen3.8-max`，0 个缺失；
- `modelUsage` 是 Provider 对该 Session 内全部模型执行的累计值，包含主 Agent、子 Agent 和其他子请求；
- 与 Registry 顶层 usage 相比，它额外计入 **114,249,418 tokens（+2.55%）**，其中 uncached input +81,494,327、cache read +25,739,587、cache write +1,020,510、output +5,994,994；
- `Total = uncached input + cache read + cache write + output`，四类互斥相加。

Runtime 本身不会为 Provider 管理的子 Agent 单独登记 Worker Session，因此无法列出每个子 Agent 的 ID 或逐子 Agent 成本；终态 `modelUsage` 是当前可获得的最完整累计口径。

### Qwen3.8-Max 官方价格折算

官方模型 ID 是 `qwen3.8-max`。所有 501 个终态事件的 `canonicalModel` 都是该模型；`worker_sessions.backend = claude` 表示使用 Claude CLI 协议适配器，并不表示底层模型是 Claude。

本报告按阿里云百炼中国区（华北 2，北京）原价、显式上下文缓存口径折算：

| Token 类别 | 单价（元 / 1M tokens） |
|---|---:|
| Uncached input | 12 |
| Explicit cache hit / cache read | 1 |
| Explicit cache creation / cache write | 15 |
| Output | 36 |

价格公式：

```text
cost_cny = uncached_input / 1M × 12
         + cache_read    / 1M × 1
         + cache_write   / 1M × 15
         + output        / 1M × 36
```

这是按官方原价对 Token 账本的折算，不等于 Provider 账单中的 `total_cost_usd`，也未计免费额度、限时折扣、税费或汇率。官方还列出隐式缓存命中价 1.5 元 / 1M tokens；若把全部 cache-read 都按隐式缓存计费，总价将由 **¥9,493.27** 增至 **¥11,658.51**。本实验有独立 `cacheCreationInputTokens` 与 `cacheReadInputTokens`，因此主表采用显式缓存创建/命中价格。

官方资料：

- [qwen3.8-max 模型信息](https://help.aliyun.com/zh/model-studio/qwen3-8-max)
- [模型调用价格](https://help.aliyun.com/zh/model-studio/model-pricing)
- [上下文缓存（Context Cache）](https://help.aliyun.com/zh/model-studio/context-cache)

## CUDA

基线：26,694.1 µs。

| Arm | GeoMean µs | Epoch wall | Agent-h | O/E sessions | Uncached in M | Cache read M | Cache write M | Output M | Total M | Cost ¥ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| evolve (active+challenger) | 1,295.489 | 24h 58m | 38.12 | 38/9 | 7.176 | 351.988 | 11.803 | 5.126 | 376.093 | 799.67 |
| ablation-retained | 1,199.026 | 23h 21m | 40.45 | 40/0 | 7.181 | 374.746 | 10.299 | 5.417 | 397.642 | 810.39 |
| ablation-pooled | **1,086.193** | 24h 09m | 41.73 | 40/0 | 7.059 | 384.431 | 10.506 | 5.619 | 407.615 | 829.02 |
| ablation-isolated-01 | 1,655.317 | 20h 42m | 20.57 | 20/0 | 3.754 | 186.091 | 5.391 | 2.721 | 197.958 | 409.97 |
| ablation-isolated-02 | 1,517.628 | 22h 47m | 22.68 | 20/0 | 4.137 | 189.557 | 5.522 | 3.026 | 202.242 | 430.96 |
| AKA-1 | 1,275.351 | 26h 23m | — | 19 Ep/0 Ev | 0.017 | 265.896 | 8.526 | 3.164 | 277.603 | 507.91 |
| AKA-2 | 1,161.882 | 35h 09m | — | 19 Ep/0 Ev | 2.258 | 353.816 | 10.632 | 3.946 | 370.651 | 682.44 |

Runtime CUDA 合计：**1,581.549M tokens、163.55 Agent-hours、¥3,280.01**。AKA-1：19 Episode，12 晋升/7 Pivot/0 拒绝/0 协议失败；AKA-2：19 Episode，12 晋升/6 Pivot/0 拒绝/1 协议失败。

AKA 的 `19 Ep/0 Ev` 表示 19 个优化 Episode、无 Evolver；Token 与费用包含嵌套子 Agent 和独立 policy review，但排除 V0/V1 Bootstrap。

## Triton

基线：1,686.1 µs。

| Arm | GeoMean µs | Epoch wall | Agent-h | O/E sessions | Uncached in M | Cache read M | Cache write M | Output M | Total M | Cost ¥ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| evolve (active+challenger) | 1,080.145 | 24h 16m | 38.85 | 38/9 | 6.358 | 333.799 | 10.174 | 5.112 | 355.442 | 746.72 |
| ablation-retained | 1,138.087 | 17h 58m | 30.98 | 40/0 | 4.498 | 263.118 | 7.627 | 3.975 | 279.218 | 574.59 |
| ablation-pooled | 1,095.815 | 21h 10m | 38.55 | 40/0 | 5.801 | 293.995 | 9.384 | 5.188 | 314.368 | 691.13 |
| ablation-isolated-01 | **1,044.368** | 21h 00m | 20.91 | 20/0 | 3.235 | 193.464 | 4.951 | 2.746 | 204.396 | 405.39 |
| ablation-isolated-02 | 1,058.340 | 21h 58m | 21.87 | 20/0 | 3.808 | 177.817 | 5.348 | 2.906 | 189.879 | 408.35 |
| AKA-1 | 1,105.962 | 29h 02m | — | 19 Ep/0 Ev | 0.125 | 311.289 | 9.500 | 3.364 | 324.277 | 576.38 |
| AKA-2 | 1,293.979 | 22h 28m | — | 19 Ep/0 Ev | 0.723 | 216.481 | 6.900 | 2.613 | 226.939‡ | 422.96–430.72‡ |

Runtime Triton 合计：**1,343.303M tokens、151.15 Agent-hours、¥2,826.18**。AKA-1：19 Episode，7 晋升/10 Pivot/0 拒绝/2 协议失败；AKA-2：19 Episode，5 晋升/9 Pivot/3 拒绝/2 协议失败。

AKA 的 `19 Ep/0 Ev` 表示 19 个优化 Episode、无 Evolver；Token 与费用包含嵌套子 Agent 和独立 policy review，但排除 V0/V1 Bootstrap。‡ AKA-2 有一次超时 policy review 仅留下 0.222M total、无四类分项；Total 已计入，费用范围按全部为 cache read 至全部为 output 计算。

## CuteDSL

基线：303,764.9 µs。

| Arm | GeoMean µs | Epoch wall | Agent-h | O/E sessions | Uncached in M | Cache read M | Cache write M | Output M | Total M | Cost ¥ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| evolve (active+challenger) | 1,391.746 | 23h 29m | 36.37 | 38/9 | 6.073 | 324.754 | 11.093 | 4.797 | 346.718 | 736.72 |
| ablation-retained | **1,125.741** | 26h 30m | 42.85 | 40/0 | 8.059 | 411.715 | 11.118 | 5.744 | 436.636 | 881.98 |
| ablation-pooled | 1,364.712 | 26h 46m | 44.72 | 40/0 | 7.332 | 415.604 | 11.016 | 5.819 | 439.772 | 878.34 |
| ablation-isolated-01 | 1,713.009 | 21h 48m | 21.64 | 20/0 | 3.364 | 206.891 | 5.175 | 2.880 | 218.309 | 428.55 |
| ablation-isolated-02 | 1,309.801 | 23h 30m | 23.30 | 20/0 | 3.933 | 222.512 | 5.568 | 3.007 | 235.020 | 461.48 |
| AKA-1 | 1,626.643 | 33h 43m | — | 19 Ep/0 Ev | 0.501 | 317.529 | 9.866 | 3.884 | 331.779 | 611.34 |
| AKA-2 | 1,205.153 | 35h 54m | — | 19 Ep/0 Ev | 0.020 | 325.186 | 13.121 | 3.701 | 342.028 | 655.48 |

Runtime CuteDSL 合计：**1,676.455M tokens、168.87 Agent-hours、¥3,387.08**。AKA-1：19 Episode，10 晋升/7 Pivot/1 拒绝/1 协议失败；AKA-2：19 Episode，13 晋升/5 Pivot/0 拒绝/1 协议失败。

AKA 的 `19 Ep/0 Ev` 表示 19 个优化 Episode、无 Evolver；Token 与费用包含嵌套子 Agent 和独立 policy review，但排除 V0/V1 Bootstrap。

## Runtime 总 Token 与费用

以下合计仅包含 15 个 Runtime 臂，以保持原消融总账口径。AKA 完整 Token 已列在各 DSL 表中：AKA-1 合计 **933.660M tokens、¥1,695.62**；AKA-2 合计 **939.618M tokens、¥1,760.88–¥1,768.64**。AKA-2 的费用区间来自一次缺少四类分项的 0.222M-token 超时 policy review。

| 类别 | Tokens | 百万 Tokens | 单价（元 / M） | 费用（元） |
|---|---:|---:|---:|---:|
| Uncached input | 81,769,619 | 81.770 | 12 | 981.24 |
| Cache read | 4,330,482,750 | 4,330.483 | 1 | 4,330.48 |
| Cache write | 124,972,679 | 124.973 | 15 | 1,874.59 |
| Output | 64,082,298 | 64.082 | 36 | 2,306.96 |
| **合计** | **4,601,307,346** | **4,601.307** | — | **9,493.27** |

Cache read 占物理 Token 总量的 94.11%，但只占估算费用的 45.62%；Output 只占 Token 的 1.39%，却占费用的 24.30%。Cache write 占 Token 的 2.72%，占费用的 19.75%。

## 解释限制

1. 每个非 isolated 配置只有一条 Lineage；isolated 虽有两个同配置重复臂，也只有两个样本。因此排名是描述性结果，不是统计显著性结论。
2. 各臂预算不完全相同：evolve 每 DSL 有 38 个 Optimizer + 9 个 Evolver Session；retained/pooled 各有 40 个 Optimizer；每个 isolated 各有 20 个 Optimizer。不能只看最终 latency 而忽略成本。
3. Epoch wall 包含并行与等待；Agent-h 是并发 Session 时长之和。两者回答不同问题，不应相加或互相替代。
4. 子 Agent Token 使用 Provider 的终态累计 `modelUsage` 统计，优于 Registry 顶层 usage；但 Provider 没有在 Runtime 中暴露可独立核验的子 Agent ID 清单。
5. 费用使用 2026-09-02 检索到的中国区官方原价，后续官方调价会改变折算结果。
6. AKA 的墙钟按 E1 至 E19 计算，`19 Ep/0 Ev` 不是 Runtime Worker Session 数；AKA 与 Runtime 的 Agent-h、Session 并发结构不可直接比较。
7. AKA-2 Triton 有一次 0.222M-token 超时 policy review 缺少四类 Token 分项，因此该臂和 AKA-2 总费用报告为上下界。
