# Direction 谱系与 Pool 学习

[English](direction-genealogy-and-pool-learning.md) | 中文

## 状态

待实现。当前 Direction Journal 仍然是单个 Direction 生命周期的权威记录；本文提出的关系字段尚未成为
受支持的输入协议。

## 背景与问题

Pool 实验证明，并行 Trajectory 的作用不只是独立采样多个最终延迟。它们能够：

- 从同一个 Kernel 出发探索不同 Direction；
- 对同一个宏观 Direction 给出不同实现；
- 重新检查因噪声或过严局部条件而被另一条 Trajectory 否定的 Direction；
- 将未胜出 Trajectory 中的有效修改移植到胜出的 Kernel；
- 把多个已经独立测量的修改组合成新的 Direction。

Runtime 已经保存相关 Report、Experiment、Kernel Artifact 和 Result Artifact，但 Direction 之间的关系
通常只存在于 Agent 编写的自然语言中。语义上的重试、重新实现、纠错、移植或组合经常获得新的
Direction ID。因此，当前 Registry 无法可靠还原一个新 Direction 为什么产生、引用了哪些历史证据。

这也妨碍了准确归因：同一 Trajectory 的下一个 Attempt 修复错误结论，证明的是持久 Journal 和串行
Attempt 的价值；从兄弟 Trajectory 恢复修改，才是 Pool 特有的跨分支迁移。系统应当不依赖重新阅读
完整 Conversation 就能区分两者。

## 实验观察

以下观察来自 FlashAttention 的 Pool-3 运行记录。每个 DSL 的 Pool 由两条并行 Trajectory 组成；每条
Trajectory 在一个 Epoch 内串行执行三个 Attempt，然后 Runtime 选择本轮最好的正确 Kernel，作为下一
Epoch 两条 Trajectory 的共同起点。Pool-3 不继承分支内自适应 Runtime State，但 Runtime 中持久化的
Direction、Experiment、Kernel Artifact 和 Result Artifact 仍可被后续 Attempt 查询。

本次分析覆盖 Registry 中的 90 个 Pool Attempt，以及其中 66 份可用的终态 Attempt Report。24 个
Attempt 没有终态 Report，因此下面的 Direction 语义统计只覆盖 66 份 Report；Kernel 选择与性能数据
仍以 Registry 中的权威记录为准。

### Direction 总体统计

66 份 Report 共包含 299 个 Direction Event：

| Event | 数量 |
|---|---:|
| `propose` | 85 |
| `start` | 100 |
| `complete` | 67 |
| `abandon` | 38 |
| `defer` | 9 |

其中共有 104 个不同的 Direction ID，最终状态为 62 个完成、37 个放弃、5 个推迟：

| DSL | 完成 | 放弃 | 推迟 | 合计 |
|---|---:|---:|---:|---:|
| CUDA | 26 | 13 | 2 | 41 |
| Triton | 13 | 17 | 0 | 30 |
| CuteDSL | 23 | 7 | 3 | 33 |

只有 5/104 个稳定 Direction ID 跨越了多个 Attempt。大量“恢复、重新实现、移植、纠错、组合”会创建
新的 Direction ID，其谱系关系只存在于 Agent 编写的假设、分析和终态报告中。这正是本文 TODO 要
解决的核心可观测性缺口。

### Pool 是否改善了 Direction 选择

有明确证据。14 次产生改进的 Epoch 中，Trajectory 1 提供了 8 次胜出结果，Trajectory 2 提供了 6 次；
每个 DSL 都至少有两次由 Trajectory 2 胜出，不存在一条可以预先删除的固定弱分支。

| DSL | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 | Epoch 5 |
|---|---|---|---|---|---|
| CUDA | T1 | T2 | T1 | T2 | T1 |
| Triton | T1 | T1 | T2 | 保留 Incumbent | T2 |
| CuteDSL | T2 | T1 | T1 | T1 | T2 |

CUDA Epoch 4 是最清楚的方向投资组合案例。两条 Trajectory 从同一个约 `232.631 µs` Kernel 出发：

- T1 先研究中等规模 Shape，Profile 显示目标区域已经达到约 91%–93% DRAM SOL，于是放弃继续优化
  该区域，转向小规模 Decode 的 sub-wave split-KV 调度，并得到约 `0.735%` 的改善。
- T2 研究重 Prefill 的 warp specialization。第一版整体改善约 `2.62%`，但轻量 Decode Shape 退化
  9%–15%。它随后把实现修复为双 Kernel Dispatch，仅对 `splits == 1`、`pages_est >= 48` 且
  `tiles >= 64` 的重负载 Shape 使用 warp-specialized Kernel，最终得到约 `3.93%` 的改善且不再损害
  轻量 Shape。

Runtime 最终选择 T2，同时保留 T1 已测量的 Direction 和 Artifact。这说明 Pool 可以让保守方向和高风险
方向同时推进；高风险方向失败时仍有有效候选，高风险方向成功时也不会丢失另一条路线的局部发现。

从停滞长度也能观察到类似现象：CUDA 和 CuteDSL 的 Pool 最长无改进区间均为 2 个步长，而相同
Attempt 预算下的 Isolated Best-of-Two 分别为 4 和 6。Triton 的 Pool 最长无改进区间从 5 缩短为 3，
但最终性能反而比 Isolated 差约 1.1%，因此 Pool 只能降低陷入单一路径的概率，不能保证有限预算下必然
找到更好的结果。

### Pool 是否修复了 Direction 可行性的误判

存在跨 Trajectory 的直接案例。Triton Epoch 4 的 T1 测试 deep-shrink host wave override，六个预注册
条件通过五个：总体改善约 `0.229%`，Decode 区域改善约 `0.305%`，正确性通过，Deep Control 达到
1.2 倍，Prefill 基本不变。它最终因为一个理论持平单元被一次计时量化波动翻转，违反严格的零退化边界
而放弃该 Direction。报告已经判断这更可能是局部 Gate 设计问题，而不是机制不可行。

Epoch 5 的 T2 从持久历史中找到并按 Artifact 精确恢复该 Candidate，在新的评测之前冻结更符合测量
噪声的判定条件，然后重新验证并保留它，使权威延迟从约 `273.904 µs` 降至 `273.297 µs`。这证明
Pool 可以让兄弟分支重新审查“不可行”结论。由于收益较小，而且调整内部判定条件存在事后调参风险，
该案例更有力地证明了可恢复性，而不是显著的性能提升；最终接受仍由 Runtime Comparator 决定。

另一个 CUDA 案例发生在同一 Trajectory 的连续 Attempt 中：一个 Attempt 把两个 `fa_combine` 修改与
新的 `_pick_splits` Dispatch 一起测试，隐藏的 dense q-active Shape 退化，因而错误地推断 combine 修改
不能发布。下一个 Attempt 发现 combine 从未脱离有害 Dispatch 被单独测量，于是恢复 Incumbent Dispatch，
只移植 windowed register prefetch 和 generalized combine geometry，重新通过 90/90 Correctness 并获得
约 0.4%–0.7% 的改善。这个案例证明的是持久 Journal 和串行 Attempt 的价值，而不是 Pool 特有收益；
未来的关系模型必须能够区分这两类归因。

### Pool 是否能为同一 Direction 找到更好的实现

有直接证据。CuteDSL Epoch 1 的两个分支围绕同一个 Attention 优化 Direction 独立实现：

- T1 采用 pipeline、head-in-M GQA packing 和 split-KV，得到约 `288.106 µs`。
- T2 先排除 Host Overhead，再用不同的 compact/dynamic-start 处理实现 GQA packing、pipeline 和
  split-KV，随后加入 wide-M `BLOCK_M=128`，最终达到约 `258.874 µs` 并胜出。

Triton Epoch 1 也出现同类竞争：T1 使用 grouped split-KV、轻量 Host Dispatch 和 Regime/KV Gate；T2
使用通用 split-KV 和 compact global-token partial buffer。T2 的早期版本一度达到约 `550.506 µs`，而
T1 很快达到约 `290.885 µs` 并最终胜出。

因此，两个分支探索相同的宏观 Direction 不一定是无效重复。它能帮助系统区分“这个方向本身不好”和
“当前实现方式不好”。当前 Runtime 没有 Coordinator 主动分配不同方向；重复由随机性自然产生，有时
形成有效的实现级竞赛，有时也会浪费预算。

### Pool 是否能组合 Direction 并催生新 Direction

这是目前最强的 Pool 特有证据。

CUDA Epoch 4 中，T1 保留了约 `0.735%` 的 sub-wave split retuning，T2 保留了约 `3.93%` 的双 Kernel
warp specialization 并成为本轮胜者。Epoch 5 的 T1 创建了一个明确的新 Direction，把前者移植到后者：

1. 先证明两个 Dispatch 条件基本互斥：sub-wave 规则作用于 `tiles <= 27`，warp-specialized Gate 作用于
   `tiles >= 64 && splits == 1`；
2. 只移植 Host Dispatch 规则，保持 Device Kernel Source 不变；
3. 新 ABBA 改善约 `0.7286%`，几乎复现了兄弟分支原来的 `0.735%` 增量。

这是完整的“两个兄弟 Direction 分别测量，再产生多父节点组合 Direction”因果链，而不是简单选择两者
中较快的一个。

CuteDSL Epoch 2 至 Epoch 3 也出现跨分支组合。Epoch 2 选择了 T1 的 fp16 partial + split economy；未
胜出的 T2 留下 reversed work order、batched combine/grid clamp 和 compact work map 三项独立收益。
Epoch 3 的 T1 分别创建三个 Port Direction，将它们逐项移植到胜出的 Kernel 上，报告的增量依次约为
`1.409%`、`0.207%` 和 `0.205%`，最终权威 Epoch 延迟从约 `251.455 µs` 降到 `247.197 µs`。组合后
总收益小于简单相加，也让 Agent 发现这些机制存在重叠。

组合还可能暴露新的二阶 Direction。CUDA 合并 sub-wave split 和 warp specialization 后，Agent 推断更深
Split 增加了 Partial Traffic，因而尝试 2^-4 缩放的 FP16 Partial Buffer。Correctness 通过，但 Dev 结果
显示，在 2–4 CTA 的 Combine Grid 上，转换指令开销超过带宽收益，最终回退。该 Direction 没有产生
更快 Kernel，却形成了由组合后新瓶颈驱动、并被测量否定的新知识。

### Pool 是否能挽救未晋升的有效结果

CuteDSL 的一个兄弟分支曾生成 `cp.async .cg` L1 bypass + `COMB_CHUNK=12`，ABBA 显示约 `3.04%`
改善，但终态 Report/Handoff 失败，因此没有被选择。后续 Trajectory 扫描持久 Journal，找到测量结果为
`keep_after` 但 Artifact 不在 Active Tree 中的记录，按 Digest 恢复 Candidate，重新 Check/Evaluate，最终
把约 `247.197 µs` 降至 `239.798 µs`。

这说明 Pool 的失败分支可以成为后续搜索的组件库。前提是 Runtime 保存 Artifact 与权威测量，而不是只
保存当轮胜者。

### 性能与成本边界

在每个 DSL 都使用 30 个 Optimizer Attempt 的相同预算下，Pool-3 与 Isolated Best-of-Two 的结果为：

| DSL | Isolated Best-of-Two | Pool-3 | Pool 相对变化 |
|---|---:|---:|---:|
| CUDA | 233.838 µs | 222.558 µs | -4.8% |
| Triton | 270.268 µs | 273.297 µs | +1.1% |
| CuteDSL | 243.521 µs | 235.593 µs | -3.3% |

三个 DSL 的 Pool 共使用约 1,155.654M Provider Token，Isolated Best-of-Two 共使用约 1,279.457M，观察
值下降约 9.7%。这不是 Pool 的协议保证，也不能仅凭一次运行认定为因果收益。Pool 产生的最终 Candidate
数量也没有一致增加：CUDA 为 16 对 20、Triton 为 9 对 14、CuteDSL 为 18 对 11。因此 CUDA 的收益
不能简单解释为“Pool 生成了更多 Candidate”。

综合来看，当前 Pool 的实际形态是“Direction 竞赛 + 非胜出分支组件库 + 跨分支纠错循环”，而不只是
Best-of-Two。实证最强的是同一 Direction 的实现竞赛和跨分支组合；可行性误判修复也确实存在，但必须
防止事后修改局部 Gate。Triton 的反例表明，三个 Attempt 一次广播也可能过早淘汰需要更长探索周期的
Direction。

## 建议的数据模型

在不改变 Direction 稳定身份和生命周期的前提下，为其增加可选、受校验的关系：

```json
{
  "direction_id": "direction_...",
  "relationship": "combination",
  "derived_from_direction_ids": ["direction_A", "direction_B"],
  "supersedes_direction_id": null,
  "derived_from_experiment_ids": ["experiment_X", "experiment_Y"]
}
```

第一版关系类型建议包括：

- `retry`：因执行或基础设施失败而重新尝试相同假设；
- `refinement`：保留主要机制，同时缩小或改进假设；
- `reimplementation`：采用明显不同的实现验证同一机制；
- `correction`：重新检查此前对可行性或因果关系的判断；
- `port`：把已测量修改应用到不同 Kernel 节点或兄弟分支结果；
- `combination`：由两个或更多历史 Direction 构造新的假设。

所有关系必须引用 Registry 中的持久身份。关系不能把 Agent 分析升级为测量事实：Result Artifact 仍然
是 Observation 的权威来源，关系及其理由仍然属于 Agent 的分析声明。

## Runtime 与 Agent 行为

- `update-direction` 接受可选关系字段，并对不存在、不可见或不兼容的引用返回可操作的校验错误。
- 被引用的 Direction 和 Experiment 必须按照现有 Evidence Policy 对当前 Lineage 可见。
- 关系记录保持只追加；纠错通过新增事件完成，不能改写历史证据。
- Optimizer Prompt 应说明何时复用 Direction ID，何时创建派生 Direction。
- Evolver Evidence 可以汇总关系图，但必须保留到原始 Report、Experiment 和 Result Artifact 的引用。
- Inspect 和导出接口应能直接展示关系图，不要求解析 Conversation。

## 可支持的 Pool 分析

关系图应允许分别统计：

- 兄弟 Trajectory 之间的 Direction 多样性；
- 同一个 Direction 的独立实现；
- 被否定后又被纠正或恢复的 Direction；
- 从未胜出兄弟分支移植的修改；
- 组合后能够复现组成部分预期增量的新 Direction；
- 来自跨 Trajectory 迁移的收益，以及来自单条 Trajectory 串行探索的收益。

这些指标必须与权威 Kernel 测量一起解释；Direction 关系本身不能证明某个方向导致了性能变化。

## 验收标准

- Direction 可以声明零个或多个经过校验的父 Direction 与 Experiment。
- Registry 无需读取 Conversation，即可还原重试、细化、重新实现、纠错、移植和组合边。
- Inspect 可以展示一个 Epoch 或完整 Lineage 的 Direction 关系图。
- 系统能够区分跨 Trajectory 移植和同一 Trajectory 内的继续探索。
- 缺少关系字段的现有 Direction 记录仍然有效，无需为历史数据编造关系。
- 测试覆盖不可见引用、禁止的环、重复边、失败后重试、兄弟分支移植和多父节点组合。

## 非目标

- 不自动把 Agent 的因果解释当成事实。
- 不强制并行 Trajectory 探索不同 Direction。
- 不自动组合所有成功修改。
- 不替代 Kernel Retention、Agent Promotion、ABBA 或 Production Gate 的裁决。
