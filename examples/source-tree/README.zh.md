# 多文件源码树优化

[English](README.md) | 中文

这个示例只生成本任务自己的 Campaign、Evaluation Contract、公开 Shape Train 和固定适配器，
不会启动 Runtime、模型或 GPU 作业，也不复用其他 example 的配置。使用已经启动的 Runtime
服务及其匹配的 `ATREX_RUNTIME_CONFIG`；部署需要配置 `gate_policy.evaluator`、Agent backend，
且 GPU 环境必须满足 Source Manifest 中的依赖。

源码树任务请在该 Runtime 配置中设置 `campaign.optimizer.max_session_tokens: 100000000`：
每次 Bootstrap 或 Optimizer Session 独立允许 100M token，适用于所有消融臂，不是整个 Epoch
共用的额度。GDN 输入包已经配置此值。本准备脚本不会修改传入的 Runtime 配置；请使用与
单文件任务分开的配置，以保留单文件配额。Evolver 仍不限 token。修改后需重新启动 Campaign
进程读取，正在运行的 Session 不会热更新。

GDN 的完整命令见 [英文示例](README.md)。准备时指定：

- `--source-manifest`：初始 seed 的 `task/source_manifest.json`。
- `--source-repository`：初始 seed 的 `source` Git 仓库；严格按 Manifest 中的 commit 导入。
- `--task`：包含 `reference.py`、`input.py`、`shape_train.json`、`shape_valid.json` 的算子目录。
- `--optimizer-commit`：Runtime 配置的 Optimizer base repository 中的完整 commit。
- `--hardware-target`：支持原 SM103 算子的 Agate 环境，不能直接假定为 L20N。
- `--output`：新目录；不会覆盖旧运行。恢复时继续使用原生成文件。

准备后，使用公共源码树入口（不启动/停止 Runtime 或 Wiki）：

```bash
python scripts/source-tree/run.py \
  --config "$ATREX_RUNTIME_CONFIG" \
  --campaign workspaces/gdn-source-tree/campaign.json \
  --plan workspaces/gdn-source-tree/ablation.json \
  --workspace workspaces/gdn-source-tree/run --target-epoch 5
```

默认只用 CuteDSL，与单文件生产共享计划生成器。
当前计划为四种 Direction/Experiment 工具配置（全开、全关、仅 Experiment、仅 Direction）
各启用两种互通模式。每个配置/模式对应一个 Campaign/Lineage，内含三条保留 State 的 Trajectory。
共八个臂，复用同一份冻结的 Bootstrap v0，不重复 Baseline 测量，不运行 Evolver。
默认每臂 5 个 Epoch，每条轨迹每轮串行执行 3 次 Attempt：每轨迹 15 次、每臂 45 次、
全套 360 次 Optimizer Attempt，不含 Bootstrap。

| 模式 | Workflow | Epoch 内 | Epoch 结束后 |
|---|---|---|---|
| `epoch-shared` | `epoch_shared_3.py` | 各轨迹只读自身本轮历史，延续自身 Kernel | 三条轨迹共享已完成历史及选出的最佳 Kernel |
| `broadcast` | `broadcast_3.py` | 已记录的评测、Artifact 和启用的 Journal 实时互通；每个 round 广播截至当时最佳的已接受 Kernel | 三条轨迹共享已完成历史及选出的最佳 Kernel |

每种模式对应 `ablation-MODE-3`、`ablation-MODE-no-modules-3`、
`ablation-MODE-experiments-3`、`ablation-MODE-directions-3` 四个臂。
Broadcast 的后续 Attempt 文件快照也包含其他轨迹已完成的前序 round 的 report 和对话。
工作源码、scratch、实时对话保持私有；Tool State 按轨迹独立路由。
历史按需读取，不要求扫描全部旧对话。不同消融臂不共享 Bootstrap 之后的历史。

Pool-Retained、单轨迹 Retained、Retained-Evolve 等 Workflow 保留实现，但不进入新计划。
每个启用臂持有 Lineage-local `agent-v0`，固定自身 Workflow；Runtime 不根据 Label 推断拓扑。
实际并发数受部署配置限制。

各臂从同一份冻结的源码树 v0 开始，对照臂不重复 Bootstrap/评测。

`run/` 保存各臂 seed 身份、日志、结果和 `campaign-results.json` 汇总；真正的 Session/Artifact
仍在配置的 Runtime storage。一个臂失败不取消其他臂，重复运行恢复相同身份。
新计划固定每轨迹 15 次；`--target-epoch` 仅为兼容可能启用主臂的旧计划而保留。
已有工作区保留原计划；恢复旧主臂时应显式传入原来的绝对目标轮次。
八个臂合计 24 条轨迹；实际并发受部署配置限制。只跑主臂时仍可直接调用 `bootstrap` 和
`run-campaign` 两条原始 CLI 命令。

Bootstrap 启动配置的 Agent，在原始源码上完成评测、Journal 和标准报告，
随后 Runtime 按分阶段 Gate 独立终评注册 v0；固定适配器不可修改。
原 Manifest 的 `measurement`、`bringup`、旧 Repository Horizon 控制逻辑不会覆盖 Runtime。

Agent 仍用原 Core/KDA Runtime Tools 调用 `evaluate`。支持普通完整评测、`correctness_only`、
自定义 `input_path` / `shapes_path`、以及带 `comparison.baseline_path` 的 ABBA。
完整评测必须先填 `latency_prediction`：几何平均延迟降低超过 1% 为 `improved`，变化在
±1% 以内（含边界）为 `retained`，升高超过 1% 为 `degraded`。ABBA 比较 B 与 A；普通
评测比较候选与本次 Attempt 开始时的 Kernel。`correctness_only` 无需预测。Runtime
只在私有事件中保存预测供人工评测，Agent 返回结果不显示该字段。
`comparison.baseline_path` 必须指向整棵历史 Kernel 目录。只有适配器路径固定为 `work/kernel/kernel.py`；
其他源码直接保留在 `work/kernel/` 下，没有额外 `source/` 层。

```json
{"operation":"evaluate","latency_prediction":"retained"}
```

```json
{"operation":"evaluate","latency_prediction":"improved","comparison":{"method":"abba","baseline_path":"scratch/previous-kernel-tree","repeats":2}}
```

NVIDIA 源码树诊断也可以直接调用原工具：

```json
{"operation":"profile","level":"sol"}
{"operation":"profile","level":"deep","kernel_regex":".*GatedDelta.*","source":true}
{"operation":"check","sanitize":"memcheck"}
{"operation":"disassemble","fmt":"sass"}
```

这些诊断由 Runtime 将整个源码树和固定驱动交给 Agate Dev，不需要 Agent 本地执行 GPU 命令；
普通 Evaluate 和 ABBA 则使用原生 Eval 源码归档。
GPU 镜像需要提供 NCU（Profile/Disassemble）和 Compute Sanitizer（带 sanitize 的 Check）。
Check 是单个 case 的编译/运行探针，不代表完整正确性通过。即使操作 completed，仍需检查
诊断 `passed`；PTX 导出依赖工具链。详见[源码树设计和接口](../../docs/source-trees.zh.md)。
