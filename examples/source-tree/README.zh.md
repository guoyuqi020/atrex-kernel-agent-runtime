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
  --workspace workspaces/gdn-source-tree/run --target-epoch 100
```

默认只用 CuteDSL，但与单文件生产一样启用三个 Retained 重复、一个含三条 Trajectory 的
Pool-Retained，以及三个 Retained-Evolve 重复。Isolated、Pool-3、旧的 Evolve-3、
Isolated-Evolve 和 Isolated-Pool-Evolve Workflow 实现保留但停用。
生成的 `ablation.json` 及仓库中的 `ablation.example.json`
与单文件共享计划生成器。各臂复用一次 Bootstrap 的冻结源码树 v0，对照臂不重复评测。

默认每臂 100 个 Epoch、每轨迹每轮 3 次 Attempt。Retained 和 Retained-Evolve
每实例一条轨迹，各 300 次；唯一的 Pool-Retained 有三条轨迹，共 900 次。
所有启用臂均保留自适应 State。Pool 在 Epoch 边界共享最佳 Kernel，各轨迹继续继承自己的终态
State，不做合并。不同臂不共享后续历史或可写文件。
三个 Retained-Evolve 各运行一条 Active Trajectory，不做同轮 Agent 对比。Epoch N 结束后，
它们等待对应 Retained 的 Epoch N Evidence，再生成 Epoch N+1 使用的 Agent；最后一轮也会留下
可供后续继续运行的 successor。
全套共 2,700 次 Optimizer Attempt，不启动外部原版 AKA。

`run/` 保存各臂 seed 身份、日志、结果和 `campaign-results.json` 汇总；真正的 Session/Artifact
仍在配置的 Runtime storage。一个臂失败不取消其他臂，重复运行恢复相同身份。
新计划固定每轨迹 300 次；`--target-epoch` 仅为兼容可能启用主臂的旧计划而保留。
已有工作区保留原计划；恢复旧 5 轮实验且不扩展主臂时，显式传入 `--target-epoch 5`。
单文件模式默认值不变。
最多并行 21 个 Optimizer，请预留宿主内存。只跑主臂时仍可直接调用 `bootstrap` 和
`run-campaign` 两条原始 CLI 命令。

Bootstrap 启动配置的 Agent，在原始源码上完成评测、Journal 和标准报告，
随后 Runtime 按分阶段 Gate 独立终评注册 v0；固定适配器不可修改。
原 Manifest 的 `measurement`、`bringup`、旧 Repository Horizon 控制逻辑不会覆盖 Runtime。

Agent 仍用原 Core/KDA Runtime Tools 调用 `evaluate`。支持普通完整评测、`correctness_only`、
自定义 `input_path` / `shapes_path`、以及带 `baseline_path` 和 `comparison` 的 ABBA。
`baseline_path` 必须指向整棵历史 Kernel 目录。只有适配器路径固定为 `work/kernel/kernel.py`；
其他源码直接保留在 `work/kernel/` 下，没有额外 `source/` 层。

NVIDIA 源码树诊断也可以直接调用原工具：

```json
{"operation":"profile","level":"sol"}
{"operation":"profile","level":"deep","kernel_regex":".*GatedDelta.*","source":true}
{"operation":"check","sanitize":"memcheck"}
{"operation":"disassemble","fmt":"sass"}
```

Runtime 自动将整个源码树和固定驱动交给 Agate Dev，不需要 Agent 本地执行 GPU 命令。
GPU 镜像需要提供 NCU（Profile/Disassemble）和 Compute Sanitizer（带 sanitize 的 Check）。
Check 是单个 case 的编译/运行探针，不代表完整正确性通过。即使操作 completed，仍需检查
诊断 `passed`；PTX 导出依赖工具链。详见[源码树设计和接口](../../docs/source-trees.zh.md)。
