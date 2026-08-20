# 单 Epoch Lineage 示例

[English](README.md) | 中文

这个示例让一个 DSL Lineage 完整运行且只运行 Epoch 1。默认配置为：

- `challenger_count=0`：不启动 Evolver Session，也不创建 Challenger Branch；
- `trajectories_per_branch=1`：从 Epoch 起始 Kernel 启动一条独立 Trajectory；
- `attempts_per_trajectory=3`：在这条 Trajectory 内串行运行三个 Optimizer Attempt。

因此，这个 Epoch 恰好会启动三个全新的 Optimizer Session。Kernel 被保留后会成为下一个
Attempt 的输入；如果被回退，下一个 Attempt 仍从此前保留的 Kernel 开始。
由于 `challenger_count=0`，Runtime 不会单独解析 Evolver 凭据，也不会导入它的 Git Bundle；
Optimizer 仍使用 Runtime 默认的 QoderCLI 凭据。如果把 Challenger 数量设为正数，同一凭据已可供
Evolver 使用。

`run-campaign` 会在每个 Attempt 持久完成后立即向 stderr 输出进度，例如：

```text
[2026-08-18T12:01:02+00:00] active trajectory 1 attempt 1 finished
[2026-08-18T12:03:04+00:00] challenger-1 trajectory 3 attempt 2 finished
```

`active` 和 `challenger-N` 表示参与竞争的 Agent Branch；`trajectory-N` 表示该 Branch 内的一条
独立优化链。并发 Trajectory 按实际完成顺序输出。进度不写入 stdout，因此保存的 Epoch Result
仍是单个有效 JSON 文档。

在交互式终端中，带时间戳的完成日志会保留在上方，下方原地刷新这样的进度图：

```text
Epoch 1 branch progress (lineage_...)
  active
    trajectory 1   [██░] 2/3
    trajectory 2   [█░░] 1/3
  challenger-1
    trajectory 1   [░░░] 0/3
    trajectory 2   [░░░] 0/3
```

stderr 被重定向时，Runtime 会自动退化为纯文本时间戳日志，不会写入终端控制字符。

本目录自行持有 `runtime.json` 和单 Epoch `campaign.json`，只使用
`examples/shared/vecadd/` 的标准输入，不复用 Bootstrap 示例的脚本或配置。

导出远端 Agate 配置后运行：

```bash
export AGATE_URL="https://your-agate-service"
export AGATE_AK="..."
export AGATE_SK="..."
export AGATE_GPU="L20N"
export QODER_PERSONAL_ACCESS_TOKEN="..."
bash examples/lineage/run.sh
```

脚本会准备独立状态目录；如果未设置 `ATREX_WIKI_URL`，它会自动启动并等待
`http://127.0.0.1:8091` 上的 Local Wiki。随后脚本启动 Runtime、按需 Bootstrap Triton
VecAdd Lineage、运行或恢复 Epoch 1、打印 Attempt/Kernel/Agent 历史，最后关闭由它启动的
Runtime 和 Local Wiki。显式设置 `ATREX_WIKI_URL` 可以改用已有的本地或远端 Wiki；脚本不会
关闭不是由它启动的 Wiki。结果保存在
`workspaces/lineage-example/`。如果 Epoch 1 曾被中断，重新运行会恢复它；如果 Epoch 1 已完成，
则直接报告现有结果，不会创建 Epoch 2。已有 Bootstrap 身份、Optimizer commit 和 Evolver
commit 都保持固定。只有在确实需要全新 Campaign 时才应移动或删除该示例工作区。

需要逐步调试时，先生成输入文件，并在第一个终端保持 Runtime 运行：

```bash
bash examples/lineage/prepare.sh
bash examples/lineage/start-runtime.sh
```

在具有相同 Agate 环境变量的第二个终端中，按需执行 Bootstrap，并运行或恢复 Epoch 1：

```bash
bash examples/lineage/run-epoch.sh
```

Inspect 命令可以离线执行：Epoch 完成后，它直接读取持久 Registry，不要求 Runtime 继续运行。

```bash
bash examples/lineage/inspect.sh
```

`inspect.sh` 会输出保存的调度结果、Epoch 胜者决策、所有已调度 Attempt（包括 `pivot`、`blocked` 等未产出
Candidate 的结果）、Kernel 版本表以及 Lineage Agent Revision 表。`X` 对应 Attempt 历史中
的行数，并不承诺产生 `X` 个新 Kernel 版本。

运行前可以通过 `ATREX_CHALLENGER_COUNT`、`ATREX_CHALLENGER_START_EPOCH`、
`ATREX_TRAJECTORIES_PER_BRANCH` 和 `ATREX_ATTEMPTS_PER_TRAJECTORY` 覆盖拓扑参数。
已有 Lineage 的拓扑不可变；测试不同参数时，请指定新的 `ATREX_LINEAGE_STATE_DIR`，或者先移动
原示例工作区。
