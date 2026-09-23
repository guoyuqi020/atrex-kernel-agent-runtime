# 单 Epoch Lineage 示例

[English](README.md) | 中文

默认连接远端 Agate 服务；可显式覆盖环境变量。

这个示例让一个 DSL Lineage 完整运行且只运行 Epoch 1。Campaign 向 Agent 自有 Workflow
授予零个 Challenger Slot 和恰好三个 Optimizer Attempt 的预算。选中的
`workflow/isolated.py` 程序把预算组织成一条 Trajectory、三个串行 Round，显式把每轮保留的
Kernel 路由给下一轮，并在每个 Attempt 前重置自适应 State。

因此 Runtime 恰好启动三个全新的 Optimizer Session，但不替 Workflow 决定拓扑或路由。
Workflow 不请求 Challenger，所以不需要 Evolver Session 或 Evolver Git Bundle。Optimizer
使用 Runtime 默认的 QoderCLI 凭据。

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

导出Agate 配置后运行：

```bash
export AGATE_URL="https://atrex-gateway.alibaba-inc.com"
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

搜索组织是可执行的 Agent 代码。测试其他组织方式时，应选择另一份 Workflow 实现，并在新的
Campaign 工作区中为它配置匹配的 `max_challengers` 与 `optimizer_attempt_budget` 资源上限。
