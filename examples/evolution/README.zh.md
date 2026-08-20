# 三 Epoch 自进化示例

[English](README.md) | 中文

这个示例运行一个 Triton Lineage 到 Epoch 3，并展示两次受控 Agent 进化：

- Epoch 1：只有 Active，运行一个 Optimizer Attempt；
- Epoch 2：Evolver 基于 Epoch 1 Evidence 创建一个 Challenger，Active 和 Challenger 各运行一个 Attempt；
- Epoch 3：Evolver 基于 Epoch 2 Evidence 再创建一个 Challenger，Active 和 Challenger 各运行一个 Attempt；
- 到达目标 Epoch 3 后停止，不创建 Epoch 4，因此不会再调用 Evolver。

配置为 `challenger_count=1`、`challenger_start_epoch=2`、
`trajectories_per_branch=1`、`attempts_per_trajectory=1`。这里“一次 Attempt”是每个 Branch
一次，因此本例总计 5 个全新 Optimizer Session 和 2 个 Evolver Session。Challenger 不会被无条件
覆盖到 Active；每个 Epoch 仍由 Runtime 根据独立评测结果决定是否晋升。

本目录自行持有 `runtime.json` 和三 Epoch `campaign.json`，只使用 `examples/shared/` 下的标准
VecAdd 输入与通用 Helper，不依赖其他可运行示例。

先准备远端 Agate、QoderCLI 和固定 Evolver Bundle 所需环境：

```bash
export AGATE_URL="https://..."
export AGATE_AK="..."
export AGATE_SK="..."
export AGATE_GPU="L20N"
export QODER_PERSONAL_ACCESS_TOKEN="..."
```

一条命令启动 Runtime、Bootstrap、运行三个 Epoch、展示历史并停止 Runtime：

```bash
bash examples/evolution/run.sh
```

Inspect 输出首先展示 Epoch Winner 表，其中逐轮列出 `ACTIVE_BEFORE`、`CHALLENGERS`、`WINNER`
以及 `active_retained` 或 `challenger_promoted` 决策。

也可以分步调试：

```bash
bash examples/evolution/prepare.sh
bash examples/evolution/start-runtime.sh
# 另一个终端，导出相同环境后：
bash examples/evolution/run-campaign.sh
bash examples/evolution/inspect.sh
```

状态默认保存在 `workspaces/evolution-example/`。再次运行会固定复用该工作区中的 Optimizer
和 Evolver Commit，并从第一个未完成的持久步骤恢复；若 Epoch 3 已完成，则只报告既有结果。
