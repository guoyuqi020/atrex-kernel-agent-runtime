# 保留原始提示的 GDN 输入

[English](README.md) | 中文

这是 [`data/GDN`](../GDN/README.zh.md) 的独立对照输入包，保留原始优化提示。
本目录只存输入和模板；实际配置、源码副本、数据库、Session、日志和结果写入
`workspaces/GDN-full/`，也可显式指定其他工作区。

恢复内容：

- 原始 **M64-oriented** objective 和 SM103 表述。
- 全部 `range_evidence`、`value_evidence`、`coverage_regimes`，包括低并行度描述和 M64 场景名。
- seed 来源说明及初始证据里提到 M64 的原始文字。
- 原始 seed commit `a39405536f178689d7f60b551c17b2252bcee61d`，包含在离线 `source.bundle` 中。

`task/shape_train.json` 与最初导入的文件逐字节一致，SHA-256 为
`7695093a901dc50f595ba0b0cd95df273882a3f14c5c457f6b050789d9437df9`。
Kernel 源码与清理版完全相同，两个 seed 树仅 `UPSTREAM_PROVENANCE.json` 不同。
没有导入优化后的实现、获胜 Kernel、历史 Session 或实验经验。Reference、输入生成器、
adapter、输入范围及测试用例、Metadata、Roofline、Gate 策略和 Agent commit 均不变。
Campaign 使用独立 creation key，不复用清理版的实验身份。

默认仍为 **L20D / CuteDSL / Claude**、5 个 Epoch、每条轨迹每轮 3 次 Attempt、
Optimizer/Bootstrap 每 Session 100M tokens，七臂消融方案不变。
“不屏蔽”指输入恢复原始内容，并不开放隐藏测试集，也不改变 Runtime/Core/KDA 的 Prompt
投影：例如 `range_evidence`、`value_evidence` 等元数据仍可能被已有 Agent 格式化逻辑省略。
本次不修改 Agent 实现或其固定 commit。

在 Lima Ubuntu 中激活 Linux Runtime 环境后执行：

```bash
# 只准备输入快照，不启动服务、Agent 或 GPU 作业
python scripts/gdn/prepare.py --inputs data/GDN-full --backend claude

# 后续：服务进程
python scripts/gdn/run.py serve --workspace workspaces/GDN-full
# 另一个终端：Bootstrap 加单 Campaign
python scripts/gdn/run.py campaign --workspace workspaces/GDN-full --target-epoch 5
# 或选择七臂消融
python scripts/gdn/run.py ablation --workspace workspaces/GDN-full
```

准备时默认工作区是 `workspaces/<输入目录名>`，可通过 `--workspace` 覆盖。
运行入口只使用该工作区的冻结输入。Campaign 和 ablation 是两种选择，不默认同时启动。
模板与 GDN 一样使用 Runtime 8766、Wiki 8091 端口，两个 Runtime 不能同时占用同一端口。
凭据、Worker 权限、调度和独立 Wiki 服务说明见[通用启动文档](../GDN/README.zh.md)。
