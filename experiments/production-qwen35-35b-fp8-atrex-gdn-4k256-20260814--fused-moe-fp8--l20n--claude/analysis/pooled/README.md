# AKA / isolated / pooled / retained 对照数据

本目录保存主报告采用的 AKA、isolated、pooled 和 retained Session Trace 对照。AKA 的 `max-iters=20` 归档实际包含每条路线 19 个 Optimizer Episode，两遍、三个 DSL 合计 114 个；isolated 的两个重复实例、pooled 和 retained 各包含 120 个 Optimizer Attempt。Runtime 均取 Epoch 1–10。同一 DSL 从相同 Bootstrap Kernel seed 开始，Bootstrap 不计入。

## 文件

- `bash-action-index.json.gz`：AKA 的冻结调用标注与 isolated/pooled/retained 调用分类索引；记录 Trace、命令摘要和类别，不执行归档命令。
- `comparison.json`：按组、DSL、Session 和 Bash 用途汇总响应、工具、压缩、Bash 文本及相邻 thinking，并保存可回查的 Trace 路径和行号。
- 上级目录的 `pooled_comparison.py`：从只读 Registry 选择 isolated、pooled 和 retained Lineage，校验 isolated 每个实例的 20 个、pooled/retained 每个 DSL 的 40 个已完成 Optimizer Session 及共同 Kernel seed，再生成上述文件。
- 上级目录的 `checkpoint_summary.py` / `checkpoint-summary.json`：同时复算 Epoch 5 / Episode 10 的中间检查点和 Epoch 10 / 完整 AKA 运行的性能、用量与 Evolution 文件差异。

Provider 总量来自上级 `provider-usage-audit.json` 与 Runtime Session 终态 `modelUsage`。`comparison.json` 中 isolated/pooled/retained 的用量用于复核每轮均值和中位数；AKA 的完整用量还包含独立 Reviewer 与子 Agent，因此不从 Bash 文本估算反推。

## 文本统计口径

主词表为 `tiktoken 0.12.0 / o200k_base`。Bash 命令首次生成计一次，工具返回不计生成；命令和返回每保留到一次后续可见请求，计一次后续输入。压缩后仅保留 Trace 明确列出的消息，不把摘要反向分配给旧动作。

相邻 thinking 只取同一响应中调用前的 thinking，或工具结果后紧接着的 thinking；不跨人工消息、压缩和其他动作。它的后续读取是假设原文继续保留的文本场景，不是 Provider 账单，也不能解释为某个 Bash 调用独占的成本。

## 复现

```bash
TOKENIZER_ENV="$(mktemp -d /tmp/atrex-tokenizer.XXXXXX)"
python3 -m venv "$TOKENIZER_ENV/venv"
"$TOKENIZER_ENV/venv/bin/python" -m pip install 'tiktoken==0.12.0'

ARCHIVE_ROOT="$HOME/atrex-runs/production-qwen35-35b-fp8-atrex-gdn-4k256-20260814--fused-moe-fp8--l20n--claude"
"$TOKENIZER_ENV/venv/bin/python" ../pooled_comparison.py \
  --archive-root "$ARCHIVE_ROOT" \
  --output "$TOKENIZER_ENV/pooled"
```

脚本只读打开归档文件和 SQLite，不运行 Agent 留下的 Shell、Python 或 GPU 代码。输出目录不能位于原始归档中。
