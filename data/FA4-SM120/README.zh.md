# FA4 SM120 实现任务

[English](README.md) | 中文

本题保留 `data/FA4` 的生产算子、Reference、输入生成器、Shape 集合、Evaluator 和正确性策略，
但目标改为 **L20N / SM120**。题目不再提供可编辑 Vendor 实现或 SM120 Bridge。

## Candidate 与 Reference 边界

Candidate 源码树结构为：

```text
kernel.py             固定生产 ABI Adapter
implementation/       可写的 SM120 实现
reference_sm103/      只读的原 SM103-family 实现参考
PROVENANCE.json       只读来源信息
```

`kernel.py` 校验捕获的 ABI，并调用 `implementation.sm120.flash_attention_sm120`。初始函数
故意没有实现；Bootstrap 必须创建第一个正确的 SM120 CuTe 实现。Reference 保留固定版本的
FlashAttention CuTe 和 Quack 源码供阅读，但不能作为运行时 Fallback 导入。Runtime 只允许修改
`implementation/`。

合约包括 FP8 E4M3 Query 与 P64 分页 KV、BF16 输出、16 个 Query Head、1 个 KV Head、
Head Dim 256、Ragged Batch、PackGQA、右下对齐因果 Mask，以及原地更新 `out`。返回值和
`out` 都使用精确逐元素条件：
`abs(candidate-reference) <= 0.06 + 0.04 * abs(reference)`。

Production 源码 Gate 已开启，因为它现在只扫描 Agent 自写的实现，而不会扫描只读 Reference
库。Bootstrap 和普通 Evaluate 使用统一 Runtime Gate；Retention 与 Agent Promotion 使用同
Allocation ABBA。Roofline 保持同一 Workload 语义，并使用 L20N/SM120 峰值。

## 准备与运行

在具备 Runtime、模型 CLI、bwrap 和 Agate 凭据的 Linux 环境运行：

```bash
cd ~/atrex-runtime
source env.sh
python3 scripts/source-tree/task.py prepare \
  --inputs data/FA4-SM120 \
  --workspace workspaces/FA4-SM120 \
  --backend claude
```

准备过程离线执行，不提交 GPU 任务。分别启动服务和 Campaign：

```bash
python3 scripts/source-tree/task.py serve --workspace workspaces/FA4-SM120
python3 scripts/source-tree/task.py bootstrap --workspace workspaces/FA4-SM120
python3 scripts/source-tree/task.py campaign --workspace workspaces/FA4-SM120 --target-epoch 1
python3 scripts/source-tree/task.py inspect --workspace workspaces/FA4-SM120
```

也可以运行标准消融：

```bash
python3 scripts/source-tree/task.py ablation --workspace workspaces/FA4-SM120
```

所有生成状态只写入指定 Workspace，不修改题目输入。
