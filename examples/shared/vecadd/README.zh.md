# 标准 VecAdd Fixture

[English](README.md) | 中文

这是所有示例使用的唯一 VecAdd 数据集：

- `reference/`：直接调用 Agate 时使用的 PyTorch Reference、`_make_inputs` 和 Shape 元数据；
- `evaluation-contract.json`：同一可信 Workload 的 Runtime/Gateway 表示；
- `agent-problem.json`：Agent 可见的公开问题定义；
- `triton/baseline-kernel/`：Core Framework Baseline 使用的 Reference-shaped Seed；
- `triton/initial-evidence/`：可信 Epoch 0 Note；
- `triton/agate-candidate/`：直接演示 Agate CLI 的正确 Triton Candidate。

所有可运行示例只读使用本目录，并把生成状态写到 `workspaces/`。
