# 可信 Roofline 构建

[English](roofline-builder.md) | 中文

Campaign Bootstrap 可以从部署批准的完整 Atrex Bench Git Commit 执行标准
benchmark-converter，为缺少 `roofline.json` 的 Evaluation Contract 自动构建 Roofline。这是可信
控制面阶段：Core 与 Evolver 都看不到私有 Shape，也不能编写 Roofline。

## 解析顺序

1. Evaluation Contract 已包含 `roofline` 时，Runtime 原样使用。
2. `roofline` 为 null 且配置了 `campaign.roofline_builder` 时，Runtime 构建一次。
3. Builder 未配置或构建失败时，Bootstrap 仍合法，Runtime 会封存不带 Roofline 的 Contract；
   此后每个正确的 Kernel 评测都依次执行 `eval` 与 `profile --level sol`。

对于已有 Campaign，如果输入中除 Roofline 外的字段没有变化，Runtime 会加载已经封存的生成版
Evaluation Contract，不会用更新后的 Builder 重算并悄悄改变 Campaign 指标。

## Builder 契约

配置的仓库在固定 Commit 上必须包含
`skills/benchmark-converter/scripts/generate_roofline.py`。Runtime 导出该 Commit 的普通 Git Tree，
拒绝链接与 Submodule，只向临时算子目录写入 `shapes.json` 和 `metadata.json`，再使用可选的
Hardware Target 到 SKU 映射调用 Converter。Converter 必须已经为该算子实现 `op_cost`。

生成结果只有满足以下条件才会被接受：

- Shape ID 与 Evaluation Contract 完全一致；
- 每个 Shape 都有非负有限的 `semantic_W_flops`、`semantic_Q_read_bytes` 与
  `semantic_Q_write_bytes`；
- 每个 Shape 至少有一个非负有限的 `SOL_time_ms` Hardware 值。

生成的 Roofline 会在 Agent Problem 泛化和任意 DSL Baseline Session 之前写入不可变 Campaign
Evaluation Contract。因此所有 DSL Lineage 共用同一个 Roofline，Agate 也始终用同一个封存理论
下界评测所有 Kernel。

## Profile 回退与运行进度

NCU Profile 失败不会推翻已经正确的 `eval` 延迟。Runtime 会把原始 Profile 响应（包括终态失败）
与原始 Eval 响应一起写入不可变 Gateway Result Artifact。SOL 使用每个已报告 Kernel 的
`max(compute_sol_pct, mem_sol_pct)`，再按 Kernel Duration 加权汇总。

Bootstrap 会打印本次是使用显式 Roofline、复用 Campaign 已封存 Roofline、成功生成新 Roofline，
还是进入 Profile 回退，并打印 Baseline 最终采用的 SOL 路径。Agent 请求的中间 Eval 和 Runtime
权威终评都会成对执行并分别封存；每个 Attempt 完成时会打印权威 Kernel 的执行结果。运行消息写到
stderr，CLI JSON 仍单独写到 stdout。

## 配置

`campaign.roofline_builder` 是可选项。生产配置示例：

```json
{
  "repository": "/srv/atrex-bench",
  "commit": "0123456789abcdef0123456789abcdef01234567",
  "git_executable": "/usr/bin/git",
  "python_executable": "/srv/atrex-runtime/.venv/bin/python",
  "fetch_timeout_seconds": 120,
  "execution_timeout_seconds": 120,
  "max_archive_bytes": 268435456,
  "max_output_bytes": 8388608,
  "sku_by_hardware_target": {
    "L20N": "NVIDIA RTX PRO 5000 72GB Blackwell (SM120)"
  }
}
```

Repository 是部署 Allowlist，完整 Commit 固定 Cost Model 实现，`sku_by_hardware_target` 避免把
Agate GPU Alias 错当成 Converter 默认 SKU。Metadata 缺失、算子 Cost Function 缺失、Shape
覆盖不完整、SKU 未知或 Converter 失败时会选择 NCU Profile 回退，并在 Bootstrap 输出中说明。
