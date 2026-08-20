# 决策 0033：独立版本化自进化 Kernel Agent Revision

[English](0033-lineage-agent-version-labels.md) | 中文

## 状态

已接受并实现。

## 背景

Kernel Agent Revision 已有不透明 ID、Parent Link、创建时间与封存 Optimizer Artifact Digest，
可以精确标识内容，却无法让人直观阅读一条 Lineage 的 Harness 演进。它也不能复用 Kernel 的
`vN`：一个 Agent Revision 可以生产多个 Kernel；失败 Challenger 之后的新 Revision 还可能与
它拥有同一个 Active Parent。

## 决策

Registry schema 17 新增 `lineage_agent_versions`，在唯一 Lineage 内为每个 Agent Revision
分配不可变、从零开始的编号。Bootstrap 初始 Optimizer 固定为 `agent-v0`。只有可信控制器把
已验证 Evolver 输出挂接为 Epoch Challenger 时，才为其分配下一个编号。挂接会校验 Challenger
Parent 等于 Epoch Active Agent，版本映射和挂接在同一事务内提交。

映射保存引入 Epoch 与 Link Time；schema 16 迁移按 Epoch 顺序重建。Catalog 投影包含 Agent
与 Parent 版本、创建来源和时间、引入 Epoch、Active 标记、处置结果（`baseline`、
`challenger`、`promoted`、`rejected` 或 `failed`）、准确 ID、Optimizer Artifact 与
Trace/Provenance Digest。Kernel 投影同时返回生产它的 `kernel_agent_version`。

认证 API 以及 `list-agent-revisions`、`show-agent-revision` CLI 暴露该历史。JSON 仍是自动化
默认格式，`--format table` 用于人工查看。

## 影响

Kernel 与 Agent 计数器彼此独立。同一个 `agent-v0` 可以生产 `v0`、`v1`、`v2` Kernel。
Agent 数字相邻也不代表祖先关系：当 `agent-v1` 未晋升时，`agent-v2` 的
`parent_agent_version` 可以仍是 `agent-v0`。Git Commit 与 Artifact Digest 继续表示供应链和
实际执行内容身份，Lineage Label 只是稳定的人类可读投影。
