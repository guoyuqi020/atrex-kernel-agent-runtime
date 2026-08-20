# 0034：可配置 Epoch 拓扑

[English](0034-configurable-epoch-topology.md) | 中文

## 状态

已接受。

## 背景

固定的 Active 对单 Challenger Epoch 混合了三个独立预算：比较多少种 Agent 设计、每种设计
获得多少条独立 Kernel 搜索路径，以及每条路径能连续学习多长。把独立路径也称作 Lineage，
还会与持有 Agent/Kernel 历史的持久 DSL Lineage 冲突。

## 决策

每个持久 DSL Lineage 配置：

- `challenger_count`（`K`，大于等于零）；
- `challenger_start_epoch`（正数，默认 `1`）；
- `trajectories_per_branch`（`Y`，正数）；
- `attempts_per_trajectory`（`X`，正数）。

一个 Epoch 冻结一个 Active Agent、一个起始 Kernel 和一个 Evidence Checkpoint。在
`challenger_start_epoch` 之前，有效 `K` 为零；从该 Epoch 开始，Runtime 串行调用 Evolver
`K` 次，每次调用只创建一个 Challenger。第 `i` 次调用获得只读 Catalog，其中
包含 Active Revision、已保留 Agent 历史和 Challenger `1..i-1`；尚未生成的未来 Challenger
不可能提前可见。

Epoch 有 `1 + K` 个 Branch Slot。Active 的 Challenger Ordinal 为零；每个 Challenger 使用从一
开始的创建序号。每个 Branch 从相同起始 Kernel 启动 `Y` 条 Trajectory。Trajectory 彼此独立
并可并发运行；每条 Trajectory 内串行执行 `X` 个 Attempt，被保留的结果只成为该 Trajectory
下一个 Attempt 的输入。每个 Attempt 都是全新 Agent Session。因此，一个 Epoch 包含
`(1 + K) × Y × X` 个 Optimizer Session 和 `K` 个 Evolver Session。

Kernel 选择考虑全部 Trajectory 的保留结果。Agent 晋升依次比较 Active Score 与所有
Challenger Score，完全相同时保留当前者。Runtime 只在整个 Epoch 完成选择后发布一个 Evidence
Checkpoint。Attempt Evidence 只暴露同一 Trajectory 内更早的 Attempt；完成后的 Epoch Evidence
保留全部已测量 Branch 和 Trajectory Outcome。

## 结果

`K=0` 可以在保留 Epoch 边界的同时彻底关闭 Evolution。提高 `Y` 可增加独立搜索覆盖率，又
不会在路径间泄漏中间 Kernel 状态；提高 `X` 则加深单条路径的串行学习。Session 公式使资源
预算明确。Registry schema 19 通过把 `challenger_start_epoch` 迁移为 `1` 来保留已有 Lineage
行为。
