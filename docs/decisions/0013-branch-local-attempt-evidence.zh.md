# 决策 0013：持久化分支本地 Attempt Evidence

[English](0013-branch-local-attempt-evidence.md) | 中文

## 状态

已接受并实现。

## 背景

每次 Optimizer 执行都是全新进程和 Session。只传递 Epoch 起点 Evidence 与当前 Branch 最佳 Kernel 可以避免进程上下文污染，但后续 Attempt 仍需学习同一 Trajectory 此前的评测、Diff 和注释。如果只在启动时重建历史而不持久化其身份，重启或配置变化后也无法准确确定当时的模型可见输入。

## 决策

插入 Attempt 前，可信 Runtime 会从同一 Epoch、同一 Trajectory 中较小 Ordinal 的连续已完成
前缀封存 `ATTEMPT_EVIDENCE` Artifact。内容包括权威输入/输出 Kernel 与 Gateway 事实、受限
Kernel Diff、`raw_files` 保持捕获内容原样的受限 Session Projection，以及显式标记为不可信的
最终注释；配置的脱敏只作用于规范化事件摘要，且绝不包含竞争 Trajectory。

Attempt 行持久化该 Artifact Digest。Runtime 把累计 Lineage Checkpoint 与该私有 Snapshot
投影进 Optimizer 统一的只读 `input/evidence` View；两个来源 Artifact 仍保留独立身份，用于
恢复与来源追踪。基础设施重试复用两个来源 Digest，但仍获得全新进程、Session 和 Workspace。

## 影响

Epoch 内学习是显式、不可变、确定性、按 Trajectory 隔离且无需复用模型上下文即可重建的。Attempt Evidence 会增加 Artifact 存储量，仍受保留与垃圾回收影响。目标 Backend 验收仍然必需；跨 Branch 共享只在 Epoch 提交进累计 Lineage Evidence 后发生。
