# 决策 0030：分离探索性评测与可信终结

[English](0030-exploratory-and-authoritative-evaluations.md) | 中文

## 状态

已接受并实现。

## 背景

Optimizer 在修改 Kernel 时需要编译、正确性和性能反馈，因此一个 Attempt 可能先后评测多个
不同 Candidate Tree，最后才提名其中一个。旧控制路径会把第一次 `evaluate` 响应直接提交为
Attempt Outcome。这会让修复无法继续：早期错误 Candidate 永久占用 Outcome，后续正确 Candidate
与其冲突；同时也把不可信 Agent 可见的探索证据和可信保留/晋升逻辑使用的结果混为一体。

## 决策

Agent 发起的每次 `evaluate` 都是探索性评测。Runtime 把当时准确的 Candidate 目录封存为 Kernel
Artifact，把完整原始 Gateway 响应封存为 Artifact，并追加一条不可变
`GatewayEvaluationRecord`。记录包含 Attempt、Recovery Generation、Ordinal、`agent` 来源、
Idempotency Key、Candidate/Result Digest、正确性、延迟、外部 Job 身份和时间。不同请求必须使用
不同 Key；相同请求以相同 Key 重试时直接重放已有响应和记录，不会再次提交外部任务。探索记录
永远不会提交 Attempt Outcome。

`candidate_ready` 或 `baseline_ready` 提名最终的 `work/kernel` Tree。Runtime 自己封存该 Tree，
并要求存在一个针对完全相同字节的正确探索记录。Bootstrap 使用 Runtime 凭据和稳定的
Runtime-final Idempotency Key 向 Agate 重新提交评测。优化 Attempt 则临时注册准确的提名 Artifact，
并把最终权威交给 Kernel 留存：普通 Evaluate 按配置次数分别测量 A/B，以 B 的算术平均和聚合
Result 完成终结；同 Allocation ABBA 以 B 的几何平均和成对聚合 Result 完成终结。两种替换都在
Attempt 完成前发生。基础设施失败不会伪造 Outcome。

Gateway Control schema 6 保留全部评测记录以及权威 Outcome 对应的 Source Evaluation Identity。
Artifact 存活集合包含每一条记录的 Candidate 和原始 Result。认证管理 API 与 CLI 可按 Evaluation
ID 列表和读取准确的 Candidate 文件与原始 Gateway Result；ABBA 权威结果同时保存在 Kernel
Measurement 与 Kernel Revision 中。

## 影响

一个 Attempt 可以安全地探索、失败、修复并再次评测，而不会污染最终状态。即使生命周期 Event
被清理，完整 Kernel/Result 历史仍可查询。两种留存方法都复用其强制比较中的 Candidate 测量
作为最终权威，避免重复的单边 Candidate Eval，同时保留准确比较证据。Bootstrap 因没有
Incumbent，仍消耗一次独立终评。外部提交后、持久化终评前的恢复仍需保持幂等，目标部署仍需
完成该窗口的 Crash Test。
