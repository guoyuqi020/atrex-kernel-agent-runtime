# 决策 0029：保留每一次 Bootstrap 执行 Generation

[English](0029-append-only-bootstrap-generations.md) | 中文

## 状态

已接受并实现。

## 背景

一次 Campaign Bootstrap 只有一个稳定、确定性的 `bootstrap_attempt_id`，但 Provider 故障、
Token 配额终止、Runtime 重启、Agent Blocked Report 或 Authority 过期都可能要求启动新的 Core
Session。Capability Generation Fencing 已能保证重试安全，但旧 Gateway Schema 会覆盖当前
Generation，并删除未提交的 Operation Reservation。原始 Run 目录通常仍存在，却没有持久记录
关联失败 Session、Token 用量、Report、Trace、Workspace 和失败原因，Operator 无法可靠检查
每一次物理执行为何结束。

## 决策

Gateway Control Schema 5 新增以 `(bootstrap_attempt_id, recovery_generation)` 为键的 Append-only
`bootstrap_runs`。签发时创建 `issued` 记录；Workspace 创建后绑定 `run_id` 和准确路径；每次正常
返回或被捕获的失败都提交唯一的 `completed` 或 `failed` 终态。记录保存 Finish/Failure 原因、
时间戳、四类 Provider Token 与配额、Session Trace 和终止 Report Digest，以及可用时的权威
Candidate/Result Digest。Runtime Crash 留下的非终态记录会在下一 Generation 签发时标为
`superseded-by-retry`。

Gateway Operation 使用 `(attempt_id, recovery_generation, idempotency_key)` 作为键。轮换会让旧
Bearer 失效并重置当前额度，但不删除早期 Operation。Outcome Commit 仍由准确的 Authorization
Generation 隔离；已有权威 Outcome 会直接恢复，不创建新 Run。Artifact GC 会把全部 Run 和
Operation Digest 视为存活引用。

认证管理 API 与 `list-bootstrap-runs` / `show-bootstrap-run` CLI 提供查询，但绝不暴露 Capability
Token。失败 Bootstrap 异常也会标明 Attempt、Generation 与物理 Run。Schema 4 迁移会保留当前
Operation，并为每个已有 Bootstrap Subject 创建明确标记的 Legacy 记录；旧 Schema 已覆盖的历史
无法事后重建。

## 影响

Capability Generation 既继续充当安全 Fence，也成为持久执行身份。失败 Generation 可以查询和
审计，但其 Agent 结论不会因此变成权威结果。存储会随执行和 Operation 增长，因此只能通过理解
保留策略的维护流程清理 Workspace 与 Artifact。目标部署仍需在签发、Session、Gateway 和终态
记录的每个边界执行强制 Crash 测试。
