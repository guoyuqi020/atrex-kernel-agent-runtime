# 决策 0031：Evolver 只记录 Token，不实施 Token 配额

[English](0031-unbounded-evolver-token-accounting.md) | 中文

## 状态

已接受并实现。

## 背景

Optimizer Session 需要部署持有的成本边界，但 Evolver 每个 Epoch 只调用一次，需要在不受 Provider
Token 截止的情况下完成 Agent Revision 假设。旧配置为两个角色都暴露正数
`max_session_tokens`，累计 Token 达到该值时会终止 Evolver 进程组并返回 125。

## 决策

Optimizer Core 阶段继续使用正数单 Session Token 配额。Evolver 配置不再包含
`max_session_tokens`，Runtime 不再注入 `ATREX_TOKEN_BUDGET`，Evolver 不会因为 Provider Token
消耗而终止。Wall-time、输出大小、进程和 Workspace 安全限制保持有效。

Provider 记账仍是必需协议。Evolver 继续对 Stream Event 去重、优先使用终态 Usage，并报告非缓存
输入、输出、Cache Read 和 Cache Write。其严格 TokenUsageReportV1 使用
`budget_tokens=null`、`budget_exhausted=false`；Runtime 强制要求这两个值，并继续拒绝缺失或不完整
的 Provider Usage。

## 影响

Evolver 成本可观测，但不再由 Token 数限制。运维需要把 Provider 侧控制和 Wall-time 作为外部安全
机制。严格 Schema 会拒绝仍包含 Evolver `max_session_tokens` 的旧配置，必须重新生成；Optimizer
Phase 继续使用独立的正 Provider Token 配额。
