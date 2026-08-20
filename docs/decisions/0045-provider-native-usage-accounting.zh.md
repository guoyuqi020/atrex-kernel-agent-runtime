# ADR 0045：Provider 原生用量记账

## 状态

已接受。

## 决策

Runtime 按所选 Provider 权威返回的原生单位记录每个 Worker。QoderCLI 使用 `credits`；Claude、
Codex 与 Pi 使用互斥的 Provider Token Bucket。Runtime 不从 Qoder Credit 估算 Token，也不把
Qoder 填充为零的 Token 字段当作用量。

Core 接收 `ATREX_USAGE_UNIT` 与 `ATREX_USAGE_BUDGET`。Schema v2 Report 记录 `budget`、
`consumed`、真实或全零的 Token Bucket，以及可选 `credits`。实时配额按 Provider Message ID
去重累计 Qoder Credit；终态以 `result.total_credits`（或等价的 Model Usage 总值）为权威值。
Evolver 仍不设配额，但使用同一 Report，并令 `budget=null`。

Worker Session、Bootstrap Generation、Attempt Trace、Runtime Event 与未经脱敏的 Session Trace
都保留所选单位和用量。配置同时提供 `max_session_tokens` 与 `max_session_credits`，只执行与当前
Backend 对应的配额。

## 结果

- Qoder 可以在不伪造 Token 数的前提下执行 Fail-Closed 配额。
- 跨 Provider 汇总必须按单位分组，Credit 与 Token 不能相加。
- Qoder 消息若声明 Usage 却不提供 Credit，视为记账不完整并 Fail Closed。
