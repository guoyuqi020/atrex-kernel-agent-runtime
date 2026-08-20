# 决策 0037：Runtime 绑定 Agent Backend

[English](0037-runtime-bound-agent-backends.md) | 中文

## 背景

Core 已实现 Claude、Codex、QoderCLI 与 Pi Adapter，但 Backend 由可进化仓库选择；Evolver
则只实现 Claude。这样 Agent Revision 可以在与另一 Revision 比较时切换 Provider，使 Agent
设计晋升与 Provider/Model Policy 变化混杂，也使部署配置和 Example 无法准确声明实际 CLI 与
Credential 契约。

## 决策

Runtime 配置分别把 `optimizer` 和 `evolver` 绑定到 `claude`、`codex`、`qodercli` 或 `pi`，
并绑定 `reasoning_effort` 与不透明的 Backend-specific `session_settings` 字符串。Runtime 把三者
作为不可拆分的保留环境变量三元组注入，Worker 环境白名单不得覆盖。

Core 与 Evolver 保留仓库默认值用于独立开发，但托管 Session 必须使用 Runtime Binding。Core
将其用于问题泛化、Framework Baseline 与每次 Optimizer Attempt。Evolver 在没有 Token 截止的
前提下实现同样四种 Adapter 和 Token Accounting 契约。两者都在 Session Trace 中记录实际
Backend、Effort、Settings Digest、原始 Provider Stream 与标准化 Usage。Provider Usage 缺失或
不完整时仍失败关闭。

Credential 永远不是配置值，只能通过各 Worker 的显式环境白名单转发；所选 CLI 必须存在于该
Worker 的 `PATH`。

## 结果

- 同一 Runtime 部署中的 Active 与 Challenger Agent Revision 使用可比较的 Provider Policy。
- Optimizer 与 Evolver 可以选择不同 Backend。
- 修改 Runtime Binding 会改变执行策略，但不会重写 Core/Evolver Git Revision。
- Prompt、Tool、Workflow、Memory Policy 与 Adapter 实现仍由 Bundle 持有。
- 单一 Core 入口和 Bundle 持有 Framework 实现的边界不变。
- 每个目标 Backend 仍必须使用真实 Credential 和 CLI 完成目标环境验收。
