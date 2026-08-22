# 决策 0007：把 Agent 运行持久化为不可变 Trace Artifact

[English](0007-immutable-agent-traces.md) | 中文

## 状态

已接受并实现。

## 背景

每次 Optimizer 和 Evolver 调用都使用全新进程与 Session。Runtime 需要持久来源用于审计、重试归属、
Evidence，同时不能把隐藏对话连续性带入后续 Session。原始 Provider 历史
不适合写入 SQLite，单个 Trace 字段也会覆盖重试记录。

## 决策

Artifact Store 持有原始 Session Trace Tree 和结构化 Evolution Record；Registry 保存内容 Digest、
Worker 生命周期、角色、Model、Workspace、终止原因、Optimizer Token 预算和完整 Provider Token
Bucket。一个稳定 Attempt 或 Evolution Subject 可以拥有多条只追加 Worker Session。

Runtime 在启动前创建 Worker Session Record，持续更新到终态，并在可用时原样封存 Trace，不做脱敏。
Evidence 保存规范化摘要与来源 Digest；Agent View 显式物化原始 Trace Artifact。Trace 是审计输入，不是权威成功声明，也不会自动成为
Agent 记忆。

## 结果

重启和重试可以保留归属而不复用模型上下文。即使进程在捕获 Trace 前失败，Runtime 仍保留带失败
身份的终态 Worker Session Record，但不会伪造 Trace Artifact。Kernel/Agent 晋升由 Runtime 权威
Gateway Record 与 Registry Transition 决定，而不是 Provider 输出。
