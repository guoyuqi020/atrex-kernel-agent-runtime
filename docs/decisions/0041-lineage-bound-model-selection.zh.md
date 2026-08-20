# 决策 0041：把具体 Model 选择绑定到 Lineage

## 决策

Runtime 部署配置继续选择 Optimizer/Evolver Backend、凭据、可执行环境、Reasoning Effort 与
不透明 Session Settings。Campaign schema v3 独立选择具体 Model 身份：

- 顶层可选 `problem_generalization_model` 只作用于 Core 问题泛化；
- 每个 `lineages.<dsl>.models.optimizer` 作用于 Framework Baseline 和全部 Optimizer Attempt；
- 每个 `lineages.<dsl>.models.evolver` 作用于 Challenger 构建。

省略或设为 null 表示委托给所配置 Backend CLI 的默认 Model。Runtime 把 Campaign 级与 Lineage
级取值持久化到 Registry，恢复相同 Creation Key 时拒绝 Model 漂移，并通过
`ATREX_AGENT_MODEL` 注入所选值。Core 与 Evolver 在 Session Provenance 中保留该值，并转换为
Claude、Codex、QoderCLI 或 Pi 的原生命令参数。

## 后果

同一 Campaign 中不同 DSL Lineage 可以使用不同的 Optimizer/Evolver Model，同时继续共享由部署
控制的 Backend 与凭据边界。Lineage 恢复时能够复现其显式声明的 Model 身份。Campaign 作者也可
把字段保留为 null 来使用 Provider 默认值；此时 Provider 后续修改默认值有意不属于 Runtime 的
身份保证范围。

Campaign schema v3 已提供 Model 时，Codex 与 Pi 的结构化 Session Settings 不得再次声明
Model，Adapter 会拒绝有歧义的启动。修改已持久化 Model 需要使用新的 Campaign Creation Key，
不能原地修改既有 Lineage。
