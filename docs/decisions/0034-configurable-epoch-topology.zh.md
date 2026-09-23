# 0034：Agent 自有 Epoch Workflow

[English](0034-configurable-epoch-topology.md) | 中文

## 状态

已接受。本决策取代同一编号下此前由配置驱动 Epoch 拓扑的设计。

## 背景

Branch 数量、进化时机、Trajectory 布局、Attempt 轮次、Kernel 传播与自适应 State 传播都属于
搜索策略。把它们编码成 Runtime 配置，会使每种新策略都依赖控制器改动，也使 Evolver 无法改变
优化工作的组织方式。

Runtime 仍需要提供严格、可审计的资源边界；评测、隔离、选择、晋升、持久化与恢复也必须位于
不可信 Agent 之外。

## 决策

每个 Lineage 只冻结两个 Epoch 资源上限：

- `max_challengers`：Workflow 最多可物化的 Challenger Slot 数；
- `optimizer_attempt_budget`：Workflow 必须恰好分配的 Optimizer Attempt 数。

版本化 Agent Bundle 持有 `workflow/main.py`。它通过窄化的 Workflow SDK 复制或进化 Agent、
创建 Branch Pool 与 Trajectory、运行并发 Round，并显式把 Kernel 与自适应 State 输出路由给
后续 Attempt。Runtime 校验 Capability 与精确预算后执行请求，不推断拓扑，也不提供回退调度。

Workflow 不能修改评测策略、Gate 策略、资源上限、沙箱权限、Registry 事实、选择或晋升逻辑；
这些仍由可信 Runtime 负责。

## 结果

每个 Campaign 都必须包含可执行 Workflow。不同消融臂由不同 Workflow 程序定义，而不是由
Runtime 解释标签。Evolver 可以通过修改 Candidate Workflow 改变搜索组织；Runtime 则保持精简、
面向策略、可恢复且可审计。
