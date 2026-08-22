# 决策 0021：实时查询 GPU Wiki，并在返回前冻结交互

[English](0021-live-gpu-wiki-capability.md) | 中文

## 状态

已接受并实现。本决策是当前 Wiki 集成的权威决策。

## 背景

GPU Wiki 是外源知识，lineage Experience 是 Agent 自己生产的本地历史。把一个预先选择的 Wiki Snapshot 当成下一 Epoch 的公共记忆，会混淆两者职责，也使 Optimizer 无法在问题真正出现时发起聚焦查询。若实时响应没有先持久化，Session Trace 将不可复现。

## 决策

每个已配置的 Optimizer 获得 Core `wiki-query` Tool 和 Attempt 范围 Runtime
Capability。Query 只向 `POST /v1/wiki/query` 发送不可变 Manifest 中的 Attempt ID、聚焦问题和
幂等键；Agent 可见 Content 是 GPU Wiki 准确的 `records`/`notes` 投影，`records` Mapping Key
就是稳定 Record ID，每个 Value 都是完整的安全服务 Record。可信 Runtime 从权威存储重建
Campaign、lineage、Epoch、Branch、Ordinal、Kernel Agent Revision、operator、DSL、硬件、
Evaluation Contract、Epoch Evidence 和 Attempt Evidence 上下文。只有 Runtime 持有外部 Wiki
凭据。

Runtime 校验严格 Wiki Response 和规范 Content Digest，把每次完整可信 Query 交互封存为 `WIKI_INTERACTION` Artifact，并在向 Worker 返回内容前把 Artifact Digest 提交到幂等 Reservation。Core Tool 只投影知识 `content`；协议版本、交互/Snapshot 身份和完整性 Digest 保留在内部。相同 Key 与请求会重放冻结响应，不会再次访问外部服务；请求发生变化则失败。Wiki Query 不消耗 Gateway Benchmark 调用次数预算，但仍受请求/响应字节、Transport/进程 Timeout 和仅按 Provider Token 计算的 Agent 预算限制。

Runtime 不向累计 Evidence 注入 Wiki Selection，也不把 lineage Experience 存入 Wiki。Wiki 集成只读查询：Runtime 不上传消费记录、Session Trace、Kernel 历史、胜者事实、组件进化或 lineage 记忆。

## 后果

Optimizer 通过 Query 直接获得上游 Wiki 的完整安全 Record 投影，并保存实际影响工作的 Record
稳定 ID。每个模型可见响应都可归因、可重放。外部服务凭据、可信上下文和审计身份保留在模型
上下文之外。真实 Wiki 可用性仍可能影响单次 Tool 调用，因此 Agent 与 Prompt 必须显式处理失败，而不是接收
静默缓存替代。
