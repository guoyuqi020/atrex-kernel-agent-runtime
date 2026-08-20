# 决策 0042：以 Artifact 为种子的 Lineage 根节点

## 状态

已接受。

## 决策

一个 Active Campaign 可以通过以下任一种来源增加 Lineage：

- 一个已封存的 Kernel Agent Artifact Digest 加一个已封存的 Kernel Artifact Digest；或
- 一个已有 Kernel Agent Revision ID 加一个已有 Kernel Revision ID，由 Runtime 解析出对应的已封存 Artifact。

新 Lineage 复用选中的内容，但不复用 Registry Revision 身份。Runtime 会创建相互独立的新
`agent-v0` 和 `v0`，在可用时记录来源 Revision ID，并从 Epoch 1 开始运行。Agent 与 Kernel
都必须匹配请求中的 DSL；二者可以来自不同的历史 Lineage，从而支持有意组合某个 Agent
设计与另一个 Kernel 起点。

发布前，Runtime 会校验完整 Agent Bundle，并使用目标 Campaign 已封存的 Evaluation Contract
与硬件目标，对准确的 Kernel Artifact 做独立权威评测。Kernel 不正确就不会创建 Lineage。
没有 Roofline 时，会执行与普通 Runtime-final 评测一致的非权威 SOL Profile。`creation_key`
会派生稳定的 Lineage 与根 Revision ID，因此相同请求可在中断后安全恢复。

这不是第二条 Git 导入边界。标准 Campaign Bootstrap 仍通过 `base_revision.commit` 由 Commit
锚定；Core 与 Evolver 源码仍只能以完整 Commit ID 从 Git 导入。新操作只能选择 Runtime CAS
中已经封存的内容，或引用解析到这些内容的已注册 Revision。

## 接口

- CLI：`atrex-kernel-agent-runtime seed-lineage --config ... --campaign ... --spec ...`
- 管理 API：`POST /v1/admin/campaigns/{campaign_id}/lineages`

请求包含固定 DSL、可选 Optimizer/Evolver Model、Epoch 拓扑、可选初始 Evidence，以及一个带
判别字段的 `seed` 来源。返回值包含新 Lineage ID、`agent-v0`、`v0`、来源 Provenance、权威
Gateway Result 与 Latency。

## 影响

历史优化可以在不重新运行 Framework Baseline 的情况下分叉、重组或复现。新 Lineage 不建立
父 Lineage 边；来源 Provenance 只是审计元数据，它自身的 Agent 与 Kernel 版本历史仍是独立的
树。跨 Campaign 复用也是安全的，因为 Kernel 总会在目标 Campaign Contract 下重新评测。
