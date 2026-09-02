# Atrex Kernel Agent Runtime 文档

[English](README.md) | 中文

## 使用与运维

- [使用说明](user-guide.zh.md)：安装、配置、Bootstrap、运行、检查和恢复。
- [配置参考](configuration.zh.md)：Runtime schema v1 与 Campaign schema v3。
- [接口参考](interfaces.zh.md)：CLI、HTTP、Optimizer Runtime Tools 与 Evolver 文件系统。
- [部署与运维](operations.zh.md)：生产拓扑、隔离、备份与事故处理。
- [可运行示例](../examples/README.zh.md)和
  [生产运行脚本](../scripts/production/README.zh.md)。

## 理解与验证

- [设计理念](design-principles.zh.md)：为什么 Runtime 将 Agent 进化与可信控制分离、采用 Lineage
  内竞争并保持历史只追加。
- [设计与实现](../DESIGN.zh.md)：集中说明 AKA 局限、自进化所缺能力、Runtime 权责、端到端
  运行方式、实现映射和剩余风险。
- [架构](architecture.zh.md)：术语、权责、生命周期、存储、隔离与源码组织。
- [评测与晋升](evaluation.zh.md)：私有 Contract、正确性、Production Gate、Evaluate/ABBA、
  Roofline、NCU 与选择。
- [协议](protocols.zh.md)：持久身份、Artifact、Evidence、Session 与可见性规则。
- [架构决策](decisions/README.zh.md)：仍约束当前实现的设计理由。
- [测试与生产验收](testing-and-acceptance.zh.md)：仓库检查与部署证据。
- [发布检查清单](release-checklist.zh.md)和[变更日志](../CHANGELOG.zh.md)。

## 文档权威

当前代码与严格 Schema 是可执行权威。配置与接口参考定义受支持公开表面；架构和协议描述当前语义；
架构决策只解释约束原因。生产就绪由测试/验收文档与发布清单证明，而不是依赖人工维护的功能状态表。
