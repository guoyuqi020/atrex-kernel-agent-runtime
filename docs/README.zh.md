# Atrex Kernel Agent Runtime 文档

[English](README.md) | 中文

## 从这里开始

- [使用说明](user-guide.zh.md)：安装、配置、Bootstrap、运行、检查、恢复和维护。
- [接口说明](interfaces.zh.md)：CLI、HTTP API、Optimizer Runtime Tools、Evolver Tools 和外部服务
  Contract。
- [配置说明](configuration.zh.md)：Runtime schema v1 和 Campaign schema v3。
- [可运行样例](../examples/README.zh.md)：远端 Agate、Bootstrap、Lineage、Evolution、开发 Shell 和
  本地 Wiki。
- [生产运行脚本](../scripts/production/README.zh.md)：常驻控制面、多任务 Campaign、按 DSL 检查和
  故障恢复。

## 设计

- [架构设计](architecture.zh.md)：信任边界、生命周期、隔离和数据流。
- [模块设计](module-design.zh.md)：实现模块及职责。
- [代码组织](code-organization.zh.md)：依赖方向和源码结构。
- [完整仓库 Agent Revision](full-repository-optimizer-revision.zh.md)：Git 导入、封存、进化和来源。
- [性能 Gate](performance-gates.zh.md)：正确性、Production Gate、Evaluate 和 ABBA。
- [可信 Roofline 构建](roofline-builder.zh.md)：可选的 Commit 固定 Roofline 生成。
- [架构决策](decisions/README.zh.md)：只包含当前有效决策。

## 运维与发布

- [部署与运维](operations.zh.md)：生产拓扑、沙箱、备份、恢复、保留和故障处理。
- [协议](protocols.zh.md)：持久 Schema 和 Artifact 语义。
- [实现状态](implementation-status.zh.md)：已实现范围和待完成验收。
- [测试与生产验收](testing-and-acceptance.zh.md)：自动化与目标环境验证。
- [发布检查清单](release-checklist.zh.md)：打包和发布门禁。
- [变更记录](../CHANGELOG.zh.md)：按版本记录已发布行为。

## 文档权威

当前代码和严格 Schema 是可执行事实来源。接口与配置说明定义受支持的公共表面；架构决策只解释
当前约束的原因，不引入兼容行为。如果实现状态没有把目标环境验收标记为完成，则该功能已实现，
但尚不能视为生产验证完成。
