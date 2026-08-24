# 决策 0022：把本地 GPU Wiki 保持为 Wire 兼容的测试工作区

[English](0022-local-wiki-test-double.md) | 中文

## 状态

已接受并实现。

## 背景

Runtime 已经拥有严格的实时 Query Client，但生产 GPU Wiki 由外部独立开发与部署。单元 Fake 只能证明本地控制流，不能证明一个可替换 HTTP 服务能够接受准确的可信请求并返回 Digest 有效的响应。生产 Runtime 必须与参考用 `atrex-kernel-agent` 源码树解耦；本地测试服务可以有意把它固定为测试数据。

## 决策

新增可独立打包的 Python 测试服务 `local-wiki`。其生产模块不导入 `atrex_runtime` 或 `atrex-kernel-agent`，而是独立实现 v1 外部请求/响应字段、规范 JSON Digest 算法、HTTP Status 和可选 Bearer 认证。跨工作区测试使用 Runtime 的权威模型，从而发现任意一侧的漂移。

本地适配器只保留 HTTP Envelope，直接执行 Commit 固定的参考 Wiki `tools/query_nl.py`，不重新
实现 Intent 抽取、归一化、Widening、排序、硬件查询、Store 隔离或服务 Record 投影。因此 Query
返回上游 `records`/`notes` 接口，包括按稳定 ID 索引的完整安全服务 Record。工作区 Fixture 使
单元测试不依赖 Reference Checkout。

固定 Reference 保持不可变，并原子同步到可写本地 Store。SQLite 只保留准确 Query 观察；服务不提供 Feedback Endpoint。

Runtime 只通过远端 Wiki 同样使用的 `gpu_wiki.base_url` 与凭据配置访问该服务。本地响应 Content 有实现专用版本，但在外部 v1 Envelope 内仍是不透明 JSON。Runtime 不增加本地专用 Endpoint 或 Import。

## 结果

开发者无需远端部署即可验证 Query、严格请求解析、Digest 校验、认证和响应状态。切换生产服务只是配置变更。搜索质量、可用性与运维行为仍需真实服务验收；该测试替身绝不能被描述或部署成 Wiki 本体。
