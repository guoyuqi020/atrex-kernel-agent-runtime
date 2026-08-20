# 决策 0022：把本地 GPU Wiki 保持为 Wire 兼容的测试工作区

[English](0022-local-wiki-test-double.md) | 中文

## 状态

已接受并实现。

## 背景

Runtime 已经拥有严格的实时 Query 和 Epoch 后 Feedback Client，但生产 GPU Wiki 由外部独立开发与部署。单元 Fake 只能证明本地控制流，不能证明一个可替换 HTTP 服务能够接受准确的可信请求、返回 Digest 有效的响应并落实 Feedback 幂等性。生产 Runtime 必须与参考用 `atrex-kernel-agent` 源码树解耦；本地测试服务可以有意把它固定为测试数据。

## 决策

新增可独立打包的 Python 测试服务 `workspaces/local-wiki`。其生产模块不导入 `atrex_runtime` 或 `atrex-kernel-agent`，而是独立实现 v1 外部请求/响应字段、规范 JSON Digest 算法、HTTP Status、可选 Bearer 认证和 Feedback 身份语义。跨工作区测试使用 Runtime 的权威模型，从而发现任意一侧的漂移。

本地适配器只保留 HTTP Envelope，直接执行 Commit 固定的参考 Wiki `tools/query_nl.py`，不重新
实现 Intent 抽取、归一化、Widening、排序、硬件查询、Store 隔离或服务 Record 投影。因此 Query
返回上游 `records`/`notes` 接口，包括按稳定 ID 索引的完整安全服务 Record。工作区 Fixture 使
单元测试不依赖 Reference Checkout。

固定 Reference 保持不可变，并原子同步到可写本地 Store。SQLite 保留准确 Query、Feedback HTTP
观察和幂等状态。Epoch 完成后，冻结交互中每条公开 Kernel Record 都通过固定版本
`ingest_feedback.py` 形成上游 `served` Event，再由固定版本 `rebuild_importance.py` 折叠追加日志
并更新 Ranking。Runtime Report 没有权威的逐 Record 采用字段，因此适配器不会推断 `applied`、
`effective` 或 `ineffective`。中断的应用保持 Pending，并使用稳定的上游 Event Key 重放。

Runtime 只通过远端 Wiki 同样使用的 `gpu_wiki.base_url` 与凭据配置访问该服务。本地响应 Content 有实现专用版本，但在外部 v1 Envelope 内仍是不透明 JSON。Runtime 不增加本地专用 Endpoint 或 Import。

## 结果

开发者无需远端部署即可验证所有 Wiki 外部路径，包括 Query、严格请求解析、Digest 校验、认证、响应状态、至少一次 Feedback 重放、上游 Served Event Ingest 和 Ranking Rebuild。切换生产服务只是配置变更。搜索质量、远端持久化、其他生产 Feedback 处理、可用性与运维行为仍需真实服务验收；该测试替身绝不能被描述或部署成 Wiki 本体。
