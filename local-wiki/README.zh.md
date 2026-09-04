# Atrex Local GPU Wiki

[English](README.md) | 中文

这个目录是独立 Atrex GPU Wiki 的本地 HTTP 适配器，仅用于 Runtime 集成测试。HTTP 协议由
本地提供，查询行为执行语料自带的实现。

适配器不再自行实现检索算法。每次 Query 都直接执行语料自带的
`gpu-wiki/tools/query_nl.py`，因此 Bridge Agent、Intent 校验、算子别名与子算子独立检索、安全 Widening、
`kernel_wiki` 排序、`hardware_wiki` 精确查询和 Record 投影均使用它的
实现。Query 的 `content` 形如：

```json
{"query_id":"wiki-query-0123456789abcdef0123456789abcdef","records":{"stable.record.id":{"store":"gpu_wiki","wiki_id":"gpu_wiki::stable.record.id","source":"kernel_wiki","type":"technique-card","applies_to":{},"match":{},"payload":{}}},"notes":[]}
```

完整查询结果原样透传，包括归因 ID 和全部 notes。私有 `internal_gpu_wiki` Store 不可用时，
上游会返回说明，但不影响查询公开 Store。

Runtime 继续负责带版本的 HTTP Envelope、Digest 校验、Attempt Authority 和结果冻结。
每个 `records` Mapping Value 已经是完整的安全服务 Record；上游提供 `query_id` 和规范化的
`wiki_id` 用于归因。不存在独立 Read 操作。

Runtime 将 Wiki 视为只读外源知识。

## HTTP 接口

| 方法与路径 | 结果 |
| --- | --- |
| `GET /` 或 `GET /ui` | 本地浏览器查询客户端。 |
| `GET /healthz` | 进程存活检查。 |
| `GET /readyz` | 上游工具、两个 Index 与 SQLite 就绪检查。 |
| `POST /v1/knowledge/query` | 严格 Runtime Query；`content` 为上游 `query_id/records/notes`。 |

## 语料

`corpus/gpu-wiki` 是本仓库的普通内容，Checkout 后即可直接启动。启动时会复制到被忽略的可写
`state/gpu-wiki` Store，语料自带的工具因此可以记录查询反馈而不修改被跟踪的文件。修改语料只会
在下次启动时触发一次重新复制。

来源 Commit 与复制范围记录在 [corpus/README.md](corpus/README.md)。
它原有的 Apache-2.0 License 与 NOTICE 保留在同一目录下。

## 启动

默认配置不覆盖上游查询默认值，由语料自带的 `query_nl.py` 自己选择 Bridge CLI、Timeout 与 Record
上限。`agent_cli`、`query_timeout_seconds` 和 `max_results` 只作为可选 HTTP 部署 Override；
Local Wiki 不保存模型凭证。

当前 Bridge 使用 `claude`（默认）或 `qodercli` 的无工具 JSON 协议，不支持 `codex`；这只限制
Wiki 的意图提取，不影响 Optimizer/Evolver 的 Backend 选择。Runtime 上下文和 Agent 问题以文本
传入，意图提取和算子解析完全由上游负责。旧的本地 `operator_families` Override 已移除。

`max_concurrent_queries` 限制同时运行的 `query_nl.py` 子进程数，默认值为 `16`；其余请求等待
并发槽。这样既避免模型和子进程无界扩张，也不会再用一个全局锁串行阻塞所有只读查询。

```bash
PYTHONPATH=local-wiki/src \
  .venv/bin/python -m atrex_local_wiki serve \
  --config local-wiki/configs/local.example.json
```

打开 [http://127.0.0.1:8091/](http://127.0.0.1:8091/)。如果覆盖 `agent_cli`，必须使用语料自带
`tools/agent_launch.py` 接受的 Backend。

## 验证

```bash
PYTHONPATH=src:local-wiki/src .venv/bin/pytest local-wiki/tests
PYTHONPATH=local-wiki/src \
  .venv/bin/ruff check local-wiki/src local-wiki/tests
PYTHONPATH=local-wiki/src \
  .venv/bin/mypy --config-file local-wiki/pyproject.toml \
  local-wiki/src
```
