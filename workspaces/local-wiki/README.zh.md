# Atrex Local GPU Wiki

[English](README.md) | 中文

这个工作区是独立 Atrex GPU Wiki 的本地 HTTP 适配器，仅用于 Runtime 集成测试。HTTP 协议由
本地提供，查询行为执行固定 Commit 的上游实现。

适配器不再自行实现检索算法。每次 Query 都直接执行 Commit 固定的上游
`gpu-wiki/tools/query_nl.py`，因此 Bridge Agent、Intent 校验、归一化、安全 Widening、
`kernel_wiki` 排序、`hardware_wiki` 精确查询、公有/内部 Store 隔离和 Record 投影均使用上游
实现。Query 的 `content` 与上游接口一致：

```json
{"records":{"stable.record.id":{"store":"gpu_wiki","source":"kernel_wiki","type":"technique-card","applies_to":{},"match":{},"payload":{}}},"notes":[]}
```

Runtime 继续负责带版本的 HTTP Envelope、Digest 校验、Attempt Authority 和结果冻结。
每个 `records` Mapping Value 已经是完整的安全服务 Record；某条 Record 实际影响工作时，Consumer
保存它的稳定 Mapping Key。不存在独立 Read 操作。

仓库内 Vendored 的固定上游源码始终保持不可变。启动时会原子同步到 `state/gpu-wiki` 可写
Store。Runtime 将 Wiki 视为只读外源知识。

## HTTP 接口

| 方法与路径 | 结果 |
| --- | --- |
| `GET /` 或 `GET /ui` | 本地浏览器查询客户端。 |
| `GET /healthz` | 进程存活检查。 |
| `GET /readyz` | 上游工具、两个 Index 与 SQLite 就绪检查。 |
| `POST /v1/knowledge/query` | 严格 Runtime Query；`content` 为上游 `records/notes`。 |

## 固定上游版本

`reference.lock.json` 固定 Alibaba `atrex-kernel-agent` Commit
`71b16928579474c93039053d2facfeaf7134e268`。该 Commit 的准确 `gpu-wiki` Tree 已 Vendor 到
`corpus/gpu-wiki`，因此 Runtime Checkout 无需再次 Clone 即可启动。只有维护者在更新 Lock 时
刷新 Vendored Tree；Runtime 执行期间不会下载或修改它。

```bash
git -C /path/to/atrex-kernel-agent archive \
  71b16928579474c93039053d2facfeaf7134e268 gpu-wiki
```

## 启动

默认配置不覆盖上游查询默认值，由固定版本 `query_nl.py` 自己选择 Bridge CLI、Timeout 与 Record
上限。`agent_cli`、`query_timeout_seconds` 和 `max_results` 只作为可选 HTTP 部署 Override；
Local Wiki 不保存模型凭证。

`max_concurrent_queries` 限制同时运行的 `query_nl.py` 子进程数，默认值为 `16`；其余请求等待
并发槽。这样既避免模型和子进程无界扩张，也不会再用一个全局锁串行阻塞所有只读查询。

```bash
PYTHONPATH=workspaces/local-wiki/src \
  .venv/bin/python -m atrex_local_wiki serve \
  --config workspaces/local-wiki/configs/local.example.json
```

打开 [http://127.0.0.1:8091/](http://127.0.0.1:8091/)。如果覆盖 `agent_cli`，必须使用固定
版本上游 `tools/agent_launch.py` 接受的 Backend。

## 验证

```bash
PYTHONPATH=src:workspaces/local-wiki/src .venv/bin/pytest workspaces/local-wiki/tests
PYTHONPATH=workspaces/local-wiki/src \
  .venv/bin/ruff check workspaces/local-wiki/src workspaces/local-wiki/tests
PYTHONPATH=workspaces/local-wiki/src \
  .venv/bin/mypy --config-file workspaces/local-wiki/pyproject.toml \
  workspaces/local-wiki/src
```
