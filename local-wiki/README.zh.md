# Atrex Local GPU Wiki

[English](README.md) | 中文

这个目录是独立 Atrex GPU Wiki 的本地 HTTP 适配器，仅用于 Runtime 集成测试。HTTP 协议由
本地提供，查询行为执行语料自带的实现。

适配器不再自行实现检索算法。每次 Query 都直接执行语料自带的
所选语料的 `tools/query_nl.py`，因此 Bridge Agent、Intent 校验、算子别名与子算子独立检索、安全 Widening、
`kernel_wiki` 排序、`hardware_wiki` 精确查询和 Record 投影均使用它的
实现。Query 的 `content` 形如：

```json
{"query_id":"wiki-query-0123456789abcdef0123456789abcdef","records":{"stable.record.id":{"store":"gpu_wiki","wiki_id":"gpu_wiki::stable.record.id","source":"kernel_wiki","type":"technique-card","applies_to":{},"match":{},"payload":{}}},"notes":[]}
```

完整查询结果原样透传，包括归因 ID 和全部 notes。公开库的旧式私有 `internal_gpu_wiki` 槽位
不可用时，上游会返回说明，但不影响公开库查询。新版独立内部库须通过下述配置直接选择，
不能放入这个旧槽位后期待自动接入。

Runtime 继续负责带版本的 HTTP Envelope、Digest 校验、Attempt Authority 和结果冻结。
每个 `records` Mapping Value 已经是完整的安全服务 Record；上游提供 `query_id` 和规范化的
`wiki_id` 用于归因。不存在独立 Read 操作。

Runtime 将 Wiki 视为只读外源知识。

## HTTP 接口

| 方法与路径 | 结果 |
| --- | --- |
| `GET /` 或 `GET /ui` | 本地浏览器查询客户端。 |
| `GET /healthz` | 进程存活检查。 |
| `GET /readyz` | 所选语料的工具、索引及数据依赖与 SQLite 就绪检查。 |
| `POST /v1/knowledge/query` | 严格 Runtime Query；`content` 为上游 `query_id/records/notes`。 |

## 语料

`corpus/gpu-wiki` 是本仓库的普通内容，Checkout 后即可直接启动。启动时会复制到被忽略的可写
`state/gpu-wiki` Store，语料自带的工具因此可以记录查询反馈而不修改被跟踪的文件。修改语料只会
在下次启动时触发一次重新复制。

来源 Commit 与复制范围记录在 [corpus/README.md](corpus/README.md)。

当前内置 Wiki 支持可选查询追踪。在服务环境中设置 `ATREX_WIKI_PROFILE_ROOT`，指向可写的
运行数据目录，即可保存不可变查询事件；`ATREX_WIKI_TASK_ID` 用于任务归属。事件包含查询问题、
归一化意图、返回记录的 ID/排名、耗时和 Token 计数，不复制记录正文或编码 Agent 对话。
未设置 profile root 时不生成查询事件文件；追踪写入失败也不改变查询响应。输出应放在
`corpus/` 之外。

上游 AKA 的插件入口和重启接管 Orchestrator 不属于本独立 HTTP 服务。适配器直接调用
`query_nl.py`，普通 Bridge 启动不依赖 AKA Orchestrator；独立部署不要传入 AKA 专用的
`ATREX_ENVIRONMENT_RESTART_HANDOFF_ID`。
它原有的 Apache-2.0 License 与 NOTICE 保留在同一目录下。

### 内部索引知识库

适配器也支持独立内部 Wiki 的 `query_nl.py → query.py → search_index` 布局。
它的 `query_id/records/notes` 外层格式与 Runtime 兼容；嵌套 `wiki_identity`、治理元数据、
证据限制及 generation-reference 匹配说明原样保留，不转换成公开库的 Record 格式。
`reference_root` 为一个服务选择一个原生知识库，两套知识库不自动合并。

使用 [configs/internal.example.json](configs/internal.example.json) 可在公开库示例使用的同一端口
提供内部库服务。先从有访问权限的内部仓库导入，命令从 Runtime 仓库根目录执行：

```bash
# 目标目录必须新建且为空，不能将新版本直接覆盖到旧快照上。
mkdir -p local-wiki/corpus/internal_gpu_wiki
set -o pipefail
git -C /path/to/atrex-kernel-agent-internal archive \
  2076c865cc6618d810cfe2bf09b4fc395536693e:internal_source/gpu-wiki \
  | tar -x -C local-wiki/corpus/internal_gpu_wiki
PYTHONPATH=local-wiki/src .venv/bin/python -m atrex_local_wiki serve \
  --config local-wiki/configs/internal.example.json
```

来源是 `codex/ppu15-agent-wiki` 分支，来源与复制边界见 [corpus/README.md](corpus/README.md)。
内部快照和可写状态均被 Git 忽略。升级时停止服务，导出至新的空目录，再替换完整 reference
目录；重启时会刷新可写 Store。就绪检查拒绝不完整的索引库，不会退回旧检索入口。
版本摘要覆盖原生工具、声明的数据分片、治理和附加证据；缓存未变文件的哈希，避免每次查询
重读整个索引。
就绪检查验证目录布局、Manifest 格式及依赖文件存在性；分片内容和治理资格仍由上游校验。
治理文件格式错误或过期时，即使依赖文件齐全，原生检索也可能按规则隐藏记录。

HTTP 请求保留完整硬件、DSL、算子和原始问题。任意问题会走内部库的 Claude/Qoder 意图解析，
不匹配上游固定句式的免模型入口。该内部版本的 Claude 使用 `--bare`，Bridge 环境未透传
`ANTHROPIC_BASE_URL`、`ANTHROPIC_MODEL`；部署前须单独验证所用 CLI 与模型端点，不能认为复制
本地 Claude settings 就已完成自定义端点配置。原生确定性查询和模拟意图 CLI 可用于验证索引
与 HTTP 契约，无需请求模型。

本配置仅接通知识服务。另一个 Runtime 部署配置中的 `gpu_wiki.enabled` 默认 `false`；设为
`true` 才向 Bootstrap/Optimizer 加载 `wiki-query`、条件式提示词并发放 Attempt 查询权限。
它与 Direction/Experiment 模块独立。使用包含可选 Wiki 工具的 Core/KDA 源码，重启 Runtime
及 campaign worker 后，新建工作区才会使用新契约；配置开关不会替换已冻结的 Agent 源码。
详见 [Runtime 配置](../docs/configuration.zh.md#gpu_wiki)。

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
