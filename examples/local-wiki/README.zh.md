# Local Wiki 与 Runtime Tools 示例

[English](README.md) | 中文

该示例启动 Wire 兼容的本地 GPU Wiki，并展示 Core Agent 可用的准确 `wiki-query` 工作流。
Local Wiki 只是开发测试替身；生产环境通过远端
`gpu_wiki.base_url` 使用相同的外部 API。

从 Lima 挂载的工作区运行时，应创建 Linux 本地虚拟环境，不要复用仓库中的 macOS
`.venv`：

```bash
python3 -m venv ~/.venvs/atrex-runtime
source ~/.venvs/atrex-runtime/bin/activate
python -m pip install -U pip
python -m pip install -e '.[dev]' -e './local-wiki[dev]'
```

启动脚本默认使用 `PATH` 中的 `python3`；如需指定其他 Linux 解释器，可设置
`ATREX_PYTHON`。Local Wiki 的查询子进程默认继承同一个解释器，示例配置不再保存任何
绝对 `.venv` 路径。

## 1. 测试语料

`gpu-wiki` 语料位于 `local-wiki/corpus/gpu-wiki`，是本仓库的普通内容。启动时无需额外
Checkout，也不需要网络下载。

## 2. 启动 Local Wiki

在 Runtime 仓库根目录执行：

```bash
bash examples/local-wiki/start-local-wiki.sh
```

服务监听 `http://127.0.0.1:8091`。在另一个终端验证：

```bash
curl -fsS http://127.0.0.1:8091/healthz
```

启动时不会修改 Vendored 语料，而是把它同步到被 Git Ignore 的可写 Store：
`local-wiki/state/gpu-wiki`。Query 使用该 Store。

也可以打开 [http://127.0.0.1:8091/](http://127.0.0.1:8091/) 使用浏览器客户端；每次 Query
直接显示完整的安全服务 Record 及其稳定 ID。

本示例自己的 [`runtime.json`](runtime.json) 已包含：

```json
{
  "gpu_wiki": {
    "base_url": "http://127.0.0.1:8091"
  }
}
```

真实配置还包含 Timeout 和字节限制字段，不要用上面的缩略片段覆盖完整对象。

## 3. 快速路径：打开用完即毁的 Wiki Agent Shell

调试 Wiki Tool 不需要完整 Campaign Bootstrap 或 Agate Gateway。只需保持 Local Wiki 运行，
然后执行：

```bash
bash examples/local-wiki/start-temporary-agent-shell.sh
```

包装脚本会创建临时 Campaign/Lineage/Epoch/Attempt 身份，并在随机 Loopback 端口启动临时
Runtime Wiki Proxy。它注入受限 Capability 和严格的 Core Attempt 环境，但不会启动 Agent
Backend。在 Shell 中直接使用未修改的 Core Runtime Tool：

```bash
python agent/optimizer/src/runtime_tools.py \
  wiki-query --request scratch/wiki-query.json
```

该路径仍经过 Runtime Wiki Proxy，以获得可信上下文和响应冻结；它只跳过常驻 Runtime Service、
Kernel Baseline 与 Agate。退出 Shell 后，临时 Proxy、Registry、Artifacts、Capability 和
Workspace 会全部删除。

可用 `--dsl cuda`、`--dsl triton` 或 `--dsl cutedsl` 修改临时知识范围，也可以加
`--shell bash`。

## 4. 完整路径：启动本示例自己的 Runtime 并 Bootstrap

Local example 会在首次运行时自动创建
`local-wiki/state/demo.env`，权限为 `0600`。其中只保存本地 Capability Signing Key
和 Admin Token，Runtime、Bootstrap 与 Agent Shell 包装脚本会自动加载同一文件。可以提前检查
文件是否已就绪，但不需要手工 `source`：

```bash
bash examples/local-wiki/prepare-demo-env.sh
```

Agent Provider 凭据不会写入这个文件。该 Example 的 Runtime 配置把 Optimizer 绑定到
QoderCLI，并从启动环境继承 `QODER_PERSONAL_ACCESS_TOKEN`。修改本 Example `runtime.json`
中的 `campaign.optimizer.agent_backend` 可以选择其他受支持 Backend。凭据值不会打印或复制进
Workspace。

保持外部 Agate 兼容 Gateway 监听 `127.0.0.1:9000`，然后分别运行：

```bash
# 终端 2
bash examples/local-wiki/start-runtime.sh

# 终端 3
bash examples/local-wiki/bootstrap-campaign.sh
```

Bootstrap 输出包含后续调试 Shell 使用的 `lineage_id`。
该流程只使用本目录的 `runtime.json` 与 `campaign.json`；Campaign 指向
`examples/shared/vecadd/` 中的标准 VecAdd Fixture。

## 5. 在持久 Agent Session 内使用 Runtime Tools

Runtime Tools 不是 Local Wiki 的直连客户端。它们调用 Attempt 范围的 Runtime Proxy：
`/v1/wiki/query`。因此以下命令必须在 Runtime 启动的 Core Baseline 或
Optimizer Session 内执行。Runtime 会注入可信 Manifest、Proxy URL 和范围 Capability；不要在
宿主 Shell 中人工创建或复制 Capability Token。

在该 Agent Workspace 内，当前目录是 Workspace 根目录，Runtime 已提供
`agent/optimizer/src/runtime_tools.py`。先把示例 Query 复制或重写到 Workspace 自己的
`scratch/`：

```bash
cp /path/to/atrex-runtime/examples/local-wiki/wiki-query.json scratch/wiki-query.json
python agent/optimizer/src/runtime_tools.py \
  wiki-query --request scratch/wiki-query.json
```

Agent 会看到上游 GPU Wiki 的 `records/notes` Envelope：

```json
{
  "records": {
    "nvidia.hopper.any.kernel-opt.mbarrier-software-pipeline.pipeline-in-gluon": {
      "store": "gpu_wiki",
      "source": "kernel_wiki",
      "type": "technique-card",
      "applies_to": {},
      "match": {"arch": "exact"},
      "payload": {}
    }
  },
  "notes": []
}
```

每个 Mapping Value 已经是完整的安全服务 Record。某条 Record 实际影响优化时，Agent 将准确的
Mapping Key 写进 `research_sources`；不存在第二次 Read 操作。Agent 不会看到协议版本、Snapshot
身份、Interaction Artifact Digest、Content
Digest、外部凭据或可信控制上下文；Runtime 内部仍会冻结这些信息，用于幂等重放和审计。

## 6. 打开真实托管的 Agent 调试 Session

调试 Session 会创建真实的 Optimizer Workspace、Attempt Manifest、Evidence View、工作 Kernel、
Core Bundle，并注入 Attempt 范围的 Gateway/Wiki Capability，但**不会启动 Agent Backend**。
最终进程是交互式 `zsh` 或 `bash`。

先保持 Local Wiki、Agate 兼容 Gateway 和 Runtime Service 运行，并完成上一步 Bootstrap。取输出
中的 `lineage_id`，在另一个终端执行：

```bash
bash examples/local-wiki/start-agent-shell.sh
```

无参数形式会读取 Bootstrap 包装脚本原子写入的
`local-wiki/state/last-bootstrap.json`，并使用其中第一条 `lineage_id`。仍然可以显式
指定 Lineage 和 Shell：

```bash
bash examples/local-wiki/start-agent-shell.sh \
  lineage_0123456789abcdef0123456789abcdef zsh
```

脚本始终使用本目录的 `runtime.json`。它等价于：

```bash
atrex-kernel-agent-runtime dev-shell \
  --config examples/local-wiki/runtime.json \
  --lineage lineage_0123456789abcdef0123456789abcdef \
  --shell zsh
```

`--lineage` 会为当前 Epoch 创建或复用第一个 Active Attempt。若已有一个 `running` Attempt，
也可以使用 `--attempt attempt_...` 为它创建新的 `run-<uuid>` Workspace。Shell 全程持有 Lineage
Fencing Lease；退出后 Workspace 会保留，Attempt 仍是 `running`，不会伪造 Session Trace、Token
Report 或终止结果。

进入 Shell 后可以直接检查 `$ATREX_ATTEMPT_MANIFEST`、`input/`、`work/kernel/`、
`agent/optimizer/` 和 `scratch/`。使用当前平台环境中的 Python 执行 Tool：

```bash
python3 \
  agent/optimizer/src/runtime_tools.py \
  wiki-query --request scratch/wiki-query.json
```

Capability 本身不会打印，但它存在于进程环境中，因此不要把 `env` 输出复制到日志或聊天。
调试 Shell 没有 Token 配额记账，因为 Agent 未启动；如果在 Shell 中主动调用 `submit`、
`evaluate` 或 Wiki Tool，这些仍是真实受控操作，会消耗 Capability Call Budget 并写入审计记录。
不要同时对同一 Lineage 运行 Campaign Scheduler。
