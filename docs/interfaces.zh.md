# 接口说明

[English](interfaces.md) | 中文

受支持的公共表面包括一个 CLI、三类 HTTP 权限、Core Runtime Tools 和冻结的 Evolver 检查工具。
除非明确说明，JSON Object 拒绝未知字段。类型化 ID 使用稳定前缀，例如 `campaign_`、`lineage_`、
`epoch_`、`attempt_`、`kernelrev_`、`agentrev_` 和 `sha256:`。

0.1 版本的 Python Module Graph 属于内部实现 API。受支持的嵌入边界是 CLI 或 HTTP Service；除非
本页明确列出，直接导入 `atrex_runtime.*` 不提供兼容保证。

## CLI

所有命令使用 `atrex-kernel-agent-runtime`。除 `digest-evolver-bundle` 外，读取部署状态的命令都
需要 `--config`。

| 命令 | 必要选择 | 作用 |
| --- | --- | --- |
| `serve` | `--config` | 提供健康、Gateway、Wiki 和管理 HTTP API。 |
| `bootstrap` | `--config --campaign <file>` | 幂等创建/继续 Campaign 和初始 Lineage。 |
| `seed-lineage` | `--config --campaign <id> --spec <file>` | 从 Artifact/Revision Root 创建 Lineage。 |
| `run-campaign` | `--config`，`--campaign` 或重复 `--lineage`，`--target-epoch N` | 运行到绝对 Epoch；可选 `--finalize`。 |
| `cancel-campaign` | `--config --campaign` | 取消静止 Campaign。 |
| `run-task-worker` | `--config` | 领取一个持久 Task；`--watch` 持续轮询。 |
| `recover-epoch` | `--config --epoch --recovery-key --reason` | 授权一次幂等失败 Epoch 重试。 |
| `dev-shell` | `--config`，`--lineage` 或 `--attempt` | 不启动 Core，进入真实 Optimizer Workspace。 |
| `temporary-dev-shell` | `--config --campaign <file>` | 进入用完即销毁的合成 Optimizer Workspace。 |
| `evolver-dev-shell` | `--config --lineage --epoch` | 进入重建的冻结 Evolution Workspace。 |
| `temporary-evolver-dev-shell` | `--config --campaign <file>` | 进入用完即销毁的合成 Evolution Workspace。 |
| `list-epochs` | `--config`，`--campaign` 或 `--lineage` | 竞争/胜者历史；`--format json|table`。 |
| `list-attempts` | 同上 | 包括无 Candidate 在内的所有 Attempt。 |
| `show-attempt` | `--config --attempt` | Attempt 与终态。 |
| `list-worker-sessions` | `--config` 加 Campaign/Lineage/Epoch/Attempt/Subject 之一 | Model Process/Trace 目录。 |
| `show-worker-session` | `--config --session` | 单个 Session 生命周期。 |
| `list-kernels` | `--config`，`--campaign` 或 `--lineage` | Kernel 版本历史。 |
| `show-kernel` | `--config --kernel` | Kernel、Agent、权威评测和重复测量。 |
| `list-agent-revisions` | `--config`，`--campaign` 或 `--lineage` | `agent-vN` 历史。 |
| `show-agent-revision` | `--config --agent-revision` | Agent Revision 与来源。 |
| `list-bootstrap-runs` | `--config --attempt` | 全部 Bootstrap Recovery Generation。 |
| `show-bootstrap-run` | `--config --attempt --generation N` | 一次物理 Bootstrap 执行。 |
| `list-evaluations` | `--config --attempt` | 全部不可变 Kernel/Result 评测对。 |
| `show-evaluation` | `--config --evaluation` | 元数据；`--source --result` 增加受限精确内容。 |
| `gc-artifacts` | `--config --minimum-age-seconds --limit` | CAS GC 预览；删除还需 `--apply --confirm-runtime-stopped`。 |
| `gc-workspaces` | 同上 | Worker Run GC 预览及确认删除。 |
| `digest-evolver-bundle` | `--path` | 校验并计算 Bundle Digest。 |

两个 Dev Shell 都支持 `--shell zsh|bash`。JSON 是稳定机器接口；Table 和进度消息属于运维展示。

## HTTP 权限与错误

- `GET /healthz`、`GET /readyz` 无需认证。
- `POST /v1/operations`、`POST /v1/wiki/query` 使用 Attempt 范围 Bearer Capability。
- 所有 `/v1/admin/*` 使用 `Authorization: Bearer <admin-token>`。
- Gateway/Wiki：`400` 请求错误，`403` 权限无效/过期/撤销，`409` 幂等或状态冲突，`503` 依赖不可用。
- Administration：`400` 请求错误，`401` Token 错误，`404` ID 不存在，`409` 状态转换错误。

### Worker Route

| Method 与路径 | 请求/响应 |
| --- | --- |
| `POST /v1/operations` | Gateway v2；支持 `evaluate`、`submit`、`profile`、`dev`、`check`、`sol`、`disassemble`、`poll`、`jobs`、`cancel`、`env`、`health`、`config`。 |
| `POST /v1/wiki/query` | Wiki v1 `{schema_version,attempt_id,idempotency_key,query}`；返回冻结响应。 |

Candidate 操作上传完整 Base64 File Bundle，Runtime 在执行前封存。相同 Key/Request 重放已提交响应；
同 Key 不同内容返回冲突。`evaluate` 生成探索性评测记录，但不会单独保留 Kernel Revision。

### Administration Route

| Method 与路径 | 作用 |
| --- | --- |
| `POST /v1/admin/campaigns/bootstrap` | 使用 Campaign schema v3 Bootstrap；HTTP 文件路径必须为绝对路径。 |
| `GET /v1/admin/campaigns/{id}` | Campaign 状态与冻结来源。 |
| `POST /v1/admin/campaigns/{id}/lineages` | 使用 schema v1 Seed Lineage。 |
| `POST /v1/admin/campaigns/{id}/cancel` | 取消静止 Campaign。 |
| `GET /v1/admin/campaigns/{id}/{epochs,attempts,kernels,agent-revisions,worker-sessions}` | Campaign 目录。 |
| `GET /v1/admin/lineages/{id}/{epochs,attempts,kernels,agent-revisions,worker-sessions}` | Lineage 目录。 |
| `GET /v1/admin/bootstrap-attempts/{id}/runs[/N]` | Bootstrap Generation 列表/详情。 |
| `GET /v1/admin/attempts/{id}` | Attempt 详情。 |
| `GET /v1/admin/attempts/{id}/worker-sessions` | Attempt Session。 |
| `GET /v1/admin/attempts/{id}/evaluations` | Evaluation 列表。 |
| `GET /v1/admin/attempts/{id}/evaluations/{eval}` | Evaluation 详情。 |
| `GET .../evaluations/{eval}/{source,result}` | 受限精确 Candidate 文件/原始结果。 |
| `GET /v1/admin/kernels/{id}` | 含 Measurement 的 Kernel 详情。 |
| `GET /v1/admin/kernels/{id}/{source,measurements}` | 精确文件或 Measurement。 |
| `GET /v1/admin/agent-revisions/{id}` | Agent Revision 详情。 |
| `GET /v1/admin/worker-sessions/{id}` | Worker Session 详情。 |
| `GET /v1/admin/epochs/{id}/worker-sessions` | Epoch Session。 |
| `POST /v1/admin/epochs/{id}/recover` | `{schema_version:1,recovery_key,reason}`。 |
| `POST /v1/admin/tasks` | `{schema_version:1,creation_key,campaign_id,target_epoch_number,finalize}`。 |
| `GET /v1/admin/tasks/{id}` | Task 状态。 |
| `POST /v1/admin/tasks/{id}/{cancel,requeue}` | Task 状态操作。 |
| `GET /v1/admin/events` | Event 分页；支持 `after`、`limit`、重复 `kind` 和关联 ID。 |
| `GET /v1/admin/events/export` | 同过滤条件的大批量受限 NDJSON 导出。 |
| `POST /v1/admin/events/prune` | `{schema_version:1,before_sequence,limit}` 前缀清理。 |
| `GET /v1/admin/metrics` | Event/Task 计数。 |

## Optimizer/Core Runtime Tools

Core 调用仓库内 `src/runtime_tools.py`。请求是 `scratch/` 下的 JSON Object；`--request` 不能逃逸。
Attempt ID、Capability 和 Candidate 文件由工具注入。

```bash
python src/runtime_tools.py <command> --request scratch/request.json
```

| 命令 | Agent 提供的请求 |
| --- | --- |
| `gateway-execute` | Gateway Operation 与参数；Candidate 操作自动上传当前 Working Kernel。 |
| `wiki-query` | `query` 和可选 `idempotency_key`；stdout 只返回 Wiki `content`。 |
| `record-experiment` | 严格 `name,hypothesis,change,evidence,result,decision`；Decision 为 `continue/revert/pivot`。 |
| `attempt-report` | Schema-v2 终态 Report：状态、假设、瓶颈、计划、修改/Profiling/评测证据、解释、来源、经验和下一方向。 |
| `lineage-bootstrap-report` | `baseline_ready` 或 `blocked`；Ready 时声明正延迟及 Candidate/Result Digest。 |

`attempt-report` 要求非空连续 Experiment Journal 且只能写一次。Runtime 不信任 Agent 的成功文本，
会独立读取 Gateway 记录并执行 Finalization。

## Evolver Runtime Tools

Evolver 获得无 HTTP 凭据的冻结 `runtime-tools/evolver_tools.py`。它只能读取物化的 Lineage/Epoch
Snapshot，唯一受控写操作是 `candidate-reset`。

| 命令 | 参数 | 结果 |
| --- | --- | --- |
| `history` | `--limit` | 已完成 Epoch、Active、Branch、Winner 与 Kernel 版本。 |
| `branches` | `--epoch` | 每个 Branch 的 Attempt/有效/失败/保留计数及最佳 Kernel。 |
| `attempts` | `--epoch --branch`，可选 `--trajectory --limit` | Attempt Summary 和 Evidence 路径。 |
| `kernels` | 可选 `--epoch --limit` | 冻结 Kernel Catalog。 |
| `kernel-read` | `--revision`，可选 `--file` | File Manifest 或一个 UTF-8 Source。 |
| `agents` | `--limit` | 冻结 Agent Catalog。 |
| `agent-diff` | `--base --candidate` | Repository 文件状态和 Diff。 |
| `candidate-reset` | `--base <agentrev>` | 原子加载已完成历史 Agent，并记录 Candidate Base。 |
| `trace-paths` | 可选 `--epoch --limit` | 原始、未脱敏 Session Trace Tree 路径。 |

`attempts` 的 Branch 名为 `active` 或 `challenger-NNNN`。`candidate-reset` 拒绝当前 Epoch
Challenger 以及不属于已完成 Lineage 历史的 Agent。

## 外部服务 Contract

- Agate 通过发布版 `atrex-gateway-client` SDK 调用；Runtime 持有凭据和请求构造，Worker 只看到
  安全投影。
- GPU Wiki Query 为 `POST /v1/knowledge/query`；Local Wiki 实现同一 v1 Contract。
- 完整 Schema、Evidence Layout、版本和 Bundle 语义见[协议](protocols.zh.md)，所有部署字段见
  [配置说明](configuration.zh.md)。
