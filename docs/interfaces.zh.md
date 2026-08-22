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
| `list-kernel-trials` | `--config --attempt` | 查询 Attempt 内观察到的全部精确实验 Candidate。 |
| `show-kernel-trial` | `--config --trial` | 查询 Trial Operation/决策；`--source --result` 返回精确内容。 |
| `gc-artifacts` | `--config --minimum-age-seconds --limit` | CAS GC 预览；删除还需 `--apply --confirm-runtime-stopped`。 |
| `gc-workspaces` | 同上 | Worker Run GC 预览及确认删除。 |
| `digest-evolver-bundle` | `--path` | 校验并计算 Bundle Digest。 |

两个 Dev Shell 都支持 `--shell zsh|bash`。JSON 是稳定机器接口；Table 和进度消息属于运维展示。

## HTTP 权限与错误

- `GET /healthz`、`GET /readyz` 无需认证。
- `POST /v1/operations`、`POST /v1/wiki/query` 使用 Attempt 范围 Bearer Capability。
- 所有 `/v1/admin/*` 使用 `Authorization: Bearer <admin-token>`。
- Gateway/Wiki：`400` 请求错误，`403` 权限无效/过期/撤销，`409` 幂等或状态冲突，`503` 依赖不可用。
- Gateway `400` 中若包含可识别的 `operation`，响应会携带 `request_schema`：它由拒绝该请求的
  同一个 Pydantic Model 生成，是面向 Agent 的 JSON Schema。Schema 会移除 Runtime 自管字段，
  并把 `idempotency_key` 标为 Runtime 默认填充；缺少或无法识别 Operation 时则返回
  `supported_operations`。
- Administration：`400` 请求错误，`401` Token 错误，`404` ID 不存在，`409` 状态转换错误。

### Worker Route

| Method 与路径 | 请求/响应 |
| --- | --- |
| `POST /v1/operations` | Gateway v2；支持评测器 Operation，以及 Runtime 本地历史查询。 |
| `POST /v1/wiki/query` | Wiki v1 `{schema_version,attempt_id,idempotency_key,query}`；返回冻结响应。 |

Candidate 操作上传完整 Base64 File Bundle，Runtime 在执行前封存。相同 Key/Request 重放已提交响应；
同 Key 不同内容返回冲突。`evaluate` 生成探索性评测记录，但不会单独保留 Kernel Revision。

`measurements` 是不计配额的 Runtime 本地只读操作，不会调用 Agate。Runtime 从当前 Attempt
Capability 自动确定 Lineage 与可见边界。Agent 可按 `kind`、`kernel_revision_id`、
`kernel_artifact_digest`、`shape_id`、`kernel_name`、`metric` 和 `limit` 过滤，但不能自行指定
Lineage。返回值包含规范化的 Evaluate/Profile 标量、已注册时的 Kernel 版本元数据，以及原始
Gateway Result Artifact Digest。

`kernel_trials` 查询当前 Attempt 及其可见历史中的精确实验 Candidate，可按 `decision` 和 `limit`
过滤，返回 Trial ID、Observation 与 Agent 注解，但不直接返回源码。`kernel_trial_read` 接收 Trial
ID 与可选 `file`：省略 `file` 时返回已验证的文件索引，指定后返回精确 UTF-8 或 Base64 内容。
两者均不计配额、不访问 Agate，且调用方不能自行选择 Lineage 或 Attempt。
查询历史 Trial 的规范化 Evaluate/Profile 结果时，应把返回的 `candidate_artifact_digest` 传给
`measurements.kernel_artifact_digest`；Trial ID 和 Gateway Result Digest 都不是 Kernel Measurement
过滤条件。当前 Attempt 的结果来自原始 Operation 响应，Attempt 完成后才成为后续 Attempt 可查询的历史。

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
| `GET /v1/admin/attempts/{id}/kernel-trials` | 查询实验 Candidate，包括回退快照。 |
| `GET /v1/admin/attempts/{id}/kernel-trials/{trial}` | 查询 Trial Observation 与决策。 |
| `GET .../kernel-trials/{trial}/source` | 查询精确的未版本化 Candidate 文件。 |
| `GET .../kernel-trials/{trial}/results` | 查询保留的准确 Operation Result。 |
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
| `gateway-execute` | Gateway Operation 与参数；Candidate 操作上传当前 Working Kernel；`measurements`、`kernel_trials`、`kernel_trial_read` 查询 Runtime 持有的可见历史。 |
| `wiki-query` | `query` 和可选 `idempotency_key`；stdout 只返回 Wiki `content`。 |
| `record-experiment` | 增加 `candidate_artifact_digest`；测量结果在 `continue/revert/pivot` 前绑定准确 Gateway Candidate。 |
| `attempt-report` | Schema-v3 终态 Report，包含源码绑定实验，以及状态、计划、证据、解释、经验和下一方向。 |
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
