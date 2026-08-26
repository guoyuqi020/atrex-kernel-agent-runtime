# 接口说明

[English](interfaces.md) | 中文

受支持的公共表面包括一个 CLI、三类 HTTP 权限、Core Runtime Tools 和 Evolver 的冻结文件系统输入 Contract。
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
| `show-attempt` | `--config --attempt` | Attempt、处置以及输入/终止 Runtime State Digest。 |
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
  同时移除 `idempotency_key`；响应还包含精简的 `issues`，给出 Agent 可见字段路径、稳定错误码和
  修复提示，但不回显请求值。缺少或无法识别 Operation 时则返回 `supported_operations`。
- Core 工具对预期失败输出单个 JSON Object 并以非零状态退出；保留 Runtime 的 `error`、`detail`、
  `issues`、`request_schema` 或 `supported_operations`，并补充 `status="error"`、`command` 及适用时的
  `http_status`，不再输出 Python Traceback。
- Core 自管的 Trial/Artifact/Result、Wiki、Direction、Experiment 和 Attempt Report 校验器会附加
  对应命令的 JSON Schema。可见性或生命周期错误还会给出有界 `recovery`：指定安全的 list/load
  调用，或说明应复用此前返回的哪类身份；不会枚举不可访问 Lineage 的身份。
- Agate 在创建 Job 前的拒绝归类为 Candidate/源码校验；安全校验详情会在递归移除评测输入、Reference、
  Shape、Payload 和日志后返回。执行隐藏 Case 后的失败仍保持脱敏。
- Administration：`400` 请求错误，`401` Token 错误，`404` ID 不存在，`409` 状态转换错误。

### Worker Route

| Method 与路径 | 请求/响应 |
| --- | --- |
| `POST /v1/operations` | Gateway v2；仅执行 GPU/Agate Operation。 |
| `POST /v1/runtime/queries` | 使用 Gateway v2 Envelope 执行不计配额的 Runtime 本地历史与源码查询。 |
| `POST /v1/runtime/journals` | 使用 Gateway v2 Envelope 执行不计配额、由 Runtime 自管的 Direction/Experiment Mutation 与读取。 |
| `POST /v1/wiki/query` | Wiki v1 `{schema_version,attempt_id,idempotency_key,query}`；返回冻结响应。 |

Candidate 操作上传完整 Base64 File Bundle，Runtime 在执行前封存。相同 Key/Request 重放已提交响应；
同 Key 不同内容返回冲突。`evaluate` 生成探索性评测记录，但不会单独保留 Kernel Revision。Wire
Response 仍保留 `schema_version` 以支持校验和持久化。Core 工具在向 Agent 打印结果前移除该顶层
字段，并把权威的 `evaluation.correct` 和 `evaluation.latency_us` 作为 `correct`、`latency_us`
合入 `result`，同时移除顶层 `evaluation` 和等价别名。对于 `dev`、`disassemble`、`jobs`、
`poll`、`cancel`、`env`、`health` 和 `config`，Core 直接返回 Agent-safe `result` Object。`profile` 还会在展平后的安全 Job
Result 旁返回 Kernel Artifact、Kernel Trial 和 Gateway Result 身份。其嵌套 `result` 仅用数字型
不透明 `shape_id` 标识 Shape，把 Kernel Duration 和常见资源字段规范化，并保留安全的 Profiler
Counters；同时增加 `kernel_count`、`total_duration_us`、逐 Kernel `duration_share_pct`、
`dominant_kernel`、按耗时加权的 `weighted_sol_pct` 和 `dominant_bound`。具体 Shape 输入和维度
仍然不可见。

`kernel_trial_show` 按 Gateway 响应或已保留 Experiment 记录返回的已知 `kernel_trial_id`
获取一条实验 Candidate 溯源记录。`kernel_artifact_read` 接收 Trial 的 `kernel_artifact_digest`（请求字段
名为 `kernel_artifact_digest`）、必填的 `scratch/` 下目标 `file`，以及可选的 Artifact 内源路径
`artifact_file`（默认取目标文件名）。Core 工具原子写入准确字节，stdout 只返回状态、路径、字节数
和 SHA-256。`gateway_result_read` 接收 Observation 的 `gateway_result_digest`，返回当时
规范化 Agent 可见视图。Evaluate 视图包含 `operation`、`status`、正确性结论、最坏情况的
`rel_err`、`max_abs_err`、`max_rel_err`、两种聚合延迟以及按不透明 Shape ID 的延迟；私有评测输入与隐藏 Case 细节仍不暴露。这些操作均不计配额、不访问 Agate，且调用方不能自行选择 Lineage
或 Attempt。当前 Attempt 的身份信息来自原始 Operation 响应和已保留的 Experiment 记录。

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
| `GET /v1/admin/attempts/{id}` | Attempt 详情，包括输入与终止 Runtime State Digest。 |
| `GET /v1/admin/attempts/{id}/report` | Runtime 最终 Attempt Report，融合 Agent Handoff 与 parent/Candidate 的权威 Gateway 结果。 |
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
| `GET /v1/admin/agent-revisions/{id}` | Agent Revision 详情，包括 Source 与 Runtime State Digest。 |
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
python3 src/runtime_tools.py <command> --request scratch/request.json
```

| 命令 | Agent 提供的请求 |
| --- | --- |
| `gateway-execute` | GPU/Agate Operation 与参数；Candidate 操作上传当前 Working Kernel。 |
| `kernel-trial-show` | 按 Trial ID 查询 Kernel Artifact Digest 和规范化 Gateway Results；请求 JSON 不写 `operation`。 |
| `kernel-artifact-read` | 按 Artifact Digest 把准确可见 Kernel 源码复制到必填的 `scratch/` 目标；stdout 只返回写入结果。 |
| `gateway-result-read` | 按 Gateway Result Digest 读取规范化的 Agent 可见测量；请求 JSON 不写 `operation`。 |
| `wiki-query` | `query`；Core 分配请求身份，stdout 只返回 Wiki `content`。 |
| `update-direction` | 以 `propose` 创建不可变 Direction 定义，或用 `start`、`complete`、`abandon`、`block`、`defer` 与分析更新现有 Direction；Experiment 关联自动派生，返回稳定 Direction ID。 |
| `list-directions` | 请求必须指定 `scratch/` 下的安全 `file`；工具把 Direction ID、名称和当前状态原子写入该文件，stdout 只返回状态、文件路径和条目数。 |
| `load-direction` | 请求只包含 `direction_id`，返回完整规范化 Direction；支持 ID 自动包含所有绑定它的可见 Experiment，以及状态事件在内部形成的关联快照。 |
| `record-experiment` | 记录 `direction_id`、测量对应的 Kernel/Trial/Result 身份、`evidence`、`analysis` 与 Action。普通比较要求完整 `before`/`after`；只有 Bootstrap 可用 `baseline`，此时 `before=null`、`after` 完整。返回稳定 Experiment ID。 |
| `list-experiments` | 请求必须指定 `scratch/` 下的安全 `file`；工具把冻结历史及当前实时 Journal 中的 Experiment ID、序号、名称和 Action 原子写入文件，stdout 只返回状态、文件路径和条目数。 |
| `load-experiment` | 请求只包含一个 `experiment_id`，返回该 Experiment 的完整原始记录。 |
| `attempt-report` | Schema-v12 终态 Agent Handoff，包含工程证据、Direction 事件及与 Direction 绑定的 Experiment；`framework_baseline` 和普通优化均使用它，Bootstrap 只允许 `candidate_ready` 或 `blocked`；不含重复的下一方向列表或顶层 `decision`。 |

`attempt-report` 要求匹配的非空 Runtime 自管 Direction/Experiment Journal。第一次成功调用会发布不可覆盖的终态
Report；校验或工具错误不会发布 Report，因此 Agent 可以依据 `issues`、`request_schema` 和 `recovery`
修正后重试，但成功后不得再次调用。每个 Experiment
必须绑定一个可见且已开始的 Direction；终态交接前，任何 Direction 都不能保持 in_progress，未产生
Experiment 的已启动 Direction 也必须 defer 或 block；complete 和 abandon 仍要求存在支持 Experiment。
每个 Attempt 最多可以启动并推进三个不同 Direction，包括继承和本 Attempt 新增的 Direction。仅 propose
不占推进名额，Report 也不限制保持 proposed/deferred 的 Direction 数量。同一时间只能有一个 Direction
处于 `in_progress`；启动第二个 Direction 会被 Runtime 原子拒绝，并返回
`direction_concurrency_conflict`、冲突 Direction ID 与修复步骤，请求的 Direction 状态保持不变。
Direction 的规范化状态是下一方向的唯一来源。Runtime 不信任 Agent 的成功文本，
会独立读取 Gateway 记录并执行 Finalization。
`update-direction` 与 `record-experiment` 是同步 Runtime Mutation：Runtime 校验并持久追加事件后才
返回稳定 ID。权威 Journal 绑定逻辑 Attempt，而非某个物理 Session 或 Recovery Generation；不再存在
作为权威数据的 `scratch/directions.json` 或 `scratch/experiments.json`。list/load 工具直接查询实时
Runtime Journal 与授权冻结历史的合并视图，只有显式请求的紧凑索引文件会写到 `scratch/`。
Bootstrap Session 开始时没有更早 Journal；成功后，
其终态 Journal、Kernel Trial 与 Gateway Result 会成为该 Lineage 后续普通 Attempt 的根历史。
`list-experiments` 和 `load-experiment` 把当前实时 Runtime Journal 与历史持久 Journal 合并；终态
Attempt Report Artifact 只作为旧数据的兼容回退。已完成 Epoch 包含获胜分支以及所有未获胜 Active/Challenger 分支的
Journal，但不向 Agent 暴露分支、Epoch、Attempt、选中状态或当前/历史来源；普通 Agent/Kernel Evidence
仍只保留已晋升路线。运行中
Epoch 仍只可见同 Trajectory 更早 Attempt，并行分支要到 Epoch barrier 后才会可见。它们使用
Attempt-scoped Runtime Journal Endpoint，不访问 Agate、不消耗 Gateway 配额，也不能任意选择 Attempt 或 Lineage。
Direction 历史遵循相同的已完成全路径/运行中同 Trajectory 可见边界；Agent-facing 结果不暴露
Branch、Epoch、Attempt、选中状态或当前/历史来源。
`load-direction` 会根据每个可见 Experiment 的 `direction_id` 反向派生关联；因此记录 Experiment 后，
对应 Direction 的读取视图立即更新。`update-direction` 会在内部状态事件中快照这些派生 ID，Agent
无需填写；实时关联与快照关联会合并到同一个去重列表中。
`profile_evidence` 必须为 `null`，或包含 `tool_used`、`profiler`、`profile_level`、
`bottleneck_type`、`evidence_summary`、`evidence_chain` 和非空 `supporting_results` 的精确
Object。每项 Supporting Result 绑定 `operation`（`profile` 或 `dev`）、
`kernel_artifact_digest`、`kernel_trial_id` 与 `gateway_result_digest`，且至少包含一项 Profile。
Core 要求每组绑定已出现在本 Attempt 的 Experiment Journal；Runtime 再核验声明的 Operation 与
三个身份确实匹配一条持久化、当前可见的 Gateway Observation。没有执行 Profile 时必须为 `null`。
每个 Finding 必须包含非空且唯一的 `supporting_experiment_ids`；每个 ID 都必须属于同一份随 Report
附加的 Experiment Journal。这样 Finding 可通过 Experiment 中实际存在的 before/after Subject 追溯到准确
Kernel Artifact、Trial 和 Gateway Result，而无需在 Finding 中重复这些身份。
Agent-facing `gateway-execute` 不暴露低层 Agate `submit` 和独立 `sol` 接口。评测只能使用由
Runtime 构造的 `evaluate`；SOL Profile 仍通过 `profile` 的 `level="sol"` 使用。

封存的 schema-v12 内容是 Agent Handoff，并非权威结果。Runtime 为管理接口和后续 Evidence
Snapshot 派生 schema-v1 最终 Attempt Report：保留工程叙述，并补充准确的 `parent_kernel` 与
`candidate_kernel`。Kernel 身份使用 `kernel_artifact_digest`，不暴露内部 Revision ID。每个 Kernel
都包含规范化 `gateway_result`，展示 Operation、完成状态、正确性、几何平均/算术平均延迟，以及按
不透明 Shape ID 索引的延迟。Correctness 包含 `status`，以及安全聚合后的最坏 relative-L2、逐元素
绝对误差和逐元素相对误差，但不暴露产生该值的隐藏 Shape/Case。Candidate 还包含 Runtime 判定的保留状态，以及相对 parent 的整体和
逐 Shape 对比。Kernel Outcome 投影本身不会重复 Gateway Result Digest；Experiment 溯源仍保留
准确 Result Digest。
Runtime 自管的 `production_gate` 会说明内容级生产策略是未启用、通过、失败，还是尚未执行；失败时
携带可信控制层给出的准确拒绝原因。

Agent Handoff 的 Schema 和已封存 Artifact 都不包含、也不要求 retention ABBA 操作。只有在
`candidate_ready` Handoff 被持久记录之后，Runtime 才会应用配置的
`kernel_retention_comparison`。当策略为 `same_allocation_abba` 时，Runtime 自行执行 ABBA，
以该权威 Gateway 结果更新 Candidate Kernel Revision，并且只通过 Runtime Final Attempt
Report 对外展示。缺失或非 ready 的 Handoff 会直接终结，不会运行 retention comparator。

```json
{
  "schema_version": 1,
  "attempt_id": "attempt_<id>",
  "status": "candidate_ready",
  "parent_kernel": {
    "version": "v2",
    "kernel_artifact_digest": "sha256:<parent>",
    "gateway_result": {
      "operation": "evaluate",
      "status": "completed",
      "correct": true,
      "correctness": {"status":"PASS","rel_err":null,"max_abs_err":0.0009765625,"max_rel_err":0.0078125},
      "latency_us_geomean": 200.0,
      "latency_us_arith_mean": 205.0,
      "latency_us_by_shape": {"0": 120.0, "1": 290.0}
    }
  },
  "candidate_kernel": {
    "version": "v3",
    "kernel_artifact_digest": "sha256:<candidate>",
    "status": "retained",
    "gateway_result": {
      "operation": "same_allocation_abba",
      "status": "completed",
      "correct": true,
      "correctness": {"status":"PASS","rel_err":null,"max_abs_err":0.0009765625,"max_rel_err":0.0078125},
      "latency_us_geomean": 173.28,
      "latency_us_arith_mean": 180.0,
      "latency_us_by_shape": {"0": 100.0, "1": 260.0}
    },
    "comparison_with_parent": {
      "latency_us_geomean_delta": -26.72,
      "improvement_percent": 13.36,
      "latency_us_delta_by_shape": {"0": -20.0, "1": -30.0},
      "improvement_percent_by_shape": {"0": 16.667, "1": 10.345}
    }
  },
  "production_gate": {
    "enabled": true,
    "result": "PASS",
    "failure_reason": null
  }
}
```

### 已知 Kernel 证据工具示例

以下输入是 `--request` 指向文件中的 JSON；Digest 和 ID 仅为便于阅读而缩写。

`kernel-trial-show` 只返回 Kernel 身份和规范化 Gateway Results：

```json
{"kernel_trial_id":"gtrial_<id>"}
```

```json
{"kernel_artifact_digest":"sha256:<kernel>","gateway_results":[{"operation":"evaluate","status":"completed","result":{"correct":true,"correctness":{"status":"PASS","rel_err":null,"max_abs_err":0.0009765625,"max_rel_err":0.0078125},"latency_us_geomean":12.288,"latency_us_arith_mean":12.400,"latency_us_by_shape":{"0":12.288}}}]}
```

`kernel-artifact-read` 把一个 Artifact 文件复制进 `scratch/`，不会打印源码：

```json
{"kernel_artifact_digest":"sha256:<kernel>","artifact_file":"kernel.py","file":"scratch/recovered/kernel.py"}
```

```json
{"status":"completed","file":"scratch/recovered/kernel.py","bytes":4281,"sha256":"<file-sha256>"}
```

`gateway-result-read` 读取一条规范化的 Agent 可见 Result：

```json
{"gateway_result_digest":"sha256:<result>"}
```

```json
{"operation":"evaluate","status":"completed","result":{"correct":true,"correctness":{"status":"PASS","rel_err":null,"max_abs_err":0.0009765625,"max_rel_err":0.0078125},"latency_us_geomean":12.288,"latency_us_arith_mean":12.400,"latency_us_by_shape":{"0":12.288}}}
```

## Evolver 文件系统接口

Evolver 没有 Runtime Tool 或 Runtime HTTP Capability。Runtime 物化一份冻结文件视图：当前参赛仓库和
Runtime State 位于 `input/agents/`，最近完成 Epoch 的汇总和 Conversation 位于
`input/evidence/`，已完成且非当前的 Agent 版本位于 `input/historical/agent-vN/`，此前 Agent 创建报告
按 `input/evolution-reports/evo-N.json` 排序并关联 Source Base/产出 Agent 路径。唯一可写的 Agent
组件是 `candidate/source/` 与 `candidate/runtime-state/`。

`runtime-state/trajectories/<N>/{skills,tools}/` 是唯一的自适应 Skill/Tool 存储，它是 Optimizer Session
积累的非版本化状态，作用域是某个 Agent Revision 的某条 Trajectory。根级 `skills/` 和 `tools/`
是保留路径，在版本化 Agent Source 中会被拒绝。Evolver 可把冻结状态整理为唯一一份扁平的
`candidate/runtime-state/{skills,tools}/` 种子，也可修改版本化 Prompt、Workflow、Memory Policy 或实现，
以改进未来 Session 的状态使用方式。Runtime 始终分别封存 Candidate Source 与 State，并把两者组合为
同一个不可变 Agent Bundle；所有新 Trajectory 都从该 Bundle 的 State 初始化。

选择 `evolve_from_history` 时，Evolver 用所选历史 `source/` 的完整可写副本替换 Candidate Source，
可从可见历史状态整理公共种子，并声明 `kernel_agent_revision_id`；该 Revision 的 Source 是提案参考。
Runtime State 身份仍是 Runtime 私有控制数据。Runtime 验证 Source 资格、报告的 Source 根目录相对
Diff 和私有 State Diff；不再存在 Candidate Base 旁路记录或 Reset 命令。

Evolver 使用 Bundle 本地的 `evolution-report` 命令提交终态 Draft。校验错误返回 `issues`、
`request_schema` 与 `recovery`；失败不发布且可修正后重试。第一次成功调用原子发布
`scratch/evolution-report.json` 并返回紧凑回执。

## 外部服务 Contract

- Agate 通过发布版 `atrex-gateway-client` SDK 调用；Runtime 持有凭据和请求构造，Worker 只看到
  安全投影。
- GPU Wiki Query 为 `POST /v1/knowledge/query`；Local Wiki 实现同一 v1 Contract。
- 完整 Schema、Evidence Layout、版本和 Bundle 语义见[协议](protocols.zh.md)，所有部署字段见
  [配置说明](configuration.zh.md)。
