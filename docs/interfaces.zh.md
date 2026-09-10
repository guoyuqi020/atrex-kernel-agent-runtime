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
| `seed-ablation-arm` | `--config --spec <file>` | 从源 Lineage 的 Bootstrap Baseline 创建独立 Campaign 中的控制 Lineage，可配置进化调度。 |
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
- Gateway `503` 返回 `error="gateway_unavailable"` 和可操作的 `detail`（最多 8 KiB）。
  Runtime 日志保留完整异常链和 Attempt/Operation/Request Digest 关联；
  `gateway.operation_failed` 同时记录异常类型及正文。公开详情隐藏凭据，并对包含私有测试内容的
  源码树错误返回摘要。Runtime 返回 503 不代表 Agate 原始 HTTP 状态也是 503，也不保证可重试。
- Gateway `400` 中若包含可识别的 `operation`，响应会携带 `request_schema`：它由拒绝该请求的
  同一个 Pydantic Model 生成，是面向 Agent 的 JSON Schema。Schema 会移除 Runtime 自管字段，
  同时移除 `idempotency_key`；响应还包含精简的 `issues`，给出 Agent 可见字段路径、稳定错误码和
  修复提示，但不回显请求值。缺少或无法识别 Operation 时则返回 `supported_operations`。
- Core 工具对预期失败输出单个 JSON Object 并以非零状态退出；保留 Runtime 的 `error`、`detail`、
  `issues`、`request_schema` 或 `supported_operations`，并补充 `status="error"`、`command` 及适用时的
  `http_status`，不再输出 Python Traceback。
- Core 和 Kernel Design Agent 的本地 Evaluate 文件错误通过 `issues[].path` 精确指向
  出错字段，并附上对应的本地 `request_schema`。Evaluate 包含规范的
  `full`/`correctness_only` 模式、内联/文件参数及形式互斥约束，同时提供针对该字段的有界
  `recovery` 步骤。Evaluate 支持可选 `candidate_path`；其 `comparison` 对象要求
  `method: "abba"` 及 `baseline_path`，`repeats` 范围为 2–20。启用比较时 `mode` 必须为
  `full`。嵌套字段错误指向 `comparison.method`、`comparison.baseline_path` 或
  `comparison.repeats`；输入文件
  错误指向 `input_path`/`shapes_path`。这些操作已有的 Runtime `issues`、`request_schema` 和 `recovery`
  会保留，不被本地备用提示覆盖。
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
| `POST /v1/wiki/query` | 保留的 Wiki 集成端点；不再向新建托管 Agent Session 下发 Wiki 权限。 |

Candidate 操作上传完整 Base64 File Bundle，Runtime 在执行前封存。相同 Key/Request 重放已提交响应；
同 Key 不同内容返回冲突。`evaluate` 生成探索性评测记录，但不会单独保留 Kernel Revision。Runtime
将 Agate 原始响应及其 `gateway_result_digest` 保留为私有事实，只供评测、比较和审计；同时把规范化的
Agent 可见 `operation`、`status`、`result` 独立封存为 Result Artifact。Agent 始终收到
`result_artifact_digest`，初次执行和后续读取暴露同一份规范化内容，不暴露私有 Gateway Result 身份。

`evaluate` 的 Wire Request 可指定 `mode: "full" | "correctness_only"`（默认 `full`）、
`input_py`（UTF-8 Python 输入生成器源码，上限 128 KiB）和 `shapes`（以整数字符串为键的非空
Agate Shape Record Object，每条记录也是 Object）。两个输入组件可独立覆盖；未指定的源码或
Shapes 继续复用私有 Contract，Reference 和可信评测策略保持不变。`correctness_only` 不测性能、
不自动 Profile。自定义输入或仅正确性调用仍保留 Kernel Trial 和 Result Artifact 身份，其嵌套
`result` 记录 `mode` 与 `input_scope`（`custom` 或 `contract`）。这些调用不能满足
`candidate_ready` 前所需的可信 Contract 完整评测；默认 `{"operation":"evaluate"}` 行为不变。

完整文件内容见[自定义输入文件示例](evaluation.zh.md#自定义输入文件示例)，其中说明了
`input_kwargs` → `_make_inputs`、返回字典 → `Model.forward`、`init_kwargs` → Model 构造函数
的对应关系。HTTP 接收文件内容，不接收本地文件路径。

携带 `comparison: {method: "abba", repeats: 2}` 的 `evaluate` Wire Request 通过 `baseline`（A）
和 `candidate`（B）上传两个源码 Bundle。`comparison.repeats` 默认 2，范围 2–20；比较要求
`mode: "full"`，`input_py` 和 `shapes` 仍可选。Runtime 校验并封存两侧源码，在每个 Shape
Batch 的同一 Allocation 内交错观测；`comparison.repeats: 2` 生成
A、B、B、A，更长 Schedule 还须满足 Allocation 预算。ABBA 始终仅用于探索，不能满足
`candidate_ready` 或触发 Retention/Promotion。响应返回 B 的 `kernel_trial_id`、
`kernel_artifact_digest` 与一个 `result_artifact_digest`，并保留 `operation: "evaluate"`。
`result.comparison` 记录 `method: "abba"` 及 `repeats`；嵌套结果还包含
`baseline_kernel_artifact_digest`、`baseline`/`candidate` 摘要、`schedule`、所有 `measurements`，
以及 A/B `speedup` 和 `improvement_pct`。结果通过 Trial/Result Artifact 查询读取，不进入普通
Evaluate 历史路由。详见[评测说明](evaluation.zh.md#探索性-abba)。

对于 `dev`、`disassemble` 和 `env`，Core 直接返回 Agent-safe
`result` Object。`profile` 还会在展平后的安全 Job
Result 旁返回 Kernel Artifact、Kernel Trial 和 Result Artifact 身份。其嵌套 `result` 仅用数字型
不透明 `shape_id` 标识 Shape，把 Kernel Duration 和常见资源字段规范化，并保留安全的 Profiler
Counters；同时增加 `kernel_count`、`total_duration_us`、逐 Kernel `duration_share_pct`、
`dominant_kernel`、按耗时加权的 `weighted_sol_pct` 和 `dominant_bound`。具体 Shape 输入和维度
仍然不可见。

多文件源码树的 `profile/check/disassemble` 保留同一套接口，内部由 Runtime 通过 Agate Dev
运行完整封存源码树和固定诊断驱动。Check 是单个 case 的编译/运行探针，可选 Compute Sanitizer，
不是正确性 Gate；transport completed 也必须检查诊断 `passed`。NVIDIA 工具依赖、参数、导出
及限制见[源码树诊断说明](source-trees.zh.md#profilecheck-与-disassemble)。

`kernel_trial_show` 按 Gateway 响应或已保留 Experiment 记录返回的已知 `kernel_trial_id`
获取一条实验 Candidate。它返回 Kernel Artifact Digest，以及由 Result Artifact Digest、Operation 和
Status 组成的精简索引；仅对需要分析的条目调用 `result_artifact_read` 展开内容。
`kernel_artifact_read` 接收 Trial 的 `kernel_artifact_digest`（请求字段
名为 `kernel_artifact_digest`）、必填的 `scratch/` 下目标 `file`，以及可选的 Artifact 内源路径
`artifact_file`（默认取目标文件名）。Core 工具原子写入准确字节，stdout 只返回状态、路径、字节数
和 SHA-256。`result_artifact_read` 接收 Observation 的 `result_artifact_digest`，读取规范化的
Agent 可见 Result Artifact；返回的 `operation`、`status` 和 `result` 与初次调用一致。
不带比较的 Evaluate 视图包含正确性结论及最坏情况的 `rel_err`、`max_abs_err`、`max_rel_err`。
完整评测还返回两种聚合延迟及按不透明 Shape ID 的延迟；仅正确性结果不包含性能测量。
自定义输入和仅正确性视图保留 `mode` 与 `input_scope`。比较结果使用前述 `result.comparison`
标记与 A/B 摘要；私有评测输入与隐藏 Case 细节仍不暴露。
这些操作均不计配额、不访问 Agate，且调用方不能自行选择 Lineage
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

`gateway-execute` 的 `operation: "evaluate"` 除 Wire 内联字段外，还支持 `input_path` 和
`shapes_path`。它们是指向普通 UTF-8 文件的安全 Workspace 相对路径，分别承载 Python 输入源码
（上限 128 KiB）和 Shape Record JSON Object（上限 256 KiB）。Core 会拒绝绝对路径或路径穿越、
符号链接、`.runtime` 控制路径、缺失或特殊文件、无效 UTF-8/JSON，以及同一组件同时指定内联和
路径形式。文件内容在计算请求幂等键前展开，因此内容变化会生成新 Key，与内联内容等价的请求则
得到相同 Key。例如：

```json
{"operation": "evaluate", "mode": "correctness_only", "input_path": "scratch/custom-input.py", "shapes_path": "scratch/custom-shapes.json"}
```

使用 `{"operation":"evaluate","mode":"correctness_only"}` 可仅检查 Contract 输入的正确性；
使用 `{"operation":"evaluate"}` 则执行默认的可信 Contract 完整评测。

若覆盖文件在本地加载失败，先按 `issues[].path` 与 `recovery` 修正指定路径、普通文件内容、
UTF-8 编码、Shape JSON Object 或内联/路径冲突，再重试。附带的 `request_schema` 描述 Agent
可填写的 Evaluate 请求，包含内联和路径两种形式，不暴露可信请求字段或私有评测输入。

对于 `operation: "evaluate"`，Core 与 Kernel Design Agent 支持可选 `candidate_path`（默认
`work/kernel`）。启用 ABBA 比较时，还需提供
`comparison: {method: "abba", baseline_path: "scratch/baseline.py"}`。两侧路径均为指向 `.py` 源文件或 Kernel Bundle 目录的
Workspace 相对路径；单个 `.py` 文件映射为 `kernel.py`，目录保留相对文件名。两侧均应用安全
路径规则及 Candidate Bundle 大小限制。工具自动上传 `baseline` 和 `candidate`，Agent 不能
直接填写这两个 Wire 字段；本地路径在计算内容幂等键前移除，Wire `comparison` 仅保留
`method` 和可选 `repeats`。也可使用相同的输入
文件参数，让两侧共用自定义输入生成器和 Shapes。

```json
{"operation": "evaluate", "candidate_path": "scratch/candidate.py", "comparison": {"method": "abba", "baseline_path": "scratch/baseline.py", "repeats": 2}}
```

| 命令 | Agent 提供的请求 |
| --- | --- |
| `gateway-execute` | GPU/Agate Operation 与参数；Candidate 操作默认上传当前 Working Kernel。Evaluate 可通过 `candidate_path` 选择 B，通过 `comparison.baseline_path` 选择比较基线 A。 |
| `kernel-trial-show` | 按 Trial ID 查询 Kernel Artifact Digest 和精简 Result Artifact 索引；请求 JSON 不写 `operation`。 |
| `kernel-artifact-read` | 按 Artifact Digest 把准确可见 Kernel 源码复制到必填的 `scratch/` 目标；stdout 只返回写入结果。 |
| `result-artifact-read` | 按 Result Artifact Digest 读取规范化的 Agent 可见结果；请求 JSON 不写 `operation`。 |
| `update-direction` | 以 `propose` 创建不可变 Direction 定义，或用 `start`、`complete`、`abandon`、`block`、`defer` 与分析更新现有 Direction；Experiment 关联自动派生，返回稳定 Direction ID。 |
| `list-directions` | 请求必须指定 `scratch/` 下的安全 `file`；工具把 Direction ID、名称和当前状态原子写入该文件，stdout 只返回状态、文件路径和条目数。 |
| `load-direction` | 请求只包含 `direction_id`，返回完整规范化 Direction；支持 ID 自动包含所有绑定它的可见 Experiment，以及状态事件在内部形成的关联快照。 |
| `record-experiment` | 记录 `direction_id`、前后 Kernel Trial ID、`evidence`、`analysis` 与 Action；Runtime 冻结 Trial 对应的 Kernel 和 Result Artifact 身份。只有 Bootstrap 可用 `baseline` 与 `before=null`。返回稳定 Experiment ID。 |
| `list-experiments` | 请求必须指定 `scratch/` 下的安全 `file`；工具把冻结历史及当前实时 Journal 中的 Experiment ID、名称、Hypothesis、Change、Evidence、Analysis 和 Action 原子写入文件，stdout 只返回状态、文件路径和条目数。 |
| `load-experiment` | 请求只包含一个 `experiment_id`，返回该 Experiment 的完整 Agent 可见记录，不包含 Runtime 内部排序元数据。 |
| `attempt-report` | Schema-v12 终态 Agent Handoff，包含工程证据、Direction 事件及与 Direction 绑定的 Experiment；`framework_baseline` 和普通优化均使用它，Bootstrap 只允许 `candidate_ready` 或 `blocked`；不含重复的下一方向列表或顶层 `decision`。 |

示例配置与生产 Workspace 生成器将 `campaign.optimizer.max_attempt_report_bytes` 设为
`1048576`（1 MiB），限制包含工具自动附加 Journal 的完整终态 Report。Core/KDA 在提交前检查
组装后的大小，Runtime 代理在接受前检查，Worker 在 Session 结束后读取文件时再次检查。
Core/KDA 的每份 `--request` JSON 文件也限制为 1 MiB，按实际文件字节数计量，包含空白。
这两层限制相互独立；HTTP 请求体限制及自定义输入源码、Shape 文件限制保持不变。
已有 Workspace 配置会保留原值，需要显式更新才会生效。

### 模型正常退出后的报告补交

`campaign.optimizer.report_completion_retries` 默认 `2`，允许整数 `0..10`。
模型调用成功退出后，Core/KDA 用当前 Attempt Capability 调用
`POST /v1/runtime/queries` 的 `attempt_report_status`。
该 Harness 内部查询返回 `missing` 或包含封存 Report 的 `accepted`，不计工具配额，
也不启动 Agate Job；本地 Agent 文件或聊天中的“已完成”不等价于接受回执。
已接受的报告必要时恢复到本地，不重复提交。

若尚未接受，Harness 最多追加指定次数的报告补交调用，沿用同一个 Attempt 和工作区，
提示读取已有 Journal、草稿和 Trace。这是新的 Provider 对话，不是原生 resume，也不是新一轮
优化，不能编造缺失测量。尚未接受的本地终态文件会移到唯一的 scratch 备份，避免阻塞重新提交。
所有调用共同消耗原有总时间和 Token/Credit 配额；模型非零退出、超时、配额耗尽、
Provider 捕获或用量不完整时，不触发补交。`0` 关闭追加调用，但仍检查报告是否接受。

初次调用的 Trace 保留在根目录，后续调用分别放在 `continuations/001/`、`002/` 等目录。
根 `session.json` 的 `segments` 与 `report_completion` 记录分段及补交状态；
`conversation.jsonl` 按分段身份合并对话，Provider 用量累计计算。
补交耗尽时 Worker Session 记录 `report-completion-exhausted`，不会产生成功 Candidate。
普通 Epoch 可以继续下一个 Attempt，Bootstrap 则失败而非登记 Baseline。
该机制适用于 Core/KDA 的优化与 Framework Baseline，不用于 Problem Generalization 或 Evolver。
冻结的旧 Agent Commit 需要升级后才能使用这套机制。

### 终态交接与 Journal

`candidate_ready` 要求匹配的非空 Runtime 自管 Direction/Experiment Journal 及有实验支持的 Findings。
若未能开展实验，`blocked` 和 `pivot` 允许 Journal 与 Findings 为空；报告需如实说明原因，不应虚构实验。
已有 in_progress Direction 仍须先 block 或 defer。第一次成功调用 `attempt-report` 会发布不可覆盖的终态
Report；校验或工具错误不会发布 Report，因此 Agent 可以依据 `issues`、`request_schema` 和 `recovery`
修正后重试，但成功后不得再次调用。每个 Experiment 必须绑定可见的 `in_progress` Direction，
或已关闭的 Direction（`completed`、`abandoned`、`blocked`、`deferred`）。允许关闭后补交已有证据：
Experiment 追加到当前 Attempt 的 Journal，不重新打开 Direction、不改变其状态、不改写历史事件；
加载 Direction 时，其支持 Experiment ID 会自动包含补录条目。仅处于 `proposed` 的 Direction 仍须先
start。Trial 可见性、归属和证据校验不变；补录不代表可以不经 start 就恢复研究。
终态交接前，任何 Direction 都不能保持 in_progress，未产生
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
其终态 Journal、Kernel Trial 与 Result Artifact 会成为该 Lineage 后续普通 Attempt 的根历史。

采纳可见历史中的原样 Kernel 时，使用 `record-experiment` 的 `action="adopt"`，before/after 都填写
真实 Kernel Trial ID。区别于其他动作，`adopt` 允许历史 after：Runtime 要求该精确 Kernel 有成功的
普通完整 Evaluate、已提交且匹配的 Result Artifact，以及一致的算子、硬件、DSL 和封存评测 Contract。
现有历史可见边界保持不变，包括显式继承的 Bootstrap 历史。自定义输入、仅正确性检查、Profile 和探索性
ABBA 不符合采纳资格。Experiment 记录当前采纳决策，保留原始 Trial/测量身份，不新增测量或修改历史
Trial 的 disposition。这条持久化采纳记录可以满足 `candidate_ready` 预检，无需重测原样候选；如果更改
候选，则需为新的精确内容提供证据。本 Attempt 新的完整 Evaluate 已失败时，不能用更早成功记录覆盖。
请求幂等按 Attempt 与 Recovery Generation 隔离，不是全历史同 Artifact 禁止评测。
`list-experiments` 和 `load-experiment` 把当前实时 Runtime Journal 与历史持久 Journal 合并；终态
Attempt Report Artifact 只作为旧数据的兼容回退。已完成 Epoch 包含获胜分支以及所有未获胜 Active/Challenger 分支的
Journal，但不向 Agent 暴露分支、Epoch、Attempt、选中状态或当前/历史来源；普通 Agent/Kernel Evidence
按分支标签保留每个已完成分支。运行中
Epoch 仍只可见同 Trajectory 更早 Attempt，并行分支要到 Epoch barrier 后才会可见。它们使用
Attempt-scoped Runtime Journal Endpoint，不访问 Agate、不消耗 Gateway 配额，也不能任意选择 Attempt 或 Lineage。
Direction 历史遵循相同的已完成全路径/运行中同 Trajectory 可见边界；Agent-facing 结果不暴露
Branch、Epoch、Attempt、选中状态或当前/历史来源。
`load-direction` 会根据每个可见 Experiment 的 `direction_id` 反向派生关联；因此记录 Experiment 后，
对应 Direction 的读取视图立即更新。`update-direction` 会在内部状态事件中快照这些派生 ID，Agent
无需填写；实时关联与快照关联会合并到同一个去重列表中。
`profile_evidence` 必须为 `null`，或包含 `tool_used`、`profiler`、`profile_level`、
`bottleneck_type`、`evidence_summary`、`evidence_chain` 和非空 `supporting_results` 的精确
Object。每项 Supporting Result 绑定 `operation`（仅允许 `profile`）、
`kernel_artifact_digest`、`kernel_trial_id` 与 `result_artifact_digest`。Core/KDA 根据 Runtime
投影的 `citable_profile_results` 检查引用；Runtime 再独立核验三个身份与 Operation 是否匹配
持久化、当前可见的 Gateway Observation。不要求先被 Experiment 引用：历史 Profile，以及在
Experiment 快照之后取得的 Profile，都无需补录 Journal 或重新打开 Direction 即可引用。
原有历史可见性边界保持不变。尚无 Result Artifact 的进行中操作，以及非 Profile 操作，不能引用。
没有已记录的 Profile 证据时必须为 `null`。
每个 Finding 必须包含非空且唯一的 `supporting_experiment_ids`；每个 ID 都必须属于同一份随 Report
附加的 Experiment Journal。这样 Finding 可通过 Experiment 中实际存在的 before/after Subject 追溯到准确
Kernel Artifact、Trial 和 Result Artifact，而无需在 Finding 中重复这些身份。
`contributing_kernel_trial_ids` 是必填数组，列出本次 Attempt 取用过其代码或思路的历史
Kernel Trial；没有取用时为空。Core/KDA 和 Runtime 接受任意顺序及重复 ID，在提交或封存 Report 前
自动排序、去重；仍逐项校验 ID 格式，并在去重前限制输入最多 64 项。两侧都不去解析它是否在可见历史内 ——
因为该字段是 Agent 的解读而非测量事实；Experiment Subject 身份则必须通过 Runtime 自管 Trial 的核验。Runtime 会把它带入
派生的 Final Report，供后续 Attempt 与 Evolver 阅读。
Gateway 不定义低层 Agate `submit` 透传，也不定义独立的 `sol` 操作。评测只能使用由
Runtime 构造的 `evaluate`，其中可包含探索性比较；SOL Profile 仍通过 `profile` 的 `level="sol"` 使用。

封存的 schema-v12 内容是 Agent Handoff，并非权威结果。Runtime 为管理接口和后续 Evidence
Snapshot 派生 schema-v1 最终 Attempt Report：保留工程叙述，并补充准确的 `parent_kernel` 与
`candidate_kernel`。Kernel 身份使用 `kernel_artifact_digest`，不暴露内部 Revision ID。每个 Kernel
都包含规范化 `gateway_result`，展示 Operation、完成状态、正确性、几何平均/算术平均延迟，以及按
不透明 Shape ID 索引的延迟。Correctness 包含 `status`，以及安全聚合后的最坏 relative-L2、逐元素
绝对误差和逐元素相对误差，但不暴露产生该值的隐藏 Shape/Case。Candidate 还包含 Runtime 判定的保留状态，以及相对 parent 的整体和
逐 Shape 对比。Kernel Outcome 投影本身不会重复私有 Gateway Result Digest；Experiment 溯源保留
准确的 Agent 可见 Result Artifact Digest。
Runtime 自管的 `production_gate` 会说明内容级生产策略是未启用、通过、失败，还是尚未执行；失败时
携带可信控制层给出的准确拒绝原因。

Agent Handoff 的 Schema 和已封存 Artifact 都不包含、也不要求 retention ABBA 操作。只有在
`candidate_ready` Handoff 被持久记录之后，Runtime 才会应用配置的
`kernel_retention_comparison`。当策略为 `same_allocation_abba` 时，Runtime 自行执行 ABBA，
以该权威 Gateway 结果更新 Candidate Kernel Revision，并且只通过 Runtime Final Attempt
Report 对外展示。缺失或非 ready 的 Handoff 会直接终结，不会运行 retention comparator。
权威比较不会生成 Agent `gtrial`，不能等待它来补齐交接前的 Experiment Journal。Agent 主动调用的
ABBA 有自己的候选 Trial，但仍是探索性比较，不能替代提名要求的成功普通完整 Evaluate。

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

`kernel-trial-show` 只返回 Kernel 身份和精简 Result Artifact 索引：

```json
{"kernel_trial_id":"gtrial_<id>"}
```

```json
{"kernel_artifact_digest":"sha256:<kernel>","result_artifacts":[{"result_artifact_digest":"sha256:<evaluate-result>","operation":"evaluate","status":"completed"},{"result_artifact_digest":"sha256:<profile-result>","operation":"profile","status":"completed"}]}
```

`kernel-artifact-read` 把一个 Artifact 文件复制进 `scratch/`，不会打印源码：

```json
{"kernel_artifact_digest":"sha256:<kernel>","artifact_file":"kernel.py","file":"scratch/recovered/kernel.py"}
```

```json
{"status":"completed","file":"scratch/recovered/kernel.py","bytes":4281,"sha256":"<file-sha256>"}
```

`result-artifact-read` 读取一条规范化的 Agent 可见 Result Artifact：

```json
{"result_artifact_digest":"sha256:<result-artifact>"}
```

```json
{"operation":"evaluate","status":"completed","result":{"correct":true,"correctness":{"status":"PASS","rel_err":null,"max_abs_err":0.0009765625,"max_rel_err":0.0078125},"latency_us_geomean":12.288,"latency_us_arith_mean":12.400,"latency_us_by_shape":{"0":12.288}}}
```

## Evolver 文件系统接口

Evolver 没有 Runtime Tool 或 Runtime HTTP Capability。Runtime 物化一份按 Lineage 版本索引的冻结文件
视图。`input/agents/agent-vN/` 是完整 Agent Bundle，直接包含实现、配置及
`prompts/`、`memory/`、`knowledge/`、`skills/`、`tools/`、`hooks/`。可写 `candidate/` 使用相同布局。
已有 Checkpoint 替换打包默认内容，无需再分别编辑 Source/State。

`input/evidence/agent-vN/` 保存优化效果汇总和补充的 `resources/trajectories/<N>/` 快照。
仅上一个完成 Epoch 的参赛者拥有该 Epoch 的 Conversation 与 Attempt Report。历史报告
`input/evolution-reports/evo-N.json` 的 `parent.path` 和 `generated_agent.path` 指向完整 Bundle；
贡献路径属于原始生成 Session，不保证当前资源仍与原始内容一致。

从历史派生时先复制完整历史 Bundle 到 Candidate，再修改；报告所选 `kernel_agent_revision_id`，
`changed_paths` 为相对于 Bundle 根目录的排序文件 Diff，包括六目录改动。Runtime 独立校验 Diff，
封存完整 Bundle 和六目录 Checkpoint。Optimizer 的权限与继承规则不变：实现只读，六目录可写。

`contributing_paths` 记录实际吸收内容的、排序且去重的 Workspace 相对文件或目录路径，允许
`input/agents/agent-vN/` 和 `input/evidence/agent-vN/resources/`，包括 Parent 其他 Trajectory 的资源。
仅阅读和自动继承 Parent 不算贡献。路径必须存在、无链接或越界，且属于合格已评估历史或 Parent，
不能引用同 Epoch 尚未评估的 Challenger。`reuse` 要求 `[]`。Runtime 在 Evolution Trace 中保存归属
和准确内容快照；该字段不改变 Bundle Base 或 Revision 祖先关系。

Evolver 通过本地 `evolution-report` 提交 Draft；错误返回 `issues`、`request_schema` 和 `recovery`，
不发布。首次成功原子生成 `scratch/evolution-report.json`，Session 退出后 Runtime 再独立校验。

## 外部服务 Contract

- Agate 通过发布版 `atrex-gateway-client` SDK 调用；Runtime 持有凭据和请求构造，Worker 只看到
  安全投影。
- GPU Wiki Query 为 `POST /v1/knowledge/query`；Local Wiki 实现同一 v1 Contract。
- 完整 Schema、Evidence Layout、版本和 Bundle 语义见[协议](protocols.zh.md)，所有部署字段见
  [配置说明](configuration.zh.md)。
- Evaluation、Production Gate、比较、Roofline 与 SOL 语义见[评测与晋升](evaluation.zh.md)。
