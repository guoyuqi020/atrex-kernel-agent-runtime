# 配置参考

[English](configuration.md) | 中文

Runtime 只接受严格 JSON，并拒绝未知字段。本地相对路径相对于其配置文件解析。完整语法以仓库中的
Example 为模板；本文解释权责与约束，不重复容易过期的全部数值。

## 配置文档

| 文档 | Schema | 用途 |
| --- | --- | --- |
| [`runtime.example.json`](../runtime.example.json) | Runtime v1 | 部署、服务、策略、Agent Backend、存储与 Launcher。 |
| `examples/*/campaign.json` | Campaign 配置 | 算子、私有/公开 Contract、Core Commit、DSL Lineage、Model 与 Epoch 拓扑。 |
| [`lineage-seed.example.json`](../lineage-seed.example.json) | Lineage Seed v1 | 从封存 Agent/Kernel 内容增加独立 Lineage。 |
| Ablation Arm Spec | Ablation v1 | 从源 Lineage 的 Bootstrap Baseline 创建控制 Campaign，可配置独立进化调度。 |

不存在独立 Bootstrap 配置。Manifest、Report、Trace、Evidence、Journal 与 Usage File 是 Runtime
输出，不是运维配置。

## Runtime v1

### `server`

ASGI `host` 与 `port`。在 `container` 和 `sandbox` Mode 下，
`campaign.gateway_proxy_url` 必须指向该 Socket。

### `storage`

四个互不相同的位置：Registry SQLite、Gateway Control SQLite、Agate Job SQLite 和不可变 Artifact
Root。应放在专用私有目录并一致备份。

### `administration`

可选 Admin Bearer Token 环境变量名、请求/Event 上限，以及持久 Task 的 Lease/Heartbeat/Poll
策略。Lease 必须大于 Heartbeat 的两倍。

### `gateway_proxy`

Attempt Capability 签名 Key 环境变量名、请求/Candidate 上限、逐 DSL Changed-path Allowlist，以及
是否要求 Candidate 源码变化。计划重启期间签名 Key 必须保持稳定。

### `agate`

Agate URL、认证模式（`none`、`token` 或 `ak_sk`）、凭据环境变量名、请求/等待超时与健康
检查间隔。Runtime JSON 只保存 Secret 名称，不保存值。

### `kernel_agent`

Agent Bundle 与公开 Contract 大小上限。可选 `base_source` 指定批准的 Core 仓库、Git 程序、
Fetch/Archive 上限和显式允许的 Submodule。Bootstrap 仍必须提供完整 Commit SHA；Branch/Tag
不是持久身份。

### `gpu_wiki`

可选只查询 Wiki URL、Bearer Token 环境变量名、超时与 Request/Query/Response 上限。Runtime 在
返回结果前冻结查询。不存在 Feedback、Upload 或 Outbox 配置。

### `campaign`

所有 Campaign 共用的部署策略：

- Attempt、Evolution、Problem Generalization、Bootstrap 四个独立 Worker Root；
- Lineage Fencing Lease/Heartbeat；
- Runtime Gateway URL、允许操作、调用配额与 Capability 生命周期；
- `gate_policy`、Kernel Retention Comparator 与 Agent Promotion Comparator；
- Infrastructure Retry、Bootstrap Lineage 并发与 Branch 并发；
- 可选可信 Roofline Builder；
- Evidence 投影上限；
- Optimizer/Evolver Worker 策略；
- Launcher Mode 与隔离策略。

DSL 拓扑、Model 身份和 `K/Y/X` Epoch 结构属于 Campaign 定义，不属于 Runtime 服务配置。

## Gate 与比较策略

`campaign.gate_policy` 是以下内容的可信来源：

- Optimizer Correctness Case、Benchmark Iteration 与探索 Evaluate Repeat；
- 有序 Bootstrap Stage 与 Bootstrap Benchmark Iteration；
- Retention Correctness Case 与 Benchmark Iteration；
- Production Gate 开关；
- Warmup、`atol`、`rtol`、超时与锁频；
- Commit 固定 Atrex Bench Evaluator 的导入上限。

Runtime 在封存前覆盖输入 Evaluation Contract 中 Gate 持有的字段。

`kernel_retention_comparison` 与 `agent_promotion_comparison` 分别选择：

- `method: "evaluate"`：`repeats` 与 `measurement_uncertainty_us`；
- `method: "same_allocation_abba"`：`repeats`、最小提升、Allocation Timeout、Shape Batch Size
  和最大并行 Shape Batch。

ABBA Timeout 必须容纳完整交错 Schedule。详见[评测与晋升](evaluation.zh.md)。

### Roofline Builder

可选 `roofline_builder` 固定一个 Atrex Bench 仓库与完整 Commit、Git/Python 程序、
Fetch/Execution/Output 上限，以及可选 Agate Target 到 SKU 的映射。只有 Campaign Evaluation
Contract 没有显式 Roofline 时才运行。生成 Roofline 必须精确覆盖私有 Shape ID 才会被封存。

## Worker 策略

### Optimizer

`campaign.optimizer` 选择 `claude`、`codex`、`qodercli` 或 `pi`，以及 Reasoning Effort、
不透明 Backend Settings、Command Prefix、显式环境继承、隔离 Home Key、Trace/Usage 路径、
Attempt/Bootstrap 超时、终止宽限、诊断/Report 上限和逐 Session 用量配额。

QoderCLI 使用 Provider Credit；其他受支持 Backend 使用 Provider Token Bucket。Core 必须生成
Provider 原生终态 Usage Report。Bootstrap 使用 `bootstrap_timeout_seconds`，普通 Attempt 使用
`timeout_seconds`。

### Evolver

`campaign.evolver` 还固定 Evolver 仓库与完整 Commit，以及 Import/Bundle 上限。它记录 Provider
原生用量，但没有 Token/Credit 配额；进程仍受 `timeout_seconds` 限制。

Campaign Bootstrap 会冻结 Evolver Commit；Resume、Schedule 与 Debug Shell 会拒绝选择不同 Commit
的 Runtime Config。

### Environment

`environment.values` 保存静态非 Secret 值；`inherit` 要求 Runtime 进程存在指定变量；
`inherit_optional` 只在存在时复制。未声明 Ambient Variable 不会传入 Worker。不要通过这些 Map
配置 `HOME` 等隔离 Home Key。

## Launcher

`campaign.launcher.mode` 可选：

- `development`：干净环境与轻量调试边界；只用于可信本地工作。
- `container`：专用外层 OCI Container 中直接运行 bubblewrap。
- `sandbox`：通过 systemd 运行 bubblewrap，并配置逐 Session cgroup v2。

`backend_credentials` 控制所选 CLI 登录态的只读投影。两个生产 Mode 共用文件系统设置：bwrap
程序、私有 Home/Workspace Mount、Resolver、额外只读 Bind、隐藏 Host Path 和可选只读 Reference
Project Root。Reference Project 只对 Framework Bootstrap 可见，普通 Attempt 不可见。

`sandbox.resources` 设置内存、Swap、CPU Quota 与 PID 上限。Container Mode 没有逐 Session
Resource Block，由外层容器持有总限制。两个 Mode 都保留所在 Network Namespace，因此允许公网
访问和可达 Host Service。

## Evidence

`campaign.evidence.max_trace_files` 只限制参与语义投影的 Runtime 归一化 JSONL Ledger；Provider
原生主 Session/子 Agent Transcript 作为不透明原文保留，不计入该数量。`max_trace_bytes` 限制包含
这些 Provider 文件在内的完整 Session Artifact。其余配置限制投影 Event/Text 与 Kernel Diff。
Redaction Pattern 只作用于归一化投影；原始 Session Artifact 保持不可变，可物化进授权 Agent
Evidence 或由管理端读取，但不会上传到 GPU Wiki。

## Campaign 配置

必填顶层字段：

- `creation_key`、`operator` 与 Agate Environment Selector `hardware_target`；
- 私有 `evaluation_contract`；
- 优先使用 `shape_train`，迁移期可使用 `agent_problem`；若配置 Core Problem Generalization，
  二者可以都省略；
- 完整 `base_revision.commit`；
- `challenger_count`、`challenger_start_epoch`、`trajectories_per_branch` 与
  `attempts_per_trajectory`；
- 以 DSL 为 Key 的非空 `lineages` Map。

每条 Lineage 提供可选 Optimizer/Evolver Model、`baseline_kernel` 与 `initial_evidence`。省略
Model 时委托给配置的 Backend CLI 默认值。可选 `problem_generalization_model` 只作用于 Core
Problem Generalization。

可选 `first_epoch_same_agent` 默认 false，生产进化臂启用它，并要求 `challenger_count=1`。
Epoch 1 的 Active 与副本使用同一 Agent Revision、相同 Kernel/State 起点，可写 State 相互隔离；
不调用 Evolver、不创建新 Agent Revision。之后遵循 `challenger_start_epoch`，该选项随 Lineage 冻结。

Bootstrap 会查询 Agate。返回架构（例如 `sm_120`）对 Agent 可见；Canonical GPU Alias 单独封存
用于 Agate 调度。

## Lineage Seed v1

Spec 固定 DSL 与 Epoch 拓扑，并选择一种来源：

- `source_type: "artifacts"`：Agent Artifact Digest 与 Kernel Artifact Digest；
- `source_type: "revisions"`：已注册 Agent Revision ID 与 Kernel Revision ID；
- `source_type: "lineage_baseline"`：源 Lineage 的冻结 Bootstrap Baseline（用于 Ablation）。

Runtime 在发布独立 `agent-v0`/`v0` 根之前重新校验 Agent，并在目标 Campaign 下重评 Kernel。

## Ablation Arm v1

`seed-ablation-arm --spec` 接受：

- `creation_key` 与 `source_lineage_id`；
- `attempts_per_trajectory` 和可选 `trajectories_per_branch`；
- `ephemeral_agent_state`（默认 true）；
- `challenger_count`（默认 0）和 `challenger_start_epoch`（默认 2）；
- `first_epoch_same_agent`（默认 false，要求一个 Challenger）；
- 可选 `optimizer_model`。

它创建一个独立的单 Lineage Campaign，默认不生成 Challenger。当
`ephemeral_agent_state=true` 时，每个 Attempt 的 `prompts/`、`memory/`、`knowledge/`、`skills/`、`tools/` 与 `hooks/` 恢复到固定 Core Revision 的初始内容；
设为 false 时保留源 Bootstrap Deposit 和后续串行 State。对比进化频率时使用
`challenger_count=1`、`challenger_start_epoch=2`、`first_epoch_same_agent=true`、
`ephemeral_agent_state=false`，调整
`attempts_per_trajectory`。Evolver 模型及 Commit 继承自源；Optimizer 模型也默认继承，
可显式覆盖。Baseline Outcome 直接复用，不重复测量。

## 校验规则

- Secret 名称必须是合法环境变量标识符，值保持在 JSON 外。
- Git 生产身份必须是小写完整 Commit SHA。
- Agent Protocol 内路径必须是安全相对 POSIX Path；部署路径相对于 Config 解析。
- Workspace Root 与 Storage Location 必须互不相同。
- Production Launcher 只读 Bind 不能暴露 Runtime Storage 或兄弟 Worker Root。
- 相同 Creation Key 下不能修改不可变 Campaign 输入。
