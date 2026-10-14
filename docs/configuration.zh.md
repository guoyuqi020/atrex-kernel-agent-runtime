# 配置参考

[English](configuration.md) | 中文

Runtime 只接受严格 JSON，并拒绝未知字段。本地相对路径相对于其配置文件解析。完整语法以仓库中的
Example 为模板；本文解释权责与约束，不重复容易过期的全部数值。

## 配置文档

| 文档 | Schema | 用途 |
| --- | --- | --- |
| [`runtime.example.json`](../runtime.example.json) | Runtime v1 | 部署、服务、策略、Agent Backend、存储与 Launcher。 |
| `examples/*/campaign.json` | Campaign 配置 | 算子、私有/公开 Contract、Agent Commit、DSL Lineage、Model、Workflow 命令与 Runtime 资源包络。 |
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

Runtime 使用有界的 `get_job(wait=False)` 查询与异步间隔等待 Agate Job。状态不变时按
1/2/4/8/10 秒逐步放缓，状态变化后重新从 1 秒开始。`wait_timeout_s` 限制单次状态查询，
最多 30 秒，不会结束整个工具调用。Evaluate、ABBA、Profile、Check、Disassemble、Dev、
Bootstrap 与权威终评仍等到终态才发布结果。网络重试保留 5/10/20/40/60 秒策略、尊重
`Retry-After`，退避使用异步等待；保留提交幂等键、Job ID 和已经成功的 OSS 上传阶段。

SDK 调用按事件循环使用独立并发额度：普通请求 32、OSS 操作 8、环境信息 4、取消／健康
检查 4、本地打包 4，不占用 AnyIO 默认的 40 个工作槽位。轮询间隔与 Runtime 重试退避
不占线程。Agate 安装包保持不变；Runtime 本地客户端子类关闭 SDK 内部的提交过载睡眠，
同时通过公开参数 `transport_retries=0` 关闭网络重试睡眠。单次有界 SDK 调用仍在线程中
执行；取消时等待在途调用返回再清理文件，之后不再查询／重试。同步部署兼容入口仍保留，
服务和评测路径使用异步边界。

### `kernel_agent`

Agent Bundle 与公开 Contract 大小上限。可选 `base_source` 指定已初始化的本地 Optimizer
Checkout、Git 程序、命令/Archive 上限和显式允许的 Submodule 来源身份。Bootstrap 必须提供完整
Commit SHA，并直接封存对应本地 Commit，不执行 Fetch 或其他网络访问；Branch/Tag 不是持久身份。

### `gpu_wiki`

可选只查询 Wiki URL、Bearer Token 环境变量名、超时与 Request/Query/Response 上限。Runtime 在
返回结果前冻结查询。不存在 Feedback、Upload 或 Outbox 配置。

`enabled` 默认为 `false`。设置 `gpu_wiki.enabled: true` 后，Bootstrap、Optimizer（含开发
Shell）才会看到 `wiki-query` 工具与对应说明、获得 Attempt 范围查询权限；Runtime 同时启用
`/v1/wiki/query`。它独立于 `optimizer.tool_modules` 的 Direction／Experiment 开关。
`gpu_wiki` 缺省/null、未填 `enabled` 或设为 `false` 时，均不加载 Wiki 工具/说明、不发查询
权限，API 入口不可用，也不要求 Wiki 凭据。切换时保留 URL 和限制字段，无需再配置一个
Gateway operation。Evolver 能看到后续 Optimizer 的工具可用性，但自身不会获得 Wiki 权限。

例如在 [runtime.example.json](../runtime.example.json) 的 `gpu_wiki` 中把 `"enabled": false`
改为 `true`，其他字段保留。修改后重启 Runtime 和 campaign worker，新建工作区获得新的
不可变工具契约；恢复已有工作区仍会复用原契约，直到创建新工作区。
已冻结的 Agent 版本须包含本次新增工具实现；切换部署配置不会更新固定的 Agent
源码或已经运行的会话。关闭开关并重启服务后，旧 Wiki capability 也无法通过该 API 查询。
知识服务是独立进程；内部知识源使用 `local-wiki/configs/internal.example.json` 选择。

Agent 只填写 `{"query":"一个具体的硬件或优化问题"}`。工具负责 Attempt 身份、幂等键和认证，
Runtime 补充可信算子/DSL/硬件上下文并冻结返回值。Wiki 历史结论只作为有适用范围的参考，
不能替代本次 kernel 的精度和性能评测。

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

`campaign.optimizer.output_limit_recovery_retries` 控制已分类输出上限失败后的 Core/KDA 新对话恢复。
默认 `2`，允许整数 `0..10`，与仅补交报告的 `report_completion_retries` 配额独立。
恢复保留 Attempt 工作区以及剩余时间/用量预算。
详见[输出上限恢复](interfaces.zh.md#输出上限失败后的恢复)。


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

Runtime 绑定的 Core/KDA Bootstrap 和 Optimizer Session 将 Runtime CLI 的 HTTP 请求交给
独立执行器：发送请求前暂停 Provider 所属进程组（包括 subagent），所有请求及 Runtime 记录的
在途请求结束后才恢复。Shell `timeout`、后台执行或 CLI 被杀都不能解除等待。Runtime 和 HTTP
worker 保持运行；显式取消 Session 或触发策略终止时，仍会停止 Agent 并清理 HTTP worker。

Claude 的 `PreToolUse` hook 同时保持 Runtime CLI Bash 调用前台运行，等待额度至少为 72 小时。
没有 Runtime 请求时，其他 Bash 调用保持原行为。HTTP 读取超时使用同一请求身份重连；主动杀掉
CLI 可能丢失其输出，请求结束后用相同内容重试仍遵循原有恢复／去重契约。
部署时必须刷新冻结的 Core/KDA Bundle，仅更新服务无法启用这个等待屏障。

`campaign.optimizer.tool_modules` 可独立启用 `directions` 和 `experiments` 两组 Runtime
工具。默认是 `["directions", "experiments"]`；也可以只填其中一项，或填 `[]` 全部关闭。
该配置同时适用于 Bootstrap 和优化 Attempt，Session 实时契约仅列出已启用的命令。仅启用
Directions 时，关闭 Direction 不要求 Experiment 支持 ID；仅启用 Experiments 时，
`record-experiment` 不填写 `direction_id`。
Session 提示词也随此配置调整：保留通用的工作区和评测说明，只展示已启用 Journal 工具的
操作要求。默认配置仍使用原始提示词；已启动的 Session 沿用启动时的契约与提示词。
关闭 Experiments 后，仍可通过 `candidate_ready` 提名可见历史中的原样 Kernel：Runtime 自动
核验并复用其匹配的成功普通完整 Evaluate 证据。算子、硬件、DSL、封存 Contract 和精确 Kernel
内容必须一致；自定义输入、仅正确性检查、Profile 和探索性 ABBA 不符合条件。无需记录 Experiment
或重复 Evaluate，原始测量身份保持不变。本 Attempt 新的完整 Evaluate 已失败时，不能用历史成功
覆盖。启用 Experiments 时，历史复用仍需显式调用 `record-experiment` 并设置 `action="adopt"`。

`LineageSeedSpecV1`／`AblationArmSpecV1` 可设置 `result_artifact_read_enabled: false`
（默认 `true`），并在 Lineage 创建时冻结。它从 Optimizer 契约和提示词移除
`result-artifact-read`，服务端也拒绝直接 HTTP 读取。配合 `tool_modules: []` 即为
`no-result-read` 消融配置；旧 Kernel Trial 查询工具原本已不可用。测量仍随原工具调用返回，
并在 Runtime 中保留供人类分析。历史报告、会话、源码读取、帕累托边界、其他已启用查询及
Agent 自己的笔记仍可用。同一幂等键的传输重放不变；新请求重复提交已完成任务时直接返回可见的缓存结果，不新增 GPU 调用或测量记录。
不可见同伴仍隔离并独立测量。开启结果读取的臂保留原来的重复错误；运行中的重复请求继续等待。Bootstrap 与旧 Lineage 保持默认行为。

`LineageSeedSpecV1`／`AblationArmSpecV1` 可设置冻结的 `dev_only: true`，同时必须设置
`tool_modules: []` 和 `result_artifact_read_enabled: false`。Agent 的 Gateway 操作仅允许
`dev`；源码回读遵守原有历史可见性规则，结果回读、Pareto、Journal 和 Wiki 查询关闭，
直接 HTTP 请求同样受限。可见 CLI 保留 `gateway-execute`（Dev）、`kernel-artifact-read`、
`attempt-report`、本地 `runtime-contract`；
内部报告回执恢复仍保留。Agent 可通过自写 Dev 脚本编译、检查正确性、计时和诊断环境，
但脚本输出不会直接成为权威评测结果。

该臂允许未调用 Agent Evaluate 就提交 `candidate_ready`。Runtime 在报告提交后始终先运行
独立的私有正式验证，再走原有采纳流程；不能把任意 Dev stdout 作为通过或加速的依据。
Bootstrap、报告结构、源码限制、预算和文件历史可见性保持原有规则；历史报告／会话及自身
笔记仍可用，因此不是“无记忆”消融。`dev-only` 是第六种配置：默认共享方案每种共享模式
七臂，共十四臂；FA4/L20D、GDN/L20D、Qwen3.5 FA/L20N 的独立方案为七种配置各三次重复，
共二十一条 Lineage。


`raw: true` 是第七种配置，要求 `dev_only: false`、`tool_modules: []` 和
`result_artifact_read_enabled: false`。不开放任何 Runtime 命令，包括源码回读、契约查询、
报告提交；服务端也拒绝 Raw Attempt 的所有 Runtime 操作。仅 Raw 会话获得部署配置对应的
Agate 地址和认证环境变量（`AGATE_URL`，及 `AGATE_TOKEN` 或 `AGATE_AK`／`AGATE_SK`）。
使用 `"$ATREX_AGATE_PYTHON" -m atrex_gateway_client.cli --help` 或 SDK 直接操作；执行环境
必须安装 Agate client。上传、轮询、重试、正确性验证和计时都由 Agent 自己管理；Runtime
调用次数限制、去重、工具等待扣时不适用于这些直连调用，会话 token 和经过时间预算仍生效。
不使用 Runtime 的补报告、输出上限恢复提示或自动历史总结。

不提供 Bootstrap 报告、历史对话、Epoch 摘要、Journal／Kernel Trail、同伴笔记或源码／结果
查询接口。Agent 可自行把笔记、源码副本、脚本和响应保存在 `tools/`，格式不限；只有本轨迹
自己写下的这些文件延续至后续会话，不继承胜出轨迹的笔记。scratch 为临时目录。下一次
`work/kernel` 仍遵守原 Workflow 的 kernel 选择规则；共享模式不会给 Raw 开启经验共享。
正常退出后，控制器封存最终源码并按原有规则私下评测，只生成内部交接收据，不伪造 Agent
结论。内部审计会话、token、评分仍供人类分析，但不再挂载为 Raw Agent 历史。
Bootstrap 本身的行为不变。凭据只注入进程环境，不写入契约、Prompt 或复用文件。

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
- 资源上限 `max_challengers` 与 `optimizer_attempt_budget`；
- 以 DSL 为 Key 的非空 `lineages` Map。

每条 Lineage 提供可选 Optimizer/Evolver Model 与 `initial_evidence`，并选择 `baseline_kernel`
或多文件源码模式的 `source_manifest` + `source_repository`。详见[源码树](source-trees.zh.md)。省略
Model 时委托给配置的 Backend CLI 默认值。可选 `problem_generalization_model` 只作用于 Core
Problem Generalization。

这两个值只是资源包络，不定义 Epoch 拓扑。版本化的 `workflow/main.py` 决定何时复制或进化
Agent、运行哪些 Branch/Trajectory、如何分配 Attempt 预算，以及下一轮使用哪个 Kernel 和自适应
State。Workflow 超出 Challenger 上限或未精确分配 Attempt 预算时，Runtime 会拒绝执行。

部署项 `campaign.max_parallel_attempts` 独立限制一个 Workflow 并行批次中同时准入的 Optimizer
Session 数量。它只是并发上限，不是拓扑定义。

Bootstrap 会查询 Agate。返回架构（例如 `sm_120`）对 Agent 可见；Canonical GPU Alias 单独封存
用于 Agate 调度。

## Lineage Seed v1

Spec 固定 DSL 与同一资源包络，并选择一种来源：

- `source_type: "artifacts"`：Agent Artifact Digest 与 Kernel Artifact Digest；
- `source_type: "revisions"`：已注册 Agent Revision ID 与 Kernel Revision ID；
- `source_type: "lineage_baseline"`：源 Lineage 的冻结 Bootstrap Baseline（用于 Ablation）。

Runtime 在发布独立 `agent-v0`/`v0` 根之前重新校验 Agent，并在目标 Campaign 下重评 Kernel。

## Ablation Arm v1

`seed-ablation-arm --spec` 接受：

- `creation_key` 与 `source_lineage_id`；
- `optimizer_attempt_budget` 与可选 `max_challengers`（默认 0）；
- `workflow_command`，用于选择完整的初始可执行组织方式；
- 可选 `optimizer_model`。

它创建一个独立的单 Lineage Campaign，默认不允许 Challenger。被选择的 Workflow 程序自行决定
Branch/Trajectory 组织、进化时机、Kernel 路由，以及是否复用已完成 Attempt 的输出 State；
Runtime 不再解释 reset/retain、首轮复制或进化频率配置。Evolver 模型及 Commit 继承自源；
Optimizer 模型也默认继承，可显式覆盖。Baseline Outcome 直接复用，不重复测量。

## 校验规则

- Secret 名称必须是合法环境变量标识符，值保持在 JSON 外。
- Git 生产身份必须是小写完整 Commit SHA。
- Agent Protocol 内路径必须是安全相对 POSIX Path；部署路径相对于 Config 解析。
- Workspace Root 与 Storage Location 必须互不相同。
- Production Launcher 只读 Bind 不能暴露 Runtime Storage 或兄弟 Worker Root。
- 相同 Creation Key 下不能修改不可变 Campaign 输入。
