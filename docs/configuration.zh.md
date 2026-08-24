# 配置参考

[English](configuration.md) | 中文

仓库暴露两类需要用户编写的控制文档：

- [`runtime.example.json`](../runtime.example.json) 是部署模板。Runtime JSON 使用
  `schema_version: 1`、拒绝未知字段，并以配置文件目录解析本地相对路径。使用时将它复制为
  已被 Git 忽略的 `runtime.json`，再按部署环境调整。
- 每个可运行工作流持有自己的 `examples/<workflow>/campaign.json`，例如
  [`examples/bootstrap/campaign.json`](../examples/bootstrap/campaign.json)。Campaign JSON 使用
  `schema_version: 3`，直接作为 `bootstrap --campaign` 的输入。

Bootstrap 没有独立配置文件；它创建或恢复 Campaign JSON 描述的不可变 Campaign。
Evaluation Contract、Agent Problem、Seed Kernel 和 Initial Evidence 是 Campaign 引用的输入资产，
不是额外的控制层配置。Runtime 生成的 Result、Manifest、Evidence Metadata、Trace 与 Token Usage
Report 都是输出协议，因此不提供容易被误认为配置的根目录 `*.example.json` 模板。

## 顶层配置组

- `server`：ASGI Host 与 Port。
- `storage`：互不相同的 Registry、Gateway Control、Agate Job SQLite 和 Artifact Root。
- `gateway_proxy`：请求/Candidate 限制、签名密钥环境变量名、各 DSL 变更路径白名单和 No-op 策略。
- `agate`：URL、认证模式、凭据环境变量名、HTTP/等待超时。
- `kernel_agent`：`max_bundle_files`、`max_bundle_bytes`、`max_entrypoint_bytes`、`max_agent_problem_bytes` 与可选 `base_source`。
- `gpu_wiki`：外部 URL、可选 Bearer Token 环境变量名，以及 Proxy/Query/Response 限制。
- `campaign`：Worker 组合与调度策略。
- `administration`：Bearer Token 环境变量名和请求/Event 限制。
- `maintenance`：离线 Artifact/Workspace 保留限制。

`kernel_agent.base_source` 声明批准的 Core 仓库、Git 可执行文件、Fetch/Archive 限制和允许 Submodule 的精确 Path-to-URL Map。Bootstrap 请求提供完整 Commit。本地 Checkout 只用于开发；Runtime 仍会导入并封存 Tracked Tree。

## Campaign 配置

Campaign 运行时分别配置 Attempt、Evolution、Problem Generalization 与 Lineage Baseline Workspace Root。Fencing 必须满足 `lease_seconds > 2 * heartbeat_seconds`。部署配置不选择 DSL；该拓扑由独立的 Campaign 定义持有。Gateway Operation 必须唯一且包含 `evaluate`，Capability 调用配额和生命周期显式配置。`dev` 与 `wiki_query` 会被完整审计但不消耗调用配额；Benchmark、编译、Profile 及其他操作仍计数。

`gate_policy` 是 Correctness 与 Performance 语义的唯一可信来源。仓库内配置与 Atrex Kernel
Agent 一致：Optimizer 探索使用 1 个 Correctness Case、1 次 Eval 和 100 ms eager 测量预算；
留存评测使用 1 个 Case 与 100 ms；Bootstrap 顺序执行 `(1 case, 5 ms)` 和
`(5 cases, 5 ms)` 两个阶段，必须全部通过，并以第二阶段延迟作为 Bootstrap `v0` 延迟。
`warmup_iters` 为 10，容差为 `0.01/0.05`，Candidate/Performance Timeout 为 `20/120` 秒，
Evaluation Job Budget 为 600 秒。

`gate_policy.production_gate` 用于开启可信的内容级 Production Gate。其 Model 默认值为关闭；
仓库内所有发布 Example 均显式开启。开启后，Candidate Source 必须是合法 Python，并包含当前
Lineage 固定 DSL 的自实现标记；Runtime 会在提交 Agate 前拒绝混用其他 DSL、PyTorch Compute
Fallback、`torch.ops`、动态外部代码加载、预构建算子库和未批准 Import。若存在
`solution.json`，它必须声明固定 DSL 且只能包含批准的依赖。无法机械确认的第三方实现依赖会
fail closed，不交给可进化 Agent 判断。权威终评与 Artifact Seed Lineage 还会再次执行同一策略。

Bootstrap 会在封存前覆盖输入 Evaluation Contract 的所有 Gate 字段：Sampling、容差、Timeout、
完整 Validation Mode、时钟 Policy、标准 Atrex Bench 版本和 Gate-owned Runner Override。输入
Contract 无法反向覆盖这些策略。已有 Campaign 保留已封存 Policy；修改 `gate_policy` 需要新的
Campaign Identity。每个优化 Attempt 由 `kernel_retention_comparison` 终结：普通 Evaluate 按
自身 `repeats` 分别测量 A/B，同 Allocation ABBA 执行交错 Schedule；B 聚合结果直接成为
Candidate Kernel 的权威 Evaluation，不再有单独的 Attempt 终评。

可选 `roofline_builder` 声明批准的 Atrex Bench Repository 与完整 Commit、绝对 Git/Python
可执行文件、Fetch/Execution/Output 限制，以及可选的准确 `sku_by_hardware_target` Map。
Evaluation Contract 没有 Roofline 时，Bootstrap 会在任何 Agent Session 前执行该 Commit 的标准
Converter，并封存校验后的结果。无法生成有效 Roofline 时，Runtime 会报告原因，并在每个正确的
Eval 后回退执行一次 NCU SOL Profile。参见[可信 Roofline 构建](roofline-builder.zh.md)。

Kernel 保留与 Agent 晋升可以分别选择普通 Evaluate：

```json
{"method":"evaluate","repeats":1,"measurement_uncertainty_us":0.0}
```

Runtime 会为 A、B 分别并发提交 `repeats` 次独立普通 Eval，要求全部 Run 通过，比较两边的算术
平均值，并把 B 的聚合延迟和聚合 Result Artifact 写入 Candidate Kernel Revision。

每次这类 Measurement，以及 Optimizer、Bootstrap 和 Lineage Seed 的 Evaluate，都使用同一套
固定可信执行器：每个 Agate Job 包含 4 个 Shape，同时最多运行 4 个 Shape Batch Job。这两个
限制是 Runtime 不变量，不是 Campaign 输入项。

也可以选择严格的同 Allocation ABBA：

```json
{
  "method": "same_allocation_abba",
  "repeats": 2,
  "minimum_improvement_percent": 0.0,
  "allocation_timeout_seconds": 600,
  "shape_batch_size": 1,
  "max_parallel_shape_batches": 4
}
```

`gate_policy.evaluator` 为普通 Eval 与 ABBA 提供同一个完整 Atrex Bench Commit：

```json
{
  "repository": "./third_party/atrex-bench",
  "commit": "FULL_LOWERCASE_COMMIT_SHA",
  "git_executable": "/usr/bin/git",
  "fetch_timeout_seconds": 120,
  "max_archive_bytes": 8388608,
  "max_bundle_files": 128,
  "max_bundle_bytes": 4194304
}
```

仓库内模板指向按 Commit 固定的 `third_party/atrex-bench` Git Submodule。该配置继续用完整
Commit 校验 Atrex Bench，并限制 Git Import 与上传的 evaluator-only Bundle。
每个 Shape Batch 只提交一个 Agate `dev`
Job，完整交错 Schedule 在其中执行。所有 A/B Run 都必须通过；Runtime 对各 Repeat 取几何平均，
并要求提升严格大于 `minimum_improvement_percent`。参见[性能门禁](performance-gates.zh.md)。

`optimizer` 配置 Runtime 权威的 `agent_backend`、`reasoning_effort` 与
`session_settings` Binding，以及 Core 命令前缀、显式/继承环境、隔离 Home Key、
Provider Usage Report 路径、Report/Diagnostic 限制、超时/Grace、`max_session_tokens` 与
`max_session_credits`。QoderCLI 选择 Credit 配额并记录 Provider 原生 Credit；Claude、Codex
与 Pi 选择 Token 配额并记录互斥的 Provider Token Bucket。
Backend 可选 `claude`、`codex`、`qodercli` 或 `pi`。该部署 Binding 作用于 Core Session，具体
`timeout_seconds` 限制普通 Optimizer Attempt 与 Problem Generalization；
`bootstrap_timeout_seconds` 单独限制每个 Lineage Bootstrap Session，默认 10800 秒（180 分钟）。
`evolver.timeout_seconds` 单独限制一次 Challenger 构建 Session；仓库模板和生产策略统一设置为
10800 秒（3 小时），终止 Grace 为 10 秒。
Model 则由 Campaign 独立选择。Core 仍持有 Prompt、
Tool、Workflow 与 Adapter 实现；`atrex-agent.json` 只提供独立运行时的默认值。

`launcher.backend_credentials` 控制宿主 CLI 登录态复用，默认开启。Runtime 只针对当前选择的
Backend，从显式 `host_home`（未配置时为 Runtime 进程的 `HOME`）发现并把 Claude 的
`.claude`/`.claude.json`、Codex 的 `.codex`、QoderCLI 的 `.qoder`/`.qodersec` 或 Pi 的
`.pi/agent` 直接投影到该 Session 的隔离 Home。凭据挂载只读且只在进程期间存在，凭据不会复制进
Workspace Artifact。可变 Provider 状态会复制或创建在私有 Session Home 中；其中 Codex
`config.toml` 可写，使 TUI 能记录工作区 Trust 而不修改宿主配置。CLI Cache 与 Session 输出仍写入 Session 自己的可写路径。生产私有 Home
遮蔽的用户级 CLI 安装根也会被只读恢复。`development_bwrap_executable` 默认为
`/usr/bin/bwrap`。
生产 Optimizer/Evolver Session 仍严格只暴露所选 Backend。交互式 Optimizer/Evolver Dev
Shell 是明确例外：它会同时只读投影所有可用 Backend 的登录态，为每个 CLI 在 Dev Shell
Session Home 下提供私有可写状态，并在同一 Shell 中暴露四种 CLI，便于人工比较和排障。

`evolver` 使用独立的四选一 Backend Binding，并配置仓库、完整 Commit、Git/Import 限制、
解释器命令前缀、Bundle/Output 限制、显式/继承环境、隔离 Home Key、Trace/Token Report、
超时/Grace 与 Diagnostic。它没有 Token 配额，但四种 Backend 都必须提交完整 Provider Usage。
Runtime 在首次请求 Challenger 时才解析该环境并导入固定 Bundle。因此，当 Epoch 拓扑为
`challenger_count=0` 时，既不要求 Evolver 凭据，也不会访问 Evolver 仓库。

`bootstrap_max_parallel_lineages` 是正数 Campaign Bootstrap 并发上限，默认值为 `1`，以保持
原有串行行为。Bootstrap 会先封存共享 Evaluation Contract、Agent Problem、Core Commit 与
Campaign 身份，再在该上限内并行生成各 DSL 的 `v0`；返回顺序仍固定为 CUDA、Triton、CuteDSL。
生产环境需要三条 DSL 同时 Bootstrap 时可配置为 `3`。

`max_parallel_branches` 是正数 Runtime 调度上限，默认值为 `4`。Evolver 调用仍然串行，使后续
提案可以检查此前的 Challenger 设计。Challenger Pool 冻结后，Active 与 Challenger Branch 在该
上限内并发运行；每个获准运行的 Branch 还可以并发执行其全部 Trajectory，而每条 Trajectory 内的
Attempt 仍保持串行。一个 Branch 的失败会独立保留，不会取消兄弟 Branch 任务；Runtime 等待兄弟
任务清理完成后再向上抛出失败。

`launcher.mode` 为必填项。两种模式都会阻止私有评测文件/路径进入 Optimizer、Baseline、Evolver
Workspace 与环境，并只返回面向 Agent 的安全 Gateway Result 投影（独立 Problem Generalization
阶段会有意接收私有输入）。`development` 保留用于可信本地调试的显式干净环境；发现宿主 CLI
登录态时，Linux bubblewrap 会创建轻量 Mount Namespace，使登录态与宿主根只读，同时保持当前
Workspace 与私有 `/tmp` 可写。它仍不提供 cgroup、Network、PID、IPC 或 Runtime Storage
隔离，无法约束恶意同用户进程，也绝不会被当作失败降级路径。在没有 bubblewrap 的平台上使用
development 时，必须显式关闭 `backend_credentials` 并通过白名单环境变量鉴权。
`container` 在运维方提供的外层 OCI 容器中通过 bubblewrap 运行每个 Worker。它要求 Linux 与
bubblewrap，但不需要 systemd、可写 cgroup 层级、sudo 或逐 Session cgroup；外层容器必须允许
bwrap 使用 User/PID/IPC/UTS Namespace。Runtime 会应用与 `sandbox` 相同的只读根、私有
`~/workspace`、兄弟 Workspace/Runtime Storage 遮蔽、Capability 丢弃和 Backend 登录态只读投影。
因此 Worker 之间具有文件系统与 Namespace 隔离，但共享外层容器的内存、CPU 与 PID 总限额。
为兼容常见 OCI seccomp 策略，`container` 会只读投影外层容器的 `/proc`，而不是新挂载 procfs；
宿主 `sandbox` 仍使用私有 procfs。必须使用专用容器，不要挂载 Docker Socket、Runtime Secret、
私有评测数据或无关宿主路径，并在 OCI 层设置资源限额。

`sandbox` 要求 Linux、bubblewrap 与启用 cgroup v2 的 systemd。固定边界如下：

- 宿主根只读，并使用私有 `/home`、`/tmp`、`/run`、`/dev` 和 `/proc`；
- Runtime Artifact/数据库 Storage、四类配置的 Workspace Root 与 `hidden_host_paths` 全部被遮蔽；
- 只有当前 Session 读写挂载到 `workspace_mount`，该路径必须位于 `sandbox_home` 下并表现为
  `~/workspace`；
- `read_only_bind_paths` 只用于显式批准的不可变依赖或 Provider 配置，且不能位于任何 Worker
  Root 或 Runtime Storage Root 下；
- `reference_projects_root` 在同样的边界约束下只读挂载到 Attempt Workspace 的 `reference/`
  目录；参考项目是按需检出的 Submodule，因此 Host Check 会拒绝缺失、符号链接或空的目录树；
- 每个 Session 独立限制 `memory_max_bytes`、`memory_swap_max_bytes`、
  `cpu_quota_percent` 与 `tasks_max`；
- Worker 共享宿主机 Network Namespace，包括宿主 DNS、路由、公网出站和可达的宿主服务；
  `resolv_conf` 会只读挂载到 Worker 的私有 `/run`。

cgroup 由 system manager transient service 创建，不使用 user-manager scope。因此
`systemd_user` 固定为 `false`，`worker_user` 必须指向已预置的非 root 账号。可信
Runtime Launcher 只需要受控的 transient service 创建权限；systemd 会在执行 bwrap 前降权到
`worker_user`。Host Check 也会要求 systemd 直接以 `worker_user` 创建四类 Worker Root 与
Probe；Lima virtiofs 等文件系统可能让成功的 `chown` 不改变 Owner，因此不能依赖 root 创建后
移交。错误 Owner 的空 Root 可以重建，非空目录则 Fail Closed。

Runtime 不提供模型代理，也不维护模型 Host 白名单。Claude、Codex、QoderCLI 与 Pi 在 Worker
中通过宿主网络使用各自原生直连行为。Provider 鉴权仍来自所选 Backend 显式白名单化的 Worker
环境或只读配置。该策略允许任意出站目的地，也不隔离宿主服务或 Worker 间流量；文件系统和进程/
资源隔离仍然生效。

`evidence` 限制 Session 规范化摘要和 Kernel Diff。脱敏规则只作用于规范化摘要；原始 Session Artifact 不会被修改，可按 Digest 为 Agent 物化。

## Campaign 定义

Bootstrap 操作只接受 Campaign schema v3。必须提供 Campaign 身份、Evaluation Contract、
`base_revision.commit`、`challenger_count`、`challenger_start_epoch`、
`trajectories_per_branch`、`attempts_per_trajectory`，以及每个 DSL 的 Seed Kernel/Initial
Evidence。`lineages` 的 Key 就是完整的初始 Bootstrap DSL 集合，不再存在独立的 `dsls` 字段或
部署默认值；之后仍可从已封存 Artifact/Revision 根增加 Lineage。
每个 `lineages.<dsl>.models` 可以分别指定可选的 `optimizer` 与 `evolver` Model；省略或设为
`null` 时使用所选 Backend CLI 的默认 Model。Optimizer Model 用于该 Lineage 的 Framework
Baseline 和全部 Attempt，Evolver Model 用于构建 Challenger。仅当省略 `agent_problem` 时，
顶层可选 `problem_generalization_model` 才用于问题泛化。这些选择会持久化到 Campaign/Lineage，
恢复已有 `creation_key` 时不允许静默改动。
`challenger_count` 可以为零，其余拓扑值必须为正数。在
`challenger_start_epoch` 之前的 Epoch 只运行 Active；默认值 `1` 保持立即进化的原有行为。
可运行工作流各自持有具体的 `examples/<workflow>/campaign.json`，并可引用
`examples/shared` 下的公共不可变输入，包括公开的 `agent_problem`；只有配置了 Core 问题泛化时
才可省略该字段。
本地 `kernel_agent`、预生成 Baseline Gateway Result 和 Baseline Latency 都会被拒绝。

凭据值不得写入 JSON。`inherit` 中缺少任一变量会让组合失败；`inherit_optional` 只转发当前存在的白名单变量，适用于 Claude/Codex 这类互斥凭据集合。Capability 签名密钥必须为 Base64，解码后至少 32 字节。

部署配置选择 Backend，而每个 Lineage 独立选择 Model，例如：

```json
{
  "problem_generalization_model": "generalization-model",
  "lineages": {
    "triton": {
      "models": {
        "optimizer": "optimizer-model",
        "evolver": "evolver-model"
      }
    }
  }
}
```

Backend、凭据、可执行文件发现、Reasoning Effort 与不透明 `session_settings` 仍属于 Runtime JSON
的部署策略。Campaign 已指定 Model 时，不要再在 Codex 或 Pi 的 `session_settings` 中指定 Model；
Adapter 会拒绝相互冲突的来源。

独立的 [`lineage-seed.example.json`](../lineage-seed.example.json) 用于在已有 Active Campaign
下配置额外 Lineage。它不包含 Commit，而是选择 Runtime 已经封存的 Agent/Kernel 内容。
`source_type` 可以是 `revisions` 或 `artifacts`；后者使用两个 `sha256:` Digest。新 Lineage
仍继承目标 Campaign 的 Operator、Hardware Target、Evaluation Contract 与 Agent Problem。

`campaign.evolver.commit` 在 Campaign 首次 Bootstrap 前只是部署输入。Bootstrap 会把完整 SHA
复制进 Campaign 记录；以后每次恢复都会校验配置值与冻结值，所以修改 Runtime JSON 只影响新
Campaign。

`session_settings` 只交给所选 Backend Adapter。凭据与 CLI 位置仍由部署环境和 `PATH` 管理，
不得放入该字符串。
