# 部署与运维

[English](operations.md) | 中文

## 进程拓扑

Agate 兼容 Gateway 与生产 GPU Wiki 作为外部服务运行。部署一个 Runtime Service 和一个或多个独立 Campaign Task/Scheduler 进程。SQLite 只支持单节点部署。Runtime 数据库使用 rollback journal，因此 Runtime 与 CLI 跨进程访问 Lima `virtiofs` 工作区也是安全的；不要使用无法可靠提供 POSIX 文件锁的远端文件系统。

Core 与 Evolver 是部署批准、按完整 Commit 固定的 Git 仓库；相邻 Submodule 只是开发 Checkout，不是 Runtime Import。`third_party/atrex-bench` Submodule 是 ABBA 与可选 Roofline 构建使用的本地可用、按 Commit 固定的可信 Evaluator Source。仓库验证前执行 `git submodule update --init --recursive` 初始化三者。

## 启动

1. 创建受限 Owner 的 Storage/Workspace 目录。
2. 选择 Worker 边界。`sandbox` 需要安装 bubblewrap、启用 cgroup v2、预置非 root
   `launcher.sandbox.worker_user`，并授予可信 Launcher 受控的 system manager transient-service
   权限，使用 `sudo .venv/bin/python scripts/validate-linux-sandbox.py` 验证准确镜像。
   `container` 则要求把完整部署放入安装 bubblewrap 且设置了内存/CPU/PID 限额的专用 OCI
   容器；容器内不需要 systemd、可写 cgroup 层级、sudo 或逐 Session cgroup，但 OCI 安全策略必须
   允许 bwrap 使用所需 Namespace。
3. 导出解码后至少 32 字节的 Base64 Capability Signing Key、Admin Bearer Token、所选 Agent
   Provider 凭据、Agate 凭据和可选 Wiki 凭据。Provider CLI 直接通过宿主网络连接，不需要
   Runtime 模型代理。
4. `sandbox` 只把最小不可变 Provider 路径加入 `launcher.sandbox.read_only_bind_paths`；
   `container` 只把必需登录状态挂入外层容器，Runtime 会把白名单子集只读投影到各 bwrap Session Home。
   禁止暴露 Docker Socket、私有评测数据或无关宿主存储。
5. 启动 Service 并检查 `/readyz`，完成配置验证。
   ASGI 启动后，Runtime 会立即探测一次 Agate，之后每隔
   `agate.health_check_interval_s` 秒探测一次（默认 30 秒）；首次状态、故障与恢复均写入 Service
   日志。该观测不改变 `/healthz` 或 `/readyz` 语义，外部服务短暂故障不会停止可信控制面。
   所有执行任务的 Agate SDK 请求使用同一套持久瞬时故障策略：分别退避 5、10、20、40 秒；第 5 次连续
   失败后不再放弃，而是每隔 60 秒持续重试，直至成功。一次成功会让下一次请求重新计数。
   不可重试的 4xx 校验或鉴权错误仍立即返回，因为必须修改请求才能成功。编译失败、正确性失败等
   Job 终态属于结果，不会被该策略重新提交。周期性健康观测仍是有界单次探针，不持有 Campaign 工作。
6. 执行 Campaign schema-v3 Bootstrap 时保持 Runtime、Gateway 和可选 Wiki 可用；Core Baseline
   Session 会回调 Runtime。
7. 使用绝对目标 Epoch 调度 Campaign。GPU Wiki 只需运行查询服务；Runtime 没有 Feedback Drainer
   或交付进程。

示例命令见仓库 [README](../README.zh.md)。配置只保存凭据环境变量名。

## Worker Workspace 所有权

Sandbox Workspace Root 必须由 `launcher.sandbox.worker_user` 实际拥有。Runtime 的 Host Check
通过 systemd 直接以该用户创建 Attempt、Evolution、Problem Generalization、Lineage Bootstrap
四类 Root 和 Probe；三个 DSL 同时 Bootstrap 时使用跨进程锁串行完成这一步。可信 Scheduler
仍可组装 Workspace，但在 bwrap 启动前会把当前 Run 交给 Worker。

Lima `virtiofs` 上的 `chown` 可能静默不生效，因此不要以 root 预建 `*-workspaces`。空的错误 Owner
目录会被安全重建；真实的非空 Owner 不匹配会 Fail Closed 并保留现场。由于 virtiofs 可能让 root
与 Worker 看到不同的数字 Owner，Runtime 会在 root 视角不匹配时通过 systemd 以 Worker 身份再次
执行 `stat`，只有 UID/GID 严格匹配才接受已有 Root。遇到所有 DSL 都在 `Sandbox path ownership`
处失败时，分别用普通 `stat` 和 `systemd-run --uid=<worker>` 视角检查，并执行 `findmnt -T`；不要用
递归 `chmod` 或删除非空 State。生产脚本的准确流程见
[生产运行脚本](../scripts/production/README.zh.md)。

## Kernel Catalog

使用 `list-kernels --campaign` 枚举全部 DSL Lineage，或用 `--lineage` 限定单条 Lineage。
`show-kernel` 返回生产该 Kernel 的 Agent Revision、Attempt 上下文、主权威评测和全部持久
Repeat Measurement。认证 HTTP API 还可导出有界的精确 Source Artifact。Catalog 查询只读，
不要求 Runtime 静止。
使用 `--format table` 可以直接查看带 Parent Link、时间、保留结果、延迟和相对 Parent 变化的
`v0`/`v1` 历史；面向自动化的默认格式仍为 JSON。

```bash
atrex-kernel-agent-runtime list-kernels --config runtime.json --campaign campaign_xxx
atrex-kernel-agent-runtime list-kernels --config runtime.json --lineage lineage_xxx --format table
atrex-kernel-agent-runtime show-kernel --config runtime.json --kernel kernelrev_xxx
```

## Attempt 历史

使用 Attempt 历史核对 `X` 个优化 Session，包括发生 Pivot、Blocked 或未产生 Candidate 的
Session。Kernel 历史会有意保持稀疏，不为无 Candidate 的 Attempt 虚构版本。

```bash
atrex-kernel-agent-runtime list-attempts --config runtime.json --lineage lineage_xxx --format table
atrex-kernel-agent-runtime show-attempt --config runtime.json --attempt attempt_xxx
```

## Epoch 胜者历史

Epoch 历史会直接显示每轮赛前 Active、全部 Challenger、最终胜者、晋升决策，以及起始与全局
最佳 Kernel 版本。

```bash
atrex-kernel-agent-runtime list-epochs \
  --config runtime.json --lineage lineage_xxx --format table
```

## Kernel Agent 历史

使用独立 Agent Catalog 查看 Bootstrap `agent-v0`、每个 Evolver Challenger、真实 Parent、
晋升结果和 Active 状态。一个 Agent 版本可以生产多个 Kernel 版本。

```bash
atrex-kernel-agent-runtime list-agent-revisions \
  --config runtime.json --lineage lineage_xxx --format table
atrex-kernel-agent-runtime show-agent-revision \
  --config runtime.json --agent-revision agentrev_xxx
```

## Attempt 评测历史

使用 `list-evaluations` 查看一个 Attempt 中 Agent 提交的全部探索 Kernel，以及 Runtime 持有的
Comparator/Finalization Record。Retention Comparator 的 Candidate 聚合是权威结果；Runtime 不会
在其后再增加一次独立 Final Eval。`show-evaluation` 标识一对不可变记录；`--source` 返回该步实际评测的准确
文件，`--result` 返回完整原始 Gateway 响应。即使 Kernel 错误、被回退或从未成为 Kernel
Revision，这些记录也仍然存在。CLI 会直接打开 Gateway Control，因此需要配置中指定的
Capability Signing Key 环境变量。

```bash
atrex-kernel-agent-runtime list-evaluations \
  --config runtime.json \
  --attempt attempt_xxx

atrex-kernel-agent-runtime show-evaluation \
  --config runtime.json \
  --evaluation geval_xxx \
  --source \
  --result
```

## Bootstrap 执行历史

一个稳定 Bootstrap Attempt 可以对应多个物理 Core Session。每个 Session 都作为 Append-only
Recovery Generation 保留，包含终态、失败原因、准确 Workspace、Provider Token 用量、Session
Trace/Report Digest 和可用的权威结果身份。CLI 会直接打开 Gateway Control，因此必须提供 Runtime
配置所命名的 Signing Key 环境变量。对应的认证 HTTP 接口是
`GET /v1/admin/bootstrap-attempts/{attempt_id}/runs` 和 `/runs/{generation}`。

```bash
atrex-kernel-agent-runtime list-bootstrap-runs \
  --config runtime.json \
  --attempt attempt_xxx

atrex-kernel-agent-runtime show-bootstrap-run \
  --config runtime.json \
  --attempt attempt_xxx \
  --generation 2
```

新的失败异常会携带 Attempt、Generation 和 Run 身份，并产生
`bootstrap.lineage_baseline_failed` 生命周期事件。

## Optimizer 调试 Shell

`dev-shell --lineage` 会在持有 Lineage Fence 时创建或复用当前 Epoch 的首个 Active Attempt，
物化与正式 Optimizer 相同的 Workspace 并注入 Gateway/Wiki Capability，但只启动交互式
`zsh/bash`。`--attempt` 可为已有 `running` Attempt 创建新的 Run Workspace。退出不会完成
Attempt；运维人员之后必须明确恢复 Campaign 或保留现场。不要与同一 Lineage 的 Scheduler 并发运行，
也不要记录包含 Capability 的完整环境。

```bash
atrex-kernel-agent-runtime dev-shell \
  --config runtime.json \
  --lineage lineage_xxx \
  --shell zsh
```

## Evolver 调试 Shell

`evolver-dev-shell` 必须同时指定 Lineage ID 和一个已经存在的绝对 Epoch 编号。它持有 Lineage
Fence，解析 Campaign 固定的 Evolver Commit，并重建目标 Epoch 的冻结 Evolution Workspace 与
环境，但不执行 Evolver Backend。视图以 Epoch 记录的 Parent Agent 和 Evidence Checkpoint 为
边界；包含已经挂载到同一 Epoch 的 Challenger，但排除目标 Epoch 及所有未来 Epoch 产生的
Kernel。退出后保留 Workspace，不修改 Epoch、Challenger、选择或晋升状态。

```bash
atrex-kernel-agent-runtime evolver-dev-shell \
  --config runtime.json \
  --lineage lineage_xxx \
  --epoch 2 \
  --shell zsh
```

该命令会准备与正式启动相同的 Contract，因此仍要求 Evolver Worker 配置中声明的继承环境变量。
Sandbox 模式下交互 Shell 使用相同 cgroup 与 bubblewrap 边界，并共享宿主网络，
直接出站；它不需要 Gateway Capability。

如果只需在没有现有 Lineage/Epoch 的情况下测试目录与 CLI，可使用
`temporary-evolver-dev-shell --config runtime.json --campaign campaign.json`。该命令根据固定的
Core 基线和初始 Evidence 合成 Active `agent-v0`，不启动 Agent 或 Runtime HTTP 服务，不伪造
Kernel 测量，并在退出时销毁工作区。

## Worker Session 检查

每个模型驱动的物理进程都独立于领域结果进入统一目录：

```bash
atrex-kernel-agent-runtime list-worker-sessions --config runtime.json --lineage "$LINEAGE_ID" --format table
atrex-kernel-agent-runtime list-worker-sessions --config runtime.json --attempt "$ATTEMPT_ID"
atrex-kernel-agent-runtime show-worker-session --config runtime.json --session "$WORKER_SESSION_ID"
```

对应鉴权 API 为 `GET /v1/admin/campaigns/{id}/worker-sessions`、
`GET /v1/admin/lineages/{id}/worker-sessions`、
`GET /v1/admin/epochs/{id}/worker-sessions`、
`GET /v1/admin/attempts/{id}/worker-sessions` 与
`GET /v1/admin/worker-sessions/{id}`。失败或超时进程即使来不及封存 Trace 也仍然可见；
目录会保留准确 Workspace 和终态诊断。

## 恢复与维护

重复 Bootstrap、绝对 Epoch 目标与 Task Creation Key 都是幂等的。Bootstrap 重试会轮换 Capability
Generation，同时保留此前 Run 与 Gateway Operation。Failed Epoch 恢复必须提供 Operator Key 和
原因。Cancellation 是协作式的，只在安全 Transition 完成。Wiki Query 会同步冻结，不存在待排空
Delivery Queue。

备份 SQLite 前必须停止 Runtime、Scheduler、Task Worker 和 Agent 进程，并一致备份 Registry、
Gateway Control、Agate Jobs 与 Artifact Store。先恢复到隔离环境，再运行 Readiness 与身份校验。
Runtime SQLite 使用 rollback journal；如果仍有 `-wal`/`-shm` 文件，说明状态由旧版本生成，只能
在所有持有进程停止后迁移。

Artifact/Workspace GC 有界且默认 Dry Run。只有在部署静止并满足事故保留要求后才能 Apply。不得直接删除 CAS 对象。

轮换 Capability Signing Key 前要静止系统或明确接受现有 Capability 全部失效。Provider/Agate/Wiki 凭据通过环境/Secret Manager 轮换，不得重写不可变 Artifact。事故处理要记录生命周期 Event 和相关 Artifact Digest。

## 安全警告

`launcher.mode=sandbox` 与 `launcher.mode=container` 都是生产模式。两者在 bwrap 或配置的 Resolver
不可用时都会 Fail Closed，绝不会降级到 development；`sandbox` 还要求 systemd/cgroup-v2，
`container` 则依赖外层 OCI 的资源总限额。两者有意共享所在环境的 Network Namespace，因此不隔离
宿主/容器服务、Worker 流量或出站目的地。
`launcher.mode=development` 有意不隔离，只适合可信本地工作。生产晋升前必须在目标镜像执行
负测：相邻/宿主读取、写出 `~/workspace`、直接 Internet/DNS 成功、Namespace 逃逸、Fork/
内存/CPU 耗尽、超时清理以及凭据挂载范围。

仓库验收脚本会实际检查文件系统、Capability、cgroup 资源、直连公网、DNS 和 Runtime Port。
`sudo` 进程只负责通过 system manager 创建 transient service，
systemd 会以已配置的非 root Worker 账号执行 bwrap。Lima 通过只是参考 Smoke 结果，
不能取代目标生产 Kernel/镜像的同脚本验收。

生产发布还必须满足[测试与生产验收](testing-and-acceptance.zh.md)中的全部门禁。
