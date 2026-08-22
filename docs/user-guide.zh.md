# 使用说明

[English](user-guide.md) | 中文

## 1. 前置条件

开发模式要求 Python 3.12+、Git、一种受支持的 Agent CLI（`claude`、`codex`、`qodercli` 或
`pi`），以及可访问的 Agate/GPU 环境。宿主机生产沙箱模式还要求 Linux、bubblewrap、
systemd+cgroup v2 和专用非 root Worker 用户；`container` 模式要求在专用外层 OCI 容器内安装
bubblewrap、允许其创建 Namespace，并由运维方设置资源总限额，但不需要嵌套 systemd/cgroup。
Worker 共享所在环境的网络。

安装前拉取全部固定版本仓库：

```bash
git submodule update --init --recursive
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

也可以安装发布 Wheel。Core、Evolver 和 Atrex Bench 不打进 Wheel；Runtime 在执行时导入配置中
指定的 Git Commit。

## 2. 运行第一个工作流

最短端到端检查：

```bash
export AGATE_URL='https://your-agate.example.com'
export AGATE_AK='...'
export AGATE_SK='...'
export AGATE_GPU='H20'
export QODER_PERSONAL_ACCESS_TOKEN='...'
bash examples/bootstrap/run.sh
```

脚本会创建隔离 Workspace、生成本地 Secret 和配置、启动 Runtime、Bootstrap 一个 Triton VecAdd
Lineage、输出检查结果并停止 Runtime。其他样例见 [examples/README.zh.md](../examples/README.zh.md)。

## 3. 配置部署

复制 [`runtime.example.json`](../runtime.example.json) 为私有部署文件。Runtime 配置是严格 schema
v1，主要需要确定：

- Registry/Gateway/Agate Job 数据库和 Artifact/Workspace 路径；
- Agate URL、GPU、认证、Timeout 和健康检查周期；
- Core Base 仓库及 Evolver 仓库/完整 Commit；
- Optimizer/Evolver Backend、启动命令、环境白名单和 Session Policy；
- Gate Policy、比较方法、Roofline Builder 和 Production Gate；
- `development`、外层 OCI `container` 或 Linux `sandbox` Launcher；
- 可选 GPU Wiki Query 服务；
- Administration 和 Maintenance 上限。

从 Example 创建 Campaign schema-v3 文件。输入的 `hardware_target` 用于选择 Agate GPU 环境；
Bootstrap 会查询远端环境，把返回的架构（例如 `sm_120`）传给 Agent，并只将规范 Agate GPU
别名用于调度。Campaign 还会选择 Operator、Evaluation Contract、精确 Core
Commit、Epoch 拓扑、每个 DSL 的 Seed Kernel/Evidence 和可选 Lineage Model。不存在独立的
Bootstrap JSON。

配置只保存环境变量名称，不保存 Secret。把真实值注入 Runtime 进程：

```bash
export ATREX_CAPABILITY_SIGNING_KEY="$(openssl rand -base64 32)"
export ATREX_ADMIN_BEARER_TOKEN="$(openssl rand -hex 32)"
export AGATE_AK='...'
export AGATE_SK='...'
# 只导出所选 Backend 需要的 Provider 凭据。
```

仓库内 Example 会自动生成 Signing/Admin 值。生产值必须在进程重启间保持稳定，并由 Secret
Manager 管理。

## 4. 启动并检查 Runtime

```bash
atrex-kernel-agent-runtime serve --config /absolute/path/runtime.json
curl -fsS http://127.0.0.1:8765/healthz
curl -fsS http://127.0.0.1:8765/readyz
```

`healthz` 检查进程存活；`readyz` 检查本地 Registry、Gateway Control、Agate Job 数据库和 Artifact
Staging。外部 Agate 健康状态单独写日志，短暂不可用不会让本地 Ready 失败。

## 5. Bootstrap Campaign

保持 Service 运行，因为 Core Session 会通过 HTTP 回调 Runtime Tools：

```bash
atrex-kernel-agent-runtime bootstrap \
  --config /absolute/path/runtime.json \
  --campaign /absolute/path/campaign.json
```

Bootstrap 以 `creation_key` 幂等。它冻结 Campaign Contract 与 Commit，可选生成公开 Agent Problem，
为每个 DSL 运行一个 Core Baseline Session，保留所有 Bootstrap Generation 和评测，最后发布
`agent-v0` 与权威 Kernel `v0`。相同输入重试会继续已有进度；改变不可变输入会被拒绝。

## 6. 运行 Epoch

把 Campaign 或指定 Lineage 运行到绝对 Epoch：

```bash
atrex-kernel-agent-runtime run-campaign \
  --config /absolute/path/runtime.json \
  --campaign campaign_0123456789abcdef0123456789abcdef \
  --target-epoch 3
```

`--target-epoch` 是绝对编号，重复命令安全。只有确定不再调度时才加 `--finalize`。Runtime 串行创建
Challenger Pool，再按 `max_parallel_branches` 执行 Branch；同一 Branch 内 Trajectory 可并发，单个
Trajectory 内 Attempt 串行。

队列模式通过 `POST /v1/admin/tasks` 提交，并运行：

```bash
atrex-kernel-agent-runtime run-task-worker --config runtime.json --watch
```

## 7. 查看结果

```bash
atrex-kernel-agent-runtime list-epochs --config runtime.json --campaign "$CAMPAIGN" --format table
atrex-kernel-agent-runtime list-attempts --config runtime.json --campaign "$CAMPAIGN" --format table
atrex-kernel-agent-runtime list-kernels --config runtime.json --campaign "$CAMPAIGN" --format table
atrex-kernel-agent-runtime list-agent-revisions --config runtime.json --campaign "$CAMPAIGN" --format table
atrex-kernel-agent-runtime list-worker-sessions --config runtime.json --campaign "$CAMPAIGN" --format table
```

使用 `show-attempt`、`show-kernel`、`show-agent-revision`、`show-worker-session` 查看精确记录。
`list-evaluations`/`show-evaluation --source --result` 展示每次探索性和 Runtime 权威 Kernel/Result。
`list-bootstrap-runs`/`show-bootstrap-run` 展示所有物理 Bootstrap Generation，包括失败记录。

## 8. 运行生产任务

生产脚本推荐把常驻控制面与算子任务分开。先启动一次 Runtime 和 Wiki：

```bash
bash scripts/production/services.sh start \
  --workspace workspaces/production/control-l20n \
  --hardware-target L20N \
  --env-file env.sh
```

随后可为不同算子和 Backend 启动独立后台任务；每个任务默认创建 CUDA、Triton、CuteDSL 三个
单 DSL Campaign，并行 Bootstrap 后运行到绝对目标 Epoch：

```bash
bash scripts/production/campaign.sh start \
  --service-workspace workspaces/production/control-l20n \
  --kernel production_qwen35_35b_inhouse_4k256/flash_attention \
  --backend qodercli \
  --target-epoch 10 \
  --env-file env.sh
```

按 DSL 查看 Epoch、Kernel 和 Agent 历史：

```bash
bash scripts/production/inspect.sh \
  --workspace workspaces/production/production-qwen35-35b-inhouse-4k256--flash-attention--l20n--qodercli \
  --dsl triton
```

启动命令不要整体包在 `sudo` 中；脚本会在需要创建 cgroup/bwrap transient service 时自行提权，
从而保留正确的宿主 Home、Provider 登录态和 Worker Workspace Owner。完整布局、日志、状态和故障
处理见[生产运行脚本](../scripts/production/README.zh.md)。

## 9. 从 Artifact/Revision 新建 Lineage

复制 [`lineage-seed.example.json`](../lineage-seed.example.json)，选择 Agent/Kernel Artifact Digest 或
已有 Revision ID：

```bash
atrex-kernel-agent-runtime seed-lineage \
  --config runtime.json \
  --campaign "$CAMPAIGN" \
  --spec lineage-seed.json
```

Runtime 会重新校验 Agent 仓库、按目标 Campaign Contract 独立评测 Kernel，并创建新的独立
`agent-v0`/`v0` Lineage；不会改变 Campaign 冻结的 Core/Evolver Commit。

## 10. 调试 Agent Workspace

以下命令创建或重建真实 Workspace 和权限，但不启动 Agent：

```bash
atrex-kernel-agent-runtime dev-shell --config runtime.json --lineage "$LINEAGE"
atrex-kernel-agent-runtime evolver-dev-shell --config runtime.json --lineage "$LINEAGE" --epoch 2
```

只用于可信调试。Optimizer Runtime Tools 与 Evolver 检查工具见[接口说明](interfaces.zh.md)。

## 11. 恢复与维护

失败 Epoch 不会被自动改写。检查后显式授权幂等重试：

```bash
atrex-kernel-agent-runtime recover-epoch --config runtime.json --epoch "$EPOCH" \
  --recovery-key incident-2026-08-20 --reason 'Agate allocation interruption'
```

Artifact/Workspace GC 默认仅预览。使用 `--apply --confirm-runtime-stopped` 前必须停止 Runtime、
Worker。备份恢复、Event 导出/清理、取消、凭据轮换和 Linux 沙箱见
[部署与运维](operations.zh.md)。
