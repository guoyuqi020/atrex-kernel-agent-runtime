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
Bootstrap 发生进程退出或基础设施错误时，会在 `campaign.max_infrastructure_retries` 上限内自动
重试；每次重试都会获得新的 Capability、Workspace、Session 和执行 Generation。Optimizer Attempt
的基础设施重试也由同一个配置项控制。
Evolver 的进程退出和基础设施错误同样使用该上限；Runtime 会先保留失败 Worker Session 和
Evolution Failure Trace，再使用新 Workspace 重试。
生成的 Epoch-0 Evidence 对后续 Optimizer/Evolver 只暴露 `bootstrap/report.json` 和
`bootstrap/conversation.jsonl`。

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

Claude Session Artifact 在 `provider/claude-session.raw-jsonl` 和 `provider/claude-subagents/` 中保留原生主/子会话。规范化 `events.jsonl` 通过 `message_id`、`source_path` 将每个响应的最新 usage 关联回原始消息及工具调用。使用逐响应统计前应检查 `session.json.response_usage_complete`；缺失或与终态总账无法核对的计数仍标为 partial。不要重复累加 stdout/native 副本，也不要把终态总账加到逐响应用量上。一个响应可能包含多个工具调用：这些计数属于响应，并不是每个工具独立计费的用量。

封存后的 `conversation.jsonl` 是阅读视图：Claude 优先使用原生内容，省去已被完整覆盖的 stdout 消息副本，保留不同的 thinking/text/tool 内容块、未被覆盖的 stdout 内容、诊断、压缩边界和终态结果。重复的初始 Prompt，以及原生队列、标题、文件历史等内部管理事件只从阅读视图中省去。封存前的实时视图仍跟随 stdout。原始 Provider 文件及规范化 usage 索引不变。

该采集能力需要更新后的 Core/Evolver Bundle Commit。现有 Campaign 冻结了 Bundle Revision，单独重启 Runtime 不会自动升级。历史运行如果禁用了原生持久化，无法从 Session 总量还原缺失的逐响应计数。

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

## 10. 创建 Ablation Arm

创建一份 Ablation v1 JSON，指定源 Lineage 与控制组拓扑：

```json
{
  "schema_version": 1,
  "creation_key": "triton-no-evolution",
  "source_lineage_id": "lineage_0123456789abcdef0123456789abcdef",
  "attempts_per_trajectory": 3,
  "trajectories_per_branch": 1,
  "ephemeral_agent_state": true,
  "optimizer_model": null
}
```

```bash
atrex-kernel-agent-runtime seed-ablation-arm \
  --config runtime.json \
  --spec ablation.json
```

Runtime 从源 Bootstrap Baseline 创建一个独立的单 Lineage Campaign，不创建 Challenger。使用
`ephemeral_agent_state=true` 可在每次 Attempt 后清空自适应 Memory/Docs/Skills/Tools；设为 false 则保留串行
State，只隔离 Evolver 修改缺失的影响。

进化对照臂设置 `challenger_count=1`、`challenger_start_epoch=2`、`first_epoch_same_agent=true`、
`ephemeral_agent_state=false`。使用返回的 Campaign ID 执行 `run-campaign --target-epoch N`。
每分支每 Epoch 1 次跑 15 轮、3 次跑 5 轮、5 次跑 3 轮，均为 30 次 Optimizer Attempt。
首轮两个分支使用同一 Agent，不调用 Evolver；之后生成进化的 Challenger。Bootstrap 直接复用。

## 11. 调试 Agent Workspace

以下命令创建或重建真实 Workspace 和权限，但不启动 Agent：

```bash
atrex-kernel-agent-runtime dev-shell --config runtime.json --lineage "$LINEAGE"
atrex-kernel-agent-runtime evolver-dev-shell --config runtime.json --lineage "$LINEAGE" --epoch 2
```

只用于可信调试。Optimizer Runtime Tools 与 Evolver 的只读文件系统输入 Contract 见
[接口说明](interfaces.zh.md)。

每个 Optimizer Workspace 都包含可写的 `memory/`（搜索记忆）、`docs/`（知识）、`skills/`（技能流程）
和 `tools/`（工具脚本），各自包含 `README.md` 索引。Session 退出时，Runtime 会封存其
准确终态，并把 Artifact Digest 记录到生产它的 Attempt。后续串行 Attempt 从该 State 继续，本地缓存
丢失后也能准确重建。Framework Bootstrap 初始化 `agent-v0` State。Evolver 从最近完成 Epoch 获胜
分支中、产出最佳 Kernel 的 Trajectory 在该 Epoch 最后一个 Attempt 后的终态 State 开始；下一
Epoch 的 Active Branch 从完全相同的 State 开始。每个新 Agent
Revision 都把 Source 与 State 一起封存为
一个逻辑 Bundle，每条新 Trajectory 获得独立 State 副本。
新增、修改、重命名或删除内容时，模型必须同步更新对应 README 中的路径、用途和适用范围；工具还需
说明调用方法、输入输出、依赖、示例和限制。四目录遵循相同的继承与清空策略。旧快照仅在复制时补齐
缺失目录和索引，不改写已存储 Artifact。这些笔记由 Agent 编写，不替代权威 Journal 和 Gateway 结果。

## 12. 恢复与维护

失败 Epoch 不会被自动改写。检查后显式授权幂等重试：

```bash
atrex-kernel-agent-runtime recover-epoch --config runtime.json --epoch "$EPOCH" \
  --recovery-key incident-2026-08-20 --reason 'Agate allocation interruption'
```

Artifact/Workspace GC 默认仅预览。使用 `--apply --confirm-runtime-stopped` 前必须停止 Runtime、
Worker。备份恢复、Event 导出/清理、取消、凭据轮换和 Linux 沙箱见
[部署与运维](operations.zh.md)。
