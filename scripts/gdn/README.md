# GDN launcher

New GDN/GDN-full deployments default to `container` mode: bwrap still isolates each Agent,
but there is no nested systemd service, user switching or per-Session cgroup limit. Run as
the current container user (prefer non-root); CLI credentials come from that user's Home.
`--worker-user` cannot select another user in this mode. Provision CPU/memory/PID limits in
the outer container. Running directly in Lima relies on VM-wide limits instead.
This mode does **not** create or start a Docker container. bwrap and namespace support must
already be available; startup reports an error rather than dropping isolation if they are not.
Existing workspaces retain their frozen launcher configuration.

## Shared Runtime for GDN and GDN-full

Prepare one service workspace and two **new, independent task workspaces** in Lima:

```bash
cd ~/atrex-runtime
source ~/.venvs/atrex-runtime/bin/activate
set -a
source env.sh
set +a

# Configure once. If 8766 is already occupied, choose a free port with --port 8767.
python scripts/gdn/prepare.py --services-only \
  --workspace workspaces/control-gdn --backend claude

python scripts/gdn/prepare.py --inputs data/GDN \
  --workspace workspaces/GDN-shared --service-workspace workspaces/control-gdn
python scripts/gdn/prepare.py --inputs data/GDN-full \
  --workspace workspaces/GDN-full-shared --service-workspace workspaces/control-gdn

# One persistent service; run separately as the same container user, without sudo.
python scripts/gdn/run.py serve --workspace workspaces/control-gdn
```

KDA no longer uses Wiki. Both templates disable it with `gpu_wiki: null`; no Wiki service,
corpus or readiness check is needed. Both tasks must use the same deployment environment.
In two other terminals/processes as the same container user with that environment:

```bash
python scripts/gdn/run.py campaign --workspace workspaces/GDN-shared
python scripts/gdn/run.py campaign --workspace workspaces/GDN-full-shared
```

Use `ablation` instead of `campaign` for seven arms **per input variant** (14 Campaigns total).
Each variant has its own Bootstrap, v0, public inputs, history and creation keys. Sharing a
Runtime does not seed GDN from GDN-full. Two ablation suites may run up to 20 Optimizers;
the branch limit is not a deployment-wide semaphore. Check host memory before parallel runs.
Both runners default to absolute Epoch 100; no processes are launched by preparation.

- `control-gdn/`: `runtime.json`, `service.json`, one `runtime-secrets.json`, and shared
  `state/` for Registry, Gateway, Artifacts and all Session workspaces.
- `GDN-shared/` and `GDN-full-shared/`: task/source snapshots, Campaign/ablation definitions,
  `service-binding.json`, and task-specific logs/results. No local Runtime config, keys or DB.
- The saved binding uses a relative service path and a Runtime config digest. `run.py` finds
  it automatically; optional `--service-workspace` verifies, but cannot override, the binding.
- Shared config is immutable after preparation. Backend/worker overrides that disagree with
  the service are rejected. To change the deployment, prepare a new service/task workspace.
- `serve` on an attached task is rejected; start only the service workspace. Stopping a task
  does not stop services. Stopping shared Runtime affects both tasks.
- Existing standalone workspaces are not migrated or merged. Continue them with the old
  entrypoint below; never copy SQLite files/keys between live workspaces to attach a task.

Inspect either task by its Campaign/Lineage ID using
`--config workspaces/control-gdn/runtime.json`. Session/Artifact paths come from that shared
config, not from the task's local log directory.

## Standalone workspace

Task inputs and templates: [`data/GDN`](../../data/GDN/README.md).
Default workspace: `workspaces/GDN`; no generated files are written to `data/`.

Reusable launch Skill: [`skills/atrex-gdn-launch`](../../skills/atrex-gdn-launch/SKILL.md),
including the Lima service-management reference and Agent UI metadata.

Use `prepare.py --inputs data/GDN-full` for the [original-hint input variant](../../data/GDN-full/README.md).
When `--workspace` is omitted, preparation uses `workspaces/<input directory name>`.
For this variant, pass `--workspace workspaces/GDN-full` to every `run.py` command.

Run in Lima Ubuntu with the Linux Runtime environment activated:

```bash
python scripts/gdn/prepare.py --backend claude
python scripts/gdn/run.py serve
# In a separate terminal as the same container user with the same environment:
python scripts/gdn/run.py campaign --target-epoch 100
```

For a separate experiment, pass `--workspace workspaces/GDN-clean` to all commands.
`run.py ablation` uses the workspace's frozen seven-arm definitions. Only Runtime is needed;
these scripts do not start Wiki. Preparation does not run Agents or evaluations.
Run roles use existing workspace snapshots; they never silently re-prepare changed task inputs.

中文：输入和模板只放在 [`data/GDN`](../../data/GDN/README.zh.md)，脚本放在此目录。
可复用的[启动 Skill](../../skills/atrex-gdn-launch/SKILL.md) 也随仓库发布，不依赖作者的本机副本。
保留原始提示的版本位于 [`data/GDN-full`](../../data/GDN-full/README.zh.md)，通过
`prepare.py --inputs data/GDN-full` 选择，默认输出到 `workspaces/GDN-full`。
实际配置、源码副本、数据库、Session、凭据、日志及结果均位于 `workspaces/GDN`。
使用新输入时，为所有命令指定相同的新 `--workspace`；恢复旧实验直接运行 `run.py`，
不会重新导入当前 `data` 中的修改。KDA 不再使用 Wiki；两份模板均设置 `gpu_wiki: null`，
不启动 Wiki，也不要求下载语料或检查 Wiki 就绪状态。

## GDN 与 GDN-full 共用服务（中文）

上面的共享模式命令在 Lima 中执行：先用 `prepare.py --services-only` 准备一次
`workspaces/control-gdn`，再分别用 `--inputs data/GDN` 和 `--inputs data/GDN-full`，
通过 `--service-workspace workspaces/control-gdn` 绑定两个新的任务工作区。
已有 8766 服务时选择空闲端口，例如准备服务时加 `--port 8767`，不要抢占旧任务端口。

只对服务工作区运行一次 `run.py serve`，无需 Wiki。之后可在两个独立进程中分别运行两条 `run.py campaign` 命令，
或将 role 改为 `ablation`，启动每份输入各自的七臂消融。准备本身不启动模型/GPU 作业。
请以同一个容器用户、相同环境变量运行服务与任务，不需要 sudo/systemd 调度权限。

GDN/GDN-full 新配置默认 `container`：保留 bwrap 工作区隔离，不切换用户，不创建每个
Session 的 systemd/cgroup。CLI 凭据来自当前用户 Home；`--worker-user` 不能指定其他用户。
CPU/内存/PID 限制由外层容器提供；直接运行在 Lima 时只有 VM 总体限制，不再有原先的
每 Session 3 GiB 限额。该模式不会自动创建 Docker 容器，环境仍需支持 bwrap/namespace。
已有工作区保留冻结模式，不自动从 sandbox 切换。

共享的是 Runtime 配置、Registry、Artifact Store、Session 存储和鉴权密钥；
两份任务的输入快照、源码、Bootstrap、v0、经验历史和结果身份仍独立。
任务目录只保存 `service-binding.json`，不复制 `runtime.json`、密钥或数据库；
后续启动自动读取绑定，无需重复传服务参数。可显式传入 `--service-workspace` 做一致性检查，
但不能借此改绑。使用共享配置和对应 Campaign/Lineage ID 执行 inspect。

绑定时会冻结服务配置摘要；配置漂移、Backend/Worker 不匹配，以及误从任务目录启动
`serve` 都会提前拒绝。已有独立工作区不自动迁移，不要手工合并数据库或复制密钥。
改变部署使用新的服务和任务工作区；停止任务不停止共享服务，停止共享 Runtime 会影响所有任务。
两套七臂消融最多约 20 个 Optimizer 并行，当前限制不是全局并发配额，请先检查内存。
