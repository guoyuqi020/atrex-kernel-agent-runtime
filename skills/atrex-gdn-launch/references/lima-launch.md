# Lima 启动细节

这些命令描述实际启动操作，不是创建/检查 Skill 时应执行的测试。
从当前仓库读取配置，若命令或字段已变化，依据代码调整，保留身份和权限边界。

## 只读检查与准备

宿主机先确认 `limactl list` 的 `ubuntu` 实例，然后进入：

```bash
limactl shell ubuntu
cd ~/atrex-runtime
source ~/.venvs/atrex-runtime/bin/activate
python -c 'import atrex_runtime, anyio, httpx, pydantic'
command -v python atrex-kernel-agent-runtime claude bwrap systemd-run
free -h
nproc
test -f /sys/fs/cgroup/cgroup.controllers
systemctl show atrex-gdn-runtime.service atrex-gdn-campaign.service atrex-gdn-ablation.service \
  -p Id -p ActiveState -p SubState -p MainPID -p WorkingDirectory -p User
```

当前部署的 GDN Runtime/Campaign 调度单元使用系统服务的 root 权限；
Sandbox Worker 和 CLI credentials 的 host_home 在 Runtime 配置中指向非 root Linux 用户。
不要把服务 root 的 `~/.claude` 当作默认 Worker 凭据来源。
Lima 中可能没有 `rg`，可退回 `grep`；缺少搜索工具不是启动故障。

只有缺少生成配置且用户要求准备/启动时，才运行：

```bash
# 按用户选择设置；恢复已有实验时保留其原 workspace。
gdn_inputs=data/GDN
gdn_workspace=workspaces/GDN
# 不屏蔽原始提示的对照：
# gdn_inputs=data/GDN-full
# gdn_workspace=workspaces/GDN-full
python scripts/gdn/prepare.py --inputs "$gdn_inputs" --workspace "$gdn_workspace" --backend claude
```

迁移来的旧 `workspaces/GDN` 保留旧输入，不能用新 data 模板覆盖。用户需要新清理版实验时，
应选未使用的工作区（如 `workspaces/GDN-clean`）。下文的 `gdn_workspace` 必须与本次选择一致。

prepare 不启动服务或 GPU 作业；它会还原源 Git Bundle 并验证输入。
已有状态时拒绝覆盖不同配置是预期行为。声明 Commit 与本地 dirty tree 是两回事，
不要通过 checkout/清理用户改动或自动 commit 来“修复”这个边界。

凭据检查避免输出秘密：

```bash
set -a
source env.sh
set +a
test -n "${AGATE_AK:-}"
test -n "${AGATE_SK:-}"
curl -fsS --max-time 5 http://127.0.0.1:8766/healthz
curl -fsS --max-time 5 http://127.0.0.1:8091/readyz
```

上面 URL 是本次默认值；配置不同则使用配置值。不运行 `env`、`set`（无参数）、
`set -x`、`cat env.sh` 或输出完整 Secrets JSON。

## 持久化启动消融任务

以下在 Lima 内、Linux venv 已激活、服务已匹配且健康、同名 runner 不在运行时使用。
普通 `python scripts/gdn/run.py ablation --workspace "$gdn_workspace"` 不会自动 sudo；
如果当前调度用户没有 systemd 系统服务管理权限，使用这类独立 transient service。
所有绝对路径从当前环境解析，不写死 macOS 路径或某个 Python 小版本：

```bash
gdn_repo="$(pwd -P)"
gdn_python="$(command -v python)"
gdn_path="$PATH"
test -f "$gdn_workspace/runtime.json"
sudo -n systemd-run \
  --unit=atrex-gdn-ablation --collect \
  --property=Type=exec \
  --property=WorkingDirectory="$gdn_repo" \
  --property=KillMode=control-group \
  --setenv=PATH="$gdn_path" \
  /bin/bash -c '
    set -euo pipefail
    set -a
    source env.sh
    set +a
    exec "$1" scripts/gdn/run.py ablation --workspace "$2"
  ' gdn-launch "$gdn_python" "$gdn_workspace"
```

先验证 `gdn_repo` 确实包含本次文件。执行前说明将产生模型/GPU 消耗；
用户已要求启动即是任务授权，环境的提权审批仍应通过正常工具机制处理。
没有 sudo 权限时报告，不改 sudoers。
新实验默认目标为 100 个 Epoch。非默认目标或恢复旧实验时在内层命令追加 `--target-epoch N`；
例如恢复旧 5 轮实验使用 `--target-epoch 5`，不要自动延长其主臂；
对照臂预算仍由计划决定。不要隐式缩小并行度或改消融拓扑。

`--collect` 允许服务结束后释放 transient 单元；若同名单元仍存在，先确认是不是活跃任务。
若同名但失效的配置阻止恢复，诊断单元状态，不强杀不明进程。
保留代码中的 Workspace lock 与 Registry lease 检查。

## 服务缺失时

先辨别未启动、启动失败、错误端口和错误部署。启动一个任务不需要重启健康的共享服务。
若此次请求包含准备所需服务，且没有占用冲突：

- 已有且工作区匹配的 GDN Runtime 单元可以启动；若 transient 单元已释放，
  可按上面相同模式另建 `atrex-gdn-runtime`，仅将内层 role 改为 `serve`。
  保留同一个 `--workspace`，与任务使用相同的 `runtime.json`、`env.sh` 和 `runtime-secrets.json`。
  不把指向另一工作区的健康 Runtime 当作本次服务，也不杀掉它来抢占端口。
- Wiki 使用仓库的 `examples/local-wiki/start-local-wiki.sh`；
  先读脚本及 `local-wiki/configs/local.example.json`，确认语料、依赖、端口、状态目录权限。
  用 Linux venv，通过 `ATREX_PYTHON` 指定解释器。优先以非 root 用户独立托管，
  不复用未知的其他任务 Wiki workspace。
- 服务启动后重新检查 health/ready，再提交 Campaign。
- 遇到缺少依赖、语料、不可用 Agate 或配置不兼容时明确报告；
  不用启动模型“试试看”代替前置排查。

## 验证与恢复

```bash
systemctl show atrex-gdn-ablation.service -p ActiveState -p SubState -p MainPID -p ExecMainStatus
sudo -n journalctl -u atrex-gdn-ablation.service -n 60 --no-pager
tail -n 40 "$gdn_workspace/ablation/bootstrap.log"
```

日志里可能包含不应复述的环境/服务信息；给用户摘要，不原样倾倒。
七臂 fan-out 后查看各臂 `campaign.log` 和 `campaign-results.json`。
服务活跃但短时间无新日志不等于卡死：继续看实际 Worker、最新 Session 与 Gateway 请求，
不据此自动重启。

恢复沿用同一文件、key 和服务命令；先确认旧 runner 已结束。
重复执行脚本会使用 Runtime 的幂等 Bootstrap/Seed 和绝对 Epoch 目标。
若用户要恢复的是旧试跑，使用独立的 `run.py campaign` 入口及旧输出，
不把 ablation 输出覆盖到旧 `bootstrap-result.json` 上。
