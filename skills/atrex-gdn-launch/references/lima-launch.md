# Lima 启动细节

这些命令描述实际启动操作，不是创建/检查 Skill 时应执行的测试。
从当前仓库读取配置，若命令或字段已变化，依据代码调整，保留身份和权限边界。

## 只读检查与准备

GDN/GDN-full 共用服务时，先按仓库 `scripts/gdn/README.md` 完成一次服务准备和两次任务绑定。
下文使用 `gdn_workspace` 指任务目录；额外设置 `gdn_service_workspace` 为其绑定的服务目录。
独立模式令二者相同。已有 `service-binding.json` 时不要因为任务目录缺少 `runtime.json`
而重新准备或启动另一个服务。服务健康 URL 从服务配置读取。

宿主机先确认 `limactl list` 的 `ubuntu` 实例，然后进入：

```bash
limactl shell ubuntu
cd ~/atrex-runtime
source ~/.venvs/atrex-runtime/bin/activate
python -c 'import atrex_runtime, anyio, httpx, pydantic'
command -v python atrex-kernel-agent-runtime claude bwrap
free -h
nproc
# 下面只用于检查仍在运行的旧 systemd 任务；新 container 模式不以此作为启动要求。
systemctl show atrex-gdn-runtime.service atrex-gdn-campaign.service atrex-gdn-ablation.service \
  -p Id -p ActiveState -p SubState -p MainPID -p WorkingDirectory -p User
```

新配置使用 container 模式，直接以当前容器用户（建议非 root）运行 Runtime/Campaign。
CLI credentials 的 host_home 在准备时取该用户 Home。不要通过 sudo 切换用户后启动；
`--worker-user` 不能在 container 模式切换用户。旧 sandbox 服务才保留原来的 root 调度权限。
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

上面的 prepare 是独立模式。共享模式改为添加 `--service-workspace "$gdn_service_workspace"`，
并使用服务已配置的 Backend/Worker；端口只能在服务准备时设置。

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
```

上面 URL 是本次默认值；配置不同则使用配置值。GDN 不使用 Wiki，无需检查 8091 端口。
不运行 `env`、`set`（无参数）、
`set -x`、`cat env.sh` 或输出完整 Secrets JSON。

## 持久化启动消融任务

以下在 Lima 内、Linux venv 已激活、服务匹配且健康、没有相同任务 runner 时使用。
新 container 模式不要求 systemd/cgroup v2，不自动提权。直接在已准备的容器/持久会话中运行：

```bash
set -a
source env.sh
set +a
python scripts/gdn/run.py ablation --workspace "$gdn_workspace"
```

长任务用部署已有的进程管理器或持久终端托管；不要强行安装 systemd 或用 sudo 绕过权限。
由外层容器配置 CPU/内存/PID 限制，bwrap 负责工作区隔离；直接在 Lima 执行只受 VM 总体限制。
若 bwrap/user namespace 被外层 seccomp/AppArmor 限制，报告探测结果，不自动开放整个容器权限。
运行脚本不创建 Docker 容器，也不改变外层容器配置。

仅恢复已冻结的 sandbox 工作区时才沿用旧 systemd 服务：
先核对该服务配置和实际进程，再通过已有单元恢复，不把新 container 任务塞进旧 root 服务。

先说明将产生模型/GPU 消耗。新实验默认目标为 100 个 Epoch；
非默认目标或恢复旧实验时追加 `--target-epoch N`，
例如恢复旧 5 轮实验使用 `--target-epoch 5`，不要自动延长主臂。
对照臂预算仍由冻结计划决定，不隐式改拓扑。保留 Workspace lock 和 Registry lease 检查。

## 服务缺失时

先辨别未启动、启动失败、错误端口和错误部署。启动一个任务不需要重启健康的共享服务。
若此次请求包含准备所需服务，且没有占用冲突：

- 默认 container 模式直接运行 `python scripts/gdn/run.py serve --workspace "$gdn_service_workspace"`。
  独立模式服务目录才与任务目录相同。采用已有进程管理器托管，不要求 systemd 或 sudo。
  两者必须使用同一份服务配置、`env.sh` 和 `runtime-secrets.json`。
  不把指向另一工作区的健康 Runtime 当作本次服务，也不杀掉它来抢占端口。
- 不启动 Wiki，也不下载 Wiki 语料；KDA 不再使用它。其他任务已运行的 Wiki 不受影响。
- Runtime 启动后重新检查 `/healthz`，再提交 Campaign。
- 遇到缺少依赖、不可用 Agate 或配置不兼容时明确报告；
  不用启动模型“试试看”代替前置排查。

## 验证与恢复

```bash
# 用实际进程管理器检查 runner 状态；旧 systemd 任务才使用 systemctl/journalctl。
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
