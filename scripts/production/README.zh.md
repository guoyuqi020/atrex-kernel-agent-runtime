# 生产运行脚本

[English](README.md) | 中文

这套脚本从 `third_party/atrex-bench/data` 中选择一个算子，创建 CUDA、Triton、CuteDSL
三个单 DSL Campaign Workspace。三个 Campaign 共享一个可信 Runtime、Registry、Artifact
Store、Wiki 和 Agate 连接，但分别持有 Campaign 定义、Bootstrap 输入、Evaluation
Contract、结果和日志。`sandbox` 模式下每个 Agent Attempt 拥有独立挂载沙箱、Home 和 cgroup；
`container` 模式在专用外层 OCI 容器内使用 bwrap 提供逐 Worker 文件系统/Namespace 边界，
由外层容器提供资源总限额。两者都直接共享所在环境的网络。

生产内容 Policy Gate 是强制项。准备、运行和服务启动入口都会检查
`campaign.gate_policy.production_gate == true`；自定义 Policy 或已有 Workspace 关闭该 Gate
时会被直接拒绝，不会启动无内容门禁的 Campaign。

## 入口选择

- 长期运行或连续提交多个算子：先执行一次 `services.sh start`，之后对每个任务使用
  `campaign.sh start`。这是推荐的生产拓扑。
- 单任务验收或独立部署：使用 `run.sh`，由任务 Workspace 自己管理 Runtime/Wiki。
- 只生成冻结配置：使用 `prepare.sh`。
- 查看结果：使用 `inspect.sh --workspace TASK [--dsl DSL]`。

不要把整个入口命令写成 `sudo bash ...`。`sandbox` 模式只在启动 bwrap/cgroup transient
service 前自行提权；`container` 模式完全不会提权。

固定调度策略：

- 三个单 DSL Campaign 由三个独立进程并行运行 framework bootstrap，各自生成正确的 `v0`；
- 每个 DSL 在自己的 Bootstrap 成功后立即进入 Epoch，不等待另外两个 DSL；
- 一个 DSL Bootstrap 或运行失败不会取消或阻塞另外两个，成功结果会保留，重复执行只恢复失败
  或未完成的 Campaign；
- 默认运行至 Epoch 5；
- Epoch 1 只有 Active，串行运行 3 个全新 Optimizer Session；
- 从 Epoch 2 开始，每个 Epoch 开始前由一个 Evolver 生成一个 Challenger；
- Active 和 Challenger 并行运行，各自串行运行 3 个 Attempt；
- Epoch 结束后独立比较并选择 Agent，下一 Epoch 再生成一个 Challenger；
- CUDA、Triton、CuteDSL 三个 Campaign 独立调度并并行推进。

默认 `event_only=true` 时，每个 DSL 还会运行不包含 Evolver 的 Ablation Campaign。
每个 Arm 都固定使用 15 个 Bootstrap 之后的 Optimizer Attempt，Bootstrap 不计入。默认 Pool
每个 Epoch 运行 3 个 Attempt，另外两个串行 Pool Arm 只改变 Attempt 的分组方式：

- `ablation-pooled`：5 个 Epoch x 3 个 Attempt；
- `ablation-pool-1`：15 个 Epoch x 1 个 Attempt；
- `ablation-pool-5`：3 个 Epoch x 5 个 Attempt。

Isolated 和 Retained 对照臂同样运行 5 个 Epoch x 3 个 Attempt。生成的 Ablation Plan
会为每个 Arm 分别保存每 Epoch Attempt 数和派生的目标 Epoch。因此 `--target-epoch`
只控制进化 Campaign，不改变固定的 Ablation 预算。

## 前置条件

Worker 边界有两种选择：

- `sandbox` 要求 Linux、`bwrap`、启用 cgroup v2 的 systemd、root/sudo 和专用非 root Worker；
- `container` 要求 Linux 与 bwrap，但不要求 systemd、可写 cgroup 层级、sudo 或逐 Session
  cgroup；OCI 安全策略必须允许 bwrap 创建所需 Namespace。Runtime 隔离各 Worker 的文件系统与
  Namespace，Docker/Kubernetes 则提供共享的内存、CPU 与 PID 总限额。不要挂载 Docker Socket、
  Runtime Secret、私有评测数据或无关宿主路径。

两种模式都需要受支持的 Agent CLI 和 Agate 凭据，并直接复用所在环境的 DNS 与公网连接。
`sandbox` 模式下，服务脚本会让 Local Wiki 以配置的非 root Sandbox Worker 身份运行；
`container` 模式沿用容器进程用户，建议直接以非 root 用户启动外层容器，并先验证 bwrap 能创建
User/PID/IPC/UTS Namespace。

准备新 Campaign 时，Core 和 Evolver Git Worktree 必须干净。生产配置只记录 Commit，因此存在
已暂存、未暂存或未跟踪的 Bundle 文件时会直接失败，不再静默使用旧 `HEAD`。创建新 Workspace
前应先提交并推送 Bundle 修改。已有 Workspace 继续固定使用自身 `production-manifest.json`
记录的 Commit；有意采用新 Agent Commit 时必须使用新的 Workspace。

### Lima 与 virtiofs

Lima 把仓库挂到 `virtiofs` 时，`chown` 可能返回成功但不改变文件显示的 UID/GID。Runtime 不依赖
“root 创建再 chown”：Sandbox host check 会通过 systemd 直接以配置的 `worker_user` 创建四个
Worker roots 和 probe。失败运行遗留的空 `root:root` root 可以自动重建；非空异常目录会 Fail
Closed，不会自动删除。如果 root 与 Worker 看到不同的数字 Owner，Runtime 会通过 systemd 以
`worker_user` 身份重新检查；Worker UID/GID 严格匹配时保留并复用非空 Root，真实不匹配仍会拒绝。

不要手工使用 `sudo mkdir -p TASK/state/*-workspaces`。如果需要预建目录，使用普通登录用户，或直接
让 Runtime 创建。Runtime SQLite 使用 rollback journal，可用于能可靠提供 POSIX 文件锁的 Lima
virtiofs；这不代表任意网络文件系统都受支持。

Agate 凭据可以提前 export，也可以复制 `environment.example` 到 Git 仓库之外并填写：

```bash
cp scripts/production/environment.example /secure/atrex-production.env
chmod 0600 /secure/atrex-production.env
```

如果没有设置 `ATREX_WIKI_URL`，配置默认使用并由服务脚本启动仓库中的 Local Wiki；设置
该变量后则只检查远端 Wiki，不会管理远端进程。

在普通 Docker 容器内首次启动常驻服务：

```bash
bash scripts/production/services.sh start \
  --workspace workspaces/production/control-l20n \
  --hardware-target L20N \
  --launcher-mode container \
  --env-file /secure/atrex-production.env
```

服务 Workspace 会固定该 Launcher 模式，后续附着的 `campaign.sh start` 会自动继承。

## 单任务一体化运行

`--kernel` 可以是完整目录、相对于 Atrex-Bench `data/` 的 `suite/operator`，或者全库唯一
的 operator 目录名。`--target-epoch` 是绝对 Epoch 编号，重复执行会从持久化状态继续。

```bash
bash scripts/production/run.sh \
  --kernel production_qwen35_35b_inhouse_4k256/causal_conv1d \
  --backend qodercli \
  --env-file /secure/atrex-production.env
```

默认绝对目标为 Epoch 5；需要继续到其他 Epoch 时可显式传入 `--target-epoch N`。

对于新版 Atrex-Bench 布局，准备流程只把 `shape_train.json` 暴露给 Agent，并将
`shape_valid.json` 作为精确 Evaluation Contract 封存。旧版 `agent_problem.json` 与
`shapes.json` 仅用于迁移回退。`metadata.json` 会私下传给评测端，其中的 `mutates_inputs` 与
`scratch_inputs` 会由远端 Correctness Gate 执行输入副作用检查。

当 `--kernel` 指向算子目录时，默认使用 `reference.py` 作为三个 bootstrap 的语义 Seed；
当它直接指向目录内的某个 Python 文件时，就使用该文件。若确定希望从 Atrex-Bench 的
已有实现开始，也可以显式使用：

```bash
--seed-source solution.py
```

首次执行会在 `workspaces/production/` 下生成一个独立工作区：

```text
runtime.json                 三个 Campaign 共用的可信 Runtime 配置
production-manifest.json     输入、布局及 Core/Evolver/Bench commit 冻结记录
runtime.env                  自动生成的 Runtime 控制面密钥（0600）
local-wiki.json              共用 Local Wiki 配置
dsls/<dsl>/                  每个 DSL 各有一个 Campaign Workspace
    campaign.json            只包含该 DSL 的 Campaign 定义
    evaluation-contract.json 该 DSL 的不可变 Evaluation Contract 副本
    production-manifest.json 带 DSL 身份的冻结记录
    inputs/                   Bootstrap Seed 与初始 Evidence
    bootstrap-result.json    该 DSL 的 Bootstrap 权威结果
    campaign-result.json     该 DSL 到目标 Epoch 的最终状态
    bootstrap.log            Bootstrap stderr/进度日志
    campaign.log             Epoch/Attempt 进度日志
bootstrap-results.json       各 DSL Bootstrap 结果或缺失状态的统一索引
campaign-results.json        各 DSL Campaign 结果或缺失状态的统一索引
campaign-run/                后台任务 PID、总日志、启动参数与终态
state/                       共用 Registry、Gateway、Artifact 与隔离 Agent Workspace
services/                    Wiki 与 Runtime 的 PID 和日志
```

已有工作区不会随源码 HEAD 变化而更换冻结 commit。生产 Policy 的摘要也会写入 Manifest；
改变算子、Backend、模型、硬件目标或调度 Policy 时，必须指定一个新的 `--workspace`。旧的
“一个 Campaign 包含三个 DSL”工作区不会被原地解释为新布局，必须先移走旧目录或选择新路径。

## 分步操作

只生成配置：

```bash
bash scripts/production/prepare.sh \
  --kernel production_qwen35_35b_inhouse_4k256/causal_conv1d \
  --backend codex \
  --hardware-target L20N \
  --workspace /data/atrex/causal-conv1d
```

启动、检查和关闭 Wiki 与 Runtime：

```bash
bash scripts/production/services.sh start  --workspace /data/atrex/causal-conv1d --env-file /secure/atrex-production.env
bash scripts/production/services.sh status --workspace /data/atrex/causal-conv1d
bash scripts/production/services.sh stop   --workspace /data/atrex/causal-conv1d
```

查看全部 DSL 或单个 DSL 的 Epoch、Kernel 和 Agent 版本历史：

```bash
bash scripts/production/inspect.sh --workspace /data/atrex/causal-conv1d
bash scripts/production/inspect.sh --workspace /data/atrex/causal-conv1d --dsl triton
```

`--dsl` 只接受 `cuda`、`triton` 或 `cutedsl`。脚本读取任务的 `runtime.json` 和该 DSL 的
`bootstrap-result.json`，因此可以在 Campaign 运行期间执行；若 Bootstrap 尚未成功完成，会明确
报告缺少结果文件。

实时查看指定 DSL 的进度：

```bash
tail -f /data/atrex/causal-conv1d/dsls/triton/bootstrap.log
tail -f /data/atrex/causal-conv1d/dsls/triton/campaign.log
```

需要下钻到 Attempt、Session 或每次评测时，先读取该 DSL 的 Campaign ID：

```bash
TASK=/data/atrex/causal-conv1d
DSL=triton
CAMPAIGN_ID="$(python3 -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["campaign_id"])' \
  "$TASK/dsls/$DSL/bootstrap-result.json")"

atrex-kernel-agent-runtime list-attempts \
  --config "$TASK/runtime.json" --campaign "$CAMPAIGN_ID" --format table
atrex-kernel-agent-runtime list-worker-sessions \
  --config "$TASK/runtime.json" --campaign "$CAMPAIGN_ID" --format table
```

从表格复制 `attempt_xxx` 后，可继续使用 `show-attempt`、`list-evaluations`；从 Session 表复制
`wsession_xxx` 后使用 `show-worker-session`。这些命令只读，不要求暂停 Runtime。

`run.sh` 在结束后保留服务进程，便于继续运行或检查；需要显式执行 `services.sh stop`。

## 常驻服务与多个任务

`services.sh start` 会在目标目录尚不存在时自动初始化一个不绑定算子和 Backend 的常驻
控制面 Workspace，然后启动 Runtime 和 Wiki：

```bash
bash scripts/production/services.sh start \
  --workspace /data/atrex/control \
  --hardware-target L20N \
  --env-file /secure/atrex-production.env
```

如果 `env-file` 已提供 `AGATE_GPU`，可以省略 `--hardware-target`。重复执行 `start` 会复用
已有控制面配置、数据库和密钥，不会重新初始化。

随后使用 `campaign.sh start` 后台运行不同算子和 Backend。这个入口只执行三个 DSL 的
Bootstrap 和 Campaign，不启动、重启或关闭任何服务：

```bash
bash scripts/production/campaign.sh start \
  --service-workspace /data/atrex/control \
  --workspace /data/atrex/tasks/flash-attention-claude \
  --kernel production_qwen35_35b_inhouse_4k256/flash_attention \
  --backend claude \
  --target-epoch 10 \
  --env-file /secure/atrex-production.env
```

任务启动后可以独立查看、停止或使用原启动参数重新启动：

```bash
bash scripts/production/campaign.sh status  --workspace /data/atrex/tasks/flash-attention-claude
bash scripts/production/campaign.sh stop    --workspace /data/atrex/tasks/flash-attention-claude
bash scripts/production/campaign.sh restart --workspace /data/atrex/tasks/flash-attention-claude
```

`status` 会区分 `starting`、`running`、`succeeded`、`failed` 和 `stopped`；总日志位于
`campaign-run/runner.log`。`stop` 会终止引用该任务 Workspace 的全部 Bootstrap、Campaign、
Sandbox 与 Agent 子进程，但不会影响共享 Runtime/Wiki。`restart` 复用持久化的精确启动参数和
绝对目标 Epoch；若要修改目标 Epoch，请先 `stop`，再用新的完整 `start` 命令启动。

任务 Workspace 拥有独立的配置、Bootstrap 输入、结果、日志及 Agent 沙箱目录；它与常驻
控制面共享 Registry、Gateway/Agate Job 数据库、Artifact Store、Capability 签名密钥和
Wiki。`campaign.sh` 启动前会检查 Runtime `/healthz` 与 Wiki `/readyz`，任一服务不可用时
直接退出，不会隐式拉起服务。同一算子和 Backend 重复使用同一任务 Workspace 时，会从
Registry 中已有的 Bootstrap、Epoch 和 Attempt 状态继续。

共享控制面模式下，任务 `runtime.json` 中的 Registry、Gateway/Agate Job Database 与 Artifact
Store 指向 Service Workspace；任务自身的 `state/` 只保存 Attempt、Evolution、Generalization 和
Bootstrap Worker Workspace。检查任务时始终向 `inspect.sh` 传任务 Workspace，而不是 Service
Workspace。
