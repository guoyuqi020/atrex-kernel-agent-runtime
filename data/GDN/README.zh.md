# GDN 源码树优化输入包

[English](README.md) | 中文

同时运行 GDN/GDN-full 请使用[共享服务入口](../../scripts/gdn/README.md)：
先准备一个 `--services-only` 工作区，再以 `--service-workspace` 分别绑定两份任务，只启动一次
Runtime。下文路径和 serve 命令描述独立模式；共享模式的 Registry、Session、Artifact 和密钥
都位于服务工作区，任务目录只保留输入快照、绑定和结果。

保留原始优化提示的独立对照输入包见 [GDN-full](../GDN-full/README.zh.md)。

目标 GPU：**L20D**。算子：`chunk_gated_delta_rule`。仅创建 **CuteDSL** Lineage。
**Optimizer 和 Evolver 均使用 Claude backend，复用 Lima 用户的 `.claude` 配置。**
这是供后续试跑使用的独立输入包；准备脚本不启动 Runtime、模型或 GPU 作业。
KDA 不再使用 Wiki，本包已禁用 Wiki，不需要启动服务或准备语料。

`data/GDN/` 只存任务输入和配置模板；启动脚本位于 `scripts/gdn/`，默认工作区为
`workspaces/GDN/`。准备阶段会把任务、Campaign 定义及初始证据复制为工作区快照，
并在工作区还原 seed。生成配置、凭据、数据库、Session、日志和结果均不写入 `data/`。
两个脚本都支持 `--workspace workspaces/GDN-clean`；准备、服务和运行必须指定同一工作区。
已注册的实验保留原输入快照，修改 `data/GDN/` 不会影响它。要使用新输入，请准备新工作区，
不要通过重新 prepare 覆盖已有实验的任务定义。

本源码树任务的 `campaign.optimizer.max_session_tokens` 为 **100,000,000（100M）/ Session**，
覆盖 Bootstrap，以及所有消融臂的每一次 Active/Challenger Attempt；不是整个 Epoch 或
Campaign 共用 100M。Evolver 仍不限 token，原有超时限制不变。新启动的 Campaign/Bootstrap
进程会读取此配置；已运行的进程不会热更新配额，修改配额也不会自动重试已失败的 Attempt。
单文件任务配置保持不变。

## 消融实验入口

在已准备本包、Runtime 正常运行的 Lima 中执行：

```bash
cd ~/atrex-runtime
source ~/.venvs/atrex-runtime/bin/activate
source env.sh
python scripts/gdn/run.py ablation
```

只启动任务，不管理服务；以与 `campaign` 相同的容器用户运行，不需要 sudo/systemd。
使用 `ablation-campaign.json` 的新 key `gdn-source-tree-l20d-claude-ablation`，
不会修改或接管当前试跑。一次完整 Bootstrap 后，六个对照臂共享新实验的冻结 v0、Agent、
修改边界、评测契约和初始证据；不重复 Baseline 测量，不导入旧试跑经验。

`ablation.json` 与单文件生产使用相同的计划生成器，默认每臂 100 个 Epoch：

| Campaign 实例 | 每 Epoch 的结构 | Optimizer Attempts | 保留 State | Evolution |
|---|---|---:|---|---:|
| `evolve-3` | Active + Challenger，各 1 条轨迹 × 3 次 | 600 | 是 | 99 |
| `ablation-isolated-01/02` | 两个独立实例，各 1 条轨迹 × 3 次 | 各 300 | 否 | 0 |
| `ablation-retained-01/02` | 两个独立实例，各 1 条轨迹 × 3 次 | 各 300 | 是 | 0 |
| `ablation-pool-3` | 同一 Active Branch，2 条轨迹 × 3 次 | 600 | 否 | 0 |
| `ablation-pool-retained-3` | 同一 Active Branch，2 条轨迹 × 3 次 | 600 | 是 | 0 |

共 **7 个 Campaign、3,000 次 Optimizer Attempt**，不含 Bootstrap 和 Evolver。
消融主臂首轮使用同一 Agent 的两份独立副本，第二轮起才调用 Evolver；原 `run.py campaign`
仍保留首轮仅 Active 的策略，二者不是同一个实验。这里不启动外部原版 AKA 对照。
重置 State 不会删除 Kernel 进展或 Runtime Journal；Pool 在 Epoch 边界共享最佳 Kernel，
Pool-Retained 还继承该轨迹的终态 State，不做合并。不同臂不共享后续历史或可写文件。

输出在 `workspaces/GDN/ablation/`：Bootstrap、冻结输入、`campaign-results.json` 汇总，以及每臂
同名目录中的 `campaign-result.json` / `campaign.log`；对照臂还保存 seed 定义和结果。
汇总含各臂 Campaign/Lineage ID，可供 inspect 使用。Session/Artifact 在共享的 `workspaces/GDN/state/`。
Attempt 进度实时写入各臂日志，臂完成时打印时间戳。失败不取消其他臂；重复运行复用身份并恢复，
完成后只报告结果。`--target-epoch` 只改变主臂目标，新计划的对照臂固定每轨迹 300 次 Attempt。
已有工作区仍保留冻结的对照臂预算；恢复旧 5 轮实验且不扩展主臂时，显式传入 `--target-epoch 5`。
修改冻结输入需新 Workspace 和 creation key。

默认最多并行 10 个 Optimizer Worker；Lima 资源有限，建议结束旧试跑后再显式启动。
添加配置不会自动启动、停止或重启任何任务。

## 内容与来源

- `task/`：导入的适配器、Source Manifest、Torch Reference、输入生成器、公开 Shape Train、
  10 个私有测试 Shape、Metadata 和 Roofline。公开目标与证据说明已调整，Source Manifest
  固定到更新来源说明后的 seed 版本。
- `source.bundle`：初始源码的离线 Git Bundle，不依赖外部 GDN 目录或联网拉取。
- `campaign.json`：固定 L20D、初始源码及 Optimizer commit、每 Epoch 的优化调度。
- `runtime.template.json`：本任务自己的 Runtime 配置模板，不交叉引用其他 example。
- `initial-evidence/`：初始来源说明，不包含历史优化经验。

工作区保存 `task/` 与 `initial-evidence/` 快照、还原的 `source/`（不要直接在此优化）、
生成的 `runtime.json` / `evaluation-contract.json`、Campaign 定义，以及记录文件哈希、
固定 commit 和本地校验结果的 `prepared.json`。

两个 Campaign 定义均固定 KDA commit `41af4a45ca4155254f3c2e8d501ae28a5fb5bb62`。
此版本不包含 KernelWiki 和 ncu-report-skill，构建无需初始化这两个 Skill 子模块；
Runtime 模板的 `allowed_submodules` 为空。新工作区使用此版本，已有工作区仍使用冻结的
Agent revision。本地未提交的 KDA 修改不会进入 Bundle。

Evaluator 与 Roofline 固定 Atrex Bench `54925ff9223aa54b901219f02fffd51d9af82e3c`，
来自子模块的 `yuxiao_dev` 分支。准备阶段使用真实加载器验证 Optimizer 导出、Evolver 拉取封存、
Evaluator 拉取导出，以及 Roofline 拉取导出和入口文件；不再只用 `git cat-file` 判断 commit 存在。
两个 Campaign 定义都参与校验，版本和 Bundle Digest 记录在 `prepared.json.source_preflight`。
失败时不发布 Runtime 配置、不启动 Agent/GPU 作业。预检不执行 Roofline 生成器，
也不代替 GPU 镜像及模型连通性测试；更新输入 pin 不会改写已有工作区的冻结配置。

原始资料来自 `GDN_AKA_REPRO_20260907` 中的：

- `gdn_fi_initial_seed/task/`、`gdn_fi_initial_seed/source/`。
- `atrex-bench/data/aka/gdn_prefill_sm103_m64_20260904/chunk_gated_delta_rule/`。

当前 seed commit：`60c83174e82e4566e6ee360fd38b85c5bb0794b6`，基于原 seed
`a39405536f178689d7f60b551c17b2252bcee61d` 与 FlashInfer 上游 commit
`2ab910c58fdd2392914ea05e2a8714946ac0eef6`。本次 seed 更新仅修改
`UPSTREAM_PROVENANCE.json`，去掉实现名称提示，Kernel 源码字节保持不变。
原 Metadata 与 Roofline 已使用 `NVIDIA L20D`，未将其他 GPU 的数据改名复用。
已有 Campaign 仍使用其冻结的 seed 和任务输入。

## 在 Lima 准备

从宿主机进入虚拟机，再使用 **Linux 的 venv**，不要使用共享目录里的 macOS `.venv`：

```bash
limactl shell ubuntu
cd ~/atrex-runtime
source ~/.venvs/atrex-runtime/bin/activate
python scripts/gdn/prepare.py --backend claude
```

新配置默认 `container`：以当前容器用户运行 bwrap，从该用户 Home 复用 CLI 配置；
建议非 root，`--worker-user` 不能切换到其他用户。不需要 systemd 或每 Session cgroup；
CPU/内存/PID 限额由外层容器配置，直接运行在 Lima 时只受 VM 总体限制。
Runtime 服务和准备脚本使用 Linux venv；Sandbox 中的 Optimizer、Evolver 和 Runtime Tools
使用全局 Python（将 venv 的解释器软链接解析为 `/usr/bin/python3.x`），不依赖被隐藏的
Home 下的 venv 路径。

准备过程会检查源码锁定、可修改范围、Production Policy、公开/私有 Shape 契约，
以及上述四种 Bundle 的实际加载路径。它不会验证远端 GPU 镜像或模型连通性，这些仍需后续实跑确认。
生成文件和 `source/` 不纳入版本控制；复制或 clone 本仓库后可用 Bundle 再生成。
如果已经产生 `state/`，脚本拒绝覆盖不同的运行配置，避免改变进行中的 Campaign。

## 后续启动

配置预留 Runtime 地址 `http://127.0.0.1:8766`，通过 `gpu_wiki: null` 禁用 Wiki 集成。
无需启动或检查 Wiki。状态、Session 和 Artifact 写入 `workspaces/GDN/state/`。
Agate 使用 `AGATE_AK` / `AGATE_SK`，`AGATE_URL` 可在准备时覆盖服务地址；不将凭据复制进输入包。
Runtime 服务与 Campaign 进程还需使用一致的 `ATREX_CAPABILITY_SIGNING_KEY` 和
`ATREX_ADMIN_BEARER_TOKEN`。所选模型 CLI 必须已经安装并配置好认证。

也可以使用 `run.py`：在已加载 Agate 环境变量的 Linux 进程中，以同一容器用户、无需 sudo，
分别运行 `python scripts/gdn/run.py serve` 和
`python scripts/gdn/run.py campaign --target-epoch 100`。它自动共用持久化的
`runtime-secrets.json`（权限 0600，Git 忽略），按顺序执行 Bootstrap 和指定 Epoch，
将结果写入 `bootstrap-result.json` / `epoch-result.json`；Bootstrap 失败就停止，不启动优化。
恢复时继续使用原配置及 secrets，不要另起一个并行的同 Campaign 调度器。

如使用 systemd 管理，可将服务单元命名为 `atrex-gdn-runtime` 和 `atrex-gdn-campaign`，
并将日志重定向到 `workspaces/GDN/services/`。查看状态可用：

```bash
systemctl status atrex-gdn-runtime atrex-gdn-campaign --no-pager
sudo tail -n 60 workspaces/GDN/services/campaign.log
```

以下为后续试跑的 CLI 入口，**本次准备没有执行**。默认 container 模式需要 bwrap/namespace
可用，不需要 systemd 调度权限；已有 sandbox 工作区仍保留原先的 systemd/Worker 要求：

```bash
# 服务进程；另一个终端执行 Bootstrap / Campaign。
atrex-kernel-agent-runtime serve --config workspaces/GDN/runtime.json

# 启动完整 Claude Bootstrap Session，在 seed 上验证/修复，再由 Runtime 终评注册 v0。
atrex-kernel-agent-runtime bootstrap \
  --config workspaces/GDN/runtime.json --campaign workspaces/GDN/campaign.json

# 将返回的 campaign_id 填入下面的位置；运行到 Epoch 100。
atrex-kernel-agent-runtime run-campaign \
  --config workspaces/GDN/runtime.json \
  --campaign CAMPAIGN_ID_FROM_BOOTSTRAP --target-epoch 100
```

Bootstrap 让 Claude 先评测原始源码，必要时只修复可修改的文件，记录 Direction/Experiment
Journal 并提交标准报告，再由 Runtime 独立终评。Campaign key 使用
`gdn-source-tree-l20d-claude-bootstrap`，与之前的无模型试跑分开；旧 v0 与 Session 记录保留。
之后再次启动相同 key 时，正常复用新流程产生的 baseline。

默认总共运行 100 个 Epoch，每个分支每 Epoch 串行 3 个 Attempt、1 条 Trajectory；Epoch 1
只有 Active，Epoch 2 起加入 1 个 Challenger（Active 和 Challenger 每轮各跑 3 个 Attempt）。
每条 Trajectory 的 Attempt 数与单文件生产默认值一致；单文件的 Epoch 目标不变。
保留原来的首轮仅 Active 策略，单 Campaign 共 597 次 Optimizer Attempt。
目标是绝对轮次：完成 Epoch 100 后重复执行只报告已有结果，不额外增加 100 轮，也不会为不运行的
Epoch 101 再触发 Evolver。
每轮 Attempt 数在 Lineage 注册时冻结；修改本配置不会改变已有的每轮 1 次 Lineage。
新调度应使用新的 Campaign creation key，保留旧历史，不要直接修改 Registry，或把旧 Campaign
当成每轮 3 次继续运行。
Kernel Retention 和 Agent Promotion 使用同 Allocation ABBA，
开启 Production Gate 和锁频。Manifest 原有 `measurement` 等旧控制字段保留为来源信息，
实际运行以 `runtime.json` 的 Gate 与 Campaign 配置为准。

源码会直接物化到 Agent 的 `work/kernel/`，入口为不可修改的 `kernel.py`。仅允许修改
`flashinfer/gdn_kernels/blackwell/`；其他文件保持冻结。Agent 可以使用 Runtime Tools 的
Evaluate、Profile、Check、Disassemble，详见[源码树接口](../../docs/source-trees.zh.md)。

远端 L20D 环境须满足原任务声明的 SM103、`torch>=2.9.0`、`nvidia-cutlass-dsl>=4.4.2`。
Profile / Disassemble 需要 NCU，Sanitized Check 需要 Compute Sanitizer。
