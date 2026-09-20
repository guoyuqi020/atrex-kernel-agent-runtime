---
name: atrex-gdn-launch
description: 在 Lima Ubuntu 中准备、启动或恢复本次 GDN 多文件源码树优化与十二臂消融实验（L20D、CuteDSL、Claude）。用于“启动 GDN 源码树优化”“跑 GDN 消融”“恢复 GDN 实验”；仅询问运行状态时使用状态检查流程，不启动任务。
---

# GDN 源码树优化启动

完成用户要求的准备/启动/恢复，并确认实际进入的阶段。创建 Skill、查看命令、检查配置
不等于授权运行模型或 GPU 作业。用户明确要求启动时，按本流程执行；不要重复索取相同授权。
先简要告知要启动哪个实验、是否复用服务以及是否存在其他任务。

## 定位当前实现

默认宿主仓库为 `~/atrex-runtime`，Lima instance 为 `ubuntu`，虚拟机中同样为
`~/atrex-runtime`。先验证路径，不把宿主 `/Users/...` 路径直接用在 Linux。
用户指定其他部署时，以实际路径为准。

在改配置或启动前，读取仓库里的：

- 所选输入包的 `README.zh.md`、`scripts/gdn/run.py`、`scripts/gdn/prepare.py`；
- 所选输入包的 `ablation-campaign.json`、`ablation.json`；恢复时以工作区冻结副本为准；
- `scripts/source-tree/run.py`；需要核对消融逻辑时读取
  `src/atrex_runtime/ablation_plan.py` 和 `scripts/production/policy.json`。

共享服务模式还需读取 `scripts/gdn/README.md` 和 `scripts/gdn/gdn_workspace.py`。
这些文件是配置与命令的依据；不要另造一套调度脚本或复制旧的 PID/Job ID/Commit。
实际启动和服务管理前，读取 [Lima 启动细节](references/lima-launch.md)。

## 输入包和工作区

- `data/GDN`：清理过优化提示的输入；准备入口默认使用这一份。
- `data/GDN-full`：恢复原始提示的对照输入，使用 `prepare.py --inputs data/GDN-full`。
  这不表示公开隐藏用例，也不改变 Agent 现有的 Prompt 投影规则。
- `data` 只存输入。准备时通过 `--workspace` 选择输出目录；省略时默认
  `workspaces/<输入目录名>`。`run.py` 不接收 `--inputs`，必须传入对应 `--workspace`。
  特别是 GDN-full，不能省略运行命令的 `--workspace workspaces/GDN-full`，
  否则会进入默认的 GDN 工作区。
- GDN/GDN-full 并跑优先使用共享模式：一次 `prepare.py --services-only --workspace SERVICE`，
  两份新任务分别 `prepare.py --inputs INPUTS --workspace TASK --service-workspace SERVICE`。
  只对 SERVICE 运行 `serve`；对 TASK 运行 `campaign`/`ablation`，自动读取冻结的
  `service-binding.json`。不把两个任务目录里的配置/密钥复制拼接成共享服务。
- 共享模式的 Registry、Session、Artifact 和 secrets 位于服务目录；任务目录保留输入和结果。
  独立模式仍把二者放在同一工作区。恢复用原快照，不自动迁移旧数据库或改绑；
  两份模板默认 Runtime 8766，仅新独立服务可用 `--port` 选择空闲端口。
- 新配置默认 `container`：保留 bwrap，以当前容器用户执行，不切换 Worker，不为 Session
  创建 systemd/cgroup。资源限制由外层容器提供；直接在 Lima 运行只有 VM 总体限制。
  这不是自动创建 Docker 容器。不要 sudo 启动新任务；旧 sandbox 工作区按冻结模式恢复。

## 识别实验，而不是只数分支

十二臂消融入口是 `python scripts/gdn/run.py ablation --workspace WORKSPACE`：

- L20D；`chunk_gated_delta_rule`；只跑 CuteDSL；Optimizer/Evolver 都为 Claude backend。
  实际模型名称来自配置/环境，不把 CLI 名称当成模型名称。
- 12 个 Campaign：`evolve-3`、`ablation-isolated-01/02`、
  `ablation-isolated-evolve-01/02`、`ablation-retained-evolve-01/02`、`ablation-retained-01/02`、
  `ablation-isolated-pool-evolve-3`、`ablation-pool-3`、`ablation-pool-retained-3`。
- 默认 100 个 Epoch，每轨迹每轮 3 次 Attempt。主臂及两个 Pool 各 600 次；
  八个独立对照各 300 次，Isolated-Pool-Evolve 为 1,200 次，共 5,400 次 Optimizer Attempt，
  不含 Bootstrap/Evolver。
- 主臂首轮是同一 Agent 的两个独立分支，之后比较 Active 与进化 Challenger。两个
  Isolated-Evolve 与 Retained-Evolve 首轮只运行 Active 副本，Epoch 2–100 只运行进化
  Challenger，不运行同轮 Active；每臂 99 次 Evolution，前者每个 Attempt 前重置 State，后者
  在三个串行 Attempt 间继承 State。
  Isolated-Pool-Evolve 同时运行 Active/Challenger，每边两条 Trajectory，但每个 Attempt 前重置
  Optimizer 产出的自适应 State。
  Active/Challenger、Trajectory、Direction 都不是独立消融臂。
- 只做一次完整 Bootstrap，再由 `seed-ablation-arm` 派生十一个对照：
  共享冻结 v0、初始 Agent/证据与评测契约，不导入之后的跨臂历史。
- `ablation` 使用自己的 creation key 与 `workspaces/GDN/ablation/` 输出，不接管旧试跑。
  `run.py campaign` 是单 Campaign 入口，也用于恢复旧试跑；用户要求单路线时使用它，
  不强制启动十二臂消融。以下数量描述十二臂模式，不适用于单 Campaign。

不要把 `--target-epoch 100` 解释为“再追加一百轮”。它是绝对目标，且在消融入口中只控制主臂；
新计划的对照臂固定每轨迹 300 次 Attempt。已有工作区保留其冻结计划；恢复旧 5 轮实验时，
显式传入 `--target-epoch 5`，不要用新默认值意外延长旧主臂。完成后重跑应复用结果，不扩展预算。

## 启动前检查

1. 在 Lima 内使用已安装 Runtime 的 **Linux venv**，当前惯例
   `~/.venvs/atrex-runtime`。检查 `python`、`atrex-kernel-agent-runtime`、
   `claude`、`bwrap` 和相关 Python 导入；只在旧 sandbox 模式检查 `systemd-run`。
   共享仓库里的 macOS `.venv` 不可用。
2. 若所选工作区既没有 `runtime.json` 也没有 `service-binding.json`，按 README 用当前容器用户执行
   `python scripts/gdn/prepare.py --inputs INPUTS --workspace WORKSPACE --backend claude`。
   已有配置与状态时先检查，
   不靠重新 prepare 覆盖；配置漂移应报告，而非清理 state 来绕过。
   `data/GDN` 只读作输入；实际使用工作区内的冻结 task/Campaign/config。
   使用更新后的输入时准备新的任务 `--workspace`；共享模式的 serve 必须指向绑定的服务工作区。
   独立模式三个入口使用同一路径。旧试跑已迁移到 `workspaces/GDN`，仍保留旧 seed 和每轮 1 次 Attempt；
   不把它当作已应用当前 data 模板的新实验。
3. 核对 L20D、CuteDSL、Claude、source manifest 的固定 seed、可改目录和 adapter，
   核对两个 Agent 仓库的声明 Commit 确实存在。未提交的源码不会进入 commit-anchored
   Agent Bundle；启动前说明这种差异，不擅自 commit/push、改 Commit 或假称已用最新代码。
4. 从受信的 `env.sh` 加载 Agate/模型环境。仅检查变量是否存在，不显示值。
   `run.py` 自动共享 `runtime-secrets.json`；不要打印、复制进 Agent、重新生成或替换已有 key。
5. 检查 Runtime 的 `/healthz`，默认端口为 8766，但以运行配置为准。
   KDA 不再使用 Wiki，新模板通过 `gpu_wiki: null` 禁用；不要启动 Wiki、下载语料或检查其就绪状态。
   旧工作区保留冻结配置，不为取消 Wiki 自动改写配置摘要或重启共享服务。
   HTTP 200 不足以证明是同一部署，要核对进程/服务及配置路径。
   复用健康服务；端口被其他进程占用时不杀进程抢占。
6. 检查现有 GDN Campaign/ablation runner、Workspace lock、`free -h`、
   `nproc` 及外层容器资源限制；仅旧 sandbox 模式要求 systemd/cgroup v2。十二臂最多并行 18 个 Optimizer；
   两份输入各跑十二臂会产生 24 个 Campaign、最多约 36 个 Optimizer，当前并发限制不是全局配额。
   container 模式没有每 Session 的 memory_max。不要声称 8 GiB Lima 必然足够。
   有旧任务占资源且用户未授权停止/并跑时，报告情况并取得运行安排。
7. 默认 `container` 模式不要求系统级调度权限；bwrap/namespace 不可用时停止并报告，
   不自动取消隔离。旧 `sandbox` 模式的调度进程仍需要系统级 systemd/Worker 切换权限。
   Worker 本身仍为配置指定的非 root 用户；不要用 root 当 Worker，或关闭 bwrap/cgroup
   来绕过权限错误。

## 启动与恢复

- 健康 Runtime 已存在时，只启动新的 ablation runner，不重启服务。
- 默认 container 模式直接运行 `run.py`，长任务交给部署已有的进程管理器/持久会话，
  不要求 systemd 或 sudo。旧 sandbox 服务按原方式恢复，不改其服务命令。
  具体命令见启动细节；先检查实际进程，避免重复启动。
- 已有消融输出时，检查 `launch-inputs.json`、各臂结果与实际运行进程。
  没有活跃 runner 才用相同入口恢复；不要删除 lock 文件、修改 SQLite fence，
  或另起不同 creation key 假装恢复。过期租约先核实持有者和期限。
- 一个臂失败不会取消其他臂。先报告失败原因及仍运行的臂；
  不自动反复重启整个实验，也不在其他臂仍由原 runner 管理时启动第二个 runner。
- 若用户明确要求停止，只处理已确认属于该实验的单元/子进程；
  不顺带停止共享 Runtime、其他任务的 Wiki、旧试跑或删除工作区。

## 启动后验收与交付

不要把进程管理器接受命令等同于实验已成功启动。确认：

- 目标 runner 活跃且无立即退出，Runtime 仍健康；
- `ablation/bootstrap.log` 出现进展，或已产生 `bootstrap-result.json`；
- 进入 fan-out 后，各臂 seed 身份独立、`campaign.log` 更新。
  Bootstrap 仍在进行时可以交付“已启动，正在 Bootstrap”，不用等待整个实验。

主要结果位置：

以下以 `workspaces/GDN` 为例；使用 GDN-full 或自定义工作区时替换此前缀。

- `workspaces/GDN/ablation/launch-inputs.json`、`bootstrap-result.json`；
- `workspaces/GDN/ablation/campaign-results.json`：十二臂 Campaign/Lineage ID、状态和路径；
- `workspaces/GDN/ablation/<arm>/campaign.log`、`campaign-result.json`；
- 原始 Session/Artifact 按实际 Runtime config 查找：共享模式位于 SERVICE/state，
  独立模式默认 `workspaces/GDN/state/`，不是各臂日志目录。

用汇总里的 ID 调用当前 CLI inspect（先看 `--help`），不要手工重建评测或触发新 GPU 测试。
系统记录的旧 `running` 状态可能没有实际进程，应交叉验证。若使用已安装的
`atrex-production-status` Skill，先读其说明；它对自定义 GDN runner 的进程发现可能不完整，
且共享 Registry 可能含旧试跑，需按本次汇总 ID 过滤。

最终用中文报告：是否启动、十二臂/旧试跑哪个模式、GPU/DSL/backend、当前阶段、复用的服务、
日志/结果入口，以及任何未解决的资源/权限/配置问题。后续持续监控只有用户要求时才配置。
