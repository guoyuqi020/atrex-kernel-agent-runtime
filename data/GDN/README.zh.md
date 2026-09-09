# GDN 源码树优化输入包

[English](README.md) | 中文

目标 GPU：**L20D**。算子：`chunk_gated_delta_rule`。仅创建 **CuteDSL** Lineage。
**Optimizer 和 Evolver 均使用 Claude backend，复用 Lima 用户的 `.claude` 配置。**
这是供后续试跑使用的独立输入包；准备脚本不启动 Runtime、Wiki、模型或 GPU 作业。

本源码树任务的 `campaign.optimizer.max_session_tokens` 为 **100,000,000（100M）/ Session**，
覆盖 Bootstrap，以及所有消融臂的每一次 Active/Challenger Attempt；不是整个 Epoch 或
Campaign 共用 100M。Evolver 仍不限 token，原有超时限制不变。新启动的 Campaign/Bootstrap
进程会读取此配置；已运行的进程不会热更新配额，修改配额也不会自动重试已失败的 Attempt。
单文件任务配置保持不变。

## 消融实验入口

在已准备本包、Runtime/Wiki 正常运行的 Lima 中执行：

```bash
cd ~/atrex-runtime
source ~/.venvs/atrex-runtime/bin/activate
source env.sh
python data/GDN/run.py ablation
```

只启动任务，不管理服务；需与原 `campaign` 入口相同的 Sandbox 调度权限。
使用 `ablation-campaign.json` 的新 key `gdn-source-tree-l20d-claude-ablation`，
不会修改或接管当前试跑。一次完整 Bootstrap 后，六个对照臂共享新实验的冻结 v0、Agent、
修改边界、评测契约和初始证据；不重复 Baseline 测量，不导入旧试跑经验。

`ablation.json` 与单文件生产使用相同的计划生成器，默认每臂 5 个 Epoch：

| Campaign 实例 | 每 Epoch 的结构 | Optimizer Attempts | 保留 State | Evolution |
|---|---|---:|---|---:|
| `evolve-3` | Active + Challenger，各 1 条轨迹 × 3 次 | 30 | 是 | 4 |
| `ablation-isolated-01/02` | 两个独立实例，各 1 条轨迹 × 3 次 | 各 15 | 否 | 0 |
| `ablation-retained-01/02` | 两个独立实例，各 1 条轨迹 × 3 次 | 各 15 | 是 | 0 |
| `ablation-pool-3` | 同一 Active Branch，2 条轨迹 × 3 次 | 30 | 否 | 0 |
| `ablation-pool-retained-3` | 同一 Active Branch，2 条轨迹 × 3 次 | 30 | 是 | 0 |

共 **7 个 Campaign、150 次 Optimizer Attempt**，不含 Bootstrap 和 Evolver。
消融主臂首轮使用同一 Agent 的两份独立副本，第二轮起才调用 Evolver；原 `run.py campaign`
仍保留首轮仅 Active 的策略，二者不是同一个实验。这里不启动外部原版 AKA 对照。
重置 State 不会删除 Kernel 进展或 Runtime Journal；Pool 在 Epoch 边界共享最佳 Kernel，
Pool-Retained 还继承该轨迹的终态 State，不做合并。不同臂不共享后续历史或可写文件。

输出在 `data/GDN/ablation/`：Bootstrap、冻结输入、`campaign-results.json` 汇总，以及每臂
同名目录中的 `campaign-result.json` / `campaign.log`；对照臂还保存 seed 定义和结果。
汇总含各臂 Campaign/Lineage ID，可供 inspect 使用。Session/Artifact 仍在共享的 `data/GDN/state/`。
Attempt 进度实时写入各臂日志，臂完成时打印时间戳。失败不取消其他臂；重复运行复用身份并恢复，
完成后只报告结果。`--target-epoch` 只改变主臂目标，对照臂固定每轨迹 15 次 Attempt。
修改冻结输入需新 Workspace 和 creation key。

默认最多并行 10 个 Optimizer Worker；Lima 资源有限，建议结束旧试跑后再显式启动。
添加配置不会自动启动、停止或重启任何任务。

## 内容与来源

- `task/`：原始适配器、Source Manifest、Torch Reference、输入生成器、公开 Shape Train、
  10 个私有测试 Shape、Metadata 和 Roofline，逐字节保留原文件。
- `source.bundle`：初始源码的离线 Git Bundle，不依赖外部 GDN 目录或联网拉取。
- `source/`：准备脚本从 Bundle 还原的只读用途 seed Git 仓库，不要在这里进行优化。
- `campaign.json`：固定 L20D、初始源码及 Optimizer commit、每 Epoch 的优化调度。
- `runtime.template.json`：本任务自己的 Runtime 配置模板，不交叉引用其他 example。
- `runtime.json`、`evaluation-contract.json`：在 Lima 生成的可用配置及封存输入契约。
- `prepared.json`：文件 SHA-256、固定 commit 和本地校验结果。
- `initial-evidence/`：初始来源说明，不包含历史优化经验。

原始资料来自 `GDN_AKA_REPRO_20260907` 中的：

- `gdn_fi_initial_seed/task/`、`gdn_fi_initial_seed/source/`。
- `atrex-bench/data/aka/gdn_prefill_sm103_m64_20260904/chunk_gated_delta_rule/`。

初始 seed commit：`a39405536f178689d7f60b551c17b2252bcee61d`；其 FlashInfer 上游 commit：
`2ab910c58fdd2392914ea05e2a8714946ac0eef6`。不包含原实验的优化结果、`solution.py` 或私有
M64 实现。原 Metadata 与 Roofline 已使用 `NVIDIA L20D`，未将其他 GPU 的数据改名复用。

## 在 Lima 准备

从宿主机进入虚拟机，再使用 **Linux 的 venv**，不要使用共享目录里的 macOS `.venv`：

```bash
limactl shell ubuntu
cd ~/atrex-runtime
source ~/.venvs/atrex-runtime/bin/activate
python data/GDN/prepare.py --backend claude
```

脚本使用当前非 root Linux 用户作为 Sandbox Worker，并从该用户 Home 复用 CLI 配置；
也可通过 `--worker-user` 指定其他已存在的非 root 用户。保留 bwrap + cgroup 隔离。
Runtime 服务和准备脚本使用 Linux venv；Sandbox 中的 Optimizer、Evolver 和 Runtime Tools
使用全局 Python（将 venv 的解释器软链接解析为 `/usr/bin/python3.x`），不依赖被隐藏的
Home 下的 venv 路径。

准备过程会检查源码锁定、可修改范围、Production Policy、公开/私有 Shape 契约及三个固定
仓库 commit。它不会验证远端 GPU 镜像或模型连通性，这些仍需后续实跑确认。
生成文件和 `source/` 不纳入版本控制；复制或 clone 本仓库后可用 Bundle 再生成。
如果已经产生 `state/`，脚本拒绝覆盖不同的运行配置，避免改变进行中的 Campaign。

## 后续启动

配置预留 Runtime 地址 `http://127.0.0.1:8766`、Wiki 地址 `http://127.0.0.1:8091`。
Wiki 是独立服务；本包不会自动启动它。状态、Session 和 Artifact 写入 `data/GDN/state/`。
Agate 使用 `AGATE_AK` / `AGATE_SK`，`AGATE_URL` 可在准备时覆盖服务地址；不将凭据复制进输入包。
Runtime 服务与 Campaign 进程还需使用一致的 `ATREX_CAPABILITY_SIGNING_KEY` 和
`ATREX_ADMIN_BEARER_TOKEN`。所选模型 CLI 必须已经安装并配置好认证。

也可以使用 `run.py`：在已加载 Agate 环境变量、具备下述 Sandbox 调度权限的 Linux 进程中，
分别运行 `python data/GDN/run.py serve` 和
`python data/GDN/run.py campaign --target-epoch 5`。它自动共用持久化的
`runtime-secrets.json`（权限 0600，Git 忽略），按顺序执行 Bootstrap 和指定 Epoch，
将结果写入 `bootstrap-result.json` / `epoch-result.json`；Bootstrap 失败就停止，不启动优化。
恢复时继续使用原配置及 secrets，不要另起一个并行的同 Campaign 调度器。

本次 Lima 试跑使用 `atrex-gdn-runtime` 和 `atrex-gdn-campaign` 两个 systemd 单元，
日志在 `services/runtime.log` / `services/campaign.log`。查看状态可用：

```bash
systemctl status atrex-gdn-runtime atrex-gdn-campaign --no-pager
sudo tail -n 60 data/GDN/services/campaign.log
```

以下为后续试跑的 CLI 入口，**本次准备没有执行**。Sandbox 调度须在具备 systemd 系统服务
管理及 Worker 切换权限的 Linux 环境中执行（与生产脚本的 root 调度方式一致）：

```bash
# 服务进程；另一个终端执行 Bootstrap / Campaign。
atrex-kernel-agent-runtime serve --config data/GDN/runtime.json

# 启动完整 Claude Bootstrap Session，在 seed 上验证/修复，再由 Runtime 终评注册 v0。
atrex-kernel-agent-runtime bootstrap \
  --config data/GDN/runtime.json --campaign data/GDN/campaign.json

# 将返回的 campaign_id 填入下面的位置；运行到 Epoch 5。
atrex-kernel-agent-runtime run-campaign \
  --config data/GDN/runtime.json \
  --campaign CAMPAIGN_ID_FROM_BOOTSTRAP --target-epoch 5
```

Bootstrap 让 Claude 先评测原始源码，必要时只修复可修改的文件，记录 Direction/Experiment
Journal 并提交标准报告，再由 Runtime 独立终评。Campaign key 使用
`gdn-source-tree-l20d-claude-bootstrap`，与之前的无模型试跑分开；旧 v0 与 Session 记录保留。
之后再次启动相同 key 时，正常复用新流程产生的 baseline。

默认总共运行 5 个 Epoch，每个分支每 Epoch 串行 3 个 Attempt、1 条 Trajectory；Epoch 1
只有 Active，Epoch 2 起加入 1 个 Challenger（Active 和 Challenger 每轮各跑 3 个 Attempt）。
Epoch 总数和每条 Trajectory 的 Attempt 数与单文件生产默认值一致；保留原来的首轮仅 Active 策略。
目标是绝对轮次：完成 Epoch 5 后重复执行只报告已有结果，不额外增加 5 轮，也不会为不运行的
Epoch 6 再触发 Evolver。
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
