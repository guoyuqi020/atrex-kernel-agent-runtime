# FA4 Prefill 多文件算子优化任务

[English](README.md) | 中文

这是从两份提供的资料组装的独立任务：Qwen3.8-Max Attention 的 Atrex-Bench 输入与评测器，以及原始 FA4 R0 源码。任务使用 **L20D、CuteDSL、Claude**；只创建一条 DSL Lineage。Wiki 关闭。这里仅存放任务输入、离线源码 Bundle 和配置模板；运行数据写入 `workspaces/FA4/`。

## 优化目标

生产接口是 `flashinfer.prefill.trtllm_batch_context_with_kv_cache`，但 Candidate 使用随任务提供的 FlashAttention CuTe 源码，而不是调用已安装的 FlashInfer。输入为 FP8 E4M3 Query/KV，输出为 BF16；16 个 Query Head、1 个 KV Head、Head Dim 256、64-token KV Page、因果 Attention。公开输入域来自 `shape_train.json`；全部 30 个精确评测 Shape 保持私有。

R0 源码固定到上游 FlashAttention Commit `b54df166ebb69b896892826014759d09b9c3c9c6`，并包含完整 CuTe 子树和 Quack 0.5.3 源码依赖。起始实现只支持 HD256 双 CTA 的 P128 分页路径。Bootstrap 应先补齐：两个独立映射的 P64 Page 组成 N128 Tile、实际 KV 长度 `seqused_k`、专用 PackGQA 支持；通过完整 Bootstrap Gate 后才注册 `v0`。不能把 R0 当成已经正确的 baseline，也不能在固定适配器里绕过这些能力缺口。

Optimizer 可以修改 `vendor/flash_attention/flash_attn/cute/` 内的文件；`kernel.py` 适配器、Quack 支持及其他源码由 Runtime 哈希锁定。初始 Evidence 包含任务说明，不包含其他运行的优化结果、Journal 或 Conversation。

这是本任务采用的保守封装策略：原始资料明确允许修改完整 CuTe 树，并禁止在 Adapter 中绕过能力缺口，但没有明确规定 Adapter 与 `vendor_support` 必须全程只读。“固定起始快照”不等同于“优化期间不可修改”；这里的只读范围由 Runtime Manifest 显式决定。

## 内容

```text
data/FA4/
├── task/
│   ├── adapter.py                 # 固定适配器，进入 Candidate 时命名为 kernel.py
│   ├── source_manifest.json        # 源码 Commit、可编辑范围和 GPU 环境依赖
│   ├── reference.py
│   ├── input.py
│   ├── shape_train.json
│   ├── shape_valid.json            # 私有
│   ├── metadata.json               # 私有，包含逐输出正确性策略
│   └── roofline.json               # 私有，复用提供的 L20D 数据
├── smoke/                          # 原始 Agate 冒烟脚本与旧 Shape 文档，逐字保留
├── source.bundle                   # 原始 FA4 / Quack 离线源码
├── evaluator.bundle                # 专用 Atrex-Bench 评测器离线源码
├── source-provenance.json
├── source-validation.json          # 提供资料中的历史冒烟验证，不是本任务的验收结果
├── asset-integrity.json            # 按提供资料核对的文件及源码 SHA256
├── initial-evidence/
├── campaign.json
└── runtime.template.json
```

离线封装的 Source Commit 是 `7b077cf391a98a06ad9464530f0a4c9c7be3f477`；它与上游 Commit 不同，因为封装加入了 Quack 依赖和来源记录，但没有修改原始 FA4 源码。Evaluator Commit 是 `ed449b63ecd8aeff4db23be0d9f658d7d50b1cfa`，来自提供的新版评测器，不使用 runtime 当前的 `third_party/atrex-bench`。准备时会验证文件 SHA256、两个 Commit，以及封装的全部 70 个 Source 和 32 个 Evaluator 文件，并实际加载封存的 Optimizer、Evolver 和 Evaluator。详细核对与评测策略差异见 [资料对齐记录](ALIGNMENT.zh.md)。

## 评测策略和限制

- 返回值及被修改的 `out` 都逐元素检查：`abs(candidate-reference) <= 0.06 + 0.04 * abs(reference)`。Evaluator 从 Metadata 加载策略，不允许通过旧的 L2 或 mismatch-rate 参数放宽。
- `workspace_buffer` 被声明为 scratch 输入；`out` 被声明为允许修改的输入。其余输入仍按 Evaluator 的输入副作用策略检查。
- Bootstrap 使用一轮 1 Case、再一轮 5 Cases 的分阶段验收。普通 Evaluate 使用 5 Cases、100 Bench Iters、一次逻辑 Evaluate；Retention 与 Agent Promotion 使用同 Allocation ABBA。这里采用 eager 模式，`warmup_iters=10` 与 `bench_iters=100` 分别是 10ms 和 100ms 预算，不是固定运行次数。每批一个 Shape，最多 16 批并发；默认锁频。ABBA 执行三次完整比较并逐 Shape 取中位数。
- 本任务关闭 **Production 静态源码 Gate**：完整上游 CuTe 源码包含测试/Benchmark 辅助逻辑，现有对全部可编辑文件的单 DSL 扫描会拒绝这些已有代码。没有修改全局 Gate；严格正确性、源码锁、范围约束和 Runtime 比较保持开启。若需要这项静态 Gate，须先适配多文件库的检测范围，不能直接切回 `true` 后假定能够运行。
- 提供的 `source-validation.json` 仅记录原始 P128 冒烟成功和目标 ABI 的预期失败。原始 `smoke.py` 的 L2 阈值不是本任务的 Gate；它不作为验收入口。
- GPU 环境需要 `torch>=2.9.0`、`nvidia-cutlass-dsl==4.6.1`。Quack 随源码提供。准备阶段不测试 GPU 镜像、模型登录或远端连接，也不会安装 GPU 环境依赖。
- 复用显式 Roofline，不启动 Roofline 构建器。原始文件的硬件标注是 `NVIDIA B300 (SM100)`；按本环境约定，Agate 资源名 `L20D` 对应 B300，不修改原始数值或伪造一份 L20D 数据。提交时只按 Runtime 的现有规则移除硬件名括号后缀。携带 Roofline 不等同于远端已返回 SOL；需要瓶颈证据时可使用 `profile`。

## 准备与运行

在 Linux / Lima 中使用已经安装 runtime 的 Python 环境；不要使用宿主 macOS 的共享 `.venv`。模型 CLI 必须已安装并登录，Linux 环境必须能运行 bwrap；`container` Launcher 不依赖 systemd 或独立 cgroup。Agate 的 `AGATE_AK`、`AGATE_SK` 应已导出，可选 `AGATE_URL` 覆盖服务地址。

```bash
cd ~/atrex-runtime
source env.sh
python3 scripts/source-tree/task.py prepare --backend claude
```

准备不启动服务、模型或 GPU Job。任务自己的 Runtime Config 默认监听 `127.0.0.1:8770`，可在准备时用 `--port` 更改；`--backend` 支持 claude/codex/qodercli/pi。所有命令都支持 `--workspace workspaces/FA4-trial-2`。改变输入、Backend 或端口，应使用新 Workspace；已存在的 Workspace 不会被覆盖。

可选：按原始脚本执行两种冒烟检查。这两条命令会直接提交 Agate Dev Job，需要 `agate` 在 PATH 中，但不需要启动 Runtime；不注册 Kernel 或 `v0`。目标检查在原始 R0 上预期失败：

```bash
python3 scripts/source-tree/task.py smoke --smoke-mode upstream-p128
python3 scripts/source-tree/task.py smoke --smoke-mode target --shape-id 0
```

这会临时物化完整 R0 源码、固定适配器和原始 `reference/shapes.json`，执行逐字保留的 `run_agate_dev.sh` / `smoke.py`；完成后清理临时目录。它不测试已被 Agent 修改的 Candidate，也不替代后续正式评测。

终端一启动 Runtime；Ctrl-C 优雅关闭：

```bash
python3 scripts/source-tree/task.py serve
```

终端二先只执行 Bootstrap，检查 bring-up：

```bash
python3 scripts/source-tree/task.py bootstrap
```

Bootstrap 成功后执行第一个 Epoch，或直接让 `campaign` 自动执行/复用 Bootstrap：

```bash
python3 scripts/source-tree/task.py campaign --target-epoch 1
python3 scripts/source-tree/task.py inspect
```

默认每个 Branch 一条 Trajectory，每个 Epoch 串行 3 个 Attempt；Epoch 1 仅 Active，从 Epoch 2 开始加入一个 Evolver 生成的 Challenger。想继续实验可运行 `campaign --target-epoch 3`；这是绝对目标 Epoch，不是额外运行三轮。不要为同一个 Campaign 同时运行两个 Scheduler。

运行目录包含 Task 快照、`source/` 与 `evaluator/` 固定 Git Checkout、`runtime.json`、`evaluation-contract.json`、`prepared.json`、Bootstrap/Epoch 结果和 `state/` 下的 Registry、Artifacts、Workspace 与 Session。不要在固定 Checkout 中优化；Agent 修改的是每次 Session 的 `work/kernel/`。Runtime 鉴权密钥在首次运行时自动生成、保存到权限 0600 的 `runtime-secrets.json`，服务与任务进程复用同一份，不必手动生成。

更多检查可以直接使用 runtime CLI，例如：

```bash
atrex-kernel-agent-runtime list-attempts --config workspaces/FA4/runtime.json --lineage <lineage_id>
atrex-kernel-agent-runtime list-worker-sessions --config workspaces/FA4/runtime.json --campaign <campaign_id>
```

ID 来自 `bootstrap-result.json`。Bootstrap 尚未成功时，该结果文件可能不存在；此时应先查看终端错误和 `state/lineage-bootstrap-workspaces/` 中的 Session。配置和源码加载成功不代表 R0 已通过目标正确性；第一次模型运行仍必须完成上述源码能力补齐。
