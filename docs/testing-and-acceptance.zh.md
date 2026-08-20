# 测试与生产验收

[English](testing-and-acceptance.md) | 中文

## 仓库验证

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check src tests scripts
.venv/bin/mypy src
.venv/bin/python scripts/smoke-wheel-independence.py
sudo .venv/bin/python scripts/validate-linux-sandbox.py
(cd src/atrex-kernel-agent-core && ../../.venv/bin/python -m pytest -q)
(cd src/atrex-kernel-agent-core && ../../.venv/bin/ruff check src tests)
(cd src/atrex-kernel-agent-core && ../../.venv/bin/mypy src tests)
(cd src/atrex-kernel-agent-evolver && ../../.venv/bin/python -m pytest -q)
(cd src/atrex-kernel-agent-evolver && ../../.venv/bin/ruff check src tests)
(cd src/atrex-kernel-agent-evolver && ../../.venv/bin/mypy src tests)
(cd workspaces/local-wiki && ../../.venv/bin/python -m pytest -q)
(cd workspaces/local-wiki && ../../.venv/bin/ruff check src tests)
(cd workspaces/local-wiki && ../../.venv/bin/mypy src tests)
```

文档不固定测试数量，因为每次实现增量都会改变数量；发布证据必须保存上述命令的实际输出。
Lima 参考环境已经覆盖 Linux Sandbox 边界、共享宿主 DNS/出站与 Runtime 访问、virtiofs Worker
Root 准备，以及 Claude/Codex/QoderCLI 真实连通性。这些证据只用于诊断，不能替代准确生产镜像
上的验收。

Runtime Suite 覆盖持久状态 Transition 与 Schema Migration、Fencing、Append-only Bootstrap Generation/部分恢复、共用 Core 阶段、完整仓库 Evolution、Token Report、普通重复 Evaluate、同 Allocation ABBA Schedule/Batch/逐轮证据、持久 Kernel/Agent/Measurement/Bootstrap Run Catalog 查询和有界 Source 导出、按 Generation 隔离的 Gateway/Wiki Capability 与 Operation 保留、Evidence 可见性与按 Digest 原始物化、Wiki Outbox、生命周期 Event、管理、Readiness、保留、Git Archive 安全、精确 Commit Provenance、应用启动和 Wheel 独立性。

## 真实 Agent Backend 连通性

可选验收脚本会把每个被选 Backend 分别放入生产同款 `BwrapSandboxLauncher` 边界，应用 cgroup
v2 限额、挂载/进程隔离、只读 Resolver 和共享宿主网络。命令直接由
当前 Checkout 中的 Core Backend Adapter 生成；一次最小非交互请求必须返回成功的 Provider
终态 Event：

```bash
sudo --preserve-env=PATH,CODEX_HOME,OPENAI_API_KEY,ANTHROPIC_AUTH_TOKEN,ANTHROPIC_API_KEY,QODER_PERSONAL_ACCESS_TOKEN \
  .venv/bin/python scripts/validate-agent-backends.py
```

缺少二进制或凭据会报告 `skipped`；Provider、代理、Sandbox 或终态 Event 错误会报告
`failed`。CI 或发布验收应增加 `--require-all`。可以单独选择 Backend：

```bash
sudo --preserve-env=PATH,CODEX_HOME \
  .venv/bin/python scripts/validate-agent-backends.py \
  --backend codex --require-all --json

sudo --preserve-env=PATH,QODER_PERSONAL_ACCESS_TOKEN \
  .venv/bin/python scripts/validate-agent-backends.py \
  --backend qodercli --require-all
```

脚本从原始非 root `SUDO_USER` 的 Home 发现登录状态，只把当前 Backend 的凭据路径只读挂载。
目标镜像使用非标准位置时，可以重复使用 `--credential BACKEND=PATH`，并通过
`--executable BACKEND=PATH`、`--model BACKEND=MODEL` 和 `--settings BACKEND=JSON` 覆盖。
脚本不会打印凭据值。

在 Lima 中，需要在 Ubuntu Guest 内安装 Runtime 和所选 CLI，并以普通 Guest 用户完成登录，
随后再用 `sudo` 执行同一命令。macOS 虚拟环境和宿主 CLI 二进制不能在 Linux 中复用。通过该
检查只证明一次真实模型请求经过参考 Sandbox 的直连网络路径，不代表完整 Core/Evolver 流程
已经验收。

## 强制生产证据

仓库测试不能证明下列事项。生产发布记录必须在准确镜像与硬件上附带每项证据：

| 门禁 | 验收条件 |
| --- | --- |
| Package/来源 | Runtime Wheel 在无源码 Checkout 环境运行；批准的 Core/Evolver Commit、Tree、封存 Digest 与发布元数据一致。 |
| Secret/配置 | 未知字段、弱或缺失 Secret、不安全路径、缺失继承环境变量和非完整 Commit 在工作前失败。 |
| 恶意代码隔离 | 在目标 Linux 镜像证明 Core/Evolver 无法读取相邻 Workspace/未声明宿主 Home 路径、无法写出 `~/workspace`、无法在进程树终止后存活、无法逃逸 Mount/Process Namespace，也无法超过 CPU/内存/PID 限制。另行证明声明的共享网络行为：公网 DNS/出站、Loopback Runtime，以及可达宿主/Peer 服务不会被 Runtime 限制。网络隔离不属于该边界；命令契约单测不等于通过目标环境门禁。 |
| 进程生命周期 | Timeout、Cancellation、后代进程、Provider 故障和 Host 终止不残留进程，并收敛到明确可恢复状态。 |
| Gateway/GPU | 所选 Core Binding 经 Runtime Proxy 在固定 Agate、Driver、GPU、Evaluation Contract 上跑通全部授权与拒绝操作。 |
| Evolution 语义 | 配置的 Challenger Pool、同起点并发 Trajectory、每 Trajectory 串行 Attempt、独立 Kernel/Agent 选择、同 DSL、全新 Session、Rollback 和失败证据在真实 Provider 下可复现。 |
| Token 记账 | 四类 Provider Token 准确；缺失或不一致 Report 在预期模型请求边界 Fail Closed。 |
| Evidence/Wiki | Optimizer 已晋升历史、包含 Agent 胜者与精确 Kernel 的 Evolver 全分支历史、同 Trajectory 当前 Attempt、完整已测量 Epoch Outcome、原始 Session 物化、Wiki 返回前冻结和原始 Feedback 至少一次交付/重试/保留符合真实服务。 |
| Crash 恢复 | 在 Bootstrap、Challenger、Attempt、Selection、Evidence 发布、Task Lease、Wiki Lease 强制终止，不产生重复权威 Outcome。 |
| 运维/Soak | Backup/Restore、Readiness 故障、GC、轮换、磁盘压力、取消、可观测性保留和代表性多 DSL Soak 保持约定边界。 |

每份记录必须包含发布身份、Wheel Digest、Core/Evolver Commit/Tree/Artifact、镜像、OS/Kernel、实现后的隔离策略、Driver/GPU、Agate/Wiki/Provider/Model 版本、准确命令、时间、已处理运维日志、Registry 备份、相关 Artifact Digest、结果、批准人和豁免。豁免必须注明 Owner、到期日、影响范围、补偿控制与回滚触发条件。
