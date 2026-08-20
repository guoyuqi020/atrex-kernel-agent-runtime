# Atrex Kernel Agent Runtime

[English](README.md) | 中文

Atrex Kernel Agent Runtime 是自进化 GPU Kernel Agent 的可信 Python 控制面。它负责固定并加载
Optimizer/Evolver Commit、调度 Lineage 内竞争、通过 Agate 评测 Kernel、记录每个 Agent、Kernel、
Attempt、Session 和测量结果，并且只依据 Runtime 权威结果晋升。

Runtime 不实现 Agent 框架。Core 和 Evolver 持有 Prompt、Tool、Workflow 以及
Claude/Codex/QoderCLI/Pi Adapter；Runtime 持有不可变来源、Model/Backend 绑定、Capability、评测
策略、持久化、沙箱策略和晋升权限。

## 主要生命周期

```text
Campaign Bootstrap -> agent-v0 + Kernel v0
        |
        v
Epoch：创建 Challenger -> 执行 Branch/Trajectory -> 比较 Kernel -> 选择 Agent
        |
        v
不可变 Evidence + Wiki Feedback -> 下一 Epoch
```

- Campaign 冻结 Core/Evolver Commit、Evaluation Contract、Gate Policy、硬件和 DSL Lineage。
- 每个 Lineage 独立演进 Agent（`agent-vN`）和 Kernel（`vN`）。
- 每个 Attempt 都是全新 Agent Session，并可执行多次探索性 Gateway 评测。
- Runtime Finalization 在保留 Kernel 前执行正确性检查、可选 Production Gate，以及配置的普通
  Evaluate 或同 Allocation ABBA 比较。
- Evolver 可以新建、复用或基于历史 Agent Revision 进化。Active 与 Challenger Branch 在配置的
  上限内并发执行。
- Optimizer 可以实时查询外部 GPU Wiki。Runtime 在返回知识前冻结完整查询交互，并在 Epoch 完成后
  才上传消费和 Trace Feedback。

## 仓库结构

- `src/atrex_runtime/`：可信 Runtime Package
- `src/atrex-kernel-agent-core/`：独立版本的 Optimizer Git Submodule
- `src/atrex-kernel-agent-evolver/`：独立版本的 Evolver Git Submodule
- `third_party/atrex-bench/`：固定版本的评测器源码 Submodule
- `examples/`：相互独立、可运行的工作流样例
- `docs/`：设计、使用、接口、配置、运维和发布文档
- `workspaces/local-wiki/`：仅开发测试使用的 GPU Wiki 兼容服务

## 快速开始

基础要求为 Python 3.12+、Git、Agate 服务和一种受支持的 Agent CLI。生产沙箱还要求 Linux、
bubblewrap 与 cgroup v2/systemd。Sandbox Worker 直接共享宿主网络。

```bash
git submodule update --init --recursive
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'

export AGATE_URL='https://your-agate.example.com'
export AGATE_AK='...'
export AGATE_SK='...'
export AGATE_GPU='H20'
export QODER_PERSONAL_ACCESS_TOKEN='...'

bash examples/bootstrap/run.sh
```

每个 Example 都在 `workspaces/` 下生成独立配置和状态，不复用项目根目录的 `runtime.json`。真实
部署请复制 `runtime.example.json`；只有可信本地调试才选择 `development`，生产环境应配置 Linux
`sandbox` Launcher。完整步骤见[使用说明](docs/user-guide.zh.md)。

## 文档

- [文档导航](docs/README.zh.md)
- [架构与信任设计](docs/architecture.zh.md)
- [模块设计](docs/module-design.zh.md)
- [使用说明](docs/user-guide.zh.md)
- [CLI、HTTP 和 Runtime Tools 接口](docs/interfaces.zh.md)
- [配置说明](docs/configuration.zh.md)
- [性能与正确性 Gate](docs/performance-gates.zh.md)
- [部署与运维](docs/operations.zh.md)
- [生产运行脚本](scripts/production/README.zh.md)
- [发布检查清单](docs/release-checklist.zh.md)
- [可运行样例](examples/README.zh.md)

## 开发验证

```bash
.venv/bin/ruff check src tests scripts
.venv/bin/mypy src
.venv/bin/pytest
.venv/bin/python scripts/smoke-wheel-independence.py
```

Core、Evolver、本地 Wiki、Linux 沙箱和外部服务验收命令见
[测试与生产验收](docs/testing-and-acceptance.zh.md)。本地测试通过不能替代目标 Linux、Agent
Provider、Agate、Wiki 和 GPU 环境验收。

许可证：Apache-2.0。
