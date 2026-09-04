# 示例

[English](README.md) | 中文

- [`shared/`](shared/README.zh.md)：多个可运行示例共同使用的标准只读 VecAdd Fixture 与通用
  Helper；该目录本身不是可运行流程。
- [`bootstrap/`](bootstrap/README.zh.md)：通过 Core、Runtime Tools、配置选择的 Agent Backend 和远端 Agate
  真实运行单 DSL VecAdd Campaign Bootstrap。
- [`lineage/`](lineage/README.zh.md)：Bootstrap 一个 Triton VecAdd Lineage，并按可配置的
  Challenger、Trajectory 和串行 Attempt 数运行一个 Epoch。
- [`evolution/`](evolution/README.zh.md)：运行三个 Epoch，每个 Branch 一次 Attempt，并仅在前
  两个 Epoch 之后创建 Challenger。
- [`optimizer-dev-shell/`](optimizer-dev-shell/README.zh.md)：创建带有实时 Gateway Authority
  的一次性 Optimizer 兼容工作区，不运行 Bootstrap、不持久化 Lineage，也不启动 Agent。
- [`evolver-dev-shell/`](evolver-dev-shell/README.zh.md)：无需 Bootstrap、Runtime 服务或 Agent
  进程，打开一份用完即销毁的合成 Evolution Workspace。
- [`agate/`](agate/README.zh.md)：使用官方 CLI 直接调用真实远端 Agate 服务，演示评测、
  Profiling、编译检查、反汇编、开发命令与任务管理。
- [`local-wiki/`](local-wiki/README.zh.md)：启动独立本地 GPU Wiki，通过浏览器/API 查询。
  Agent Wiki 工具及对应 Shell 演示暂时不可用。
- [`kernel-design-agents/kernel-agent.example.json`](kernel-design-agents/kernel-agent.example.json)：
  KDA Optimizer 的 `kernel_agent` 配置段，包含 Skill 子模块白名单和完整 Bundle 限额。
  不是完整 Runtime 配置或可运行脚本；用法见 [KDA Optimizer](../docs/user-guide.zh.md#kda-optimizer)。

每个 Runtime Example 都在自己的 `runtime.json` 中分别选择
`campaign.optimizer.agent_backend` 与 `campaign.evolver.agent_backend`。可选值为 `claude`、
`codex`、`qodercli` 和 `pi`；仓库内 Runtime 默认由 Optimizer 与 Evolver 共用 QoderCLI。

Example Wrapper 从当前 `PATH` 解析 `python3`、`atrex-kernel-agent-runtime` 和 `agate`；也可分别
通过 `ATREX_PYTHON`、`ATREX_RUNTIME_CLI` 与 `AGATE_BIN` 显式覆盖。Lima 挂载目录中的 macOS
虚拟环境不能复用，必须先在 Linux 内创建并激活自己的环境。
