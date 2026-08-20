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
- [`local-wiki/`](local-wiki/README.zh.md)：启动本地 GPU Wiki，并通过 Runtime Tools 演示
  Agent 的 `wiki-query` 流程；也可打开面向 Wiki 调试的托管 Shell。

每个 Runtime Example 都在自己的 `runtime.json` 中分别选择
`campaign.optimizer.agent_backend` 与 `campaign.evolver.agent_backend`。可选值为 `claude`、
`codex`、`qodercli` 和 `pi`；仓库内 Runtime 默认由 Optimizer 与 Evolver 共用 QoderCLI。

Example Wrapper 从当前 `PATH` 解析 `python3`、`atrex-kernel-agent-runtime` 和 `agate`；也可分别
通过 `ATREX_PYTHON`、`ATREX_RUNTIME_CLI` 与 `AGATE_BIN` 显式覆盖。Lima 挂载目录中的 macOS
虚拟环境不能复用，必须先在 Linux 内创建并激活自己的环境。
