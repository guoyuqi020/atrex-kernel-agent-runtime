# 临时 Evolver 调试 Shell

[English](README.md) | 中文

该示例打开一个用完即销毁、带沙箱隔离、与 Evolver 兼容的工作区。它不会启动 Runtime HTTP
服务、Bootstrap、Campaign、Lineage、Epoch、Optimizer 或 Evolver。

Runtime 会把固定 Commit 的 Core 基线导入为合成的 `agent-v0`，封装配置中的初始 Evidence，
再构造正常的 Evolver 目录。示例不会运行或伪造 Kernel 评测；当前 Agent 池仅包含合成的
Active `agent-v0` 父版本。

## 运行

在 Linux 上运行，需要免密 `sudo`、`bwrap`、`systemd-run` 以及已安装的 Coding Agent CLI。
QoderCLI 是默认元数据选项，但交互 Shell 会暴露宿主机上所有可用的 Claude、Codex、
QoderCLI 和 Pi 登录态：

```bash
bash examples/evolver-dev-shell/run.sh zsh qodercli
bash examples/evolver-dev-shell/run.sh bash codex
```

Backend 参数只决定合成输入里的 Evolver 元数据，不会启动 Backend 进程。示例不需要 Agate
凭据；如果设置了 `AGATE_GPU`，它会被用作硬件标签，否则使用 `nvidia-h100`。

进入 Shell 后可以检查 `input/agents/` 中的当前参赛池、`input/evidence/<role>/` 中对应的 Session
与实测效果、`input/historical/agent-vN/` 中的历史版本，以及可写的
`candidate/` 与 `scratch/` 目录。Evolution
Manifest 位于 Runtime-private `.runtime/`，不属于 Agent-facing Workspace Contract。

```bash
find input/agents input/evidence input/historical -maxdepth 3 -type f -print
cat input/evidence/active/optimization-summary.json
```

退出后，内部 Evolution Workspace 与外层临时目录都会被销毁。JSON 摘要会报告
`workspace_destroyed: true`，并且不会创建任何持久化 Registry 对象。

用于重建真实 Epoch 历史的 `evolver-dev-shell --lineage ... --epoch ...` CLI 仍然保留；本示例
明确使用新的 `temporary-evolver-dev-shell`。
