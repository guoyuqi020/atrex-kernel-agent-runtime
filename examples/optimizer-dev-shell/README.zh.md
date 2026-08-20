# 临时 Optimizer 调试 Shell

[English](README.md) | 中文

该示例只创建一个可丢弃的 Optimizer 兼容工作区，不运行 Bootstrap，不创建持久化 Campaign 或
Lineage，也不启动任何 Agent Backend。

Runtime 会注入固定版本的 Core Bundle、已知正确的 Triton VecAdd Kernel、空的首次 Attempt
Evidence、可写 Candidate/Scratch 目录，以及临时且受限的 Gateway Capability。Shell 存续期间
仍可使用正常的 Runtime Tools。

Shell 固定使用生产 `sandbox` Launcher：通过 bubblewrap 隔离文件系统，以独立 Network
Namespace 隔离网络，并通过 systemd/cgroup-v2 限制资源。Wrapper 要求 Linux、相应系统工具和
免密 `sudo`；它让可信 Runtime/Launcher 以 root 身份重新进入，交互 Shell 仍以调用者的非 root
用户运行。调用者的 Backend Home 和 PATH 会被显式保留，因此用户本地安装的 `qodercli` 及其
只读登录状态仍可在 Sandbox 内使用。

## 运行

先导出远端 Agate 配置：

```bash
export AGATE_URL="https://your-agate.example"
export AGATE_AK="..."
export AGATE_SK="..."
export AGATE_GPU="H100"
bash examples/optimizer-dev-shell/run.sh zsh qodercli
```

Shell 与 Backend 参数都可省略，也可交换顺序。Backend 参数决定 Session Context 显示的
Runtime Binding，默认是 `qodercli`。与生产 Agent Session 不同，交互 Dev Shell 会同时投影
所有可用的 `claude`、`codex`、`qodercli` 和 `pi` 登录态，因此在同一个 Shell 内即可调用所有
已安装 CLI。明确选择 Backend 仍适合调试对应的 Runtime Binding：

```bash
bash examples/optimizer-dev-shell/run.sh bash claude
bash examples/optimizer-dev-shell/run.sh bash codex
```

每次执行都会通过 `mktemp` 创建唯一目录、启动独立 Runtime，
然后直接进入 Shell；不会调用 Qoder、Claude、Codex 或其他 Agent Backend。退出 Shell 后，
Capability 会被撤销，Runtime 会停止，包含 SQLite 数据库和 Artifact Store 在内的整个临时目录
都会被删除。

例如，默认 Shell 内可直接调用两种 Provider：

```bash
claude -p "Reply with exactly: hello"
codex exec --ephemeral --skip-git-repo-check "Reply with exactly: hello"
```

进入 Shell 后可以检查 `attempt.json`、`input/`、`agent/`、`work/kernel/`、`sessions/` 和
`scratch/`。例如，可继续通过下面的方式调用 Runtime Tool：

```bash
python agent/optimizer/src/runtime_tools.py gateway-execute --request scratch/request.json
```
