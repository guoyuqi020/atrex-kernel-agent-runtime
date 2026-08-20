# 可运行的 Campaign Bootstrap 示例

[English](README.md) | 中文

本示例会真实 Bootstrap 一条 Triton VecAdd Lineage：启动 Runtime 控制服务，以
`framework_baseline` 模式运行固定 Commit 的 Core；Core 通过 Runtime Gateway Tool 把
Candidate 提交到真实远端 Agate；只有权威评测正确后，Runtime 才会登记 Baseline Kernel
并将 Lineage 置为 Ready。

这不是 Mock 流程，会调用 QoderCLI 并消耗远端 GPU 资源。示例不会启动 Local Agate；除非
显式设置 `ATREX_WIKI_URL`，否则 GPU Wiki 也默认关闭。

本目录自行持有 [`runtime.json`](runtime.json) 部署模板与 [`campaign.json`](campaign.json)
拓扑；二者只引用 [`../shared/vecadd`](../shared/vecadd) 中的标准只读 VecAdd 输入，不依赖其他
可运行示例。

## 前置条件

安装本仓库开发环境，并导出 QoderCLI 凭据与远端 Agate 连接。`AGATE_GPU` 必须是
`agate env` 返回的准确环境名称。

```bash
# ~/.qoder 与 ~/.qodersec 已有有效登录态时可省略：
# export QODER_PERSONAL_ACCESS_TOKEN="..."
export AGATE_URL="https://your-agate-service.example.com"
export AGATE_AK="..."
export AGATE_SK="..."
export AGATE_GPU="H20"
```

Wrapper 可以通过配置的 Worker Environment 传递 `QODER_PERSONAL_ACCESS_TOKEN`；未提供时，
Runtime 会把宿主 `.qoder` 与 `.qodersec` 只读挂入 Session Home。Agate 与 Agent 凭据都不会
写入生成文件或保留进 Workspace Artifact。

## 执行 Bootstrap

推荐入口会自动启动自己拥有的 Runtime、等待健康、执行 Bootstrap、输出 Inspect 结果，
并在成功、失败或中断时停止 Runtime：

```bash
bash examples/bootstrap/run.sh
```

如果配置端口上已经存在 Runtime，该脚本会拒绝继续，因此不会停止并非由它创建的进程。
Runtime 日志保存在 `workspaces/bootstrap-example/runtime.log`。

需要手工调试时，可以先单独生成并检查非敏感输入：

```bash
bash examples/bootstrap/prepare.sh
```

然后在第一个终端保持 Runtime 运行：

```bash
bash examples/bootstrap/start-runtime.sh
```

在具有相同 Agate 环境变量的第二个终端执行真实 Bootstrap，然后 Inspect：

```bash
bash examples/bootstrap/bootstrap.sh
bash examples/bootstrap/inspect.sh
```

Bootstrap 可能耗时较长，因为 Core 会启动真实 Agent Session，并远端评测生成的
Baseline。默认 Evaluation Job 预算为 3600 秒，Runtime 单次请求超时为 1800 秒，
Agate 总等待时间为 3900 秒，示例中每个 Core Session 的 Optimizer Token 配额为 20,000,000 个
Provider Token。启动两个命令前可以覆盖：

```bash
export AGATE_HTTP_TIMEOUT=3600
export AGATE_JOB_TIMEOUT=7200
export AGATE_WAIT_TIMEOUT=7500
export ATREX_OPTIMIZER_MAX_SESSION_TOKENS=25000000
```

Token 配额按同一 Session 内所有模型请求累计，包含 Provider 报告的非缓存输入、缓存读取、
缓存写入和输出 Token；它不是模型 Context Window 的大小。
还可以使用 `ATREX_CHALLENGER_COUNT`、`ATREX_CHALLENGER_START_EPOCH`、
`ATREX_TRAJECTORIES_PER_BRANCH` 和 `ATREX_ATTEMPTS_PER_TRAJECTORY` 覆盖生成的 Bootstrap
拓扑；这些值只影响 Baseline 登记后的普通 Epoch，不改变 Framework Baseline Session 本身。

`inspect.sh` 只输出包含持久化身份、Baseline Kernel 和 Agent Revision 的 Bootstrap 结果，
不会再自动查询 Kernel Catalog；需要完整权威评测记录时，可以单独使用 Runtime 的
`list-kernels` 命令。每条 Lineage 结果包含准确的 `bootstrap_attempt_id` 和结构化
`baseline_kernel.producer`；普通 Epoch `attempt_id` 仍按设计保持为空。手工模式 Inspect
完成后，通过 `Ctrl-C` 停止 `start-runtime.sh`。

## 生成状态与幂等性

示例会把生成配置、本地 Runtime 签名/管理 Secret、SQLite、Artifact、Workspace 和最近一次
结果写入 `workspaces/bootstrap-example/`，该目录已被 Git 忽略。本地 Secret 文件权限为
`0600`，其中只有自动生成的 Runtime 控制 Secret，不包含 Agate 凭证。

新 Workspace 的默认 Creation Key 根据 `AGATE_GPU` 和当前 Core Commit 生成。一旦生成了
Campaign 定义与 Runtime Config，重复执行会固定原 Creation Key、Optimizer Commit、Evaluation
Contract 和 Evolver Commit；本地 Core/Evolver HEAD 变化不会让已有 Campaign 静默换线。
需要让另一个不可变 Campaign 使用新 Commit 时，应指定新的状态目录；创建该新 Workspace
时可以设置 `ATREX_BOOTSTRAP_CREATION_KEY`。

本示例自己持有的定义为：

- `campaign.json`：Campaign schema v3 模板，其 `lineages` Key 决定 DSL 集合并可选择各 Lineage 的 Model；
- `runtime.json`：本示例完整的 Runtime 部署模板。

Evaluation Contract、Agent Problem、Baseline Kernel 和 Initial Evidence 统一来自
`examples/shared/vecadd/` 的标准 Fixture。
