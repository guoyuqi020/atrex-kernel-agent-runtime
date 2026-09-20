# Agate localhost 后端

[English](agate-localhost.md) | 中文

Runtime 默认连接官方远端 Agate 服务。本文说明如何显式覆盖为可选的 localhost 部署：
`http://127.0.0.1:8000`、GPU 选择器 `local`。Localhost 是 Agate 的部署后端，不是新的 Runtime
评测器或 SDK 传输方式。Evaluate、ABBA、
Profile、Dev、Check、Disassemble 沿用同一套 SDK 调用及 Runtime 记录、去重、重试和结果投影策略。

## 在 GPU 机器上部署 Agate

Runtime 依赖的 HTTP Client 不包含 Agate Server。使用官方服务端源码及 GPU Python 环境，
将[本地服务配置](../scripts/shared/agate-local.example.json)复制到 Git 之外，按机器实际情况设置
`python_bin`、`gpu_model`、设备编号以及状态/工作目录的绝对路径。
Agate 的 `app.local.detect.detect_local_device()` 可探测设备信息。

在 Agate 服务端仓库执行：

```bash
AGATE_CONFIG_FILE=/path/to/agate-local.json python3 -m app.main
```

按服务端的部署配置将监听地址限制为 loopback、端口设为 8000。服务端配置和鉴权由部署负责；若启用鉴权，export 该服务对应的
`AGATE_AK` 和 `AGATE_SK`。无鉴权的 loopback 服务可以不设置这两个变量。生成的 Runtime 配置
只保存凭据变量名；只要存在任一凭据，本地连接也会使用 AK/SK，且要求完整的一对。显式非本机
地址沿用 AK/SK 鉴权。

本地执行器以 Gateway 用户的权限执行子进程，本身不是安全沙箱。应使用专用 GPU 执行机器或
容器，不携带 Runtime 凭据、Registry 文件或无关数据；Optimizer 的 bwrap 并不隔离 Gateway
执行的代码。不要在 Optimizer/Evolver 沙箱内启动服务，也不要在缺少适当鉴权和执行隔离的情况下
公开暴露服务。

## 连接 Runtime

显式覆盖远端默认值：

```bash
export AGATE_URL=http://127.0.0.1:8000
export AGATE_GPU=local
bash examples/agate/check-service.sh
bash scripts/production/services.sh start --workspace workspaces/production/control-local
bash scripts/production/campaign.sh start \
  --service-workspace workspaces/production/control-local \
  --kernel suite/operator --backend claude
```

Agate 必须先启动。服务脚本仅管理 Runtime/Wiki，不安装、启动或关闭 Agate。
`local` 指 Agate 服务所在机器的 GPU，不是 Agent 沙箱。Runtime 若运行在容器内，须确保能从该
容器访问此地址；否则设置可达的 GPU 执行器地址及对应凭据。

GDN/FA4 输入包保持原先声明的 `L20D` 调度目标，在对应 GPU 上运行时
应将 `L20D` 加入本地 Cluster 的 aliases。Runtime 仍从 Agate 查询实际架构并告知 Agent。
已有 Campaign 的 GPU/输入身份已封存，切换执行环境应使用新工作区；更新模板不会覆盖已生成的
常驻服务配置。
