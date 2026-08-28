# 真实 Agate 服务示例

[English](README.md) | 中文

本示例通过当前 `PATH` 解析到的官方 `agate` CLI 直接调用真实远端 Agate 服务，不会在本机
启动或模拟 Agate。

标准 Candidate、可信 PyTorch Reference、输入生成器和 Shape 元数据只保存在一份
`examples/shared/vecadd/` 中，本目录脚本以只读方式使用这些公共输入。

## 配置连接

设置真实服务地址；如果服务要求鉴权，再选择一种受支持的凭证形式：

```bash
export AGATE_URL="https://your-agate-service.example.com"

# Bearer Token 鉴权：
export AGATE_TOKEN="..."

# 或改用 AK/SK 鉴权：
# export AGATE_AK="..."
# export AGATE_SK="..."
```

不要把凭证写入 `runtime.json`，也不要提交到仓库。官方 CLI 会从环境变量读取
`AGATE_TOKEN` 或 `AGATE_AK`/`AGATE_SK`。如果服务采用无鉴权模式，则无需设置鉴权变量。

脚本直接运行当前环境中的 `agate`，不会固定仓库虚拟环境路径。需要时可把 `AGATE_BIN` 设置为
另一个命令或当前平台上的路径。macOS 创建的虚拟环境不能在 Linux 中复用；从 Linux 执行前应创建
并激活 Linux 自己的虚拟环境。

先查询服务及其准确的 GPU 环境名称：

```bash
bash examples/agate/check-service.sh
export AGATE_GPU="H20"  # 替换成 env 命令返回的某个值
```

`check-service.sh` 只查询服务，不会提交 GPU Job。

## 提交真实 GPU 工作

下面每个命令都会向配置的远端服务提交任务，并可能消耗 GPU 资源：

```bash
# 正确性和性能评测，并等待最终结果。
bash examples/agate/evaluate.sh

# Survey 级 Profiling；设置 AGATE_PROFILE_LEVEL=sol 可运行 NCU SpeedOfLight。
bash examples/agate/profile.sh

# 编译检查；也可以选择在 compute-sanitizer 下执行。
bash examples/agate/check-kernel.sh
AGATE_SANITIZE=memcheck bash examples/agate/check-kernel.sh

# 按服务能力生成 SASS/PTX。
bash examples/agate/disassemble.sh

# 在远端 Worker 执行开发命令；默认命令是 nvidia-smi。
bash examples/agate/dev.sh
bash examples/agate/dev.sh 'python -c "import torch; print(torch.cuda.get_device_name())"'
```

可选变量包括：

- `AGATE_CORRECTNESS_CASES` 和 `AGATE_BENCH_ITERS`：控制评测工作量；
- `AGATE_MODE=correctness`：只检查正确性，不测性能；
- `AGATE_ARCH` 和 `AGATE_SANITIZE`：配置 `check-kernel.sh`；
- `AGATE_DISASSEMBLY_FORMAT=auto|sass|ptx|isa`：配置 `disassemble.sh`；
- `AGATE_PROFILE_LEVEL=survey|sol|deep`：选择 Profiling 深度；Deep 模式还必须设置
  `AGATE_KERNEL_NAME`；
- `AGATE_HTTP_TIMEOUT`：单次 HTTP 请求超时，默认 `1800` 秒；
- `AGATE_JOB_TIMEOUT`：远端 Worker 执行预算，默认 `3600` 秒；
- `AGATE_WAIT_TIMEOUT`：客户端总体等待时间，默认 `3900` 秒；
- `AGATE_POLL_SECONDS`：轮询间隔，默认 `5` 秒。

总体等待时间有意设置得比远端 Job 预算更长，以便客户端有时间取回终态结果。
`evaluate-async.sh` 只使用 HTTP 和 Job 超时，不会等待或轮询。

这里特意不提供 `run-all`：这些操作会分配真实远端资源。

## 异步任务

提交后不等待，将 Agate Job ID 留作 Evidence，之后再查询或取消：

```bash
bash examples/agate/evaluate-async.sh
bash examples/agate/get-job.sh ev_your_job_id
bash examples/agate/list-jobs.sh --kind eval --limit 20
bash examples/agate/cancel-job.sh ev_your_job_id
```

## 与 Runtime 的关系

这些脚本是面向操作者的 Agate 与示例 Kernel 冒烟测试，会有意绕过 Runtime。正式的
Optimizer Session 中，Agent 应调用 Runtime Gateway Tool：Runtime 会把请求绑定到 Attempt，
校验 Capability 和 Ownership，注入冻结的 Evaluation Contract，记录幂等与 Evidence，
并确保 Agate 凭证不会进入 Agent 工作区。

若要让 Runtime 连接同一个服务，只在 `runtime.json` 的 `agate` 段写入非敏感连接策略，
选择 `auth_mode`（`none`、`token` 或 `ak_sk`），并指定相应凭证环境变量的名称。例如，
Token 模式增加 `"token_env": "AGATE_TOKEN"`；AK/SK 模式增加
`"access_key_env": "AGATE_AK"` 和 `"secret_key_env": "AGATE_SK"`。
基于 Runtime 的示例应在自己的 `runtime.json` 中完成该配置，不导入其他示例的配置。
