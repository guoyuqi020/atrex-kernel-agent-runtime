# 决策 0001：使用发布版 Agate Python SDK

[English](0001-agate-sdk-adapter.md) | 中文

## 状态

于 2026-08-15 接受。

## 背景

可信 Gateway Proxy 必须暴露 Agate 远程命令接口，同时不能向 Worker 暴露 Gateway 凭据。当前支持的 `atrex-gateway-client>=0.14.3,<0.15` 提供零运行时依赖的同步 `Client`、可插拔认证、包含原生多文件源码归档和 ABBA 的稳定 Eval Request Builder、Typed Job 提交、分段长轮询、取消、Job 与环境查询、存活检查和结构化 `GatewayError` 字段。

## 决策

Runtime 直接使用 `build_eval_request_from_content` 和适用的 `Client` 方法，并在 AnyIO Worker Thread 中执行同步 SDK 调用。Runtime 不启动 `agate` CLI，也不复制 Agate 的 HTTP 或 AK/SK 实现。面向 Agent 的协议 v2 仅暴露 evaluate/profile/dev/check/disassemble/env；Runtime 内部保留 Job 查询、轮询、取消、健康检查和连接信息，用于恢复与管理。包升级不是 Gateway 请求且会修改可信 Python 安装，因此仍由部署管理。

所有新提交的单文件、多文件普通 Evaluate 和 ABBA 均使用原生 Eval。多文件请求携带完整 OSS 源码归档和显式入口；ABBA 两侧使用独立的 Candidate/Baseline 归档。不支持的请求或旧 SDK 明确报错，不回退 Dev。源码树诊断和 Agent Dev 仍使用 Dev；已提交的旧 Dev 评测仅保留只读恢复能力，不创建新的 Dev 评测作业。归档与排程限制详见[源码树](../source-trees.zh.md)。

单文件和多文件 ABBA 的 `repeats` 都必须是 `{2, 4, 6, 8, 10, 12, 14, 16}` 中的严格整数，Runtime 发送 `repeats / 2` 个原生 ABBA Block。非法值返回参数校验错误，不回退 Dev。无论源码形式如何，缺少原生 Request Builder 都返回明确配置错误，不提交 GPU 作业。

SDK 是上游 Wire 的权威实现。Runtime 自己负责校验部署配置、解析封存的 Campaign Evaluation Contract、封存 Candidate、校验响应 JSON 与 Atrex-Bench Result 字段、分类故障、持久化外部 Job 归属，以及提交权威 Attempt Outcome。只有 `evaluate` 可以提交 Outcome；原始 EvalRequest Submit 和 SOL Result 只用于诊断。Job 列表、轮询和取消保持 Attempt 范围。

## 影响

`atrex-gateway-client` 成为从其内网 Package Index 获取并锁定范围的生产依赖。升级 SDK 时必须使用新的发布包运行 Adapter Contract Test。权威评测提交阶段的 Agate Validation Rejection 作为失败 Candidate Outcome；传输错误、未知状态和格式错误响应仍是基础设施故障。非权威失败 Job 返回结构化 Failed Result。Job List、Poll 和 Cancel 必须存在持久的 `(Attempt, job id)` 绑定。Runtime 通过 SDK 的 OSS API 上传 Kernel 源码归档，不向 Agent 暴露上传凭据。
