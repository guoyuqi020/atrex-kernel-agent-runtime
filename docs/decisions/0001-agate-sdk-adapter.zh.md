# 决策 0001：使用发布版 Agate Python SDK

[English](0001-agate-sdk-adapter.md) | 中文

## 状态

于 2026-08-15 接受。

## 背景

可信 Gateway Proxy 必须暴露 Agate 远程命令接口，同时不能向 Worker 暴露 Gateway 凭据。已安装的 `atrex-gateway-client` 0.12.1 提供零运行时依赖的同步 `Client`、可插拔认证、稳定 Eval Request Builder、Typed Job 提交、分段长轮询、取消、Job 与环境查询、存活检查和结构化 `GatewayError` 字段。

## 决策

Runtime 直接使用 `build_eval_request_from_content` 和适用的 `Client` 方法，并在 AnyIO Worker Thread 中执行同步 SDK 调用。Runtime 不启动 `agate` CLI，也不复制 Agate 的 HTTP 或 AK/SK 实现。协议 v2 为 eval/submit/get/profile/dev/check/sol/disassemble/jobs/cancel/env/health/config 提供规范命令等价接口。包升级不是 Gateway 请求且会修改可信 Python 安装，因此仍由部署管理。

SDK 是上游 Wire 的权威实现。Runtime 自己负责校验部署配置、解析封存的 Campaign Evaluation Contract、封存 Candidate、校验响应 JSON 与 Atrex-Bench Result 字段、分类故障、持久化外部 Job 归属，以及提交权威 Attempt Outcome。只有 `evaluate` 可以提交 Outcome；原始 EvalRequest Submit 和 SOL Result 只用于诊断。Job 列表、轮询和取消保持 Attempt 范围。

## 影响

`atrex-gateway-client` 成为从其内网 Package Index 获取并锁定范围的生产依赖。升级 SDK 时必须使用新的发布包运行 Adapter Contract Test。权威评测提交阶段的 Agate Validation Rejection 作为失败 Candidate Outcome；传输错误、未知状态和格式错误响应仍是基础设施故障。非权威失败 Job 返回结构化 Failed Result。Job List、Poll 和 Cancel 必须存在持久的 `(Attempt, job id)` 绑定。OSS 附件流式传输保留为未来独立 Artifact Capability。
