# 决策 0020：保留失败 Evolution 运行的受限证据

[English](0020-failed-evolution-evidence.md) | 中文

## 状态

已接受并实现。

## 背景

成功 Evolver 运行已经产生不可变来源记录，但超时、进程失败和被拒绝 Candidate Manifest 只有生命周期 Event 与可变执行 Workspace。这使失败自进化更难审计，也无法形成统一 Artifact 保留策略。直接封存整个不可信 Workspace 既不安全也没有大小上限。

## 决策

每次失败 Evolver 调用都会尝试封存一个 `EVOLUTION` Failure Artifact。严格版本 1 记录包含不可变 Input Manifest、失败阶段、异常类型；当 Coding Agent 已返回结构化结果时，还包含不含秘密的 Agent Descriptor、Return Code、受限 stdout/stderr、已校验 Token Report，以及可选的独立封存 Session Trace。它不复制 Candidate Tree、异常消息、Prompt、完整 argv、环境值或 Secret。

Worker Timeout/Failure 或 `evolution.candidate_rejected` Event 携带 `failure_artifact_digest`。如果失败证据封存本身失败，原异常仍是权威错误，Event 只记录 `failure_retention_error_type`；保留故障绝不能掩盖或重新分类主故障。失败 Trace 绝不会创建或晋升 Kernel Agent Revision。

## 影响

成功与失败的自进化现在都能通过不可变且受限 Artifact 审计，同时晋升语义保持不变。配置后，原始 Session 以显式内容寻址方式保留；执行 Workspace 在受限离线 Workspace GC 前继续作为部署诊断状态。Failure Artifact 会进入以 Event 为根的传递 Artifact GC 保留集合。
