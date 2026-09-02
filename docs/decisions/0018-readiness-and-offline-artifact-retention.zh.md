# 决策 0018：本地就绪与离线 Artifact 保留

[English](0018-readiness-and-offline-artifact-retention.md) | 中文

## 状态

已接受并实现。

## 背景

进程存活无法区分 ASGI Loop 可响应和本地持久存储不可用。不可变 CAS 对象也会无限积累，其中包括 Artifact 已封存、后续数据库写入失败后留下的对象。仅按目录扫描删除并不安全，因为权威引用同时存在于 Registry 与 Gateway Control 数据库，保留 Event 可能保存诊断 Artifact 引用，在线进程还可能正处于“已封存 Artifact、尚未提交引用”的窗口。

## 决策

`GET /readyz` 对 Registry、Gateway Control 数据库、Agate Job Store 和 Artifact Store Staging 区执行受限读写探针。响应只包含失败的本地依赖名，不包含异常文本。`/healthz` 仍只表示存活。外部 Agate、GPU Wiki、Agent Provider、cgroup 和 GPU 被有意排除：这些依赖暂时不可用时，可信控制面仍应能够启动，以便检查和恢复。

Artifact 保留是显式离线 CLI 操作。Collector 汇总 Registry 列、已提交 Gateway Outcome 和保留 Runtime Event Payload 引用的全部 Artifact Digest，再沿这些已验证 Artifact 内嵌的现存 Digest Token 求传递 CAS 闭包。它只考虑早于 Operator 指定最小年龄的不可达对象，删除前验证每个完整 CAS 对象，遇到意外 Entry 会失败，并在 Operator 指定对象上限处停止。默认行为是 dry-run。实际删除要求 `--confirm-runtime-stopped`；全部 Runtime、Worker、Bootstrap 和 Scheduler 进程必须确实停止，因为该确认是运维前置条件，而不是系统自动推断的锁。

实际执行后会发出 `artifact.gc_completed`。文件系统删除与 SQLite 审计追加之间有意不提供跨存储事务；被删除对象不在完整持久引用快照中，缺少审计 Event 也不会使其变成有引用对象。

## 影响

负载均衡器可以摘除自身持久 Store 不可用的进程，同时无需把启动与外部服务耦合。磁盘保留通过保守且可审计的维护操作控制，不再依赖手工删除 CAS。每次实际执行前，Operator 必须保留一致备份、提供用于打开 Gateway 状态的 Capability 签名密钥、选择适合部署的年龄并确认静止。目标镜像中的就绪故障和 GC/恢复演练仍是生产验收要求。
