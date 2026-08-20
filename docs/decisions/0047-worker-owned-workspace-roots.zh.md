# 决策 0047：由 Worker 创建 Sandbox Root

## 状态

已接受。

## 决策

Sandbox Host Check 通过以配置的非 root `worker_user` 运行的 systemd transient service，直接
创建 Attempt、Evolution、Problem Generalization、Lineage Bootstrap 四类 Workspace Root 和
Probe。多个 Scheduler 进程使用文件锁串行完成准备。可信 Runtime 可以用 root 身份组装 Run，
但会在 bwrap 启动前把完整的当前 Run 交给 Worker。

Root 准备不采用“root 创建后 chown”。失败准备遗留的错误 Owner 空 Root 可以删除并以 Worker
身份重建；非空 Ownership 不匹配通常会 Fail Closed，并保留给运维检查。Lima `virtiofs` 可能让
root 与 Worker 看到同一路径的不同数字 Owner。因此 root 视角不匹配时，Runtime 会再通过以
Worker 身份运行的 systemd transient service 执行一次 `stat`。只有 Worker 视角严格匹配配置的
Worker UID/GID 时才接受非空 Root。Campaign Restart 后，旧 bwrap service 退出时可能短暂遗留
system manager 的 `root:root` virtiofs 视角，因此 Runtime 只对这一特征不匹配做最长 30 秒的
有界重试；其他不匹配或 30 秒内未收敛仍然 Fail Closed。

## 结果

- 即使 Lima virtiofs 上 `chown` 返回成功却不改变可见 Owner，Sandbox 仍可工作；
- 仅由 virtiofs 视角差异造成误判时，已有非空 Bootstrap 状态不会丢失；
- 三个 DSL Bootstrap 进程可以安全共享一份 Runtime 配置，不会竞争 Host Probe；
- 运维人员不得使用 sudo 预建 Worker Root；
- Runtime 不会递归删除或改写非空的错误 Owner Root。
