# 决策 0046：Sandbox Worker 共享宿主网络

## 状态

已接受。

## 决策

Sandbox Worker 保留宿主 Network Namespace。Runtime 不再创建 Bridge、veth、地址租约、
Firewall Chain、NAT 规则或每 Worker Network Namespace。私有 `/run` 中只读投影 Resolver
文件，Provider CLI 使用宿主 DNS、路由和出站连接。

bwrap 的挂载/User/PID/IPC/UTS/cgroup 边界、非 root Worker 身份、私有可写 Workspace、
Runtime Storage 遮蔽、Capability 丢弃与 systemd cgroup 限额仍是强制项。

## 结果

- Provider CLI 的网络行为与宿主调用一致，避免 Namespace 特有的 DNS/路由故障；
- Runtime 可以监听 Loopback，Worker 可直接调用；
- Worker 可以访问宿主服务、其他 Worker 和任意出站目的地；网络隔离明确不属于 Sandbox
  安全边界。
