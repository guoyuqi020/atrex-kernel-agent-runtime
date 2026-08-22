# 0048：外层容器 Worker Launcher

## 决策

当资源总边界由外层 OCI 容器提供时，Runtime 支持 `launcher.mode="container"`。每个 Worker 仍会
通过 bubblewrap 启动，使用与宿主 Sandbox 相同的只读根、私有 `~/workspace`、Runtime Storage/
兄弟 Root 遮蔽、Capability 丢弃、Namespace 和 Backend 登录态只读投影。Launcher 直接调用 bwrap，
不调用 `systemd-run`，也不读写任何 cgroup 接口。

该模式提供逐 Session 文件系统和 User/PID/IPC/UTS Namespace 隔离，但不提供逐 Session 资源
计量；所有 Worker 共享外层容器的内存、CPU 与 PID 总限额。运维方必须使用专用容器，允许 bwrap
创建所需 Namespace，在 OCI 层设置资源总限额，并避免挂载 Docker Socket、Runtime Secret、
私有评测数据或无关宿主路径。需要逐 Session cgroup 的部署继续使用 `sandbox`。

生产 Service Workspace 会冻结 Launcher 模式，附着的 Campaign Workspace 自动继承；
`container` 模式不会请求 sudo 提权。
