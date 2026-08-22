# 0048: Outer-container Worker launcher

## Decision

Runtime supports `launcher.mode="container"` for deployments whose aggregate resource boundary is
an outer OCI container. Each Worker is still launched through bubblewrap with the same read-only
root, private `~/workspace`, Runtime-storage/sibling-root masking, dropped capabilities, namespaces,
and read-only Backend login-state projection used by the host Sandbox. The launcher invokes bwrap
directly: it does not invoke `systemd-run` or read/write a cgroup interface.

The mode provides per-Session filesystem and user/PID/IPC/UTS namespace isolation, but not
per-Session resource accounting. All Workers share the outer container's memory/CPU/PID limits.
Operators must use a dedicated container, allow bwrap's namespace operations, set aggregate limits
in the OCI runtime, and avoid mounting the Docker socket, Runtime secrets, private evaluator data,
or unrelated host paths. Deployments requiring per-Session cgroups continue to use `sandbox`.

Production service workspaces freeze the launcher mode. Attached Campaign workspaces inherit it,
and container mode does not request sudo escalation.
