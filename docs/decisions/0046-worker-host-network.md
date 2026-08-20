# Decision 0046: Sandboxed Workers Share Host Networking

## Status

Accepted.

## Decision

Sandbox Workers retain the host network namespace. Runtime does not create bridges, veth pairs,
address leases, firewall chains, NAT rules, or per-Worker network namespaces. A read-only resolver
file is projected into the private `/run`, while provider CLIs use host DNS, routing, and egress.

The bwrap mount/user/PID/IPC/UTS/cgroup boundary, non-root Worker identity, private writable
Workspace, hidden Runtime storage, dropped capabilities, and systemd cgroup limits remain mandatory.

## Consequences

- Provider CLIs behave like host CLI invocations and avoid namespace-specific DNS/routing failures.
- Runtime can listen on loopback and Workers can call it directly.
- Workers can reach host services, other Workers, and arbitrary outbound destinations; network
  isolation is explicitly outside the Sandbox security boundary.
