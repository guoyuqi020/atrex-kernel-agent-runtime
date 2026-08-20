# Decision 0047: Worker-Created Sandbox Roots

## Status

Accepted.

## Decision

The Sandbox host check creates Attempt, Evolution, Problem Generalization, and Lineage Bootstrap
workspace roots and its probe directly through a systemd transient service running as the
configured non-root `worker_user`. Concurrent scheduler processes serialize this preparation with
a filesystem lock. Trusted Runtime code may assemble a Run as root, but hands the complete current
Run to the Worker before bwrap starts.

Runtime does not use root-create-then-chown for root preparation. An empty foreign-owned root left
by a failed preparation may be removed and recreated as the Worker. A non-empty ownership mismatch
normally fails closed and is preserved for operator inspection. Lima `virtiofs` may expose the
same path with different numeric ownership to root and the Worker. When root's view mismatches,
Runtime therefore performs a second `stat` through a systemd transient service running as the
Worker. Runtime accepts a non-empty root only when that Worker view exactly matches the configured
Worker UID/GID. A campaign restart may briefly retain the system manager's `root:root` virtiofs
view after the old bwrap service exits, so Runtime retries only that characteristic mismatch for a
bounded 30 seconds. Any other mismatch, or a root view that does not converge, still fails closed.

## Consequences

- Lima virtiofs is supported even when `chown` reports success without changing visible ownership.
- Existing non-empty Bootstrap state survives a false mismatch caused only by virtiofs views.
- Three DSL Bootstrap processes can share one Runtime configuration without racing host probes.
- Operators must not pre-create Worker roots with sudo.
- Runtime never recursively deletes or rewrites a non-empty wrong-owner root.
