# Deployment and Operations

English | [中文](operations.zh.md)

## Process topology

Run the Agate-compatible Gateway and production GPU Wiki as external services. Run one Runtime service process and one or more independent Campaign task/scheduler processes. SQLite deployments are single-node. Runtime databases use rollback journals so Lima `virtiofs` workspaces remain safe across the Runtime and CLI processes; do not use a remote filesystem whose POSIX file locks are not reliable.

Optimizer and Evolver are deployment-approved Git repositories pinned by full commit. Bootstrap imports the Optimizer and its Skill submodules only from their initialized local checkouts; it never fetches them. The `third_party/atrex-bench` submodule is the locally available, commit-pinned trusted evaluator source used by ABBA and optional Roofline construction. Run `git submodule update --init --recursive` during deployment preparation, before Runtime startup and repository verification.

## Startup

1. Create storage/workspace directories with restricted ownership.
2. Choose a Worker boundary. For `sandbox`, install bubblewrap and enable cgroup v2, provision the
   non-root `launcher.sandbox.worker_user`, grant narrowly controlled system-manager transient-service
   authority, and run `sudo .venv/bin/python scripts/validate-linux-sandbox.py`. For `container`,
   run the complete deployment in a dedicated OCI container with bubblewrap and explicit
   memory/CPU/PID limits. It needs no systemd, writable cgroup hierarchy, sudo, or per-Session
   cgroup, but the OCI security policy must allow bwrap's namespace operations.
3. Export a Base64 capability signing key of at least 32 decoded bytes, the admin bearer token,
   selected Agent-provider credentials, Agate credentials, and optional Wiki credentials. Provider
   CLIs connect directly through the host network; no Runtime model proxy is required.
4. In `sandbox`, make required provider CLI/config paths system-readable and add only the minimum
   immutable paths to `launcher.sandbox.read_only_bind_paths`. In `container`, mount only the required
   provider login state into the outer container; Runtime projects the allowlisted subset read-only
   into each bwrap Session Home. Never expose a Docker socket, private evaluator data, or unrelated
   host storage.
5. Validate configuration by starting the service and checking `/readyz`.
   After ASGI startup, Runtime also probes Agate immediately and every
   `agate.health_check_interval_s` seconds (30 seconds by default). Initial state, failure, and
   recovery are written to the service log. This observation does not change `/healthz` or
   `/readyz`; a temporary external outage does not stop the trusted control plane.
   Every operational Agate SDK request uses the same persistent transient-failure policy: retry after 5, 10,
   20, and 40 seconds; after the fifth consecutive failure, continue retrying every 60 seconds
   without a terminal attempt limit. A successful call resets the count for the next request.
   Non-retryable 4xx validation or authorization errors still return immediately because the
   request must change before it can succeed. Terminal job results such as compilation or
   correctness failure are results, not transport errors, and are not resubmitted by this policy.
   One collection failure is intercepted: Job `status=failed`, `error_class=infra`,
   `reason=logs_unavailable`, and `error.details.backend_state=succeeded`. During submit-and-wait,
   Runtime resubmits the same workload with a replacement idempotency key and waits for the new
   Job ID; it does not poll the failed Job again. Delays are 5, 10, 20, 40, then 60 seconds
   indefinitely, until recovery or cancellation. Evaluate retries only the affected Shape batch.
   This also applies to Profile, Dev, Check, Disassemble, Bootstrap, seed evaluation and ABBA.
   Each replacement follows normal Job binding/event recording. Explicit polls remain read-only;
   other Job outcomes, transport retries and Agent error formatting are unchanged.
   The periodic health observation remains a bounded one-shot probe; it never owns Campaign work.
6. Keep Runtime, Gateway, and optional Wiki available while running Campaign schema-v3 bootstrap;
   Core baseline sessions call back through Runtime.
7. Schedule a Campaign by absolute target Epoch. GPU Wiki needs only its query service; Runtime has
   no feedback drainer or delivery process.

Example commands are in the repository [README](../README.md). Configuration contains credential environment-variable names only.

## Worker workspace ownership

Each Sandbox workspace root must be owned by `launcher.sandbox.worker_user`. Runtime's host check
asks systemd to create the Attempt, Evolution, Problem Generalization, and Lineage Bootstrap roots
and probe directly as that user. A cross-process lock serializes this preparation when three DSLs
bootstrap concurrently. The trusted scheduler may still assemble a Run, but hands the current Run
to the Worker before bwrap starts.

On Lima `virtiofs`, `chown` may silently leave ownership unchanged. Do not pre-create
`*-workspaces` as root. Runtime can safely recreate an empty wrong-owner root, but fails closed and
preserves any real non-empty mismatch. Because root and the Worker may see different numeric owners
on virtiofs, Runtime verifies a root-view mismatch with a second `stat` executed as the Worker; it
accepts the existing root only on an exact Worker UID/GID match. If every DSL fails at
`Sandbox path ownership`, inspect `stat` both normally and through `systemd-run --uid=<worker>`, plus
`findmnt -T`; do not recursively chmod or delete non-empty state. See the
[Production runner](../scripts/production/README.md) for the exact workflow.

## Kernel catalog

Use `list-kernels --campaign` to enumerate every DSL lineage, or `--lineage` for one lineage. Use
`show-kernel` for the producing Agent revision, Attempt context, primary authoritative evaluation,
and all durable repeated measurements. The authenticated HTTP API additionally exports the bounded
exact source Artifact. Catalog queries are read-only and do not require Runtime quiescence.
Use `--format table` for a human-readable `v0`/`v1` history with parent links, timestamps, retention
results, latency, and parent-relative change; JSON remains the default for automation.

```bash
atrex-kernel-agent-runtime list-kernels --config runtime.json --campaign campaign_xxx
atrex-kernel-agent-runtime list-kernels --config runtime.json --lineage lineage_xxx --format table
atrex-kernel-agent-runtime show-kernel --config runtime.json --kernel kernelrev_xxx
```

## Attempt history

Use Attempt history to account for every `X` optimization session, including sessions that pivoted,
were blocked, or produced no Candidate. Kernel history intentionally remains sparse.

```bash
atrex-kernel-agent-runtime list-attempts --config runtime.json --lineage lineage_xxx --format table
atrex-kernel-agent-runtime show-attempt --config runtime.json --attempt attempt_xxx
```

## Epoch winner history

Use Epoch history to see the Active before each competition, all Challengers, the winner, the
promotion decision, and the starting/global-best Kernel versions.

```bash
atrex-kernel-agent-runtime list-epochs \
  --config runtime.json --lineage lineage_xxx --format table
```

## Kernel Agent history

Use the independent Agent catalog to inspect Bootstrap `agent-v0`, every Evolver Challenger, its
actual parent, promotion result, and active state. One Agent version may produce several Kernel
versions.

```bash
atrex-kernel-agent-runtime list-agent-revisions \
  --config runtime.json --lineage lineage_xxx --format table
atrex-kernel-agent-runtime show-agent-revision \
  --config runtime.json --agent-revision agentrev_xxx
```

## Attempt evaluation history

Use `list-evaluations` to inspect every exploratory Kernel submitted by an Agent and the
Runtime-owned comparator/finalization records for one Attempt. The retention comparator's Candidate
aggregate is authoritative; Runtime does not add another independent final Eval afterward.
`show-evaluation` identifies one immutable pair;
`--source` includes the exact files evaluated at that step and `--result` includes the complete raw
Gateway response. These records exist even when the Kernel was incorrect, reverted, or never became
a Kernel Revision. The CLI opens Gateway Control directly and therefore requires the configured
capability-signing-key environment.

```bash
atrex-kernel-agent-runtime list-evaluations \
  --config runtime.json \
  --attempt attempt_xxx

atrex-kernel-agent-runtime show-evaluation \
  --config runtime.json \
  --evaluation geval_xxx \
  --source \
  --result
```

## Bootstrap execution history

One stable Bootstrap Attempt may have multiple physical Core Sessions. Each Session is retained as
an append-only recovery Generation, including terminal status, failure reason, exact workspace,
provider-token usage, Session Trace/Report Digests, and available authoritative result identities.
The CLI requires the signing-key environment named by Runtime configuration because it opens Gateway
Control directly. The authenticated HTTP equivalents are
`GET /v1/admin/bootstrap-attempts/{attempt_id}/runs` and `/runs/{generation}`.

```bash
atrex-kernel-agent-runtime list-bootstrap-runs \
  --config runtime.json \
  --attempt attempt_xxx

atrex-kernel-agent-runtime show-bootstrap-run \
  --config runtime.json \
  --attempt attempt_xxx \
  --generation 2
```

New failures include Attempt, Generation, and Run identity in the exception and emit
`bootstrap.lineage_baseline_failed` lifecycle events.

## Optimizer debug shell

`dev-shell --lineage` creates or reuses the current Epoch's first Active Attempt while holding the
lineage fence. It materializes the same workspace and injects Gateway/Wiki capability, but starts
only interactive `zsh/bash`. `--attempt` creates a new run workspace for an existing running
Attempt. Exiting does not complete the Attempt; an operator must explicitly resume the Campaign or
retain the scene. Never run it concurrently with the same lineage's scheduler or log the complete
capability-bearing environment.

```bash
atrex-kernel-agent-runtime dev-shell \
  --config runtime.json \
  --lineage lineage_xxx \
  --shell zsh
```

## Evolver debug shell

`evolver-dev-shell` requires both a Lineage ID and an existing absolute Epoch number. It holds the
lineage fence, resolves the Campaign-pinned Evolver commit, and reconstructs the selected Epoch's
frozen Evolution workspace and environment without executing the Evolver backend. The view is
anchored to the Epoch's parent Agent and Evidence checkpoint. It includes already attached
same-Epoch Challengers, but excludes Kernels produced by the target Epoch and every later Epoch.
Exiting retains the workspace and changes no Epoch, Challenger, selection, or promotion state.

```bash
atrex-kernel-agent-runtime evolver-dev-shell \
  --config runtime.json \
  --lineage lineage_xxx \
  --epoch 2 \
  --shell zsh
```

The command requires the Evolver worker's configured inherited environment because it prepares the
same launch contract. In Sandbox mode the interactive shell uses the same bubblewrap mount/process
boundary and cgroup limits with shared host networking. It does not require a Gateway capability.

For layout and CLI testing without an existing Lineage or Epoch, use
`temporary-evolver-dev-shell --config runtime.json --campaign campaign.json`. It synthesizes an
active `agent-v0` from the pinned Core base and initial Evidence, starts no Agent or Runtime HTTP
service, fabricates no Kernel measurements, and destroys its workspace on exit.

## Worker Session inspection

Every model-backed physical process is indexed independently of its domain outcome:

```bash
atrex-kernel-agent-runtime list-worker-sessions --config runtime.json --lineage "$LINEAGE_ID" --format table
atrex-kernel-agent-runtime list-worker-sessions --config runtime.json --attempt "$ATTEMPT_ID"
atrex-kernel-agent-runtime show-worker-session --config runtime.json --session "$WORKER_SESSION_ID"
```

Authenticated API equivalents are `GET /v1/admin/campaigns/{id}/worker-sessions`,
`GET /v1/admin/lineages/{id}/worker-sessions`,
`GET /v1/admin/epochs/{id}/worker-sessions`,
`GET /v1/admin/attempts/{id}/worker-sessions`, and
`GET /v1/admin/worker-sessions/{id}`. Failed and timed-out processes remain visible even when no
trace could be sealed; the catalog still retains their exact workspace and terminal diagnostic.

## Recovery and maintenance

Repeated Bootstrap, absolute Epoch targets, and Task creation keys are idempotent. Bootstrap retries
rotate Capability Generation while retaining earlier Runs and Gateway operations. Failed Epoch
recovery requires an operator key and reason. Cancellation is cooperative and only finalizes at
safe transitions. Wiki queries are frozen synchronously; there is no delivery queue to drain.

Before SQLite backup, quiesce Runtime, schedulers, task workers, and Agent processes, then back up
Registry, Gateway control, Agate jobs, and Artifact store consistently. Restore into an isolated
environment first and run readiness plus identity checks. Runtime SQLite files use rollback
journals; a leftover `-wal`/`-shm` pair indicates state created by an older release and should only
be migrated while every owning process is stopped.

Artifact and workspace GC are bounded and dry-run by default. Apply them only while the deployment is quiescent and after incident-retention requirements are satisfied. Never delete objects directly from the CAS.

Rotate the capability signing key only after quiescing or intentionally invalidating outstanding capabilities. Rotate provider/Agate/Wiki credentials through environment/secret management; never rewrite immutable Artifacts. Record lifecycle events and relevant Artifact digests during incidents.

## Security warning

`launcher.mode=sandbox` and `launcher.mode=container` are production modes. Both fail closed when
bwrap or the configured resolver is unavailable and never fall back to development. `sandbox` also
requires systemd/cgroup-v2; `container` relies on outer OCI resource limits instead. Both intentionally
share their surrounding network namespace, so they do not isolate host/container services, Worker
traffic, or outbound destinations.
`launcher.mode=development` is intentionally unisolated and suitable only for trusted local work.
Before production promotion, run the target-image suite: sibling/host reads, writes outside
`~/workspace`, direct Internet/DNS success, namespace escape, fork/memory/CPU exhaustion, timeout
cleanup, and credential-mount scope.

The repository acceptance script exercises the implemented filesystem, capability, cgroup-resource,
direct-public-network, DNS, and Runtime-port behavior. Its `sudo` invocation creates the
transient service through the system manager, while systemd executes bwrap as the configured
non-root Worker account. Passing it in Lima is a reference smoke result, not a substitute for running
it on the production kernel and image.

Production promotion additionally requires every gate in [Testing and Production Acceptance](testing-and-acceptance.md).
