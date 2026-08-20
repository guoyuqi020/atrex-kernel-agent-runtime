# Runnable Campaign Bootstrap example

English | [中文](README.zh.md)

This example bootstraps one real Triton VecAdd lineage. It starts the Runtime control service,
launches the commit-pinned Core in `framework_baseline` mode, lets Core submit its candidate through
the Runtime Gateway Tool to a real remote Agate service, and registers a ready Lineage only after a
correct authoritative evaluation.

This is not a mock workflow. It invokes QoderCLI and consumes remote GPU resources. It does not start
a Local Agate service. GPU Wiki is disabled unless `ATREX_WIKI_URL` is explicitly provided.

This directory owns its [`runtime.json`](runtime.json) deployment template and
[`campaign.json`](campaign.json) topology. Both point only to the canonical read-only VecAdd inputs
under [`../shared/vecadd`](../shared/vecadd); no other runnable example is used.

## Prerequisites

Install the repository development environment and export the QoderCLI credential plus remote Agate
connection. `AGATE_GPU` must exactly match one environment returned by `agate env`.

```bash
# Optional when ~/.qoder and ~/.qodersec contain a valid login:
# export QODER_PERSONAL_ACCESS_TOKEN="..."
export AGATE_URL="https://your-agate-service.example.com"
export AGATE_AK="..."
export AGATE_SK="..."
export AGATE_GPU="H20"
```

The wrapper may pass `QODER_PERSONAL_ACCESS_TOKEN` through the configured Worker environment. When
it is absent, Runtime mounts host `.qoder` and `.qodersec` read-only into the Session Home. Agate
and Agent credentials are never written to generated files or retained in Workspace artifacts.

## Run Bootstrap

The recommended entrypoint starts its own Runtime, waits until it is healthy, runs Bootstrap,
prints the inspection result, and stops Runtime on success, failure, or interruption:

```bash
bash examples/bootstrap/run.sh
```

It refuses to proceed when a Runtime is already reachable on the configured port, so it never
stops a process it did not create. Runtime logs are saved to
`workspaces/bootstrap-example/runtime.log`.

For manual debugging, inspect the generated non-secret inputs first:

```bash
bash examples/bootstrap/prepare.sh
```

Then keep Runtime running in the first terminal:

```bash
bash examples/bootstrap/start-runtime.sh
```

Run the real Bootstrap from a second terminal with the same exported Agate variables, then inspect:

```bash
bash examples/bootstrap/bootstrap.sh
bash examples/bootstrap/inspect.sh
```

Bootstrap can take a long time because Core starts a real Agent session and evaluates the generated
baseline remotely. The default evaluation Job budget is 3600 seconds, Runtime's single-request
timeout is 1800 seconds, its total Agate wait timeout is 3900 seconds, and the example's per-Core-
Session Optimizer quota is 20,000,000 provider tokens. They can be changed before both commands are
started:

```bash
export AGATE_HTTP_TIMEOUT=3600
export AGATE_JOB_TIMEOUT=7200
export AGATE_WAIT_TIMEOUT=7500
export ATREX_OPTIMIZER_MAX_SESSION_TOKENS=25000000
```

The token quota is cumulative across model requests and includes uncached input, cache reads,
cache writes, and output tokens reported by the provider. It is not the model context-window size.
The generated Bootstrap topology can also be overridden with `ATREX_CHALLENGER_COUNT`,
`ATREX_CHALLENGER_START_EPOCH`, `ATREX_TRAJECTORIES_PER_BRANCH`, and
`ATREX_ATTEMPTS_PER_TRAJECTORY`; these values affect ordinary Epochs after the baseline has been
registered, not the Framework Baseline Session itself.

`inspect.sh` prints the Bootstrap result containing the durable identities, baseline Kernel, and
Agent revision. It does not automatically query the Kernel Catalog; use the Runtime
`list-kernels` command when the complete authoritative evaluation record is needed. Each Lineage
result includes the exact `bootstrap_attempt_id` and a structured
`baseline_kernel.producer`; its ordinary Epoch `attempt_id` remains null by design. In manual mode,
stop `start-runtime.sh` with `Ctrl-C` after inspection.

## Generated state and idempotency

The example writes generated configuration, local Runtime signing/admin secrets, SQLite databases,
Artifacts, workspaces, and the last result under `workspaces/bootstrap-example/`. That directory is
ignored by Git. The local secret file has mode `0600`; it contains only generated Runtime control
secrets, never Agate credentials.

For a new workspace, the default creation key is derived from `AGATE_GPU` and the current Core
commit. Once the generated Campaign definition and Runtime config exist, re-running preserves the original creation
key, Optimizer commit, Evaluation Contract, and Evolver commit. Local Core/Evolver HEAD changes do
not silently switch an existing Campaign. Use a different state directory when another immutable
Campaign should intentionally adopt newer commits; `ATREX_BOOTSTRAP_CREATION_KEY` applies while
creating that new workspace.

The checked-in example-owned definitions are:

- `campaign.json`: Campaign schema v3 template; its `lineages` keys select the DSLs and optional per-Lineage models;
- `runtime.json`: this example's complete Runtime deployment template.

The Evaluation Contract, Agent Problem, baseline Kernel, and initial Evidence are the shared
canonical VecAdd fixtures in `examples/shared/vecadd/`.
