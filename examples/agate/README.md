# Real Agate service example

English | [中文](README.zh.md)

This example calls a real remote Agate service through the official `agate` CLI resolved from the
current `PATH`. It does not start or emulate Agate locally.

The canonical candidate and matching trusted PyTorch reference, input generator, and shape metadata
live once under `examples/shared/vecadd/`. These scripts consume those shared read-only inputs.

## Configure the connection

Set the real service URL and, when the service requires authentication, exactly one supported
credential form:

```bash
export AGATE_URL="https://your-agate-service.example.com"

# Bearer token authentication:
export AGATE_TOKEN="..."

# Or AK/SK authentication instead:
# export AGATE_AK="..."
# export AGATE_SK="..."
```

Do not put credentials in `runtime.json` or commit them to the repository. The official CLI reads
`AGATE_TOKEN` or `AGATE_AK`/`AGATE_SK` from the environment. No authentication variable is needed
for a service configured without authentication.

The scripts run `agate` from the active environment and never pin a repository virtual-environment
path. Set `AGATE_BIN` to an explicit command or platform-local path when needed. A virtual
environment created on macOS cannot be reused inside Linux; create and activate a Linux environment
before running these scripts.

First inspect the service and its exact GPU environment names:

```bash
bash examples/agate/check-service.sh
export AGATE_GPU="H20"  # replace with one value returned by the env command
```

`check-service.sh` performs service queries only; it does not submit a GPU job.

## Submit real GPU work

Each command below submits work to the configured remote service and may consume GPU resources:

```bash
# Correctness plus performance evaluation; waits for the result.
bash examples/agate/evaluate.sh

# Survey-level profiling; set AGATE_PROFILE_LEVEL=sol for NCU SpeedOfLight.
bash examples/agate/profile.sh

# Compile, optionally under compute-sanitizer.
bash examples/agate/check-kernel.sh
AGATE_SANITIZE=memcheck bash examples/agate/check-kernel.sh

# Produce SASS/PTX output according to service support.
bash examples/agate/disassemble.sh

# Execute a development command on the remote worker. The default is nvidia-smi.
bash examples/agate/dev.sh
bash examples/agate/dev.sh 'python -c "import torch; print(torch.cuda.get_device_name())"'
```

Useful optional variables are:

- `AGATE_CORRECTNESS_CASES` and `AGATE_BENCH_ITERS` for evaluation workload size;
- `AGATE_MODE=correctness` to skip performance measurement;
- `AGATE_ARCH` and `AGATE_SANITIZE` for `check-kernel.sh`;
- `AGATE_DISASSEMBLY_FORMAT=auto|sass|ptx` for `disassemble.sh`;
- `AGATE_PROFILE_LEVEL=survey|sol|deep`; deep profiling also requires
  `AGATE_KERNEL_NAME`;
- `AGATE_HTTP_TIMEOUT` for one HTTP request (default `1800` seconds),
  `AGATE_JOB_TIMEOUT` for the remote Worker budget (default `3600` seconds), and
  `AGATE_WAIT_TIMEOUT` for total client-side waiting (default `3900` seconds);
- `AGATE_POLL_SECONDS` for the polling interval (default `5` seconds).

The waiting timeout is deliberately greater than the remote Job timeout so the client has time to
retrieve the terminal result. `evaluate-async.sh` uses the HTTP and Job timeout settings but does
not wait or poll.

There is deliberately no `run-all` script: these operations allocate real remote resources.

## Asynchronous jobs

Submit without waiting, retain the returned Agate job ID as evidence, then inspect or cancel it:

```bash
bash examples/agate/evaluate-async.sh
bash examples/agate/get-job.sh ev_your_job_id
bash examples/agate/list-jobs.sh --kind eval --limit 20
bash examples/agate/cancel-job.sh ev_your_job_id
```

## Relationship to Runtime

These scripts are an operator-facing smoke test of Agate and the example kernel. They intentionally
bypass Runtime. During an Optimizer session, the Agent should call the Runtime Gateway Tool instead:
Runtime binds the request to an Attempt, enforces capability and ownership checks, injects the
sealed evaluation contract, records idempotency and evidence, and keeps Agate credentials outside
the Agent workspace.

To connect Runtime itself to the same service, put the non-secret connection policy in the `agate`
section of `runtime.json`, select `auth_mode` (`none`, `token`, or `ak_sk`), and name the
corresponding credential environment variables. For example, token mode adds
`"token_env": "AGATE_TOKEN"`; AK/SK mode adds `"access_key_env": "AGATE_AK"` and
`"secret_key_env": "AGATE_SK"`. A Runtime-based example should make that change in its own
`runtime.json` rather than importing another example's configuration.
