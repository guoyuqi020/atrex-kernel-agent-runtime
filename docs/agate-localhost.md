# Agate localhost backend

English | [中文](agate-localhost.zh.md)

Runtime defaults to the official remote Agate service. This document describes the optional,
explicit localhost override at `http://127.0.0.1:8000` with GPU selector `local`. Localhost is an
Agate deployment backend, not a new Runtime evaluator or an SDK transport:
Evaluate, ABBA, Profile, Dev, Check and Disassemble keep the same SDK calls and Runtime recording,
deduplication, retry and result-projection policies.

## Deploy Agate on the GPU node

The HTTP client dependency does not include the Agate server. Use the official server checkout and
its GPU Python environment. Copy [the local server configuration](../scripts/shared/agate-local.example.json)
outside Git; set `python_bin`, `gpu_model`, device number and absolute state/work directories for
that machine. Agate's `app.local.detect.detect_local_device()` can discover the device fields.

From that Agate server checkout:

```bash
AGATE_CONFIG_FILE=/path/to/agate-local.json python3 -m app.main
```

Expose the service on loopback port 8000. Agate server configuration and authentication remain
deployment-owned. If authentication is enabled, export the server's matching `AGATE_AK` and
`AGATE_SK`; an unauthenticated loopback service may omit both. Generated Runtime configs store only
credential environment variable names. Localhost with either credential present still selects AK/SK
auth and requires the complete pair. Explicit non-loopback endpoints retain AK/SK auth.

The local executor runs submitted code as subprocesses with the Gateway user's permissions; it is
not a security sandbox. Use a dedicated GPU execution host/container without Runtime credentials,
Registry files or unrelated data. Optimizer bwrap isolation does not isolate Gateway-executed code.
Do not deploy this server in the Optimizer/Evolver sandbox, and do not expose it publicly without
appropriate authentication and execution isolation.

## Connect Runtime

Override the remote defaults explicitly:

```bash
export AGATE_URL=http://127.0.0.1:8000
export AGATE_GPU=local
bash examples/agate/check-service.sh
bash scripts/production/services.sh start --workspace workspaces/production/control-local
bash scripts/production/campaign.sh start \
  --service-workspace workspaces/production/control-local \
  --kernel suite/operator --backend claude
```

Agate must already be running. Service scripts manage Runtime/Wiki only; they do not install, start
or stop Agate. `local` means the GPU of the Agate server, not the Agent sandbox. With Runtime in a
container, ensure this endpoint is actually reachable from that container; otherwise set `AGATE_URL`
to the GPU executor's accessible address and supply its credentials.

GDN/FA4 input packs retain their declared
`L20D` scheduling target; configure `L20D` as an alias of the local cluster when running those packs
on the matching GPU. Runtime continues to query Agate for the actual architecture shown to Agents.
Existing Campaigns retain sealed GPU/input identities, so use a new workspace when changing the
execution environment. Updating a template does not rewrite already generated service configs.
