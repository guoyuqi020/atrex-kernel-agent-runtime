# Three-Epoch Evolution Example

English | [中文](README.zh.md)

Defaults connect to the remote Agate service; explicit environment overrides remain supported.

This example runs one Triton Lineage through Epoch 3 and exercises two controlled Agent evolutions:

- Epoch 1 explicitly replicates Active, then Active and its replica run one Attempt each;
- Epoch 2 creates one Challenger from Epoch 1 Evidence, then Active and Challenger run one Attempt each;
- Epoch 3 creates one Challenger from Epoch 2 Evidence, then Active and Challenger run one Attempt each;
- the target stops at Epoch 3, so Runtime never creates Epoch 4 or invokes Evolver again.

The Campaign grants one Challenger slot and an exact budget of two Optimizer Attempts per Epoch.
The Agent-owned Workflow decides the first-Epoch replication, later evolution timing, Branches,
Trajectory layout, Kernel routing, and State routing. The whole example contains six fresh
Optimizer Sessions and two Evolver Sessions. Runtime evaluates the resulting Branches independently
and promotes a Challenger only when it wins.

This directory owns its `runtime.json` and three-Epoch `campaign.json`. It uses only the canonical
VecAdd inputs and generic helpers under `examples/shared/`, never another runnable example.

Export the Agate and pinned Evolver Bundle requirements first:

```bash
export AGATE_URL="https://atrex-gateway.alibaba-inc.com"
export AGATE_AK="..."
export AGATE_SK="..."
export AGATE_GPU="L20N"
export QODER_PERSONAL_ACCESS_TOKEN="..."
```

Run Runtime, Bootstrap, all three Epochs, inspection, and cleanup with one command:

```bash
bash examples/evolution/run.sh
```

Inspection begins with an Epoch winner table showing `ACTIVE_BEFORE`, `CHALLENGERS`, `WINNER`, and
the `active_retained` or `challenger_promoted` decision for every Epoch.

For split-terminal debugging:

```bash
bash examples/evolution/prepare.sh
bash examples/evolution/start-runtime.sh
# In another terminal with the same exported environment:
bash examples/evolution/run-campaign.sh
bash examples/evolution/inspect.sh
```

State defaults to `workspaces/evolution-example/`. Reruns retain the workspace-pinned Optimizer and
Evolver commits and resume the first incomplete durable step. Once Epoch 3 is complete, rerunning
only reports the existing result.
