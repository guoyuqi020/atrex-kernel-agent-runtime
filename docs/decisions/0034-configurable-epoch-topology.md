# 0034: Configurable Epoch Topology

English | [中文](0034-configurable-epoch-topology.zh.md)

## Status

Accepted.

## Context

A fixed Active-versus-one-Challenger Epoch conflates three independent budgets: how many Agent
designs to compare, how many independent Kernel-search paths each design receives, and how long each
path may continue learning. Calling the independent paths “lineages” also conflicts with the durable
DSL Lineage that owns Agent and Kernel history.

## Decision

Each durable DSL Lineage configures:

- `challenger_count` (`K`, zero or greater);
- `challenger_start_epoch` (positive, default `1`);
- `trajectories_per_branch` (`Y`, positive); and
- `attempts_per_trajectory` (`X`, positive).

One Epoch freezes one Active Agent, one starting Kernel, and one Evidence checkpoint. Before
`challenger_start_epoch`, its effective `K` is zero; from that Epoch onward Runtime invokes the
Evolver sequentially `K` times, and one invocation creates exactly one Challenger. Invocation
`i` receives a read-only catalog containing the Active revision, retained Agent history, and
Challengers `1..i-1`; future Challengers cannot be visible before they exist.

The Epoch has `1 + K` Branch slots. Active uses Challenger ordinal zero; each Challenger uses its
one-based creation ordinal. Every Branch launches `Y` Trajectories from the same starting Kernel.
Trajectories are independent and may run concurrently. Within a Trajectory, `X` Attempts run
serially, and a retained result becomes only that Trajectory's next input. Every Attempt is a fresh
Agent Session. The Epoch therefore contains `(1 + K) × Y × X` Optimizer Sessions and `K` Evolver
Sessions.

Kernel selection considers retained results from all Trajectories. Agent promotion compares the
Active score with every Challenger score and retains the incumbent on an exact tie. Runtime publishes
one Evidence checkpoint only after the entire Epoch is selected. Attempt Evidence exposes only prior
Attempts in the same Trajectory; completed Epoch Evidence preserves all measured Branch and
Trajectory outcomes.

## Consequences

Setting `K=0` cleanly disables Evolution while preserving the Epoch boundary. Increasing `Y`
improves independent search coverage without leaking intermediate Kernel state between paths.
Increasing `X` deepens one path's serial learning. Resource planning is explicit from the Session
formula. Registry schema 19 preserves existing Lineages by migrating `challenger_start_epoch` to
`1`.
