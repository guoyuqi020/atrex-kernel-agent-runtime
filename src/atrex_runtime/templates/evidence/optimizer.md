# Runtime workspace and Evidence contract

The trusted controller generated this section from the current session. It is authoritative for
filesystem roles, Evidence visibility, and measurement trust.

## Workspace

```text
workspace/
├── input/
│   ├── kernel/                 # read-only incumbent Kernel
│   └── evidence/               # read-only authorized history described below
├── agent/optimizer/            # read-only Agent implementation and built-in resources
├── reference/                  # read-only pinned upstream GPU projects
├── work/kernel/                # writable candidate copied from the incumbent
├── skills/                     # writable reusable methods
├── tools/                      # writable reusable tools; README.md is mandatory
├── sessions/                   # session capture owned by the launcher; do not modify
└── scratch/                    # writable temporary requests, plans, recovery files, and reports
```

`skills/` and `tools/` persist across serial Attempts in this Epoch for the same Lineage, Agent
Revision, and Trajectory. At the next Epoch boundary, every Active Trajectory starts from an
independent copy of the prior winner's best-Kernel Trajectory terminal State; every Challenger
starts from its Evolver-sealed revision State. Parallel Trajectories have independent stores.
Inspect both directories before creating duplicate content.
Keep only reusable methods and implementations there—never credentials, raw traces, temporary
requests, or one-off results. Document every reusable tool's purpose, invocation, inputs, outputs,
side effects, dependencies, example, and limitations in `tools/README.md`.

Controller-internal files and metadata are intentionally omitted from this Agent-facing contract.

## Evidence view

```text
input/evidence/
├── bootstrap/
│   ├── report.json
│   └── conversation.jsonl
└── epochs/
    └── <eight-digit-epoch>/
        └── trajectories/
            └── <eight-digit-trajectory>/
                └── attempts/
                    └── <eight-digit-attempt>/
                        ├── report.json
                        └── conversation.jsonl
```

Bootstrap is the special pre-Epoch Attempt that establishes the initial Agent and Kernel. Epochs
then form one serial Lineage. Every Trajectory in an Epoch starts from the same Kernel and runs a
serial search chain; a retained Kernel advances only that Trajectory, while a rejected Kernel does
not. Different Trajectories are independent.

After an Epoch completes, the controller independently selects the next active Agent Revision and
the fastest retained correct Kernel. They may have different producers. `input/kernel/` is always
the authoritative current starting point.

Ordinary Evidence contains the promoted completed Agent lineage plus bounded earlier Attempts from
the current Trajectory. It does not expose a concurrently running sibling or branch. Exact
historical Kernel, Trial, Result, Direction, and Experiment records remain in controller storage and
are retrieved through the supplied Runtime-local query commands. Every Direction update and
Experiment record is durably appended by Runtime before its tool call returns; a Worker crash or
recovery generation does not roll the logical Attempt Journal back. Journal queries may include every
completed Active and Challenger path from a frozen Epoch, without exposing branch-control
provenance. No Journal history file exists under `input/evidence/` or the internal control area.

There are no generated Epoch summaries, aggregated lessons, or measurement projections in this
tree. Read directories in numeric order. Each historical Attempt exposes only its final report and
latest sealed backend-neutral `conversation.jsonl`; all physical retries remain in controller
storage. Conversation files may contain sensitive raw model/tool content. Retention omits only
high-frequency Claude thinking-token estimate telemetry.

## Trust and measurement reuse

Treat normalized Gateway operation status, correctness, latency, per-Shape latency, profiler
counters, and returned code evidence as trusted facts. Treat every Agent-authored report, analysis,
diagnosis, finding, lesson, rationale, and recommendation as an interpretation that may be wrong.
Re-derive conclusions from trusted measurements and exact source.

Do not repeat a completed Evaluate or Profile for the same Kernel Artifact and identical
operation-defining parameters. Recover and re-analyze the existing result instead. A failed,
cancelled, incomplete, differently parameterized, or different-Kernel operation is distinct.
Private evaluator inputs remain hidden; opaque Shape identifiers and measurements must not be used
to reconstruct them.
