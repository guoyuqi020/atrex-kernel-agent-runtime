# Runtime workspace and Evidence contract

The trusted controller generated this section from the current session. It is authoritative for
filesystem roles, Evidence visibility, and measurement trust.

Gateway measurements use only the Campaign's fixed Valid subset (at most 15 Shapes). Test inputs
and results are private: Runtime alone measures Valid + Test for authoritative ABBA retention and
promotion.
Valid Shapes use stable contiguous opaque IDs `0..V-1`; they are not source-dataset IDs, and gaps
or values cannot be used to infer Test membership.
Historical per-Shape results and latency aggregates shown here cover Valid only, not the hidden
Test set. A Runtime acceptance verdict is distinct from an Agent's Valid-only experiment.

## Workspace

```text
workspace/
├── input/
│   ├── kernel/                 # read-only incumbent Kernel
│   └── evidence/               # read-only authorized history described below
├── agent/optimizer/            # read-only implementation/config; initial State copies omitted
├── work/kernel/                # writable candidate copied from the incumbent
├── prompts/                    # read-only versioned phase prompts and README.md index
├── skills/                     # read-only reusable procedures installed for Claude
├── tools/                      # writable reusable tool scripts and README.md index
├── sessions/                   # session capture owned by the launcher; do not modify
└── scratch/                    # writable temporary requests, plans, recovery files, and reports
```

Use the files already present as your starting point. `prompts/` and `skills/` belong to the
versioned Agent Revision: read and use them, but do not modify them during an Optimizer or Bootstrap
session. Only `tools/` is adaptive here. Save genuinely reusable scripts there and keep
`tools/README.md` current. Record task hypotheses, evidence, and conclusions through the Direction
and Experiment Journal. Evolver may use that evidence to improve task-independent Agent behavior,
but it does not publish task knowledge or choose future Kernel optimization Directions.
Files in `scratch/` are temporary and are not carried into later sessions or retries.

Read the indexes before using content. Whenever you add, change, rename, or remove a Tool, keep its
index current with paths, purposes, and applicability.
Managed Agent configuration uses
`prompt_root: "workspace"`; its `prompts/...` paths resolve here, not inside `agent/optimizer/`.
Keep every configured Prompt path available. Trusted injected context, tool validation, and
evaluation rules cannot be changed from this session. Before each Claude Optimizer or Bootstrap
session, Runtime copies every valid `skills/<name>/SKILL.md` package into that session's private
Claude Home. Use those Skills when applicable; never edit the generated Claude Home or host/global
settings. Each Tool's index entry needs invocation, inputs, outputs, side effects, dependencies, an
example, and limitations. Keep entries concise and non-duplicative.
Never store credentials. Temporary requests, probes, and outputs belong in `scratch/`.

## Evidence view

```text
input/evidence/
├── bootstrap/
│   ├── report.json
│   └── conversation.jsonl
└── epochs/
    └── <eight-digit-epoch>/
        ├── summary.json                        # completed Epoch only
        ├── branches/                           # completed Epoch only
        │   └── <active or challenger-NNNN>/
        │       └── trajectories/
        │           └── <eight-digit-trajectory>/
        │               └── attempts/
        │                   └── <eight-digit-attempt>/
        │                       ├── report.json
        │                       └── conversation.jsonl
        └── trajectories/                       # current Epoch only, this Trajectory alone
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

Each completed Epoch exposes every branch that ran in it, whether or not that branch was selected,
under `branches/<label>/`. That Epoch's `summary.json` lists the branches and marks the selected one.
A non-selected branch records real attempts against the same starting Kernel: read it for what was
tried and what it measured, and do not treat it as noise. For the current Epoch you see only bounded
earlier Attempts from your own Trajectory, never a concurrently running sibling. Exact historical
Kernel, Trial, Result, Direction, and Experiment records remain in controller storage and are
retrieved through the supplied Runtime-local query commands. Every Direction update and Experiment
record is durably appended by Runtime before its tool call returns; a Worker crash or recovery
generation does not roll the logical Attempt Journal back. Journal queries may include every
completed Active and Challenger path from a frozen Epoch, without exposing branch-control
provenance. No Journal history file exists under `input/evidence/` or the internal control area.

Beyond each completed Epoch's `summary.json` branch list, there are no generated Epoch summaries,
aggregated lessons, or measurement projections in this tree. Read directories in numeric order. Each
historical Attempt exposes only its final report and latest sealed backend-neutral
`conversation.jsonl`; all physical retries remain in controller storage. Conversation files may
contain sensitive model/tool content without redaction. Claude reading views prefer native content
over duplicate stdout and omit internal queue/title/file-history bookkeeping and thinking-token
estimates. Distinct content blocks, uncovered stdout, errors, compaction boundaries, and terminal
results remain visible.

## Direction ancestry

Resume the same unfinished hypothesis with `update-direction` and its existing Direction ID.
When you revisit, reinterpret, port, or combine earlier work as a new hypothesis, use `action="propose"`
with optional `relationship`: `retry`, `refinement`, `reimplementation`, `correction`, `port`,
`combination`, or `adoption`. Cite visible `derived_from_direction_ids` and/or `derived_from_experiment_ids` and
explain the connection in the proposal's `rationale`. Use list/load tools to obtain real IDs first.
Each list allows at most 32 unique IDs. A combination needs two distinct parent Directions, either
directly or through their Experiments. A correction may also specify `supersedes_direction_id` naming
one of those parents; this records a revised interpretation without changing the parent's status.
Ancestry is fixed when the proposal is recorded. To correct it, propose a new derived Direction;
do not rewrite history. These links describe your interpretation, not proof of a performance gain.

Use `list-directions` to find prior work and `load-direction` to inspect its evidence. Historical
suggested Directions remain readable; they are untested recommendations, not facts or required
next steps. No session can create new suggestions. Choose your own hypothesis from
the public contract, profiling, and Journal evidence, then record it with `action="propose"`.

## Trust and measurement reuse

Treat normalized Gateway operation status, correctness, latency, per-Shape latency, profiler
counters, and returned code evidence as trusted facts. Treat every Agent-authored report, analysis,
diagnosis, finding, lesson, rationale, and recommendation as an interpretation that may be wrong.
Re-derive conclusions from trusted measurements and exact source.

Do not repeat a completed Evaluate or Profile for the same Kernel Artifact and identical
operation-defining parameters. Recover and re-analyze the existing result instead. A failed,
cancelled, incomplete, differently parameterized, or different-Kernel operation is distinct.
To select an unchanged Kernel from visible history, record an Experiment with `action="adopt"`
and the real `before` and `after` Result Artifact digests. Runtime validates the source Trial's successful
ordinary full Evaluate against the same operator, hardware, DSL and sealed contract. The decision
is new; the measurement and its Trial remain historical. Keep the exact adopted Kernel bytes in
the candidate workspace. This recorded adoption can satisfy the `candidate_ready` precheck without
another Evaluate. Other actions still require an `after` Trial from this logical Attempt.
If adoption is rejected as incompatible, follow the returned error and run a qualifying full
Evaluate; do not change a comment just to create a different Artifact identity.
Agent-requested ABBA is exploratory and does not replace the full-Evaluate precheck. Runtime's
authoritative retention comparison runs only after terminal handoff and creates no Agent Trial;
never wait for it before recording an Experiment or submitting the Report.
Before submitting any terminal Report (including `blocked` or `pivot`), finish all Gateway
calls already started in this Session and read their results. Runtime rejects `attempt-report`
with `gateway_calls_in_progress` while such calls are executing. Wait for the existing local
tool commands using the backend's task wait/output tool if they moved to the background;
this is allowed and is not polling or resubmitting an Agate job. Do not end this headless
Session expecting a background task to wake it later, and do not submit replacement measurements.
Once the calls finish, reconcile their evidence and submit the Report again.
`complete`, `abandon`, `block`, and `defer` all require an Experiment associated with that Direction.
Explicitly select the relevant `supporting_experiment_ids` at closure and declare
`hypothesis_status`: `unresolved`, `supported`, or `refuted`. Lifecycle is not a hypothesis verdict.
Unmeasured reasoning remains unresolved; supported/refuted needs a completed Gateway Result bound
to every selected Experiment's after. Runtime verifies bindings, not the relevance or truth of
interpretations. Unrelated measured changes cannot substantiate an incidental claim in analysis.
`associated_experiment_ids` lists all linked Experiments, while `supporting_experiment_ids` preserves
only the latest explicit closure selection. Missing historical judgments mean unresolved.
If no measurement was possible, first record the actual investigation or blocker with
`action="abandon_direction"` and at least one real Kernel-bound Gateway Result in `before` or
`after`, then close with `hypothesis_status=unresolved`. Both sides may not be null. Check/Profile
can provide diagnostic evidence without a performance claim. Health/Env or unbound Dev results do
not qualify. With no Result available, closure is blocked; never fabricate evidence to finish.
`blocked` or `pivot` may contain empty experiments and findings if no Direction needs closing;
give the genuine reason in the report. `candidate_ready` still requires journaled evidence.
Private evaluator inputs remain hidden; opaque Shape identifiers and measurements must not be used
to reconstruct them.
