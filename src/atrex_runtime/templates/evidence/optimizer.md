# Runtime workspace and Evidence contract

The trusted controller generated this section from the current session. It is authoritative for
filesystem roles, Evidence visibility, and measurement trust.

## Workspace

```text
workspace/
├── input/
│   ├── kernel/                 # read-only incumbent Kernel
│   └── evidence/               # read-only authorized history described below
├── agent/optimizer/            # read-only implementation/config; initial State copies omitted
├── work/kernel/                # writable candidate copied from the incumbent
├── prompts/                    # read-only versioned phase prompts and README.md index
├── insights/                   # read-only, evidence-derived decision guidance
├── skills/                     # read-only reusable procedures installed for Claude
├── tools/                      # writable reusable tool scripts and README.md index
├── sessions/                   # session capture owned by the launcher; do not modify
└── scratch/                    # writable temporary requests, plans, recovery files, and reports
```

Use the files already present as your starting point. `prompts/`, `insights/`, and `skills/` belong
to the versioned Agent Revision: read and use them, but do not modify them during an Optimizer or
Bootstrap session. Only `tools/` is adaptive here. Save genuinely reusable scripts there and keep
`tools/README.md` current. Record hypotheses, evidence, and conclusions through the Direction and
Experiment Journal instead of creating free-form Insights. Evolver curates Journal and Session
evidence between Agent revisions and owns changes to Prompts, Insights, and Skills.
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

## Trust and measurement reuse

Treat normalized Gateway operation status, correctness, latency, per-Shape latency, profiler
counters, and returned code evidence as trusted facts. Treat every Agent-authored report, analysis,
diagnosis, finding, lesson, rationale, and recommendation as an interpretation that may be wrong.
Re-derive conclusions from trusted measurements and exact source.

Do not repeat a completed Evaluate or Profile for the same Kernel Artifact and identical
operation-defining parameters. Recover and re-analyze the existing result instead. A failed,
cancelled, incomplete, differently parameterized, or different-Kernel operation is distinct.
To select an unchanged Kernel from visible history, record an Experiment with `action="adopt"`
and the real `before` and `after` Kernel Trial IDs. Runtime validates the source Trial's successful
ordinary full Evaluate against the same operator, hardware, DSL and sealed contract. The decision
is new; the measurement and its Trial remain historical. Keep the exact adopted Kernel bytes in
the candidate workspace. This recorded adoption can satisfy the `candidate_ready` precheck without
another Evaluate. Other actions still require an `after` Trial from this logical Attempt.
If adoption is rejected as incompatible, follow the returned error and run a qualifying full
Evaluate; do not change a comment just to create a different Artifact identity.
Agent-requested ABBA is exploratory and does not replace the full-Evaluate precheck. Runtime's
authoritative retention comparison runs only after terminal handoff and creates no Agent Trial;
never wait for it before recording an Experiment or submitting the Report.
If no Experiment was possible, `blocked` or `pivot` may contain empty experiments and findings;
give the genuine reason in the report and block or defer any in-progress Direction first. Do not
invent an Experiment to satisfy a count. `candidate_ready` still requires journaled evidence.
Private evaluator inputs remain hidden; opaque Shape identifiers and measurements must not be used
to reconstruct them.
