# Runtime workspace and Evidence contract

The trusted controller generated this section from the current session. It is authoritative for
filesystem roles, Evidence visibility, and measurement trust.

Gateway measurements use the Campaign's complete Valid population with no Shape-count cap. New
Campaigns have an empty Test population; authoritative ABBA retention and promotion measure the
same complete Valid set.
Valid Shapes use stable contiguous opaque IDs `0..V-1`; they are not source-dataset IDs, and gaps
or values cannot be used to infer Test membership.
Historical per-Shape results and latency aggregates shown here cover the complete Valid set. A
Runtime acceptance verdict remains distinct from an Agent experiment because Runtime performs its
own authoritative measurement.

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
serial search chain. Kernel sharing across Trajectories depends on the configured workflow;
always treat `input/kernel/` as this Attempt's authoritative incumbent.

After an Epoch completes, the controller independently selects the next active Agent Revision and
the fastest retained correct Kernel. They may have different producers. `input/kernel/` is always
the authoritative current starting point.

Each completed Epoch exposes every branch that ran in it, whether or not that branch was selected,
under `branches/<label>/`. That Epoch's `summary.json` lists the branches and marks the selected one.
A non-selected branch records real attempts against the same starting Kernel and can contain useful
positive or negative evidence. Inspect it when relevant to the current question, not as a mandatory
reading assignment. For the current Epoch you see only bounded
earlier Attempts from your own Trajectory, never a concurrently running sibling. Exact historical
Kernel, Trial, Result, Direction, and Experiment records remain in controller storage and are
retrieved through the supplied Runtime-local query commands. Every Direction update and Experiment
record is durably appended by Runtime before its tool call returns; a Worker crash or recovery
generation does not roll the logical Attempt Journal back. Journal queries may include every
completed Active and Challenger path from a frozen Epoch, without exposing branch-control
provenance. No Journal history file exists under `input/evidence/` or the internal control area.
The filesystem view above is distinct from live query visibility: a Broadcast workflow can expose
other Trajectories' recorded Journal and Artifact evidence through tools without sharing their
working files or live conversations. With Epoch-boundary sharing, sibling history becomes visible
only after the Epoch completes. Do not infer isolation or additional access from directory layout.

Each completed Epoch's `summary.json` is a branch index, not an aggregate of optimization lessons.
This filesystem tree does not contain an additional generated lesson summary or measurement table;
that does not mean structured history is unavailable through enabled Journal tools. Numbered
directories encode chronology, not a requirement to read every earlier Attempt. Each historical
Attempt exposes only its final report and latest sealed backend-neutral
`conversation.jsonl`; all physical retries remain in controller storage. Conversation files may
contain sensitive model/tool content without redaction. Claude reading views prefer native content
over duplicate stdout and omit internal queue/title/file-history bookkeeping and thinking-token
estimates. Distinct content blocks, uncovered stdout, errors, compaction boundaries, and terminal
results remain visible.

## Recover relevant history

Start from `input/kernel/` and a concrete question, such as whether a proposed change was already
tested or which measurement supports a prior conclusion. Use enabled Journal indexes and selected
records before opening historical reports or conversations. If an ID is already known, load that
record directly. Follow the retrieval order below; recovering state does not require replaying the
Lineage. These queries expose only the current Session's authorized history.

### Direction history

Call `list-directions` with `{"file":"scratch/directions-index.json"}`. It writes IDs, names,
lifecycle status, hypothesis status, and ancestry links to that file; the response contains only
status, file, and count. Read the index to select relevant entries. Call `load-direction` with
`{"direction_id":"direction_<id>"}` to recover a selected hypothesis, rationale, plan, success and
stop criteria, and latest analysis. The index alone does not explain what was tried or why a
hypothesis was judged supported or refuted.
Both queries show `in_progress(self)` for a Direction started by this Attempt and
`in_progress(other)` for one owned by another Attempt, regardless of who proposed it.
Close only your own open Direction before switching or handing off; shared visibility does not
transfer ownership. Other lifecycle statuses are unchanged.
If you already have a Kernel Artifact digest, `find-kernel-directions` accepts
`{"kernel_artifact_digest":"sha256:<digest>"}` and returns `direction_ids` linked through visible
Experiments. With no recorded association it can return an empty list, including when Experiment
recording is disabled; use the Direction index for hypotheses that have no such link.

### Experiment history

Call `list-experiments` with `{"file":"scratch/experiments-index.json"}`. It writes IDs, names,
hypotheses, changes, evidence, analyses, actions, and `knowledge_used` to that file; the response contains only status,
file, and count. This index already contains useful recorded conclusions. Call `load-experiment`
with `{"experiment_id":"experiment_<id>"}` when you need the complete selected record, especially
its exact `before`/`after` Kernel and Result Artifact digests. If the index answers the question,
do not load a record merely to repeat the same prose. Recorded interpretations remain fallible.
For a known Kernel Artifact digest, `find-kernel-experiments` accepts
`{"kernel_artifact_digest":"sha256:<digest>"}` and returns `experiment_ids` citing it as before or
after. An empty list means no visible recorded association, not that the Kernel was never evaluated.

When both Journal modules are enabled, `load-direction` also provides `associated_experiment_ids`
for all linked visible Experiments and `supporting_experiment_ids` for the latest explicit closure
selection. Load the relevant Experiments to follow that evidence; do not infer support from lifecycle
status alone. An Experiment's Direction association is not required when that module is disabled.

When recording an Experiment that applies Wiki knowledge, include optional `knowledge_used` entries
with the exact `record_id`, a concise `finding`, and its concrete `application` in this Experiment.
Use `[]` when none was used; old records without this field also read as `[]`. Historical knowledge
can be cited even when live Wiki access is disabled. This is your attribution, not measurement
evidence or proof that the knowledge helped; retain the required before/after Result references.

### Measurements, source, and gaps

Use `result-artifact-read` with a real `result_artifact_digest` to inspect a specific Evaluate,
Profile, or other recorded observation. Use `kernel-artifact-read` with a real
`kernel_artifact_digest`, `artifact_file`, and a destination `file` under `scratch/` to inspect the
exact source. Use `kernel-pareto-frontier` when the question concerns visible per-Shape latency
winners; these come from correct full contract Evaluations. Do not recover measurements or source
by scraping a conversation when the corresponding Artifact is available.

If Journal records are absent or leave a concrete gap, inspect the relevant historical
`report.json`. Only then search a selected `conversation.jsonl` for missing details such as an exact
Dev command, a failed probe, or an unrecorded implementation rationale. Load tools return structured
records and evidence links, not every command or the full Session transcript. An empty Journal
index does not prove that no prior work occurred.

Do not read or flatten all old conversations by default. Identify the missing fact and limit the
search to relevant Attempts and excerpts. Apply the same scope to delegated work: give a specific
question and selected files, not an instruction to reconstruct the entire history. Reuse evidence
already recovered in this Session and stop expanding history once there is enough support for the
next optimization or a concrete blocker.

## Direction ancestry

Resume the same unfinished hypothesis with `update-direction` and its existing Direction ID.
An unclaimed proposed or inherited Direction may be started with its existing ID. If Runtime
reports `direction_trajectory_conflict`, propose a new Direction with
`relationship="reimplementation"` and the inherited Direction ID as its parent.
When you revisit, reinterpret, port, or combine earlier work as a new hypothesis, use `action="propose"`
with optional `relationship`: `retry`, `refinement`, `reimplementation`, `correction`, `port`,
`combination`, or `adoption`. Cite visible `derived_from_direction_ids` and/or `derived_from_experiment_ids` and
explain the connection in the proposal's `rationale`. Use list/load tools to obtain real IDs first.
Each ancestry ID array allows at most 32 unique IDs; this is not a limit on Journal query results.
A combination needs two distinct parent Directions, either
directly or through their Experiments. A correction may also specify `supersedes_direction_id` naming
one of those parents; this records a revised interpretation without changing the parent's status.
Ancestry is fixed when the proposal is recorded. To correct it, propose a new derived Direction;
do not rewrite history. These links describe your interpretation, not proof of a performance gain.

Historical suggested Directions remain readable; they are untested recommendations, not facts or required
next steps. No session can create new suggestions. Choose your own hypothesis from
the public contract, profiling, and Journal evidence, then record it with `action="propose"`.

## Trust and measurement reuse

Treat normalized Gateway operation status, correctness, latency, per-Shape latency, profiler
counters, and returned code evidence as trusted facts. Treat every Agent-authored report, analysis,
diagnosis, finding, lesson, rationale, and recommendation as an interpretation that may be wrong.
Re-derive conclusions from trusted measurements and exact source.

Evaluate returns correctness and latency without automatically running Profile. When you need
SOL or hardware counters to test a bottleneck hypothesis, call `profile` explicitly with the
appropriate level (`sol`, `survey`, or `deep`); a completed Evaluate alone is not profiling evidence.

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
`complete`, `abandon`, `block`, and `defer` declare `hypothesis_status`: unresolved, supported,
or refuted, separately from lifecycle. Untested or inconclusive work can close unresolved with
empty support, even before any experiment or Gateway call. Select relevant `supporting_experiment_ids`
only when available; direct `supporting_results` bind exact Kernel and Result digests regardless of
Journal modules. Judged conclusions need a tested scope and completed evidence suitable for their
claim kind; insufficient support remains unresolved with assessment notes. Invalid references are
errors. Abandoning work does not refute a hypothesis, and no diagnostic call is required merely to
close it. `associated_experiment_ids` lists linked Experiments; selected support is a separate list.

Record one reusable claim per Finding. Set `claim_kind` to `observation` (what was observed),
`implementation_outcome` (the measured outcome of a particular implementation), or
`causal_hypothesis` (an explanation of why). Supply the exact sentence in `claim`, its tested
Kernel/hardware/Shape scope in `scope`, and `assessment=unresolved|supported|refuted`.
Use `root_cause: null` when the cause is unknown; neither a report nor a completed Attempt requires
a causal explanation. Missing historical assessment means unresolved.

Bind evidence to that claim using `supporting_results`, available with every Journal module
selection: `[{"kernel_artifact_digest":"sha256:<kernel>","result_artifact_digests":["sha256:<result>"]}]`.
When Experiments are enabled, optional `supporting_experiment_ids` may reference this Attempt's
Journal as an additional organization of evidence; direct Result bindings do not require an
Experiment. Reuse matching visible historical Results; do not repeat GPU work just to record a claim.
Runtime validates exact visible bindings and completed operation eligibility: observations may use
Check, Dev, Profile, or Evaluate; implementation outcomes require Dev or full Evaluate; causal
hypotheses require Dev, Profile, or full Evaluate. Check-only or correctness-only evidence cannot
support a performance or causal judgment. These checks do not establish causal relevance or truth.
Missing claim, scope, or suitable completed evidence downgrades a requested supported/refuted
Finding to unresolved with `assessment_notes`; invalid or invisible references are still errors.

Keep an untested explanation unresolved even beside a successful measured optimization. General
`analysis`, diagnosis, root-cause text, and lessons remain Agent interpretations, not verified
conclusions. Before rejecting a potentially useful mechanism, identify the smallest targeted probe
that could distinguish it from alternatives; if it is not worth testing now, defer it unresolved.
For example, observing `autovec_copy` in source does not establish that output cost is negligible.
A failed implementation does not refute every implementation of its proposed mechanism.

`blocked` or `pivot` may contain empty experiments and findings; give the genuine stopping reason.
`candidate_ready` still requires the enabled modules' nomination evidence and at least one Finding.
Private evaluator inputs remain hidden; opaque Shape identifiers and measurements must not be used
to reconstruct them.
