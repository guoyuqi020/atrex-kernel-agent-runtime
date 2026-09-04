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
├── prompts/                    # writable phase prompts and README.md index
├── memory/                     # reusable search memories and lessons; README.md index
├── knowledge/                  # reusable knowledge and reference notes; README.md index
├── skills/                     # reusable procedures; README.md index
├── tools/                      # reusable tool scripts; README.md index
├── hooks/                      # reusable Claude/Codex hooks and config snippets; README.md index
├── sessions/                   # session capture owned by the launcher; do not modify
└── scratch/                    # writable temporary requests, plans, recovery files, and reports
```

Use the files already present as your starting point. Save reusable content in the writable
`prompts/`, `memory/`, `knowledge/`, `skills/`, `tools/`, and `hooks/` directories, and keep their
README indexes current. The controller manages persistence and reuse between sessions.
Files in `scratch/` are temporary and are not carried into later sessions or retries.

Read the indexes before adding content. Whenever you add, change, rename, or remove a file, update
that directory's `README.md` so it remains a current index of paths, purposes, and applicability.
Use `prompts/` for phase prompts and reusable tool instructions. Managed Agent configuration uses
`prompt_root: "workspace"`; its `prompts/...` paths resolve here, not inside `agent/optimizer/`.
Keep referenced files available. Edits apply to later fresh Sessions, not the already submitted
current Prompt. Trusted injected context, tool validation, and evaluation rules cannot be changed
by editing these files. The six initial State directories are omitted from the Source workspace
copy; the sealed Source Artifact remains complete.
Use `memory/` for reusable search experiences and decisions, `knowledge/` for sourced knowledge, `skills/` for
repeatable procedures, and `tools/` for scripts. Tool entries also need invocation, inputs, outputs,
side effects, dependencies, an example, and limitations in `tools/README.md`.
Use `hooks/` for reusable Claude/Codex hook scripts and configuration snippets. Its README must identify
the backend, event, command, dependencies, side effects, activation steps, and verification status.
Before each Claude/Codex Optimizer or Bootstrap session, Runtime installs `skills/<name>/SKILL.md`
directories and the selected `hooks/claude.json` or `hooks/codex.json` into a private CLI Home.
Use native `{"hooks": {event: [matcher groups]}}` command-hook definitions; reference scripts with
`python3 "$WORKSPACE_ROOT/hooks/example.py"`. Installation does not execute hooks. Edit the reusable
originals for later sessions, not generated CLI settings; host/global settings must not be changed.
Other backends preserve these resources without auto-installation. See both README indexes for details.
Keep content concise and non-duplicative; link Journal and measurement identities instead of copying
raw traces or one-off outputs. Memory and
Knowledge are Agent-authored interpretations, not authoritative Journal/measurement replacements.
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
Private evaluator inputs remain hidden; opaque Shape identifiers and measurements must not be used
to reconstruct them.
