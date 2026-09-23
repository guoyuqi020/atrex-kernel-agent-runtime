# Evidence input

Kernel Artifacts may contain a multi-file source tree, not just `kernel.py`. The next Optimizer
receives that tree in `work/kernel/` and must inspect the relevant source files. Runtime injects
its fixed adapter and editable-root rules; evolving the Agent does not change them. Historical
Kernel reuse restores the entire Artifact, including its locked support files.

## Runtime services for the next Optimizer

This catalog describes services available in Optimizer and Bootstrap sessions, subject to their
Capabilities and phase rules. It does not authorize this Evolver to call them. Here, read frozen
Evidence files and use only the Session-context `evolution_report.tool` for submission.

| Command | Service |
| --- | --- |
| `gateway-execute` | Executes one supported GPU operation; Runtime owns Job tracking, infrastructure retries, request deduplication, result projection, and measurement persistence. |
| `kernel-artifact-read` / `result-artifact-read` | Copies a selected Kernel Artifact source file into `scratch/`, or reads a normalized Agent-visible Result Artifact by digest. |
| `list-directions` / `load-direction` | Writes the visible Direction index to a requested `scratch/` file, or loads one complete Direction and its evidence links. |
| `list-experiments` / `load-experiment` | Writes the visible Experiment index to a requested `scratch/` file, or loads one complete recorded Experiment. |
| `update-direction` / `record-experiment` | Immediately persists Direction lifecycle changes or evidence-linked Experiments; enforces visibility, state, and provenance rules. |
| `attempt-report` | Validates and publishes the terminal handoff. Invalid drafts can be repaired and resubmitted; Runtime alone decides Kernel retention and Agent promotion. |

Supported `gateway-execute` operations:

- `evaluate`: correctness and per-Shape timing; optional `comparison.method="abba"` compares two
  Kernel sources. ABBA is an Evaluate option, not a separate operation.
- `profile`: profiler counters, per-Kernel durations, resource usage, and available SOL evidence;
  private contract cases are identified only by opaque numeric Shape IDs.
- `dev`: runs a bounded remote GPU command with selected workspace files for custom probes.
- `check`: compilation or sanitizer checks.
- `disassemble`: the Gateway's disassembly operation; use only evidence actually returned.
- `env`: remote GPU environment and capabilities.

For the standard Agent, the exact CLI form in a later session is
`python3 agent/optimizer/src/runtime_tools.py <command> --request scratch/<request>.json`.
Read the Candidate's `atrex-agent.json` `prompt_fragments.attempt_tools` file for request fields,
examples, and result semantics, and `src/runtime_tools.py` / `src/tool_contracts.py` for bindings
and local validation. These paths describe the standard implementation, not a requirement to keep
it. Runtime request errors return the operation's `issues`, `request_schema`, and `recovery`;
use the supplied contracts, not invented endpoints or guessed IDs.

Agent Gateway operations use only the Campaign's fixed Valid subset; authoritative Runtime ABBA
uses Valid + Test, each containing at most 15 Shapes. Test inputs and per-Shape results are never
exposed. Valid Shapes use stable contiguous opaque IDs `0..V-1`, not source-dataset IDs. Latencies
in this Evidence view cover Valid only; Kernel acceptance and Branch selection
are Runtime verdicts, not Valid-only performance decisions. Do not attempt to reconstruct hidden
Test data.

Runtime also controls private Evaluation inputs, resource limits, Session capture, version sealing,
recovery, and Gate/comparison policy. Candidate code cannot grant access to hidden Shapes, the
Registry database, credentials, or management APIs. A convenience interface can be implemented as
an Agent-side composite Tool, not a new Runtime endpoint. For example, in a later Optimizer session,
`list-experiments` → `load-experiment` → `result-artifact-read` can feed a compact per-Shape history
comparison, retaining source Result identities and separating measured facts from new analysis.

Runtime injects this frozen view. Missing participants, Sessions, history, or State are unavailable;
do not infer them.

Some Evolutions also expose `input/observer/active/`, identified by
`observer_active_lineage` in Session context. It is a separate read-only Isolated Lineage, frozen
through the preceding Epoch. Its `agents/` and `evidence/` trees use the same Source, summary,
Session, report, Journal, review, and reusable-resource conventions described below. Use it as an
additional behavioral comparison when improving the Challenger Lineage. It is not a Branch of the
current Lineage, not a Candidate base, and not a source of shared Kernel or Journal state. Never
infer same-Epoch Active results, and never rewrite Challenger history from observer records.

```text
input/
├── agents/agent-vN/
│   ├── src/ and Agent configuration
│   └── {prompts,skills,tools}/
├── evidence/
│   ├── latest-epoch-facts.json
│   ├── journal/
│   │   ├── directions/{index.json,direction_<id>.json}
│   │   └── experiments/{index.json,experiment_<id>.json}
│   ├── review/
│   │   ├── evolution-change-audit.json
│   │   ├── trajectory-comparison.json
│   │   └── workflow-friction.json
│   └── agent-vN/
│       ├── resources/trajectories/trajectory-NNNNNNNN/{prompts,skills,tools}/
│       ├── optimization-summary.json
│       ├── sessions/trajectory-NNNNNNNN/attempt-NNNNNNNN.conversation.jsonl
│       └── reports/trajectory-NNNNNNNN/attempt-NNNNNNNN.report.json
└── evolution-reports/evo-N.json
```

Every visible Agent revision appears exactly once, keyed by its Lineage version `agent-vN`.
If the same Agent ran both Branches in the first Epoch, both histories appear under that version:
Trajectory slots for the replica follow Active's slots in State, Sessions, and Reports;
`latest_epoch.branch` is `active_and_replica`. It is a
parallel run of the same Agent, not an Evolution; no Evolution Report exists for it.
`input/agents/agent-vN/` is one complete read-only Agent Bundle;
`input/evidence/agent-vN/` is what Runtime derived about it. No directory name encodes an Epoch role.

Every version has an `optimization-summary.json`. Only the branches that competed in the most
recent completed Epoch also have `sessions/` and `reports/`, and all of those sets come from that same
Epoch, so their behavior is directly comparable. Every other version has Source, State, and a career
summary but no conversations and no Attempt reports. Bootstrap, older-Epoch conversations, and detailed
Runtime history are not exposed.

`input/evidence/latest-epoch-facts.json` is a compact cross-Branch index for the latest completed
Epoch. It records each Attempt's Runtime status, failure reason (with private evaluator details
withheld), report status, Candidate Artifact/Result identities and outcome, and whether it became
Branch best, plus Direction/Experiment IDs and the final selection reason. Outcome and failure fields
come from Runtime's frozen records; Direction/Experiment IDs index Agent-authored Journals and do
not certify their analyses. Read this file and the optimization summaries first. Use the IDs and
per-Agent reports to select which conversations need close inspection. A missing Candidate by itself
does not diagnose an Agent or Journal failure; check the Runtime failure reason and Session first.

`review/` contains conservative Runtime-derived indexes over the same frozen latest-Epoch Evidence:

- `evolution-change-audit.json` matches prior Evolution `changed_paths` to observed discovery,
  invocation, execution failure, and Attempt-report citation in the evaluated generated Agent.
- `trajectory-comparison.json` groups Attempts by Branch and Trajectory and reports exact stable-ID
  overlaps, outcomes, failures, and Valid-domain best latency without merging similar prose.
- `workflow-friction.json` indexes Runtime-tool failures, failed-then-successful repair loops, and
  identical normalized request/probe construction across multiple Sessions.

These are navigation aids, not semantic verdicts. `not_observed` does not prove a change was useless;
a cited or successfully invoked Tool does not prove causal benefit; a repeated construction may be an
intentional retry or revalidation. Start with the indexes, then inspect only the cited raw Sessions,
Reports, Directions, and Experiments needed to classify a material signal.

`journal/directions/index.json` and `journal/experiments/index.json` index Bootstrap and the completed Lineage's
append-only Journal. Read selected `<id>.json` files for full Direction events and Experiments,
including entries from Attempts without terminal Reports. Gateway measurements are facts;
Agent-authored analyses are interpretations. Historical `suggest` records remain readable;
they are untested recommendations, not facts. Evolver cannot create or change Directions.
Compare related hypotheses and Experiments across Branches and check interpretations against
trusted outcomes. Use that evidence only to diagnose task-independent Agent defects in process,
tooling, evidence handling, or orchestration. Do not copy task facts into the Candidate or prescribe,
rank, suppress, merge, reopen, or require a concrete Kernel optimization direction.

The Agent Bundle declares one executable Epoch Workflow through `atrex-bundle.json`; production and
control Lineages may begin from different selected programs. The selected program and its helpers
are part of the Agent Revision and may be evolved.
Runtime executes the Active Revision's Workflow once per Epoch and exposes only bounded services:
create an Active replica, invoke Evolver for a Challenger, run an exact Branch/Trajectory/Attempt
organization, select the trusted best Kernel, compare Agents, and commit the Epoch. This is enough
to implement organizations such as one-Agent pooling or Active-versus-Challenger evolution in code.
The fixed Optimizer Attempt budget must be spent exactly. Runtime still owns Worker execution,
Gateway evaluation, retries, comparison, promotion, recovery, and durable state. Use Conversation
and outcome evidence before changing Workflow code; a changed Workflow is evaluated only after its
Agent Revision becomes Active in a later Epoch.

Each Session-context entry's `relationship` names its Epoch role. The entries whose relationship is
`active` or `challenger` are exactly the last completed Epoch's comparison pool, so their `version`
tells you which `agent-vN` competed in it:

- `active` — ran as that Epoch's Active branch.
- `challenger` — ran as that Epoch's Challenger at its `challenger_ordinal`.
- `current_epoch_challenger` — created earlier in the current Epoch. It has not competed, so it has no
  conversations or Attempt Reports. Use it only to avoid duplicating an existing proposal; do not copy
  its content, treat it as outcome Evidence, or credit it as a contributor.
- `lineage_history` — a completed version outside that pool.

The Session context does not carry an Epoch number. Read `latest_epoch.epoch_number` from either pool
member's summary to learn which Epoch it was.

The Session context marks exactly one visible Agent with `parent: true`. That revision is your Source
Base and won the last completed Epoch. Read `latest_epoch.branch` to see which side each competitor ran
as, and `latest_epoch.outcome` to see which one won — do not infer either from a path.

`latest_epoch.selection_reason` records the rule applied in the final pairwise selection step that
left the Epoch winner in place. With multiple Challengers it is not the complete tournament history
and does not explain every losing Agent individually. Never infer more than the recorded value:

- `authoritative_comparison` — Runtime used the configured authoritative Kernel comparator in that
  step. Its accepted/rejected verdict decided the step; this does not imply the retained winner's raw
  latency point estimate was lower.
- `identical_kernel` — both sides reached the same best Kernel, so Runtime retained the incumbent in
  that step without another comparison.
- `latency` — the better best-Kernel latency won, and the difference exceeded measurement uncertainty.
- `secondary_criteria` — the latencies tied within measurement uncertainty, so the decision fell to
  reaching the best result earlier, then more strict improvements, then more valid candidates, then
  fewer failures.
- `incumbent_retained` — everything tied, so the incumbent Active kept its position.
- `null` — no comparison ran, because the Epoch had no Challenger, both Branches used the same Agent,
  or because it completed before
  Runtime recorded reasons.

A `secondary_criteria` or `incumbent_retained` result is evidence about consistency and convergence
speed, not raw speed. Treat it accordingly when attributing the outcome.

## Prior Evolutions

`input/evolution-reports/evo-N.json` orders available Agent-creation reports. Bootstrap is
`agent-v0`; the first evolved revision is normally `evo-1`.

```json
{
  "evolution_number": 1,
  "parent": {
    "path": "input/agents/agent-v0"
  },
  "generated_agent": {
    "path": "input/agents/agent-v1"
  },
  "report": {
    "proposal_type": "evolved",
    "hypothesis": "The Agent-level causal hypothesis.",
    "expected_effect": "The expected next-Epoch behavior.",
    "changed_paths": ["prompts/episode.md"],
    "contributing_paths": ["input/agents/agent-v2"],
    "unimplemented_capabilities": []
  }
}
```

`parent` is the selected Bundle Base, including for `evolve_from_history`; `generated_agent` is the
produced revision. Each path points to that Agent's single visible location under `input/agents/`.

`changed_paths` lists files changed by that Evolution relative to its Bundle root. Older reports
may cover implementation files only. To inspect content, compare the named Bundles, but remember that
the Parent's visible reusable resources can reflect later optimization sessions rather than the
original Evolution input.

`contributing_paths` records the original workspace-relative files or directories whose content
that Evolution incorporated, from Agent Bundles or Evidence resources, including Parent resources.
It excludes mere reading and automatic Parent inheritance, and is empty when nothing was incorporated.
These are paths in the producing Session; later resources may differ or disappear. Runtime retains
exact content snapshots in the private Evolution Trace; do not treat current files as the old snapshot.
Older ID-only reports identify contributing Bundles, not precise files or Trajectory resources.
This is provenance, not parentage: `parent` remains the single Bundle base for the diff.

Treat every other report field as intent to test against Source, conversations, Attempt reports, and
optimization summaries. Bootstrap and unavailable legacy reports have no file.

For a previous-change audit, match `generated_agent.path` to the last completed Epoch's participants
using the Session-context relationships, then inspect that version's `sessions/` and `reports/`.
An Agent promoted earlier may now be the Active. A newer `current_epoch_challenger` is unevaluated;
versions outside the last Epoch pool have no conversations here. Missing observations cannot show
whether a Tool was unused or ineffective. Per-Trajectory `resources/` and Conversations can help
identify Tool rewrites, but a terminal resource file alone does not establish the exact code executed
earlier in a session.

## Agent Bundles and reusable resources

Every `input/agents/agent-vN/` is a complete read-only Bundle. `candidate/` starts as a writable
copy of the `parent: true` Bundle. There is no separate Source/State pair to assemble or synchronize.

The Parent combines its implementation with the latest completed Epoch winner's best-Kernel
Trajectory terminal resources, falling back to its Epoch-start snapshot, revision seed, then packaged
defaults. This is also the next Active's starting snapshot. Other visible Bundles use their revision
seeds. Per-Trajectory learned resources remain available under each Evidence entry's `resources/`;
they are supplementary observations, not extra Candidate copies. Runtime never merges them automatically.

Each of the three reusable directories has a mandatory `README.md` index. Prompts contains phase
instructions; Skills contains reusable, task-independent procedures; and Tools contains executable
helpers. Optimizer and Bootstrap sessions can modify only Tools. This Evolver owns the versioned
curation of all three. Keep them concise and non-duplicative, and update the relevant index after
additions, changes, removals, or renames.

Treat Tool-to-Skill promotion as evidence-driven curation. Inspect the Tool source, its actual
invocations in conversations, the associated Attempt reports, and authoritative outcomes. Promote a
Tool only when those records show a repeatable procedure worth triggering in future Claude sessions;
do not turn every one-off probe, task-private script, or failed helper into a Skill. Package a promoted
procedure as `skills/<skill-name>/SKILL.md`. Its YAML frontmatter must have a non-empty `name` matching
the directory and a `description` with concrete trigger conditions. Keep the body concise and include
the procedure, prerequisites, validation criteria, dependencies, and limitations; keep supporting
scripts and references inside the same Skill package. Remove or reduce a redundant `tools/` copy when
the Skill becomes the canonical owner, and update both indexes. Before the next Claude Optimizer or
Bootstrap session, Runtime installs valid Skill packages into that session's private Claude Home; it
never modifies Evolver or host/global configuration. Other backends may read workspace resources but
native Skill discovery is not promised.

Task-specific hypotheses, Kernel directions, measurements, conclusions, and Artifact identities
belong only to Runtime Journals and Reports. Static, task-independent reference material belongs in
a Skill's references. You may combine generic procedures from eligible Agents and their
Trajectories, remove redundant content, and incorporate stable behavior into prompts or
implementation. Credit contributing revisions. Do not draw from an unevaluated
`current_epoch_challenger`.

Edit `candidate/prompts/` for later Optimizer sessions; preserve configured prompt paths.
Managed Optimizer sessions resolve those paths against their inherited writable `prompts/`.
Changes do not alter a prompt already loaded or override trusted injected context and enforcement.
Candidate resources seed its next optimization trajectories; they do not overwrite the next Active's
independent starting copy.

## Optimization summary

```json
{
  "kernel_agent_revision_id": "agentrev_0123456789abcdef0123456789abcdef",
  "version": "agent-v3",
  "path": "input/agents/agent-v3",
  "resources_path": "input/evidence/agent-v3/resources",
  "latest_epoch": {
    "epoch_number": 3,
    "branch": "challenger",
    "challenger_ordinal": 1,
    "outcome": "won",
    "selection_reason": "authoritative_comparison",
    "attempt_count": 2,
    "correct_attempt_count": 1,
    "incorrect_attempt_count": 1,
    "no_candidate_attempt_count": 0,
    "best_kernel": {
      "gateway_result": {
        "status": "completed",
        "correct": true,
        "correctness": {
          "status": "PASS",
          "rel_err": null,
          "max_abs_err": 0.0009765625,
          "max_rel_err": 0.0078125
        },
        "latency_us_by_shape": {"0": 8.0, "1": 10.0},
        "latency_us_geomean": 8.94427190999916,
        "latency_us_arith_mean": 9.0
      }
    }
  },
  "career": {
    "epoch_participation_count": 3,
    "win_count": 1,
    "loss_count": 2
  }
}
```

`path` identifies the complete Bundle; `resources_path` identifies supplementary per-Trajectory
resources. `latest_epoch` is `null` before the revision completes an Epoch.
Otherwise `branch` is `active` or `challenger`, `challenger_ordinal` is `null` for the Active branch,
and the three mutually exclusive Attempt outcome counts sum to `attempt_count`. `best_kernel` is `null`
without a correct Candidate; otherwise it contains the fastest correct Attempt's authoritative Gateway
result. Shape keys are opaque. Per-Shape latency and its geometric and arithmetic means are
microseconds. Career wins plus losses equal completed-Epoch participation.

## Attempt reports

`reports/trajectory-NNNNNNNN/attempt-NNNNNNNN.report.json` is the Optimizer's own account of one
Attempt: `hypothesis`, `diagnosis`, `approach`, `experiments`, `findings`, `knowledge_used`,
`evidence_summary`, `contributing_result_artifact_digests` naming the historical results it drew from,
`parent_kernel`, `candidate_kernel` including `comparison_with_parent`,
`production_gate`, and a closing `analysis`. It is an untrusted interpretation like a conversation:
use it to explain what the Agent believed and attempted, then verify against the measured Gateway
results in the optimization summary.
Reports also retain `direction_events`, including the selected `supporting_experiment_ids` and
Agent-declared `hypothesis_status` (`unresolved`, `supported`, `refuted`). These are not Runtime
certifications of scientific conclusions. Lifecycle closure, including abandonment, does not imply
falsification. Missing historical judgments mean unresolved. When producing memory, Skills, or
workflow changes, preserve uncertainty and the exact scope of selected Experiments: do not promote
untested interpretations from unrelated measurements into established constraints.

## Conversation JSONL

Each line is one increasing-sequence record. Relevant types are `session_start`, Runtime `message`,
Backend `provider_event`/`provider_text`/`provider_binary`, and terminal `session_end`. Content is
unredacted. Claude reading views prefer native content over duplicate stdout, omit internal
queue/title/file-history bookkeeping and thinking-token estimates, and preserve distinct content
blocks, uncovered stdout, errors, compaction boundaries, and terminal results. Use them to understand
behavior and tool use, not as performance Evidence.

## Session context

The final JSON supplies:

- immutable `dsl` and current `evolution_number`;
- `visible_agent_repositories[]` identity, version, `relationship`
  (`active` / `challenger` / `current_epoch_challenger` / `lineage_history`), `challenger_ordinal`,
  `parent` marker, ancestry, and paths to the Bundle, summary, available Sessions, Attempt
  reports, and supplementary resources;
- `evidence` and `evolution_reports` roots;
- writable `candidate`;
- `evolution_report.draft`, exact publication `tool`, and final `published` path.
