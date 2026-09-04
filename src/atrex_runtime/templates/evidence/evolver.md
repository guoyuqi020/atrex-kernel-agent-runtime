# Evidence input

Runtime injects this frozen view. Missing participants, Sessions, history, or State are unavailable;
do not infer them.

```text
input/
├── agents/agent-vN/
│   ├── src/ and Agent configuration
│   └── {prompts,memory,knowledge,skills,tools,hooks}/
├── evidence/agent-vN/
│   ├── resources/trajectories/trajectory-NNNNNNNN/{prompts,memory,knowledge,skills,tools,hooks}/
│   ├── optimization-summary.json
│   ├── sessions/trajectory-NNNNNNNN/attempt-NNNNNNNN.conversation.jsonl
│   └── reports/trajectory-NNNNNNNN/attempt-NNNNNNNN.report.json
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

## Agent Bundles and reusable resources

Every `input/agents/agent-vN/` is a complete read-only Bundle. `candidate/` starts as a writable
copy of the `parent: true` Bundle. There is no separate Source/State pair to assemble or synchronize.

The Parent combines its implementation with the latest completed Epoch winner's best-Kernel
Trajectory terminal resources, falling back to its Epoch-start snapshot, revision seed, then packaged
defaults. This is also the next Active's starting snapshot. Other visible Bundles use their revision
seeds. Per-Trajectory learned resources remain available under each Evidence entry's `resources/`;
they are supplementary observations, not extra Candidate copies. Runtime never merges them automatically.

Each of the six reusable directories has a mandatory `README.md` index:
Prompts contains phase instructions; Memory contains search experiences and decisions;
Knowledge contains reusable knowledge; Skills contains procedures; Tools contains scripts;
Hooks contains backend hook scripts and configuration snippets. Keep content concise, reusable, and
non-duplicative. Update the relevant index after additions, changes, removals, or renames.
For Hooks document backend, event, command, dependencies, side effects, activation, and verification;
storing a hook does not activate it.

These are Agent-authored materials, not authoritative results. You may combine supported content
from eligible Agents and their Trajectories, remove redundant content, and incorporate stable behavior
into prompts or implementation. Credit contributing revisions. Do not draw from an unevaluated
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
`evidence_summary`, `contributing_kernel_trial_ids` naming the historical Kernel Trials it drew from,
`parent_kernel`, `candidate_kernel` including `comparison_with_parent`,
`production_gate`, and a closing `analysis`. It is an untrusted interpretation like a conversation:
use it to explain what the Agent believed and attempted, then verify against the measured Gateway
results in the optimization summary.

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
