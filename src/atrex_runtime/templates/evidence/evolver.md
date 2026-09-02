# Evidence input

Runtime injects this frozen view. Missing participants, Sessions, history, or State are unavailable;
do not infer them.

```text
input/
├── agents/agent-vN/
│   ├── source/
│   └── runtime-state/trajectories/<ordinal>/{skills,tools}/
├── evidence/agent-vN/
│   ├── optimization-summary.json
│   ├── sessions/trajectory-NNNNNNNN/attempt-NNNNNNNN.conversation.jsonl
│   └── reports/trajectory-NNNNNNNN/attempt-NNNNNNNN.report.json
└── evolution-reports/evo-N.json
```

Every visible Agent revision appears exactly once, keyed by its Lineage version `agent-vN`.
`input/agents/agent-vN/` is its sealed Source and accumulated per-Trajectory State;
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
- `null` — no comparison ran, either because the Epoch had no Challenger or because it completed before
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
    "source_path": "input/agents/agent-v0/source",
    "runtime_state_path": "input/agents/agent-v0/runtime-state"
  },
  "generated_agent": {
    "source_path": "input/agents/agent-v1/source",
    "runtime_state_path": "input/agents/agent-v1/runtime-state"
  },
  "report": {
    "proposal_type": "evolved",
    "hypothesis": "The Agent-level causal hypothesis.",
    "expected_effect": "The expected next-Epoch behavior.",
    "changed_paths": ["prompts/episode.md"],
    "contributing_source_paths": ["input/agents/agent-v2/source"],
    "unimplemented_capabilities": []
  }
}
```

`parent` is the selected Source Base, including for `evolve_from_history`; `generated_agent` is the
produced revision. Each path points to that Agent's single visible location under `input/agents/`.

`changed_paths` is the exact set of versioned Source files that Evolution changed, relative to the
Source root. Runtime already checked it against the sealed Source, so it is recorded fact, not a claim
you need to re-derive. It covers versioned Source only: an empty array means that Evolution changed
Runtime State alone, and a `reuse` proposal is always empty. It tells you *which* files moved, not
*what* changed inside them—diff the two named Source trees for that.

`contributing_source_paths` names the other Agents whose Source, Skills, or Tools that Evolution drew
content from, and is empty when it drew from none. It is provenance, not parentage: `parent` remains the
single Source base the diff was measured against.

Treat every other report field as intent to test against Source, conversations, Attempt reports, and
optimization summaries. Bootstrap and unavailable legacy reports have no file.

## Source and Runtime State

Source under `input/agents/` is a sealed read-only snapshot. The complete
`candidate/source/` copy is writable and is the versioned Agent code you may freely modify.
Runtime State is non-versioned adaptive `skills/` and `tools/` accumulated independently by
Trajectory; it may be narrow, stale, or wrong.

The Candidate starts with the complete Source of the `parent: true` Agent and one flat
`candidate/runtime-state/` seed. Runtime selects the latest completed Epoch's winning Agent and
best-Kernel Trajectory terminal State, falling back to its Epoch-start State, revision seed, then
empty State. Runtime never merges Trajectories. The sealed Candidate State initializes
every Trajectory of the new revision and the next Active.

Every visible version's Source and Runtime State is readable, not only the one you start from. You may
study all of them, but may merge content only from the Active and completed Lineage history—not an
unevaluated `current_epoch_challenger`—into the single writable Candidate. The revision you declare as
the Source base fixes only what your Source diff is measured against; other eligible revisions you
drew content from are recorded separately as provenance.

## Optimization summary

```json
{
  "kernel_agent_revision_id": "agentrev_0123456789abcdef0123456789abcdef",
  "version": "agent-v3",
  "source_path": "input/agents/agent-v3/source",
  "runtime_state_path": "input/agents/agent-v3/runtime-state",
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

`source_path` and `runtime_state_path` are this revision's single location under `input/agents/`; use
them to move from Evidence to Source. `latest_epoch` is `null` before the revision completes an Epoch.
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
Backend `provider_event`/`provider_text`/`provider_binary`, and terminal `session_end`. The stream is
unredacted except that high-frequency Claude thinking-token estimates may be omitted. Use it to
understand behavior and tool use, not as performance Evidence.

## Session context

The final JSON supplies:

- immutable `dsl` and current `evolution_number`;
- `visible_agent_repositories[]` identity, version, `relationship`
  (`active` / `challenger` / `current_epoch_challenger` / `lineage_history`), `challenger_ordinal`,
  `parent` marker, Source ancestry, and paths to Source, summary, available Sessions, available Attempt
  reports, and Runtime State;
- `evidence` and `evolution_reports` roots;
- writable `candidate.source` and `candidate.runtime_state`;
- `evolution_report.draft`, exact publication `tool`, and final `published` path.
