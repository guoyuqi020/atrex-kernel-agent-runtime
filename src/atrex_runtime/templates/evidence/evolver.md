# Evidence input

Runtime injects this frozen view. Missing participants, Sessions, history, or State are unavailable;
do not infer them.

```text
input/
├── agents/
│   ├── active/{source,runtime-state/trajectories/<ordinal>/{skills,tools}}/
│   └── challenger-NNNN/{source,runtime-state/trajectories/<ordinal>/{skills,tools}}/
├── evidence/
│   ├── active/{optimization-summary.json,sessions/...}
│   └── challenger-NNNN/{optimization-summary.json,sessions/...}
├── evolution-reports/evo-N.json
└── historical/agent-vN/
    ├── source/
    ├── optimization-summary.json
    └── runtime-state/trajectories/<ordinal>/{skills,tools}/
```

`input/agents/` is the current competition pool. Its matching `input/evidence/` entry contains the
participant's latest completed Epoch: one Runtime-derived summary and, for each completed Attempt,
`sessions/trajectory-NNNNNNNN/attempt-NNNNNNNN.conversation.jsonl`. A new Challenger may have
`latest_epoch: null` and no Sessions.

`input/historical/` contains completed non-current versions, their accumulated per-Trajectory State,
and career summary, but no conversations. Bootstrap, older-Epoch conversations, and detailed Runtime
history are not exposed.

## Prior Evolutions

`input/evolution-reports/evo-N.json` orders available Agent-creation reports. Bootstrap is
`agent-v0`; the first evolved revision is normally `evo-1`.

```json
{
  "evolution_number": 1,
  "parent": {
    "source_path": "input/historical/agent-v0/source",
    "runtime_state_path": "input/historical/agent-v0/runtime-state"
  },
  "generated_agent": {
    "source_path": "input/historical/agent-v1/source",
    "runtime_state_path": "input/historical/agent-v1/runtime-state"
  },
  "report": {
    "proposal_type": "evolved",
    "hypothesis": "The Agent-level causal hypothesis.",
    "expected_effect": "The expected next-Epoch behavior.",
    "unimplemented_capabilities": []
  }
}
```

`parent` is the selected Source Base, including for `evolve_from_history`; `generated_agent` is the
produced revision. Each path points to that Agent's single visible location under `input/agents/` or
`input/historical/`. Compare Source trees for the actual change. Treat report claims as intent to test
against Source, conversations, and optimization summaries. Bootstrap and unavailable legacy reports
have no file.

## Source and Runtime State

Source under `input/agents/` and `input/historical/` is a sealed read-only snapshot. The complete
`candidate/source/` copy is writable and is the versioned Agent code you may freely modify. Runtime
State is non-versioned adaptive `skills/` and `tools/` accumulated independently by Trajectory; it
may be narrow, stale, or wrong.

The Candidate starts with complete Active Source and one flat `candidate/runtime-state/` seed. Runtime
selects the latest completed Epoch's winning Agent and best-Kernel Trajectory terminal State, falling
back to its Epoch-start State, revision seed, then empty State. Runtime never merges Trajectories. The
sealed Candidate State initializes every Trajectory of the new revision and the next Active.

## Optimization summary

```json
{
  "kernel_agent_revision_id": "agentrev_0123456789abcdef0123456789abcdef",
  "latest_epoch": {
    "epoch_number": 3,
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

`latest_epoch` is `null` before the revision completes an Epoch. Otherwise its three mutually
exclusive Attempt outcome counts sum to `attempt_count`. `best_kernel` is `null` without a correct
Candidate; otherwise it contains the fastest correct Attempt's authoritative Gateway result. Shape
keys are opaque. Per-Shape latency and its geometric and arithmetic means are microseconds. Career
wins plus losses equal completed-Epoch participation.

## Conversation JSONL

Each line is one increasing-sequence record. Relevant types are `session_start`, Runtime `message`,
Backend `provider_event`/`provider_text`/`provider_binary`, and terminal `session_end`. The stream is
unredacted except that high-frequency Claude thinking-token estimates may be omitted. Use it to
understand behavior and tool use, not as performance Evidence.

## Session context

The final JSON supplies:

- immutable `dsl` and current `evolution_number`;
- `visible_agent_repositories[]` identity, relationship, Source ancestry, and paths to Source,
  summary, available Sessions, and Runtime State;
- `evidence` and `evolution_reports` roots;
- writable `candidate.source` and `candidate.runtime_state`;
- `evolution_report.draft`, exact publication `tool`, and final `published` path.
