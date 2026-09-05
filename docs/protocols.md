# Durable protocols

English | [中文](protocols.zh.md)

This document defines persisted identities and visibility rules. Callable commands and routes are
in [Interface Reference](interfaces.md); deployment fields are in
[Configuration](configuration.md); measurement policy is in
[Evaluation and Promotion](evaluation.md). Pydantic models and database migrations remain the
executable schema authority.

## Identity and versioning

Typed IDs are opaque and prefix-qualified: `campaign_`, `lineage_`, `epoch_`, `attempt_`,
`workersession_`, `kernelrev_`, `agentrev_`, and `sha256:`. Callers must not parse semantics
from their suffixes.

Content and presentation identities are separate:

- Artifact digest identifies exact bytes and supports verification/deduplication.
- Kernel Revision ID plus Lineage-local `vN` identifies one retained historical reference.
- Agent Revision ID plus Lineage-local `agent-vN` identifies one Agent Bundle reference.
- Kernel Trial ID identifies an unversioned measured Candidate inside Attempt-visible history.

Rejected, reverted, blocked, pivoted, or infrastructure-failed Attempts remain durable but consume
no `vN` unless a Kernel Revision is actually published.

## Campaign and Lineage

A Campaign freezes operator, resolved hardware architecture, Agate GPU selector, Evaluation
Contract, public Agent Problem, Core source provenance, Evolver commit, and policy. A creation key is
idempotent only while all immutable inputs match.

Each Lineage owns exactly one DSL, independent Agent/Kernel version trees, models, Epoch topology,
active Agent, best Kernel, common Evidence checkpoint, and Runtime State lineage. Adding a seeded
Lineage creates fresh `agent-v0`/`v0` roots; source Revision IDs are provenance, not shared
version ancestry.

An Ablation Arm is a separate Campaign and Lineage rooted from another Lineage's sealed Bootstrap
baseline. It has no Challengers. Its optional ephemeral-State behavior is part of Lineage identity.

## Epoch, Branch, Trajectory, and Attempt

One Epoch records the pre-competition Active Agent, ordered Challenger participants, frozen
starting Kernel/Evidence/State, Branch and Trajectory outcomes, selected best Kernel, Agent winner,
and publication checkpoint.

Branch names describe competition roles, not ancestry. Every Branch starts from the same frozen
Epoch inputs. A Trajectory is an independent serial chain of Attempts within one Branch. An Attempt
is the logical unit of one Optimizer step; automatic infrastructure retries do not create another
Attempt.

Runtime fencing associates a renewable generation with every scheduler owner. A stale owner may
finish local work but cannot commit a transition.

## Bootstrap Generations and Worker Sessions

One stable Bootstrap Attempt may contain multiple append-only physical Generations. Each Generation
gets a new Capability generation, Workspace, Worker Session, usage report, trace, report, operations,
failure, and outcome. Earlier Generations are never rewritten.

Every model-backed process is a Worker Session with subject, role, Backend/model, start/end state,
workspace, terminal diagnostic, provider usage, and optional immutable Session Artifact. Session
identity is physical execution identity; Attempt, Bootstrap Attempt, Generalization subject, or
Evolution subject is logical ownership.

Authoritative Session Artifacts retain backend-neutral `conversation.jsonl` plus normalized event
ledgers. Claude high-frequency `system/thinking_tokens` estimate telemetry is omitted; terminal
provider usage remains mandatory.

## Evolver and Agent State

The commit-pinned Evolver Bundle declares its entrypoint in `atrex-evolver-bundle.json` schema 1.
Runtime sends exactly `Run the versioned Evolver Bundle once.` on stdin. Evolution input schema 11
contains Parent identity, DSL, Evidence checkpoint, workspace paths, and the frozen Agent catalog;
it grants no Gateway/Wiki or Runtime query capability. Trace schema 9 retains process, usage,
report, Candidate identity, and contribution snapshots. Provider usage is required; there is no
Evolver token cutoff.

Every `input/agents/agent-vN/` is a complete read-only Bundle; `candidate/` uses the same layout.
Evidence summaries and supplementary per-Trajectory resources are under `input/evidence/agent-vN/`.
Only participants in the last completed Epoch expose its Conversations and Attempt Reports.
Historical creation reports remain under `input/evolution-reports/`.

An Evolution report declares proposal mode, selected `kernel_agent_revision_id`, hypothesis,
expected effect, exact Bundle-relative `changed_paths`, `contributing_paths`, and unimplemented
capabilities. Historical derivation copies the selected complete Bundle before editing it.
The local `evolution-report` tool validates drafts and returns `issues`, `request_schema`, and
`recovery` on error without publishing. The first valid call atomically publishes
`scratch/evolution-report.json`; Runtime independently revalidates it after the Session exits.

Every new revision seals the complete Bundle and six-directory State checkpoint, recording
`optimizer_digest` and `runtime_state_digest`. Each new Trajectory starts from a separate copy;
Optimizer implementation permissions and State inheritance remain unchanged.

`contributing_paths` records sorted, unique workspace-relative files or directories actually incorporated from
`input/agents/agent-vN/` or `input/evidence/agent-vN/resources/`, including Parent resources from other
Trajectories. Mere reading and automatic Parent inheritance are not contributions. Paths must exist,
contain no links/traversal, and belong to eligible evaluated history or Parent, never a same-Epoch
unevaluated Challenger. `reuse` requires `[]`. Runtime records ownership and exact content snapshots
in the Evolution Trace; the field does not change the Bundle base or revision ancestry.

## Artifacts and measurements

The local CAS stores canonical manifests and payloads for Agent source, Kernel source, Runtime
State, Evaluation Contract, Agent Problem, Gateway result, reports, traces, and Evidence. Artifacts
are immutable. Registry rows and Gateway-control records are the authority that gives an Artifact
domain meaning.

Every candidate-bearing Gateway operation seals Kernel source before external execution. An
`evaluate` or `profile` creates immutable operation/result and normalized measurement records.
These facts remain valid if the Agent later restores another Kernel.

Attempt Evidence schema v2 is immutable and includes only earlier Attempts from the same Branch slot
and Trajectory. Completed Epoch Evidence preserves Challenger and Trajectory identities, summaries,
reports, diffs, normalized Session projections, normalized Evaluate/Profile measurements, lessons,
and source digests. Agent view schema v1 is role-scoped: an Optimizer sees every completed branch's
Attempt reports and conversations under a per-branch layer, with each Epoch summary naming the
selected branch, plus bounded current same-Trajectory Attempts with branch-control identities
removed.
Runtime derives the compact Evolver filesystem view from complete durable Evidence: each current
participant receives an authoritative latest-Epoch optimization summary and one Conversation per
Attempt; completed non-current Agent versions receive a summary beside their source. Older detailed
branch trees and exact Kernel Artifacts remain in Runtime's Registry and Artifact stores rather than
being duplicated into the Evolution workspace. The Evolver requires no live Gateway authority.
Authoritative Session retention omits Claude `system/thinking_tokens` estimate telemetry, and the
derived copies defensively apply the same rule to older Session Artifacts.

Kernel Trials group one exact Kernel Artifact with visible Gateway observations. A Trial can be
recovered through a recorded Experiment, read by Artifact digest, and inspected without contacting
Agate. Normalized measurements are internal durable facts, not a separate unrestricted Agent query
surface.

The Kernel Revision's primary evaluation is the Runtime-selected Bootstrap or retention result.
Comparator repeats remain separate durable measurements. Exact raw jobs are administration-visible;
Workers receive sanitized projections.

## Direction and Experiment Journal

Directions represent research/exploration hypotheses, not only code edits. Definitions are
immutable; status changes are append-only. At most one Direction may be in progress at a time and
one Attempt may advance at most three distinct Directions, while proposing additional future
Directions remains allowed.

Experiments bind one Direction to measured `before`/`after` Kernel Trials plus factual evidence,
interpretation, and action. The Agent supplies only each Trial ID; Runtime resolves and freezes its
exact Kernel and Result Artifact identities when appending the Experiment. Bootstrap establishes
its first measured anchor with `action="baseline"`, `before=null`, and a measured `after` Trial.

Journal calls synchronously validate and persist before replying. Their authority is scoped to the
logical Attempt, so a failed physical Session or retry generation does not erase records. The
terminal `attempt-report` snapshots the same Journal, requires no Direction to remain in progress,
and can be retried after validation failure until the first successful write-once publication.
Runtime, not the Agent report, decides retention.

Runtime derives a Final Attempt Report by joining the Agent handoff with authoritative parent and
Candidate outcomes, comparison, Production Gate result, correctness, latency by opaque Shape ID,
profile evidence, and Direction-bound Findings.

## Runtime State

Versioned Core Source includes initial `prompts/`, `memory/`, `knowledge/`, `skills/`, `tools/`, and `hooks/` seeds.
Runtime copies them when no inherited State exists. Learned State remains a separate Artifact:

```text
runtime-state/
  trajectories/<N>/
    prompts/README.md
    memory/README.md
    knowledge/README.md
    skills/README.md
    tools/README.md
    hooks/README.md
```

An Optimizer workspace presents one Trajectory's State as writable root `prompts/`, `memory/`, `knowledge/`, `skills/`,
`tools/`, and `hooks/`. Each README indexes that directory and tracks additions, edits, renames, and removals.
Runtime seals the terminal contents and restores them for the next serial Attempt. Evolver receives
frozen participant/historical State and writes one flat Candidate seed at
`candidate/{prompts,memory,knowledge,skills,tools,hooks}/`. A new Agent Revision records both source and State
digests as one logical Bundle; each new Trajectory receives an independent copy.

Without inherited State, Runtime copies the six initial directories from the pinned Core Source.
An Ablation Lineage with ephemeral Agent State returns to that seed on every Attempt/retry.
All six directories are sealed together and share the same inheritance and isolation rules.
Adaptive Knowledge are distinct from implementation documentation inside versioned Source.

## Evidence visibility

Visibility prevents information leakage across concurrent search:

- Optimizer sees promoted completed history plus earlier Attempts in its own current Trajectory.
- Optimizer does not see competing Branches while an Epoch is running.
- After the Epoch barrier, completed winning and losing journals may inform later Direction and
  Experiment history, but branch/selection provenance remains hidden from Optimizer tools.
- Evolver sees frozen latest-completed-Epoch Active/Challenger summaries and Conversations,
  historical Agent source/State summaries, and prior Evolution reports.
- Evolver has no Gateway, Wiki, journal, or Runtime-query authority; it reads the frozen filesystem
  and submits only through Bundle-local `evolution-report`.

Detailed Registry/Evidence trees are not duplicated into Agent workspaces. Runtime materializes only
the role-scoped files required for that Session.

## Agent Bundle and Evolution

Core and Evolver entry manifests each declare one repository-relative command. Runtime imports an
exact full commit, strips Git metadata, rejects unsafe tree content, and seals the complete source.
Runtime owns Backend/model policy and supplies phase, paths, usage, and scoped authority through the
launch contract.

Evolution input freezes current participants, visible historical Agents, Evidence, prior reports,
DSL, and Candidate seed. One output uses:

- `evolved`: new revision parented by Active;
- `reuse`: existing visible revision unchanged;
- `evolve_from_history`: new revision parented by a visible historical revision.

Runtime validates selected source, source-relative changed paths, private State diff, same-DSL
identity, file policy, and manifest before sealing. Evolved content is Lineage-local; Runtime never
pushes it to the Core repository.

## Capabilities and external services

Gateway and Wiki Capabilities are signed, Attempt-bound, operation-limited, expiring, revocable, and
generation-scoped. Idempotency keys replay the same committed response and reject changed requests.
Runtime-local journal/history queries do not call Agate or consume Gateway quota.

Agate credentials and private request construction stay in Runtime. GPU Wiki uses one live Query
contract; Runtime freezes the complete interaction before returning only knowledge content to Core.
There is no Wiki feedback/upload protocol.

Provider usage is fail-closed: QoderCLI reports credits; Claude, Codex, and Pi report provider-token
buckets. Optimizer has a configured per-Session quota. Evolver has no usage cutoff but must still
produce a complete terminal usage report unless process failure/timeout is the primary outcome.
