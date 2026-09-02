# Design principles

English | [中文](design-principles.zh.md)

Atrex Kernel Agent Runtime exists to let a GPU Kernel Agent change itself without letting the
changing component become the authority over evaluation, promotion, isolation, or history. The
central design rule is therefore:

> Agents propose and explain changes; Runtime controls identity, execution, measurement,
> comparison, persistence, and promotion.

This document explains why the system is shaped this way. [Architecture](architecture.md) describes
the resulting components and lifecycle; [Protocols](protocols.md) defines their durable contracts.

## Why a Runtime control plane is needed before Agent evolution

Agent evolution is not the original reason to separate Runtime from the Kernel Agent. Even a fixed
Optimizer needs a trusted control plane when a long-running optimization Harness encodes control,
memory, measurement, and observability mainly through prompts and workspace files.

The upstream Atrex Kernel Agent pinned under `third_party/atrex-kernel-agent/` has already improved
substantially over its early clean-session workflow. Its current Long Horizon engine asks the Agent
to append decisive experiments to an episode Journal as they happen, refreshes a non-canonical
`memory/live.json` view, lets the Supervisor construct canonical `memory/vN.json`, and independently
verifies terminal candidates. These changes reduce end-of-session recall loss and keep Agent claims
out of the final promotion authority. They do not, however, make the Journal an automatic record of
what actually happened.

### Limits of a prompt-governed Harness

1. **Workflow is encoded as binding instruction.** Fast and full episodes prescribe fixed planning,
   review, profiling, implementation, evaluation, recording, and handoff protocols. The Agent pays
   the instruction and execution cost even when a stage has little value for the current operator,
   DSL, or evidence. Adding more recovery and evidence rules also increases the protocol surface the
   Agent must understand correctly.
2. **Cross-session cognition remains summary-centered.** Canonical Memory, plans, profiles, and
   archived Journals exist on disk, but the next episode primarily reconstructs prior work from a
   compressed `memory/vN.json` view. Detailed evidence is retained without becoming a uniform,
   query-first history of Directions, Kernel Trials, measurements, and failures.
3. **Journal completeness still depends on the Agent.** After an experiment, the Agent must decide
   that it is worth recording, parse the evaluator output, construct structured `evaluation` fields,
   and invoke the Journal CLI. If it forgets, crashes, times out, compresses away an earlier event, or
   reports a result incorrectly, neither the Journal nor its `memory/live.json` mirror contains the
   missing fact.
4. **Recovery repairs protocol, not absent evidence.** Same-session handoff recovery and interrupted
   worktree recovery are useful, but they cannot reconstruct a Kernel modification, measurement, or
   failure that was never captured. Recovery may restore execution while still preserving only an
   incomplete account of why the execution reached that state.
5. **Intermediate facts and Agent interpretation are not fully separated.** Terminal promotion can
   use Supervisor verification, but an episode Journal's intermediate correctness, latency,
   bottleneck, and decision fields are still an Agent-authored interpretation of tool output. A
   structurally valid report is not proof that it faithfully represents the original result.
6. **In-flight observability remains file-driven.** Active state, live memory, Journals, and telemetry
   improve visibility, but progress appears only when their producers update them. They do not by
   themselves provide one durable view joining the complete Session trace, every Runtime Tool call,
   exact Kernel bytes, Gateway result, Experiment, and later selection.

The architectural lesson is not that Agents must stop writing plans or analyses. It is that an
Agent-authored report must not be the only place where an execution fact exists.

### Capture facts at the execution boundary

Runtime records facts when the authorized operation occurs, before returning control to the Agent:

| Event | Runtime-owned durable fact | Agent-owned interpretation |
| --- | --- | --- |
| Kernel submission | exact Kernel Artifact, digest, Attempt, Session, and timestamp | intended change and hypothesis |
| Evaluate or Profile | operation, request binding, Gateway-result Artifact, normalized correctness, and per-Shape measurements | bottleneck analysis and significance |
| Tool failure | operation, stable error category, response, and retry history | diagnosis and proposed repair |
| Experiment | immutable before/after Kernel and Gateway-result references | analysis, lesson, and next decision |
| Attempt completion | retained Session trace, Journal snapshot, and nominated measured Candidate | terminal narrative and open Directions |
| Promotion | Runtime comparison, Gate outcome, ancestry, and selected Revision | no unilateral Agent authority |

The Agent can still omit or revise an analysis. It cannot make an executed measurement disappear,
replace its original values, or create a promotion merely by writing a convincing report. A terminal
report is therefore a validated view over already durable facts, not the source of those facts.

This separation is useful before any self-evolution is enabled. Agent evolution later builds on the
same boundary instead of creating it.

## 1. Keep the control plane stable and the workers evolvable

Runtime is ordinary trusted Python, not an Agent. It performs deterministic scheduling and owns
policy. Core and Evolver are commit-pinned Agent Bundles running as untrusted Workers.

An Optimizer may change Kernel source and adaptive `skills/` and `tools/`. An Evolver may change
the Optimizer's source, workflow, skills, and tools. Neither may change the Registry, reveal private
validation inputs, grant itself capabilities, choose its own promotion result, or rewrite history.

This boundary permits broad Agent evolution while keeping failures recoverable and results
comparable.

## 2. Separate proposal, measurement, and promotion

Kernel generation, GPU measurement, and version promotion are different authorities:

1. the Optimizer proposes a Kernel and analysis;
2. Agate executes the Runtime-constructed request;
3. Runtime records the exact Kernel Trial and Gateway result;
4. Runtime applies the configured comparison and decides whether to retain a Kernel Revision;
5. Runtime separately decides whether an Agent Revision wins the Epoch.

Gateway measurements are durable facts. Agent explanations, plans, and nominations are Evidence,
not authority. A Kernel may be retained without promoting its Agent, and an Agent comparison may
use the best Kernel produced by a Branch rather than its last Attempt.

## 3. Evolve locally, not globally

Each DSL in each operator Campaign owns a separate Lineage. CUDA, Triton, and CuteDSL can benefit
from different prompts, tools, search strategies, and accumulated state; an improvement in one does
not imply a safe global Agent upgrade.

Local evolution also makes attribution clearer: every Agent and Kernel version is measured under
one immutable operator, hardware target, Evaluation Contract, DSL, and ancestry. Reuse across
Lineages requires an explicit new seed instead of invisible global promotion.

## 4. Use two loops with different horizons

The inner loop optimizes Kernels. Every Attempt is a fresh Optimizer Session that may explore and
measure multiple Kernel Trials before submitting one terminal report.

The outer loop optimizes the Agent. At an Epoch boundary, Evolver studies completed conversations,
outcomes, Agent source, and reusable state, then creates Challenger proposals. Active and
Challenger Agents run the next controlled competition from the same Kernel and State starting
point.

Agent evolution is therefore scheduled evidence-driven experimentation, not an emergency mutation
triggered inside a failing Session.

## 5. Prefer fresh context and durable state

An Attempt retry starts a new physical Session. Context is never treated as the durable memory of a
Lineage. Runtime persists the data needed across Sessions:

- immutable Agent and Kernel Artifacts;
- exact Kernel Trials and Gateway results;
- Direction, Experiment, Attempt, and Evolution reports;
- retained conversations and provider usage;
- adaptive Runtime State in `skills/` and `tools/`.

This prevents context pollution, makes retries observable, and lets later Agents inspect history
without replaying an unbounded conversation into every prompt.

## 6. Make state append-only and recovery explicit

Content-addressed Artifacts never change. Logical versions (`vN`, `agent-vN`) are Lineage-local
labels over immutable records. Creation keys make repeated commands idempotent; leases and fences
prevent concurrent schedulers from committing the same transition.

A failed Session, Bootstrap Generation, or Epoch remains in history. Recovery advances a new
Generation or Session instead of editing the failed record. Active remains usable until a measured
Challenger wins, so self-evolution always has a rollback point.

## 7. Expose the train domain, keep validation private

Agents receive the public operator contract and `shape_train` domain, but not exact validation
Shapes, reference/input implementations, private metadata, or Roofline inputs. Agent-visible Shape
IDs are opaque.

This gives the Optimizer enough structure to generalize while preventing direct memorization of the
acceptance set. Runtime constructs correctness, performance, Production Gate, and promotion
requests from the sealed private Evaluation Contract.

## 8. Give Workers narrow tools, not ambient authority

Workers receive Attempt-scoped capabilities and a single workspace. Runtime Tools bind requests to
the current Attempt and expose only authorized history. Agate and GPU Wiki remain external
services behind Runtime-controlled clients and projections.

The GPU Wiki supplies external knowledge only. Agent-produced experiments, conversations, and
adaptive state belong to their Lineage rather than being silently aggregated into the Wiki.

## 9. Do not add an evolvable coordinator without evidence

Campaign scheduling, concurrency, retries, and promotion are deterministic control-plane work and
remain in Runtime. Evolver already owns the high-value meta-level decision: how the Optimizer should
change. A separate Coordinator Agent would add cost, another mutable authority, and harder
attribution without currently owning a distinct optimization problem.

The architecture can introduce a Coordinator later only if it has a measurable objective and a
bounded authority that Runtime can independently evaluate.

## Consequences

| Choice | Benefit | Cost |
| --- | --- | --- |
| Commit-pinned Agent Bundles | Reproducible source and controlled evolution | Updating an Agent requires a new Revision |
| Fresh Session per Attempt | Clean context and isolated failure | More process startup and explicit State handling |
| Lineage-local promotion | Specialized Agents and clear attribution | No automatic global knowledge transfer |
| Active/Challenger competition | Rollback-safe Agent evolution | Extra evaluation cost |
| Private validation Contract | Lower overfitting risk | Agents cannot directly debug exact hidden Cases |
| Immutable Artifacts and append-only history | Auditability and recovery | Storage and lifecycle machinery |
| Runtime-owned comparison | Trustworthy selection | Agents cannot unilaterally declare success |

## Deliberate non-goals

The current Runtime does not attempt multi-node scheduling, cross-Campaign memory aggregation,
global Agent promotion, recursive Evolver self-evolution, or an Agent-controlled trust boundary.
These may be added only with explicit authority, evidence, and recovery semantics.
