# Full-repository Optimizer revision

English | [中文](full-repository-optimizer-revision.zh.md)

## Decision

One pinned commit of `git@github.com:guoyuqi020/atrex-kernel-agent-core.git` is the complete Base Revision of the Optimizer. Runtime does not repackage the repository into independently versioned prompt, workflow, memory-policy, tool, or DSL-overlay components.

The repository commit and the immutable Artifact digest serve different purposes. Git identifies the upstream source and supports synchronization and review. The Artifact digest is the exact source snapshot used by a lineage after Git metadata has been removed and file-policy validation has completed.

## Runtime loading

The trusted control plane performs these operations before a Campaign starts:

1. fetch the deployment-approved repository;
2. resolve and verify an explicit full commit SHA;
3. export the complete tracked tree without `.git` metadata;
4. reject unresolved submodules, symbolic links, special files, oversized content, and an invalid Runtime entry manifest;
5. seal the complete tree as one immutable Optimizer Artifact;
6. register the repository URL, commit SHA, source-tree identity, and Artifact digest as Base Revision provenance.

Branches and tags are synchronization inputs, not production identities. An existing lineage never changes when the upstream branch advances.

## Runtime entry manifest

Core is a maintained fork of the upstream repository rather than a path-compatible overlay. A small
root manifest declares a framework-neutral repository command. The manifest does not enumerate the
whole payload because the whole tracked repository is the revision.

The command owns Core's Agent-framework selection and prompt/workflow assembly. The manifest has no framework-specific adapter section. It cannot define credentials, Gateway/Wiki authority, mounts, network access, evaluation rules, or promotion policy. Runtime validates the command path and supplies trusted task context and scoped capabilities separately.

## Optimizer execution

Runtime materializes the complete Optimizer revision read-only in every fresh Attempt, executes `entrypoint.command` through a deployment-owned command prefix, and passes trusted Campaign, lineage, Epoch, DSL, hardware, Evaluation Contract, Evidence, Gateway, Wiki, report, and token-budget context through the versioned Attempt manifest and environment protocol. Core chooses and invokes its Agent framework; Runtime has no parallel framework-specific launch path.

The Runtime entrypoint must execute exactly one Attempt. The old outer Campaign CLI does not automatically satisfy this contract and is not allowed to start another Campaign, create Git Worktrees, launch a local Gateway, manage canonical memory, or perform promotion. Those responsibilities remain in Runtime even if migration-source code is temporarily present in the Optimizer revision.

## Self-evolution

The separately versioned fixed Evolver Bundle receives a read-only parent Agent, completed Epoch
Evidence, and writable Candidate `source/` plus `runtime-state/{skills,tools}/` initialized from
Active Source and the latest completed Epoch winner's best-Kernel Trajectory terminal State after
that Epoch's last Attempt. The next Epoch's Active Branch uses the same State seed. Missing terminal
State falls back to that Trajectory's Epoch-start State, the revision seed, and empty default. For
`evolve_from_history`, Evolver replaces Source with an
ordinary writable copy from one visible historical Agent and may synthesize the common seed from
visible historical state; final sealing verifies the declared Agent revision, reported Source Diff,
and private State Diff and
stores complete Source plus State as one logical Bundle. Evolver may change any
validated Optimizer-owned file, including Agent-framework selection, prompts, workflows, Skill/Tool
lifecycle mechanisms, helper implementations, and the revision-wide Candidate state seed. Top-level
`skills/` and `tools/` are invalid in source. It cannot modify Runtime code,
Evolver code, or policy because those files are not present in the Candidate. Recursive Evolver
self-evolution remains deferred and requires its own evaluation and promotion design.

The candidate is accepted only when its complete repository snapshot passes file policy, manifest validation, size limits, and independent Active-versus-Challenger evaluation. A successful lineage revision remains private to that lineage. Runtime never pushes, merges, rebases, or creates refs in the upstream Core repository.

Upstreaming a successful evolved revision is a separate reviewed export process. Upstream synchronization likewise creates a new candidate Base commit; it never mutates an existing lineage revision.

## Core simplification strategy

Core records the upstream base commit in Git history, while synchronization is a reviewed merge or
patch-port into the maintained `src/`, `prompts/`, and manifest layout. Runtime never performs this
synchronization automatically. Structural divergence is accepted where Runtime-owned control-plane
code has been removed; each upstream update must rerun Core unit/static checks and Runtime contract
tests before producing a new Base Commit.

| Existing Core responsibility | Treatment |
| --- | --- |
| Episode prompt and Kernel optimization workflow | Consolidated under root `prompts/`; Runtime operations use exact hyphenated Core CLI commands |
| Useful result-analysis knowledge | Keep only when compatible with structured Runtime results and the fixed writable roots |
| Agent implementation and Backend adapters | Maintained under `src/`; each Backend remains part of the evolvable repository |
| `long_horizon/` Episode, Journal, Handoff, and comparison code | Removed after its durable semantics moved to Runtime Evidence and Registry state |
| Git Worktree, Campaign scheduling, process/session supervision | Removed from Core; Runtime owns these responsibilities |
| Local Gateway and old sandbox transport | Replaced by the framework-neutral Core `gateway-execute` protocol client; every Backend uses the same Runtime contract |
| Local GPU Wiki and direct query script | Replaced by the Core `wiki-query` protocol client; Wiki content is an external service |
| Canonical memory mutation and iteration marker scripts | Replace with Runtime Evidence, structured experiment/report protocols, and the selected Agent framework's trace |
| Evaluation adapters and hidden-input handling | Replace with the Runtime Evaluation Contract and Gateway |
| Reference-project checkouts and nested dependencies | Remove unless a specific helper still requires them and Runtime policy admits them |

Only the Core entry command determines active Optimizer behavior. Historical upstream code that is
not part of the maintained Bundle is consulted through Git history, not retained as inactive source.
