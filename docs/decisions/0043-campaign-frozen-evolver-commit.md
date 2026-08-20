# Decision 0043: Freeze the Evolver commit into Campaign identity

## Status

Accepted.

## Decision

The first successful Campaign Bootstrap copies the deployment-selected full
`campaign.evolver.commit` into the durable Campaign record. Every later Bootstrap retry,
`run-campaign`, durable Campaign Task, and Evolver dev shell must use that exact Commit. A Runtime
configuration that selects another Commit is rejected before the Evolver Bundle is resolved or run.

Artifact-seeded Lineages inherit the destination Campaign's frozen Evolver Commit. Their selected
Agent Artifact replaces Core Bootstrap for the new root, but does not establish another Evolver
identity.

Registry schema 23 adds the nullable `campaigns.evolver_commit` column. Null exists only for
Campaigns created before this decision. The next identical Bootstrap retry or fenced Campaign run
binds such a legacy Campaign once and emits `campaign.evolver_commit_bound`; subsequent changes are
rejected. Evolver dev shell refuses a legacy-null Campaign because it must not choose provenance as
a side effect of a debugging operation.

## Consequences

Optimizer and Evolver provenance now have symmetric resume behavior: Kernel Agent revisions are
content-addressed, while the executable Evolver repository is fixed per Campaign. Updating the
deployment default affects only new Campaigns. Intentionally testing another Evolver Commit requires
a new Campaign or an explicitly designed future migration operation; editing Runtime JSON cannot
silently change an existing experiment.
