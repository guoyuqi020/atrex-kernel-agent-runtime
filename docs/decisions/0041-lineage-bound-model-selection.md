# Decision 0041: Bind concrete model selection to Lineage

## Decision

Runtime deployment configuration continues to select the Optimizer and Evolver Backend, credentials,
executable environment, reasoning effort, and opaque session settings. Campaign schema v3 selects
concrete model identities separately:

- optional top-level `problem_generalization_model` applies only to Core problem generalization;
- each `lineages.<dsl>.models.optimizer` applies to framework baseline and every Optimizer Attempt;
- each `lineages.<dsl>.models.evolver` applies to Challenger construction.

An omitted or null model delegates selection to the configured Backend CLI default. Runtime persists
the Campaign-level and Lineage-level values in the Registry, rejects model drift when resuming the same
creation key, and injects the selected value through `ATREX_AGENT_MODEL`. Core and Evolver preserve the
value in Session provenance and translate it to the native Claude, Codex, QoderCLI, or Pi command.

## Consequences

Different DSL Lineages in one Campaign can use different Optimizer and Evolver models while retaining
one deployment-controlled Backend and credential boundary. A Lineage resume is reproducible with
respect to its declared model identity. Campaign authors may rely on provider defaults by leaving the
fields null, but a later provider-default change is then intentionally outside Runtime's identity
guarantee.

Codex and Pi structured session settings cannot also declare a model when Campaign schema v3 provides
one; the Adapter rejects the ambiguous launch. Changing a persisted model requires a new Campaign
creation key rather than mutating an existing Lineage.
