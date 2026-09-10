# GDN inputs with original hints

English | [中文](README.zh.md)

For concurrent GDN/GDN-full runs, follow the [shared-service commands](../../scripts/gdn/README.md).
Attach two new task workspaces to one prepared service; do not start two Runtime listeners.
The paths and commands below are for standalone mode. Shared mode stores Session/Artifact/DB
and secrets under the service workspace, while task inputs and results remain separate.

This self-contained counterpart to [`data/GDN`](../GDN/README.md) retains the original
optimization hints. This directory holds inputs/templates only; generated files belong
to `workspaces/GDN-full/` or an explicitly selected workspace.

Restored content:

- The original **M64-oriented** objective and SM103 wording.
- All `range_evidence`, `value_evidence`, and `coverage_regimes`, including low-parallelism
  descriptions and M64 regime names.
- The original seed provenance and initial-evidence descriptions mentioning M64.
- Seed commit `a39405536f178689d7f60b551c17b2252bcee61d`, included in `source.bundle`.

`task/shape_train.json` is byte-for-byte the original imported file (SHA-256
`7695093a901dc50f595ba0b0cd95df273882a3f14c5c457f6b050789d9437df9`). Kernel source files
are identical to the cleaned package; only `UPSTREAM_PROVENANCE.json` differs between the
seed trees. No optimized implementation, winning Kernel, Session history or memory is imported.
Reference, inputs, adapter, shape domains/cases, Metadata, Roofline, Gate policy and Agent
commits match the cleaned package. Campaign creation keys are distinct from the cleaned package.

Defaults remain **L20D / CuteDSL / Claude**, 100 Epochs, three Attempts per trajectory,
100M tokens per Optimizer/Bootstrap Session, and the same seven-arm ablation plan.
New deployments use `container` (bwrap, current user, no systemd/per-Session cgroup).
Outer-container resource limits are required separately; no Docker container is created by the scripts.
“Full” means original input content, not access to hidden cases. Runtime/Core/KDA prompt
projection is unchanged: provenance metadata such as `range_evidence` and `value_evidence`
may still be omitted by the existing Agent formatter.

Both Campaign definitions pin KDA commit `41af4a45ca4155254f3c2e8d501ae28a5fb5bb62`,
matching GDN. It bundles neither KernelWiki nor ncu-report-skill; no Skill submodule checkout
is needed and `allowed_submodules` is empty. Existing workspaces keep their frozen Agent
revisions. Local uncommitted KDA edits are not included in the Bundle.

Evaluator and Roofline share GDN's Atrex Bench pin
`54925ff9223aa54b901219f02fffd51d9af82e3c`. The same preparation preflight exercises all
four real Bundle loaders and records `prepared.json.source_preflight`; a local commit-object
existence check alone is not considered sufficient. Existing workspace pins are not overwritten.

In Lima Ubuntu, with the Linux Runtime environment activated:

```bash
# Prepare only: no services, Agents, or GPU jobs
python scripts/gdn/prepare.py --inputs data/GDN-full --backend claude

# Later, service process
python scripts/gdn/run.py serve --workspace workspaces/GDN-full
# In a separate terminal: Bootstrap plus the single Campaign
python scripts/gdn/run.py campaign --workspace workspaces/GDN-full --target-epoch 100
# Alternative experiment: seven-arm ablation
python scripts/gdn/run.py ablation --workspace workspaces/GDN-full
```

Preparation defaults to `workspaces/<input directory name>`; override with `--workspace`.
Run roles use that workspace's frozen inputs. Campaign and ablation are alternatives, not
commands to launch together by default. Runtime uses port 8766, like GDN;
two Runtime services cannot bind the same port. Wiki is disabled (`gpu_wiki: null`);
no Wiki service or corpus is required. See the [common launch guide](../GDN/README.md)
for credentials, worker permissions and scheduling.
