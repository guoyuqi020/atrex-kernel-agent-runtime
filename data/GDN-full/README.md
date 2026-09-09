# GDN inputs with original hints

English | [中文](README.zh.md)

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
commits remain unchanged. Campaign creation keys are distinct from the cleaned package.

Defaults remain **L20D / CuteDSL / Claude**, five Epochs, three Attempts per trajectory,
100M tokens per Optimizer/Bootstrap Session, and the same seven-arm ablation plan.
“Full” means original input content, not access to hidden cases. Runtime/Core/KDA prompt
projection is unchanged: provenance metadata such as `range_evidence` and `value_evidence`
may still be omitted by the existing Agent formatter. No Agent code or pinned commit is changed.

In Lima Ubuntu, with the Linux Runtime environment activated:

```bash
# Prepare only: no services, Agents, or GPU jobs
python scripts/gdn/prepare.py --inputs data/GDN-full --backend claude

# Later, service process
python scripts/gdn/run.py serve --workspace workspaces/GDN-full
# In a separate terminal: Bootstrap plus the single Campaign
python scripts/gdn/run.py campaign --workspace workspaces/GDN-full --target-epoch 5
# Alternative experiment: seven-arm ablation
python scripts/gdn/run.py ablation --workspace workspaces/GDN-full
```

Preparation defaults to `workspaces/<input directory name>`; override with `--workspace`.
Run roles use that workspace's frozen inputs. Campaign and ablation are alternatives, not
commands to launch together by default. Runtime uses port 8766 and Wiki 8091, like GDN;
two Runtime services cannot bind the same port. See the [common launch guide](../GDN/README.md)
for credentials, worker permissions, scheduling and independent Wiki setup.
