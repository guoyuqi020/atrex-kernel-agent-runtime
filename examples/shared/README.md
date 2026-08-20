# Shared example resources

English | [中文](README.zh.md)

This directory contains read-only inputs and generic launch helpers used by more than one runnable
example. Runnable workflows never source or execute another example directory.

- `vecadd/` is the single canonical VecAdd fixture: trusted reference, input generator, shapes,
  Evaluation Contract, Agent Problem, Triton baseline seed, initial Evidence, and direct-Agate
  candidate.
- `prepare_campaign.py` materializes an example-owned Runtime config and Campaign definition into
  that example's generated workspace.
- `runtime-common.sh` provides generic secret, prerequisite, Bootstrap, Lineage-ID, and owned-Runtime
  lifecycle helpers. Each caller declares its own Runtime and Campaign templates first.
- `local-secrets.sh` is the single implementation for private generated Runtime control secrets.

Shared files are inputs, not an independently runnable example.
