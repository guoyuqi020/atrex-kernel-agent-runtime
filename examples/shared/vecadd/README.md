# Canonical VecAdd fixture

English | [中文](README.zh.md)

This is the only VecAdd dataset used by the examples:

- `reference/`: PyTorch reference, `_make_inputs`, and Shape metadata for direct Agate commands;
- `evaluation-contract.json`: the Runtime/Gateway representation of the same trusted workload;
- `agent-problem.json`: public Agent-facing problem statement;
- `triton/baseline-kernel/`: reference-shaped seed for Core framework baseline;
- `triton/initial-evidence/`: trusted Epoch-zero notes;
- `triton/agate-candidate/`: correct Triton candidate for direct Agate CLI demonstrations.

Runnable examples treat this tree as read-only and write all generated state under `workspaces/`.
