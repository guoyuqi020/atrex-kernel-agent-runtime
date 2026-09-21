# FA4 SM120 task alignment

English | [中文](ALIGNMENT.zh.md)

The SM120 task deliberately reuses the original FA4 task's operator semantics while changing
only architecture-dependent material:

| Asset | Relationship to `data/FA4` |
| --- | --- |
| `adapter.py`, `reference.py`, `input.py` | byte-identical |
| `shape_train.json`, `shape_valid.json` | byte-identical; all 30 Shapes retained |
| Evaluator Bundle | byte-identical and pinned to the same Commit |
| Metadata | same capture evidence; task ID and target hardware identify L20N/SM120 |
| Roofline | same semantic FLOPs/bytes; SOL recomputed from L20N SM120 peaks |
| Source Bundle | same upstream FA4/Quack closure plus one committed SM120 Vendor overlay |
| Initial Evidence | rewritten for SM120; contains no SM100 2CTA optimization prescription |

The packaged overlay is inside `interface.py`; the fixed adapter remains byte-identical. It
turns the unsupported combination `SM120 + FP8 + paged KV` into a correctness-first route by
materializing BF16 dense K/V before the existing SM120 FA4 kernel. This makes every temporary
allocation, conversion and Host synchronization part of the timed Candidate boundary and leaves
them available for optimization.

The task does not claim that historical L20D latency applies to L20N. It retains those records
only as Shape provenance. L20N SOL is computed as
`max(FP8 FLOPs / 549e12, bytes / 1.344e12)`.

The optional target smoke uses the same elementwise `atol=0.06`, `rtol=0.04` predicate as the
authoritative evaluator. It is a one-Shape diagnostic, not a substitute for Runtime validation.

`asset-integrity.json` records the complete task and Bundle hashes. Preparation verifies every
file, both Git Commits, public/private Shape consistency and the pinned Optimizer/Evolver before
publishing a workspace.
