# FA4 SM120 task alignment

English | [中文](ALIGNMENT.zh.md)

The SM120 task reuses the original workload while replacing the Candidate boundary:

| Asset | Relationship to `data/FA4` |
| --- | --- |
| `reference.py`, `input.py` | byte-identical |
| `shape_train.json`, `shape_valid.json` | byte-identical; all 30 Shapes retained |
| Evaluator Bundle | byte-identical and pinned to the same Commit |
| Metadata | same capture evidence; task ID and target identify L20N/SM120 |
| Roofline | same semantic FLOPs/bytes; SOL recomputed from L20N/SM120 peaks |
| Fixed Adapter | same production ABI, but dispatches only to the new SM120 implementation |
| Source Bundle | immutable SM103-family reference plus an empty editable SM120 implementation |

No Vendor tree or SM120 fallback is supplied. The immutable reference preserves the pinned
SM103-family path used by the original task while excluding SM120 source and the mixed-arch
dispatch interface, so an Agent can study its design while Runtime permits writes
only under `implementation/`. The Candidate must therefore implement SM120 rather than tuning or
repairing a provided SM120 bridge.

Historical L20D measurements remain Shape provenance and do not claim L20N performance. L20N SOL
is `max(FP8 FLOPs / 549e12, bytes / 1.344e12)`. `asset-integrity.json` records all task and Bundle
hashes; preparation verifies the source and evaluator Commits before publishing a workspace.
