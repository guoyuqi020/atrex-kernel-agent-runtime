# Original AKA PPU CUDA comparison

Three independent sandboxes each run both operators: `gated_residual_combine_bf16`
(24 shapes) and `compressed_sparse_attention_paged_causal_bf16_interleaved_kv_append`
(25 shapes), on Agate production `ZW-M890P`.

AKA is upstream `main` commit `602bc381aec89a0238d18dbc903717ce848294ed`.
Launch arguments explicitly select `--framework Cuda --max-iters 15` and each
operator's isolated workspace. Other optimizer options retain upstream defaults:
Claude, leaderboard mode, two initial fast episodes with five trials each,
unlimited token budget, and target utilization 90. Claude uses the separately
provisioned account configuration; no credentials belong in this directory.

`operators/` contains evaluator inputs and metadata. It does not include the
curated `solution.py`. AKA establishes its own V0 from the reference wrapper.
CSA uses the independently validated compact input generator from
`../agate-inputs/`, with unchanged cases and reference. `experiment.json` records
source provenance and hashes.

The internal PPU Wiki is the complete `internal_source/gpu-wiki` subtree from
internal commit `2076c865cc6618d810cfe2bf09b4fc395536693e`. The public AKA plugin's
legacy two-store adapter cannot query this newer indexed store. A knowledge-only
plugin overlay selects the internal native `tools/query_nl.py`, preserving its
JSON responses, canonical record IDs and query evidence. AKA orchestration,
agent prompts and iteration logic are unchanged. The plugin still exposes
`gpu-wiki.query` with its original request/output schema; no Runtime service is
used by these experiments.

`launch.py` starts both independent tasks and refuses to overwrite an existing
launch record. Run data lives under `/workspace/runs/<operator>/` in each sandbox;
local deployment receipts belong in `~/atrex-runs/abla/20261011-aka-ppu-three-replicas/`.
