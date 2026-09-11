# Analysis evidence

`generate.py` combines two evidence families:

- Runtime: the Registry is authoritative for Campaign, Epoch, Attempt, Session, Kernel Revision, Agent Revision, measurement, and Token facts; `campaign-results.json` supplies the frozen arm labels.
- AKA: the run7/run8 Manifest supplies Episode outcomes and final PASS candidates; raw Claude traces supply Provider usage and tool-call counts. Usage is deduplicated by taking the final event for each assistant `message.id`.

Runtime input expects an extracted production root containing:

- `control-l20n/state/registry.sqlite`;
- `control-l20n/state/artifacts/sha256/<digest>/` for the Result, Kernel, Attempt Report, Runtime State, and Evolution artifacts referenced by the Registry;
- `production-qwen35-35b-fp8-atrex-gdn-4k256-20260814--flash-attention--l20n--claude/campaign-results.json`.

AKA input is discovered under `~/atrex-runs` from these two directories:

- `production-qwen35-35b-fp8-atrex-gdn-4k256-20260814--flash-attention--l20n--claude--standalone-run7`;
- `production-qwen35-35b-fp8-atrex-gdn-4k256-20260814--flash-attention--l20n--claude--standalone-run8`.

Their small workspace and trace archives are extracted automatically into `/tmp/atrex-flash-aka`. The three Runtime Bootstrap Kernel artifacts must also be extracted because the generator compares their source with the AKA V1 Commits. Pass their common `artifacts/sha256` parent through `--runtime-baseline-artifacts`.

Run:

```bash
python3 analysis/generate.py \
  --extracted-root /path/to/extracted/production \
  --runtime-baseline-artifacts /path/to/extracted/production/control-l20n/state/artifacts/sha256
```

Generated outputs are `summary.json`, `latency-curves/`, `best-kernels/`, and `retained-state-examples/`. `latency-curves/` includes AKA run7/run8 Best-of-Two alongside all Runtime arms. The committed outputs are the frozen evidence used by [`REPORT.md`](../REPORT.md); rerunning is only needed when a source archive changes.
