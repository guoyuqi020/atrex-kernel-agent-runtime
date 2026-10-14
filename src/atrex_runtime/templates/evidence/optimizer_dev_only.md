# Runtime workspace and Evidence contract: Dev-only arm

This Lineage disables both Journal modules, Kernel Trail, Result Artifact reads, and all
Agent-facing Runtime operations except Dev, Kernel source reads and terminal report handoff. Runtime-contract is
local schema discovery. This restriction does not change the controller's private evaluation
or Kernel selection rules.

Start from read-only `input/kernel/`; edit the candidate under `work/kernel/`. `agent/`,
`prompts/`, `skills/`, and `input/` are read-only. Reusable executable helpers belong in `tools/`;
keep its README index current. Temporary scripts, requests and output belong in `scratch/`,
which is not carried into later sessions. Do not write credentials or modify session capture.

Authorized history still exists under `input/evidence/`: Bootstrap and earlier Attempts expose
`report.json` and `conversation.jsonl`; completed Epochs have `summary.json` indexes. This is
not a no-memory arm. Recover only relevant facts from selected reports, then selected conversation
excerpts if needed. Do not read all old traces by default. Your current dialogue and own notes
may retain tool results. Use `kernel-artifact-read` to retrieve source from a visible Kernel
Artifact by its digest. Its CLI request contains `kernel_artifact_digest`, `artifact_file`
(path inside the source artifact), and `file` (destination under `scratch/`). This returns source,
not measurements, and follows the same history visibility rules. Result Artifact, Pareto,
Journal and Wiki queries remain disabled.
{{VISIBILITY}}

Run custom compilation, correctness checks, timing, profiling, or environment inspection through
Dev with your own script and public-contract inputs. Dev seals and uploads the current candidate
and returns stdout/stderr and job status; a successful process is not an official correctness or
performance verdict. No private evaluator inputs or credentials are exposed. Do not reconstruct
private cases from measurements. Record the harness, coverage, timing method, and limitations.
Repeated delivery of the same request may return its cached response; do not assume a new request
key is a new independent measurement without checking the returned job identity.

Use the live contract for the terminal Attempt Report schema. The CLI supplies empty tool_modules,
experiments and direction_events automatically. Submit only status, final_candidate,
contributing_result_artifact_digests and blocker. Keep custom Dev analysis in your own notes;
no analysis or Findings belong in the handoff. Do not invent official measurements or Result IDs.
A candidate_ready Report nominates the exact current source without requiring an Agent Evaluate.
After handoff Runtime privately validates it on the complete Valid population and applies the
normal retention policy. Dev output never substitutes for that gate. New Campaigns use 100% Valid
and 0% Test. Do not wait for a post-handoff measurement before submitting the report.
