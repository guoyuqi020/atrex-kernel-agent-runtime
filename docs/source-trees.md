# Multi-file Kernel source trees

English | [中文](source-trees.zh.md)

## Declaration and ownership

A Campaign Lineage selects **either** `baseline_kernel` (the existing single-file workflow)
**or** `source_manifest` plus `source_repository`. Paths resolve relative to the Campaign file.
Both forms still require `initial_evidence`; models, DSL identity, Epochs and evolution remain
unchanged. The [preparation example](../examples/source-tree/README.md) accepts the original GDN
manifest without converting its source into a generated single-file wrapper.

```json
{
  "cutedsl": {
    "source_manifest": "task/source_manifest.json",
    "source_repository": "source",
    "initial_evidence": "initial-evidence"
  }
}
```

The manifest uses `source.revision`, `source.archive_paths`, `source.package_root`, `adapter`,
`editable_roots`, and optional `runtime_requirements`. Runtime imports the **exact commit**,
ignoring checkout modifications, and copies the fixed adapter to the Evaluation Contract's
`candidate_path` (normally `kernel.py`). The original checkout is never mounted writable.
The complete seed is sealed in CAS. Runtime stores its digest, immutable file hashes, package
root, source revision, editable roots, and dependency requirements in the Campaign's sealed
`kernel_sources[DSL]` contract. These locks are not part of the Agent-editable Candidate.

GDN's old `measurement`, `bringup`, and lifecycle fields are accepted as source-document
metadata, **not** as Runtime policy. Runtime Gate settings control correctness cases, performance
iterations, repeats, clock locking and promotion. Only snapshot source imports are supported;
nonempty `runtime_support` uploads and alternative repository-search modes are rejected.
Provision declared distributions/versions in the Agate environment; Runtime does not install
arbitrary Agent-selected packages during evaluation.

## Workspace and identity

The seed is materialized read-only as `input/kernel/` and copied directly to `work/kernel/`:

```text
work/kernel/
├── kernel.py                       fixed adapter
├── LICENSE                         fixed
├── UPSTREAM_PROVENANCE.json         fixed
└── flashinfer/
    └── gdn_kernels/blackwell/        editable source subtree
```

Runtime injects the actual scope into the Optimizer prompt and updates the fragment's integrity
hash. Read-only file modes prevent accidental fixed-file edits. **Acceptance enforcement** is
the Runtime's source-lock validation, not file modes or the prompt: edits/deletions of fixed
files, out-of-scope additions, symlinks, import shadowing of protected dependencies, and prebuilt
loading artifacts are rejected. New/deleted source files inside editable roots are allowed.
An unchanged seed or historical tree is valid; no mandatory-edit rule applies to this mode.

Evaluate submission and final nomination use the same complete-tree sealing rule. Generated
Python caches, `.git` metadata and empty directories do not affect the identity. Other files
are not silently omitted. Keep build products in `scratch/`. Each Trial/Kernel Revision points
to the full tree; historical read, rollback and adoption preserve all its files. No Registry
schema or additional Kernel version hierarchy is introduced.

Production Gate validates the locked support files and scans editable source as one DSL
implementation. Bundled package and relative imports are allowed; this does not grant permission
to call an unbundled prebuilt kernel library or change DSL. It is a static policy check, not proof
that arbitrary Python execution cannot interfere with an evaluator.

## Evaluation and Bootstrap

The Agent-facing `evaluate` tool remains unchanged. Core/KDA already send directory bundles.
For source trees, Runtime transports each shape batch through **Agate Dev with a fixed driver**
and the deployment's commit-pinned Atrex Bench evaluator. The source snapshot, adapter, package
root, private input generator/shapes and Gate options are staged together. This is a logical
Evaluate operation in the durable journal, not Agent-controlled Dev evidence. Native single-file
Eval transport remains unchanged.

The common SDK submission boundary automatically moves Dev file maps larger than 4 MiB
(decoded UTF-8 bytes, including both ABBA snapshots and the evaluator) to Agate OSS. Runtime
creates one deterministic ZIP, calls `prepare_uploads` and `upload_file`, and attaches its opaque
reference via `oss_files`. A small inline bootstrap verifies SHA-256 and restores the exact paths
before the original command runs. Upload preparation, PUT and submission follow the usual retry
policy independently; submission retries reuse the uploaded reference. Logical request/cache
identity, Gate inputs and same-allocation ABBA semantics do not change. No additional Agent tool,
OSS credentials or configuration is required; the Agate service must support the SDK upload API.

Supported: full Evaluate, `correctness_only`, custom input/shape overrides, ordinary repeated
measurement, Agent exploratory ABBA, and Runtime authoritative ABBA. Source-tree ABBA uses the Dev
driver, while single-file ABBA uses Agate's native Eval ABBA API. In either path every step gets a
fresh process and independent JIT caches within the same allocation, and the complete A/B schedule
shares the existing Runtime clock-lock policy.

Source-tree Bootstrap launches the configured Optimizer backend for a complete framework-baseline
Session (Claude in the GDN kit). Runtime adds the source scope to a session-local copy of the
Bootstrap prompt, loaded by Core/KDA on every backend; the pinned Bundle and reusable prompts
remain unchanged. The Agent evaluates the supplied seed, repairs only
editable sources if needed, records the normal Direction/Experiment Journal, and submits a standard
Attempt Report. An unchanged correct seed is a valid nomination. Runtime seals the entire source
tree using the same rules as exploratory Evaluate, verifies matching Agent evidence, and applies
the independent Bootstrap Gate before registering v0. The real Session trace and standard report
become Lineage history. Retry and finalization recovery use the normal Bootstrap machinery.

Existing v0 records remain immutable and are reused on resume. To test this flow after a former
model-free Bootstrap, use a new Campaign creation key; do not rewrite historical reports or
weaken Journal validation.

## Profile, Check and Disassemble

The existing tools also support whole source trees through a Runtime-generated Dev driver:

```json
{"operation":"profile","level":"sol","shape_id":"0"}
{"operation":"profile","level":"deep","kernel_regex":".*GatedDelta.*","source":true,"launch_count":1}
{"operation":"check"}
{"operation":"check","sanitize":"memcheck"}
{"operation":"disassemble","fmt":"sass"}
```

- Profile uses NCU: `survey` collects launch statistics and duration, `sol` collects SpeedOfLight,
  `deep` collects the full set and requires a Kernel filter. `counters` adds metrics; `kernel_name`
  is an exact demangled match, while `kernel_regex` is a regex. `source=true` requests correlated
  source/SASS output. `top_kernels` limits the structured view; raw CSV retains collected launches.
- Runtime selects one opaque `shape_id` (first sorted ID by default), warms up the model outside
  the profiled region and allocates fresh inputs before the measured forward. `launch_skip` and
  `launch_count` select launches **within that one forward**, not benchmark repeats. Defaults are
  0 and 10; a filter/range yielding no kernels is an explicit failure.
- Check invokes the fixed adapter's Model and one forward, triggering lazy JIT compilation.
  It is a launch probe, **not a numerical correctness test**. Optional `sanitize` runs
  `memcheck`, `racecheck`, `initcheck`, or `synccheck` with a nonzero error exit code.
  Optional `arch` must match the allocated GPU's base compute capability (e.g. `sm_103`);
  cross-compilation and architecture suffixes are not supported.
- Disassemble collects an NCU report and exports SASS (`auto`/`sass`) or PTX (`ptx`). PTX requires
  a supporting NCU/toolchain and available PTX in the report; absence is reported, not fabricated.
  Check and Disassemble use the first sorted opaque case. These drivers currently target NVIDIA;
  `rocprofv3` and AMD `fmt=isa` are explicitly rejected.

Each job gets fresh JIT caches and follows the Contract's `lock_clocks`. No evaluator code or
Reference model is uploaded for these diagnostics, only the selected input generator/case and
sealed Candidate. Declared dependencies must already be provisioned; `requirements` are validated
against installed distributions, not installed by Dev. There is no dependency installer in either
`deps_mode`. The Dev allocation is capped at 600 seconds, reserving 30 seconds for cleanup.

The logical operation remains Profile/Check/Disassemble in Runtime's durable job ownership and
result history. Job binding and upstream idempotency keys are scoped to the Attempt recovery
generation: a new Session after an interrupted Attempt does not collide with old jobs. Within
one generation, retrying an already submitted request resumes its persisted job instead of
allocating another Dev job. Historical job ownership and results remain intact.
`passed=false` / `status=error` means the diagnostic failed even if transport
completed. Missing structured output cannot be accepted as success. Results contain parsed Kernel
metrics and `exports` (CSV/SASS/PTX text with byte count, SHA-256 and explicit truncation at 512 KiB
per export); these are recorded in Result Artifacts. Binary `.ncu-rep` files are not transferred
back by this driver. Tool stdout/stderr are retained in trusted raw evidence, not the Agent's
hidden-case result view. Diagnostic evidence never substitutes for Evaluate or promotion evidence.

The NCU export flags follow the [NVIDIA CLI documentation](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html);
sanitizer error handling follows [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html).

Current limits: automatic NCU fallback is still skipped for these trees; an available Roofline
still supplies SOL, and explicit Profile returns measured compute/memory SOL. Disposable
Optimizer dev-shell still takes the single-file baseline form; source-tree tasks can use the
existing durable-Lineage dev-shell after Bootstrap. GPU execution and environment compatibility
must be validated on the selected deployment; local tests use a CPU evaluator double.

## Implementation and verification

- `kernel_sources.py`: source import, lock, scope validation, canonical tree and prompt projection.
- `gateway/source_tree.py`: logical Evaluate ↔ trusted Dev transport, restart-safe result parsing.
- `gateway/source_diagnostics.py`: fixed source-tree NCU, compile/launch and sanitizer drivers.
- `gateway/abba.py`, `abba_remote.py`: shared fixed driver, complete A/B trees and process isolation.
- `gateway/proxy.py`, `workers/core.py`: identical submission/nomination sealing.
- `workers/lineage_bootstrap.py`, `composition/bootstrap.py`, `gateway/finalization.py`: Agent Bootstrap, source nomination and authoritative finalization.
- `tests/test_kernel_sources.py`: exact commit, scope attacks, directory identity, prompt digest,
  source execution, recovery polling, ABBA cache isolation and Bootstrap gate coverage.
- `tests/test_source_diagnostics.py`: real driver/child execution with CPU GPU-tool doubles,
  NCU arguments, sanitizer failures, exports and private-result projections.
