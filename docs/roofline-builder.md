# Trusted Roofline construction

English | [中文](roofline-builder.zh.md)

Runtime can construct a missing `roofline.json` during Campaign Bootstrap by executing the
canonical Atrex Bench benchmark-converter from a deployment-approved full Git commit. This is a
trusted control-plane phase: Core and Evolver neither see the private Shapes nor author the
Roofline.

## Resolution order

1. If the submitted Evaluation Contract already contains `roofline`, Runtime uses it unchanged.
2. If `roofline` is null and `campaign.roofline_builder` is configured, Runtime builds it once.
3. If the Builder is absent or fails, Bootstrap remains valid and Runtime seals the contract without
   a Roofline. Every correct Kernel evaluation then runs `eval` followed by
   `profile --level sol`.

For an existing Campaign, Runtime loads the already sealed generated Evaluation Contract when the
submitted non-Roofline fields are unchanged. It does not re-run a newer Builder and silently alter
the Campaign metric.

## Builder contract

The configured repository must contain
`skills/benchmark-converter/scripts/generate_roofline.py` at the configured commit. Runtime exports
that exact plain Git tree, rejects links and submodules, writes only `shapes.json` and
`metadata.json` into a temporary operator directory, and invokes the converter with an optional
hardware-target-to-SKU mapping. The converter must have an `op_cost` implementation for the
operator.

Generated output is accepted only when it:

- contains exactly the Evaluation Contract Shape IDs;
- supplies non-negative finite `semantic_W_flops`, `semantic_Q_read_bytes`, and
  `semantic_Q_write_bytes` for every Shape; and
- supplies at least one non-negative finite `SOL_time_ms` hardware value for every Shape.

The resulting Roofline is embedded in the immutable Campaign Evaluation Contract before Agent
Problem generalization or any DSL baseline Session. All DSL Lineages therefore share the same
Roofline and Agate evaluates every Kernel against the same sealed theoretical floor.

## Profile fallback and progress output

An NCU Profile failure does not invalidate a correct `eval` latency. Runtime stores the raw Profile
response (including a terminal failure) beside the raw Eval response in the immutable Gateway
Result Artifact. SOL is the duration-weighted mean of each reported Kernel's
`max(compute_sol_pct, mem_sol_pct)` value.

Bootstrap prints whether it used an explicit Roofline, reused a Campaign-sealed Roofline,
generated a new one, or entered Profile fallback. It also prints the resulting baseline SOL path.
This applies to Agent-requested intermediate Evals as well as Runtime's authoritative final Eval;
each paired result is sealed independently. Each completed Attempt prints the authoritative
Kernel's execution result. Operational messages go to stderr, while CLI JSON remains on stdout.

## Configuration

`campaign.roofline_builder` is optional. A production example is:

```json
{
  "repository": "/srv/atrex-bench",
  "commit": "0123456789abcdef0123456789abcdef01234567",
  "git_executable": "/usr/bin/git",
  "python_executable": "/srv/atrex-runtime/.venv/bin/python",
  "fetch_timeout_seconds": 120,
  "execution_timeout_seconds": 120,
  "max_archive_bytes": 268435456,
  "max_output_bytes": 8388608,
  "sku_by_hardware_target": {
    "L20N": "NVIDIA RTX PRO 5000 72GB Blackwell (SM120)"
  }
}
```

The repository is a deployment allowlist, the full commit fixes the cost-model implementation,
and `sku_by_hardware_target` prevents an Agate GPU alias from being mistaken for the converter's
default SKU. Missing metadata, missing operator cost functions, incomplete Shape coverage, an
unknown SKU, or a failed converter selects the NCU Profile fallback and is reported in Bootstrap
output.
