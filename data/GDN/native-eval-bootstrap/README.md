# GDN / L20D independent comparison

Uses the original GDN-clean Bootstrap (2026-09-09), with native Eval compatibility changes only. The CuTe helper retains the exact cached get_num_sm implementation and removes unused code. The 203037-byte kernel file is reformatted to 127617 bytes, removing comments and docstrings; executable AST is unchanged. These changes do not alter the kernel algorithm, dispatch or JIT.

All 10 original evaluation shapes passed production native Eval ev_3b1cd5669f60; common baseline is 233.07089396164332 us geometric mean. No post-Bootstrap optimization records are imported.

Three upstream AKA repeats use CUDA and 15 iterations. Runtime retains CuteDSL, seven independent configurations with three repeats each, five epochs with three attempts per epoch, 100M session tokens and 8 hours active execution time. All use Qwen 3.8 Max. Wiki is disabled. Remote native Eval has a 128 KiB limit per Python source file.

`evaluation-contract.json` preserves the historical Bootstrap contract for reference. The deployed contract uses the compatible source revision in `task/source_manifest.json`, requirements `torch>=2.9.0` and `nvidia-cutlass-dsl==4.6.1`, and is sealed as `sha256:8aa0a8a16ee8a0182e760b420cf1145c534b8de85c0810cc76a3a3ea7eb7c246`. Full deployment receipts live in `~/atrex-runs/abla/20261013-gdn-l20d-four-sandboxes/summary.json`.

The sixth configuration is Dev-only: no Journals or result reads, visible Kernel source reads and terminal report/discovery remain; other Agent Runtime queries are disabled. Runtime privately validates nominated source.

Raw adds direct Agate access, no Runtime tools or supplied history, and only trajectory-owned files as memory. Private final evaluation and kernel routing remain unchanged. See `docs/configuration.md` for the boundary and budget semantics.
