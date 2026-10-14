# Qwen3.5 Flash Attention on L20N

Historical experiment name: GDN/L20N. This is the 90-shape flash-attention task, not the L20D chunk_gated_delta_rule operator in data/GDN.

All runs import the identical evaluated CUDA Bootstrap (kernel SHA256 `3d82b5baaee925405e1f98becd94571a9ca8f2fb4bead454e66ca9c7f2b7a209`, historical geometric mean 4320.617583365619 us). Upstream AKA uses current main, CUDA, 15 iterations and its other defaults. Runtime uses seven independent configurations with three repeats each, five epochs and three attempts per epoch. GPU Wiki is disabled. All GPU work goes to production Agate.

The sixth configuration is Dev-only: no Journals or result reads, visible Kernel source reads and terminal report/discovery remain; other Agent Runtime queries are disabled. Runtime privately validates nominated source.

Raw adds direct Agate access, no Runtime tools or supplied history, and only trajectory-owned files as memory. Private final evaluation and kernel routing remain unchanged. See `docs/configuration.md` for the boundary and budget semantics.
