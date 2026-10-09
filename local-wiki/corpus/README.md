# GPU Wiki corpus

`gpu-wiki/` is the knowledge corpus this service serves. It originated as a copy of the `gpu-wiki`
tree from `alibaba/atrex-kernel-agent` and is now ordinary content of this repository, so Local Wiki
examples and integration tests require no second checkout.

Synced from [`alibaba/atrex-kernel-agent` main](https://github.com/alibaba/atrex-kernel-agent/tree/3d27c1eb1d75f390df29e63aa93ddbeecd928e3f/gpu-wiki)
at commit `3d27c1eb1d75f390df29e63aa93ddbeecd928e3f` (main verified on 2026-10-09).
All tracked files in that tree are copied
unchanged, including query tools, schemas, records, and mining skills. Upstream's nested
`3rdparty` Git submodules are not vendored; they are not required by the query service.
Local Wiki's HTTP adapter lives outside the copied tree under `src/atrex_local_wiki`.

Operator aliases, component decomposition, retrieval, and bridge behavior belong to the copied
implementation. Do not add a second normalization or ranking algorithm in the HTTP adapter.

Upstream now includes optional query evidence through `tools/wiki_trace.py` and bridge process
registration during an AKA restart handoff. The standalone Runtime adapter uses the direct
`query_nl.py` interface; AKA's plugin launcher and orchestrator are outside this copy boundary.
Ordinary bridge queries do not require them. Do not set AKA's restart-handoff environment variables
for this standalone service; that mode intentionally requires the owning AKA orchestrator.

Known upstream validation issue at this commit: `schema/kernel/render_template.py --check`
reports `TEMPLATE.md` as stale in both the source checkout and this copy. It is left unchanged
to preserve source parity; the record gates, hardware index check, and retrieval unit tests pass.

Runtime never mutates this directory. Local Wiki copies it into the ignored writable
`../state/gpu-wiki/` store before serving requests, which is what lets the corpus tools record query
feedback without modifying tracked files.

The original Apache-2.0 license and NOTICE are preserved beside this file.

## Optional internal source

The separately imported, Git-ignored `internal_gpu_wiki/` snapshot comes from
[`tre-infra-open/atrex-kernel-agent`, `codex/ppu15-agent-wiki`](https://code.alibaba-inc.com/tre-infra-open/atrex-kernel-agent/tree/codex/ppu15-agent-wiki/internal_source/gpu-wiki):

- Commit: `2076c865cc6618d810cfe2bf09b4fc395536693e` (verified 2026-10-09).
- Subtree: `internal_source/gpu-wiki`.
- Tree object: `5d17eedd45eb0e1d9f419e9b55e80b70503654f8`.
- Copy boundary: the complete Git subtree, unchanged, including native tools, source records,
  search shards and governance history. This is a separate internal source; the public corpus's
  license/provenance above does not describe its contents.

Use `configs/internal.example.json` to select it directly. Its `search_index` retrieval layout is
not compatible with the public corpus's legacy sibling-store discovery. No internal records are
vendored into this Runtime Git repository. An authorized checkout/import is required on each host.

The PPU FP8 prefill additions are historical mechanism summaries for ZW-M890P / ppu0015 and the
specified SDK, with evidence qualifications. They are not a complete reproducible GPU experiment
package or proof that a technique improves a different operator such as gated residual combine.
