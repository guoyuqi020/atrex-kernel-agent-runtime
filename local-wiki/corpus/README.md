# GPU Wiki corpus

`gpu-wiki/` is the knowledge corpus this service serves. It originated as a copy of the `gpu-wiki`
tree from `alibaba/atrex-kernel-agent` and is now ordinary content of this repository, so Local Wiki
examples and integration tests require no second checkout.

Synced from `third_party/atrex-kernel-agent/gpu-wiki` at commit
`d2ff1dfcdf60761f0675aa4393ba28f22b1cc049`. All tracked files in that tree are copied
unchanged, including query tools, schemas, records, and mining skills. Upstream's nested
`3rdparty` Git submodules are not vendored; they are not required by the query service.
Local Wiki's HTTP adapter lives outside the copied tree under `src/atrex_local_wiki`.

Operator aliases, component decomposition, retrieval, and bridge behavior belong to the copied
implementation. Do not add a second normalization or ranking algorithm in the HTTP adapter.

Known upstream validation issue at this commit: `schema/kernel/render_template.py --check`
reports `TEMPLATE.md` as stale in both the source checkout and this copy. It is left unchanged
to preserve source parity; the record gates, hardware index check, and retrieval unit tests pass.

Runtime never mutates this directory. Local Wiki copies it into the ignored writable
`../state/gpu-wiki/` store before serving requests, which is what lets the corpus tools record query
feedback without modifying tracked files.

The original Apache-2.0 license and NOTICE are preserved beside this file.
