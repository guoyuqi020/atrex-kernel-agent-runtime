# Vendored GPU Wiki corpus

`gpu-wiki/` is an unmodified copy of the `gpu-wiki` tree from
`alibaba/atrex-kernel-agent`. Its exact repository, commit, and Git tree are recorded in
`../reference.lock.json`.

The source is vendored so Local Wiki examples and integration tests require no second repository
checkout. Runtime never mutates this directory; Local Wiki copies it into the ignored writable
`../state/gpu-wiki/` store before serving requests.

The upstream Apache-2.0 license and NOTICE are preserved beside this file.
