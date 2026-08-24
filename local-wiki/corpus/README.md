# GPU Wiki corpus

`gpu-wiki/` is the knowledge corpus this service serves. It originated as a copy of the `gpu-wiki`
tree from `alibaba/atrex-kernel-agent` and is now ordinary content of this repository, so Local Wiki
examples and integration tests require no second checkout.

Runtime never mutates this directory. Local Wiki copies it into the ignored writable
`../state/gpu-wiki/` store before serving requests, which is what lets the corpus tools record query
feedback without modifying tracked files.

The original Apache-2.0 license and NOTICE are preserved beside this file.
