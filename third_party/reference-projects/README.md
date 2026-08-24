# GPU kernel reference projects

[English](README.md) | [中文](README.zh.md)

Each directory is a pinned Git submodule of an upstream GPU kernel project, kept as optimization
reference material. Runtime never builds, imports, or executes them, and release archives exclude
the whole `third_party/` tree.

| Project | Upstream |
| --- | --- |
| `aiter` | ROCm AI Tensor Engine for ROCm |
| `composable_kernel` | ROCm Composable Kernel |
| `cuLA` | inclusionAI CUDA linear algebra |
| `cute-gemm` | CuTe GEMM examples |
| `cutex` | CUDA template extensions |
| `cutlass` | NVIDIA CUTLASS |
| `DeepGEMM` | DeepSeek DeepGEMM |
| `flash-attention` | Dao-AILab FlashAttention |
| `flashinfer` | FlashInfer LLM serving kernels |
| `FlashMLA` | DeepSeek FlashMLA |
| `FlyDSL` | ROCm FlyDSL |
| `hpc-ops` | Tencent HPC operators |
| `LeetCUDA` | xlite-dev LeetCUDA |
| `quack` | Dao-AILab Quack |
| `tilelang` | TileLang |
| `triton` | Triton language and compiler |

## Checkout

Every entry carries `update = none`, so the documented `git submodule update --init --recursive`
skips all of them and a normal checkout stays small. Name a project and pass `--checkout` to
obtain one:

```bash
git submodule update --init --checkout third_party/reference-projects/cutlass
```

Each pin is older than its upstream branch tip, so clone the full history rather than a shallow
one; a depth-1 fetch cannot reach the recorded commit.
