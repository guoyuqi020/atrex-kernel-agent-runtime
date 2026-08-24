# GPU Kernel 参考项目

[English](README.md) | 中文

每个目录都是一个上游 GPU Kernel 项目的固定版本 Git Submodule，作为优化参考资料保存。Runtime
不构建、不导入、也不执行它们，发布归档同样排除整个 `third_party/` 目录。

| 项目 | 上游 |
| --- | --- |
| `aiter` | ROCm AI Tensor Engine for ROCm |
| `composable_kernel` | ROCm Composable Kernel |
| `cuLA` | inclusionAI CUDA 线性代数 |
| `cute-gemm` | CuTe GEMM 样例 |
| `cutex` | CUDA Template Extensions |
| `cutlass` | NVIDIA CUTLASS |
| `DeepGEMM` | DeepSeek DeepGEMM |
| `flash-attention` | Dao-AILab FlashAttention |
| `flashinfer` | FlashInfer LLM 推理 Kernel |
| `FlashMLA` | DeepSeek FlashMLA |
| `FlyDSL` | ROCm FlyDSL |
| `hpc-ops` | 腾讯 HPC 算子 |
| `LeetCUDA` | xlite-dev LeetCUDA |
| `quack` | Dao-AILab Quack |
| `tilelang` | TileLang |
| `triton` | Triton 语言与编译器 |

## 检出

每个条目都带 `update = none`，因此文档中的 `git submodule update --init --recursive` 会跳过它们，
常规检出保持精简。指定项目并加上 `--checkout` 即可获取其中一个：

```bash
git submodule update --init --checkout third_party/reference-projects/cutlass
```

每个固定 Commit 都早于上游分支最新提交，因此需要克隆完整历史而不是浅克隆；深度为 1 的 Fetch
无法到达记录的 Commit。
