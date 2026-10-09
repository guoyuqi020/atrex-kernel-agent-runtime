# Fresh CUDA bootstrap

Write the CUDA C++ implementation from scratch. The supplied kernel.py is an API placeholder and has no computational implementation. Do not use Triton, CuteDSL, or prebuilt compute kernels. PyTorch reference is the evaluator oracle, not an optimization seed.

The fixed task is gated_residual_combine_bf16 on Alibaba PPU ZW-M890P. Read the public operator contract for precise BF16 rounding, layout, and input immutability requirements. The bootstrap is limited to 8 hours. All 24 private evaluator cases belong to Valid; Test is empty.

Use a self-contained kernel.py containing authored __global__ CUDA source with an approved loader such as torch.utils.cpp_extension.load_inline; Runtime enforces the CUDA-only policy. Query Runtime env and use dev for compiler/toolchain diagnostics on the actual PPU before assuming NVIDIA-specific instructions, flags, clock controls, or profilers. CUDA-compatible architecture labels do not imply NVIDIA hardware. Native Eval is required for normal evaluations and ABBA comparisons.

No prior kernel or optimization history is imported. After a correct bootstrap, Runtime automatically seeds eight independent ablation arms: epoch-shared or realtime shared, crossed with both journals, neither journal, experiments only, or directions only. Each arm has three trajectories and five epochs.
