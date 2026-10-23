# Fresh CUDA bootstrap for dual_cache_latent_bf16

Implement the public operator contract from scratch in CUDA C++ on NVIDIA L20D (SM103). The supplied kernel.py is only an API placeholder. Do not import the curated solution or use Triton/CuteDSL/prebuilt compute operators. All ten evaluator shapes are Valid; Test is empty. After Bootstrap, Runtime starts only the three-trajectory real-time shared broadcast workflow with Direction and Experiment journals enabled.
