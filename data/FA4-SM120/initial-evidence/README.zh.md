# FA4 SM120 实现任务

在 **SM120** 上实现并优化公开的生产 Attention 合约。固定的
`work/kernel/kernel.py` 保留生产 ABI，并调用由你维护的
`work/kernel/implementation/`。

目标包括 FP8 E4M3 Query 与分页 KV、BF16 输出、Head Dimension 256、16 个 Query Head、
1 个 KV Head、生产边界上的 P64 HND Page、Ragged Batch、右下对齐因果 Mask，以及原地写入
给定 `out` Tensor。布局、Scaling 和 Shape 语义以公开合约为准。

这是一个要求**原生 FP8 计算路径**的 FA4 任务，而不只是兼容 FP8 输入。主要的 QK 和
Probability-V 矩阵乘必须使用 SM120 FP8 Tensor Core MMA，或等价的原生 FP8 MMA 数据路径。
如果先把完整的 Query、Key 或 Value Tensor 转换成 BF16/FP32，再运行通用的高精度 Attention，
则不符合本题预期。Softmax、Scaling、归约以及为了数值正确性所需的累加可以使用更高精度，
但主要矩阵乘数据路径必须保持 FP8。

`work/kernel/reference_sm103/` 是原 SM103-family 题目实现的只读副本，用于参考 HD256、
分页 KV、PackGQA、Mask、调度和启动设计。重点文件包括 `flash_fwd_sm100.py`、
`sm100_hd256_2cta_fmha_forward.py`、`paged_kv.py`、`pack_gqa.py`、
`mask.py`、`tile_scheduler.py` 和 `utils.py`。SM103 的 Tensor Memory、`tcgen05`、TMA、
Cluster 和双 CTA 假设只能作为设计参考，不能直接套用到 SM120。

Reference 不是 Candidate 的运行时依赖，也不可修改。不要从 Candidate 中导入它；应当把
必要思路移植或重新设计到 `implementation/`，形成自写的 SM120 CuTe 算子。可以自由新增、
替换或删除 `implementation/` 下的文件，但不能修改固定 Adapter、Reference 或 Provenance。

初始实现会主动抛出 `NotImplementedError`。Bootstrap 必须先写出并测得第一个正确的 SM120
实现，Runtime 才能注册 `v0`。以 Runtime Evaluate/Profile/Check 结果为证据；SM120 源码和
混合架构调度入口已明确剔除。本题不包含已有 SM120 实现、优化结果、Journal 或 Conversation
历史。
