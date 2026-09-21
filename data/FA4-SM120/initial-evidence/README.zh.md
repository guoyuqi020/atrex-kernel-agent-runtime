# FA4 SM120 Prefill 源码树任务

使用已提供的 FlashAttention-4 CuTe 源码，在 **SM120** 上实现并优化公开生产 Attention
合约。完整源码位于 `work/kernel/`；修改前先阅读其中固定的 `PROVENANCE.json`。

目标合约包括：FP8 E4M3 Query 与分页 KV、BF16 输出、Head Dimension 256、16 个 Query
Head、1 个 KV Head、P64 NHD Page、Ragged Batch、右下对齐的因果 Mask，以及原地写入给定
`out` Tensor。Tensor 布局、Scaling 和 Shape 语义以公开合约为准。不要因为同属 Blackwell
而套用 SM100 的能力假设。

R0 Vendor 在 `vendor/flash_attention/flash_attn/cute/interface.py` 中提供了一条明确的
correctness-first SM120 路径：它把 `seqused_k` 同步到 Host，将 P64 Page 物化成 Dense K/V，
把 Q/K/V 从 FP8 转为 BF16，然后调用 SM120 FA4 的 M64/N64 warp-MMA Kernel。这是可测量的
功能起点，而不是可接受的性能方案。修改前先测量，并在消除这些开销时保持正确性。

SM120 上优先考虑的方向包括：

- 消除 Host 长度同步和 Python 逐请求拼装；
- 在 CuTe Mainloop 中直接读取独立映射的 P64 KV Page；
- 直接消费 FP8 Q/K/V，不物化 BF16 Tensor；
- 保持 `seqused_k`、PackGQA 和因果尾部对齐语义；
- 在 SM120 的 99-KiB 共享内存限制内优化 Tile、Stage、Warp 分工和调度。

SM100 HD256 双 CTA/Tensor-Memory 路径、`tcgen05` 及其 TMA 假设不适用于 SM120。起点相关
文件包括 `interface.py`、`flash_fwd_sm120.py`、`flash_fwd.py`、`paged_kv.py`、
`pack_gqa.py`、`mask.py`、`tile_scheduler.py` 和 `utils.py`。修改只能发生在 Runtime 声明的
可编辑 Vendor 目录；固定 Adapter 与 Quack 支持不可修改，也不能将任务替换为单文件
Fallback 或拉取已有优化实现。

以 Runtime Evaluate/Profile 结果作为证据。Shape 来自同一生产 Callable 在 L20D 上的捕获，
但本题的执行目标与 Roofline 均为 L20N/SM120。本题不包含此前优化得到的 SM120 实现或历史。
