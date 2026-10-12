# CSA paged causal：PPU CUDA 从零实现

目标为 `compressed_sparse_attention_paged_causal_bf16_interleaved_kv_append`，硬件 `ZW-M890P`。
从空 Model.forward 开始，由 Qwen 3.8 Max 编写 CUDA C++；不使用上游私有 vLLM 适配器作为种子。

- 25 个 shape 全部 Valid，Test 为 0；等权几何平均 latency，无已验证 SOL%。
- 保留上游 reference、shape 和 metadata；输入使用相邻 agate-inputs 的 8,012 字节精简版，按 SHA-256 校验并冻结。
- 保留 output 返回别名和 kv_cache/output 原地修改契约，校验全部声明状态。精度为 atol=0.01、rtol=0.05，relative L2 <=0.001。
- Bootstrap 限时 8 小时；成功后自动启动八臂：轮末/实时共享 × 全开/全关/仅 Experiment/仅 Direction。
- 每臂三条 trajectory、五个 epoch，每条每轮三次 attempt；每臂最多三次并发。无 Evolver。
- GPU Wiki 可选开启，查询使用 Agent 原始问题。使用正式 Agate、原生 Eval/ABBA，关闭锁频。
- Runtime/Campaign 外层时限 72 小时，单独创建新工作区。

准备时指定 `--task-dir data/ppubench/csa-paged-cuda`，其他准备及启动方式见相邻
[gated-residual-combine-cuda 文档](../gated-residual-combine-cuda/README.md)。
