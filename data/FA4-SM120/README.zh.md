# FA4 SM120 源码树优化任务

[English](README.md) | 中文

这是 `data/FA4` 的 SM120 对应题目。它保持相同的生产
`flashinfer.prefill.trtllm_batch_context_with_kv_cache` ABI、Reference、输入生成器和 30 个
私有 Shape，但目标硬件改为 **L20N / SM120**，Vendor 起点也改为 SM120 FA4 路径。被优化的
不是已安装的 FlashInfer 实现。

## 初始实现

固定 Adapter 接收 FP8 E4M3 Query 和 P64 分页 KV，输出 BF16，并保持 16 个 Query Head、
1 个 KV Head、Head Dim 256、Ragged Length、PackGQA 和因果语义。

上游 FlashAttention Commit `b54df166ebb69b896892826014759d09b9c3c9c6` 在 SM120 上只提供
FP16/BF16 Dense Kernel。因此封装后的 Source Commit
`b6bfe3d177aab2b930f4d6485227002b65cbb2de` 在可编辑 Vendor 内加入 correctness-first 桥接：

1. 将 `seqused_k` 同步到 Host；
2. 把独立映射的 P64 Page 拼成 Dense K/V；
3. 将 Q/K/V 从 FP8 转成 BF16；
4. 调用上游 SM120 M64/N64 FA4 Kernel。

这提供了一条明确、可测量的 R0。优化目标是消除上述开销，形成 SM120 原生 FP8 paged-KV
路径。SM100 HD256 双 CTA、Tensor Memory、`tcgen05` 及其 TMA 路线不能作为 SM120 假设。

只有 `vendor/flash_attention/flash_attn/cute/` 可编辑；Adapter、Quack、Reference 和合约固定。
详细优化说明见[初始 Evidence](initial-evidence/README.zh.md)，与原题的关系见
[对齐核对](ALIGNMENT.zh.md)。

## 评测

- 硬件目标为 `L20N`，Agent 可见架构为 `sm_120`。
- 返回值和被修改的 `out` 均逐元素检查：
  `abs(candidate-reference) <= 0.06 + 0.04 * abs(reference)`。Runtime 会把这套精确策略注入
  每个 Bootstrap 和 Optimizer Session。
- Bootstrap 先测 1 Case，再测 5 Cases；Optimizer Evaluate 使用 5 Cases 和 100ms Bench 预算。
- Retention 与 Agent Promotion 使用同 Allocation ABBA；每批一个 Shape，最多并发 16 批。
- Roofline 按 L20N SM120 的 549 TFLOP/s FP8 与 1344 GB/s HBM 峰值重新计算。
- Production 静态源码 Gate 仍关闭，因为它会扫描完整 CuTe 库；正确性、源码锁、可编辑范围和
  Runtime 比较仍然生效。

Shape 数值来自同一生产 Callable 在 L20D 上的捕获，用于定义 Workload，而不是声明 SM120
性能。Metadata 中逐 Shape 的 L20D 历史测量保留为来源证据；任务目标和 Roofline 明确为
L20N/SM120。

## 准备与运行

在已安装 Runtime、模型 CLI 和 bwrap 的 Linux 环境中运行，并先导出 Agate 凭据。

```bash
cd ~/atrex-runtime
source env.sh
python3 scripts/source-tree/task.py prepare \
  --inputs data/FA4-SM120 \
  --workspace workspaces/FA4-SM120 \
  --backend claude
```

准备过程离线执行，不启动服务、模型或 GPU Job。生成的 Runtime 默认端口为 8771。可以选择
直接运行上游 SM120 BF16 冒烟或目标 Shape 冒烟：

```bash
python3 scripts/source-tree/task.py smoke --workspace workspaces/FA4-SM120 \
  --smoke-mode sm120-bf16
python3 scripts/source-tree/task.py smoke --workspace workspaces/FA4-SM120 \
  --smoke-mode target --shape-id 0
```

分别在两个终端运行 Runtime 和 Campaign：

```bash
python3 scripts/source-tree/task.py serve --workspace workspaces/FA4-SM120
```

```bash
python3 scripts/source-tree/task.py bootstrap --workspace workspaces/FA4-SM120
python3 scripts/source-tree/task.py campaign --workspace workspaces/FA4-SM120 --target-epoch 1
python3 scripts/source-tree/task.py inspect --workspace workspaces/FA4-SM120
```

也可以运行标准消融入口：

```bash
python3 scripts/source-tree/task.py ablation --workspace workspaces/FA4-SM120
```

所有运行数据写入 `workspaces/FA4-SM120`，不会修改 `data/FA4-SM120` 中的输入。
