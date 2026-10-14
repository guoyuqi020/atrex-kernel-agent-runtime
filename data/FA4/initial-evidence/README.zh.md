# FA4 Prefill 源码树任务

使用已提供的原始 FlashAttention CuTe 源码，实现并优化公开生产 Attention 合约。完整源码已位于 `work/kernel/`，固定的 `PROVENANCE.json` 记录了源版本和起点的能力缺口。

起始路线为 HD256、双 CTA、M128/N128、FP8 分页 KV。上游支持一个 P128 Page，目标使用独立映射的 P64 Page。需要扩展 N128 路径以组合两个 P64 Page，尊重真实 KV 长度 `seqused_k`，并支持专用 PackGQA。Tensor 布局、Scaling、Ragged 序列、因果 Mask 和 `out` 修改语义以公开合约为准。

相关文件位于 `vendor/flash_attention/flash_attn/cute/`，包括 `sm100_hd256_2cta_fmha_forward.py`、`interface.py`、`mask.py`、`tile_scheduler.py`、`paged_kv.py`、`pack_gqa.py`、`flash_fwd_sm100.py` 和 `utils.py`。修改只能发生在 Runtime 指定的可编辑目录；适配器和 Quack 支持固定，不允许在适配器中绕过能力缺口、改为单文件实现或引入已优化的外部版本。

原始 R0 在目标合约上预期失败。Bootstrap 需要修复并提交正确的测量候选，之后 Runtime 才会注册 `v0`。原来的 P128 冒烟验证仅证明原始能力可以编译执行，其 L2 阈值不是目标正确性策略。通过 Runtime 工具进行目标评测与 Profile，不要从冒烟结果推断验收成功。这里不包含以往优化经验或 C05/Increment 优化源码。

起始源码包含原生 Eval 兼容补丁，不包含旧运行的优化实现。使用 `vendor.flash_attention.flash_attn` /
`vendor_support.quack` 静态导入，不添加 `sys.path` 或动态导入。上传的每个 Python 文件都会校验，包括
未使用的模块；`source_archive_invalid` 要求修改源码，重复提交相同归档不能解决问题。
新增 ParamsBase 字段时，按完整 dataclass 字段顺序更新 `__quack_field_items__`；新增 NamedTuple
参数通过 `register_namedtuple_type` / `mlir_namedtuple` 注册。可选调度配置位于 `fa4_config`。
