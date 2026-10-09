# PPU Benchmark

Qwen3.8 Flash Next PPU TP2 C16 题库，目标卡为 `ZW-M890P`，包含 **18 个算子、460 个 case**。

来源：[Atrex Bench CR 30328999](https://code.alibaba-inc.com/tre-infra/atrex-bench/codereview/30328999)，分支 `codex/qwen38flash-ppu-0-64k-bench`，固定提交 `0e1413ea89367be7d38c059e458f23f5a7c33f81`。

## 目录

保留上游 Operator／Collection 目录结构及源文件原始字节：

```text
ppubench/
├── curated/
│   ├── operators/<operator_id>/reference.py
│   └── collections/qwen38flash_next_ppu_tp2_c16_0_64k_20260928/
│       ├── collection.json
│       └── operators/<operator_id>/
│           ├── input.py
│           ├── solution.py
│           ├── shape_cases.json
│           ├── shape_range.json
│           ├── metadata.json
│           └── roofline.json
└── source-provenance.json
```

`source-provenance.json` 记录来源提交、上游文件路径、大小和逐文件 SHA-256。
`solution.py` 是上游交付实现；输入校准数据、全部 case、状态修改契约及历史验收信息一并保留。

## 接入评测

此目录是原始题库快照。标准 Eval bundle 需要将共享 `reference.py` 与对应 Collection 的 `input.py`、`solution.py`、`metadata.json` 放在同一算子目录，并将 `shape_cases.json` 投影为 `shapes.json`。
校验和导出时使用 `data/ppubench/curated` 作为 `--curated-root`；说明文档和来源清单放在其外层，以满足上游对发布目录的严格校验。
上游配套转换器位于 [CR 30003525](https://code.alibaba-inc.com/tre-infra/atrex-bench/codereview/30003525)，本次核验版本为 `b01dae5d1132e6720bf8be2e6b10675bb0493b9b`；固定公共 harness 提交为 `22d6a57b98ebb45c8db76e98815158d820c1dca3`。

接入时保留 metadata 中的状态与输出契约，并显式采用题目记录的正确性配置。PPU 评测关闭锁频；三个 TP collective 题目需要双 rank。部分输入生成器的大小、原始适配器的导入及私有运行依赖仍需与目标网关适配。
