# 公共示例资源

[English](README.md) | 中文

本目录保存多个可运行示例共同使用的只读输入与通用启动辅助。任何可运行流程都不会再 Source
或执行另一个示例目录。

- `vecadd/` 是唯一标准 VecAdd Fixture：可信 Reference、输入生成器、Shapes、Evaluation
  Contract、Agent Problem、Triton Baseline Seed、Initial Evidence 和直接调用 Agate 的 Candidate。
- `prepare_campaign.py` 把某个示例自己持有的 Runtime Config 与 Campaign 定义物化到该示例的
 生成 Workspace。
- `runtime-common.sh` 提供通用 Secret、前置检查、Bootstrap、Lineage ID 和 Runtime 生命周期
  Helper；调用方必须先声明自己的 Runtime/Campaign 模板。
- `local-secrets.sh` 是生成私有 Runtime 控制 Secret 的唯一实现。

公共文件只是输入，不是独立可运行的示例。
