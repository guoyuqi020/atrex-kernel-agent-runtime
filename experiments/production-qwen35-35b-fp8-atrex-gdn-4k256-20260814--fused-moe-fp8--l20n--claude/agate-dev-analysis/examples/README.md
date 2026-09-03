# Dev 原始样例

这些文件用于阅读和核查，**不要直接执行**。命令可能依赖历史沙箱、远端 GPU、特定 API 和当时的文件内容。原始命令保存为 `command.txt`；本次没有运行这些命令或上传任何文件。

| 样例 | 观察点 | 命令/请求 | 辅助代码 | 实际结果 |
|---|---|---|---|---|
| retained / Triton：候选筛选 | 多规模、CUDA Event 中位计时，候选与 variant 对比 | [命令](retained-triton-benchmark/command.txt) · [请求](retained-triton-benchmark/request.json) | [调用前计时器](retained-triton-benchmark/at-call/tools/bench_moe_latency.py) | [输出](retained-triton-benchmark/stdout.txt) |
| retained / CuteDSL：配对计时 | 交错 A/B；数值不相等仍可正常退出 | [命令](retained-cutedsl-paired-ab/command.txt) · [请求](retained-cutedsl-paired-ab/request.json) | [最终计时器快照](retained-cutedsl-paired-ab/files/tools/paired_ab_probe.py) | [输出](retained-cutedsl-paired-ab/stdout.txt) |
| retained / CUDA：阶段计时 | 关闭模块 Graph replay，观测每阶段 launch | [命令](retained-cuda-stage-timing/command.txt) · [请求](retained-cuda-stage-timing/request.json) | [最终探针快照](retained-cuda-stage-timing/files/scratch/stage_ab_probe.py) | [输出](retained-cuda-stage-timing/stdout.txt) |
| retained / CUDA：压力测试 | 自建 Reference、极端专家路由、异常 ID | [命令](retained-cuda-stress/command.txt) · [请求](retained-cuda-stress/request.json) | [最终探针快照](retained-cuda-stress/files/tools/moe_stress_probe.py) | [输出](retained-cuda-stress/stdout.txt) |
| retained / CUDA：MMA 布局 | 两种 PTX fragment 映射的独立实验 | [命令](retained-cuda-mma/command.txt) · [请求](retained-cuda-mma/request.json) | [调用前探针](retained-cuda-mma/at-call/scratch/mma_proto2.py) | [输出](retained-cuda-mma/stdout.txt) |
| retained / CuteDSL：远端源码检查 | 读取 GPU 镜像实际安装的 MMA API 与示例 | [命令](retained-cutedsl-inspection/command.txt) · [请求](retained-cutedsl-inspection/request.json) | [调用前探针](retained-cutedsl-inspection/at-call/scratch/inspect_mma.py) | [输出](retained-cutedsl-inspection/stdout.txt) |
| AKA / CuteDSL：A/A 与 A/B | 实例化顺序对照、两个阶段消除偏差 | [命令](aka-cutedsl-paired-ab/command.txt) | [调用前计时器](aka-cutedsl-paired-ab/at-call/profiles/episode_13/harness/paired_time_driver.py) | [输出](aka-cutedsl-paired-ab/output.txt) |
| AKA / Triton：NCU 采集 | 远端 CSV 解析、stdout 传输、单位与退出码风险 | [命令](aka-triton-ncu/command.txt) | [最终 wrapper 快照](aka-triton-ncu/files/profiles/episode_15/harness/collect_ncu.sh) | [输出](aka-triton-ncu/output.txt) |
| AKA / Triton：上传失败 | 本地已写探针，包装命令却未带入远端 | [命令](aka-triton-upload-failure/command.txt) | [调用前探针](aka-triton-upload-failure/at-call/profiles/episode_14/harness/mma_roofline_probe.py) | [输出](aka-triton-upload-failure/output.txt) |

每个目录的 `provenance.json` 标注来源与版本性质。`files/` 是工作区最终快照，`at-call/` 是能从 Trace 的字面 Write/Edit 恢复的调用前内容；不可互换。retained 的候选 `kernel.py` 和 `result.json` 单独从该次操作的不可变 Artifact 复制。AKA 框架与远端注入文件未完整归档到这些样例中，不保证独立可运行。
