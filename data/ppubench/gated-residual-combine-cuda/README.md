# Gated residual combine：PPU CUDA 从零实现

目标为 `ZW-M890P` 上的 `gated_residual_combine_bf16`。Claude 使用
`qwen3.8-max`，从只有接口、没有计算实现的 `kernel.py` 开始编写 CUDA C++。
不导入题库中的 `solution.py`，不使用 Triton、CuteDSL 或预制计算内核。

## 文件归属

- 本目录是题目配置与要求的唯一维护入口，纳入 Runtime Git 版本管理。
- `../../../scripts/ppubench/` 保存准备、运行和沙箱部署脚本。
- 运行工作区保存配置快照、固定源码版本、凭据、数据库、会话、日志和结果。
  `~/atrex-runs` 用于运行数据和备份，不保存可维护的题目定义或部署源码包。

准备时会将整个题目目录复制到工作区的 `task-definition/`，并记录输入 SHA-256。
运行只读取工作区内的冻结配置；修改这里的模板只影响新工作区。
准备脚本拒绝非空工作区，也拒绝把运行工作区建在 `data/` 下。

| 文件 | 内容 |
|---|---|
| `task.json` | 题库来源、CUDA DSL、shape 数量、网关及运行环境配置 |
| `agent-problem.json` | 发给模型的算子语义、精度、布局和实现要求 |
| `inputs/initial-evidence/README.md` | 从零写 CUDA 的 Bootstrap 指引 |
| `inputs/empty-kernel/kernel.py` | 只保留 `Model` 接口的空实现 |
| `campaign.template.json` | 目标卡、模型、种子与初始 evidence 路径 |
| `evaluation.template.json` | Native Eval 参数、容差、eager/trusted 与关闭锁频 |
| `policy.json` | 生产门禁、并发、Bootstrap/Optimizer 时限和会话预算 |
| `ablation.json` | 完整冻结的八臂计划，不随全局消融默认值变化 |

原始题目数据仍来自相邻的 `../curated/`。准备时复制 Reference、输入生成器和
shape/metadata 文件，保持其原始字节；上游实现不作为 Bootstrap 起点。
KDA、Core、Evolver、Bench 和 Runtime 的当前 Git HEAD 在准备时记录；Campaign 的
KDA 与 Runtime 的 Evaluator/Evolver 配置固定为这些提交，不使用旧实验的版本。

## 实验配置

- 24 个 shape 全部归 Valid，Test 为 0；报告等权几何平均 latency。
  原始 Roofline 没有已验证的 `SOL_time_ms`，此任务不计算 SOL%。
- 保留 BF16 乘法的中间舍入，再进行 BF16 加法；支持输入 view 的非零 storage offset，
  不得修改输入。
- Bootstrap 最多 8 小时，成功后自动启动全部八臂；失败时不启动后续优化。
- 轮末共享与实时共享，各包含全开、全关、仅 Experiment、仅 Direction 四种组合。
- 每臂 3 条 trajectory、5 个 epoch；每条每轮 3 次 attempt，共 15 次，
  每臂最多并发 3 次。臂间不共享 Bootstrap 之后的历史，不运行 Evolver。
- 生产门禁开启，PPU 禁用锁频；普通 Evaluate 和 ABBA 均使用原生 Eval。
- GPU Wiki 默认开启，Bootstrap 与全部八臂均可查询，与 Direction/Experiment 开关独立。
  自然语言检索需要模型解析，单次查询等待上限为 600 秒。
  服务地址由 `policy.json` 的 `runtime.wiki_url` 指定；`task.json` 中设置
  `runtime.gpu_wiki.enabled: false`（或 `runtime.gpu_wiki: null`）可关闭。

## 准备与运行

在已配置 Runtime Python 环境的仓库根目录执行：

```bash
python scripts/ppubench/campaign.py prepare \
  --workspace workspaces/residual-ppu-cuda-trial \
  --creation-key residual-ppu-cuda-trial
```

准备操作只生成并校验本地文件，不启动模型或 GPU 作业。`--port` 可覆盖模板中的
8771；默认 creation key 为工作区目录名，新实验应使用不同工作区与 key。

Runtime 及模型凭据应由运行环境提供。先按 [内部 Wiki 部署说明](../../../local-wiki/README.md#internal-indexed-corpus)
导入语料并启动知识服务；默认配置为 `local-wiki/configs/internal.example.json`，监听
`http://127.0.0.1:8091`。知识服务单独运行，下面的命令及沙箱 `launch` 不会自动启动它。
再启动 Runtime，并在另一个进程运行：

```bash
atrex-kernel-agent-runtime serve --config workspaces/residual-ppu-cuda-trial/runtime.json
python scripts/ppubench/campaign.py run --workspace workspaces/residual-ppu-cuda-trial
```

两个进程需要相同的 Runtime 签名/管理凭据及 Agate、Claude 环境。已配置好的沙箱
可使用 `scripts/ppubench/sandbox_ops.py launch --record ... --workspace ...` 完成准备、
凭据加载、健康检查及后台启动。日志和启动回执写入指定 `--record`，外层时限为 72 小时。
沙箱部署脚本不包含密钥；只读取沙箱中已经配置的凭据。
