# Token 分析：用量复核与文本估算

本目录保存 [实验报告](../report.md) 的离线分析方法与复核数据。[Provider 用量复核](#provider-用量复核) 包含总账校正和 pooled/evolve 横向统计；下述 Bash 分析覆盖从同一 Bootstrap DSL seed 出发的 114 个 AKA 主 Optimizer Episode、120 个 retained Optimizer Attempt，共 13,745 / 6,922 次调用，不含 Bootstrap 或独立 Reviewer/子 Agent。

## 文件

- `provider-usage-audit.json`：237 份 AKA Trace 复核所得精确用量、来源摘要及按实验时价格快照重算的费用；包含分项未知的 Policy Review 说明。
- `bash_text_tokens.py`：只读取 Trace、做分词和汇总，不执行任何 Trace 命令。
- `bash-action-index.json.gz`：冻结此前 Shell/heredoc/Python AST 审核得到的动作标签；按 Session 保存相对 Trace 路径、Trace SHA-256、调用行号、命令 SHA-256、字符数和类别。它不包含命令或返回原文。
- `bash-text-token-summary.json`：分组、DSL、Session 的统计；分类汇总允许重叠。记录 tokenizer 版本、统计范围和索引摘要。
- `bash_context_tokens.py`：按可见请求与压缩保留列表，估算 Bash 原始文本的生成与累计读取；复用同一冻结分类索引。
- `bash-context-token-summary.json`：累计读取的组别、类别、DSL、Session 汇总，以及压缩/请求覆盖诊断和两套词表、三种保留规则对照。
- `test_bash_text_tokens.py` / `test_bash_context_tokens.py`：去重、流式请求合并、命令生成/返回读取、压缩保留及子链隔离的单元测试。
- `shell_read_audit.py` / `shell-read-audit.json`：细分原 `shell_read_or_filter` 粗标签，区分写入式匹配、接口执行后过滤、帮助、具体文件对象；保存互斥汇总与每类前五条高权重调用的引用。
- `test_shell_read_audit.py`：检查 heredoc 写入、原地编辑、GPU 输出过滤与文件读取的区分。
- `bash_adjacent_thinking.py` / `bash-adjacent-thinking-summary.json`：提取 Bash 相邻思考，分别保存生成量、假设上下文累计读取、前后共享去重、与非 Bash 工具混合关联及互斥用途分配；每个 Session 附三条最高累计权重片段的位置，不复制思考原文。
- `test_bash_adjacent_thinking.py`：检查前后共享、批量工具调用、流式去重、说明文字穿插、非 Bash 工具/人工消息/压缩边界以及子链隔离。

索引冻结的是本轮已有分类，不是通用自动分类器。报告已说明静态识别的限制；该脚本不会重新解释或执行动态 Shell 内容。

## 复现

使用 Python 3.11 或更新版本，另建临时环境即可，不需要修改 Runtime 依赖。以下命令在本目录执行：

```bash
TOKENIZER_ENV="$(mktemp -d /tmp/atrex-tokenizer.XXXXXX)"
python3 -m venv "$TOKENIZER_ENV/venv"
"$TOKENIZER_ENV/venv/bin/python" -m pip install 'tiktoken==0.12.0'
export TIKTOKEN_CACHE_DIR="$TOKENIZER_ENV/cache"

# 这一步仅下载公开词表，不读取 Trace。
"$TOKENIZER_ENV/venv/bin/python" -c 'import tiktoken; [tiktoken.get_encoding(n) for n in ("o200k_base", "cl100k_base")]'

# 根据归档实际解压目录修改此变量。此后的分词只在本地执行。
ARCHIVE_ROOT="$HOME/atrex-runs/production-qwen35-35b-fp8-atrex-gdn-4k256-20260814--fused-moe-fp8--l20n--claude"
"$TOKENIZER_ENV/venv/bin/python" bash_text_tokens.py \
  --archive-root "$ARCHIVE_ROOT" \
  --index bash-action-index.json.gz \
  --output "$TOKENIZER_ENV/bash-text-token-summary.json"

# 累计读取口径；只读归档，不需要 Provider API 或完整逐响应 usage。
"$TOKENIZER_ENV/venv/bin/python" bash_context_tokens.py \
  --archive-root "$ARCHIVE_ROOT" \
  --index bash-action-index.json.gz \
  --output "$TOKENIZER_ENV/bash-context-token-summary.json"

"$TOKENIZER_ENV/venv/bin/python" shell_read_audit.py \
  --archive-root "$ARCHIVE_ROOT" \
  --index bash-action-index.json.gz \
  --output "$TOKENIZER_ENV/shell-read-audit.json"

"$TOKENIZER_ENV/venv/bin/python" bash_adjacent_thinking.py \
  --archive-root "$ARCHIVE_ROOT" \
  --index bash-action-index.json.gz \
  --output "$TOKENIZER_ENV/bash-adjacent-thinking-summary.json"

"$TOKENIZER_ENV/venv/bin/python" -m unittest discover -p 'test_*.py'
```

脚本会检查 Trace 和命令摘要、Bash 调用数、命令及返回字符数。归档有改动或匹配失败时直接报错，不静默改用不同样本。输出可与本目录的汇总 JSON 比较，统计不依赖临时分析工作区。

## 每条文本仅计一次

使用 [tiktoken](https://github.com/openai/tiktoken) 的 `encode_ordinary`，对命令和各份返回分别分词再求和。主词表为 `o200k_base`，对照词表为 `cl100k_base`；二者都不是 Claude 官方计费 tokenizer。命令取 `input.command` 原文，返回按 `tool_use_id` 取可见 `tool_result` 文本；文本块沿用此前字符审核的换行拼接规则。重复工具调用 ID、相同结果块不重复计数，不同 ID 的重复执行分别计入。

每个动作标签关联的是整条 Bash 调用文本，例如 `gateway-execute … | tail` 同时进入 GPU 入口和文本过滤两类。不要相加分类行；`total` 是独立去重总量。缺少返回只意味着本次 Trace 未观察到返回。后续 `TaskOutput`、其他工具的读取不追溯到原始 Bash，后续 Bash 读取归自己的调用。

本估算不含 Prompt、thinking、消息包装、工具 Schema、图片以及历史上下文的重复读取和缓存计费，不代表实际消耗或删除某个动作可节省的净用量。报告保留独立 Provider 总账，不再使用此前的 Session 均摊估计。

## 累计读取：主估算

本口径使用相同命令/返回文本，但不止计算一次。按 Trace 文件顺序，先对每个新出现的 `(parent_tool_use_id 所属链, assistant message.id)` 计算历史文本读取，再加入本次响应里的命令；工具返回在出现后才进入对应链的上下文。流式块不是新请求；命令生成计 output 一次，工具返回不计 output，只计后续 input。

每个文本片段保存事件 UUID、所属调用、链和首次出现行号。在 `compact_boundary` 处读取 `preservedMessages.allUuids`（AKA）或 `preserved_messages.all_uuids`（retained），只保留列表中的原始片段。缺少保留元数据就报错，不静默当作没有发生压缩。摘要无法逐句关联回 Bash，故不归入原动作。两次压缩之间假设原文保持不变；日志不可见的裁剪、替换或请求重试不能精确还原。

主估算 `preserved_messages_total = command_once + command_input + result_input`。其中 `*_input` 为文本分词数乘以它在所属链的后续可见请求中保留的次数；命令与返回分别追踪。JSON 字段前缀是词表名与场景名，`generated` 仅包含命令首次生成。`visible_once` 与原单次文本统计逐类别一致；不要将一次 `result_once` 额外加到累计账上。`child_total` 是 `total` 的子集，不可再加一次。

场景对照如下：

- `preserved_messages`：主估算，使用明确的原文保留列表。
- `reset_at_compaction`：每次压缩后清空全部旧 Bash 原文，用于量化保留列表的影响。
- `ignore_compaction`：故意忽略压缩、保留到链结束，展示直接乘剩余请求数的问题，不作为主估算。
- `optional_single_compaction_pass_input`：额外假设每个压缩边界发生一次读取压缩前 Bash 上下文的摘要请求；没有将它当成已观察到的请求，也不加入主估算。

这不是全文请求的精确重放，场景差异也不是置信区间或实际账单的上下界。主估算仍排除 Prompt、thinking、解释、其他工具、工具 Schema、摘要、隐藏请求和独立子 Agent/Reviewer 归档。归档主 Trace 内可见的子链按 `parent_tool_use_id` 单独重放，子链响应不读取主链文本；未记录的父上下文继承不推测。裁剪通知内展示的返回文本按实际可见片段计数，不打开通知中指向的大文件充当原始返回。

不按 Session 平均用量分摊，也不把终态总量按文本比例强行分配。AKA 的完整逐响应 input usage 只用于检查 Bash 子集估算是否超过同一请求的总 input；17,891 次检查没有发现超过的情况，但这不证明词表/上下文复原完全精确。Retained 仅 3 个响应具有完整 input 分类字段，因此不能用它的流式 input 值缩放或截断估算。各片段的缓存命中范围不可见，不划分 cache read/cache write/uncached input，不据此估算货币费用或删除操作的净节省。

## Shell 文本命令细查

原 `shell_read_or_filter` 标签保留以便与此前数据对账，但它按工具名称命中，包含 `cat > file <<EOF` 和 `sed -i`，不是严格的纯读取标签。`shell_read_audit.py` 用 `classify()` 中明确的优先级对这一子集做整条调用的互斥分类；先移除 heredoc 正文，区分写入式匹配，再识别帮助、接口调用、文件路径用途。分类是启发式，不解释任意动态 Shell，也不把一个复合调用拆成独立计费动作。零值不代表该系统完全不做这种工作，只表示该互斥子类中没有匹配。

`exact_repeats_within_session` 以 Session、原始命令摘要、完整返回文本列表摘要分别检查额外重复调用；不忽略空格，不合并不同 Session，不推断语义等价。高权重调用引用含路径、行号与摘要，但不复制命令/返回原文。若要抽查原文，可显式给 `--raw-output` 指定临时 JSON 路径；该文件含本地原始内容，不需上传。用途拆分总量应与 `bash-context-token-summary.json` 的 `shell_read_or_filter` 分类完全一致。

## Bash 相邻 thinking

`bash_adjacent_thinking.py` 只提取显式 `thinking` 块内的 `thinking` 字符串，不计算 signature、普通解释、隐去的推理或工具返回中引用的历史思考。前置关联限定在同一个 assistant `message.id` 的工具批次，后置关联限定在当前思考前紧邻的结果批次；该批次的 tool_use/tool_result 即使交错写入，也按原调用的 message ID 识别。普通 assistant text 和非对话元数据可以穿插，但人工消息、压缩边界和其他动作不会被越过。结果跟随原工具调用所属链。

同一 `(chain, message.id, canonical block)` 只保留第一次出现。当前归档中每个响应恰有一块可见 thinking，没有重复或前缀累积块。相邻两条 Bash 可以共享同一段 thinking，取关联调用 ID 的并集后只计一次。用途分配时除以不同关联 Bash ID 的数量，跨类别的分数份额可以相加；`blocks` 在类别下因此可能是小数，而总表始终为整数。`relations` 中 before / after / both 是互斥分组，`mixed_non_bash` 是同时邻接非 Bash 的子集，不能重复相加。

`generated` 是可见思考分词数，仅生成一次。`preserved_messages_input` 是假设原文被后续请求保留时的累计读取，`preserved_messages_total` 为两者之和；重用 Bash 的请求去重与 UUID 压缩保留规则。**思考可能在未显示的请求构造过程中被剥离；原始请求 payload 不可见，因此不能验证实际回传量。** 生成-only 与保留场景分别展示，不把后者说成精确账单、置信区间或净节省。不额外推测压缩生成调用。另提供 reset-at-compaction 和 ignore-compaction 对照场景。

原有 Bash-only JSON 和命令/返回口径不变，新增数据是独立的相邻关联项。一个 thinking 还可能关联 Read/Edit 等工具，相关占比单列，不能再与其他工具的相邻思考总量直接相加。可用 `--raw-output /tmp/adjacent-thinking-details.json` 输出所有片段到调用的索引（行号、UUID、调用 ID 和计数，不含思考原文），以便按原始 Trace 抽查。

## Shell 案例证据索引

对应 [主报告的代表性 Shell 案例](../report.md#代表性-shell-案例)。两侧在各自 DSL 内从同一 Bootstrap seed 开始；AKA 为两次运行合计，retained 为双 Trajectory，不含 Bootstrap。下表仅含单条命令及返回，不含相邻思考；k 为千 Token，“可见”为一次分词量，“累计”包括后续上下文读取，均为估算而非精确计费。未单列单条调用数值的样本标为 —。

| 编号 | 会话与原始 Trace | 对应调用 | 可见，k | 累计，k |
|---|---|---|---:|---:|
| S1 | [AKA-2 / CUDA / E19](/Users/guoyuqi/atrex-runs/production-qwen35-35b-fp8-atrex-gdn-4k256-20260814--fused-moe-fp8--l20n--claude/AKA/atrex-runs2.with-traces/atrex-runs2/claude-session-traces/projects/-root-atrex-runs2--atrex-long-horizon-worktrees-kernel-opt-fused-moe-fp8-cuda-l20n-production-e0019-90c2f208/a63a18bd-bb07-4b7a-b5a1-40316d97a226.jsonl:412) | 读取 `profile_nvidia.sh`，随后检查 `ncu_utils.py`、`analyze_reports.py`；数值仅计 wrapper 读取。 | 4.151 | 240.803 |
| S2 | [AKA-1 / CUDA / E16](/Users/guoyuqi/atrex-runs/production-qwen35-35b-fp8-atrex-gdn-4k256-20260814--fused-moe-fp8--l20n--claude/AKA/atrex-runs.with-traces/atrex-runs/claude-session-traces/projects/-root-atrex-runs--atrex-long-horizon-worktrees-kernel-opt-fused-moe-fp8-cuda-l20n-production-e0016-d932ade8/6d1d011a-793a-45e0-b44f-cfc670ddf483.jsonl:531) | NCU 执行 `profile_driver.py`，截取最后 40 行；返回 RNG、AbsFunctor、MaxNan，而非 `k_gemm`。 | 9.592 | 479.684 |
| S3 | [AKA-2 / Triton / E7](/Users/guoyuqi/atrex-runs/production-qwen35-35b-fp8-atrex-gdn-4k256-20260814--fused-moe-fp8--l20n--claude/AKA/atrex-runs2.with-traces/atrex-runs2/claude-session-traces/projects/-root-atrex-runs2--atrex-long-horizon-worktrees-kernel-opt-fused-moe-fp8-triton-l20n-production-e0007-3149d99b/2b4f84bc-2910-425c-8d1e-1aea5abe8a7b.jsonl:18) | 拼接 `memory/v7.json`、`v6.json`、`v5.json` 后截取前 200 行，不保证三份文件全部可见。 | 5.305 | 429.727 |
| S4 | [AKA-1 / Triton / E13](/Users/guoyuqi/atrex-runs/production-qwen35-35b-fp8-atrex-gdn-4k256-20260814--fused-moe-fp8--l20n--claude/AKA/atrex-runs.with-traces/atrex-runs/claude-session-traces/projects/-root-atrex-runs--atrex-long-horizon-worktrees-kernel-opt-fused-moe-fp8-triton-l20n-production-e0013-9b56efaf/f8a17ec0-f5b9-4144-b93f-0bbb8b8f743c.jsonl:687) | 搜索自身 Claude JSONL，再整理 `--input`、独立 argv 和隐藏 Shape 注入约定。 | 6.315 | 404.260 |
| S5 | [AKA-1 / CuteDSL / E1](/Users/guoyuqi/atrex-runs/production-qwen35-35b-fp8-atrex-gdn-4k256-20260814--fused-moe-fp8--l20n--claude/AKA/atrex-runs.with-traces/atrex-runs/claude-session-traces/projects/-root-atrex-runs--atrex-long-horizon-worktrees-kernel-opt-fused-moe-fp8-cutedsl-l20n-production-e0001-12b93f3f/55f3cd86-e247-4ea0-8dd0-ebe8b3987eec.jsonl:124) | 读取 CUTLASS `wmma_programming.rst`，研究 FP8 fragment、`ldmatrix`、流水线和 shared-memory layout。 | 4.436 | 230.709 |
| S6 | [Retained / CUDA / E9](/Users/guoyuqi/atrex-runs/production-qwen35-35b-fp8-atrex-gdn-4k256-20260814--fused-moe-fp8--l20n--claude/runtime/workspace-full-20260902.unpacked/production/control-l20n/state/artifacts/sha256/95bf5c5d9c17844990a8c9f2ac33f83b55f1d20ef908f905b4a0251f886ef469/payload/conversation.jsonl:49) | 读取七份 Skill 后复核历史分析的适用性。 | — | — |

S1 的目标为 `k_gemm_fused` 和 `k_gemm`；S5 的关键区分是硬件 block-scaled MMA 的 scale 要求与当前算子的 float32 scale。S2 还说明按行截取不等于限制 Token，Wiki 的单行 JSON 也有此问题。

Retained 的 harness 阅读类中，187 次调用涉及 `runtime_tools.py`，覆盖 71 个 Session，命令/返回累计约 2.598M Token，说明工具协议理解仍有成本。

### 重复执行核查

在 Shell 文本子集中，按同一 Session 内完全相同的命令文本匹配，AKA / retained 的额外调用为 **124 / 1 次**；命令及完整返回都相同的为 **5 / 0 次**。该检查不识别路径、空格或读取范围变化后的语义重复，也不判断重试是否必要。

累计读取不是重复执行次数：一次调用产生的文本也可多次进入后续请求。统计与高权重调用见 [用途分类索引](shell-read-audit.json)，识别规则见 [离线分类脚本](shell_read_audit.py)。

### 相邻思考去重示例

以 S1 为例，逐片段的原文分词与保留场景如下。这里只计算已观察到的后续请求，思考实际是否回传仍有前文所述不确定性。

| Trace 行号 | 首次生成 Token | 假设后续保留次数 | 生成＋累计读取 Token | 关联调用 |
|---|---:|---:|---:|---|
| 412 | 355 | 58 | 20,945 | 第 413 行 Bash。 |
| 417 | 963 | 57 | 55,854 | 分析第 413 行返回，同时准备第 418、420 行 Bash。 |

第 417 行只计一次，再在三个不同关联调用间各分配三分之一；不是在前后调用中分别计全量，也不代表三个调用真实独占了同等成本。完整汇总与片段索引见 [相邻思考结果](bash-adjacent-thinking-summary.json)。

## Provider 用量复核

本节保存总账核查与 pooled/evolve 横向数据；研究结论和 Bash 分类统一见 [主报告](../report.md)。所有数值对应本轮归档配置，不采用当前源码已调整的 15-Attempt 设置。M 为百万 Token，显示值按原始精度汇总后舍入。

### 范围与去重

- AKA：114 份优化 Episode Trace、113 份独立 Policy Review Trace、10 份嵌套子 Agent Trace；另一次 Policy Review 仅留下 221,597 total tokens，计入总量而不推断分项。
- AKA 按 message ID 合并流式记录，取最后观察到的完整 usage；不取首条，也不分别取各字段最大值，因为 input 可能在终态重新拆分为 uncached/cache read/cache write。跨文件未发现重复 assistant message ID。
- Runtime：复核全部 501 个非 Bootstrap Session，终态 `modelUsage` 合计 **4,601,307,346 Token**，与初版 Runtime 总账一致。该数覆盖全部配置，下面横向表仅选 pooled、retained、evolve。
- 两侧均排除 Bootstrap。Runtime 的 `modelUsage` 已含 Provider 管理的子请求，不再重复相加；AKA 分别汇总主 Trace、独立子 Trace 和 Review。总量为 uncached input、cache read、cache write、output 之和。
- 行为计数来自可见 Trace，响应按所属链的 message ID 去重；Runtime 中间 usage 不完整，不据此推算逐请求的权威上下文长度。

### 各配置用量

AKA 为两次运行合计并包含子 Agent/Review；evolve 包含 Evolver。

| DSL | AKA 两遍：完整 Trace | Runtime pooled：双 Trajectory | Runtime retained：双 Trajectory | Runtime evolve |
|---|---:|---:|---:|---:|
| CUDA | 667.349 | 407.615 | 397.642 | 376.093 |
| Triton | 553.115 | 314.368 | 279.218 | 355.442 |
| CuteDSL | 691.442 | 439.772 | 436.636 | 346.718 |
| 合计 | **1,911.906** | **1,161.755** | **1,113.496** | **1,078.253** |

每个 DSL：AKA 共 38 个 Episode；pooled/retained 各 40 个 Attempt；evolve 为 38 个 Optimizer Attempt 加 9 个 Evolver Session。Episode/Attempt 都可包含多次实验，AKA 前两个 fast Episode 还要求各做五个 Trial，因此轮次数不代表等搜索预算。

以下拆出优化主循环：AKA 不含独立子 Agent/Reviewer；evolve 不含 Evolver；Runtime Optimizer 用量仍包含 Provider 管理的子请求。

| 指标 | AKA 两遍 | Runtime pooled | Runtime retained | Runtime evolve Optimizer |
|---|---:|---:|---:|---:|
| Episode / Attempt 数 | 114 | 120 | 120 | 114 |
| Token 总量 M | 1,859.139 | 1,161.755 | 1,113.496 | 907.493 |
| 每 Episode / Attempt 平均 M | 16.308 | 9.681 | 9.279 | 7.961 |
| 可见模型响应总数 | 17,891 | 11,594 | 11,115 | 9,119 |
| 每 Episode / Attempt 平均响应数 | 156.9 | 96.6 | 92.6 | 80.0 |
| 工具调用数 | 21,156 | 13,689 | 13,389 | 11,184 |
| 上下文压缩次数 | 230 | 143 | 142 | 116 |

AKA 优化主 Trace 的用量分项如下，与包含子 Agent/Review 的全量表区分：

| 类别 | M tokens | 占比 |
|---|---:|---:|
| Cache read | 1,785.294 | 96.03% |
| Cache write | 53.772 | 2.89% |
| Output | 19.966 | 1.07% |
| Uncached input | 0.107 | 0.006% |

### AKA 账本校正

合并报告时重新读取 237 份 AKA Trace，校正后数值与此前核查一致。每个实例/DSL 的精确分项、费用及 Trace SHA-256 保存在 [Provider 复核数据](provider-usage-audit.json)；费用沿用 2026-09-02 的 12/1/15/36 元/M 快照，未按当前价格重估。

用 `attempt.json.tokens` 代替完整主 Trace、取子 Agent 每个 message 的首条 usage，再加独立 Review，可以复现初版报告；改用完整 Trace 终态 usage 后，少计项如下：

| 差额来源 | 少计 M tokens |
|---|---:|
| 5 个 Episode 的 `attempt.json.tokens` 小于完整主 Trace | 16.619 |
| 子 Agent 首条流式 usage 尚未完整 | 22.009 |
| 合计 | **38.628** |

五个 Episode 的完整 Trace 相对 `attempt.json.tokens` 的差额：

| AKA 实例 | DSL | Episode | 少计 Token |
|---|---|---:|---:|
| AKA-1 | CuteDSL | 1 | 15,238,138 |
| AKA-1 | CUDA | 16 | 107,603 |
| AKA-2 | CUDA | 3 | 77,754 |
| AKA-2 | CUDA | 6 | 108,838 |
| AKA-2 | Triton | 17 | 1,086,234 |
| **合计** | | | **16,618,567** |

AKA-1 / CuteDSL / E1 的 `resume_count=2`，账本为 12,717,544 Token，完整同一 Session 为 27,955,682。首条 Prompt 明确为优化 Episode 1，并非 Bootstrap。这五项也不能全部归因于显式 resume，其中部分 `resume_count=0`。

流式漏计样本：[AKA-2 / CUDA / E5 子 Agent Trace](/Users/guoyuqi/atrex-runs/production-qwen35-35b-fp8-atrex-gdn-4k256-20260814--fused-moe-fp8--l20n--claude/AKA/atrex-runs2.with-traces/atrex-runs2/claude-session-traces/projects/-root-atrex-runs2--atrex-long-horizon-worktrees-kernel-opt-fused-moe-fp8-cuda-l20n-production-e0005-9085e72d/28cf71f2-5a6c-4ab1-aa9d-c4efd57b9341/subagents/agent-a2ebf60612717ebce.jsonl) 中，`msg_d6656534-bbc4-9ba2-b3f3-389acea9a4ac` 首条为 `input_tokens=6694, output_tokens=0`，后续更新为 `input_tokens=6, cache_creation_input_tokens=7196, output_tokens=707`。首条不是完整结算。

校正后 AKA-1 为 **952.215M**，AKA-2 为 **959.692M**，合计 **1,911.906M**；初版报告为 **1,873.279M**，少计约 **38.628M（2.06%）**。Runtime 使用终态 `modelUsage`，未发现同类差额。统一后的报告已采用校正用量与重算费用，运行归档未改写。

### 辅助归因核查

AKA 已保存的独立 Policy Review 为 27.713M，另有 0.222M 仅保留总数；嵌套子 Agent 为 24.833M；七个 `invalid_handoff` Episode 的全部用量为 24.220M。最后一项包含探索本身，不是格式失败造成的净损失。

按最近 phase marker 归类，profile/research/implementation/planning/recording 分别为 35.4%/15.6%/20.0%/9.2%/2.6%；但只有约 45.8% 的主 Trace Token 位于紧邻匹配的 start/end 区间，因此这些标签不能充当可靠的语义成本分摊。

两个补充样本：

- [AKA-1 / CUDA / E1](/Users/guoyuqi/atrex-runs/production-qwen35-35b-fp8-atrex-gdn-4k256-20260814--fused-moe-fp8--l20n--claude/AKA/atrex-runs.with-traces/atrex-runs/claude-session-traces/projects/-root-atrex-runs--atrex-long-horizon-worktrees-kernel-opt-fused-moe-fp8-cuda-l20n-production-e0001-98c7e9c0/dabd3345-2804-4e96-b4f5-51b9a17be2cf.jsonl:3)：10.581M；配对较完整的阶段用量为 planning 3.487M、implementation 2.634M、benchmark 1.394M、recording 1.233M，未标记 1.833M。Trace 第 92/95 行重试 validator 参数，第 116 行调用 reviewer helper，第 159 行处理 Git/policy request，第 329 行发生 Journal 导入错误。
- [Pooled / CUDA / E1 / T1 / A1](/Users/guoyuqi/atrex-runs/production-qwen35-35b-fp8-atrex-gdn-4k256-20260814--fused-moe-fp8--l20n--claude/runtime/workspace-full-20260902.unpacked/production/control-l20n/state/artifacts/sha256/e32ba23303bcb884ed27459d69c71289d4db51a3b045de79c39c1ecb77f081f0/payload/conversation.jsonl:330)：121 个响应、10.728M。第 330/472 行调用 Profile，第 535 行读取已有 Gateway Result，第 547 行提交报告；使用 Runtime 接口仍需构造请求、修复 probe 和记录交接。
