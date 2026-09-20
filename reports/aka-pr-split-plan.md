# AKA simplified PR 拆分计划

## 拆分原则

将原 PR 按七个明确的评审主题重新组织，而不是按照开发过程的 Commit 顺序切分。原始大提交同时修改启动入口、工具和流程，不能直接 cherry-pick 后声称各批独立。应从上游 main 出发逐批迁入，并把后续安全性、恢复和用量修复合并到所属主题中。

当前准备的第一批以 AKA 上游 main 的 `d643fb5bc004ed021d1a60aa99bf9513204d3025` 为基础。原 `simplified` 分支保持不变。

评审目标是：第 1～6 批主要完成可观测性和执行职责迁移，第 7 批才改变优化流程。这里的“行为等价”不包括保留已确认的安全漏洞、错误计费或错误接收逻辑；这些修复必须明确说明。

## 七批边界

| PR | 主题与主要内容 | 不应混入的内容 | 依赖与验收 |
| --- | --- | --- | --- |
| 1 | Session 可观测性与用量：统一 conversation、Provider stream/native 子 Agent 记录、逐响应及逐 invocation 用量、partial/unavailable、增量读取与容量保护 | HTTP Runtime、bwrap、权限模型、Prompt 改写、Setup/Fast 删除、Git/评测/Journal 迁移 | 可独立合并。原启动命令、流程和平台路径仍可用；写盘失败不能杀死 CLI；重试与子 Agent 不重复计费 |
| 2 | Agent Workspace 隔离：可独立启用的启动封装、挂载白名单、凭证范围、辅助 Session 只读输入、scratch 生命周期 | 提前剥夺旧 Agent 仍依赖的 Git 和 Gateway 能力、删减优化步骤 | 依赖 PR1 或独立运行。保留明确的旧流程兼容路径；完整隐藏 Git 的切换放 PR6 |
| 3 | Supervisor GPU/Wiki Runtime：HTTP 服务、轻量 sandbox.py client、私有 evaluator 与输入、请求授权、参数校验和结果投影；同步 GPU Measurement/KernelWiki Skill | Journal/Report 协议变化、测量策略变化、优化流程删除 | 接入 PR2 的隔离路径。全量路由与旧接口能力对照，授权缩写绕过等安全修复随本批合入 |
| 4 | Measurement Record 与可靠执行：精确源码/请求/结果持久化、全局 ID、跨 Episode 去重、基础设施重试、ABBA checkpoint/resume、排除 cancelled/不确定结果 | Journal 状态机、Prompt 工作流简化 | 依赖 PR3。缓存损坏、任务取消、暂态读失败、恢复与重提都需测试；历史格式策略要明确 |
| 5 | Runtime Journal / Episode Report：Direction/Experiment 持久化与查询、Gateway Record 引用、状态校验、可修正重交；同步 runtime-records Skill | Git 接管、删除原来的 plan/profile/phase 工作流 | 依赖 PR4。迁移期为旧流程保留必要的报告投影/适配，不能要求提前使用第 7 批 Prompt |
| 6 | Supervisor 接管 Git、验收和晋级：候选提交、源码/测量/实验/报告绑定、Gate、ABBA 复用、promotion audit、恢复与 Long Horizon 集成 | 删除 Setup/Fast、改变探索策略和 Prompt | 依赖 PR2～5。到此才完全隐藏 Git、禁止 Agent 提交；原优化流程仍能端到端运行 |
| 7 | 优化流程简化：移除不再需要的 Setup/Fast、plan/profile/Phase Marker、旧 CLI/Skill/兼容分支，重写 Prompt 和使用文档 | 新增基础设施机制 | 基于 PR6；单独评估性能、Token 与成功率变化，使效果回退可归因 |

### 两个需要调整的评审边界

1. **PR2 隐藏 Git 与 PR6 接管 Git 不能脱节。** 旧 Agent 还要提交候选时，先把 .git 隐藏会直接破坏流程。PR2 可以交付隔离能力及挂载模型，但完全收回 Git 的默认切换必须随 PR6 的 Supervisor 提交流程完成。类似地，在 PR3 的 HTTP 代理就绪前，不能切断旧 Agent 的 GPU 工具。
2. **三次测量取 per-shape median 属于评测策略变化。** 虽可归入 PR4 的主题，但不能混称为无行为变化的持久化重构。建议独立为本批中的策略提交/开关，先维持上游默认；切换默认时另行报告测量成本与接受结果影响。如果坚持严格等价，应把默认切换推迟到第 7 批之后。

新代码应携带对应修复，而不是先提交已知有漏洞的版本，再用很多补丁修复。文档也应每批描述当时可用的行为，而不是等第 7 批才补齐。

## 已准备的 PR1

- 分支：`codex/pr1-session-observability`。
- 本地 Commit：`f01957d`；19 个文件，新增 3,391 行、删除 27 行，其中约一半新增内容为测试。尚未推送。
- 验证：58 项测试通过，Python 编译检查和 Git whitespace 检查通过；原 CLI 的 `--help` 可正常运行。未启动真实模型或 GPU 任务。
- 独立 Worktree：`workspaces/aka-pr1-session-observability`，位于 Runtime 仓库本地工作区中，Git 历史属于 AKA。
- 原 AKA Worktree 仍在 `third_party/atrex-kernel-agent` 的 `simplified` 分支。
- 没有复制 Supervisor Runtime 或 agent_sandbox；采集直接接到既有 `run_bounded`，只抽取必要的安全文件读取逻辑。
- Long Horizon 将每次 invocation 放进既有 Episode archive；单次会话默认放到候选工作区之外，避免进入候选 Commit。
- 保留 Setup、Framework Baseline、Fast/Full、Prompt、Phase Marker、Git、Gateway、Journal 和晋级逻辑。Phase Marker 与计费事件的时序也需保留，不能直接沿用 simplified 已删 Phase Marker 后的合并方式。
- 唯一 Provider 启动参数调整是取消 Qoder 的 `--no-session-persistence`，用于生成可捕获的 native transcript；不加入 Codex 的 Git 绕过参数，也不改变 HOME/凭证配置。
- 本批不保证捕获任意绕过共享启动入口的外部 reviewer 或第三方进程；只记录共享入口 invocation 及 Provider 可识别的 native 会话/子会话。未导出的调用不能标成完整。

## PR1 提交说明草稿

**Title:** Add live session transcripts and reconciled provider usage

### Summary

Extract the session-observability changes from the simplified AKA proposal into a standalone change on upstream main. Keep the current optimization workflow, prompts, Gateway, Journal, Git ownership, and promotion policy unchanged.

### Changes

- Save a live `conversation.jsonl`, provider stream/native transcripts, and `token-usage.json` for each Agent invocation.
- Include the initial/resume prompt and provider-exported subagent conversations without requiring an AKA header on native JSONL.
- Reconcile unique response and terminal counters; distinguish exact, partial, and unavailable usage, with Qoder credits stored separately.
- Preserve Codex resume deltas and root phase-event ordering without inventing child phase attribution.
- Bound incremental transcript reads/discovery and keep draining child pipes after capture failures.
- Store Long Horizon invocations in the existing Episode archive; preserve existing launch/timeout/resume behavior.

### Non-goals

No HTTP Supervisor Runtime, Bubblewrap requirement, Gateway/Journal API redesign, Git authority migration, new evaluation policy, or workflow simplification. No changes to Agent prompts or installed skills.

### Compatibility notes

Works through the existing local coordinator path without a new platform requirement. Qoder native session persistence is enabled for capture. Trace files can contain sensitive prompt/tool content and are not added to the existing upload manifest. Provider calls not exported through the supported session sources remain explicitly outside the coverage guarantee.

### Validation

Run the commands in `docs/session-observability.md`. Fixtures and local subprocesses cover success/timeout/resume, native children, duplicate counters, shared-home filtering, capacity limits, capture write failures, and phase-order preservation. These checks do not claim real GPU/model end-to-end qualification.

## 后续提交方式

PR1 先独立评审。PR2～7 应按依赖顺序建立小分支；如并行展示，采用 stacked PR，并明确每个 PR 的 base branch。合并一批后，下一批 rebase 到新的上游 main，再检查文件清单和端到端回归。

当前只准备本地 PR1 分支，不更改原大 PR，也不自动关闭原 PR。发布前检查目标是 AKA 的 main，而不是 Runtime main 或旧 core main；把新分支推到 fork 后再发起跨仓库 PR。
