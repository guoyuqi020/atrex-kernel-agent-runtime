# ADR 0036：面向 Evolver 的冻结 Runtime Tools

[中文](0036-frozen-evolver-runtime-tools.zh.md) | [English](0036-frozen-evolver-runtime-tools.md)

## 状态

已接受并实现。

## 背景

ADR 0035 向 Evolver 暴露所有已完成分支和精确 Kernel Artifact，但手工遍历长 Epoch Tree
成本高且容易遗漏。Runtime 管理 API 已能渲染版本化历史，但向 Evolver 授予其 Bearer Token
会暴露实时可变状态和超出单次 Evolution Session 的权限。仓库持有的 Helper 又可能被进化，
或与 Runtime Evidence Contract 漂移。

## 决策

Evolution Input schema v4 新增固定 `runtime-tools/` 路径。每个 Evolver 进程启动前，Runtime 冻结：

- `catalog.json`：精确 Lineage 内 `vN`/`agent-vN` 标签、Revision 身份、Parent Link、
  Provenance、评测事实、Disposition 和源路径；
- `kernels/<kernel-revision-id>/`：Lineage 的全部历史精确 Kernel Artifact；
- `evolver_tools.py`：Runtime 持有、仅依赖标准库的检索与受约束 Candidate 控制 Client。

Client 提供有界 JSON `history`、`branches`、`attempts`、`kernels`、`kernel-read`、`agents`、
`agent-diff` 和 `trace-paths` 命令。唯一写操作是 `candidate-reset --base <agentrev>`：它只接受
不可变 Manifest 中标为 `lineage_history` 的条目，拒绝 Link 和特殊文件，先构造完整可写副本，
再原子替换 `candidate/`，并在 `scratch/candidate-base.json` 记录所选 Base。Runtime 在启动前
把工具和所有输入目录封为只读。Evolver Session Context 提供精确 Interpreter/Command；最终
封存会核对 Base 记录、提案与真实仓库 Diff。

这是本地冻结 Workspace 操作面，不是 HTTP Capability。它不携带 Admin、Registry、Gateway、Wiki、评测或
晋升 Credential，也不能观察 Workspace 准备后的变化。原始 Evidence 和仓库文件仍可直接验证。

## 后果

- Evolver 获得确定性 Inspect Workflow，不削弱可信控制边界。
- 版本标签来自 Registry Catalog，不从目录顺序推断。
- 即使第一个 Epoch 尚未完成，也能读取全部历史 Kernel 源码。
- Evolution Workspace 会随 Lineage Kernel 历史增长；保留容量仍是运维问题。
- 严格校验 schema v3 的旧 Evolver Bundle 无法消费 schema v4，需显式升级固定 Commit。
