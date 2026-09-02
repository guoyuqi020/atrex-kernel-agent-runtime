# ADR 0036：Evolver 仅使用文件系统 Evidence

[English](0036-frozen-evolver-runtime-tools.md) | 中文

## 状态

已接受并实现；取代此前的 Evolver Runtime Tools 设计。

## 背景

Evolver 只针对一个不可变 Lineage Checkpoint 离线改造 Agent。它不像 Optimizer 那样执行 Kernel
评测、查询 Wiki、追加 Journal 或访问其他可变 Runtime 状态。专用命令 Client、HTTP Capability、查询
Catalog 和 Candidate Reset 记录，只是在重复 Runtime 本可物化为普通只读文件的事实。

## 决策

Evolution Input schema v11 不包含 Runtime 查询权限。Runtime 物化：

- `input/agents/agent-vN/` 下每个可见 Agent 唯一一份仓库与逐 Trajectory Runtime State；目录按稳定
  Lineage 版本命名，而不是按临时 Epoch 角色命名；
- `input/evidence/agent-vN/` 下每个可见版本一份 Runtime 派生的优化汇总，以及最近完成 Epoch 全部
  Branch 的逐 Attempt Conversation 与 Attempt Report；
- `input/evolution-reports/evo-N.json` 下可用的历史 Agent 创建报告；
- 标明当前 Parent，以及同一 Epoch 中已经创建但尚未评测的 Challenger 的只读 Catalog。

Evolver 直接读取这些文件，只能修改 Candidate `source/`、Candidate `runtime-state/` 和 `scratch/`。
Runtime 初始复制 Active Source，以及最近完成 Epoch 获胜分支中产出最佳 Kernel 的 Trajectory 在该
Epoch 最后一个 Attempt 后的终态 State；下一 Epoch 的 Active Branch 使用相同 State 种子。缺失终态
State 时依次回退到该 Trajectory 的 Epoch 起始 State、Revision Seed 和空默认值。选择
`evolve_from_history` 时，Evolver 用所选历史 Source 替换 Candidate Source，并可从可见历史状态整理
公共种子。新 Revision 也可以融合多个合格可见 Agent 的 Source、Skill 或 Tool，并把这些贡献者与唯一
Source Base 分开记录。Runtime 验证资格、报告的 Source 根目录相对 Diff 与私有 State Diff；每个新 Revision 把 Source 和 State 封存为一个
逻辑 Bundle。不信任也不需要 Candidate Base 旁路记录。

删除 Evolver 查询接口、查询 Capability、公开 Helper、Candidate Allowlist 和私有查询 Snapshot。详细历史与
完整 Evolution Trace 继续保存在 Runtime 现有 Evidence 与 Registry Store 中，只投影紧凑的 Agent 创建报告。

## 后果

- Evolver 的完整输入 Contract 是一棵可直接检查的冻结文件树。
- Workspace 和 Prompt 更简单。
- 不再存在 Evolver 专用 HTTP Credential 或查询服务。
- 历史 Base 替换由 Agent 执行，但 Runtime 会独立验证。
- 多 Agent 贡献是显式 Provenance，不增加祖先边。
- Optimizer 仍需要实时服务和 Journal，因此其 Runtime Tools 保持不变。
