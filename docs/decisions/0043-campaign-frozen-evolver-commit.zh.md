# 决策 0043：把 Evolver Commit 冻结进 Campaign 身份

## 状态

已接受。

## 决策

第一次成功 Campaign Bootstrap 会把部署选择的完整 `campaign.evolver.commit` 写进持久
Campaign 记录。此后的 Bootstrap 重试、`run-campaign`、持久 Campaign Task 和 Evolver
dev-shell 都必须使用完全相同的 Commit。如果 Runtime 配置选择了另一个 Commit，系统会在解析或
启动 Evolver Bundle 前拒绝执行。

以 Artifact 为种子的 Lineage 继承目标 Campaign 已冻结的 Evolver Commit。它选中的 Agent
Artifact 会替代新根的 Core Bootstrap，但不会建立另一套 Evolver 身份。

Registry schema 23 增加可空的 `campaigns.evolver_commit`。Null 只用于本决策前创建的 Campaign。
下一次相同 Bootstrap 重试或持有 Fencing 的 Campaign 运行会为这类旧 Campaign 一次性绑定
Commit，并记录 `campaign.evolver_commit_bound`；之后再修改就会被拒绝。Evolver dev-shell
不会为旧的 Null Campaign 自动选择来源，因为调试操作不应产生此类 Provenance 副作用。

## 影响

Optimizer 与 Evolver Provenance 现在具有对称的恢复语义：Kernel Agent Revision 由内容寻址，
执行 Evolver 的仓库则按 Campaign 固定。修改部署默认值只影响新 Campaign。若要有意测试另一个
Evolver Commit，必须新建 Campaign，或未来设计显式迁移操作；仅修改 Runtime JSON 不会再静默
改变已有实验。
