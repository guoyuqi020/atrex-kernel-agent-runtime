# 决策 0032：分离 CAS 内容身份与 Lineage 内 Kernel 版本

[English](0032-lineage-kernel-version-labels.md) | 中文

## 状态

已接受并实现。

## 背景

内容寻址 Artifact 使用 `sha256:<digest>` 标识，适合校验和去重，却无法直观展示 Kernel 在何时
被引用、与此前 Kernel 有何关系。如果把时间戳或展示版本写入参与哈希的 Manifest，相同内容会产生
不同 Digest。时间也不是去重对象的固有属性：同一个 Artifact 可以在不同时间被多个持久记录引用。

不透明 Kernel Revision ID 与 Parent ID 已能保存身份和祖先关系，但操作者很难直接阅读
`v0`、`v1`、`v2` 历史。Attempt Evaluation 还是可探索、可重试的序列，不能与持久 Kernel
Revision 混为一谈。

## 决策

CAS 身份保持不变。Registry schema 16 新增 `lineage_kernel_versions`，在唯一 Lineage 内为每个
Kernel Revision 分配不可变、从零开始的 `revision_number`。Bootstrap 把 Baseline 固定为 `v0`；
每个终态 Attempt Kernel 在注册它的同一事务中获得下一个编号。映射保存 `linked_at`，分别约束
Kernel 唯一性与 `(lineage, revision_number)` 唯一性，并要求子 Kernel 的 Parent 已位于同一
Lineage。schema 15 迁移按 Epoch/Attempt 领域顺序重建稳定编号。

Catalog 投影新增 `version`、`revision_number`、`parent_version`、处置结果、语义创建时间和相对
Parent 的性能变化，同时保留全部不透明 ID 与 Digest。Artifact 引用也投影为包含 `digest`、
`kind`、`referenced_at` 的对象；旧 Digest 字段继续保留。`list-kernels --format table` 输出便于
人工阅读的历史。

探索 Gateway Evaluation 使用独立的 `g<recovery_generation>-e<ordinal>` 标签，不占用 Kernel
的 `vN` 编号。只有可信控制器注册终态 Kernel Revision 时，Runtime-final Candidate 才获得版本。

## 影响

Kernel 版本不会因重启、后续插入或查询排序变化而改变。多个兄弟版本可以连续编号但拥有相同
`parent_version`，因此 Parent Link 而非数字相邻关系定义真实演进。CAS 去重与校验不受影响。
操作者必须把 `referenced_at` 理解为语义引用时间，而不是 Blob 固有创建时间。
