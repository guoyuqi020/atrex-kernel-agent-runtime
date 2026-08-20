# 决策 0019：使用隔离 Wheel Smoke 约束参考 Checkout 独立性

[English](0019-isolated-wheel-smoke.md) | 中文

## 状态

已接受并实现。

## 背景

Runtime 必须使用发布版 Agate SDK，不能导入其源码树；Wheel 也不能把相邻 Core、Evolver 或 Atrex Bench Checkout 当作 Python Package 导入。Editable 源码测试或仅成功构建 Wheel 都不能证明这种隔离。

## 决策

`npm run smoke:wheel` 使用本地构建 Backend 生成 Wheel，不访问 Package Index、也不安装依赖地把它安装到全新临时 Target；随后从子解释器搜索路径删除仓库 Root/Source Entry，导入完整 Runtime，并解析打包的 CUDA、Triton 和 CuteDSL Base Revision。任何从该 Target 外加载的 ATREX Module 或 Distribution Metadata，以及包含 `deepseek-harness` 或 `atrex-kernel-agent` 的声明依赖，都会使检查失败。

## 影响

仓库验证现在包含自动化负向源码解析门禁。该 Smoke 复用已安装的第三方依赖，因此生产验收仍要求在真正干净的部署主机上使用固定依赖 Artifact 完成安装与执行。
