# 发布检查清单

[English](release-checklist.md) | 中文

## 源码与打包

- [ ] `pyproject.toml` 版本与 Tag 一致，工作树不包含生成状态。
- [ ] Core、Evolver、Atrex Bench Submodule Commit 已 Push、可访问并记录。
- [ ] 干净 Checkout 中 `python -m build` 成功。
- [ ] `scripts/smoke-wheel-independence.py` 证明 sdist/Wheel 不包含相邻仓库，且安装后的 Wheel
  不会导入相邻源码树。
- [ ] Wheel/sdist 只包含 Runtime 代码和文档/许可证元数据，不含 Secret、数据库、Workspace、
  Session Trace 或本地凭据。

## 质量门禁

- [ ] Runtime Ruff、Strict Mypy 和完整 Pytest 通过。
- [ ] 固定 Commit 的 Core/Evolver Ruff、Strict Mypy 和 Pytest 通过。
- [ ] Local Wiki 使用相同 Query Contract 通过测试。
- [ ] 所有 JSON Example 可解析，所有 Markdown 相对链接可解析。
- [ ] 在干净环境从 Wheel 执行 CLI `--help`、Bootstrap、无 Challenger Epoch 和 Evolution Epoch。

## 目标环境验收

- [ ] 精确生产 Linux 镜像通过 `scripts/validate-linux-sandbox.py` 的文件系统、Namespace、cgroup、
  共享宿主网络、DNS 和清理测试。
- [ ] 四类 Worker Root 与并发 Host Probe 均由 `worker_user` 创建；准确部署文件系统（使用
  virtiofs 时包括它）无需 Owner 修复即可通过 Bootstrap。
- [ ] 所选 Claude/Codex/QoderCLI/Pi Backend 完成真实 Session，并记录原始 Trace 与 Provider Token。
- [ ] 所有启用的 Gateway Operation 经 Runtime 连接生产 Agate/目标 GPU 成功。
- [ ] 在代表性算子上验证普通 Evaluate/ABBA 稳定性、Clock Lock、容差、Production Gate 以及可选
  Roofline/NCU SOL Fallback。
- [ ] 演练生产 GPU Wiki Query。

## 运维

- [ ] Capability Signing Key 和 Admin Token 进入 Secret Manager，并在计划重启间保持稳定。
- [ ] 演练存储权限、磁盘预算、备份恢复、Event 导出/清理、Wiki 重试和离线 GC。
- [ ] 使用独立 Service/Task Workspace 演练 `services.sh`、后台 `campaign.sh` 与逐 DSL
  `inspect.sh`。
- [ ] 在 Bootstrap、Attempt 评测、Epoch Commit、Evolution、Wiki 投递和 Task Lease 阶段强制终止；
  重启后不存在重复权限或丢失权威结果。
- [ ] Metrics/日志、外部告警、Incident Owner、回滚 Commit 和 Release Notes 就绪。

任何适用的目标环境项未完成时，不应宣称生产就绪。已知待补证据见[实现状态](implementation-status.zh.md)。
