# Runtime 代码组织

[English](code-organization.md) | 中文

本文定义 Runtime 源码的可维护性边界。它是实现指导，不是协议；模块之间移动代码时，不得改变持久身份、SQLite Schema、Artifact 格式、Worker 工作区布局以及 HTTP/CLI 返回格式。

## 组织原则

依赖应当向内收敛：

```text
入口层（CLI / ASGI）
        |
        v
装配层 + 展示层
        |
        v
应用服务（bootstrap / controller / workers）
        |
        v
领域模型 + ports
        ^
        |
基础设施适配器（registry / artifacts / Agate / Wiki / Git）
```

- 入口层只解析输入、调用一个应用操作并输出结果，不负责拼装 SDK 凭据，也不重复实现领域对象投影。
- 装配模块负责把配置转换成运行对象，但不包含业务决策。
- 展示函数把持久模型转换成 CLI 与 HTTP 共用的稳定 JSON Read Model。
- 应用服务实现用例；在需要替换或测试隔离的位置依赖 Port。
- 基础设施适配器可以依赖领域类型；领域代码不得反向导入 SQLite、HTTP、Subprocess 或 SDK 实现。

## 当前源码边界

现有顶层目录已经表达了有效且稳定的职责，应继续保留：

| 区域 | 职责 |
|---|---|
| `api/` | 认证管理 HTTP 与 ASGI 生命周期 |
| `cli/` | 公开命令入口、Parser、命令族与进度渲染 |
| `composition/` | Bootstrap、Campaign、Wiki Worker 的配置到对象装配 |
| `domain/` | 不可变身份、状态模型和领域错误 |
| `controller/` | Campaign/Epoch 编排、Lease、Evidence 构造与持久 Task |
| `workers/` | Optimizer/Evolver 进程、工作区、沙箱和 Report 处理 |
| `gateway/` | Agate 适配、Capability 控制、代理、评测和指标 |
| `registry/` | 持久状态 Port 与 SQLite 实现 |
| `artifacts/` | 内容寻址 Artifact 存储 |
| `knowledge/` | 外部 Wiki 协议、代理与反馈 Outbox |
| `kernel_agents/` | Git 导入与不可变 Agent Revision 构造 |

共享叶子边界避免应用服务重复实现机械逻辑：

- `presentation.py` 统一维护 Administration HTTP 与 CLI Inspect 共用的 JSON 投影。
- `gateway/configuration.py` 成为 Agate 配置及 Secret 到连接对象的唯一解析入口。
- `asgi.py`、`filesystem.py` 与 `serialization.py` 维护受限 Transport、安全文件和规范 JSON 原语。
- `gateway/candidate.py` 维护已验证 Kernel Artifact 解析。

`composition/bootstrap.py` 现在统一装配 Campaign Bootstrap、Base Agent 导入和 Artifact-rooted Lineage Seed。CLI 与 ASGI 不再分别构造这些对象图。

第二轮整理在不改变公开 CLI 与 HTTP 契约的前提下，继续分离命令及控制职责：

- `cli/parser.py` 维护完整参数 Schema，其余 `cli/` 模块维护各自命令族；`cli/__init__.py` 只保留 Serve 和分发。已安装的 `atrex_runtime.cli:main` 入口保持不变，`cli/__main__.py` 同时支持 Module 方式执行。
- `controller/tasks.py` 维护持久 Task 的 Lease、Heartbeat 和执行；`api/administration.py` 保持为认证 HTTP 控制面，`api/app.py` 负责 ASGI 生命周期装配。
- `composition/bootstrap.py`、`composition/campaign.py` 和 `composition/knowledge.py` 统一维护部署对象装配，不在旧平铺路径保留兼容 Shim。
- `gateway/control_models.py` 维护不可变 Capability、Bootstrap 和 Evaluation 记录；`gateway/control_schema.py` 维护 Schema 创建及旧版本 Migration；`gateway/control.py` 保留 SQLite Authority 操作以及兼容导出。
- `gateway/execution.py` 统一 Finalization、Lineage Seed 和 Measurement 的阻塞 SDK 调用、JSON Object 校验、结构化 Candidate 拒绝以及 Eval/Profile 结果封存。

最后一轮减法删除了 CLI 私有转发别名，合并了重复的终端表格渲染，并把共用的 Agate Evaluation Request 构造收敛到 `gateway/execution.py`。这些改动直接减少重复代码，没有增加新的装配层；持久化记录和外部接口均保持不变。

Finalization、Lineage Seed 和 Measurement 继续分别维护评测策略；共用 Agate 执行和结果封存位于
`gateway/execution.py`。`config.py`、`ports.py`、`secrets.py` 等跨域叶子模块继续放在包根目录；
把它们移入泛化 Utility 目录反而会模糊依赖。

## 不应合并的内容

- Attempt Evidence 与 Epoch/Evolver Evidence 的可见性和信任语义不同。
- Optimizer 与 Evolver 的工作区及进程权限不同，必须继续分离。
- Gateway 探索性评测和 Runtime 权威终评的状态副作用不同；即使以后共享执行 Helper，也不能合并语义。
- Registry 与 Artifact Store 的一致性模型不同，不应抽象成同一个持久层。

## 重构门禁

每次结构调整都必须通过 Ruff、Strict Mypy 和完整测试。数据库 Migration、协议字段、配置键、CLI 输出或 Worker 可见布局的变化，必须作为独立设计变更处理，不能夹带在代码整理中。
