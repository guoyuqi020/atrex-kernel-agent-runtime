# Gateway 单 wheel 离线包

目标为 **Linux x86_64、CPython 3.12、glibc >= 2.17**。此非官方重新打包产物包含相同版本的 Gateway Server、Client 和 Server 的基础 Python 依赖，保留上游源码、依赖元数据与许可证。依赖位于私有 `_vendor/`，仅在 Gateway 命令进程中启用，不覆盖环境已有的同名依赖包。建议使用独立环境，不与官方 Server/Client 同时安装（它们提供同名命令）。

## 安装与使用

```bash
python -m pip install --no-index ./atrex_gateway_standalone-0.13.18-cp312-cp312-manylinux_2_17_x86_64.whl
python -m atrex_gateway_standalone --check
atrex-gateway serve --local
```

另一个终端使用 `AGATE_URL=http://127.0.0.1:8000 agate health`。也可以运行 `python -m atrex_gateway_standalone server --help` 和 `python -m atrex_gateway_standalone client --help`。

SDK 需要显式启用私有模块：

```python
from atrex_gateway_standalone import activate

activate()
from atrex_gateway_client import Client
```

**离线安装 Gateway 不等于离线运行完整 GPU 评测环境。** 本包不包含 Python 解释器、PyTorch、NumPy、CUDA、驱动、DSL 编译工具链、atrex-bench 或 deployment extra（Ray/数据库客户端等）。GPU 任务仍可能需要安装指定版本的评测器或自定义依赖；完全断网执行任务需要另外预置它们。`serve --local` 还需要可用的 GPU 环境；Local Backend 会以服务用户权限执行提交代码，不能作为不可信用户的安全边界。

## 重建

构建环境需要 `packaging`；产物本身没有额外的 pip 依赖。先下载目标平台的 wheel，不在构建机器安装这些包：

```bash
python -m pip download atrex-gateway-server==0.13.18 atrex-gateway-client==0.13.18 \
  --dest wheelhouse --only-binary=:all: \
  --implementation cp --python-version 3.12 --abi cp312 --platform manylinux_2_17_x86_64 \
  --index-url http://artlab.alibaba-inc.com/1/pypi/pypi-releases \
  --extra-index-url http://artlab.alibaba-inc.com/1/pypi/simple \
  --trusted-host artlab.alibaba-inc.com
python scripts/packaging/build_gateway_standalone.py --wheelhouse wheelhouse --output-dir dist
```

构建器独立校验目标平台依赖闭包（包括 `uvicorn[standard]` 的 extras）、版本约束、Python 版本、二进制 wheel 标签和文件冲突；拒绝缺失依赖及无法安全处理的 wheel 路径/安装方案。`manifest.json` 记录每个输入 wheel 的版本和 SHA256，所有上游许可证保留在私有依赖目录，外层 `RECORD` 覆盖全部产物文件。同一 wheelhouse 重建得到相同字节；输出文件已存在时拒绝覆盖。
