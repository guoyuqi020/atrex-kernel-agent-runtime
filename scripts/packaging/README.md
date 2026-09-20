# Offline Gateway single wheel

Target: **Linux x86_64, CPython 3.12, glibc >= 2.17**. This unofficial repack includes matching Gateway Server/Client versions and the Server's base Python dependencies. Original sources, metadata and licenses are preserved. Dependencies live under private `_vendor/` and are enabled only in Gateway command processes, without overwriting existing dependency packages. Use a dedicated environment; upstream Server/Client installations provide the same command names.

## Install and use

```bash
python -m pip install --no-index ./atrex_gateway_standalone-0.13.18-cp312-cp312-manylinux_2_17_x86_64.whl
python -m atrex_gateway_standalone --check
atrex-gateway serve --local
```

From another terminal: `AGATE_URL=http://127.0.0.1:8000 agate health`. Module alternatives: `python -m atrex_gateway_standalone server --help` and `python -m atrex_gateway_standalone client --help`.

To use the privately bundled SDK:

```python
from atrex_gateway_standalone import activate

activate()
from atrex_gateway_client import Client
```

**Offline Gateway installation does not provide a complete offline GPU evaluation environment.** Python, PyTorch, NumPy, CUDA, drivers, DSL toolchains, atrex-bench and deployment extras (Ray/database clients) are excluded. GPU jobs may still install their pinned evaluator or custom dependencies; provision these separately before disconnected job execution. `serve --local` requires a usable GPU environment. The Local Backend executes submitted code with the service user's permissions, not as a security boundary for untrusted callers.

## Rebuild

The builder needs `packaging`; the output has no additional pip dependencies. Download target wheels without installing them on the build host:

```bash
python -m pip download atrex-gateway-server==0.13.18 atrex-gateway-client==0.13.18 \
  --dest wheelhouse --only-binary=:all: \
  --implementation cp --python-version 3.12 --abi cp312 --platform manylinux_2_17_x86_64 \
  --index-url http://artlab.alibaba-inc.com/1/pypi/pypi-releases \
  --extra-index-url http://artlab.alibaba-inc.com/1/pypi/simple \
  --trusted-host artlab.alibaba-inc.com
python scripts/packaging/build_gateway_standalone.py --wheelhouse wheelhouse --output-dir dist
```

The builder independently checks target dependency closure (including `uvicorn[standard]` extras), version constraints, Python compatibility, wheel tags and file collisions. Missing dependencies and unsupported installation schemes/unsafe paths are rejected. `manifest.json` lists each input wheel version and SHA256. Upstream licenses remain in the private dependency tree; the outer `RECORD` covers every output file. Builds from identical wheels are byte-reproducible, and existing output files are never overwritten.
