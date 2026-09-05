# Best Kernel Sources

本目录保存主报告三种 DSL 中最低 latency 曲线对应的完整 `kernel.py`。文件从归档 Artifact Store 逐字节复制，没有格式化或改写。

| DSL | 配置 | Latency，µs | 生产 Attempt | Kernel Artifact Digest | 完整源码 |
|---|---|---:|---|---|---|
| CUDA | Pooled | 1,086.325 | `attempt_9496a9c9dcbd40448d35ca47c2d7e736` | `sha256:d7808aa57eeb04ba862b58dc41dd3b8cb5d283b16487a4c6c5ee23d31db38f59` | [`cuda/kernel.py`](cuda/kernel.py) |
| Triton | Isolated-01 | 1,044.369 | `attempt_f13ccf039c004fd88382cb7e318ccfd1` | `sha256:5281eef53ea7e7c68bc3096ebe7ae685823305cbe56520de987cdd3f4a3b0ad6` | [`triton/kernel.py`](triton/kernel.py) |
| CuteDSL | Retained | 1,126.074 | `attempt_a8e0d7e04443411aa00c88887f31ec4b` | `sha256:142269d721bf38c86ad602c6d3dab79b34062681f28e4e46a06c7d8a58ab6b7f` | [`cutedsl/kernel.py`](cutedsl/kernel.py) |

Artifact Digest 是 Runtime 对整个 Kernel Artifact 的标识，不是裸 `kernel.py` 文件内容的 SHA-256。
