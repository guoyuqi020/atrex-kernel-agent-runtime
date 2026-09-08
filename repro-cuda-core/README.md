# cuda.core 掉 shape 问题复现包

复现 2026-09-07 生产事故:cuda 臂 attempt `ce5e1c19` 的 dirB 候选在权威 evaluate 下
随机丢失 shape(该次丢 58 和 80),根因是部分 agate pod(driver 580.159.03 队列)的
frozen 环境缺 `cuda.core`,candidate import 即崩:

```
File "/tmp/agate-ev_2c38c1615cfe/candidate/kernel.py", line 80, in <module>
    from cuda.core import LaunchConfig, Program, ProgramOptions, Stream, launch
ModuleNotFoundError: No module named 'cuda.core'
```

## 文件

| 文件 | 说明 |
|---|---|
| `kernel.py` | 触发问题的原始候选字节(50,516 B,sha256 `877badef43ce…`,来自 artifact `4446b6e70a1f…`,line 80 即 traceback 位置) |
| `reference/` | 完整 90-shape 参考包(reference.py / input.py / shapes.json / metadata.json / roofline.json),从 `dsls/cuda/evaluation-contract.json` 原样导出,roofline 已按 runtime 逻辑去掉硬件后缀 |
| `reference-s58/` | 只含 shape 58(当时被丢的 shape 之一)的单 shape 参考包,用于快速抽签 |
| `make_payload.py` | 按 `agate.py::_build_request` 的字段逐一构造 eval 请求(spec.languages=["cuda"]、target_hardware=["L20N"]、num_correctness_cases=5、bench_iters=100、deps_mode=freeze_installed、mode=full、lock_clocks、harness=atrex_bench、requirements 空) |
| `eval-full90.json` / `eval-shape58.json` | 已生成的请求(可直接 `agate submit`) |
| `repro-loop.sh` | 批量抽签脚本,自动判定是否复现并统计 driver 分布 |

凭据/地址走当前环境变量:`AGATE_AK`、`AGATE_SK`、`AGATE_URL=https://atrex-gateway.alibaba-inc.com`
(与 runtime.json 的 agate 配置一致;CLI 默认 profile prod 也是同一地址)。

## 命令

单次提交(单 shape,等待结果):

```bash
cd /root/atrex-kernel-agent-runtime/repro-cuda-core
agate submit eval-shape58.json --wait --wait-timeout 1800
```

全量 90 shape(注意:这是 **1 个 job 落在 1 个 pod**,只相当于 1 次抽签;
runtime 的 batched 路径是每 shape 一个 job × 90 = 90 次抽签,所以整轮 eval 的
掉 shape 概率 ≈ 1-(1-0.0022)^90 ≈ 18%):

```bash
agate submit eval-full90.json --wait --wait-timeout 3900
```

批量抽签(推荐,复现效率最高;50 次、8 并发):

```bash
./repro-loop.sh 50 8
```

事后人工核验某个 job(看 driver 与 compile reason):

```bash
agate get <job_id>          # result.environment.driver_version / result.passed.compile
```

## 复现判据

命中 = 同时满足(与事故当日的 46 个失败 batch 完全一致的形态):

1. `job.status == "succeeded"`(基础设施故障被误报为成功)
2. `result.passed.compile.<shape>.status == "failed"`,reason 含
   `ModuleNotFoundError: No module named 'cuda.core'`(kernel.py line 80)
3. `result.passed.correctness/performance == "skipped"`
   ("Skipped because compile stage failed.")→ 该 shape 无 latency 表项
4. `result.environment.driver_version == "580.159.03"`

## 概率与时效性说明

- 事故窗口内:580.159.03 队列 46 失败 / 20,651 batch ≈ **0.22%/job**;
  580.126.09 队列 0 / 9,455 = 0%。
- 失败只出现在 09-07 09:57 之后(evaluate 负载从 3/h 爬到 50/h 的扩容期),
  疑似自动扩容起的 pod frozen env 不完整。**坏 pod 队列若已下线,今天可能抽签
  多次也无法复现** —— 抽不到时应先用 `agate get` 看 driver_version 分布确认
  580.159.03 队列是否还在服务。
- 同一份字节在健康 pod 上必然通过(bootstrap 与其余 20k+ batch 均成功),
  这正是该缺陷"随机掉 shape、kernel 本身无错"的原因。

## 备选:不构造 payload 的 CLI 一行式(非字节级等价)

```bash
agate eval --gpu L20N --candidate kernel.py --reference-dir reference-s58 \
  --operator qwen35_35b_fp8_atrex_gdn_4319x256_flash_attention \
  --num-correctness-cases 5 --bench-iters 100 \
  --deps-mode freeze_installed --lock-clocks --mode full --harness atrex_bench
```

差异:CLI 路径 `spec.languages` 固定为 `["triton"]`(runtime 提交的是 `["cuda"]`),
且不带 metadata/roofline 之外的 runtime 覆写;复现 pod 环境问题本身不受影响,
但要求字节级对齐时用 `make_payload.py + agate submit`。
