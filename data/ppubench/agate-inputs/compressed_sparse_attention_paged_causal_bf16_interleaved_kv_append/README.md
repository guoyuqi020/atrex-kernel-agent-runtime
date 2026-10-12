# CSA paged causal：Agate 输入精简版

本目录的 `input.py` 对应
`compressed_sparse_attention_paged_causal_bf16_interleaved_kv_append`，
从 **10,762 字节减为 8,012 字节**，低于网关对 `reference.input_py`
解码后 UTF-8 源码的 **8,192 字节**限制，余量 180 字节。

只删除模块说明、两个未使用的来源哈希常量、私有函数和局部变量的类型注解，
缩短内部名称并压缩空白。原始说明和哈希仍保存在 `curated/` 原件中。
`adaptation.json` 记录原件路径、双方哈希、字节数和完整名称映射。

保持 `_make_inputs` 参数名、默认值及其类型注解，所有输出键、校准数值、
校验条件与错误文本不变。保留所有 Torch 运算及其顺序，包括随机采样和种子、
大 KV cache 分配、非连续 value stride、storage offset、未使用槽位的初始值。
没有删减 25 个 shape，也没有减小张量或采样规模。

组装这个算子的 Eval bundle 时，将本目录的 `input.py` 作为 bundle 的
`input.py`（即请求的 `reference.input_py`），其余 reference、shape、metadata
继续使用原题文件。当前不会自动替换 `curated/` 或任何运行中工作区的输入。
不要对上传文件运行通用格式化器；增加空白也可能使其再次超过限制。

验证：

- `tests/test_ppubench_compact_input.py` 核对源文件哈希、UTF-8 大小和入口签名；
  逆转声明的名称替换并去掉非运行注解后，完整 AST 必须与原件相同。
  这会检查全部运算、控制流、字符串、参数和调用顺序，不依赖抽样 shape。
- 本地网关源码 `app.security.validator.validate_input_generator` 校验通过。
  此项是本地校验，并非正式网关的新一次提交或 PPU 数值评测。

这次只解决输入源码大小。原始 `solution.py` 的 `os` / `pathlib` 导入及
私有后端依赖仍需另外适配，不能据此将整个题目标记为已在 Agate 跑通。
