# GDN launcher

Task inputs and templates: [`data/GDN`](../../data/GDN/README.md).
Default workspace: `workspaces/GDN`; no generated files are written to `data/`.

Use `prepare.py --inputs data/GDN-full` for the [original-hint input variant](../../data/GDN-full/README.md).
When `--workspace` is omitted, preparation uses `workspaces/<input directory name>`.
For this variant, pass `--workspace workspaces/GDN-full` to every `run.py` command.

Run in Lima Ubuntu with the Linux Runtime environment activated:

```bash
python scripts/gdn/prepare.py --backend claude
python scripts/gdn/run.py serve
# In a separate terminal with the same environment and required sandbox privileges:
python scripts/gdn/run.py campaign --target-epoch 5
```

For a separate experiment, pass `--workspace workspaces/GDN-clean` to all commands.
`run.py ablation` uses the workspace's frozen seven-arm definitions. Wiki is independent;
these scripts do not start or stop it. Preparation does not run Agents or evaluations.
Run roles use existing workspace snapshots; they never silently re-prepare changed task inputs.

中文：输入和模板只放在 [`data/GDN`](../../data/GDN/README.zh.md)，脚本放在此目录。
保留原始提示的版本位于 [`data/GDN-full`](../../data/GDN-full/README.zh.md)，通过
`prepare.py --inputs data/GDN-full` 选择，默认输出到 `workspaces/GDN-full`。
实际配置、源码副本、数据库、Session、凭据、日志及结果均位于 `workspaces/GDN`。
使用新输入时，为所有命令指定相同的新 `--workspace`；恢复旧实验直接运行 `run.py`，
不会重新导入当前 `data` 中的修改。Wiki 需独立启动。
