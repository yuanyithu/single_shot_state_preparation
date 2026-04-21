# data 目录整理说明

当前 `data/` 采用以下约定：

- `nd3_runs/`
  - 服务器批量实验产物，保留原始运行目录、日志、`manifest.json`、分块结果与汇总图。
- `local_threshold_scout/`
  - 本地 threshold scout 脚本的按时间戳输出目录。
- `local_runs/`
  - 后续本地直接运行 `main.py` 等脚本时的默认落盘位置，避免结果继续散落在根目录。
- `legacy_local_runs/`
  - 早期直接保存在 `data/` 根目录的本地实验结果归档。
  - `baseline_multisize/`：最早的多尺寸基线扫描。
  - `kernel_mix/`：随机闭环 / kernel mix 相关本地试验。
  - `q0_geometric_multistart/`：`q=0` geometric multistart 相关本地试验。

根目录原则上只保留说明性文件和一级分类目录，不再直接存放实验 `.npz` / `.png` 结果。
