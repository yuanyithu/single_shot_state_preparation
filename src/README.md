# Source Code

Python 源码统一放在这个目录下。

- `main.py`：核心采样与本地结果落盘入口
- `production_chunked_scan.py`：生产扫描提交、merge、preflight
- `analyze_threshold_crossing.py` / `analyze_no_threshold_evidence.py`：后处理分析
- `plot_scan_results.py` / `plot_threshold_search_overview.py`：绘图
- `build_toric_code_examples.py`、`mcmc.py`、`preprocessing.py`、`linear_section.py`：基础构件
- `exact_enumeration.py`、`diagnose_q0_mixing.py`：校验与诊断

从仓库根目录运行时，优先使用 `python src/<script>.py ...`。
