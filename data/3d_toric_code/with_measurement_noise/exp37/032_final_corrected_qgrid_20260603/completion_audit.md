# exp37 completion audit

日期：2026-06-03

目标：修复 exp36 在 3D toric code 中因 logical-sector 处理错误/冻结导致的错误结果，并在 `p=0.05`、`q=0.08..0.23`、`L=3,4,5` 上给出可信 `q_top`。

## 结论

当前 final artifact 为 `final_qgrid.npz`、`final_qgrid.csv`、`final_qgrid.md`。三尺寸、16 个 q 点共 48 行全部为 `PASS:4`。

## 逐项证据

1. 修正 logical-sector observable。
   - 代码：`src/exp37_sector_ti.py` 使用 `corrected_c_eta_section`，即 `x + r(Hx + Heta) + r(Heta)`。
   - manifest：`030`、`031` 和 `029` 均记录 `sector_observable=corrected_c_eta_section`。
   - 小 L 校验：`029_corrected_sector_l2_exact_smoke_20260603/exact/exact_sector_weights.json` 给出 L=2 exact；同目录 AIS smoke 在 `q=0.23` 得到 `q_top=0.978321`，exact 为 `0.979631`，差约 `0.0013`。

2. 不依赖冷端 MCMC 跨 logical sector。
   - 生产 estimator：`ais_estimator=flip_reweight`。
   - 每个 AIS 末态枚举 8 个 zero-syndrome logical representatives 并重加权扇区贡献，因此输出是 sector weight/free-energy estimate，而不是冻结链的 sector histogram。

3. 覆盖目标网格。
   - `final_qgrid.npz` 字段：
     - `lattice_size_list.shape=(3,)`，值为 `3,4,5`。
     - `q_values.shape=(16,)`，值为 `0.08..0.23`。
     - `p_value=0.05`。
     - `q_top_per_disorder.shape=(3,16,4)`。

4. 保存 plan.md 要求的基本产物。
   - `final_qgrid.npz` 包含 `weights_per_disorder (3,16,4,8)`、`delta_f_per_disorder (3,16,4,8)`、`q_top_per_disorder`、`q_top_stderr_per_disorder`、`q_top_ci95_per_disorder`、`weights_stderr_per_disorder`、`delta_f_stderr_per_disorder`、`flags_per_disorder`、`ais_ess_per_disorder`、`ais_ess_fraction_per_disorder`。
   - 源 AIS NPZ 生成时尚未保存 `delta_f_stderr_per_disorder`；本 final NPZ 用 `weights_stderr_per_disorder` 的 delta-method 近似补齐，并在 `manifest_json` 中记录。后续代码已修正为直接保存该字段。

5. PASS/WARN 状态。
   - `flags_per_disorder` 的唯一取值为 `PASS`。
   - `final_qgrid.md` 每行 `flags=PASS:4`。
   - `030` 的 L5 `q=0.08/0.09` 曾因 `ESS/R<0.02` warning；最终用 `031` 的 t16 reinforcement 替换，`q=0.08 min ESS/R=0.0521`、`q=0.09 min ESS/R=0.0385`。

6. 结果来源可追溯。
   - `source_experiment_per_point` 与 `source_run_per_point` 保存在 `final_qgrid.npz`。
   - 合并规则：除 `L=5,q=0.08/0.09` 来自 `031_l5_lowq_t16_R4_20260603/L5_q008_009_t16_nd3` 外，其余行来自 `030_ais_corrected_flip_grid513_R4_d4_20260603`。

7. 验证命令。
   - `conda run -n 12 python -m py_compile src/exp37_sector_ti.py src/summarize_exp37_ais.py src/finalize_exp37_qgrid.py`
   - `conda run -n 12 python src/finalize_exp37_qgrid.py --output-dir data/3d_toric_code/with_measurement_noise/exp37/032_final_corrected_qgrid_20260603`
   - 最终 NPZ inspection 确认 48 行全 PASS、L5 低 q source override 正确、`delta_f_stderr_per_disorder` 全部有限。

## 剩余限制

本轮目标是用尽可能短的计算时间给出 corrected `q_top`，因此 disorder 数为 4，不是高统计量 threshold 定值。`L=3` 在高 q 方差较大，最终表已用 `total_sem_q_top` 体现。
