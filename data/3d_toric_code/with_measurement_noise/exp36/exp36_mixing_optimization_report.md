# exp36 mixing 优化记录

目标：在 `p=0.05, q=0.08..0.23` 附近提高 3D toric code 的 logical-sector mixing 和 PT transport，最终给出可用于重新生产扫描的参数/程序方案。

## 2026-05-27 首轮准备

- 已提交并推送 common-beta 同步 PT 语义修正：`2dc407066 Fix sync PT ladder temperature semantics`。
- 已提交并推送 exp35 审计文档整理：`a1bc41891 Document exp35 audit under experiment folder`。
- 当前工作区在本轮开始时已清洁。

exp35 失败基线：

- 旧 exp35 pooled 数据没有保存 per-temperature logical-sector flip/winding 接受率，只保存冷端 winding 接受率、PT swap 接受率和冷端 never-flip 链数。
- 代表点 `L=6,q=0.08`：`never=6.55/8`，cold winding acceptance `2.994e-06`，mean PT min swap `5.893e-05`。
- 代表点 `L=5,q=0.08`：`never=7.27/8`，cold winding acceptance `3.470e-06`，mean PT min swap `3.944e-04`。
- 因此 exp36 的第一步必须补齐诊断，否则无法定量判断哪个温度、哪个 move 或哪个 PT ladder 是瓶颈。

首轮实施计划：

1. 在 PT 生产结果中保存每温度 winding accepted/attempted/rate，以及相邻温度 swap accepted/attempted/rate。
2. 增加可选的 PT sector 诊断：按 measurement cadence 统计每个温度的 logical-sector signature flip 次数、首次改变位置和 sector histogram，但不保存完整 trace。
3. 本地 smoke 验证新字段可读、旧默认行为不破坏。
4. 服务器首轮并行探索建议：
   - A: common-beta `K=9, q_hot=0.44, winding_repeat_factor=1`，作为修正 PT 后基线。
   - B: common-beta `K=17, q_hot=0.44, winding_repeat_factor=1`，检验 swap bottleneck 是否主要来自温度间隔。
   - C: common-beta `K=17, q_hot=0.49, winding_repeat_factor=1`，检验更热端是否显著提高 hot-sector flips。
   - D: common-beta `K=17, q_hot=0.49, winding_repeat_factor=4`，检验增加 nontrivial winding proposal 密度是否把 hot flips 输送到 cold。

判据：

- 首要：每温度 winding acceptance、sector flip count、cold/hot sector flip count、hot-to-cold transport proxy。
- 次要：mean PT min swap acceptance、adaptive PT flow `f(k)` 是否接近线性、`never/8` 是否显著下降。
- 只有当 `L=5/6, q=0.08..0.12` 的 cold never-flip 和 PT min swap 同时改善，才进入大样本生产。

### 本地验证结果

代码改动：

- `run_parallel_tempering_measurement` 新增压缩 sector 诊断开关 `track_logical_sector_diagnostics`。
- `run_disorder_average_simulation` 和 `production_chunked_scan.py` 会保存：
  - `chain_pt_winding_acceptance_rate_per_temperature_per_disorder_per_start_replica(_tensor)`
  - `chain_pt_winding_accepted/attempted_count_per_temperature_per_disorder_per_start_replica(_tensor)`
  - `chain_pt_swap_acceptance_rate_per_pair_per_disorder_per_start_replica(_tensor)`
  - `chain_pt_swap_accept/attempt_count_per_pair_per_disorder_per_start_replica(_tensor)`
  - 开启 `--track-pt-sector-diagnostics` 时保存 per-temperature sector flip count、first change index、sector histogram、hot-to-cold sector delivery count。

验证：

- `PYTHONPATH=src conda run -n 12 python -m py_compile src/mcmc_parallel_tempering.py src/main.py src/production_chunked_scan.py` 通过。
- `PYTHONPATH=src conda run -n 12 python -m unittest discover -s tests` 通过。
- 直连 smoke 输出：`data/3d_toric_code/with_measurement_noise/exp36/diagnostic_smoke_20260527/diagnostic_smoke_L3_p0p05_q0p23_K3.npz`。
- chunk/merge smoke 输出：`data/3d_toric_code/with_measurement_noise/exp36/production_diagnostic_smoke_20260527/production_diagnostic_smoke.npz`。

chunk/merge smoke 的关键字段形状：

- `chain_pt_winding_acceptance_rate_per_temperature_per_disorder_per_start_replica_tensor`: `(1,1,1,1,1,3)`。
- `chain_pt_swap_acceptance_rate_per_pair_per_disorder_per_start_replica_tensor`: `(1,1,1,1,1,2)`。
- `chain_pt_sector_flip_count_per_temperature_per_disorder_per_start_replica_tensor`: `(1,1,1,1,1,3)`。
- `chain_pt_sector_histogram_per_temperature_per_disorder_per_start_replica_tensor`: `(1,1,1,1,1,3,8)`。

smoke 数值仅用于确认字段：

- per-temperature winding acceptance: `[0.0, 0.0, 0.3761]`。
- per-temperature sector flip count: `[0, 0, 2]`。
- hot-to-cold sector delivery count: `0`。

下一步远端 pilot：

- 使用当前代码提交后的版本，在 nd-1/nd-2/nd-3 分别跑 A-D 中的不同配置。
- 优先取难点 `L=5,6`、`p=0.05`、`q=0.08,0.12`，每点先用小样本 `32` disorder、`4` start chains、`512` measurements、显式 burn-in cap，打开 `--track-pt-sector-diagnostics`。
- 如果 `K=17,q_hot=0.49,winding_repeat_factor=4` 仍显示 hot 有 flip 但 delivery=0，则下一轮应重点改 PT ladder/adaptive flow；如果 hot 本身也少 flip，则重点改 sector-changing proposal。

### 远端 pilot 启动记录

已在远端共享存储启动首批 q=0.08 pilot。代码通过 `rsync` 从本机同步，运行命令显式记录 `--git-commit-sha f1afc9b6a`。

共同参数：

- `code_family=3d_toric`
- `p=0.05`
- `q=0.08`
- `L=5,6`
- `num_disorder_samples_total=8`
- `num_start_chains=4`
- `num_measurements_per_disorder=512`
- `num_sweeps_between_measurements=6`
- `num_burn_in_sweeps=300`
- `max_effective_num_burn_in_sweeps=1500`
- `adaptive_pt_rounds=3`
- `adaptive_pt_calibration_sweeps=256`
- `observable_temperature_mode=cold`
- `track_pt_sector_diagnostics=True`
- `cluster_update=False`

配置：

| config | node | screen | K | q_hot | winding_repeat_factor | run root |
|---|---|---|---:|---:|---:|---|
| A | nd-1 | `exp36_A` | 9 | 0.44 | 1 | `.single_shot/exp36/exp36_mixing_pilot_20260527/A_common_beta_K9_qhot044_wr1` |
| B | nd-2 | `exp36_B` | 17 | 0.44 | 1 | `.single_shot/exp36/exp36_mixing_pilot_20260527/B_common_beta_K17_qhot044_wr1` |
| C | nd-3 | `exp36_C` | 17 | 0.49 | 1 | `.single_shot/exp36/exp36_mixing_pilot_20260527/C_common_beta_K17_qhot049_wr1` |
| D | nd-1 | `exp36_D` | 17 | 0.49 | 4 | `.single_shot/exp36/exp36_mixing_pilot_20260527/D_common_beta_K17_qhot049_wr4` |

启动状态：

- 四个任务均已通过 exact/preflight validation。
- 四个任务均已进入 `Launching chunk workers: 4 workers for 16 chunks`。
- 截至本记录，尚未完成合并；下一轮需要先检查 screen/log，再收集 final NPZ。

### 首轮 pilot 局部结果与诊断

首轮任务后来提前停止，只保留已完成 chunk 用于定位瓶颈。本地同步并补齐远端 A 的 L=6 chunk 后，可用正式数据为：

- A: `K=9,q_hot=0.44,winding_repeat_factor=1`，`L=6,p=0.05,q=0.08`，4 个 disorder chunk，16 条 start-chain 样本。
- B: `K=17,q_hot=0.44,winding_repeat_factor=1`，`L=6,p=0.05,q=0.08`，2 个 disorder chunk，8 条 start-chain 样本。
- C/D 只有 preflight，没有正式 L=6 chunk，不用于定量结论。

共同采样参数：

- `num_measurements_per_disorder=512`
- `num_sweeps_between_measurements=6`
- `num_burn_in_sweeps=300`
- `max_effective_num_burn_in_sweeps=1500`
- `num_start_chains=4`
- `adaptive_pt_rounds=3`
- `adaptive_pt_calibration_sweeps=256`
- `observable_temperature_mode=cold`
- `track_pt_sector_diagnostics=True`
- `cluster_update=False`

关键结论：

- A/B 的冷端 `k=0` logical sector 都完全没有翻转。
  - A: `never=[4,4,4,4]`，cold sector flips `0/16`。
  - B: `never=[4,4]`，cold sector flips `0/8`。
- 热端会频繁翻转 logical sector，说明问题不是热端本身缺少 sector-changing move。
  - A hot `k=8`: winding acceptance `0.386410`，sector flips 平均 `384.812/511`。
  - B hot `k=16`: winding acceptance `0.385942`，sector flips 平均 `380.375/511`。
- hot-to-cold sector delivery 全部为 0。
  - A: `0/16`。
  - B: `0/8`。
- A/B 的 PT transport 断在热端附近。
  - A mean swap per pair: `[7.93e-4,8.48e-3,0.362,0.392,0.423,0.203,1.33e-2,4.65e-4]`，`pt_min_swap=0`。
  - B mean swap per pair: `[0.813,0.789,0.754,0.717,0.673,0.623,0.568,0.506,0.541,0.482,0.423,0.371,0.328,0.309,0.308,0.000]`，最后一跳 `15-16` 完全为 0。
- B 的 adaptive ladder 从 `k=15` 到热端 `k=16` 跳跃过大：
  - `k=15`: mean `(p,q)=(0.255386,0.291545)`。
  - `k=16`: `(p,q)=(0.427823,0.440000)`。
  - 这解释了最后一跳 swap acceptance 为 0：热端已经能跨 sector，但不能把状态输送回冷端。

因此首轮定量诊断把瓶颈定位为 adaptive PT ladder 在热端形成过大的 terminal gap，而不是 hot-sector flip scarcity。后续应优先修复/限制 ladder gap，并与 static common-beta ladder 对照。

### adaptive PT gap cap 修复

已实现并提交 adaptive ladder gap cap：`02e4878ef Cap adaptive PT ladder gaps`。

修复内容：

- `adaptive_ladder_from_flow(..., max_log_gap_factor=...)` 增加 log-ladder 相邻 gap 上限。
- 默认 `DEFAULT_ADAPTIVE_PT_MAX_LOG_GAP_FACTOR=1.5`。
- 当 cap 生效时 status 记录为 `ok_capped_gap`。
- 生产扫描保存 gap-cap 相关配置与实际 ladder。

验证：

- `PYTHONPATH=src conda run -n 12 python -m py_compile src/mcmc_diagnostics.py src/main.py src/production_chunked_scan.py src/mcmc_parallel_tempering.py` 通过。
- `PYTHONPATH=src conda run -n 12 python -m unittest discover -s tests` 通过。
- 本地 smoke 输出：`data/3d_toric_code/with_measurement_noise/exp36/adaptive_gap_cap_smoke_20260527/adaptive_gap_cap_smoke.npz`。
- smoke 中 heat ladder log gap `max/uniform = 1.5`，确认 cap 生效。

下一轮 pilot 设计：

- E: static common-beta `K=17,q_hot=0.44,winding_repeat_factor=1,adaptive_rounds=0`，检验不用 adaptive flow 时最后一跳是否消失。
- F: capped adaptive `K=17,q_hot=0.44,winding_repeat_factor=1,adaptive_rounds=3`，直接检验 gap cap 是否修复 B 的 terminal gap。
- G: capped adaptive `K=33,q_hot=0.44,winding_repeat_factor=1,adaptive_rounds=3`，提高温度分辨率。
- H: capped adaptive `K=33,q_hot=0.49,winding_repeat_factor=2`，检查更高热端和更多 winding proposal 是否能在保持 transport 的同时增加 sector exploration。

pilot2 先只跑难点 `L=6,p=0.05,q=0.08`，每配置少量 disorder，用 `--track-pt-sector-diagnostics` 评估：

- cold sector flips 是否从 0 变为正数；
- hot-to-cold sector delivery 是否非零；
- `pt_min_swap` 是否不再被 terminal gap 压到 0；
- per-pair swap curve 是否没有孤立断点。

### pilot2 transport 对照结果

完成并同步到本地：

- `data/3d_toric_code/with_measurement_noise/exp36/pilot2_transport_20260527/E_static_K17_qhot044_wr1/E_static_K17_qhot044_wr1.npz`
- `data/3d_toric_code/with_measurement_noise/exp36/pilot2_transport_20260527/F_capped_K17_qhot044_wr1/F_capped_K17_qhot044_wr1.npz`
- `data/3d_toric_code/with_measurement_noise/exp36/pilot2_fast_probe_20260527/F_capped_K17_qhot044_wr1_m128/F_capped_K17_qhot044_wr1_m128.npz`

共同正式参数：

- `L=6,p=0.05,q=0.08`
- `num_disorder_samples_total=2`
- `num_start_chains=4`
- `num_measurements_per_disorder=512`
- `num_sweeps_between_measurements=6`
- `num_burn_in_sweeps=300`
- `max_effective_num_burn_in_sweeps=1500`
- `K=17,q_hot=0.44`
- `track_pt_sector_diagnostics=True`
- `cluster_update=False`

远端长版 `G/H`：

- `G_capped_K33_qhot044_wr1` 与 `H_capped_K33_qhot049_wr2` 在运行约 54 分钟后仍无正式 chunk 写出，已停止。
- 结论：`K=33,512 measurements,全温度 sector diagnostics` 对当前迭代太重；后续 K=33 只做短版 probe 或减少诊断频率。

正式结果：

| config | adaptive | K | q_hot | winding_repeat | pt_min_swap mean | hot-to-cold delivery | cold sector flips mean | cold chains flipped | hot sector flips mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E static | 0 | 17 | 0.44 | 1 | 0.00984 | 11/8 chains | 1.75 | 4/8 | 387.0 |
| F capped | 3 | 17 | 0.44 | 1 | 0.01028 | 18/8 chains | 1.00 | 4/8 | 380.1 |

per-pair swap 的主要变化：

- 首轮 B 的最后一跳 `15-16` swap 为 `0.000`，terminal gap 完全断开。
- pilot2 E 的最后一跳 `15-16` mean swap 为 `0.599`。
- pilot2 F 的最后一跳 `15-16` mean swap 为 `0.414`。
- 因此 gap cap/静态 ladder 已经修复了热端 terminal transport，hot-to-cold delivery 也从 0 变为非零。
- 新 bottleneck 出现在中间 pair `6-7`：
  - E: min swap pair `6-7`，mean `0.01187`，对应平均 `(p,q)=(0.2252,0.2640)->(0.2555,0.2917)`。
  - F: min swap pair `6-7`，mean `0.01280`，对应平均 `(p,q)=(0.2214,0.2605)->(0.2514,0.2880)`。

fast probe `F_m128` 也给出一致信号：

- `pt_min_swap=0.01713`
- hot-to-cold delivery `0/4`，但 measurement 太短，不能作为 delivery 结论。
- cold sector flips mean `0.5`，cold chains flipped `1/4`。
- hot sector flips mean `95.75/127`。

本轮结论：

- PT transport 已经从“热端完全断开”改善为“能把 hot sector 送回 cold”，这是实质性改善。
- 但 cold 端真正的 logical-sector sampling 仍严重不足：512 measurements 下 cold sector flips 只有 `1.0-1.75` 次/链，且只有 `4/8` 链发生过 cold-sector 改变。
- 当前主瓶颈不是 terminal hot gap，而是中间温度带 `p≈0.22-0.26,q≈0.26-0.29` 附近 swap 太低；此外 cold winding acceptance 仍只有 `1.215e-05`，cold 端自身跨 sector 很少。

下一轮方向：

1. 用短版 K=33 probe 检查加密温度点是否抬高中间 `6-7` bottleneck。
2. 比较 `winding_repeat_factor=2/4` 是否增加 sector proposal 供给；注意如果只增加 hot flips 而 cold flips 不变，说明仍是 transport 而非 proposal 瓶颈。
3. 评估更高频 swap 或更短 measurement 间隔是否能增加 replica round-trip，从而提高 cold sector flips。
4. 若 K=33 能提高中间 min swap，但 wall time 过高，则需要优化 sector 诊断开销或只在 pilot 阶段使用全温度 sector 诊断。
