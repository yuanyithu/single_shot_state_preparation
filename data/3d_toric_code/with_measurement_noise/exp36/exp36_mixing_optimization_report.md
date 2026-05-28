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

### pilot3 低热端与诊断修正

短版 probe 运行在远端并同步到：

- `data/3d_toric_code/with_measurement_noise/exp36/pilot3_cold_mixing_20260528/I_static_K17_qhot032_wr1_m128/I_static_K17_qhot032_wr1_m128.npz`
- `data/3d_toric_code/with_measurement_noise/exp36/pilot3_cold_mixing_20260528/J_static_K17_qhot032_wr4_m128/J_static_K17_qhot032_wr4_m128.npz`
- `data/3d_toric_code/with_measurement_noise/exp36/pilot3_cold_mixing_20260528/K_static_K17_qhot035_wr1_m128/K_static_K17_qhot035_wr1_m128.npz`

共同参数：

- `L=6,p=0.05,q=0.08`
- `num_disorder_samples_total=1`
- `num_start_chains=4`
- `num_measurements_per_disorder=128`
- `num_sweeps_between_measurements=6`
- `num_burn_in_sweeps=150`
- `max_effective_num_burn_in_sweeps=750`
- `K=17`
- `adaptive_pt_rounds=0`
- `track_pt_sector_diagnostics=True`

结果：

| config | q_hot | winding_repeat | min swap | bottleneck pair | hot flips mean | cold flips mean | proxy delivery |
|---|---:|---:|---:|---:|---:|---:|---:|
| F baseline | 0.44 | 1 | 0.0191 | 6-7 | 95.8 | 0.5 | 0 |
| I | 0.32 | 1 | 0.2167 | 12-13 | 95.5 | 0.0 | 8 |
| J | 0.32 | 4 | 0.2042 | 12-13 | 93.5 | 0.0 | 4 |
| K | 0.35 | 1 | 0.1410 | 10-11 | 92.2 | 0.0 | 5 |

结论：

- 降低热端到 `q_hot=0.32/0.35` 能大幅提高 PT swap bottleneck：`0.019 -> 0.14-0.22`。
- 即使 `q_hot=0.32`，hot 端 sector flips 仍接近 `95/127`，说明热端仍足够热。
- 但 128 measurements 内 cold sector flips 为 0，说明低热端改善了 transport 平滑性，却没有立刻解决 cold 端 sector sampling。
- `winding_repeat_factor=4` 没有改善 cold flips，也没有改善 proxy delivery。
- K=33 q_hot=0.44 短版仍然运行过慢，已停止；后续若再测 K=33，必须先降低 sector 诊断开销。

诊断修正：

- pilot3 暴露出旧的 `hot_to_cold_sector_delivery_count` 是 proxy：当冷端 signature 没变时，也可能因为旧 hot signature 与当前 cold signature 相等而计数。
- 已新增更严格字段 `pt_hot_to_cold_sector_change_delivery_count`：
  - 只在该 replica 自上次处于 cold 端之后确实到过 hot 端；
  - 回到 cold 端时 cold signature 相对该 replica 上次 cold signature 发生改变；
  - 且该 signature 与该 replica 最近在 hot 端记录的 signature 匹配。
- 新字段已穿透到 chunk 与 merge 输出：
  - `chain_pt_hot_to_cold_sector_change_delivery_count_per_disorder_per_start_replica(_tensor)`
  - `mean_pt_hot_to_cold_sector_change_delivery_count_curve_matrix`
- 验证：
  - `PYTHONPATH=src conda run -n 12 python -m py_compile src/mcmc_parallel_tempering.py src/main.py src/production_chunked_scan.py`
  - `PYTHONPATH=src conda run -n 12 python -m unittest discover -s tests`
  - smoke 输出：`data/3d_toric_code/with_measurement_noise/exp36/strict_delivery_smoke_20260528/strict_delivery_smoke.npz`
  - smoke 中旧 proxy delivery 为 `[1,1]`，严格 cold-change delivery 为 `[0,0]`，确认能区分“回到 cold”与“cold sector 真改变”。

下一步：

- 用严格字段重新跑短版 probe，对比 `q_hot=0.32/0.35/0.44`。
- 若严格 delivery 仍为 0，则参数优化的重点应转向更长 round-trip/更高 cold-sector change 机会，而不是继续追求 hot 端更热。
- 考虑新增 sector diagnostics stride，减少全温度 logical signature 计算成本，使 K=33 或更长 measurements 可行。

### strict delivery probe 与 sector diagnostics stride

严格 delivery 短版 probe 已完成并同步：

- `data/3d_toric_code/with_measurement_noise/exp36/strict_delivery_probe_20260528/M_strict_K17_qhot032_wr1_m128/M_strict_K17_qhot032_wr1_m128.npz`
- `data/3d_toric_code/with_measurement_noise/exp36/strict_delivery_probe_20260528/N_strict_K17_qhot035_wr1_m128/N_strict_K17_qhot035_wr1_m128.npz`
- `data/3d_toric_code/with_measurement_noise/exp36/strict_delivery_probe_20260528/O_strict_K17_qhot044_wr1_m128/O_strict_K17_qhot044_wr1_m128.npz`

共同参数：

- `L=6,p=0.05,q=0.08`
- `num_disorder_samples_total=1`
- `num_start_chains=4`
- `num_measurements_per_disorder=128`
- `num_sweeps_between_measurements=6`
- `num_burn_in_sweeps=150`
- `max_effective_num_burn_in_sweeps=750`
- `K=17`
- `adaptive_pt_rounds=0`
- `track_pt_sector_diagnostics=True`
- `cluster_update=False`

结果：

| config | q_hot | min swap | bottleneck pair | cold flips mean | hot flips mean | hot winding acc | proxy delivery | strict delivery |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M | 0.32 | 0.1723 | 12-13 | 0.0 | 95.5 | 0.00273 | 7/4 | 0/4 |
| N | 0.35 | 0.1084 | 11-12 | 0.0 | 93.8 | 0.0207 | 3/4 | 0/4 |
| O | 0.44 | 0.00889 | 6-7 | 0.0 | 97.5 | 0.385 | 0/4 | 0/4 |

定量结论：

- 三个 `q_hot` 的 strict delivery 全为 0，说明 128 measurements 内没有看到“热端产生的 sector change 被带回冷端并改变冷端 sector”的事件。
- `q_hot=0.44` 热端 winding 接受率最高，但 transport bottleneck 最差；`q_hot=0.32/0.35` 明显改善 min swap。
- sector flip 打开位置仍在中间热区：
  - M: 从 `k=11,(p,q)=(0.212,0.252)` 开始有少量 flip，`k=14,(0.259,0.295)` 后接近充分翻转。
  - N: 从 `k=9,(0.204,0.244)` 开始有少量 flip，`k=12,(0.259,0.295)` 后接近充分翻转。
  - O: 从 `k=6,(0.225,0.264)` 开始有 flip，但同一位置也是 swap bottleneck。
- O 的 `m=128` 全温度 sector 诊断耗时约 `803s`，说明继续用 stride=1 无法高效做更长 probe。

代码改动：

- 新增 `pt_sector_diagnostic_stride` / CLI `--pt-sector-diagnostic-stride`。
- `track_pt_sector_diagnostics=True` 时，只每 `stride` 个 measurement 计算一次全温度 logical-sector signature；默认 `1` 保持旧行为。
- 新增输出字段：
  - `pt_sector_diagnostic_stride`
  - `pt_sector_diagnostic_sample_count`
  - `chain_pt_sector_diagnostic_sample_count_per_disorder_per_start_replica(_tensor)`
  - `mean_pt_sector_diagnostic_sample_count_curve_matrix`
- 注意：stride 大于 1 后，sector flip rate 应使用 `sample_count-1` 归一化，而不是 `num_measurements_per_disorder-1`。

验证：

- `PYTHONPATH=src conda run -n 12 python -m py_compile src/mcmc_parallel_tempering.py src/main.py src/production_chunked_scan.py` 通过。
- `PYTHONPATH=src conda run -n 12 python -m unittest discover -s tests` 通过。
- smoke 输出：`data/3d_toric_code/with_measurement_noise/exp36/sector_stride_smoke_20260528/sector_stride_smoke.npz`。
- smoke 使用 `num_measurements=9,stride=4`，结果 `pt_sector_diagnostic_sample_count=3`，确认采样点为 `0,4,8`。

下一轮远端 probe：

- 用 stride 版本跑更长 `m=512`，先比较 `q_hot=0.32/0.35/0.44` 的 strict delivery。
- 建议 `--pt-sector-diagnostic-stride 4`，保留足够 sector 时间分辨率，同时把全温度 logical signature 开销降到约 1/4。
- 若 `q_hot=0.32/0.35` 在 `m=512` 仍无 cold flips/strict delivery，下一步应测试更长 round-trip 或显式 cold-sector assist move；若 `q_hot=0.44` 有 strict delivery 但 min swap 极低，则需要在 `q_hot≈0.35-0.44` 之间找折中或加密 bottleneck 区间。

### stride long probe 启动记录

代码版本：

- 提交并推送：`7c691922c Add PT sector diagnostic stride`。
- 远端 source：`/home/DATA1/users/yuany/.single_shot/repos/exp36_stride_probe_20260528/source`。
- run base：`/home/DATA1/users/yuany/.single_shot/exp36/exp36_stride_long_probe_20260528`。

共同参数：

- `L=6,p=0.05,q=0.08`
- `num_disorder_samples_total=1`
- `num_start_chains=4`
- `num_measurements_per_disorder=512`
- `num_sweeps_between_measurements=6`
- `num_burn_in_sweeps=150`
- `max_effective_num_burn_in_sweeps=750`
- `K=17`
- `adaptive_pt_rounds=0`
- `observable_temperature_mode=cold`
- `track_pt_sector_diagnostics=True`
- `pt_sector_diagnostic_stride=4`
- `cluster_update=False`

配置：

| config | node | screen | q_hot | seed_base | run root |
|---|---|---|---:|---:|---|
| P | nd-1 | `exp36_P_stride` | 0.32 | 366000 | `P_stride_K17_qhot032_m512_s4` |
| Q | nd-2 | `exp36_Q_stride` | 0.35 | 367000 | `Q_stride_K17_qhot035_m512_s4` |
| R | nd-3 | `exp36_R_stride` | 0.44 | 368000 | `R_stride_K17_qhot044_m512_s4` |

启动状态：

- 三个任务均通过 exact validation。
- 三个任务均通过 preflight merge。
- 三个任务均已进入 `Launching chunk workers: 1 workers for 1 chunks`。
- 下一轮先检查 final NPZ；若完成则同步到本地并比较 strict delivery、cold flips、min swap 与每温度 sector flip onset。

### stride long probe 局部结果与 round-trip 诊断

已完成并同步：

- `data/3d_toric_code/with_measurement_noise/exp36/stride_long_probe_20260528/P_stride_K17_qhot032_m512_s4/P_stride_K17_qhot032_m512_s4.npz`
- `data/3d_toric_code/with_measurement_noise/exp36/stride_long_probe_20260528/Q_stride_K17_qhot035_m512_s4/Q_stride_K17_qhot035_m512_s4.npz`

`R_stride_K17_qhot044_m512_s4` 仍在 nd-3 运行；`q_hot=0.44` 明显比低热端慢。

P/Q 共同参数：

- `L=6,p=0.05,q=0.08`
- `num_disorder_samples_total=1`
- `num_start_chains=4`
- `num_measurements_per_disorder=512`
- `pt_sector_diagnostic_stride=4`
- 每链 sector diagnostic sample count 为 `128`

结果：

| config | q_hot | min swap | bottleneck pair | cold flips mean | hot flips mean | proxy delivery | strict delivery |
|---|---:|---:|---:|---:|---:|---:|---:|
| P | 0.32 | 0.2094 | 12-13 | 0.0 | 98.5 | 22/4 | 0/4 |
| Q | 0.35 | 0.1264 | 11-12 | 0.0 | 95.3 | 20/4 | 0/4 |

定量观察：

- cold sector histogram 全部在 sector 0：P/Q 的 4 条 chain 都是 `[128,0,0,0,0,0,0,0]`。
- hot sector 在多个 sector 之间频繁切换：
  - P hot histogram mean 约 `[39.25,0,0,24.25,0,31.0,33.5,0]`。
  - Q hot histogram mean 约 `[31.0,0,0,30.0,0,37.0,30.0,0]`。
- P/Q 的 min swap 已经不低，但 cold sector 仍完全不变；因此仅靠“热端会翻 sector + 相邻 swap 接受率高”还不能证明 cold-hot-cold transport 足够。

新增程序诊断：

- 新增轻量 PT replica round-trip diagnostics，不计算 logical observable，只跟踪 `replica_id_per_temperature`。
- 每条链输出：
  - `pt_transport_position_sample_count`
  - `pt_replica_cold_visit_count`
  - `pt_replica_hot_visit_count`
  - `pt_replica_cold_to_hot_passage_count`
  - `pt_replica_hot_to_cold_passage_count`
  - `pt_replica_endpoint_round_trip_count`
  - `pt_replica_min_temperature_visited`
  - `pt_replica_max_temperature_visited`
- 生产 merge 保存对应 tensor：
  - `chain_pt_transport_position_sample_count_per_disorder_per_start_replica_tensor`
  - `chain_pt_replica_*_per_disorder_per_start_replica_tensor`
- 该诊断从 burn-in 结束后开始计数，每次生产期 swap attempt 后记录一次，因此用于区分：
  - replica 是否真的完成 cold→hot→cold round trip；
  - 或者 round trip 已发生，但 sector change 在冷却过程中回到原 sector。

验证：

- `PYTHONPATH=src conda run -n 12 python -m py_compile src/mcmc_parallel_tempering.py src/main.py src/production_chunked_scan.py` 通过。
- `PYTHONPATH=src conda run -n 12 python -m unittest discover -s tests` 通过。
- smoke 输出：`data/3d_toric_code/with_measurement_noise/exp36/roundtrip_smoke_20260528/roundtrip_smoke.npz`。
- smoke 中 round-trip 字段形状正确，例如 `chain_pt_replica_*_tensor` shape 为 `(1,1,1,1,1,4)`。

### roundtrip-only probe 启动记录

代码版本：

- 提交并推送：`46319d276 Add PT replica roundtrip diagnostics`。
- 远端 source：`/home/DATA1/users/yuany/.single_shot/repos/exp36_roundtrip_probe_20260528/source`。
- run base：`/home/DATA1/users/yuany/.single_shot/exp36/exp36_roundtrip_probe_20260528`。
- launcher 留档：`data/3d_toric_code/with_measurement_noise/exp36/launch_roundtrip_probe_20260528.sh`。

共同参数：

- `L=6,p=0.05,q=0.08`
- `num_disorder_samples_total=1`
- `num_start_chains=4`
- `num_measurements_per_disorder=1024`
- `num_sweeps_between_measurements=6`
- `num_burn_in_sweeps=150`
- `max_effective_num_burn_in_sweeps=750`
- `K=17`
- `adaptive_pt_rounds=0`
- `observable_temperature_mode=cold`
- `track_pt_sector_diagnostics=False`
- `cluster_update=False`

配置：

| config | node | screen | q_hot | seed_base | run root |
|---|---|---|---:|---:|---|
| S | nd-1 | `exp36_S_rt` | 0.32 | 370000 | `S_roundtrip_K17_qhot032_m1024` |
| T | nd-2 | `exp36_T_rt` | 0.35 | 371000 | `T_roundtrip_K17_qhot035_m1024` |
| U | nd-3 | `exp36_U_rt` | 0.44 | 372000 | `U_roundtrip_K17_qhot044_m1024` |

启动状态：

- 三个任务均通过 exact validation。
- 三个任务均通过 preflight merge。
- 三个任务均已进入 `Launching chunk workers: 1 workers for 1 chunks`。
- 下一轮先同步 S/T/U final NPZ，读取 round-trip tensor；若 P/Q 的 low-hot ladder 已有充分 endpoint round trips 但 cold sector 仍不变，下一步应测试 sector-preserving 冷却失败机制或显式 cold-sector assist move，而不是继续只优化 swap。

### per-temperature logical-sector 反转定量汇总

输出文件：

- `data/3d_toric_code/with_measurement_noise/exp36/analysis_20260528/pt_temperature_logical_sector_summary.csv`
- `data/3d_toric_code/with_measurement_noise/exp36/analysis_20260528/pt_temperature_logical_sector_summary.json`
- `data/3d_toric_code/with_measurement_noise/exp36/analysis_20260528/pt_temperature_logical_sector_summary.md`

定义：

- `winding_acceptance_rate` 是该温度槽上 nontrivial winding proposal 的 Metropolis 接受率。
- `sector_flip_rate` 是 logical-sector signature 在相邻诊断采样之间改变的频率；stride=4 时一个诊断间隔等于 4 个 measurement。
- 这些是 temperature-slot 诊断，不是 identity-tracked replica 的 sector 历史；round-trip 另用 `replica_id_per_temperature` 诊断。

配置汇总：

| config | chains | measurements | stride | samples | min swap | bottleneck pair | cold flips | cold flip rate | hot flips | hot flip rate | hot winding acc | strict delivery | proxy delivery |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M strict `q_hot=0.32` | 4 | 128 | 1 | 128 | 0.172266 | 12 | 0.000 | 0.000000 | 95.500 | 0.751969 | 0.002727 | 0 | 7 |
| N strict `q_hot=0.35` | 4 | 128 | 1 | 128 | 0.108366 | 11 | 0.000 | 0.000000 | 93.750 | 0.738189 | 0.020724 | 0 | 3 |
| O strict `q_hot=0.44` | 4 | 128 | 1 | 128 | 0.008893 | 6 | 0.000 | 0.000000 | 97.500 | 0.767717 | 0.385348 | 0 | 0 |
| P stride `q_hot=0.32` | 4 | 512 | 4 | 128 | 0.209445 | 12 | 0.000 | 0.000000 | 98.500 | 0.775591 | 0.003405 | 0 | 22 |
| Q stride `q_hot=0.35` | 4 | 512 | 4 | 128 | 0.126374 | 11 | 0.000 | 0.000000 | 95.250 | 0.750000 | 0.020201 | 0 | 20 |
| R stride `q_hot=0.44` | 4 | 512 | 4 | 128 | 0.013605 | 6 | 0.000 | 0.000000 | 91.500 | 0.720472 | 0.385099 | 0 | 3 |
| E pilot2 static `q_hot=0.44` | 8 | 512 | 1 | 512 | 0.011866 | 6 | 1.750 | 0.003425 | 387.000 | 0.757339 | 0.385821 | - | 11 |
| F pilot2 capped `q_hot=0.44` | 8 | 512 | 1 | 512 | 0.012795 | 6 | 1.000 | 0.001957 | 380.125 | 0.743885 | 0.386536 | - | 18 |
| I pilot3 `q_hot=0.32,wr=1` | 4 | 128 | 1 | 128 | 0.216733 | 12 | 0.000 | 0.000000 | 95.500 | 0.751969 | 0.003294 | - | 8 |
| J pilot3 `q_hot=0.32,wr=4` | 4 | 128 | 1 | 128 | 0.204216 | 12 | 0.000 | 0.000000 | 93.500 | 0.736220 | 0.003239 | - | 4 |
| K pilot3 `q_hot=0.35,wr=1` | 4 | 128 | 1 | 128 | 0.140975 | 10 | 0.000 | 0.000000 | 92.250 | 0.726378 | 0.020778 | - | 5 |

关键逐温度结论：

- `q_hot=0.32` 的长 probe P 中，cold `k=0,(p,q)=(0.05,0.08)` 完全无 sector flip；从 `k=11,(0.212,0.252)` 有明显翻转，`k=12,(0.228,0.267)` 后 flip rate 到 `0.215`，`k=14,(0.259,0.295)` 后接近热端水平。
- `q_hot=0.35` 的长 probe Q 中，cold 仍完全无 sector flip；从 `k=8,(0.185,0.226)` 开始零星翻转，`k=11,(0.241,0.279)` 后 flip rate 到 `0.297`，`k=12,(0.259,0.295)` 后接近热端水平。
- `q_hot=0.44` 的长 probe R 中，hot winding acceptance 很高，但 bottleneck pair `6-7` swap 只有 `0.013605`；sector flip 从 `k=4,(0.161,0.203)` 开始，`k=6,(0.225,0.264)` 后大量翻转。
- P/Q/R 的 strict delivery 都是 `0`，说明这些 run 中没有观察到“热端产生的 sector change 被送回冷端并改变冷端 sector”的事件。

roundtrip-only S/T/U 已同步：

| config | q_hot | min swap | bottleneck pair | roundtrip sum | roundtrip per chain | c2h sum | h2c sum |
|---|---:|---:|---:|---:|---|---:|---:|
| S | 0.32 | 0.193066 | 13 | 234 | `[52,69,56,57]` | 250 | 250 |
| T | 0.35 | 0.134755 | 10 | 164 | `[37,41,37,49]` | 184 | 182 |
| U | 0.44 | 0.013563 | 6 | 37 | `[13,9,10,5]` | 49 | 53 |

解释：

- S/T 显示低热端 ladder 已经有大量 endpoint round trips；结合 P/Q 的 cold sector 完全不翻，瓶颈不再是 replica 到不了热端，而是热端 sector change 在回到 cold 的过程中没有改变 cold ensemble 的 logical sector。
- U 的 round trips 明显少，和 `q_hot=0.44` 的中间 swap bottleneck 一致；继续把热端加热到 `0.44` 不划算。

### repeated PT swap sweeps 与下一轮 probe

物理动机：

- 3D `winding_moves` 是整张非平庸 sheet，权重为 `L^2`；在 `L=6,p=0.05` 冷端直接接受率只有 `O(1e-5)`，单纯增加 winding repeat 不能根治。
- P/Q 和 S/T 说明 `q_hot=0.32/0.35` 已有 round trip，但 strict delivery 为 0；下一步优先提高 PT ladder 上 sector-changing 状态的输运/回流机会，而不是继续把热端升到 `0.44`。

程序改动：

- 新增 `pt_swap_sweeps_per_attempt` / CLI `--pt-swap-sweeps-per-attempt`。
- `pt_swap_attempt_every_num_sweeps` 仍控制 cadence；新参数控制每次 cadence 内连续做多少个 alternating even/odd adjacent swap sweep。
- 默认 `1` 保持旧行为；`2` 等价于每次 cadence 连续做一轮 even 和一轮 odd，可提高 temperature-index diffusion 常数但不改变目标分布。
- 结果和 manifest/NPZ 中记录：
  - `pt_swap_attempt_every_num_sweeps`
  - `pt_swap_sweeps_per_attempt`

验证：

- 提交并推送：`60497ad7969de1ad5217a7cbeee716832184d726 Add repeated PT swap sweeps`。
- `python -m py_compile src/mcmc_parallel_tempering.py src/main.py src/production_chunked_scan.py` 通过。
- `PYTHONPATH=src python -m unittest discover -s tests` 通过，包含新增 `tests/test_pt_swap_sweeps.py`。
- 本地 smoke：`data/3d_toric_code/with_measurement_noise/exp36/swap_sweeps_smoke_20260528/swap_sweeps_smoke.npz`。
  - `pt_swap_sweeps_per_attempt=2`
  - final NPZ 中 `chain_pt_swap_attempt_count... = [10,10,10,10]`
  - `chain_pt_transport_position_sample_count = 17`

远端 probe 启动记录：

- launcher：`data/3d_toric_code/with_measurement_noise/exp36/launch_swap_sweep_probe_20260528.sh`
- 远端 source：`/home/DATA1/users/yuany/.single_shot/repos/exp36_swap_sweep_probe_20260528/source`
- run base：`/home/DATA1/users/yuany/.single_shot/exp36/exp36_swap_sweep_probe_20260528`

共同参数：

- `L=6,p=0.05,q=0.08`
- `num_disorder_samples_total=1`
- `num_start_chains=4`
- `num_measurements_per_disorder=512`
- `num_sweeps_between_measurements=6`
- `num_burn_in_sweeps=150`
- `max_effective_num_burn_in_sweeps=750`
- `K=17`
- `q_hot=0.32`
- `adaptive_pt_rounds=0`
- `observable_temperature_mode=cold`
- `track_pt_sector_diagnostics=True`
- `pt_sector_diagnostic_stride=4`
- `cluster_update=False`

配置：

| config | node | screen | pt swap sweeps | seed_base | run root |
|---|---|---|---:|---:|---|
| V | nd-1 | `exp36_V_swap1` | 1 | 382000 | `V_swap1_K17_qhot032_m512_s4` |
| W | nd-2 | `exp36_W_swap2` | 2 | 383000 | `W_swap2_K17_qhot032_m512_s4` |
| X | nd-3 | `exp36_X_swap4` | 4 | 384000 | `X_swap4_K17_qhot032_m512_s4` |

启动状态：

- 三个任务均已通过 exact validation 和 preflight merge。
- 三个任务均已进入 `Launching chunk workers: 1 workers for 1 chunks`。
- 下一轮同步 final NPZ，比较 cold sector flips、strict delivery、endpoint round trips、transport samples 和 wall time。

### swap-sweep probe 结果与 winding-plane heatbath

`X_swap4_K17_qhot032_m512_s4` 已完成并同步到本地：

- `data/3d_toric_code/with_measurement_noise/exp36/swap_sweep_probe_20260528/X_swap4_K17_qhot032_m512_s4/X_swap4_K17_qhot032_m512_s4.npz`

V/W/X 共同参数：

- `L=6,p=0.05,q=0.08`
- `num_disorder_samples_total=1`
- `num_start_chains=4`
- `num_measurements_per_disorder=512`
- `num_sweeps_between_measurements=6`
- `num_burn_in_sweeps=150`
- `max_effective_num_burn_in_sweeps=750`
- `K=17,q_hot=0.32`
- `pt_sector_diagnostic_stride=4`

结果：

| config | swap sweeps | min swap | bottleneck pair | cold flips | hot flips mean | proxy delivery | strict delivery | roundtrip sum | roundtrip per chain | transport samples | wall mean |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|---|---:|
| V | 1 | 0.167844 | 12 | `[0,0,0,0]` | 97.25 | 19 | 0 | 95 | `[22,26,25,22]` | `[3073]*4` | 36.70s |
| W | 2 | 0.190999 | 13 | `[0,0,0,0]` | 93.75 | 23 | 0 | 150 | `[37,28,39,46]` | `[6145]*4` | 35.46s |
| X | 4 | 0.186846 | 12 | `[0,0,0,0]` | 99.25 | 20 | 0 | 140 | `[37,37,34,32]` | `[12289]*4` | 89.20s |

结论：

- 增加 `pt_swap_sweeps_per_attempt` 能增加 transport sampling，但没有产生任何 cold sector flip，也没有 strict hot-to-cold sector-change delivery。
- `swap_sweeps=4` 的 wall time 明显增加，roundtrip 反而没有相对 `swap_sweeps=2` 单调增加；继续加 swap sweep 不是主方向。
- 当前 bottleneck 不再是简单的 temperature-index diffusion，而是 sector-changing configuration 在冷却到 cold ensemble 时被压回原 logical sector。

物理与程序判断：

- 现有代码里已有直接 logical-sector changing move，即 `winding_moves`。
- 对 3D toric code，`winding_moves` 是整张非平庸 sheet，权重为 `L^2`；`L=6` 时每次翻 36 条边。
- 在 cold `p=0.05`，接受率只有 `O(1e-5)`，所以“直接翻 sector”虽然存在，但几乎不可能在 cold 端发生。

新增程序方案：

- 新增默认关闭的 `winding_plane_heatbath_sweeps` / CLI `--winding-plane-heatbath-sweeps`。
- 该更新把同一方向的所有平行 winding sheet 组成一个小闭子群，对其 `2^L` 个组合做 exact heatbath 抽样。
- 目标分布不变：这是对当前状态沿 winding-plane orbit 的条件分布精确采样，不是无权重地强行改 sector。
- 该 move 允许一次同时翻多个平行 sheet，测试目的不是降低 cold 端自由能代价，而是让中温/热端的 nontrivial sector 子空间更充分热化，再看 PT 是否能把 sector change 带回 cold。
- 输出新增：
  - `winding_plane_heatbath_sweeps`
  - `chain_pt_winding_plane_heatbath_changed_count_per_temperature_per_disorder_per_start_replica_tensor`
  - `chain_pt_winding_plane_heatbath_attempted_count_per_temperature_per_disorder_per_start_replica_tensor`
  - `mean_pt_winding_plane_heatbath_changed_count_per_temperature_curve_tensor`

验证：

- `python -m py_compile src/main.py src/mcmc_parallel_tempering.py src/production_chunked_scan.py` 通过。
- `PYTHONPATH=src python -m unittest discover -s tests` 通过，6 tests。
- 本地 production smoke：
  - `data/3d_toric_code/with_measurement_noise/exp36/winding_heatbath_smoke_20260528/winding_heatbath_smoke.npz`
  - `winding_plane_heatbath_sweeps=1`
  - heatbath changed count `[0,0,8]`，attempted count `[108,108,108]`
  - sector flip count `[0,0,3]`

本地 L=4 快速对照：

| config | heatbath sweeps | cold flips | hot flips mean | strict delivery | min swap | cold heatbath changed | hot heatbath changed | wall |
|---|---:|---|---:|---:|---:|---|---|---|
| HB0 | 0 | `[0,0]` | 23.5 | 0 | 0.1190 | `[0,0]` | `[0,0]` | ~0.91s |
| HB1 | 1 | `[0,0]` | 23.0 | 0 | 0.1458 | `[0,0]` | `[146,128]` | ~1.15s |

解释：

- heatbath 在热端确实大量改变 winding plane，但本地短 L=4 probe 仍没有 cold sector flip。
- 这说明它不是直接修复 cold 自身势垒的 move；需要远端 L=6 比较 `q_hot=0.32/0.35` 下 strict delivery 是否改善。

下一轮远端 probe：

| config | node | q_hot | heatbath sweeps | swap sweeps | purpose |
|---|---|---:|---:|---:|---|
| Y | nd-1 | 0.32 | 0 | 1 | 同代码新基线 |
| Z | nd-2 | 0.32 | 1 | 1 | 测 heatbath 是否改善 strict delivery |
| AA | nd-3 | 0.35 | 1 | 1 | 稍热端配合 heatbath 的折中测试 |

共同参数沿用 V/W/X：`L=6,p=0.05,q=0.08,K=17,m=512,stride=4,disable_cluster_update`。

### winding-plane heatbath 远端结果与 cluster-q ladder 修复

Y/Z/AA 已完成并同步到本地：

- `data/3d_toric_code/with_measurement_noise/exp36/winding_heatbath_probe_20260528/Y_hb0_K17_qhot032_m512_s4/Y_hb0_K17_qhot032_m512_s4.npz`
- `data/3d_toric_code/with_measurement_noise/exp36/winding_heatbath_probe_20260528/Z_hb1_K17_qhot032_m512_s4/Z_hb1_K17_qhot032_m512_s4.npz`
- `data/3d_toric_code/with_measurement_noise/exp36/winding_heatbath_probe_20260528/AA_hb1_K17_qhot035_m512_s4/AA_hb1_K17_qhot035_m512_s4.npz`

共同参数：

- `L=6,p=0.05,q=0.08`
- `num_disorder_samples_total=1`
- `num_start_chains=4`
- `num_measurements_per_disorder=512`
- `num_sweeps_between_measurements=6`
- `num_burn_in_sweeps=150`
- `max_effective_num_burn_in_sweeps=750`
- `K=17`
- `pt_sector_diagnostic_stride=4`
- `adaptive_pt_rounds=0`
- `cluster_update=False`

结果：

| config | q_hot | heatbath sweeps | min swap | bottleneck pair | cold flips | hot flips mean | strict delivery | proxy delivery | roundtrip sum | roundtrip per chain | heatbath hot changed mean | wall mean |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|---:|---:|
| Y | 0.32 | 0 | 0.196886 | 12 | `[0,0,0,0]` | 97.75 | 0 | 21 | 101 | `[31,24,23,23]` | 0.00 | 43.50s |
| Z | 0.32 | 1 | 0.210230 | 12 | `[0,0,0,0]` | 96.50 | 0 | 26 | 116 | `[23,27,36,30]` | 195.75 | 64.10s |
| AA | 0.35 | 1 | 0.123888 | 11 | `[0,0,0,0]` | 95.75 | 0 | 17 | 80 | `[16,22,22,20]` | 997.75 | 165.67s |

bottleneck 对应温度：

- Y/Z: pair `12-13`，`(p,q)=(0.228076,0.266725)->(0.243638,0.280969)`。
- AA: pair `11-12`，`(p,q)=(0.241217,0.278765)->(0.258878,0.294745)`。

定量结论：

- heatbath 在热端确实强烈改变 winding-plane 子空间：Z hot changed mean `195.75`，AA hot changed mean `997.75`。
- 但 cold sector flips 仍全为 0，strict delivery 仍全为 0；因此它没有把热端/中温 sector 热化转化为 cold logical sector 改变。
- q_hot=0.35 配 heatbath 的代价明显偏大，wall mean `165.67s`，roundtrip 反而少于 q_hot=0.32。
- 当前证据支持：主要瓶颈不是热端没有 sector-changing move，也不是 replica 不能 round trip，而是回到 cold ensemble 时 nontrivial sector 被自由能壁垒压回原 sector。

下一步程序改动：

- 修复 q>0 cluster update，使其支持随温度变化的 `q_k` ladder。
- 原 cluster update 只接收标量 `q`，因此 production 对 `sync_enlarge` 强制 `--disable-cluster-update`；现在 `build_cluster_controller` 接收 `syndrome_error_probability_ladder`，每个温度用自己的 `p_k,q_k` 构造 active pins/checks。
- 该 move 保持目标分布不变，物理作用是给每个温度加入全局 FK/cluster 型大尺度更新，可能比固定 winding sheet 更容易产生跨 logical sector 的大块变化。
- 限制：该 cluster 公式需要所有 `p_k,q_k<0.5`，因此本轮只测试 `q_hot=0.32/0.35`；若未来热端越过 0.5，仍需显式关闭 cluster 或另写适用于高温端的 move。

验证：

- `python -m py_compile src/cluster_update.py src/mcmc_parallel_tempering.py src/main.py src/production_chunked_scan.py` 通过。
- `PYTHONPATH=src python -m unittest discover -s tests` 通过，7 tests。
- 新增测试 `tests/test_cluster_q_ladder.py` 验证 `sync_enlarge` + cluster 可以运行并产生 cluster attempts。
- 本地 production smoke：
  - `data/3d_toric_code/with_measurement_noise/exp36/cluster_sync_smoke_20260528/cluster_sync_smoke.npz`
  - `cluster_update_config_enabled=True`
  - `cluster_update_enabled=True`
  - `cluster_num_attempts=1`
  - `pt_syndrome_error_probability_ladder=[0.08,0.13938655,0.20475691,0.26672472,0.32]`

下一轮远端 probe：

| config | node | q_hot | cluster rho | heatbath sweeps | purpose |
|---|---|---:|---:|---:|---|
| AB | nd-1 | 0.32 | 0.05 | 0 | cluster-q ladder 基线测试 |
| AC | nd-2 | 0.32 | 0.20 | 0 | 提高 cluster 预算是否增加 sector 改变 |
| AD | nd-3 | 0.35 | 0.05 | 0 | 稍热端配合 cluster 的折中测试 |

共同参数沿用 Y/Z/AA，但不传 `--disable-cluster-update`。

### cluster-q ladder 与中间热端 probe 结果

代码提交：

- `d000b8c46 Support cluster update on sync PT ladders`

远端 cluster probe 已完成并同步：

- `data/3d_toric_code/with_measurement_noise/exp36/cluster_probe_20260528/AB_cluster_K17_qhot032_rho005_m512_s4/AB_cluster_K17_qhot032_rho005_m512_s4.npz`
- `data/3d_toric_code/with_measurement_noise/exp36/cluster_probe_20260528/AC_cluster_K17_qhot032_rho020_m512_s4/AC_cluster_K17_qhot032_rho020_m512_s4.npz`
- `data/3d_toric_code/with_measurement_noise/exp36/cluster_probe_20260528/AD_cluster_K17_qhot035_rho005_m512_s4/AD_cluster_K17_qhot035_rho005_m512_s4.npz`

远端中间热端 probe 已完成并同步：

- `data/3d_toric_code/with_measurement_noise/exp36/mid_qhot_probe_20260528/AE_static_K17_qhot038_m512_s4/AE_static_K17_qhot038_m512_s4.npz`
- `data/3d_toric_code/with_measurement_noise/exp36/mid_qhot_probe_20260528/AF_static_K17_qhot040_m512_s4/AF_static_K17_qhot040_m512_s4.npz`

共同参数：

- `L=6,p=0.05,q=0.08`
- `num_disorder_samples_total=1`
- `num_start_chains=4`
- `num_measurements_per_disorder=512`
- `num_sweeps_between_measurements=6`
- `num_burn_in_sweeps=150`
- `max_effective_num_burn_in_sweeps=750`
- `K=17`
- `pt_sector_diagnostic_stride=4`
- `adaptive_pt_rounds=0`

结果汇总：

| config | q_hot | cluster | heatbath | min swap | bottleneck pair | cold flips | hot flips mean | strict delivery | proxy delivery | roundtrip sum | hot winding acc | wall mean |
|---|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| Y | 0.32 | off | 0 | 0.196886 | 12 | `[0,0,0,0]` | 97.75 | 0 | 21 | 101 | 0.00372 | 43.50s |
| AB | 0.32 | rho=0.05 | 0 | 0.192962 | 12 | `[0,0,0,0]` | 99.25 | 0 | 24 | 105 | 0.00372 | 40.77s |
| AC | 0.32 | rho=0.20 | 0 | 0.174778 | 13 | `[0,0,0,0]` | 94.00 | 0 | 17 | 97 | 0.00372 | 38.14s |
| AA | 0.35 | off | 1 | 0.123888 | 11 | `[0,0,0,0]` | 95.75 | 0 | 17 | 80 | 0.01972 | 165.67s |
| AD | 0.35 | rho=0.05 | 0 | 0.153061 | 10 | `[2,0,0,0]` | 100.50 | 0 | 19 | 87 | 0.02215 | 139.25s |
| AE | 0.38 | off | 0 | 0.073129 | 9 | `[0,0,0,0]` | 90.00 | 0 | 11 | 41 | 0.07420 | 66.44s |
| AF | 0.40 | off | 0 | 0.067896 | 8 | `[0,0,0,0]` | 97.50 | 0 | 13 | 39 | 0.14256 | 77.49s |
| R | 0.44 | off | 0 | 0.013605 | 6 | `[0,0,0,0]` | 91.50 | 0 | 3 | - | 0.38510 | 222.78s |

cluster 诊断：

| config | cluster attempts | nonzero moves | wall fraction | 主要非零温度范围 | 备注 |
|---|---:|---:|---:|---|---|
| AB | 107 | 43 | 0.052 | `k=3..7` | move fraction 最大约 `0.0017`，没有 cold flips |
| AC | 399 | 146 | 0.171 | `k=3..9` | move fraction 最大约 `0.0064`，没有 cold flips |
| AD | 87 | 47 | 0.081 | `k=3..10,13` | `k=13` 单次 mean move fraction `0.140`；出现 cold flips `[2,0,0,0]` |

结论：

- `q_hot=0.38/0.40` 不是有效折中：hot winding acceptance 确实升高到 `0.074/0.143`，但 min swap 降到 `0.073/0.068`，roundtrip 只有 `41/39`，cold flips 和 strict delivery 仍全为 0。
- `q_hot=0.32` 的 cluster 尝试次数可以增加，但只产生很小的 cluster move，没有 cold-sector 改变。
- `q_hot=0.35 + cluster rho=0.05` 是目前唯一在本轮看到 cold-sector 改变的配置：`[2,0,0,0]`。但 strict delivery 仍为 0，且只发生在 1 条链上，不能视为充分 mixing。
- 当前最佳判断：若只靠现有 kernel move + PT，推荐的生产方向仍是 `q_hot=0.32` 保 transport；若以主动寻找 sector change 为目标，应继续围绕 `q_hot=0.35 + cluster` 做重复种子/更长链验证，而不是继续提高 `q_hot` 或使用 winding-plane heatbath。

下一轮建议：

1. 对 `q_hot=0.35 + cluster rho=0.05` 做 2 个独立 seed 重复，确认 AD 的 cold flips 是否可复现。
2. 小样本测试 `q_hot=0.35 + cluster rho=0.10/0.20`，看高温 cluster 大 move 次数是否能提高 cold flips，但要严格记录 wall time。
3. 如果 strict delivery 继续为 0，需要新增更细的 cluster-stage sector-change 诊断，区分 cold flips 是 cluster 本身、mid-temperature cluster 后 PT 输运，还是普通 local path 造成。

### 001/002 编号目录与 cluster-stage sector 诊断

目录规范更新：

- exp36 后续迭代实验使用编号子目录，避免 root 下继续堆叠无编号 probe。
- 本地 smoke: `001_cluster_stage_diag_smoke_20260528/`
- 下一轮远端实验: `002_cluster_stage_repeats_20260528/`

代码改动：

- 提交：`879c4261 Track cluster-stage sector changes`
- 在 PT sector diagnostics 打开时，cluster scheduler 每次真正选中一个温度准备执行 cluster update 前，只计算该温度的 logical-sector signature。
- cluster update 后再次计算同一温度 signature，统计：
  - `pt_cluster_sector_attempted_count_per_temperature`
  - `pt_cluster_sector_nonzero_count_per_temperature`
  - `pt_cluster_sector_changed_count_per_temperature`
- 对应穿透到 disorder-average、chunk 和 merge 输出：
  - `chain_pt_cluster_sector_attempted_count_per_temperature_per_disorder_per_start_replica(_tensor)`
  - `chain_pt_cluster_sector_nonzero_count_per_temperature_per_disorder_per_start_replica(_tensor)`
  - `chain_pt_cluster_sector_changed_count_per_temperature_per_disorder_per_start_replica(_tensor)`
  - `mean_pt_cluster_sector_*_count_per_temperature_curve_tensor`

验证：

- `python -m py_compile src/cluster_update.py src/mcmc_parallel_tempering.py src/main.py src/production_chunked_scan.py` 通过。
- `PYTHONPATH=src python -m unittest discover -s tests` 通过，7 tests。
- 本地 production smoke:
  - `data/3d_toric_code/with_measurement_noise/exp36/001_cluster_stage_diag_smoke_20260528/cluster_stage_diag_smoke.npz`
  - `chain_pt_cluster_sector_attempted... = [0,0,0,0,16]`
  - `chain_pt_cluster_sector_nonzero... = [0,0,0,0,16]`
  - `chain_pt_cluster_sector_changed... = [0,0,0,0,12]`
  - 说明新诊断能在最终 merge NPZ 中记录 cluster 本身造成的 logical-sector 改变。

002 远端计划：

| run | node | q_hot | cluster rho | seed | purpose |
|---|---|---:|---:|---:|---|
| run01 | nd-1 | 0.35 | 0.05 | 413000 | 复现 AD 的 cold flips |
| run02 | nd-2 | 0.35 | 0.05 | 414000 | 独立 seed 复现 |
| run03 | nd-3 | 0.35 | 0.10 | 415000 | 检查增加 cluster 预算是否提高 sector flips |

共同参数：`L=6,p=0.05,q=0.08,K=17,m=512,stride=4,num_start_chains=4,adaptive_pt_rounds=0`。

### 002 cluster-stage repeats 最终结果

输出文件：

- `data/3d_toric_code/with_measurement_noise/exp36/002_cluster_stage_repeats_20260528/002_summary.json`
- `data/3d_toric_code/with_measurement_noise/exp36/002_cluster_stage_repeats_20260528/002_summary.md`

共同参数：

- `L=6,p=0.05,q=0.08`
- `K=17,q_hot=0.35`
- `num_disorder_samples_total=1`
- `num_start_chains=4`
- `num_measurements_per_disorder=512`
- `num_sweeps_between_measurements=6`
- `num_burn_in_sweeps=150`
- `max_effective_num_burn_in_sweeps=750`
- `pt_sector_diagnostic_stride=4`
- `adaptive_pt_rounds=0`
- `winding_repeat_factor=1`
- `winding_plane_heatbath_sweeps=0`

结果汇总：

| run | rho | min swap | bottleneck pair | cold flips | hot flips mean | strict delivery | proxy delivery | roundtrip sum | cluster nonzero | cluster-stage sector changes |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| AD previous | 0.05 | 0.153061 | 10 | `[2,0,0,0]` | 100.50 | 0 | 19 | 87 | 47 | - |
| run01 | 0.05 | 0.129513 | 11 | `[0,0,0,0]` | 94.75 | 0 | 14 | 79 | 38 | 0 |
| run02 | 0.05 | 0.121795 | 10 | `[0,0,0,0]` | 96.50 | 0 | 12 | 67 | 26 | 0 |
| run03 | 0.10 | 0.167321 | 10 | `[10,4,2,2]` | 100.50 | 0 | 18 | 76 | 77 | 9 |

cluster-stage 逐温度诊断：

- run01: sector-attempt `85`、nonzero `35`、sector-changed `0`；主要发生在 `k=5,6`。
- run02: sector-attempt `100`、nonzero `23`、sector-changed `0`；主要发生在 `k=1,3,4,5,6`。
- run03: sector-attempt `88`、nonzero `63`、sector-changed `9`；changed by temperature 为 `[0,0,0,0,0,1,0,2,1,1,0,1,0,1,2,0,0]`。

结论：

- 两条新的 `rho=0.05` repeat 均没有 cold logical-sector flip，也没有 cluster-stage sector change；因此 AD 的 `[2,0,0,0]` 不是稳定可复现信号。
- `rho=0.10` 的 run03 是当前最强 positive signal：四条 chain 都发生 cold sector flip，总数 `18`，并且新诊断直接记录到 `9` 次 cluster update 本身造成 logical-sector change。
- strict hot-to-cold sector-change delivery 仍为 `0`，说明这不是简单的“热端 winding sector change 被 PT 带回 cold”。更合理的图像是：`q_hot=0.35,rho=0.10` 下 cluster 在中温区产生跨 sector 大更新，再经后续 PT/局部更新改变 cold 槽的 logical sector。
- 当前还不能认为 mixing 充分。下一步应围绕 `q_hot=0.35,rho≈0.10` 做独立 seed repeat，并小幅扫描 `rho=0.15`，判断 run03 是否可复现以及 cluster 预算是否继续带来收益。

### 003 cluster-rho refinement 计划

目的：验证 002 run03 是否可复现，并判断 `rho=0.10` 到 `0.15` 的 cluster 预算增加是否继续提高 cold logical-sector flip。

计划目录：

- `data/3d_toric_code/with_measurement_noise/exp36/003_cluster_rho_refine_20260528/`

远端配置：

| run | node | q_hot | cluster rho | seed | purpose |
|---|---|---:|---:|---:|---|
| run01 | nd-1 | 0.35 | 0.10 | 416000 | 独立 seed 复现 002 run03 |
| run02 | nd-2 | 0.35 | 0.10 | 417000 | 第二个独立 seed 复现 |
| run03 | nd-3 | 0.35 | 0.15 | 418000 | 小幅提高 cluster 预算，检查收益与成本 |

共同参数沿用 002：`L=6,p=0.05,q=0.08,K=17,m=512,stride=4,num_start_chains=4,adaptive_pt_rounds=0,winding_plane_heatbath_sweeps=0`。

启动状态：

- 代码同步到远端：`/home/DATA1/users/yuany/.single_shot/repos/003_cluster_rho_refine_20260528/source`。
- run base：`/home/DATA1/users/yuany/.single_shot/exp36/003_cluster_rho_refine_20260528`。
- launcher：`data/3d_toric_code/with_measurement_noise/exp36/003_cluster_rho_refine_20260528/launch_cluster_rho_refine_20260528.sh`。
- 三个任务均通过 quick exact validation、preflight chunk 和 preflight merge。
- 截至本记录，三个 screen 均在运行 chunk worker，尚未生成 final NPZ：
  - nd-1: `exp36_003_r1`
  - nd-2: `exp36_003_r2`
  - nd-3: `exp36_003_r3`

下一轮先检查三个 final NPZ；若完成，同步到 003 本地目录，生成 `003_summary.json/md` 并比较 `rho=0.10` repeat 与 `rho=0.15` 的 cold flips、cluster-stage sector changes、min swap、roundtrip 和 wall time。

### 003 cluster-rho refinement 最终结果

输出文件：

- `data/3d_toric_code/with_measurement_noise/exp36/003_cluster_rho_refine_20260528/003_summary.json`
- `data/3d_toric_code/with_measurement_noise/exp36/003_cluster_rho_refine_20260528/003_summary.md`

共同参数：

- `L=6,p=0.05,q=0.08`
- `K=17,q_hot=0.35`
- `num_disorder_samples_total=1`
- `num_start_chains=4`
- `num_measurements_per_disorder=512`
- `num_sweeps_between_measurements=6`
- `num_burn_in_sweeps=150`
- `max_effective_num_burn_in_sweeps=750`
- `pt_sector_diagnostic_stride=4`
- `adaptive_pt_rounds=0`

结果：

| run | rho | min swap | bottleneck pair | cold flips | hot flips mean | strict delivery | proxy delivery | roundtrip sum | cluster nonzero | cluster-stage sector changes | cluster wall fraction |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| run01 | 0.10 | 0.141549 | 11 | `[0,0,0,0]` | 97.00 | 0 | 17 | 74 | 94 | 1 | 0.095 |
| run02 | 0.10 | 0.129513 | 11 | `[0,0,0,0]` | 91.50 | 0 | 14 | 67 | 66 | 0 | 0.096 |
| run03 | 0.15 | 0.151361 | 10 | `[0,0,0,0]` | 96.00 | 0 | 13 | 65 | 102 | 19 | 0.145 |

逐温度 cluster-stage sector change：

- run01 `rho=0.10`: `[0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]`。
- run02 `rho=0.10`: 全 0。
- run03 `rho=0.15`: `[0,0,0,0,0,0,0,0,0,1,3,2,13,0,0,0,0]`，集中在 `k=9..12`。

结论：

- 两条新的 `rho=0.10` repeat 均没有 cold logical-sector flip；002 run03 的 `[10,4,2,2]` 不是稳定可复现效果。
- `rho=0.15` 显著增加 cluster-stage sector change，但 cold flips 仍为 `[0,0,0,0]`。这说明中温 sector change 可以被制造出来，但在冷却/输运到 cold ensemble 前大多丢失。
- 单纯增大 cluster 预算不是稳健修复，且 `rho=0.15` 的 chunk wall time 约 `600s`，明显慢于 `rho=0.10` 的约 `221-233s`。
- 当前瓶颈已进一步定位为：`k≈9..12` 附近会发生 sector-changing cluster move，但这些改变不能稳定传到 `k=0`。下一步应优化 PT/cooling 机制，而不是继续增加 rho。

### 004 建议方向

目的：增强中温 sector change 向 cold 的保留与输运。

候选方向：

1. 在 `q_hot=0.35,rho=0.15` 的基础上增加 `pt_swap_sweeps_per_attempt=2`，测试更多相邻 swap 是否能把 `k=9..12` 的 sector change 更快带向 cold。此前 `q_hot=0.32` 上 swap sweep 没用，但那时没有大量 cluster-stage sector change；现在物理条件不同。
2. 测试更长生产期 `m=1024`、较低诊断频率 `stride=8`，判断 cold flip 是否只是等待时间不足；保留 cluster-stage 诊断。
3. 如果代码允许，下一步应实现 identity-tracked sector-change delivery：记录“某个 replica 在中温 cluster 后 sector 改变，随后是否到达 cold 且保持改变”，直接量化冷却保留率。

### 004 cluster cold-delivery 诊断实现与计划

代码改动：

- 提交：`4e5832423 Track cluster sector cold delivery`
- 在已有 cluster-stage sector change 诊断基础上，新增 identity-tracked cold-delivery 统计。
- 当某个温度 `k>0` 的 cluster update 直接改变 logical-sector signature 时，记录当前 temperature slot 上的 `replica_id`、change 前后 signature 和 origin temperature。
- 该 replica 后续首次到达 cold slot `k=0` 时，按 cold signature 分类：
  - `cold_arrival`: 该 pending sector-change replica 到达 cold。
  - `cold_survived`: 到达 cold 时 signature 等于 cluster 后 signature。
  - `cold_reverted`: 到达 cold 时 signature 回到 cluster 前 signature。
  - `cold_other`: 到达 cold 时 signature 既不是 cluster 前，也不是 cluster 后。
  - `pending_overwritten`: 同一 replica 在到达 cold 前又发生新的 cluster-sector change，旧 pending 被覆盖。
  - `pending_remaining`: run 结束时仍未到达 cold 的 pending change。

新增输出字段：

- `pt_cluster_sector_cold_arrival_count_per_origin_temperature`
- `pt_cluster_sector_cold_survived_count_per_origin_temperature`
- `pt_cluster_sector_cold_reverted_count_per_origin_temperature`
- `pt_cluster_sector_cold_other_count_per_origin_temperature`
- `pt_cluster_sector_pending_overwritten_count_per_origin_temperature`
- `pt_cluster_sector_pending_remaining_count_per_origin_temperature`
- 以上字段也穿透到 disorder-average、chunk 和 production merge tensor/mean curve。

验证：

- `python -m py_compile src/mcmc_parallel_tempering.py src/main.py src/production_chunked_scan.py` 通过。
- `PYTHONPATH=src conda run --no-capture-output -n 12 python -m unittest discover -s tests` 通过，7 tests。
- 本地 production smoke：
  - `data/3d_toric_code/with_measurement_noise/exp36/004_cluster_delivery_diag_smoke_20260528/cluster_delivery_diag_smoke.npz`
  - final NPZ 中新增 `chain_pt_cluster_sector_cold_*_tensor` 和 `mean_pt_cluster_sector_cold_*_curve_tensor` 字段均存在，shape 为 `(1,1,1,1,1,5)` / `(1,1,5)`。

004 远端计划：

- 目录：`data/3d_toric_code/with_measurement_noise/exp36/004_cluster_cold_delivery_20260528/`
- 远端 source：`/home/DATA1/users/yuany/.single_shot/repos/004_cluster_cold_delivery_20260528/source`
- run base：`/home/DATA1/users/yuany/.single_shot/exp36/004_cluster_cold_delivery_20260528`

| run | node | rho | swap sweeps | measurements | stride | seed | purpose |
|---|---|---:|---:|---:|---:|---:|---|
| run01 | nd-1 | 0.15 | 1 | 512 | 4 | 419000 | 003 run03 同类复现，新诊断基线 |
| run02 | nd-2 | 0.15 | 2 | 512 | 4 | 420000 | 测试更多 swap 是否提高 cold arrival/survival |
| run03 | nd-3 | 0.15 | 1 | 1024 | 8 | 421000 | 测试更长生产期是否提高 cold arrival/survival |

共同参数：`L=6,p=0.05,q=0.08,K=17,q_hot=0.35,num_start_chains=4,adaptive_pt_rounds=0,winding_plane_heatbath_sweeps=0`。

### 004 cluster cold-delivery 最终结果

输出文件：

- `data/3d_toric_code/with_measurement_noise/exp36/004_cluster_cold_delivery_20260528/004_summary.json`
- `data/3d_toric_code/with_measurement_noise/exp36/004_cluster_cold_delivery_20260528/004_summary.md`

共同参数：

- `L=6,p=0.05,q=0.08`
- `K=17,q_hot=0.35`
- `cluster rho=0.15`
- `num_start_chains=4`
- `adaptive_pt_rounds=0`
- `winding_plane_heatbath_sweeps=0`

结果：

| run | swap sweeps | measurements | stride | min swap | bottleneck pair | cold flips | hot flips mean | roundtrip sum | cluster nonzero | changed | arrival | survived | reverted | other | pending |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| run01 | 1 | 512 | 4 | 0.104526 | 11 | `[0,0,0,0]` | 95.75 | 59 | 90 | 0 | 0 | 0 | 0 | 0 | 0 |
| run02 | 2 | 512 | 4 | 0.120552 | 10 | `[0,0,0,0]` | 93.50 | 82 | 98 | 0 | 0 | 0 | 0 | 0 | 0 |
| run03 | 1 | 1024 | 8 | 0.109806 | 10 | `[0,0,0,0]` | 94.50 | 143 | 215 | 38 | 30 | 13 | 10 | 7 | 2 |

run03 逐温度 cold-delivery 诊断：

- changed by temperature: `[0,0,0,0,0,0,0,0,0,0,2,0,30,2,2,2,0]`。
- arrival by origin: `[0,0,0,0,0,0,0,0,0,0,2,0,24,2,1,1,0]`。
- survived by origin: `[0,0,0,0,0,0,0,0,0,0,0,0,12,0,0,1,0]`。
- reverted by origin: `[0,0,0,0,0,0,0,0,0,0,2,0,7,1,0,0,0]`。

结论：

- 三条 run 的 cold logical-sector flips 均为 `[0,0,0,0]`，因此 `rho=0.15` 仍没有给出稳健 cold mixing。
- `swap_sweeps=2` 短链只把 roundtrip 从 `59` 提高到 `82`，但没有产生 cluster-stage sector change，也没有改善 cold flips。
- `m=1024` 长链给出更细的物理图像：中温 cluster sector change 确实可以被 PT 带到 cold，30/38 次在 run 内到达 cold，其中 13 次到达时仍保持 cluster 后 signature；但这些事件没有变成按当前 cold-slot sector 诊断可见的持久 cold flip。
- 下一步应追踪 cold arrival 后的驻留时间和 persistence：区分“到达 cold 后立即离开/未被诊断采到”、“在 cold 停留但后续 local update 回退”、“到达时 signature 与 cold-slot 观测定义/诊断 cadence 不一致”。继续单纯增加 cluster 预算或 swap sweep 的边际收益已经很弱。

### 005 cold-arrival persistence 诊断实现与计划

目的：解释 004 run03 中 `30` 次 cluster-sector change 到达 cold、`13` 次首次到达时仍 survived，但 cold-slot sector flip 仍为 0 的原因。

代码改动：

- 在 pending cluster-sector change 首次到达 cold 并完成 `arrival/survived/reverted/other` 分类后，不再只清空事件；额外记录一个 cold-dwell tracking window。
- 该 window 按 replica 追踪：origin temperature、cluster 前后 signature、cold 内最近一次 signature、cold 内 transport-position sample 数、是否被下一次 sector diagnostic 采到。
- 当该 replica 离开 cold，或 run 结束时仍在 cold，统计：
  - `pt_cluster_sector_cold_diagnostic_survived/reverted/other/missed_count_per_origin_temperature`：到达 cold 后第一次 sector diagnostic 看到的状态；若离开 cold 前没有诊断样本则记为 missed。
  - `pt_cluster_sector_cold_departure_survived/reverted/other_count_per_origin_temperature`：离开 cold 时的状态。
  - `pt_cluster_sector_cold_dwell_sample_sum/max_per_origin_temperature`：arrival 后在 cold 的 transport-position sample 数总和/最大值。
  - `pt_cluster_sector_cold_active_remaining_count_per_origin_temperature`：run 结束时仍在 cold dwell window 中的事件数。
- 这些字段已穿透到 disorder-average、chunk 和 production merge tensor/mean curve；旧 chunk 缺字段时 merge 仍填 0。
- 该改动只增加诊断输出，不改变 Markov chain proposal、acceptance 或 swap 决策。

验证：

- `python -m py_compile src/mcmc_parallel_tempering.py src/main.py src/production_chunked_scan.py` 通过。
- `PYTHONPATH=src conda run --no-capture-output -n 12 python -m unittest discover -s tests` 通过，7 tests；`tests/test_cluster_q_ladder.py` 已检查新增字段 shape。
- 本地 production smoke：
  - `data/3d_toric_code/with_measurement_noise/exp36/005_cold_persistence_diag_smoke_20260528/cold_persistence_diag_smoke.npz`
  - final NPZ 中新增 `chain_pt_cluster_sector_cold_diagnostic_*_tensor`、`chain_pt_cluster_sector_cold_departure_*_tensor`、`chain_pt_cluster_sector_cold_dwell_*_tensor` 与对应 mean curve 字段均存在，shape 为 `(1,1,1,1,1,5)` / `(1,1,5)`。
  - smoke 太短，没有 cold-arrival 事件，因此新增字段数值全 0 是预期。

005 远端计划：

- 目录：`data/3d_toric_code/with_measurement_noise/exp36/005_cold_persistence_probe_20260529/`
- 目标：重复 004 的 `rho=0.15,q_hot=0.35` 情形，但用新 persistence 字段定量区分 cold arrival 后是否被 diagnostic 采到、离开 cold 前是否已经回退。

| run | node | rho | swap sweeps | measurements | stride | seed | purpose |
|---|---|---:|---:|---:|---:|---:|---|
| run01 | nd-1 | 0.15 | 1 | 1024 | 4 | 422000 | 比 004 run03 更细诊断频率，追踪 arrival 后是否被采到 |
| run02 | nd-2 | 0.15 | 2 | 1024 | 4 | 423000 | 检查更多 swap 是否缩短或延长 cold dwell，并影响 departure 状态 |
| run03 | nd-3 | 0.15 | 1 | 2048 | 8 | 424000 | 增加等待时间，测试 arrival/persistence 统计是否积累到更稳定信号 |

共同参数：`L=6,p=0.05,q=0.08,K=17,q_hot=0.35,num_start_chains=4,adaptive_pt_rounds=0,winding_plane_heatbath_sweeps=0,cluster rho=0.15`。

启动状态：

- 代码提交并推送：`23621bd42 Track cluster cold persistence diagnostics`。
- launcher 提交并推送：`03472e18d Plan exp36 cold persistence probes`。
- 远端 source：`/home/DATA1/users/yuany/.single_shot/repos/005_cold_persistence_probe_20260529/source`。
- run base：`/home/DATA1/users/yuany/.single_shot/exp36/005_cold_persistence_probe_20260529`。
- launcher：`data/3d_toric_code/with_measurement_noise/exp36/005_cold_persistence_probe_20260529/launch_cold_persistence_probe_20260529.sh`。
- 三条任务均已通过 quick exact validation、preflight chunk 和 preflight merge，并进入 `Launching chunk workers: 1 workers for 1 chunks`。
- screen：nd-1 `exp36_005_r1`，nd-2 `exp36_005_r2`，nd-3 `exp36_005_r3`。

下一轮先检查三个 final NPZ；若完成，同步到本地 005 目录并生成 `005_summary.json/md`，重点比较：cold flips、cluster changed/arrival/survived/reverted、diagnostic survived/reverted/missed、departure survived/reverted/other、cold dwell sample sum/max、roundtrip 和 wall time。

### 005 cold-arrival persistence 最终结果

输出文件：

- `data/3d_toric_code/with_measurement_noise/exp36/005_cold_persistence_probe_20260529/005_summary.json`
- `data/3d_toric_code/with_measurement_noise/exp36/005_cold_persistence_probe_20260529/005_summary.md`

结果：

| run | swap | m | stride | min swap | cold flips | roundtrip | changed | arrival | arr survived | arr reverted | diag survived | diag missed | dep survived | dep reverted | dep other | dwell sum/max |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| run01 | 1 | 1024 | 4 | 0.142370 | `[0,2,0,0]` | 180 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 4/4 |
| run02 | 2 | 1024 | 4 | 0.122715 | `[0,0,0,0]` | 201 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0/0 |
| run03 | 1 | 2048 | 8 | 0.121453 | `[0,0,0,0]` | 344 | 131 | 74 | 20 | 26 | 2 | 68 | 20 | 26 | 28 | 180/6 |

逐温度要点：

- run01: changed/arrival 都只在 origin `k=6` 有 1 次；到达 cold 时已 reverted，离开 cold 时仍 reverted。
- run02: 没有 cluster-stage sector change。
- run03: changed by temperature 为 `[0,0,0,0,0,0,0,0,0,0,5,20,2,0,104,0,0]`，arrival by origin 为 `[0,0,0,0,0,0,0,0,0,0,4,17,1,0,52,0,0]`，主要来自 `k=14`，其次 `k=11`。
- run03: diagnostic survived by origin 为 `[0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0]`，diagnostic missed 为 `[0,0,0,0,0,0,0,0,0,0,4,15,1,0,48,0,0]`。
- run03: departure survived/reverted/other by origin 分别为 `[0,0,0,0,0,0,0,0,0,0,1,7,0,0,12,0,0]`、`[0,0,0,0,0,0,0,0,0,0,2,7,1,0,16,0,0]`、`[0,0,0,0,0,0,0,0,0,0,1,3,0,0,24,0,0]`。

结论：

- 005 证明 004 的 `survived-at-arrival` 不是纯诊断假象：run03 中 20/74 次到达 cold 时确实保持 cluster 后 signature。
- 但 cold persistence 很弱：74 次 cold arrival 中只有 2 次赶上下一次 sector diagnostic 且仍 survived，68 次 missed；cold dwell sample 总数 180，平均约 2.43，最大也只有 6。
- 离开 cold 时仍 survived 的有 20 次，但 reverted 和 other 合计 54 次；同时 cold-slot sector flips 仍为 `[0,0,0,0]`。这说明 arrival 事件多数太短，或者到达/离开 cold 的 signature 变化没有转化成按 measurement cadence 统计的稳定 cold sector sampling。
- 下一步若继续优化 mixing，应从“增加 cold dwell / 降低 cold departure 后回退概率 / 让 cold slot 在 arrival 后多做可记录测量”入手，而不是继续只增加热端 sector proposal、swap sweep 或 cluster 预算。可测试的方向包括：arrival 后的 cold hold/measurement 诊断、减慢 swap cadence 对 persistence 的影响、或在 cold 附近加密/重排 ladder 以增加接近 cold 的停留时间。

### 006 cold-dwell schedule probe 计划

物理动机：005 中 run03 有 74 次 cold arrival，但 cold dwell sample 总数只有 180，平均约 2.43，最大 6；68/74 次错过下一次 sector diagnostic。006 不改变目标分布，只改变调度，测试 cold arrival 是否只是被当前 measurement/diagnostic cadence 漏掉。

目录：`data/3d_toric_code/with_measurement_noise/exp36/006_cold_dwell_schedule_probe_20260529/`。

共同参数：`L=6,p=0.05,q=0.08,K=17,q_hot=0.35,cluster rho=0.15,num_start_chains=4,adaptive_pt_rounds=0,winding_plane_heatbath_sweeps=0`。

| run | node | swap cadence | sweeps/measurement | measurements | stride | seed | purpose |
|---|---|---:|---:|---:|---:|---:|---|
| run01 | nd-1 | 1 | 6 | 1024 | 1 | 425000 | 同 005 run01 生产长度，但每个 measurement 都做 sector diagnostic，判断 missed 是否主要来自 stride |
| run02 | nd-2 | 6 | 6 | 1024 | 1 | 426000 | 降低 swap cadence，让 arrival 后更可能在 cold 停到下一次 measurement |
| run03 | nd-3 | 1 | 1 | 2048 | 1 | 427000 | 高频 measurement，直接检查 transient arrival 是否进入 measured cold history |

判据：

- 若 stride=1 或 measurement 高频后 cold flips 明显增加，而 departure 统计仍显示 arrival 很短，说明主要是 measurement/diagnostic cadence 漏掉 transient。
- 若降低 swap cadence 增加 dwell sum/max、diag survived 和 cold flips，则后续可考虑 cold dwell/hold 类型调度优化。
- 若三者仍无 cold flips，则瓶颈不是测量 cadence，而是 cold ensemble 本身不支持这些 arrived sector 稳定存在；下一步应考虑更物理的 near-cold ladder/cluster move 或重新评估 observable sector 定义。

启动状态：

- 提交并推送：`71c09c16b Plan exp36 cold dwell schedule probes`。
- 远端 source：`/home/DATA1/users/yuany/.single_shot/repos/006_cold_dwell_schedule_probe_20260529/source`。
- run base：`/home/DATA1/users/yuany/.single_shot/exp36/006_cold_dwell_schedule_probe_20260529`。
- launcher：`data/3d_toric_code/with_measurement_noise/exp36/006_cold_dwell_schedule_probe_20260529/launch_cold_dwell_schedule_probe_20260529.sh`。
- 三条任务均已通过 quick exact validation、preflight chunk 和 preflight merge，并进入 `Launching chunk workers: 1 workers for 1 chunks`。
- screen：nd-1 `exp36_006_r1`，nd-2 `exp36_006_r2`，nd-3 `exp36_006_r3`。

下一轮先检查三个 final NPZ；若完成，同步到本地 006 目录，生成 `006_summary.json/md`，重点比较 cold flips、arrival 的 diagnostic missed 比例、dwell sample sum/max、departure 状态、roundtrip 和 wall time。

### 007 cold-edge hold 调度实现与计划

006 截至本轮检查时：run01/run02 已完成，run03 `run03_measure1_m2048_seed427000` 仍在 nd-3 运行，CPU 正常，无 final NPZ。因此 006 尚不能给最终结论；已有 partial 结果仍是：run01 `stride=1` 无 cluster-stage sector change、cold flips `[0,0,0,0]`；run02 `swap cadence=6` 有 `changed=4,arrival=2,diag survived/reverted=1/1`，但 roundtrip 降到 `30`，cold flips 仍为 `[0,0,0,0]`。

物理动机：005 run03 显示 `74` 次 cold arrival 里只有 `2` 次赶上下一次 sector diagnostic，cold dwell 平均约 `2.43`、最大 `6`。006 run02 用全局降低 swap cadence 的方法虽然让 arrival 被 diagnostic 捕获，但也把 roundtrip 明显压低。更有针对性的调度是只降低 cold edge `(k=0,k=1)` 的 swap 频率，让到达 cold 的 replica 多停留，同时不直接降低中温/热端的 temperature-index diffusion。

代码改动：新增 `pt_cold_edge_swap_stride` / CLI `--pt-cold-edge-swap-stride`。

- `N=1` 保持旧行为。
- `N>1` 时，当 alternating swap sweep 本来会尝试 pair `0-1`，只每 `N` 次 eligible sweep 尝试一次；其它温度对仍按原 cadence 尝试。
- 这个调度固定且与 Markov 状态无关；每个被执行的 swap 仍使用原 Metropolis ratio，因此不改变目标分布，只改变不同局部 reversible kernels 的施加频率。
- 生产 chunk、manifest 和 final NPZ 均保存 `pt_cold_edge_swap_stride`。

验证：

- `PYTHONPATH=src conda run --no-capture-output -n 12 python -m py_compile src/mcmc_parallel_tempering.py src/main.py src/production_chunked_scan.py` 通过。
- `PYTHONPATH=src conda run --no-capture-output -n 12 python -m unittest discover -s tests` 通过，8 tests。
- 本地 production smoke：`data/3d_toric_code/with_measurement_noise/exp36/007_cold_edge_hold_probe_20260529/local_cold_edge_stride_smoke/local_cold_edge_stride_smoke.npz`。
  - `pt_cold_edge_swap_stride=2`。
  - pair attempts 为 `[2,4,4,4]`，确认只稀释 cold edge，其它 pair 未稀释。

007 远端计划：

目录：`data/3d_toric_code/with_measurement_noise/exp36/007_cold_edge_hold_probe_20260529/`。

共同参数：`L=6,p=0.05,q=0.08,K=17,q_hot=0.35,cluster rho=0.15,num_start_chains=4,adaptive_pt_rounds=0,winding_plane_heatbath_sweeps=0,num_measurements=1024,num_sweeps_between_measurements=6,pt_sector_diagnostic_stride=1`。

| run | node | cold-edge stride | seed | purpose |
|---|---|---:|---:|---|
| run01 | nd-1 | 1 | 428000 | 新代码基线，对照 006 run01 |
| run02 | nd-2 | 2 | 429000 | 轻度增加 cold dwell |
| run03 | nd-3 | 4 | 430000 | 更强 cold hold，观察 transport 损伤 |

判据：

- 若 stride `2/4` 明显增加 `cold_dwell_sample_sum/max`、`diagnostic survived` 和 cold flips，同时 roundtrip 仍可接受，则 cold-edge dwell 是有效调度方向。
- 若 roundtrip 明显下降且 cold flips 仍为 0，则说明简单 cold hold 不能解决 cold persistence，应改为 near-cold ladder 加密或重新设计中温 sector change 到 cold 的保留机制。

启动状态：

- 代码提交并推送：`f05c1a818 Add exp36 cold-edge PT hold scheduling`。
- launcher 提交并推送：`a3dccc37e Record exp36 cold-edge hold launch`。
- 远端 source：`/home/DATA1/users/yuany/.single_shot/repos/007_cold_edge_hold_probe_20260529/source`。
- run base：`/home/DATA1/users/yuany/.single_shot/exp36/007_cold_edge_hold_probe_20260529`。
- launcher：`data/3d_toric_code/with_measurement_noise/exp36/007_cold_edge_hold_probe_20260529/launch_cold_edge_hold_probe_20260529.sh`。
- 三条任务均通过 quick exact validation、preflight chunk 和 preflight merge，并进入 `Launching chunk workers: 1 workers for 1 chunks`。
- screen：nd-1 `exp36_007_r1`，nd-2 `exp36_007_r2`，nd-3 `exp36_007_r3`。

下一轮先检查 006 run03 与 007 三条 final NPZ；若完成，分别同步并生成 `006_summary.json/md` 与 `007_summary.json/md`。007 重点比较 cold-edge stride 对 cold flips、cluster changed/arrival、diagnostic missed、dwell sum/max、departure 状态和 roundtrip 的影响。
