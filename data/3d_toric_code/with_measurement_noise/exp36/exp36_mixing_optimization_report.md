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
