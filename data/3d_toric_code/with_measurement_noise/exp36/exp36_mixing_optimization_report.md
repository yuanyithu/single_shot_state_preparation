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
