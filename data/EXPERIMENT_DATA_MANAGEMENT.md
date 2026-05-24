# Experiment Data Management

Updated: 2026-05-24

This repository keeps experiment code, README files, and text notes in Git. Generated experiment data are local artifacts and must not be uploaded to GitHub. The current policy is conservative local cleanup plus Git index cleanup only; do not rewrite Git history unless a separate repository slimming task explicitly calls for `git filter-repo` or BFG.

## Git Policy

- Do not track generated data: `*.npz`, `*.npy`, `*.json`, `*.csv`, `*.tsv`, `*.log`, `*.png`, `*.pdf`, `chunks/`, `remote_run/`, `remote_runs/`, `pooled/`, or `analysis/`.
- Do not track data-local launcher/preflight files such as `data/**/*.sh`, `data/**/*.txt`, or nested `data/**/.gitignore`; keep durable notes in Markdown instead.
- Keep `README.md` files and this management note in Git.
- Use `git rm -r --cached data` to stop tracking old generated files without deleting local copies, then re-add only Markdown/text documentation that should remain versioned.
- Do not use `git add .` for data governance commits.

## Retention Levels

- **P0 paper-critical local archive**: keep locally; do not upload generated artifacts.
- **P1 paper/background summary**: keep README/final summary locally; generated raw data may be removed.
- **P2 historical diagnostic**: keep README/text summary locally; raw data and plots may be removed after conclusions are reflected in reports.
- **P3 covered or scout raw**: raw/intermediate artifacts can be removed.

## Directory Inventory

Sizes are from `du -sh` on 2026-05-24 before cleanup.

### 2D toric code

| Directory | Size | Purpose | Retention | Raw data policy |
|---|---:|---|---|---|
| `2d_toric_code/with_measurement_noise/measurement_noise_overnight_nd3_20260421_004035` | 65M | Early q-positive measurement-noise overnight scan | P2 | Remove chunks/raw; keep README if needed. |
| `2d_toric_code/with_measurement_noise/measurement_noise_threshold_search_nd3_20260421_104427` | 27M | Early threshold-search scan | P2 | Remove chunks/raw; keep README if needed. |
| `2d_toric_code/with_measurement_noise/no_threshold_final_nd3_20260421_225039` | 3.5M | 2D no-threshold final evidence | P1 | Keep final README/summary locally; raw generated files are not Git-tracked. |
| `2d_toric_code/with_measurement_noise/no_threshold_evidence_nd3_20260422` | 12M | 2D no-threshold evidence aggregation | P1 | Keep final README/summary locally; raw generated files are not Git-tracked. |
| `2d_toric_code/without_measurement_noise/baseline_multisize_local` | 388K | Local baseline smoke/baseline scan | P2 | Remove raw generated files if space is needed. |
| `2d_toric_code/without_measurement_noise/kernel_mix_local` | 316K | Local kernel-mixing test | P2 | Remove raw generated files if space is needed. |
| `2d_toric_code/without_measurement_noise/q0_control_extension_nd3_20260421_225303` | 3.0M | 2D q=0 background control extension | P1 | Keep final README/summary locally; raw generated files are not Git-tracked. |
| `2d_toric_code/without_measurement_noise/q0_control_summary_20260422` | 264K | 2D q=0 control summary | P1 | Keep README/summary; no raw archive requirement. |
| `2d_toric_code/without_measurement_noise/q0_geometric_multistart_local` | 296K | Local multistart check | P2 | Remove raw generated files if space is needed. |
| `2d_toric_code/without_measurement_noise/q0_threshold_deep_nd3_20260420_221142` | 5.0M | 2D q=0 deep background control | P1 | Keep final README/summary locally; raw generated files are not Git-tracked. |

### 3D toric code without measurement noise

| Directory | Size | Purpose | Retention | Raw data policy |
|---|---:|---|---|---|
| `3d_toric_code/without_measurement_noise/exp01_q0_pipeline_smoke` | 372K | q=0 pipeline smoke test | P3 | Remove chunks/preflight raw. |
| `3d_toric_code/without_measurement_noise/exp02_q0_low_p_scout` | 16M | q=0 low-p scout | P3 | Remove chunks/preflight raw. |
| `3d_toric_code/without_measurement_noise/exp03_q0_right_shift_scout` | 16M | q=0 right-shift scout | P3 | Remove chunks/preflight raw; keep README. |
| `3d_toric_code/without_measurement_noise/exp04_q0_crossing_window_scout` | 19M | q=0 crossing-window calibration record | P0 | Keep local archive; do not track generated data in Git. |
| `3d_toric_code/without_measurement_noise/exp09_q0_oneday_deep_fixed` | 8.7M | Superseded q=0 one-day deep attempt | P3 | Remove raw generated files after retaining README. |
| `3d_toric_code/without_measurement_noise/exp10_q0_oneday_deep_relaunch` | 12M | Strongest 3D q=0 deep comparison | P0 | Keep local archive; do not track generated data in Git. |

### 3D toric code with measurement noise

| Directory | Size | Purpose | Retention | Raw data policy |
|---|---:|---|---|---|
| `3d_toric_code/with_measurement_noise/exp05_q005_local_precheck_after_fix` | 692K | Local q=0.005 precheck | P3 | Remove raw chunks/preflight; keep README. |
| `3d_toric_code/with_measurement_noise/exp06_zero_disorder_quick_scan` | 480K | Zero-disorder quick scan | P3 | Keep summary only. |
| `3d_toric_code/with_measurement_noise/exp07_q005_broad_scan` | 952K | q=0.005 broad scout | P3 | Remove raw chunks/preflight; keep README. |
| `3d_toric_code/with_measurement_noise/exp08_q005_oneday_deep_scan` | 1.6M | q=0.005 oneday deep scout | P3 | Remove raw chunks/preflight; keep README. |
| `3d_toric_code/with_measurement_noise/exp11_q001_oneday_deep_partial` | 904K | q=0.001 partial scout | P3 | Remove raw chunks/preflight; keep README/summary. |
| `3d_toric_code/with_measurement_noise/exp12_q005_fine_20260425_nd1` | 10M | Old q-positive fine scan | P2 | Old observable result; keep docs only. |
| `3d_toric_code/with_measurement_noise/exp13_q001_coarse_20260425_nd2` | 9.7M | Old q-positive coarse scan | P2 | Old observable result; keep docs only. |
| `3d_toric_code/with_measurement_noise/exp14_q001_fine_20260425_nd3` | 8.9M | Old q-positive fine scan | P2 | Old observable result; keep docs only. |
| `3d_toric_code/with_measurement_noise/exp15_q001_left_denseA_20260425_nd1` | 14M | Old q-positive left dense scan A | P2 | Old observable result; keep docs only. |
| `3d_toric_code/with_measurement_noise/exp16_q001_left_denseB_20260425_nd2` | 14M | Old q-positive left dense scan B | P2 | Old observable result; keep docs only. |
| `3d_toric_code/with_measurement_noise/exp17_q001_left_fine_20260425_nd3` | 15M | Old q-positive left fine scan | P2 | Old observable result; keep docs only. |
| `3d_toric_code/with_measurement_noise/exp18_q001_left_combined_summary` | 584K | Combined old q=0.001 summary | P2 | Keep README/text summary; generated plots/data not tracked. |
| `3d_toric_code/with_measurement_noise/exp19_q050_quick_p010_020_20260425_nd1` | 7.2M | Old q=0.05 quick scan | P2 | Old observable result; keep docs only. |
| `3d_toric_code/with_measurement_noise/exp20a_q050_heavy_p018_022_20260425_nd1` | 14M | Old q=0.05 heavy scan A | P2 | Old observable result; keep docs only. |
| `3d_toric_code/with_measurement_noise/exp20b_q050_heavy_p018_022_20260425_nd2` | 14M | Old q=0.05 heavy scan B | P2 | Old observable result; keep docs only. |
| `3d_toric_code/with_measurement_noise/exp20c_q050_heavy_p018_022_20260425_nd3` | 14M | Old q=0.05 heavy scan C | P2 | Old observable result; keep docs only. |
| `3d_toric_code/with_measurement_noise/exp21_q050_heavy_p018_022_combined_summary` | 3.8M | Combined old q=0.05 summary | P2 | Keep README/text summary; generated plots/data not tracked. |
| `3d_toric_code/with_measurement_noise/exp22a_q050_L6_p018_022_20260425_nd1` | 4.4M | Old q=0.05 L=6 scan A | P2 | Old observable result; keep docs only. |
| `3d_toric_code/with_measurement_noise/exp22b_q050_L6_p018_022_20260425_nd2` | 4.4M | Old q=0.05 L=6 scan B | P2 | Old observable result; keep docs only. |
| `3d_toric_code/with_measurement_noise/exp22c_q050_L6_p018_022_20260425_nd3` | 4.4M | Old q=0.05 L=6 scan C | P2 | Old observable result; keep docs only. |
| `3d_toric_code/with_measurement_noise/exp23_q050_L3456_p018_022_combined_summary` | 340K | Combined old q=0.05 L=3-6 summary | P2 | Keep README/text summary. |
| `3d_toric_code/with_measurement_noise/exp24a_q001_q050_q100_p018_022_dense_20260426_nd1` | 153M | Old q-positive dense scan A | P2 | Old observable result; raw can be removed. |
| `3d_toric_code/with_measurement_noise/exp24b_q001_q050_q100_p018_022_dense_20260426_nd2` | 153M | Old q-positive dense scan B | P2 | Old observable result; raw can be removed. |
| `3d_toric_code/with_measurement_noise/exp24c_q001_q050_q100_p018_022_dense_20260426_nd3` | 153M | Old q-positive dense scan C | P2 | Old observable result; raw can be removed. |
| `3d_toric_code/with_measurement_noise/exp25_q001_q050_q100_p018_022_dense_combined_summary` | 76M | Combined old q-positive dense summary | P2 | Keep README/text summary; generated plots/data not tracked. |
| `3d_toric_code/with_measurement_noise/exp26a_fixed_p010_q000_100_20260426_nd1` | 84M | Old fixed-p q scan A | P2 | Old observable result; raw can be removed. |
| `3d_toric_code/with_measurement_noise/exp26b_fixed_p010_q000_100_20260426_nd2` | 84M | Old fixed-p q scan B | P2 | Old observable result; raw can be removed. |
| `3d_toric_code/with_measurement_noise/exp26c_fixed_p010_q000_100_20260426_nd3` | 84M | Old fixed-p q scan C | P2 | Old observable result; raw can be removed. |
| `3d_toric_code/with_measurement_noise/exp27_fixed_p010_q000_100_combined_summary` | 42M | Combined old fixed-p q summary | P2 | Keep README/text summary; generated plots/data not tracked. |
| `3d_toric_code/with_measurement_noise/exp28a_fixed_p010_q000_100_L6_20260427_nd1` | 28M | Old fixed-p q L=6 scan A | P2 | Old observable result; raw can be removed. |
| `3d_toric_code/with_measurement_noise/exp28b_fixed_p010_q000_100_L6_20260427_nd2` | 28M | Old fixed-p q L=6 scan B | P2 | Old observable result; raw can be removed. |
| `3d_toric_code/with_measurement_noise/exp28c_fixed_p010_q000_100_L6_20260427_nd3` | 28M | Old fixed-p q L=6 scan C | P2 | Old observable result; raw can be removed. |
| `3d_toric_code/with_measurement_noise/exp29_fixed_p010_q000_100_L3456_combined_summary` | 14M | Combined old fixed-p q L=3-6 summary | P2 | Keep README/text summary; generated plots/data not tracked. |
| `3d_toric_code/with_measurement_noise/exp30_cluster_runtime_compare_q050_p020_L345_local` | 28K | Runtime comparison scout | P3 | Keep summary only. |
| `3d_toric_code/with_measurement_noise/exp31_pq_grid_q000_050_p001_120_20260511_nd1` | 214M | Old p-q grid, pre-corrected observable | P2 | Keep README/text summary; raw/intermediate can be removed. |
| `3d_toric_code/with_measurement_noise/exp32_fixed_p050_q000_075_L34567_20260514_nd23` | 1.3G | Old fixed-p q scan, pre-corrected observable | P2 | Keep README/text summary; raw/intermediate can be removed. |
| `3d_toric_code/with_measurement_noise/exp34_fast3d_p050_q000_075_L345_corrected_observable_20260518_nd123` | 167M | Corrected-observable fast scout | P2 | Keep README/text summary; raw/intermediate can be removed after final archive exists. |
| `3d_toric_code/with_measurement_noise/exp34_fixed_p050_q000_080_L34567_corrected_observable_20260520_partial_nd12` | 102M | Corrected-observable partial snapshot superseded by 20260524 final | P3 | Delete whole directory after dry-run. |
| `3d_toric_code/with_measurement_noise/exp34_fixed_p050_q000_080_L34567_corrected_observable_20260522_partial_nd12` | 75M | Corrected-observable partial snapshot superseded by 20260524 final | P3 | Delete whole directory after dry-run. |
| `3d_toric_code/with_measurement_noise/exp34_fixed_p050_q000_080_L34567_corrected_observable_20260524_final_stopped_after_L6q060_nd12` | 135M | Corrected-observable final stopped snapshot for paper | P0 | Keep local archive; do not track generated data in Git. |
| `3d_toric_code/with_measurement_noise/prod_3d_q0050_p020_opt_20260523_1845` | 212K | Production optimization diagnostic | P2 | Keep Chinese summary/small local JSON; do not track generated data in Git. |
| `3d_toric_code/with_measurement_noise/profile_3d_opt_lowdiag_q0050_p020_L45_20260523_180331` | 1.7M | Profiling diagnostic | P2 | Keep Chinese summary and small local summary; remove raw profiling files. |
| `3d_toric_code/with_measurement_noise/profile_3d_opt_obsfast_q0050_p020_L5_20260523_181445` | 480K | Profiling diagnostic | P2 | Keep Chinese summary and small local summary; remove raw profiling files. |
| `3d_toric_code/with_measurement_noise/profile_3d_opt_q0050_p020_L45_20260523_173455` | 1.3M | Profiling diagnostic | P2 | Keep Chinese summary and small local summary; remove raw profiling files. |
| `3d_toric_code/with_measurement_noise/profile_3d_q_positive_calibration_20260523_153408` | 88K | Profiling calibration diagnostic | P2 | Keep summary only; remove raw profiling/repo snapshots. |
| `3d_toric_code/with_measurement_noise/profile_3d_q_positive_default_20260523_153613` | 9.7M | Profiling default diagnostic | P2 | Keep summary only; remove raw profiling/repo snapshots. |

## Operational Notes

- The old `q != 0` numerical results before the corrected observable path are historical diagnostics only and must not be used as paper physics conclusions.
- The corrected-observable final archive is `3d_toric_code/with_measurement_noise/exp34_fixed_p050_q000_080_L34567_corrected_observable_20260524_final_stopped_after_L6q060_nd12`.
- The strongest 3D `q=0` no-measurement-noise comparison is `3d_toric_code/without_measurement_noise/exp10_q0_oneday_deep_relaunch`.
- The 3D `q=0` crossing-window calibration record is `3d_toric_code/without_measurement_noise/exp04_q0_crossing_window_scout`.
- After the 2026-05-24 cleanup, `data/` was reduced from about `3.2G` to about `246M`. Some profiling/optimization raw files remain because they are owned by remote uid/gid `2004` and local passwordless `sudo` was unavailable; they are still ignored by Git and should be removed manually when filesystem ownership permits.
