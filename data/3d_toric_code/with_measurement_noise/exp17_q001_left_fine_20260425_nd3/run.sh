#!/usr/bin/env bash
set -euo pipefail
export MPLCONFIGDIR=$HOME/.single_shot/mpl-cache
export CONDA_NO_PLUGINS=true
cd '/home/DATA1/users/yuany/.single_shot/repos/3d_toric_exp17_q001_left_fine_20260425_nd3'
run_root='/home/DATA1/users/yuany/.single_shot/runs/3d_toric_exp17_q001_left_fine_20260425_nd3/q_0p0100'
output_stem='scan_result_multi_L_3d_toric_q0p0100_left_fine_common_random'
conda run -n 11 python src/production_chunked_scan.py submit   --run-root "$run_root"   --code-family 3d_toric   --workers 48   --chunk-size 2   --num-disorder-samples-total 48   --data-error-probabilities '0.2050,0.2075,0.2100,0.2125,0.2150,0.2175,0.2200,0.2225,0.2250,0.2275,0.2300'   --lattice-sizes '3,4,5'   --syndrome-error-probability 0.0100   --num-burn-in-sweeps 800   --max-effective-num-burn-in-sweeps 3000   --num-sweeps-between-measurements 6   --num-measurements-per-disorder 1536   --q0-num-start-chains 8   --num-start-chains 8   --num-replicas-per-start 1   --pt-p-hot 0.44   --pt-num-temperatures 7   --pt-swap-attempt-every-num-sweeps 1   --seed-base 2026042517   --burn-in-scaling-reference-num-qubits 18   --output-stem "$output_stem"   --common-random-disorder-across-p   --git-commit-sha '5982451bf586e55adf4bddd729c9f4b74184abd5'
conda run -n 11 python src/analyze_threshold_crossing.py "$run_root/$output_stem.npz" --output-dir "$run_root" --output-stem "$output_stem" --summary-path "$run_root/threshold_summary.json"
conda run -n 11 python src/plot_threshold_search_overview.py '/home/DATA1/users/yuany/.single_shot/runs/3d_toric_exp17_q001_left_fine_20260425_nd3'
echo '[launcher] completed 3d_toric_exp17_q001_left_fine_20260425_nd3'
