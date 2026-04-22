#!/usr/bin/env bash
set -euo pipefail

export MPLCONFIGDIR=$HOME/.single_shot/mpl-cache
cd /home/DATA1/users/yuany/.single_shot/repo
output_stem="scan_result_multi_L_3d_toric_q0_threshold_scout"
conda run -n 11 python production_chunked_scan.py submit   --run-root /home/DATA1/users/yuany/.single_shot/runs/3d_toric_q0_threshold_scout_20260422_101018_extension_stageB_nd3   --code-family 3d_toric   --workers 96   --chunk-size 16   --num-disorder-samples-total 256   --data-error-probabilities 0.1600\,0.1700\,0.1800\,0.1900\,0.2000\,0.2100\,0.2200\,0.2300\,0.2400\,0.2500\,0.2600\,0.2700\,0.2800   --lattice-sizes 3\,4\,5   --syndrome-error-probability 0.0   --num-burn-in-sweeps 1200   --num-sweeps-between-measurements 6   --num-measurements-per-disorder 240   --q0-num-start-chains 8   --seed-base 20260423   --burn-in-scaling-reference-num-qubits 18   --output-stem "$output_stem"   --git-commit-sha f92e18ca36b89aa5054d78d2a81abfc56db4a8c2
conda run -n 11 python analyze_threshold_crossing.py   /home/DATA1/users/yuany/.single_shot/runs/3d_toric_q0_threshold_scout_20260422_101018_extension_stageB_nd3/scan_result_multi_L_3d_toric_q0_threshold_scout.npz   --output-dir /home/DATA1/users/yuany/.single_shot/runs/3d_toric_q0_threshold_scout_20260422_101018_extension_stageB_nd3   --output-stem scan_result_multi_L_3d_toric_q0_threshold_scout   --summary-path /home/DATA1/users/yuany/.single_shot/runs/3d_toric_q0_threshold_scout_20260422_101018_extension_stageB_nd3/threshold_summary.json
