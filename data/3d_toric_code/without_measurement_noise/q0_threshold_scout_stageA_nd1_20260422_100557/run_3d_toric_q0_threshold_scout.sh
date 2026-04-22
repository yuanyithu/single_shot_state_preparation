#!/usr/bin/env bash
set -euo pipefail

export MPLCONFIGDIR=$HOME/.single_shot/mpl-cache
cd /home/DATA1/users/yuany/.single_shot/repo
output_stem="scan_result_multi_L_3d_toric_q0_threshold_scout"
conda run -n 11 python production_chunked_scan.py submit   --run-root /home/DATA1/users/yuany/.single_shot/runs/3d_toric_q0_threshold_scout_20260422_100557_rightshift_stageA   --code-family 3d_toric   --workers 48   --chunk-size 16   --num-disorder-samples-total 256   --data-error-probabilities 0.1000\,0.1100\,0.1200\,0.1300\,0.1400\,0.1500\,0.1600\,0.1700\,0.1800\,0.1900\,0.2000   --lattice-sizes 3\,4\,5   --syndrome-error-probability 0.0   --num-burn-in-sweeps 1200   --num-sweeps-between-measurements 6   --num-measurements-per-disorder 240   --q0-num-start-chains 8   --seed-base 20260422   --burn-in-scaling-reference-num-qubits 18   --output-stem "$output_stem"   --git-commit-sha fec48073369b6012f6982d48eb40fb3173346f1d
conda run -n 11 python analyze_threshold_crossing.py   /home/DATA1/users/yuany/.single_shot/runs/3d_toric_q0_threshold_scout_20260422_100557_rightshift_stageA/scan_result_multi_L_3d_toric_q0_threshold_scout.npz   --output-dir /home/DATA1/users/yuany/.single_shot/runs/3d_toric_q0_threshold_scout_20260422_100557_rightshift_stageA   --output-stem scan_result_multi_L_3d_toric_q0_threshold_scout   --summary-path /home/DATA1/users/yuany/.single_shot/runs/3d_toric_q0_threshold_scout_20260422_100557_rightshift_stageA/threshold_summary.json
