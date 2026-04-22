#!/usr/bin/env bash
set -euo pipefail

export MPLCONFIGDIR=$HOME/.single_shot/mpl-cache
cd /home/DATA1/users/yuany/.single_shot/repo
output_stem="scan_result_multi_L_3d_toric_q0_threshold_scout"
conda run -n 11 python production_chunked_scan.py submit   --run-root /home/DATA1/users/yuany/.single_shot/runs/3d_toric_q0_threshold_scout_20260421_235447   --code-family 3d_toric   --workers 48   --chunk-size 16   --num-disorder-samples-total 256   --data-error-probabilities 0.0200\,0.0300\,0.0400\,0.0500\,0.0600\,0.0700\,0.0800\,0.0900\,0.1000\,0.1100\,0.1200   --lattice-sizes 3\,4\,5   --syndrome-error-probability 0.0   --num-burn-in-sweeps 1200   --num-sweeps-between-measurements 6   --num-measurements-per-disorder 240   --q0-num-start-chains 8   --seed-base 20260429   --burn-in-scaling-reference-num-qubits 18   --output-stem "$output_stem"   --git-commit-sha 67037b3bfe1449e6243f79223880259f54d5024a
conda run -n 11 python analyze_threshold_crossing.py   /home/DATA1/users/yuany/.single_shot/runs/3d_toric_q0_threshold_scout_20260421_235447/scan_result_multi_L_3d_toric_q0_threshold_scout.npz   --output-dir /home/DATA1/users/yuany/.single_shot/runs/3d_toric_q0_threshold_scout_20260421_235447   --output-stem scan_result_multi_L_3d_toric_q0_threshold_scout   --summary-path /home/DATA1/users/yuany/.single_shot/runs/3d_toric_q0_threshold_scout_20260421_235447/threshold_summary.json
