#!/usr/bin/env bash
set -euo pipefail
export MPLCONFIGDIR=$HOME/.single_shot/mpl-cache
export CONDA_NO_PLUGINS=true
cd /home/DATA1/users/yuany/.single_shot/repos/3d_toric_exp22b_q050_L6_p018_022_20260425_nd2
conda run -n 11 python src/production_chunked_scan.py submit \
  --run-root /home/DATA1/users/yuany/.single_shot/runs/3d_toric_exp22b_q050_L6_p018_022_20260425_nd2/q_0p0500_L6 \
  --code-family 3d_toric \
  --workers 48 \
  --chunk-size 2 \
  --num-disorder-samples-total 96 \
  --data-error-probabilities 0.1800,0.1900,0.2000,0.2100,0.2200 \
  --lattice-sizes 6 \
  --syndrome-error-probability 0.0500 \
  --num-burn-in-sweeps 1000 \
  --num-sweeps-between-measurements 6 \
  --num-measurements-per-disorder 2048 \
  --q0-num-start-chains 8 \
  --num-start-chains 8 \
  --num-replicas-per-start 1 \
  --pt-p-hot 0.44 \
  --pt-num-temperatures 7 \
  --pt-swap-attempt-every-num-sweeps 1 \
  --seed-base 2026043561 \
  --burn-in-scaling-reference-num-qubits 18 \
  --max-effective-num-burn-in-sweeps 3000 \
  --output-stem scan_result_L6_3d_toric_q0p0500_p018_022 \
  --common-random-disorder-across-p \
  --git-commit-sha d7f03f04
