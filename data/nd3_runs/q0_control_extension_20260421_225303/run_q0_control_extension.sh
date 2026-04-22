#!/usr/bin/env bash
set -euo pipefail
export MPLCONFIGDIR=/home/DATA1/users/yuany/.single_shot/mpl-cache
cd /home/DATA1/users/yuany/.single_shot/repo
exec /home/DATA1/users/yuany/miniconda3/bin/conda run -n 11 python /home/DATA1/users/yuany/.single_shot/repo/production_chunked_scan.py submit --run-root /home/DATA1/users/yuany/.single_shot/runs/q0_control_extension_20260421_225303 --workers 96 --chunk-size 32 --num-disorder-samples-total 1024 --data-error-probabilities 0.0950\,0.0975\,0.1000\,0.1025\,0.1050\,0.1075\,0.1100 --lattice-sizes 9\,11 --syndrome-error-probability 0.0 --num-burn-in-sweeps 2000 --num-sweeps-between-measurements 10 --num-measurements-per-disorder 600 --q0-num-start-chains 4 --seed-base 20260427 --burn-in-scaling-reference-num-qubits 18 --output-stem scan_result_multi_L_q0_control_extension --git-commit-sha a197215bd18e9ffc160b4864b7f54239ff4e39da 
