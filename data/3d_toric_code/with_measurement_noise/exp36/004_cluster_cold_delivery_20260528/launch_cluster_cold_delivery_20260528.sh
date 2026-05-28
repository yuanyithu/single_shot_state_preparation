#!/usr/bin/env bash
set -euo pipefail

RUN_BASE=/home/DATA1/users/yuany/.single_shot/exp36/004_cluster_cold_delivery_20260528
SOURCE_DIR=/home/DATA1/users/yuany/.single_shot/repos/004_cluster_cold_delivery_20260528/source
GIT_SHA=4e58324239c9565d38d80ce921dad06395871113

mkdir -p "$RUN_BASE"
test -f "$SOURCE_DIR/src/production_chunked_scan.py"

launch_one() {
  local node="$1"
  local screen_name="$2"
  local label="$3"
  local swap_sweeps="$4"
  local measurements="$5"
  local stride="$6"
  local seed="$7"
  local run_root="$RUN_BASE/$label"
  ssh -n "$node" "mkdir -p '$run_root' && cd '$SOURCE_DIR' && screen -dmS '$screen_name' bash -lc 'set -euo pipefail; PYTHONPATH=src conda run --no-capture-output -n 11 python src/production_chunked_scan.py submit --run-root \"$run_root\" --code-family 3d_toric --lattice-sizes 6 --data-error-probabilities 0.05 --syndrome-error-probability 0.08 --num-disorder-samples-total 1 --chunk-size 1 --workers 1 --num-burn-in-sweeps 150 --max-effective-num-burn-in-sweeps 750 --num-sweeps-between-measurements 6 --num-measurements-per-disorder \"$measurements\" --q0-num-start-chains 4 --num-start-chains 4 --num-replicas-per-start 1 --pt-ladder-mode sync_enlarge --pt-q-hot 0.35 --pt-num-temperatures 17 --pt-swap-attempt-every-num-sweeps 1 --pt-swap-sweeps-per-attempt \"$swap_sweeps\" --adaptive-pt-rounds 0 --winding-repeat-factor 1 --winding-plane-heatbath-sweeps 0 --observable-temperature-mode cold --track-pt-sector-diagnostics --pt-sector-diagnostic-stride \"$stride\" --cluster-budget-fraction-rho 0.15 --seed-base \"$seed\" --git-commit-sha \"$GIT_SHA\" --output-stem \"$label\" > \"$run_root/outer.log\" 2>&1'"
}

launch_one nd-1 exp36_004_r1 run01_rho015_swap1_m512_seed419000 1 512 4 419000
launch_one nd-2 exp36_004_r2 run02_rho015_swap2_m512_seed420000 2 512 4 420000
launch_one nd-3 exp36_004_r3 run03_rho015_swap1_m1024_seed421000 1 1024 8 421000

echo "Launched 004 cluster cold-delivery diagnostics under $RUN_BASE"
