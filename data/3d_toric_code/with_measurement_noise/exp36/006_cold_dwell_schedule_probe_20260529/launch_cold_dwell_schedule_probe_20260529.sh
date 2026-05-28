#!/usr/bin/env bash
set -euo pipefail

RUN_BASE=/home/DATA1/users/yuany/.single_shot/exp36/006_cold_dwell_schedule_probe_20260529
SOURCE_DIR=/home/DATA1/users/yuany/.single_shot/repos/006_cold_dwell_schedule_probe_20260529/source
GIT_SHA=fc3ac62d7

mkdir -p "$RUN_BASE"
test -f "$SOURCE_DIR/src/production_chunked_scan.py"

launch_one() {
  local node="$1"
  local screen_name="$2"
  local label="$3"
  local swap_every="$4"
  local sweeps_between_measurements="$5"
  local measurements="$6"
  local stride="$7"
  local seed="$8"
  local run_root="$RUN_BASE/$label"
  ssh -n "$node" "mkdir -p '$run_root' && cd '$SOURCE_DIR' && screen -dmS '$screen_name' bash -lc 'set -euo pipefail; PYTHONPATH=src conda run --no-capture-output -n 11 python src/production_chunked_scan.py submit --run-root \"$run_root\" --code-family 3d_toric --lattice-sizes 6 --data-error-probabilities 0.05 --syndrome-error-probability 0.08 --num-disorder-samples-total 1 --chunk-size 1 --workers 1 --num-burn-in-sweeps 150 --max-effective-num-burn-in-sweeps 750 --num-sweeps-between-measurements \"$sweeps_between_measurements\" --num-measurements-per-disorder \"$measurements\" --q0-num-start-chains 4 --num-start-chains 4 --num-replicas-per-start 1 --pt-ladder-mode sync_enlarge --pt-q-hot 0.35 --pt-num-temperatures 17 --pt-swap-attempt-every-num-sweeps \"$swap_every\" --pt-swap-sweeps-per-attempt 1 --adaptive-pt-rounds 0 --winding-repeat-factor 1 --winding-plane-heatbath-sweeps 0 --observable-temperature-mode cold --track-pt-sector-diagnostics --pt-sector-diagnostic-stride \"$stride\" --cluster-budget-fraction-rho 0.15 --seed-base \"$seed\" --git-commit-sha \"$GIT_SHA\" --output-stem \"$label\" > \"$run_root/outer.log\" 2>&1'"
}

launch_one nd-1 exp36_006_r1 run01_stride1_m1024_seed425000 1 6 1024 1 425000
launch_one nd-2 exp36_006_r2 run02_swap6_m1024_seed426000 6 6 1024 1 426000
launch_one nd-3 exp36_006_r3 run03_measure1_m2048_seed427000 1 1 2048 1 427000

echo "Launched 006 cold-dwell schedule probes under $RUN_BASE"
