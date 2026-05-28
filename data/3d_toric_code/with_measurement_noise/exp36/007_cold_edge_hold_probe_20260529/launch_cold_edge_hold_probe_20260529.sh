#!/usr/bin/env bash
set -euo pipefail

REMOTE_SOURCE="/home/DATA1/users/yuany/.single_shot/repos/007_cold_edge_hold_probe_20260529/source"
REMOTE_RUN_BASE="/home/DATA1/users/yuany/.single_shot/exp36/007_cold_edge_hold_probe_20260529"
GIT_COMMIT_SHA="f05c1a818"

launch_run() {
  local node="$1"
  local screen_name="$2"
  local run_name="$3"
  local cold_edge_stride="$4"
  local seed_base="$5"
  local run_root="${REMOTE_RUN_BASE}/${run_name}"

  ssh yuany "mkdir -p '${run_root}' && ssh '${node}' 'cd ${REMOTE_SOURCE} && screen -dmS ${screen_name} bash -lc '\''set -euo pipefail; PYTHONPATH=src conda run --no-capture-output -n 11 python src/production_chunked_scan.py submit --run-root \"${run_root}\" --code-family 3d_toric --lattice-sizes 6 --data-error-probabilities 0.05 --syndrome-error-probability 0.08 --num-disorder-samples-total 1 --chunk-size 1 --workers 1 --num-burn-in-sweeps 150 --max-effective-num-burn-in-sweeps 750 --num-sweeps-between-measurements 6 --num-measurements-per-disorder 1024 --q0-num-start-chains 4 --num-start-chains 4 --num-replicas-per-start 1 --pt-ladder-mode sync_enlarge --pt-q-hot 0.35 --pt-num-temperatures 17 --pt-swap-attempt-every-num-sweeps 1 --pt-swap-sweeps-per-attempt 1 --pt-cold-edge-swap-stride ${cold_edge_stride} --adaptive-pt-rounds 0 --winding-repeat-factor 1 --winding-plane-heatbath-sweeps 0 --observable-temperature-mode cold --track-pt-sector-diagnostics --pt-sector-diagnostic-stride 1 --cluster-budget-fraction-rho 0.15 --seed-base ${seed_base} --git-commit-sha ${GIT_COMMIT_SHA} --output-stem ${run_name} > \"${run_root}/outer.log\" 2>&1'\'''"
}

launch_run nd-1 exp36_007_r1 run01_coldedge1_m1024_seed428000 1 428000
launch_run nd-2 exp36_007_r2 run02_coldedge2_m1024_seed429000 2 429000
launch_run nd-3 exp36_007_r3 run03_coldedge4_m1024_seed430000 4 430000

echo "Launched 007 cold-edge hold probe under ${REMOTE_RUN_BASE}"
