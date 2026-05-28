#!/usr/bin/env bash
set -euo pipefail

RUN_BASE=/home/DATA1/users/yuany/.single_shot/exp36/exp36_cluster_probe_20260528
SOURCE_DIR=/home/DATA1/users/yuany/.single_shot/repos/exp36_cluster_probe_20260528/source
GIT_SHA=d000b8c46a0b20a834679c58288ba80b6a4d3064

mkdir -p "$RUN_BASE"
test -f "$SOURCE_DIR/src/production_chunked_scan.py"

launch_one() {
  local node="$1"
  local screen_name="$2"
  local label="$3"
  local qhot="$4"
  local cluster_rho="$5"
  local seed="$6"
  local run_root="$RUN_BASE/$label"
  ssh -n "$node" "mkdir -p '$run_root' && cd '$SOURCE_DIR' && screen -dmS '$screen_name' bash -lc 'set -euo pipefail; PYTHONPATH=src conda run --no-capture-output -n 11 python src/production_chunked_scan.py submit --run-root \"$run_root\" --code-family 3d_toric --lattice-sizes 6 --data-error-probabilities 0.05 --syndrome-error-probability 0.08 --num-disorder-samples-total 1 --chunk-size 1 --workers 1 --num-burn-in-sweeps 150 --max-effective-num-burn-in-sweeps 750 --num-sweeps-between-measurements 6 --num-measurements-per-disorder 512 --q0-num-start-chains 4 --num-start-chains 4 --num-replicas-per-start 1 --pt-ladder-mode sync_enlarge --pt-q-hot \"$qhot\" --pt-num-temperatures 17 --pt-swap-attempt-every-num-sweeps 1 --pt-swap-sweeps-per-attempt 1 --adaptive-pt-rounds 0 --winding-repeat-factor 1 --winding-plane-heatbath-sweeps 0 --observable-temperature-mode cold --track-pt-sector-diagnostics --pt-sector-diagnostic-stride 4 --cluster-budget-fraction-rho \"$cluster_rho\" --seed-base \"$seed\" --git-commit-sha \"$GIT_SHA\" --output-stem \"$label\" > \"$run_root/outer.log\" 2>&1'"
}

launch_one nd-1 exp36_AB_cluster AB_cluster_K17_qhot032_rho005_m512_s4 0.32 0.05 407000
launch_one nd-2 exp36_AC_cluster AC_cluster_K17_qhot032_rho020_m512_s4 0.32 0.20 408000
launch_one nd-3 exp36_AD_cluster AD_cluster_K17_qhot035_rho005_m512_s4 0.35 0.05 409000

echo "Launched cluster probe under $RUN_BASE"
