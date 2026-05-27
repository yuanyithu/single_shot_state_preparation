#!/usr/bin/env bash
set -euo pipefail

RUN_BASE=/home/DATA1/users/yuany/.single_shot/exp36/exp36_swap_sweep_probe_20260528
SOURCE_DIR=/home/DATA1/users/yuany/.single_shot/repos/exp36_swap_sweep_probe_20260528/source
GIT_SHA=60497ad7969de1ad5217a7cbeee716832184d726

mkdir -p "$(dirname "$SOURCE_DIR")" "$RUN_BASE"
test -f "$SOURCE_DIR/src/production_chunked_scan.py"

launch_one() {
  local node="$1"
  local screen_name="$2"
  local label="$3"
  local swap_sweeps="$4"
  local seed="$5"
  local run_root="$RUN_BASE/$label"
  ssh "$node" "mkdir -p '$run_root' && cd '$SOURCE_DIR' && screen -dmS '$screen_name' bash -lc 'set -euo pipefail; PYTHONPATH=src conda run --no-capture-output -n 11 python src/production_chunked_scan.py submit --run-root \"$run_root\" --code-family 3d_toric --lattice-sizes 6 --data-error-probabilities 0.05 --syndrome-error-probability 0.08 --num-disorder-samples-total 1 --chunk-size 1 --workers 1 --num-burn-in-sweeps 150 --max-effective-num-burn-in-sweeps 750 --num-sweeps-between-measurements 6 --num-measurements-per-disorder 512 --q0-num-start-chains 4 --num-start-chains 4 --pt-ladder-mode sync_enlarge --pt-q-hot 0.32 --pt-num-temperatures 17 --pt-swap-attempt-every-num-sweeps 1 --pt-swap-sweeps-per-attempt \"$swap_sweeps\" --adaptive-pt-rounds 0 --observable-temperature-mode cold --track-pt-sector-diagnostics --pt-sector-diagnostic-stride 4 --disable-cluster-update --seed-base \"$seed\" --git-commit-sha \"$GIT_SHA\" --output-stem \"$label\" > \"$run_root/outer.log\" 2>&1'"
}

launch_one nd-1 exp36_V_swap1 V_swap1_K17_qhot032_m512_s4 1 382000
launch_one nd-2 exp36_W_swap2 W_swap2_K17_qhot032_m512_s4 2 383000
launch_one nd-3 exp36_X_swap4 X_swap4_K17_qhot032_m512_s4 4 384000

echo "Launched swap sweep probe under $RUN_BASE"
