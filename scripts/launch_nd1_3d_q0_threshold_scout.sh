#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
MASTER_RUN_SUFFIX="${MASTER_RUN_SUFFIX:-}"
MASTER_RUN_ID="${MASTER_RUN_ID:-3d_toric_q0_threshold_scout_${RUN_TIMESTAMP}${MASTER_RUN_SUFFIX}}"
REMOTE_COMPUTE_HOST="${REMOTE_COMPUTE_HOST:-nd-1}"
REMOTE_BASE='/home/DATA1/users/yuany/.single_shot'
REMOTE_RUN_ROOT="$REMOTE_BASE/runs/$MASTER_RUN_ID"
REMOTE_LOG_PATH="$REMOTE_BASE/logs/${MASTER_RUN_ID}.log"
REMOTE_RUNNER_PATH="$REMOTE_RUN_ROOT/run_3d_toric_q0_threshold_scout.sh"
REMOTE_SCREEN_NAME="ssprep_${MASTER_RUN_ID}"
COMMIT_SHA="$(git -C "$PROJECT_ROOT" rev-parse HEAD)"

LATTICE_SIZES="${LATTICE_SIZES:-3,4,5}"
DATA_ERROR_PROBABILITIES="${DATA_ERROR_PROBABILITIES:-0.1000,0.1100,0.1200,0.1300,0.1400,0.1500,0.1600,0.1700,0.1800,0.1900,0.2000}"
NUM_DISORDER_SAMPLES_TOTAL="${NUM_DISORDER_SAMPLES_TOTAL:-256}"
CHUNK_SIZE="${CHUNK_SIZE:-16}"
REQUESTED_WORKERS="${REQUESTED_WORKERS:-48}"
NUM_BURN_IN_SWEEPS="${NUM_BURN_IN_SWEEPS:-1200}"
NUM_SWEEPS_BETWEEN_MEASUREMENTS="${NUM_SWEEPS_BETWEEN_MEASUREMENTS:-6}"
NUM_MEASUREMENTS_PER_DISORDER="${NUM_MEASUREMENTS_PER_DISORDER:-240}"
Q0_NUM_START_CHAINS="${Q0_NUM_START_CHAINS:-8}"
SEED_BASE="${SEED_BASE:-20260422}"
BURN_IN_SCALING_REFERENCE_NUM_QUBITS="${BURN_IN_SCALING_REFERENCE_NUM_QUBITS:-18}"
DRY_RUN="${DRY_RUN:-0}"


require_clean_worktree() {
  if [[ -n "$(git -C "$PROJECT_ROOT" status --porcelain)" ]]; then
    echo "Working tree must be clean before remote deployment." >&2
    exit 1
  fi
}


quote_arg() {
  printf '%q' "$1"
}


fallback_archive_sync() {
  echo "Syncing local HEAD to ${REMOTE_COMPUTE_HOST} via git archive." >&2
  git -C "$PROJECT_ROOT" archive --format=tar HEAD \
    | ssh yuany "ssh ${REMOTE_COMPUTE_HOST} 'rm -rf $(quote_arg "$REMOTE_BASE/repo") && mkdir -p $(quote_arg "$REMOTE_BASE/repo") && tar -xf - -C $(quote_arg "$REMOTE_BASE/repo")'"
}


verify_remote_env() {
  ssh yuany "ssh ${REMOTE_COMPUTE_HOST} 'set -e; command -v screen >/dev/null 2>&1; command -v conda >/dev/null 2>&1; conda run -n 11 python -c \"import numpy, matplotlib\" >/dev/null'"
}


build_remote_runner_script() {
  cat <<EOF
#!/usr/bin/env bash
set -euo pipefail

export MPLCONFIGDIR=\$HOME/.single_shot/mpl-cache
export CONDA_NO_PLUGINS=true
cd $(quote_arg "$REMOTE_BASE/repo")
output_stem="scan_result_multi_L_3d_toric_q0_threshold_scout"
conda run -n 11 python src/production_chunked_scan.py submit \
  --run-root $(quote_arg "$REMOTE_RUN_ROOT") \
  --code-family 3d_toric \
  --workers $(quote_arg "$REQUESTED_WORKERS") \
  --chunk-size $(quote_arg "$CHUNK_SIZE") \
  --num-disorder-samples-total $(quote_arg "$NUM_DISORDER_SAMPLES_TOTAL") \
  --data-error-probabilities $(quote_arg "$DATA_ERROR_PROBABILITIES") \
  --lattice-sizes $(quote_arg "$LATTICE_SIZES") \
  --syndrome-error-probability 0.0 \
  --num-burn-in-sweeps $(quote_arg "$NUM_BURN_IN_SWEEPS") \
  --num-sweeps-between-measurements $(quote_arg "$NUM_SWEEPS_BETWEEN_MEASUREMENTS") \
  --num-measurements-per-disorder $(quote_arg "$NUM_MEASUREMENTS_PER_DISORDER") \
  --q0-num-start-chains $(quote_arg "$Q0_NUM_START_CHAINS") \
  --seed-base $(quote_arg "$SEED_BASE") \
  --burn-in-scaling-reference-num-qubits $(quote_arg "$BURN_IN_SCALING_REFERENCE_NUM_QUBITS") \
  --output-stem "\$output_stem" \
  --git-commit-sha $(quote_arg "$COMMIT_SHA")
conda run -n 11 python src/analyze_threshold_crossing.py \
  $(quote_arg "$REMOTE_RUN_ROOT/scan_result_multi_L_3d_toric_q0_threshold_scout.npz") \
  --output-dir $(quote_arg "$REMOTE_RUN_ROOT") \
  --output-stem scan_result_multi_L_3d_toric_q0_threshold_scout \
  --summary-path $(quote_arg "$REMOTE_RUN_ROOT/threshold_summary.json")
EOF
}


launch_remote_runner() {
  local runner_tmp
  runner_tmp="$(mktemp)"
  build_remote_runner_script > "$runner_tmp"

  ssh yuany "ssh ${REMOTE_COMPUTE_HOST} 'mkdir -p $(quote_arg "$REMOTE_RUN_ROOT") $(quote_arg "$REMOTE_BASE/logs") $(quote_arg "$REMOTE_BASE/mpl-cache")'"
  ssh yuany "ssh ${REMOTE_COMPUTE_HOST} 'cat > $(quote_arg "$REMOTE_RUNNER_PATH")'" < "$runner_tmp"
  ssh yuany "ssh ${REMOTE_COMPUTE_HOST} 'chmod +x $(quote_arg "$REMOTE_RUNNER_PATH") && screen -dmS $(quote_arg "$REMOTE_SCREEN_NAME") bash -lc \"exec $(quote_arg "$REMOTE_RUNNER_PATH") >> $(quote_arg "$REMOTE_LOG_PATH") 2>&1\" && printf \"MASTER_RUN_ID=%s\nREMOTE_COMPUTE_HOST=%s\nSCREEN_NAME=%s\nLOG_PATH=%s\nRUN_ROOT=%s\n\" $(quote_arg "$MASTER_RUN_ID") $(quote_arg "$REMOTE_COMPUTE_HOST") $(quote_arg "$REMOTE_SCREEN_NAME") $(quote_arg "$REMOTE_LOG_PATH") $(quote_arg "$REMOTE_RUN_ROOT")'"

  rm -f "$runner_tmp"
}


main() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf "MASTER_RUN_ID=%s\nREMOTE_COMPUTE_HOST=%s\nRUN_ROOT=%s\nLOG_PATH=%s\nP_VALUES=%s\nSEED_BASE=%s\n" \
      "$MASTER_RUN_ID" "$REMOTE_COMPUTE_HOST" "$REMOTE_RUN_ROOT" "$REMOTE_LOG_PATH" \
      "$DATA_ERROR_PROBABILITIES" "$SEED_BASE"
    exit 0
  fi
  require_clean_worktree
  fallback_archive_sync
  verify_remote_env
  launch_remote_runner
}


main "$@"
