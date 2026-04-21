#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MASTER_RUN_ID="3d_toric_q0_threshold_deep_$(date +%Y%m%d_%H%M%S)"
REMOTE_RUN_ROOT='$HOME/.single_shot/runs/'"$MASTER_RUN_ID"
REMOTE_LOG_PATH='$HOME/.single_shot/logs/'"${MASTER_RUN_ID}.log"
REMOTE_RUNNER_PATH="$REMOTE_RUN_ROOT/run_3d_toric_q0_threshold_deep.sh"
REMOTE_SCREEN_NAME="ssprep_${MASTER_RUN_ID}"
COMMIT_SHA="$(git -C "$PROJECT_ROOT" rev-parse HEAD)"

LATTICE_SIZES='3,4,5'
NUM_DISORDER_SAMPLES_TOTAL='1024'
CHUNK_SIZE='32'
REQUESTED_WORKERS='48'
NUM_BURN_IN_SWEEPS='1800'
NUM_SWEEPS_BETWEEN_MEASUREMENTS='8'
NUM_MEASUREMENTS_PER_DISORDER='480'
Q0_NUM_START_CHAINS='8'
SEED_BASE='20260430'
BURN_IN_SCALING_REFERENCE_NUM_QUBITS='18'


require_clean_worktree() {
  if [[ -n "$(git -C "$PROJECT_ROOT" status --porcelain)" ]]; then
    echo "Working tree must be clean before remote deployment." >&2
    exit 1
  fi
}


quote_arg() {
  printf '%q' "$1"
}


resolve_scout_summary_path() {
  if [[ $# -ge 1 ]]; then
    printf '%s\n' "$1"
    return 0
  fi
  local latest_path
  latest_path="$(find "$PROJECT_ROOT/data/nd1_runs" -name 'threshold_summary.json' -path '*3d_toric_q0_threshold_scout_*' | sort | tail -n 1)"
  if [[ -z "$latest_path" ]]; then
    echo "Provide a local scout threshold_summary.json path." >&2
    exit 2
  fi
  printf '%s\n' "$latest_path"
}


build_p_csv_from_summary() {
  local summary_path="$1"
  python3 - "$summary_path" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    summary = json.load(handle)

window = summary["recommended_server_window"]
center = 0.5 * (float(window["p_min"]) + float(window["p_max"]))
step = 0.005
half_width = 0.015
start = max(0.0, center - half_width)
stop = center + half_width

values = []
current = start
while current <= stop + 1.0e-12:
    values.append(f"{current:0.4f}")
    current += step

print(",".join(values), end="")
PY
}


fallback_archive_sync() {
  echo "Syncing local HEAD to nd-1 via git archive." >&2
  git -C "$PROJECT_ROOT" archive --format=tar HEAD \
    | ssh yuany "ssh nd-1 'rm -rf ~/.single_shot/repo && mkdir -p ~/.single_shot/repo && tar -xf - -C ~/.single_shot/repo'"
}


verify_remote_env() {
  ssh yuany "ssh nd-1 'set -e; command -v screen >/dev/null 2>&1; command -v conda >/dev/null 2>&1; conda run -n 11 python -c \"import numpy, matplotlib\" >/dev/null'"
}


build_remote_runner_script() {
  local data_error_probabilities="$1"
  cat <<EOF
#!/usr/bin/env bash
set -euo pipefail

export MPLCONFIGDIR=\$HOME/.single_shot/mpl-cache
cd \$HOME/.single_shot/repo
output_stem="scan_result_multi_L_3d_toric_q0_threshold_deep"
conda run -n 11 python production_chunked_scan.py submit \
  --run-root $(quote_arg "$REMOTE_RUN_ROOT") \
  --code-family 3d_toric \
  --workers $(quote_arg "$REQUESTED_WORKERS") \
  --chunk-size $(quote_arg "$CHUNK_SIZE") \
  --num-disorder-samples-total $(quote_arg "$NUM_DISORDER_SAMPLES_TOTAL") \
  --data-error-probabilities $(quote_arg "$data_error_probabilities") \
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
conda run -n 11 python analyze_threshold_crossing.py \
  $(quote_arg "$REMOTE_RUN_ROOT/scan_result_multi_L_3d_toric_q0_threshold_deep.npz") \
  --output-dir $(quote_arg "$REMOTE_RUN_ROOT") \
  --output-stem scan_result_multi_L_3d_toric_q0_threshold_deep \
  --summary-path $(quote_arg "$REMOTE_RUN_ROOT/threshold_summary.json")
EOF
}


launch_remote_runner() {
  local data_error_probabilities="$1"
  local runner_tmp
  runner_tmp="$(mktemp)"
  build_remote_runner_script "$data_error_probabilities" > "$runner_tmp"

  ssh yuany "ssh nd-1 'mkdir -p $(quote_arg "$REMOTE_RUN_ROOT") \$HOME/.single_shot/logs \$HOME/.single_shot/mpl-cache'"
  ssh yuany "cat > $(quote_arg "$REMOTE_RUNNER_PATH") && ssh nd-1 'chmod +x $(quote_arg "$REMOTE_RUNNER_PATH") && screen -dmS $(quote_arg "$REMOTE_SCREEN_NAME") bash -lc \"exec $(quote_arg "$REMOTE_RUNNER_PATH") >> $(quote_arg "$REMOTE_LOG_PATH") 2>&1\" && printf \"MASTER_RUN_ID=%s\nSCREEN_NAME=%s\nLOG_PATH=%s\nRUN_ROOT=%s\nP_VALUES=%s\n\" $(quote_arg "$MASTER_RUN_ID") $(quote_arg "$REMOTE_SCREEN_NAME") $(quote_arg "$REMOTE_LOG_PATH") $(quote_arg "$REMOTE_RUN_ROOT") $(quote_arg "$data_error_probabilities")'" < "$runner_tmp"

  rm -f "$runner_tmp"
}


main() {
  require_clean_worktree
  local scout_summary_path
  local data_error_probabilities
  scout_summary_path="$(resolve_scout_summary_path "$@")"
  data_error_probabilities="$(build_p_csv_from_summary "$scout_summary_path")"
  fallback_archive_sync
  verify_remote_env
  launch_remote_runner "$data_error_probabilities"
}


main "$@"
