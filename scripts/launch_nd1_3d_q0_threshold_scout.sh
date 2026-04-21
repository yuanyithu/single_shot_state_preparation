#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MASTER_RUN_ID="3d_toric_q0_threshold_scout_$(date +%Y%m%d_%H%M%S)"
REMOTE_BASE='$HOME/.single_shot'
REMOTE_LOGS_DIR='$HOME/.single_shot/logs'
REMOTE_MASTER_RUN_ROOT="$REMOTE_BASE/runs/$MASTER_RUN_ID"
REMOTE_LOG_PATH="$REMOTE_LOGS_DIR/${MASTER_RUN_ID}.log"
REMOTE_SCREEN_NAME="ssprep_${MASTER_RUN_ID}"
COMMIT_SHA="$(git -C "$PROJECT_ROOT" rev-parse HEAD)"

LATTICE_SIZES='3,4,5'
DATA_ERROR_PROBABILITIES='0.0200,0.0300,0.0400,0.0500,0.0600,0.0700,0.0800,0.0900,0.1000,0.1100,0.1200'
NUM_DISORDER_SAMPLES_TOTAL='256'
CHUNK_SIZE='16'
REQUESTED_WORKERS='48'
NUM_BURN_IN_SWEEPS='1200'
NUM_SWEEPS_BETWEEN_MEASUREMENTS='6'
NUM_MEASUREMENTS_PER_DISORDER='240'
Q0_NUM_START_CHAINS='8'
SEED_BASE='20260429'
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


fallback_archive_sync() {
  echo "Syncing local HEAD to nd-1 via git archive." >&2
  git -C "$PROJECT_ROOT" archive --format=tar HEAD \
    | ssh yuany "ssh nd-1 'rm -rf ~/.single_shot/repo && mkdir -p ~/.single_shot/repo && tar -xf - -C ~/.single_shot/repo'"
}


build_nd1_bootstrap_script() {
  cat <<'EOF'
#!/usr/bin/env bash

set -euo pipefail

master_run_id="$1"
commit_sha="$2"
lattice_sizes="$3"
data_error_probabilities="$4"
num_disorder_samples_total="$5"
chunk_size="$6"
requested_workers="$7"
num_burn_in_sweeps="$8"
num_sweeps_between_measurements="$9"
shift 9
num_measurements_per_disorder="$1"
q0_num_start_chains="$2"
seed_base="$3"
burn_in_scaling_reference_num_qubits="$4"

remote_base="$HOME/.single_shot"
repo_dir="$remote_base/repo"
logs_dir="$remote_base/logs"
mpl_cache_dir="$remote_base/mpl-cache"
master_run_root="$remote_base/runs/$master_run_id"
log_path="$logs_dir/${master_run_id}.log"
screen_name="ssprep_${master_run_id}"

mkdir -p "$remote_base" "$logs_dir" "$master_run_root" "$mpl_cache_dir"

if [[ ! -f "$repo_dir/production_chunked_scan.py" ]]; then
  echo "Missing production_chunked_scan.py in $repo_dir" >&2
  exit 18
fi
if ! command -v screen >/dev/null 2>&1; then
  echo "screen is required on nd-1 but was not found." >&2
  exit 19
fi

screen --version

if command -v nproc >/dev/null 2>&1; then
  cpu_count="$(nproc)"
else
  cpu_count="$(python3 - <<'PY'
import os
print(os.cpu_count() or 1)
PY
)"
fi

workers="$requested_workers"
if (( workers > cpu_count )); then
  workers="$cpu_count"
fi

conda_bin=''
if command -v conda >/dev/null 2>&1; then
  conda_bin="$(command -v conda)"
elif [[ -x "$HOME/miniconda3/bin/conda" ]]; then
  conda_bin="$HOME/miniconda3/bin/conda"
elif [[ -x "$HOME/anaconda3/bin/conda" ]]; then
  conda_bin="$HOME/anaconda3/bin/conda"
fi

if [[ -z "$conda_bin" ]]; then
  echo "conda is required on nd-1." >&2
  exit 20
fi

if ! "$conda_bin" run -n 11 python -c "import numpy, matplotlib" >/dev/null 2>&1; then
  echo "conda env 11 with numpy/matplotlib is required on nd-1." >&2
  exit 21
fi

touch "$log_path"

printf -v quoted_repo_dir '%q' "$repo_dir"
printf -v quoted_master_run_root '%q' "$master_run_root"
printf -v quoted_log_path '%q' "$log_path"
printf -v quoted_conda_bin '%q' "$conda_bin"

runner_script_path="$master_run_root/run_3d_toric_q0_threshold_scout.sh"
cat > "$runner_script_path" <<EOF_RUNNER
#!/usr/bin/env bash
set -euo pipefail

export MPLCONFIGDIR=$(printf '%q' "$mpl_cache_dir")
repo_dir=$quoted_repo_dir
cd "$repo_dir"
conda_bin=$quoted_conda_bin
master_run_root=$quoted_master_run_root
output_stem="scan_result_multi_L_3d_toric_q0_threshold_scout"
"\$conda_bin" run -n 11 python "$repo_dir/production_chunked_scan.py" submit \
  --run-root "\$master_run_root" \
  --code-family 3d_toric \
  --workers $(printf '%q' "$workers") \
  --chunk-size $(printf '%q' "$chunk_size") \
  --num-disorder-samples-total $(printf '%q' "$num_disorder_samples_total") \
  --data-error-probabilities $(printf '%q' "$data_error_probabilities") \
  --lattice-sizes $(printf '%q' "$lattice_sizes") \
  --syndrome-error-probability 0.0 \
  --num-burn-in-sweeps $(printf '%q' "$num_burn_in_sweeps") \
  --num-sweeps-between-measurements $(printf '%q' "$num_sweeps_between_measurements") \
  --num-measurements-per-disorder $(printf '%q' "$num_measurements_per_disorder") \
  --q0-num-start-chains $(printf '%q' "$q0_num_start_chains") \
  --seed-base $(printf '%q' "$seed_base") \
  --burn-in-scaling-reference-num-qubits $(printf '%q' "$burn_in_scaling_reference_num_qubits") \
  --output-stem "\$output_stem" \
  --git-commit-sha $(printf '%q' "$commit_sha")
"\$conda_bin" run -n 11 python "$repo_dir/analyze_threshold_crossing.py" \
  "\$master_run_root/\$output_stem.npz" \
  --output-dir "\$master_run_root" \
  --output-stem "\$output_stem" \
  --summary-path "\$master_run_root/threshold_summary.json"
EOF_RUNNER
chmod +x "$runner_script_path"

screen -dmS "$screen_name" bash -lc \
  "exec $(printf '%q' "$runner_script_path") >> $quoted_log_path 2>&1"

printf 'MASTER_RUN_ID=%s\n' "$master_run_id"
printf 'SCREEN_NAME=%s\n' "$screen_name"
printf 'LOG_PATH=%s\n' "$log_path"
printf 'MASTER_RUN_ROOT=%s\n' "$master_run_root"
printf 'CPU_COUNT=%s\n' "$cpu_count"
printf 'WORKERS=%s\n' "$workers"
EOF
}


run_nd1_bootstrap() {
  local bootstrap_script_path
  local relay_remote_path
  local remote_command

  bootstrap_script_path="$(mktemp)"
  relay_remote_path="/tmp/ssprep_nd1_3d_q0_threshold_scout_${MASTER_RUN_ID}.sh"
  build_nd1_bootstrap_script > "$bootstrap_script_path"

  remote_command="cat > $(quote_arg "$relay_remote_path") && ssh nd-1 'bash -s -- \
$(quote_arg "$MASTER_RUN_ID") \
$(quote_arg "$COMMIT_SHA") \
$(quote_arg "$LATTICE_SIZES") \
$(quote_arg "$DATA_ERROR_PROBABILITIES") \
$(quote_arg "$NUM_DISORDER_SAMPLES_TOTAL") \
$(quote_arg "$CHUNK_SIZE") \
$(quote_arg "$REQUESTED_WORKERS") \
$(quote_arg "$NUM_BURN_IN_SWEEPS") \
$(quote_arg "$NUM_SWEEPS_BETWEEN_MEASUREMENTS") \
$(quote_arg "$NUM_MEASUREMENTS_PER_DISORDER") \
$(quote_arg "$Q0_NUM_START_CHAINS") \
$(quote_arg "$SEED_BASE") \
$(quote_arg "$BURN_IN_SCALING_REFERENCE_NUM_QUBITS")' < $(quote_arg "$relay_remote_path"); rc=\$?; rm -f $(quote_arg "$relay_remote_path"); exit \$rc"

  if ssh yuany "$remote_command" < "$bootstrap_script_path"; then
    rc=0
  else
    rc=$?
  fi
  rm -f "$bootstrap_script_path"
  return "$rc"
}


main() {
  require_clean_worktree
  fallback_archive_sync
  run_nd1_bootstrap
  echo "ND-1 3D toric q=0 scout launched: $MASTER_RUN_ID"
}


main "$@"
