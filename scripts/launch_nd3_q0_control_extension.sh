#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="q0_control_extension_$(date +%Y%m%d_%H%M%S)"
REMOTE_BASE='$HOME/.single_shot'
REMOTE_LOGS_DIR='$HOME/.single_shot/logs'
REMOTE_RUN_ROOT="$REMOTE_BASE/runs/$RUN_ID"
REMOTE_LOG_PATH="$REMOTE_LOGS_DIR/${RUN_ID}.log"
REMOTE_SCREEN_NAME="ssprep_${RUN_ID}"
COMMIT_SHA="$(git -C "$PROJECT_ROOT" rev-parse HEAD)"

LATTICE_SIZES='9,11'
DATA_ERROR_PROBABILITIES='0.0950,0.0975,0.1000,0.1025,0.1050,0.1075,0.1100'
NUM_DISORDER_SAMPLES_TOTAL='1024'
CHUNK_SIZE='32'
REQUESTED_WORKERS='96'
NUM_BURN_IN_SWEEPS='2000'
NUM_SWEEPS_BETWEEN_MEASUREMENTS='10'
NUM_MEASUREMENTS_PER_DISORDER='600'
Q0_NUM_START_CHAINS='4'
SEED_BASE='20260427'
BURN_IN_SCALING_REFERENCE_NUM_QUBITS='18'
OUTPUT_STEM='scan_result_multi_L_q0_control_extension'


require_clean_worktree() {
  local branch
  branch="$(git -C "$PROJECT_ROOT" branch --show-current)"
  if [[ "$branch" != "main" ]]; then
    echo "Expected current branch to be main, found: $branch" >&2
    exit 1
  fi
  if [[ -n "$(git -C "$PROJECT_ROOT" status --porcelain)" ]]; then
    echo "Working tree must be clean before remote deployment." >&2
    exit 1
  fi
}


quote_arg() {
  printf '%q' "$1"
}


fallback_archive_sync() {
  echo "Syncing local HEAD to nd-3 via git archive." >&2
  git -C "$PROJECT_ROOT" archive --format=tar HEAD \
    | ssh yuany "ssh nd-3 'rm -rf ~/.single_shot/repo && mkdir -p ~/.single_shot/repo && tar -xf - -C ~/.single_shot/repo'"
}


build_nd3_bootstrap_script() {
  cat <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

run_id="$1"
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
output_stem="$5"

remote_base="$HOME/.single_shot"
repo_dir="$remote_base/repo"
venv_dir="$remote_base/.venv"
logs_dir="$remote_base/logs"
mpl_cache_dir="$remote_base/mpl-cache"
run_root="$remote_base/runs/$run_id"
log_path="$logs_dir/${run_id}.log"
screen_name="ssprep_${run_id}"

mkdir -p "$remote_base" "$logs_dir" "$run_root" "$mpl_cache_dir"

if [[ ! -f "$repo_dir/production_chunked_scan.py" ]]; then
  echo "Missing production_chunked_scan.py in $repo_dir" >&2
  exit 18
fi
if ! command -v screen >/dev/null 2>&1; then
  echo "screen is required on nd-3 but was not found." >&2
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

python_cmd=()
if [[ -n "$conda_bin" ]] && "$conda_bin" run -n 11 python -c "import numpy, matplotlib" >/dev/null 2>&1; then
  python_cmd=("$conda_bin" "run" "-n" "11" "python")
else
  if [[ ! -x "$venv_dir/bin/python" ]]; then
    python3 -m venv "$venv_dir"
    "$venv_dir/bin/pip" install --upgrade pip
    "$venv_dir/bin/pip" install numpy matplotlib
  fi
  python_cmd=("$venv_dir/bin/python")
fi

touch "$log_path"

printf -v quoted_repo_dir '%q' "$repo_dir"
printf -v quoted_log_path '%q' "$log_path"
printf -v python_cmd_serialized '%q ' "${python_cmd[@]}"
printf -v quoted_run_root '%q' "$run_root"

runner_script_path="$run_root/run_q0_control_extension.sh"
cat > "$runner_script_path" <<EOF_RUNNER
#!/usr/bin/env bash
set -euo pipefail
export MPLCONFIGDIR=$(printf '%q' "$mpl_cache_dir")
cd $quoted_repo_dir
python_cmd=($python_cmd_serialized)
"${python_cmd[@]}" "$repo_dir/production_chunked_scan.py" submit \
  --run-root $quoted_run_root \
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
  --output-stem $(printf '%q' "$output_stem") \
  --git-commit-sha $(printf '%q' "$commit_sha")
EOF_RUNNER
chmod +x "$runner_script_path"

screen -dmS "$screen_name" bash -lc \
  "exec $(printf '%q' "$runner_script_path") >> $quoted_log_path 2>&1"

printf 'RUN_ID=%s\n' "$run_id"
printf 'SCREEN_NAME=%s\n' "$screen_name"
printf 'LOG_PATH=%s\n' "$log_path"
printf 'RUN_ROOT=%s\n' "$run_root"
printf 'FINAL_NPZ=%s\n' "$run_root/$output_stem.npz"
printf 'CPU_COUNT=%s\n' "$cpu_count"
printf 'WORKERS=%s\n' "$workers"
EOF
}


run_nd3_bootstrap() {
  local bootstrap_script_path
  local relay_remote_path
  local remote_command

  bootstrap_script_path="$(mktemp)"
  relay_remote_path="/tmp/ssprep_nd3_q0_control_${RUN_ID}.sh"
  build_nd3_bootstrap_script > "$bootstrap_script_path"

  remote_command="cat > $(quote_arg "$relay_remote_path") && ssh nd-3 'bash -s -- \
$(quote_arg "$RUN_ID") \
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
$(quote_arg "$BURN_IN_SCALING_REFERENCE_NUM_QUBITS") \
$(quote_arg "$OUTPUT_STEM")' < $(quote_arg "$relay_remote_path"); rc=\$?; rm -f $(quote_arg "$relay_remote_path"); exit \$rc"

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
  echo "Pushing current main to origin..."
  git -C "$PROJECT_ROOT" push origin main
  echo "Launching q=0 control extension run: $RUN_ID"
  fallback_archive_sync
  run_nd3_bootstrap
  echo "ND-3 q=0 control extension screen launched."
}


main "$@"
