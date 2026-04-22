#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="threshold_deep_$(date +%Y%m%d_%H%M%S)"
REMOTE_BASE='$HOME/.single_shot'
REMOTE_REPO_DIR='$HOME/.single_shot/repo'
REMOTE_VENV_DIR='$HOME/.single_shot/.venv'
REMOTE_LOGS_DIR='$HOME/.single_shot/logs'
REMOTE_RUN_ROOT="$REMOTE_BASE/runs/$RUN_ID"
REMOTE_LOG_PATH="$REMOTE_LOGS_DIR/${RUN_ID}.log"
REMOTE_SCREEN_NAME="ssprep_${RUN_ID}"
REPO_URL="$(git -C "$PROJECT_ROOT" remote get-url origin)"
COMMIT_SHA="$(git -C "$PROJECT_ROOT" rev-parse HEAD)"
LATTICE_SIZES='3,5,7'
DATA_ERROR_PROBABILITIES='0.0900,0.0925,0.0950,0.0975,0.1000,0.1025,0.1050,0.1075,0.1100,0.1125,0.1150,0.1175,0.1200,0.1225,0.1250'
NUM_DISORDER_SAMPLES_TOTAL='512'
CHUNK_SIZE='32'
REQUESTED_WORKERS='96'
NUM_BURN_IN_SWEEPS='1500'
NUM_SWEEPS_BETWEEN_MEASUREMENTS='5'
NUM_MEASUREMENTS_PER_DISORDER='400'
Q0_NUM_START_CHAINS='4'
SEED_BASE='20260422'
BURN_IN_SCALING_REFERENCE_NUM_QUBITS='18'


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


build_nd3_bootstrap_script() {
  cat <<'EOF'
#!/usr/bin/env bash

set -euo pipefail

run_id="$1"
repo_url="$2"
commit_sha="$3"
skip_git_sync="$4"
lattice_sizes="$5"
data_error_probabilities="$6"
num_disorder_samples_total="$7"
chunk_size="$8"
requested_workers="$9"
shift 9
num_burn_in_sweeps="$1"
num_sweeps_between_measurements="$2"
num_measurements_per_disorder="$3"
q0_num_start_chains="$4"
seed_base="$5"
burn_in_scaling_reference_num_qubits="$6"

remote_base="$HOME/.single_shot"
repo_dir="$remote_base/repo"
venv_dir="$remote_base/.venv"
logs_dir="$remote_base/logs"
run_root="$remote_base/runs/$run_id"
chunks_dir="$run_root/chunks"
log_path="$logs_dir/${run_id}.log"
screen_name="ssprep_${run_id}"

mkdir -p "$remote_base" "$logs_dir" "$run_root" "$chunks_dir"

if [[ "$skip_git_sync" != "1" ]]; then
  if [[ -e "$repo_dir" && ! -d "$repo_dir/.git" ]]; then
    rm -rf "$repo_dir"
  fi
  if [[ -d "$repo_dir/.git" ]]; then
    git -C "$repo_dir" fetch origin || exit 17
    git -C "$repo_dir" checkout main || exit 17
    git -C "$repo_dir" pull --ff-only origin main || exit 17
  else
    rm -rf "$repo_dir"
    git clone "$repo_url" "$repo_dir" || exit 17
    git -C "$repo_dir" checkout main || exit 17
  fi
  git -C "$repo_dir" checkout "$commit_sha" || exit 17
fi

if [[ ! -f "$repo_dir/src/production_chunked_scan.py" ]]; then
  echo "Missing src/production_chunked_scan.py in $repo_dir" >&2
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
if [[ -n "$conda_bin" ]] && "$conda_bin" run -n 12 python -c "import numpy, matplotlib" >/dev/null 2>&1; then
  python_cmd=("$conda_bin" "run" "-n" "12" "python")
else
  if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 is required to build the fallback venv." >&2
    exit 20
  fi
  if [[ ! -x "$venv_dir/bin/python" ]]; then
    python3 -m venv "$venv_dir"
    "$venv_dir/bin/pip" install --upgrade pip
    "$venv_dir/bin/pip" install numpy matplotlib
  fi
  "$venv_dir/bin/python" -c "import numpy, matplotlib" >/dev/null 2>&1
  python_cmd=("$venv_dir/bin/python")
fi

python_version_output="$("${python_cmd[@]}" --version 2>&1)"
touch "$log_path"

if screen -ls | grep -q "[.]${screen_name}[[:space:]]"; then
  echo "screen session already exists: $screen_name" >&2
  exit 21
fi

runner_cmd=(
  "${python_cmd[@]}"
  "$repo_dir/src/production_chunked_scan.py"
  submit
  --run-root "$run_root"
  --workers "$workers"
  --chunk-size "$chunk_size"
  --num-disorder-samples-total "$num_disorder_samples_total"
  --data-error-probabilities "$data_error_probabilities"
  --lattice-sizes "$lattice_sizes"
  --num-burn-in-sweeps "$num_burn_in_sweeps"
  --num-sweeps-between-measurements "$num_sweeps_between_measurements"
  --num-measurements-per-disorder "$num_measurements_per_disorder"
  --q0-num-start-chains "$q0_num_start_chains"
  --seed-base "$seed_base"
  --burn-in-scaling-reference-num-qubits "$burn_in_scaling_reference_num_qubits"
  --git-commit-sha "$commit_sha"
)

printf -v quoted_runner_cmd '%q ' "${runner_cmd[@]}"
printf -v quoted_repo_dir '%q' "$repo_dir"
printf -v quoted_log_path '%q' "$log_path"
screen -dmS "$screen_name" bash -lc \
  "cd $quoted_repo_dir && exec $quoted_runner_cmd >> $quoted_log_path 2>&1"

if ! screen -ls | grep -q "[.]${screen_name}[[:space:]]"; then
  echo "Failed to create detached screen session $screen_name" >&2
  exit 22
fi

printf 'RUN_ID=%s\n' "$run_id"
printf 'SCREEN_NAME=%s\n' "$screen_name"
printf 'LOG_PATH=%s\n' "$log_path"
printf 'RUN_ROOT=%s\n' "$run_root"
printf 'FINAL_NPZ=%s\n' "$run_root/scan_result_multi_L_q0_geometric_multistart_threshold_deep.npz"
printf 'FINAL_PNG=%s\n' "$run_root/scan_result_multi_L_q0_geometric_multistart_threshold_deep.png"
printf 'CPU_COUNT=%s\n' "$cpu_count"
printf 'WORKERS=%s\n' "$workers"
printf 'PYTHON_VERSION=%s\n' "$python_version_output"
EOF
}


run_nd3_bootstrap() {
  local skip_git_sync="$1"
  local bootstrap_script_path
  local relay_script_path
  local relay_remote_path
  local remote_command
  local rc

  bootstrap_script_path="$(mktemp)"
  relay_remote_path="/tmp/ssprep_nd3_bootstrap_${RUN_ID}.sh"
  build_nd3_bootstrap_script > "$bootstrap_script_path"

  remote_command="cat > $(quote_arg "$relay_remote_path") && ssh nd-3 'bash -s -- \
$(quote_arg "$RUN_ID") \
$(quote_arg "$REPO_URL") \
$(quote_arg "$COMMIT_SHA") \
$(quote_arg "$skip_git_sync") \
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


fallback_archive_sync() {
  echo "Syncing local HEAD to nd-3 via git archive." >&2
  git -C "$PROJECT_ROOT" archive --format=tar HEAD \
    | ssh yuany "ssh nd-3 'rm -rf ~/.single_shot/repo && mkdir -p ~/.single_shot/repo && tar -xf - -C ~/.single_shot/repo'"
}


main() {
  require_clean_worktree

  echo "Pushing current main to origin..."
  git -C "$PROJECT_ROOT" push origin main

  echo "Launching ND-3 production run: $RUN_ID"
  fallback_archive_sync
  run_nd3_bootstrap "1"
  echo "ND-3 production screen launched via archive sync."
}


main "$@"
