#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MASTER_RUN_ID="measurement_noise_overnight_$(date +%Y%m%d_%H%M%S)"
REMOTE_REPO_DIR='$HOME/.single_shot/repo'
REMOTE_BASE='$HOME/.single_shot'
REMOTE_LOGS_DIR='$HOME/.single_shot/logs'
REMOTE_MASTER_RUN_ROOT="$REMOTE_BASE/runs/$MASTER_RUN_ID"
REMOTE_LOG_PATH="$REMOTE_LOGS_DIR/${MASTER_RUN_ID}.log"
REMOTE_SCREEN_NAME="ssprep_${MASTER_RUN_ID}"
COMMIT_SHA="$(git -C "$PROJECT_ROOT" rev-parse HEAD)"

LATTICE_SIZES='3,5,7'
DATA_ERROR_PROBABILITIES='0.0900,0.0925,0.0950,0.0975,0.1000,0.1025,0.1050,0.1075,0.1100,0.1125,0.1150,0.1175,0.1200,0.1225,0.1250'
SYNDROME_ERROR_PROBABILITIES='0.0100,0.0200,0.0300'
NUM_DISORDER_SAMPLES_TOTAL='2048'
CHUNK_SIZE='64'
REQUESTED_WORKERS='96'
NUM_BURN_IN_SWEEPS='2000'
NUM_SWEEPS_BETWEEN_MEASUREMENTS='10'
NUM_MEASUREMENTS_PER_DISORDER='800'
Q0_NUM_START_CHAINS='4'
SEED_BASE='20260424'
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


fallback_archive_sync() {
  echo "Syncing local HEAD to nd-3 via git archive." >&2
  git -C "$PROJECT_ROOT" archive --format=tar HEAD \
    | ssh yuany "ssh nd-3 'rm -rf ~/.single_shot/repo && mkdir -p ~/.single_shot/repo && tar -xf - -C ~/.single_shot/repo'"
}


build_nd3_bootstrap_script() {
  cat <<'EOF'
#!/usr/bin/env bash

set -euo pipefail

master_run_id="$1"
commit_sha="$2"
lattice_sizes="$3"
data_error_probabilities="$4"
syndrome_error_probabilities="$5"
num_disorder_samples_total="$6"
chunk_size="$7"
requested_workers="$8"
num_burn_in_sweeps="$9"
shift 9
num_sweeps_between_measurements="$1"
num_measurements_per_disorder="$2"
q0_num_start_chains="$3"
seed_base="$4"
burn_in_scaling_reference_num_qubits="$5"

remote_base="$HOME/.single_shot"
repo_dir="$remote_base/repo"
venv_dir="$remote_base/.venv"
logs_dir="$remote_base/logs"
mpl_cache_dir="$remote_base/mpl-cache"
master_run_root="$remote_base/runs/$master_run_id"
log_path="$logs_dir/${master_run_id}.log"
screen_name="ssprep_${master_run_id}"

mkdir -p "$remote_base" "$logs_dir" "$master_run_root" "$mpl_cache_dir"

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

printf -v quoted_repo_dir '%q' "$repo_dir"
printf -v quoted_master_run_root '%q' "$master_run_root"
printf -v quoted_log_path '%q' "$log_path"
printf -v python_cmd_serialized '%q ' "${python_cmd[@]}"

runner_script_path="$master_run_root/run_measurement_noise_overnight.sh"
cat > "$runner_script_path" <<EOF_RUNNER
#!/usr/bin/env bash
set -euo pipefail

export MPLCONFIGDIR=$(printf '%q' "$mpl_cache_dir")
cd $quoted_repo_dir
python_cmd=($python_cmd_serialized)
lattice_sizes=$(printf '%q' "$lattice_sizes")
data_error_probabilities=$(printf '%q' "$data_error_probabilities")
syndrome_error_probabilities=$(printf '%q' "$syndrome_error_probabilities")
num_disorder_samples_total=$(printf '%q' "$num_disorder_samples_total")
chunk_size=$(printf '%q' "$chunk_size")
workers=$(printf '%q' "$workers")
num_burn_in_sweeps=$(printf '%q' "$num_burn_in_sweeps")
num_sweeps_between_measurements=$(printf '%q' "$num_sweeps_between_measurements")
num_measurements_per_disorder=$(printf '%q' "$num_measurements_per_disorder")
q0_num_start_chains=$(printf '%q' "$q0_num_start_chains")
seed_base=$(printf '%q' "$seed_base")
burn_in_scaling_reference_num_qubits=$(printf '%q' "$burn_in_scaling_reference_num_qubits")
master_run_root=$quoted_master_run_root
commit_sha=$(printf '%q' "$commit_sha")

IFS=',' read -r -a q_values <<< "\$syndrome_error_probabilities"
q_index=0
for syndrome_error_probability in "\${q_values[@]}"; do
  q_tag="\${syndrome_error_probability/./p}"
  q_tag="\${q_tag//-/m}"
  run_root="\$master_run_root/q_\$q_tag"
  current_seed_base=\$((seed_base + q_index * 1000000000))
  echo "[launcher] starting q=\$syndrome_error_probability run_root=\$run_root seed_base=\$current_seed_base"
  "\${python_cmd[@]}" "$repo_dir/src/production_chunked_scan.py" submit \
    --run-root "\$run_root" \
    --workers "\$workers" \
    --chunk-size "\$chunk_size" \
    --num-disorder-samples-total "\$num_disorder_samples_total" \
    --data-error-probabilities "\$data_error_probabilities" \
    --lattice-sizes "\$lattice_sizes" \
    --syndrome-error-probability "\$syndrome_error_probability" \
    --num-burn-in-sweeps "\$num_burn_in_sweeps" \
    --num-sweeps-between-measurements "\$num_sweeps_between_measurements" \
    --num-measurements-per-disorder "\$num_measurements_per_disorder" \
    --q0-num-start-chains "\$q0_num_start_chains" \
    --seed-base "\$current_seed_base" \
    --burn-in-scaling-reference-num-qubits "\$burn_in_scaling_reference_num_qubits" \
    --common-random-disorder-across-p \
    --git-commit-sha "\$commit_sha"
  q_index=\$((q_index + 1))
done

echo "[launcher] all measurement-noise overnight runs completed"
EOF_RUNNER
chmod +x "$runner_script_path"

screen -dmS "$screen_name" bash -lc \
  "exec $(printf '%q' "$runner_script_path") >> $quoted_log_path 2>&1"

if ! screen -ls | grep -q "[.]${screen_name}[[:space:]]"; then
  echo "Failed to create detached screen session $screen_name" >&2
  exit 22
fi

printf 'MASTER_RUN_ID=%s\n' "$master_run_id"
printf 'SCREEN_NAME=%s\n' "$screen_name"
printf 'LOG_PATH=%s\n' "$log_path"
printf 'MASTER_RUN_ROOT=%s\n' "$master_run_root"
printf 'Q_VALUES=%s\n' "$syndrome_error_probabilities"
printf 'CPU_COUNT=%s\n' "$cpu_count"
printf 'WORKERS=%s\n' "$workers"
printf 'PYTHON_VERSION=%s\n' "$python_version_output"
EOF
}


run_nd3_bootstrap() {
  local bootstrap_script_path
  local relay_remote_path
  local remote_command

  bootstrap_script_path="$(mktemp)"
  relay_remote_path="/tmp/ssprep_nd3_measurement_noise_${MASTER_RUN_ID}.sh"
  build_nd3_bootstrap_script > "$bootstrap_script_path"

  remote_command="cat > $(quote_arg "$relay_remote_path") && ssh nd-3 'bash -s -- \
$(quote_arg "$MASTER_RUN_ID") \
$(quote_arg "$COMMIT_SHA") \
$(quote_arg "$LATTICE_SIZES") \
$(quote_arg "$DATA_ERROR_PROBABILITIES") \
$(quote_arg "$SYNDROME_ERROR_PROBABILITIES") \
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

  echo "Pushing current main to origin..."
  git -C "$PROJECT_ROOT" push origin main

  echo "Launching overnight measurement-noise production run: $MASTER_RUN_ID"
  fallback_archive_sync
  run_nd3_bootstrap
  echo "ND-3 measurement-noise overnight screen launched."
}


main "$@"
