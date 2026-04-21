#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MASTER_RUN_ID="measurement_noise_threshold_search_$(date +%Y%m%d_%H%M%S)"
REMOTE_REPO_DIR='$HOME/.single_shot/repo'
REMOTE_BASE='$HOME/.single_shot'
REMOTE_LOGS_DIR='$HOME/.single_shot/logs'
REMOTE_MASTER_RUN_ROOT="$REMOTE_BASE/runs/$MASTER_RUN_ID"
REMOTE_LOG_PATH="$REMOTE_LOGS_DIR/${MASTER_RUN_ID}.log"
REMOTE_SCREEN_NAME="ssprep_${MASTER_RUN_ID}"
COMMIT_SHA="$(git -C "$PROJECT_ROOT" rev-parse HEAD)"

LATTICE_SIZES='3,5,7'
Q_AND_P_WINDOWS=$'0.0025|0.0850,0.0875,0.0900,0.0925,0.0950,0.0975,0.1000,0.1025,0.1050,0.1075,0.1100\n0.0050|0.0800,0.0825,0.0850,0.0875,0.0900,0.0925,0.0950,0.0975,0.1000,0.1025,0.1050,0.1075\n0.0075|0.0725,0.0750,0.0775,0.0800,0.0825,0.0850,0.0875,0.0900,0.0925,0.0950,0.0975,0.1000\n0.0100|0.0600,0.0625,0.0650,0.0675,0.0700,0.0725,0.0750,0.0775,0.0800,0.0825,0.0850,0.0875,0.0900\n0.0150|0.0450,0.0475,0.0500,0.0525,0.0550,0.0575,0.0600,0.0625,0.0650,0.0675,0.0700,0.0725,0.0750,0.0775,0.0800\n0.0200|0.0350,0.0375,0.0400,0.0425,0.0450,0.0475,0.0500,0.0525,0.0550,0.0575,0.0600,0.0625,0.0650,0.0675,0.0700'
Q_AND_P_WINDOWS_BASE64="$(printf '%s' "$Q_AND_P_WINDOWS" | base64 | tr -d '\n')"
NUM_DISORDER_SAMPLES_TOTAL='2048'
CHUNK_SIZE='64'
REQUESTED_WORKERS='96'
NUM_BURN_IN_SWEEPS='2000'
NUM_SWEEPS_BETWEEN_MEASUREMENTS='10'
NUM_MEASUREMENTS_PER_DISORDER='800'
Q0_NUM_START_CHAINS='4'
SEED_BASE='20260426'
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
q_and_p_windows_base64="$4"
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
venv_dir="$remote_base/.venv"
logs_dir="$remote_base/logs"
mpl_cache_dir="$remote_base/mpl-cache"
master_run_root="$remote_base/runs/$master_run_id"
log_path="$logs_dir/${master_run_id}.log"
screen_name="ssprep_${master_run_id}"

if command -v python3 >/dev/null 2>&1; then
  q_and_p_windows="$(python3 -c 'import base64, sys; print(base64.b64decode(sys.argv[1]).decode(), end="")' "$q_and_p_windows_base64")"
elif command -v base64 >/dev/null 2>&1; then
  q_and_p_windows="$(printf '%s' "$q_and_p_windows_base64" | base64 --decode)"
else
  echo "python3 or base64 is required to decode q/p window payload." >&2
  exit 17
fi

mkdir -p "$remote_base" "$logs_dir" "$master_run_root" "$mpl_cache_dir"

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

runner_script_path="$master_run_root/run_measurement_noise_threshold_search.sh"
cat > "$runner_script_path" <<EOF_RUNNER
#!/usr/bin/env bash
set -euo pipefail

export MPLCONFIGDIR=$(printf '%q' "$mpl_cache_dir")
cd $quoted_repo_dir
python_cmd=($python_cmd_serialized)
lattice_sizes=$(printf '%q' "$lattice_sizes")
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
q_and_p_windows=$(printf '%q' "$q_and_p_windows")

q_index=0
while IFS='|' read -r syndrome_error_probability data_error_probabilities; do
  [[ -z "\$syndrome_error_probability" ]] && continue
  q_tag="\${syndrome_error_probability/./p}"
  q_tag="\${q_tag//-/m}"
  run_root="\$master_run_root/q_\$q_tag"
  current_seed_base=\$((seed_base + q_index * 1000000000))
  output_stem="scan_result_multi_L_q\${q_tag}_measurement_noise_threshold_search_common_random"
  final_npz="\$run_root/\${output_stem}.npz"
  echo "[launcher] starting q=\$syndrome_error_probability run_root=\$run_root seed_base=\$current_seed_base"
  "\${python_cmd[@]}" "$repo_dir/production_chunked_scan.py" submit \
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
    --output-stem "\$output_stem" \
    --common-random-disorder-across-p \
    --git-commit-sha "\$commit_sha"
  "\${python_cmd[@]}" "$repo_dir/analyze_threshold_crossing.py" \
    "\$final_npz" \
    --output-dir "\$run_root" \
    --output-stem "\$output_stem" \
    --summary-path "\$run_root/threshold_summary.json"
  q_index=\$((q_index + 1))
done <<< "\$q_and_p_windows"

echo "[launcher] all measurement-noise threshold-search runs completed"
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
printf 'Q_AND_P_WINDOWS=%s\n' "$q_and_p_windows"
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
  relay_remote_path="/tmp/ssprep_nd3_measurement_threshold_${MASTER_RUN_ID}.sh"
  build_nd3_bootstrap_script > "$bootstrap_script_path"

  remote_command="cat > $(quote_arg "$relay_remote_path") && ssh nd-3 'bash -s -- \
$(quote_arg "$MASTER_RUN_ID") \
$(quote_arg "$COMMIT_SHA") \
$(quote_arg "$LATTICE_SIZES") \
$(quote_arg "$Q_AND_P_WINDOWS_BASE64") \
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

  echo "Launching measurement-noise threshold-search production run: $MASTER_RUN_ID"
  fallback_archive_sync
  run_nd3_bootstrap
  echo "ND-3 measurement-noise threshold-search screen launched."
}


main "$@"
