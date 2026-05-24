#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${RUN_ID:-profile_3d_q_positive_$(date +%Y%m%d_%H%M%S)}"
REMOTE_BASE="${REMOTE_BASE:-.single_shot/profile_3d_q_positive}"
REMOTE_RUN_ROOT="$REMOTE_BASE/runs/$RUN_ID"
REMOTE_SRC_DIR="$REMOTE_RUN_ROOT/repo"
REMOTE_LOG_PATH="$REMOTE_RUN_ROOT/${RUN_ID}.log"
REMOTE_SCREEN_NAME="profile3dq_${RUN_ID}"
LOCAL_IDENTITY_FILE="/Users/jarvis/.ssh/old/id_ed25519"

Q_VALUE="${Q_VALUE:-0.005}"
SUITE="${SUITE:-default}"
MAX_WALL_SECONDS="${MAX_WALL_SECONDS:-14400}"
NUM_BURN_IN_SWEEPS="${NUM_BURN_IN_SWEEPS:-256}"
NUM_MEASUREMENTS="${NUM_MEASUREMENTS:-768}"
NUM_SWEEPS_BETWEEN_MEASUREMENTS="${NUM_SWEEPS_BETWEEN_MEASUREMENTS:-4}"
SEED_BASE="${SEED_BASE:-2026052301}"
REQUESTED_WORKERS_HIGH="${REQUESTED_WORKERS_HIGH:-80}"
REQUESTED_WORKERS_MEDIUM="${REQUESTED_WORKERS_MEDIUM:-48}"
LATTICE_SIZES="${LATTICE_SIZES:-}"
P_VALUES="${P_VALUES:-}"
CONFIG_LABELS="${CONFIG_LABELS:-}"
NUM_DISORDERS="${NUM_DISORDERS:-}"
L5_NUM_DISORDERS="${L5_NUM_DISORDERS:-}"
STAGE_SIGNATURE_MODE="${STAGE_SIGNATURE_MODE:-stage}"


quote_arg() {
  printf '%q' "$1"
}

remote_arg() {
  if [[ -z "$1" ]]; then
    printf '%q' "__EMPTY_ARG__"
  else
    quote_arg "$1"
  fi
}


ensure_root_ssh_config() {
  local root_ssh_dir="/var/root/.ssh"
  local root_config_path="/var/root/.ssh/config"
  local config_payload

  config_payload="$(cat <<EOF_CONFIG
Host Jumper
  HostName hepthu.com
  User test
  Port 48571
  IdentityFile $LOCAL_IDENTITY_FILE
  StrictHostKeyChecking accept-new

Host yuany
  HostName 192.168.3.150
  ProxyJump Jumper
  User yuany
  IdentityFile $LOCAL_IDENTITY_FILE
  StrictHostKeyChecking accept-new
EOF_CONFIG
)"

  if [[ "$(id -u)" == "0" ]]; then
    mkdir -p "$root_ssh_dir"
    chmod 700 "$root_ssh_dir"
    printf '%s\n' "$config_payload" > "$root_config_path"
    chmod 600 "$root_config_path"
    return
  fi

  if command -v sudo >/dev/null 2>&1; then
    printf '%s\n' "$config_payload" | sudo tee "$root_config_path.tmp" >/dev/null
    sudo mkdir -p "$root_ssh_dir"
    sudo mv "$root_config_path.tmp" "$root_config_path"
    sudo chmod 700 "$root_ssh_dir"
    sudo chmod 600 "$root_config_path"
    return
  fi

  echo "Cannot write $root_config_path: sudo is unavailable and current user is not root." >&2
  exit 11
}


selected_source_archive() {
  COPYFILE_DISABLE=1 tar -C "$PROJECT_ROOT" -cf - \
    src/profile_3d_q_positive.py \
    src/build_toric_code_examples.py \
    src/cluster_update.py \
    src/linear_section.py \
    src/main.py \
    src/mcmc.py \
    src/mcmc_diagnostics.py \
    src/preprocessing.py
}


remote_nd3() {
  ssh yuany "ssh nd-3 $(quote_arg "$*")"
}


copy_sources_to_nd3() {
  local remote_prepare
  remote_prepare="rm -rf $(quote_arg "$REMOTE_SRC_DIR") && mkdir -p $(quote_arg "$REMOTE_SRC_DIR")"
  remote_nd3 "$remote_prepare"
  selected_source_archive | ssh yuany "ssh nd-3 tar -xf - -C $(quote_arg "$REMOTE_SRC_DIR")"
}


build_remote_launcher() {
  cat <<'EOF_REMOTE'
#!/usr/bin/env bash

set -euo pipefail

run_id="$1"
remote_src_dir_arg="$2"
remote_run_root_arg="$3"
remote_log_path_arg="$4"
screen_name="$5"
q_value="$6"
suite="$7"
max_wall_seconds="$8"
num_burn_in_sweeps="$9"
shift 9
num_measurements="$1"
num_sweeps_between_measurements="$2"
seed_base="$3"
requested_workers_high="$4"
requested_workers_medium="$5"
lattice_sizes="${6:-}"
p_values="${7:-}"
config_labels="${8:-}"
num_disorders="${9:-}"
l5_num_disorders="${10:-}"
stage_signature_mode="${11:-stage}"
if [[ "$lattice_sizes" == "__EMPTY_ARG__" ]]; then lattice_sizes=""; fi
if [[ "$p_values" == "__EMPTY_ARG__" ]]; then p_values=""; fi
if [[ "$config_labels" == "__EMPTY_ARG__" ]]; then config_labels=""; fi
if [[ "$num_disorders" == "__EMPTY_ARG__" ]]; then num_disorders=""; fi
if [[ "$l5_num_disorders" == "__EMPTY_ARG__" ]]; then l5_num_disorders=""; fi
if [[ "$stage_signature_mode" == "__EMPTY_ARG__" ]]; then stage_signature_mode="stage"; fi

mkdir -p "$remote_src_dir_arg" "$remote_run_root_arg" "$(dirname "$remote_log_path_arg")"
remote_src_dir="$(cd "$remote_src_dir_arg" && pwd)"
remote_run_root="$(cd "$remote_run_root_arg" && pwd)"
remote_log_path="$(cd "$(dirname "$remote_log_path_arg")" && pwd)/$(basename "$remote_log_path_arg")"

if ! command -v screen >/dev/null 2>&1; then
  echo "screen is required on nd-3 but was not found." >&2
  exit 21
fi

conda_bin=""
if command -v conda >/dev/null 2>&1; then
  conda_bin="$(command -v conda)"
elif [[ -x "$HOME/miniconda3/bin/conda" ]]; then
  conda_bin="$HOME/miniconda3/bin/conda"
elif [[ -x "$HOME/anaconda3/bin/conda" ]]; then
  conda_bin="$HOME/anaconda3/bin/conda"
fi
if [[ -z "$conda_bin" ]]; then
  echo "conda was not found on nd-3." >&2
  exit 22
fi

if ! "$conda_bin" run -n 11 python -c "import numpy, numba" >/dev/null 2>&1; then
  echo "conda env 11 must import numpy and numba before profiling." >&2
  exit 23
fi

if command -v nproc >/dev/null 2>&1; then
  cpu_count="$(nproc)"
else
  cpu_count="$("$conda_bin" run -n 11 python -c "import os; print(os.cpu_count() or 1)")"
fi

load1="$(awk '{print $1}' /proc/loadavg 2>/dev/null || echo 999)"
workers="$("$conda_bin" run -n 11 python -c "import sys; load=float(sys.argv[1]); high=int(sys.argv[2]); medium=int(sys.argv[3]); cpus=int(sys.argv[4]); print(0 if load >= 90 else min(cpus, high if load < 64 else medium))" "$load1" "$requested_workers_high" "$requested_workers_medium" "$cpu_count")"
if [[ "$workers" == "0" ]]; then
  echo "nd-3 load1=$load1 >= 90; refusing to start profile." >&2
  exit 24
fi

if screen -ls | grep -q "[.]${screen_name}[[:space:]]"; then
  echo "screen session already exists: $screen_name" >&2
  exit 25
fi

cat > "$remote_run_root/preflight.txt" <<EOF_PREFLIGHT
run_id=$run_id
host=$(hostname)
started_at=$(date -Is)
cpu_count=$cpu_count
load1=$load1
workers=$workers
conda=$conda_bin
screen=$(screen --version 2>&1)
disk=$(df -h "$remote_run_root" | tail -n 1)
python=$("$conda_bin" run -n 11 python --version 2>&1)
numba=$("$conda_bin" run -n 11 python -c "import numba; print(numba.__version__)")
numpy=$("$conda_bin" run -n 11 python -c "import numpy; print(numpy.__version__)")
EOF_PREFLIGHT

runner_path="$remote_run_root/run_profile.sh"
extra_args=()
if [[ -n "$lattice_sizes" ]]; then
  extra_args+=(--lattice-sizes "$lattice_sizes")
fi
if [[ -n "$p_values" ]]; then
  extra_args+=(--p-values "$p_values")
fi
if [[ -n "$config_labels" ]]; then
  extra_args+=(--config-labels "$config_labels")
fi
if [[ -n "$num_disorders" ]]; then
  extra_args+=(--num-disorders "$num_disorders")
fi
if [[ -n "$l5_num_disorders" ]]; then
  extra_args+=(--l5-num-disorders "$l5_num_disorders")
fi
printf -v extra_args_quoted ' %q' "${extra_args[@]}"
cat > "$runner_path" <<EOF_RUNNER
#!/usr/bin/env bash
set -euo pipefail
export NUMBA_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd $(printf '%q' "$remote_src_dir")
exec $(printf '%q' "$conda_bin") run --no-capture-output -n 11 python src/profile_3d_q_positive.py \
  --code-family 3d_toric \
  --q $(printf '%q' "$q_value") \
  --suite $(printf '%q' "$suite") \
  --run-root $(printf '%q' "$remote_run_root") \
  --workers $(printf '%q' "$workers") \
  --max-wall-seconds $(printf '%q' "$max_wall_seconds") \
  --num-burn-in-sweeps $(printf '%q' "$num_burn_in_sweeps") \
  --num-measurements $(printf '%q' "$num_measurements") \
  --num-sweeps-between-measurements $(printf '%q' "$num_sweeps_between_measurements") \
  --stage-signature-mode $(printf '%q' "$stage_signature_mode") \
  --seed-base $(printf '%q' "$seed_base")$extra_args_quoted
EOF_RUNNER
chmod +x "$runner_path"
touch "$remote_log_path"

screen -dmS "$screen_name" bash -lc "exec $(printf '%q' "$runner_path") >> $(printf '%q' "$remote_log_path") 2>&1"
if ! screen -ls | grep -q "[.]${screen_name}[[:space:]]"; then
  echo "Failed to create detached screen session $screen_name" >&2
  exit 26
fi

printf 'RUN_ID=%s\n' "$run_id"
printf 'SCREEN_NAME=%s\n' "$screen_name"
printf 'REMOTE_RUN_ROOT=%s\n' "$remote_run_root"
printf 'REMOTE_SRC_DIR=%s\n' "$remote_src_dir"
printf 'REMOTE_LOG_PATH=%s\n' "$remote_log_path"
printf 'CPU_COUNT=%s\n' "$cpu_count"
printf 'LOAD1=%s\n' "$load1"
printf 'WORKERS=%s\n' "$workers"
EOF_REMOTE
}


launch_on_nd3() {
  local remote_launcher_local
  local remote_launcher_path
  remote_launcher_local="$(mktemp)"
  remote_launcher_path="/tmp/${RUN_ID}_nd3_profile_launcher.sh"
  build_remote_launcher > "$remote_launcher_local"
  ssh yuany "cat > $(quote_arg "$remote_launcher_path") && ssh nd-3 'bash -s -- \
$(quote_arg "$RUN_ID") \
$(quote_arg "$REMOTE_SRC_DIR") \
$(quote_arg "$REMOTE_RUN_ROOT") \
$(quote_arg "$REMOTE_LOG_PATH") \
$(quote_arg "$REMOTE_SCREEN_NAME") \
$(quote_arg "$Q_VALUE") \
$(quote_arg "$SUITE") \
$(quote_arg "$MAX_WALL_SECONDS") \
$(quote_arg "$NUM_BURN_IN_SWEEPS") \
$(quote_arg "$NUM_MEASUREMENTS") \
$(quote_arg "$NUM_SWEEPS_BETWEEN_MEASUREMENTS") \
$(quote_arg "$SEED_BASE") \
$(quote_arg "$REQUESTED_WORKERS_HIGH") \
$(quote_arg "$REQUESTED_WORKERS_MEDIUM") \
$(remote_arg "$LATTICE_SIZES") \
$(remote_arg "$P_VALUES") \
$(remote_arg "$CONFIG_LABELS") \
$(remote_arg "$NUM_DISORDERS") \
$(remote_arg "$L5_NUM_DISORDERS") \
$(remote_arg "$STAGE_SIGNATURE_MODE")' < $(quote_arg "$remote_launcher_path"); rc=\$?; rm -f $(quote_arg "$remote_launcher_path"); exit \$rc" < "$remote_launcher_local"
  rm -f "$remote_launcher_local"
}


main() {
  ensure_root_ssh_config
  copy_sources_to_nd3
  launch_on_nd3
}


main "$@"
