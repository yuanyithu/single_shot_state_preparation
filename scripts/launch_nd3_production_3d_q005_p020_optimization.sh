#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ID="${RUN_ID:-prod_3d_q0050_p020_opt_${RUN_TIMESTAMP}}"
REMOTE_BASE="${REMOTE_BASE:-/home/DATA1/users/yuany/.single_shot/prod_3d_q0050_p020_opt}"
REMOTE_RUN_ROOT="$REMOTE_BASE/runs/$RUN_ID"
REMOTE_SRC_DIR="$REMOTE_RUN_ROOT/repo"
REMOTE_LOG_PATH="$REMOTE_RUN_ROOT/${RUN_ID}.log"
REMOTE_SCREEN_NAME="prod3dq_${RUN_ID}"
LOCAL_IDENTITY_FILE="/Users/jarvis/.ssh/old/id_ed25519"

WORKERS="${WORKERS:-8}"
LATTICE_SIZES="${LATTICE_SIZES:-4,5}"
P_VALUE="${P_VALUE:-0.2}"
Q_VALUE="${Q_VALUE:-0.05}"
NUM_DISORDER_SAMPLES_TOTAL="${NUM_DISORDER_SAMPLES_TOTAL:-2}"
CHUNK_SIZE="${CHUNK_SIZE:-1}"
NUM_BURN_IN_SWEEPS="${NUM_BURN_IN_SWEEPS:-128}"
MAX_EFFECTIVE_NUM_BURN_IN_SWEEPS="${MAX_EFFECTIVE_NUM_BURN_IN_SWEEPS:-256}"
NUM_MEASUREMENTS_PER_DISORDER="${NUM_MEASUREMENTS_PER_DISORDER:-768}"
NUM_SWEEPS_BETWEEN_MEASUREMENTS="${NUM_SWEEPS_BETWEEN_MEASUREMENTS:-4}"
NUM_START_CHAINS="${NUM_START_CHAINS:-4}"
NUM_REPLICAS_PER_START="${NUM_REPLICAS_PER_START:-1}"
PT_P_HOT="${PT_P_HOT:-0.44}"
PT_NUM_TEMPERATURES="${PT_NUM_TEMPERATURES:-7}"
PT_SWAP_ATTEMPT_EVERY_NUM_SWEEPS="${PT_SWAP_ATTEMPT_EVERY_NUM_SWEEPS:-1}"
SEED_BASE="${SEED_BASE:-2026052307}"


quote_arg() {
  printf '%q' "$1"
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
    sudo mkdir -p "$root_ssh_dir"
    printf '%s\n' "$config_payload" | sudo tee "$root_config_path" >/dev/null
    sudo chmod 700 "$root_ssh_dir"
    sudo chmod 600 "$root_config_path"
    return
  fi

  echo "Cannot write $root_config_path: sudo is unavailable and current user is not root." >&2
  exit 11
}


selected_source_archive() {
  COPYFILE_DISABLE=1 tar -C "$PROJECT_ROOT" -cf - \
    src/build_toric_code_examples.py \
    src/cluster_update.py \
    src/exact_enumeration.py \
    src/linear_section.py \
    src/main.py \
    src/mcmc.py \
    src/mcmc_convergence_gate.py \
    src/mcmc_diagnostics.py \
    src/mcmc_parallel_tempering.py \
    src/plot_scan_results.py \
    src/preprocessing.py \
    src/production_chunked_scan.py
}


remote_nd3() {
  ssh yuany "ssh nd-3 $(quote_arg "$*")"
}


copy_sources_to_nd3() {
  remote_nd3 "rm -rf $(quote_arg "$REMOTE_SRC_DIR") && mkdir -p $(quote_arg "$REMOTE_SRC_DIR")"
  selected_source_archive | ssh yuany "ssh nd-3 tar -xf - -C $(quote_arg "$REMOTE_SRC_DIR")"
}


build_remote_launcher() {
  cat <<'EOF_REMOTE'
#!/usr/bin/env bash

set -euo pipefail

remote_src_dir="$1"
remote_run_root="$2"
remote_log_path="$3"
screen_name="$4"
workers="$5"
lattice_sizes="$6"
p_value="$7"
q_value="$8"
num_disorder_samples_total="$9"
shift 9
chunk_size="$1"
num_burn_in_sweeps="$2"
max_effective_num_burn_in_sweeps="$3"
num_measurements_per_disorder="$4"
num_sweeps_between_measurements="$5"
num_start_chains="$6"
num_replicas_per_start="$7"
pt_p_hot="$8"
pt_num_temperatures="$9"
shift 9
pt_swap_attempt_every_num_sweeps="$1"
seed_base="$2"

mkdir -p "$remote_run_root" "$(dirname "$remote_log_path")"
remote_src_dir="$(cd "$remote_src_dir" && pwd)"
remote_run_root="$(cd "$remote_run_root" && pwd)"
remote_log_path="$(cd "$(dirname "$remote_log_path")" && pwd)/$(basename "$remote_log_path")"

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

if ! "$conda_bin" run -n 11 python -c "import numpy, numba, matplotlib" >/dev/null 2>&1; then
  echo "conda env 11 must import numpy, numba and matplotlib before running." >&2
  exit 23
fi

load1="$(awk '{print $1}' /proc/loadavg 2>/dev/null || echo 999)"
if "$conda_bin" run -n 11 python -c "import sys; raise SystemExit(0 if float(sys.argv[1]) < 90 else 1)" "$load1"; then
  :
else
  echo "nd-3 load1=$load1 >= 90; refusing to start production optimization run." >&2
  exit 24
fi

if screen -ls | grep -q "[.]${screen_name}[[:space:]]"; then
  echo "screen session already exists: $screen_name" >&2
  exit 25
fi

cat > "$remote_run_root/preflight.txt" <<EOF_PREFLIGHT
host=$(hostname)
started_at=$(date -Is)
load1=$load1
workers=$workers
python=$("$conda_bin" run -n 11 python --version 2>&1)
numba=$("$conda_bin" run -n 11 python -c "import numba; print(numba.__version__)")
numpy=$("$conda_bin" run -n 11 python -c "import numpy; print(numpy.__version__)")
matplotlib=$("$conda_bin" run -n 11 python -c "import matplotlib; print(matplotlib.__version__)")
EOF_PREFLIGHT

runner_path="$remote_run_root/run_production_optimization.sh"
cat > "$runner_path" <<EOF_RUNNER
#!/usr/bin/env bash
set -euo pipefail
export NUMBA_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR="\$HOME/.single_shot/mpl-cache"
mkdir -p "\$MPLCONFIGDIR"
cd $(printf '%q' "$remote_src_dir")

run_submit() {
  local label="\$1"
  local single_fraction="\$2"
  local use_pt="\$3"
  local seed="\$4"
  local run_root="$(printf '%q' "$remote_run_root")/\$label"
  local output_stem="scan_\$label"
  local args=(
    $(printf '%q' "$conda_bin") run --no-capture-output -n 11 python src/production_chunked_scan.py submit
    --run-root "\$run_root"
    --code-family 3d_toric
    --workers $(printf '%q' "$workers")
    --chunk-size $(printf '%q' "$chunk_size")
    --num-disorder-samples-total $(printf '%q' "$num_disorder_samples_total")
    --data-error-probabilities $(printf '%q' "$p_value")
    --lattice-sizes $(printf '%q' "$lattice_sizes")
    --syndrome-error-probability $(printf '%q' "$q_value")
    --num-burn-in-sweeps $(printf '%q' "$num_burn_in_sweeps")
    --max-effective-num-burn-in-sweeps $(printf '%q' "$max_effective_num_burn_in_sweeps")
    --num-sweeps-between-measurements $(printf '%q' "$num_sweeps_between_measurements")
    --num-measurements-per-disorder $(printf '%q' "$num_measurements_per_disorder")
    --q0-num-start-chains $(printf '%q' "$num_start_chains")
    --num-start-chains $(printf '%q' "$num_start_chains")
    --num-replicas-per-start $(printf '%q' "$num_replicas_per_start")
    --num-zero-syndrome-sweeps-per-cycle 1
    --winding-repeat-factor 1
    --single-bit-proposal-fraction "\$single_fraction"
    --observable-temperature-mode cold
    --disable-cluster-update
    --common-random-disorder-across-p
    --seed-base "\$seed"
    --output-stem "\$output_stem"
    --git-commit-sha local-production-optimization
  )
  if [[ "\$use_pt" == "1" ]]; then
    args+=(
      --pt-p-hot $(printf '%q' "$pt_p_hot")
      --pt-num-temperatures $(printf '%q' "$pt_num_temperatures")
      --pt-swap-attempt-every-num-sweeps $(printf '%q' "$pt_swap_attempt_every_num_sweeps")
    )
  fi
  printf '[%(%Y-%m-%dT%H:%M:%S%z)T] start %s single_fraction=%s use_pt=%s\n' -1 "\$label" "\$single_fraction" "\$use_pt"
  "\${args[@]}"
  printf '[%(%Y-%m-%dT%H:%M:%S%z)T] done %s\n' -1 "\$label"
}

run_submit pt7_single005_coldobs 0.05 1 $(printf '%q' "$seed_base")
run_submit pt7_single010_coldobs 0.10 1 $(( $(printf '%q' "$seed_base") + 1000 ))
run_submit pt7_single100_coldobs 1.00 1 $(( $(printf '%q' "$seed_base") + 2000 ))
run_submit nopt_single010 0.10 0 $(( $(printf '%q' "$seed_base") + 3000 ))
EOF_RUNNER

chmod +x "$runner_path"
touch "$remote_log_path"
screen -dmS "$screen_name" bash -lc "exec $(printf '%q' "$runner_path") >> $(printf '%q' "$remote_log_path") 2>&1"
if ! screen -ls | grep -q "[.]${screen_name}[[:space:]]"; then
  echo "Failed to create detached screen session $screen_name" >&2
  exit 26
fi

printf 'SCREEN_NAME=%s\n' "$screen_name"
printf 'REMOTE_RUN_ROOT=%s\n' "$remote_run_root"
printf 'REMOTE_SRC_DIR=%s\n' "$remote_src_dir"
printf 'REMOTE_LOG_PATH=%s\n' "$remote_log_path"
printf 'LOAD1=%s\n' "$load1"
EOF_REMOTE
}


main() {
  local remote_launcher

  ensure_root_ssh_config
  copy_sources_to_nd3

  remote_launcher="$REMOTE_RUN_ROOT/launch_remote.sh"
  build_remote_launcher | ssh yuany "ssh nd-3 'cat > $(quote_arg "$remote_launcher") && chmod +x $(quote_arg "$remote_launcher")'"

  remote_nd3 "$(quote_arg "$remote_launcher") $(quote_arg "$REMOTE_SRC_DIR") $(quote_arg "$REMOTE_RUN_ROOT") $(quote_arg "$REMOTE_LOG_PATH") $(quote_arg "$REMOTE_SCREEN_NAME") $(quote_arg "$WORKERS") $(quote_arg "$LATTICE_SIZES") $(quote_arg "$P_VALUE") $(quote_arg "$Q_VALUE") $(quote_arg "$NUM_DISORDER_SAMPLES_TOTAL") $(quote_arg "$CHUNK_SIZE") $(quote_arg "$NUM_BURN_IN_SWEEPS") $(quote_arg "$MAX_EFFECTIVE_NUM_BURN_IN_SWEEPS") $(quote_arg "$NUM_MEASUREMENTS_PER_DISORDER") $(quote_arg "$NUM_SWEEPS_BETWEEN_MEASUREMENTS") $(quote_arg "$NUM_START_CHAINS") $(quote_arg "$NUM_REPLICAS_PER_START") $(quote_arg "$PT_P_HOT") $(quote_arg "$PT_NUM_TEMPERATURES") $(quote_arg "$PT_SWAP_ATTEMPT_EVERY_NUM_SWEEPS") $(quote_arg "$SEED_BASE")"
}


main "$@"
