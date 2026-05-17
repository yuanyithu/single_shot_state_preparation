#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_DATE="${RUN_DATE:-${RUN_TIMESTAMP%%_*}}"
MASTER_RUN_ID="${MASTER_RUN_ID:-3d_toric_exp33_fixed_p050_q000_075_L34567_corrected_observable_${RUN_TIMESTAMP}}"
LOCAL_RUN_ID="${LOCAL_RUN_ID:-exp33_fixed_p050_q000_075_L34567_corrected_observable_${RUN_DATE}_nd123}"
REMOTE_BASE="${REMOTE_BASE:-/home/DATA1/users/yuany/.single_shot}"
REMOTE_HOSTS="${REMOTE_HOSTS:-nd-1,nd-2,nd-3}"
COMMIT_SHA="${COMMIT_SHA:-$(git -C "$PROJECT_ROOT" rev-parse HEAD)}"

LATTICE_SIZES="${LATTICE_SIZES:-3,4,5,6,7}"
Q_VALUES="${Q_VALUES:-0.0000,0.0050,0.0100,0.0150,0.0200,0.0250,0.0300,0.0350,0.0400,0.0450,0.0500,0.0550,0.0600,0.0650,0.0700,0.0750}"
FIXED_P="${FIXED_P:-0.0500}"
NUM_DISORDER_SAMPLES_TOTAL="${NUM_DISORDER_SAMPLES_TOTAL:-1024}"
CHUNK_SIZE="${CHUNK_SIZE:-4}"
NUM_BURN_IN_SWEEPS="${NUM_BURN_IN_SWEEPS:-1000}"
MAX_EFFECTIVE_NUM_BURN_IN_SWEEPS="${MAX_EFFECTIVE_NUM_BURN_IN_SWEEPS:-3000}"
NUM_SWEEPS_BETWEEN_MEASUREMENTS="${NUM_SWEEPS_BETWEEN_MEASUREMENTS:-6}"
NUM_MEASUREMENTS_PER_DISORDER="${NUM_MEASUREMENTS_PER_DISORDER:-2048}"
Q0_NUM_START_CHAINS="${Q0_NUM_START_CHAINS:-8}"
NUM_START_CHAINS="${NUM_START_CHAINS:-8}"
NUM_REPLICAS_PER_START="${NUM_REPLICAS_PER_START:-1}"
PT_P_HOT="${PT_P_HOT:-0.44}"
PT_NUM_TEMPERATURES="${PT_NUM_TEMPERATURES:-7}"
PT_SWAP_ATTEMPT_EVERY_NUM_SWEEPS="${PT_SWAP_ATTEMPT_EVERY_NUM_SWEEPS:-1}"
BURN_IN_SCALING_REFERENCE_NUM_QUBITS="${BURN_IN_SCALING_REFERENCE_NUM_QUBITS:-18}"
SEED_BASE_ND1="${SEED_BASE_ND1:-202605171}"
SEED_BASE_ND2="${SEED_BASE_ND2:-202605172}"
SEED_BASE_ND3="${SEED_BASE_ND3:-202605173}"
RESUME="${RESUME:-0}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_SYNC="${SKIP_SYNC:-0}"


quote_arg() {
  printf '%q' "$1"
}


host_tag() {
  local host="$1"
  printf '%s' "${host//-/}"
}


default_workers_for_host() {
  case "$1" in
    nd-2)
      printf '80'
      ;;
    nd-1)
      printf '80'
      ;;
    nd-3)
      printf '96'
      ;;
    *)
      printf '48'
      ;;
  esac
}


seed_base_for_host() {
  case "$1" in
    nd-2)
      printf '%s' "$SEED_BASE_ND2"
      ;;
    nd-1)
      printf '%s' "$SEED_BASE_ND1"
      ;;
    nd-3)
      printf '%s' "$SEED_BASE_ND3"
      ;;
    *)
      printf '%s' "$SEED_BASE_ND2"
      ;;
  esac
}


require_no_tracked_worktree_changes() {
  local tracked_status
  tracked_status="$(
    git -C "$PROJECT_ROOT" status --porcelain --untracked-files=no -- \
      src \
      scripts/launch_exp33_fixed_p050_q_scan_nd123.sh \
      scripts/collect_exp33_fixed_p050_q_scan_nd123.sh
  )"
  if [[ -n "$tracked_status" ]]; then
    echo "Tracked deployment-path changes are not included in git archive deployment:" >&2
    echo "$tracked_status" >&2
    exit 1
  fi
}


sync_remote_repo() {
  local host="$1"
  local repo_dir="$2"

  echo "[launcher] syncing HEAD $COMMIT_SHA to $host:$repo_dir"
  git -C "$PROJECT_ROOT" archive --format=tar HEAD \
    | ssh yuany "ssh ${host} 'mkdir -p $(quote_arg "$repo_dir") && tar -xf - -C $(quote_arg "$repo_dir")'"
}


verify_remote_env() {
  local host="$1"
  ssh yuany "ssh ${host} 'set -euo pipefail; hostname; nproc; command -v screen >/dev/null; command -v conda >/dev/null; export CONDA_NO_PLUGINS=true; conda run -n 11 python -c \"import importlib.util, numpy, matplotlib, ldpc; print(\\\"python_env_ok=1\\\"); print(\\\"ldpc_available=1\\\"); print(\\\"numba_available=\\\" + str(importlib.util.find_spec(\\\"numba\\\") is not None))\"'"
}


build_remote_runner_script() {
  local host="$1"
  local host_run_id="$2"
  local repo_dir="$3"
  local master_run_root="$4"
  local requested_workers="$5"
  local seed_base="$6"

  {
    printf '#!/usr/bin/env bash\n'
    printf 'set -euo pipefail\n\n'
    printf 'host=%q\n' "$host"
    printf 'host_run_id=%q\n' "$host_run_id"
    printf 'repo_dir=%q\n' "$repo_dir"
    printf 'master_run_root=%q\n' "$master_run_root"
    printf 'requested_workers=%q\n' "$requested_workers"
    printf 'seed_base=%q\n' "$seed_base"
    printf 'commit_sha=%q\n' "$COMMIT_SHA"
    printf 'lattice_sizes_csv=%q\n' "$LATTICE_SIZES"
    printf 'q_values_csv=%q\n' "$Q_VALUES"
    printf 'fixed_p=%q\n' "$FIXED_P"
    printf 'num_disorder_samples_total=%q\n' "$NUM_DISORDER_SAMPLES_TOTAL"
    printf 'chunk_size=%q\n' "$CHUNK_SIZE"
    printf 'num_burn_in_sweeps=%q\n' "$NUM_BURN_IN_SWEEPS"
    printf 'max_effective_num_burn_in_sweeps=%q\n' "$MAX_EFFECTIVE_NUM_BURN_IN_SWEEPS"
    printf 'num_sweeps_between_measurements=%q\n' "$NUM_SWEEPS_BETWEEN_MEASUREMENTS"
    printf 'num_measurements_per_disorder=%q\n' "$NUM_MEASUREMENTS_PER_DISORDER"
    printf 'q0_num_start_chains=%q\n' "$Q0_NUM_START_CHAINS"
    printf 'num_start_chains=%q\n' "$NUM_START_CHAINS"
    printf 'num_replicas_per_start=%q\n' "$NUM_REPLICAS_PER_START"
    printf 'pt_p_hot=%q\n' "$PT_P_HOT"
    printf 'pt_num_temperatures=%q\n' "$PT_NUM_TEMPERATURES"
    printf 'pt_swap_attempt_every_num_sweeps=%q\n' "$PT_SWAP_ATTEMPT_EVERY_NUM_SWEEPS"
    printf 'burn_in_scaling_reference_num_qubits=%q\n' "$BURN_IN_SCALING_REFERENCE_NUM_QUBITS"
    printf 'resume=%q\n' "$RESUME"
    cat <<'EOF_RUNNER'

export MPLCONFIGDIR="$HOME/.single_shot/mpl-cache"
export CONDA_NO_PLUGINS=true

log_msg() {
  printf '[%(%Y-%m-%dT%H:%M:%S%z)T] %s\n' -1 "$*"
}

format_probability_tag() {
  local value="$1"
  value="${value/./p}"
  value="${value//-/m}"
  printf '%s' "$value"
}

mkdir -p "$master_run_root" "$HOME/.single_shot/logs" "$HOME/.single_shot/mpl-cache"
cd "$repo_dir"

if command -v nproc >/dev/null 2>&1; then
  cpu_count="$(nproc)"
else
  cpu_count="$(conda run -n 11 python -c 'import os; print(os.cpu_count() or 1)')"
fi
workers="$requested_workers"
if (( workers > cpu_count )); then
  workers="$cpu_count"
fi

conda run -n 11 python -c "import importlib.util, numpy, matplotlib, ldpc; print('python_env_ok=1'); print('ldpc_available=1'); print('numba_available=' + str(importlib.util.find_spec('numba') is not None))"

IFS=',' read -r -a lattice_sizes <<< "$lattice_sizes_csv"
IFS=',' read -r -a q_values <<< "$q_values_csv"
fixed_p_tag="$(format_probability_tag "$fixed_p")"
index_path="$master_run_root/run_index.tsv"
printf 'host\thost_run_id\tL\tq\tfixed_p\trun_root\toutput_stem\tseed_base\tworkers\tcompleted_at\n' > "$index_path"

log_msg "starting exp33 corrected-observable fixed-p q scan host=$host host_run_id=$host_run_id cpu_count=$cpu_count workers=$workers resume=$resume"
log_msg "grid L=$lattice_sizes_csv q=$q_values_csv fixed_p=$fixed_p"

lattice_order_index=0
for lattice_size in "${lattice_sizes[@]}"; do
  lattice_seed_base=$((seed_base + lattice_order_index * 100000000))
  log_msg "starting L=$lattice_size lattice_seed_base=$lattice_seed_base"
  for q_value in "${q_values[@]}"; do
    q_tag="$(format_probability_tag "$q_value")"
    run_root="$master_run_root/L${lattice_size}/q_${q_tag}"
    output_stem="scan_result_L${lattice_size}_p${fixed_p_tag}_q${q_tag}_exp33_corrected_observable_bplsd_common_random"
    submit_args=(
      conda run -n 11 python src/production_chunked_scan.py submit
      --run-root "$run_root"
      --code-family 3d_toric
      --workers "$workers"
      --chunk-size "$chunk_size"
      --num-disorder-samples-total "$num_disorder_samples_total"
      --data-error-probabilities "$fixed_p"
      --lattice-sizes "$lattice_size"
      --syndrome-error-probability "$q_value"
      --num-burn-in-sweeps "$num_burn_in_sweeps"
      --max-effective-num-burn-in-sweeps "$max_effective_num_burn_in_sweeps"
      --num-sweeps-between-measurements "$num_sweeps_between_measurements"
      --num-measurements-per-disorder "$num_measurements_per_disorder"
      --q0-num-start-chains "$q0_num_start_chains"
      --seed-base "$lattice_seed_base"
      --burn-in-scaling-reference-num-qubits "$burn_in_scaling_reference_num_qubits"
      --output-stem "$output_stem"
      --common-random-disorder-across-p
      --git-commit-sha "$commit_sha"
    )
    if [[ "$q_value" != "0.0000" && "$q_value" != "0" && "$q_value" != "0.0" ]]; then
      submit_args+=(
        --num-start-chains "$num_start_chains"
        --num-replicas-per-start "$num_replicas_per_start"
        --pt-p-hot "$pt_p_hot"
        --pt-num-temperatures "$pt_num_temperatures"
        --pt-swap-attempt-every-num-sweeps "$pt_swap_attempt_every_num_sweeps"
      )
    fi
    if [[ "$resume" == "1" ]]; then
      submit_args+=(--resume)
    fi

    log_msg "submit L=$lattice_size q=$q_value run_root=$run_root"
    "${submit_args[@]}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$host" "$host_run_id" "$lattice_size" "$q_value" "$fixed_p" \
      "$run_root" "$output_stem" "$lattice_seed_base" "$workers" \
      "$(date -Is)" >> "$index_path"
    log_msg "completed L=$lattice_size q=$q_value"
  done
  lattice_order_index=$((lattice_order_index + 1))
done

log_msg "all exp33 corrected-observable fixed-p q scan jobs completed host=$host host_run_id=$host_run_id"
EOF_RUNNER
  }
}


launch_one_host() {
  local host="$1"
  local tag
  local host_run_id
  local repo_dir
  local run_root
  local log_path
  local runner_path
  local screen_name
  local requested_workers
  local seed_base
  local runner_tmp
  local remote_command

  tag="$(host_tag "$host")"
  host_run_id="${MASTER_RUN_ID}_${tag}"
  repo_dir="$REMOTE_BASE/repos/$host_run_id"
  run_root="$REMOTE_BASE/runs/$host_run_id"
  log_path="$REMOTE_BASE/logs/${host_run_id}.log"
  runner_path="$run_root/run_exp33_fixed_p050_q_scan.sh"
  screen_name="ssprep_${host_run_id}"
  requested_workers="$(default_workers_for_host "$host")"
  seed_base="$(seed_base_for_host "$host")"

  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'HOST=%s\nHOST_RUN_ID=%s\nRUN_ROOT=%s\nLOG_PATH=%s\nSCREEN_NAME=%s\nWORKERS=%s\nSEED_BASE=%s\n' \
      "$host" "$host_run_id" "$run_root" "$log_path" "$screen_name" "$requested_workers" "$seed_base"
    return 0
  fi

  verify_remote_env "$host"
  if [[ "$SKIP_SYNC" != "1" ]]; then
    sync_remote_repo "$host" "$repo_dir"
  fi

  runner_tmp="$(mktemp)"
  build_remote_runner_script "$host" "$host_run_id" "$repo_dir" "$run_root" "$requested_workers" "$seed_base" > "$runner_tmp"

  ssh yuany "ssh ${host} 'mkdir -p $(quote_arg "$run_root") $(quote_arg "$REMOTE_BASE/logs") $(quote_arg "$REMOTE_BASE/mpl-cache")'"
  ssh yuany "ssh ${host} 'cat > $(quote_arg "$runner_path")'" < "$runner_tmp"

  printf -v remote_command 'chmod +x %q && if screen -ls | grep -q %q; then echo %q >&2; exit 24; fi && screen -dmS %q bash -lc %q && printf "HOST_RUN_ID=%%s\nREMOTE_COMPUTE_HOST=%%s\nSCREEN_NAME=%%s\nLOG_PATH=%%s\nRUN_ROOT=%%s\n" %q %q %q %q %q' \
    "$runner_path" "[.]${screen_name}[[:space:]]" "screen session already exists: $screen_name" \
    "$screen_name" "exec $(quote_arg "$runner_path") >> $(quote_arg "$log_path") 2>&1" \
    "$host_run_id" "$host" "$screen_name" "$log_path" "$run_root"
  ssh yuany "ssh ${host} $(quote_arg "$remote_command")"

  rm -f "$runner_tmp"
}


main() {
  local hosts
  local local_output_dir
  local host

  local_output_dir="$PROJECT_ROOT/data/3d_toric_code/with_measurement_noise/$LOCAL_RUN_ID"
  printf 'MASTER_RUN_ID=%s\nLOCAL_RUN_ID=%s\nLOCAL_OUTPUT_DIR=%s\nCOMMIT_SHA=%s\nRESUME=%s\n' \
    "$MASTER_RUN_ID" "$LOCAL_RUN_ID" "$local_output_dir" "$COMMIT_SHA" "$RESUME"

  if [[ "$DRY_RUN" != "1" ]]; then
    require_no_tracked_worktree_changes
  fi

  IFS=',' read -r -a hosts <<< "$REMOTE_HOSTS"
  for host in "${hosts[@]}"; do
    launch_one_host "$host"
  done
}


main "$@"
