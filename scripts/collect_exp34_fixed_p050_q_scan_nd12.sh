#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_BASE="${REMOTE_BASE:-/home/DATA1/users/yuany/.single_shot}"
REMOTE_LOGIN="${REMOTE_LOGIN:-yuany}"
REMOTE_HOSTS="${REMOTE_HOSTS:-nd-1,nd-2}"
HOST_RUN_IDS="${HOST_RUN_IDS:-}"
LOCAL_RUN_ID="${LOCAL_RUN_ID:-exp34_fixed_p050_q000_080_L34567_corrected_observable_nd12}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/data/3d_toric_code/with_measurement_noise/$LOCAL_RUN_ID}"
ALLOW_PARTIAL="${ALLOW_PARTIAL:-1}"
RUN_ANALYSIS="${RUN_ANALYSIS:-1}"


host_tag() {
  local host="$1"
  printf '%s' "${host//-/}"
}


copy_remote_run_files() {
  local remote_run_root="$1"
  local local_host_dir="$2"

  ssh "$REMOTE_LOGIN" "
    cd '$remote_run_root' &&
    find . \\( -path '*/chunks' -o -path '*/preflight/chunks' \\) -prune -o \
      -type f \\( \
        -name 'manifest.json' -o \
        -name 'scan_result_*.npz' -o \
        -name 'scan_result_*.png' -o \
        -name '*_convergence.json' -o \
        -name 'run_index.tsv' \
      \\) -print0 |
    tar --null -T - -czf -
  " | tar -xzf - -C "$local_host_dir"
}


main() {
  local hosts
  local run_ids
  local host
  local index
  local host_run_id
  local tag
  local remote_run_root
  local local_host_dir
  local analysis_args

  if [[ -z "$HOST_RUN_IDS" ]]; then
    echo "Set HOST_RUN_IDS to comma-separated nd-1,nd-2 run ids." >&2
    exit 2
  fi

  IFS=',' read -r -a hosts <<< "$REMOTE_HOSTS"
  IFS=',' read -r -a run_ids <<< "$HOST_RUN_IDS"
  if [[ "${#hosts[@]}" -ne "${#run_ids[@]}" ]]; then
    echo "REMOTE_HOSTS and HOST_RUN_IDS must have the same length." >&2
    exit 3
  fi

  mkdir -p "$OUTPUT_DIR/remote_runs" "$OUTPUT_DIR/logs"
  for index in "${!hosts[@]}"; do
    host="${hosts[$index]}"
    host_run_id="${run_ids[$index]}"
    tag="$(host_tag "$host")"
    remote_run_root="$REMOTE_BASE/runs/$host_run_id"
    local_host_dir="$OUTPUT_DIR/remote_runs/$tag"
    mkdir -p "$local_host_dir"

    echo "[collector] copying $host:$remote_run_root -> $local_host_dir"
    copy_remote_run_files "$remote_run_root" "$local_host_dir"
    scp "$REMOTE_LOGIN:${REMOTE_BASE}/logs/${host_run_id}.log" "$OUTPUT_DIR/logs/${host_run_id}.log"
    cp "$local_host_dir/run_index.tsv" "$OUTPUT_DIR/logs/${host_run_id}_run_index.tsv" 2>/dev/null || true
  done

  if [[ "$RUN_ANALYSIS" == "1" ]]; then
    analysis_args=(
      conda run -n 12 python "$PROJECT_ROOT/src/analyze_exp33_fixed_p_q_scan.py"
      --output-dir "$OUTPUT_DIR"
      --host-tags nd1,nd2
      --q-values 0.0000,0.0100,0.0200,0.0300,0.0400,0.0500,0.0600,0.0700,0.0800
      --output-stem fixed_p050_q000_080_exp34_corrected_observable_nd12_pooled
    )
    if [[ "$ALLOW_PARTIAL" == "1" ]]; then
      analysis_args+=(--allow-partial)
    fi
    "${analysis_args[@]}"
  fi

  printf 'OUTPUT_DIR=%s\n' "$OUTPUT_DIR"
}


main "$@"
