#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_BASE="${REMOTE_BASE:-/home/DATA1/users/yuany/.single_shot}"
REMOTE_HOSTS="${REMOTE_HOSTS:-nd-1,nd-2,nd-3}"
HOST_RUN_IDS="${HOST_RUN_IDS:-}"
LOCAL_RUN_ID="${LOCAL_RUN_ID:-exp33_fixed_p050_q000_075_L34567_corrected_observable_nd123}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/data/3d_toric_code/with_measurement_noise/$LOCAL_RUN_ID}"
ALLOW_PARTIAL="${ALLOW_PARTIAL:-1}"
RUN_ANALYSIS="${RUN_ANALYSIS:-1}"


host_tag() {
  local host="$1"
  printf '%s' "${host//-/}"
}


quote_arg() {
  printf '%q' "$1"
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
    echo "Set HOST_RUN_IDS to comma-separated nd-1,nd-2,nd-3 run ids." >&2
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
    scp -r "yuany:${remote_run_root}/L"* "$local_host_dir/"
    scp "yuany:${REMOTE_BASE}/logs/${host_run_id}.log" "$OUTPUT_DIR/logs/${host_run_id}.log"
    scp "yuany:${remote_run_root}/run_index.tsv" "$OUTPUT_DIR/logs/${host_run_id}_run_index.tsv" || true
  done

  if [[ "$RUN_ANALYSIS" == "1" ]]; then
    analysis_args=(
      conda run -n 12 python "$PROJECT_ROOT/src/analyze_exp33_fixed_p_q_scan.py"
      --output-dir "$OUTPUT_DIR"
      --host-tags nd1,nd2,nd3
    )
    if [[ "$ALLOW_PARTIAL" == "1" ]]; then
      analysis_args+=(--allow-partial)
    fi
    "${analysis_args[@]}"
  fi

  printf 'OUTPUT_DIR=%s\n' "$OUTPUT_DIR"
}


main "$@"
