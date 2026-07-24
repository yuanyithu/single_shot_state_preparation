#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "usage: run_stage.sh STAGE STAGE_DIR LOG_FILE -- WORKFLOW_ARGS..." >&2
  exit 64
fi

stage=$1
stage_dir=$2
log_file=$3
shift 3
[[ $1 == -- ]] || exit 64
shift

case "$stage" in
  schedule|preflight-node|preflight-combine|measurement-node) ;;
  *) echo "invalid T1 stage: $stage" >&2; exit 64 ;;
esac
[[ ${EXP102_SOURCE_COMMIT:-} =~ ^[0-9a-f]{40}$ ]] || {
  echo "T1 stage must run inside run_verified_source.sh" >&2
  exit 66
}
[[ $# -ge 1 ]] || exit 64
case "$stage:$1" in
  schedule:build-schedule|preflight-node:preflight-node|\
  preflight-combine:combine-preflight|measurement-node:run-node) ;;
  *) echo "workflow action does not match T1 stage" >&2; exit 68 ;;
esac

mkdir -p "$stage_dir" "$(dirname "$log_file")"
for marker in RUNNING SUCCESS FAILED; do
  [[ ! -e "$stage_dir/$marker" ]] || {
    echo "immutable T1 stage marker already exists: $stage_dir/$marker" >&2
    exit 69
  }
done
[[ ! -e "$log_file" ]] || {
  echo "immutable T1 stage log already exists: $log_file" >&2
  exit 69
}

printf '{"pid":%d,"source_commit":"%s","stage":"%s","started_utc":"%s"}\n' \
  "$$" "$EXP102_SOURCE_COMMIT" "$stage" "$(date -u +%FT%TZ)" \
  >"$stage_dir/RUNNING"

mark_failed() {
  status=$?
  printf '{"exit_code":%d,"source_commit":"%s","stage":"%s","failed_utc":"%s"}\n' \
    "$status" "$EXP102_SOURCE_COMMIT" "$stage" "$(date -u +%FT%TZ)" \
    >"$stage_dir/FAILED"
  exit "$status"
}
trap mark_failed ERR INT TERM HUP

python -m data.expander_code.exp102.validation.052_q0_random_full_column_t1_m8_20260724.workflow \
  "$@" >"$log_file" 2>&1
printf '{"source_commit":"%s","stage":"%s","completed_utc":"%s"}\n' \
  "$EXP102_SOURCE_COMMIT" "$stage" "$(date -u +%FT%TZ)" \
  >"$stage_dir/SUCCESS"
trap - ERR INT TERM HUP
