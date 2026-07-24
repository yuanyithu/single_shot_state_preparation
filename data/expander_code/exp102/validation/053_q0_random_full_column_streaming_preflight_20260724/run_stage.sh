#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "usage: run_stage.sh STAGE STAGE_DIR LOG_FILE -- COMMAND..." >&2
  exit 64
fi

stage=$1
stage_dir=$2
log_file=$3
shift 3
[[ $1 == -- ]] || exit 64
shift

case "$stage" in
  preflight-node|preflight-combine) ;;
  *) echo "invalid streaming preflight stage: $stage" >&2; exit 64 ;;
esac
[[ ${EXP102_SOURCE_COMMIT:-} =~ ^[0-9a-f]{40}$ ]] || {
  echo "streaming stage must run inside run_verified_source.sh" >&2
  exit 66
}
[[ $# -ge 2 && $1 == python ]] || exit 64
case "$stage:$2" in
  preflight-node:data/expander_code/exp102/validation/053_q0_random_full_column_streaming_preflight_20260724/run_preflight.py) ;;
  preflight-combine:data/expander_code/exp102/validation/053_q0_random_full_column_streaming_preflight_20260724/combine_preflights.py) ;;
  *) echo "streaming command does not match stage" >&2; exit 68 ;;
esac

mkdir -p "$stage_dir" "$(dirname "$log_file")"
for marker in RUNNING SUCCESS FAILED; do
  [[ ! -e "$stage_dir/$marker" ]] || {
    echo "immutable streaming marker already exists: $stage_dir/$marker" >&2
    exit 69
  }
done
[[ ! -e "$log_file" ]] || {
  echo "immutable streaming log already exists: $log_file" >&2
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

"$@" >"$log_file" 2>&1
printf '{"source_commit":"%s","stage":"%s","completed_utc":"%s"}\n' \
  "$EXP102_SOURCE_COMMIT" "$stage" "$(date -u +%FT%TZ)" \
  >"$stage_dir/SUCCESS"
trap - ERR INT TERM HUP
