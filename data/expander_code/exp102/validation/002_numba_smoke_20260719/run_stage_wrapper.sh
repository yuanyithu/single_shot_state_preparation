#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: run_stage_wrapper.sh STAGE_DIR LOG_FILE COMMAND [ARGS...]" >&2
  exit 64
fi

stage_dir=$1
log_file=$2
shift 2
mkdir -p "$stage_dir" "$(dirname "$log_file")"
exec 9>"$stage_dir/stage.lock"
flock -n 9 || exit 73

if [[ -e "$stage_dir/SUCCESS" ]]; then
  exit 0
fi
if [[ -e "$stage_dir/RUNNING" ]]; then
  echo "stale RUNNING marker requires an explicit resume decision: $stage_dir" >&2
  exit 74
fi

printf '{"pid":%d,"started_utc":"%s"}\n' "$$" "$(date -u +%FT%TZ)" >"$stage_dir/RUNNING.tmp.$$"
mv "$stage_dir/RUNNING.tmp.$$" "$stage_dir/RUNNING"

mark_failed() {
  status=$?
  printf '{"exit_code":%d,"failed_utc":"%s"}\n' "$status" "$(date -u +%FT%TZ)" >"$stage_dir/FAILED.tmp.$$"
  mv "$stage_dir/FAILED.tmp.$$" "$stage_dir/FAILED"
  rm -f "$stage_dir/RUNNING"
  exit "$status"
}
trap mark_failed ERR INT TERM

"$@" >"$log_file" 2>&1
printf '{"completed_utc":"%s"}\n' "$(date -u +%FT%TZ)" >"$stage_dir/SUCCESS.tmp.$$"
mv "$stage_dir/SUCCESS.tmp.$$" "$stage_dir/SUCCESS"
rm -f "$stage_dir/RUNNING" "$stage_dir/FAILED"
trap - ERR INT TERM
