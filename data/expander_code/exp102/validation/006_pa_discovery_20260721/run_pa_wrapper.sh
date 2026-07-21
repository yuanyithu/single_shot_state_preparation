#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: run_pa_wrapper.sh STAGE_DIR LOG_FILE STAGE_FINGERPRINT COMMAND [ARGS...]" >&2
  exit 64
fi

stage_dir=$1
log_file=$2
stage_fingerprint=$3
shift 3
mkdir -p "$stage_dir" "$(dirname "$log_file")"
exec 9>"$stage_dir/stage.lock"
flock -n 9 || exit 73

for marker in RUNNING SUCCESS FAILED; do
  if [[ -e "$stage_dir/$marker" ]]; then
    recorded=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["stage_fingerprint"])' "$stage_dir/$marker")
    if [[ "$recorded" != "$stage_fingerprint" ]]; then
      echo "PA stage marker identity conflict: $stage_dir/$marker" >&2
      exit 74
    fi
  fi
done
if [[ -e "$stage_dir/SUCCESS" ]]; then
  exit 0
fi
if [[ -e "$stage_dir/RUNNING" ]]; then
  echo "stale PA RUNNING marker requires an explicit resume decision" >&2
  exit 75
fi

printf '{"pid":%d,"stage_fingerprint":"%s","started_utc":"%s"}\n' \
  "$$" "$stage_fingerprint" "$(date -u +%FT%TZ)" >"$stage_dir/RUNNING.tmp.$$"
mv "$stage_dir/RUNNING.tmp.$$" "$stage_dir/RUNNING"

mark_failed() {
  status=$?
  printf '{"exit_code":%d,"stage_fingerprint":"%s","failed_utc":"%s"}\n' \
    "$status" "$stage_fingerprint" "$(date -u +%FT%TZ)" >"$stage_dir/FAILED.tmp.$$"
  mv "$stage_dir/FAILED.tmp.$$" "$stage_dir/FAILED"
  rm -f "$stage_dir/RUNNING"
  exit "$status"
}
trap mark_failed ERR INT TERM HUP

"$@" >"$log_file" 2>&1
printf '{"stage_fingerprint":"%s","completed_utc":"%s"}\n' \
  "$stage_fingerprint" "$(date -u +%FT%TZ)" >"$stage_dir/SUCCESS.tmp.$$"
mv "$stage_dir/SUCCESS.tmp.$$" "$stage_dir/SUCCESS"
rm -f "$stage_dir/RUNNING" "$stage_dir/FAILED"
trap - ERR INT TERM HUP
