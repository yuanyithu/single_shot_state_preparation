#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: run_v0_stage.sh STAGE STAGE_DIR LOG_FILE -- COMMAND [ARGS...]" >&2
  exit 64
}

[[ $# -ge 5 ]] || usage
stage=$1
stage_dir=$2
log_file=$3
shift 3
[[ $1 == -- ]] || usage
shift

case "$stage" in
  artifacts|manifest|preflight|node|analysis) ;;
  *) usage ;;
esac
[[ ${EXP102_SOURCE_COMMIT:-} =~ ^[0-9a-f]{40}$ ]] || {
  echo "V0 stage must run through run_verified_source.sh" >&2
  exit 66
}
[[ $# -gt 0 ]] || usage

mkdir -p "$stage_dir" "$(dirname "$log_file")"
exec 9>"$stage_dir/stage.lock"
flock -n 9 || exit 73
for marker in RUNNING SUCCESS FAILED; do
  [[ ! -e "$stage_dir/$marker" ]] || {
    echo "V0 stage marker already exists: $stage_dir/$marker" >&2
    exit 74
  }
done
[[ ! -e "$log_file" ]] || {
  echo "V0 stage log already exists: $log_file" >&2
  exit 75
}

command_sha256=$(printf '%q\0' "$@" | sha256sum | awk '{print $1}')
printf '{"stage":"%s","source_commit":"%s","command_sha256":"%s"}\n' \
  "$stage" "$EXP102_SOURCE_COMMIT" "$command_sha256" >"$stage_dir/RUNNING"

mark_failed() {
  status=$?
  printf '{"stage":"%s","source_commit":"%s","command_sha256":"%s","exit_code":%d}\n' \
    "$stage" "$EXP102_SOURCE_COMMIT" "$command_sha256" "$status" >"$stage_dir/FAILED"
  rm -f "$stage_dir/RUNNING"
  exit "$status"
}
trap mark_failed ERR INT TERM HUP

"$@" >"$log_file" 2>&1
printf '{"stage":"%s","source_commit":"%s","command_sha256":"%s"}\n' \
  "$stage" "$EXP102_SOURCE_COMMIT" "$command_sha256" >"$stage_dir/SUCCESS"
rm -f "$stage_dir/RUNNING"
