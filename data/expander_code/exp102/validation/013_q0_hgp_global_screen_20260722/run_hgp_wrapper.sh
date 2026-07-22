#!/usr/bin/env bash
set -euo pipefail

workflow_module=data.expander_code.exp102.validation.013_q0_hgp_global_screen_20260722.workflow

usage() {
  cat >&2 <<'EOF'
usage: run_hgp_wrapper.sh STAGE STAGE_DIR LOG_FILE \
  [--require-success PATH]... -- python -m WORKFLOW_MODULE ACTION [ARGS...]

STAGE is one of: build-schedule, build-artifacts, preflight,
freeze-control, screen, analyze. The wrapper must run inside
run_verified_source.sh, which exports EXP102_SOURCE_COMMIT after verifying the
deployed archive. The wrapper derives the stage fingerprint from the exact
command and prerequisite marker hashes; callers cannot supply it.
EOF
  exit 64
}

[[ $# -ge 7 ]] || usage

stage=$1
stage_dir=$2
log_file=$3
shift 3

case "$stage" in
  build-schedule|build-artifacts|preflight|freeze-control|screen|analyze) ;;
  *)
    echo "unknown HGP screen stage: $stage" >&2
    exit 64
    ;;
esac

[[ ${EXP102_SOURCE_COMMIT:-} =~ ^[0-9a-f]{40}$ ]] || {
  echo "run_hgp_wrapper.sh must be launched by run_verified_source.sh" >&2
  exit 66
}

prerequisites=()
while [[ $# -gt 0 && $1 == --require-success ]]; do
  [[ $# -ge 2 ]] || usage
  prerequisites+=("$2")
  shift 2
done
[[ $# -ge 5 && $1 == -- ]] || usage
shift

[[ $1 == python && $2 == -m && $3 == "$workflow_module" ]] || {
  echo "HGP stage command must invoke the frozen workflow module directly" >&2
  exit 68
}
action=$4
for argument in "$@"; do
  [[ $argument != -h && $argument != --help ]] || {
    echo "help commands cannot create immutable HGP stage markers" >&2
    exit 68
  }
done
case "$stage:$action" in
  build-schedule:build-schedule|build-artifacts:build-artifacts|\
  preflight:preflight-node|preflight:combine-preflight|\
  freeze-control:build-control|screen:run-node|analyze:analyze) ;;
  *)
    echo "workflow action $action is invalid for stage $stage" >&2
    exit 68
    ;;
esac

if [[ $stage == build-schedule && ${#prerequisites[@]} -ne 0 ]]; then
  echo "build-schedule must not inherit authority from another stage" >&2
  exit 67
fi
if [[ $stage != build-schedule && ${#prerequisites[@]} -eq 0 ]]; then
  echo "$stage requires at least one prerequisite SUCCESS marker" >&2
  exit 67
fi

case "$stage" in
  build-artifacts) allowed_prerequisite='build-schedule' ;;
  preflight) allowed_prerequisite='build-artifacts|preflight' ;;
  freeze-control) allowed_prerequisite='preflight' ;;
  screen) allowed_prerequisite='freeze-control' ;;
  analyze) allowed_prerequisite='screen' ;;
  build-schedule) allowed_prerequisite='' ;;
esac

prerequisite_sha256=()
for marker in "${prerequisites[@]}"; do
  [[ -f $marker && ${marker##*/} == SUCCESS ]] || {
    echo "missing prerequisite SUCCESS marker: $marker" >&2
    exit 69
  }
  metadata=$(python -c '
import json, re, sys
with open(sys.argv[1], encoding="ascii") as handle:
    value = json.load(handle)
stage = value.get("stage")
source = value.get("source_commit")
fingerprint = value.get("stage_fingerprint")
if not isinstance(stage, str):
    raise SystemExit(2)
if source != sys.argv[2]:
    raise SystemExit(3)
if not isinstance(fingerprint, str) or re.fullmatch(r"[0-9a-f]{64}", fingerprint) is None:
    raise SystemExit(4)
print(stage)
' "$marker" "$EXP102_SOURCE_COMMIT") || {
    echo "invalid prerequisite SUCCESS marker: $marker" >&2
    exit 69
  }
  [[ $metadata =~ ^($allowed_prerequisite)$ ]] || {
    echo "invalid prerequisite stage $metadata for $stage" >&2
    exit 69
  }
  prerequisite_sha256+=("$(sha256sum "$marker" | awk '{print $1}')")
done

stage_fingerprint=$(python -c '
import hashlib, json, sys
stage, source, separator, *values = sys.argv[1:]
index = values.index(separator)
prerequisites = values[:index]
command = values[index + 1:]
identity = {
    "stage": stage,
    "source_commit": source,
    "prerequisite_success_sha256": prerequisites,
    "command": command,
}
payload = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
print(hashlib.sha256(payload.encode("ascii")).hexdigest())
' "$stage" "$EXP102_SOURCE_COMMIT" __COMMAND__ \
  "${prerequisite_sha256[@]}" __COMMAND__ "$@")

[[ ! -e $log_file ]] || {
  echo "HGP screen log already exists: $log_file" >&2
  exit 70
}
mkdir -p "$stage_dir" "$(dirname "$log_file")"
exec 9>"$stage_dir/stage.lock"
flock -n 9 || exit 73

for marker in RUNNING SUCCESS FAILED; do
  if [[ -e "$stage_dir/$marker" ]]; then
    recorded=$(python -c '
import json, sys
with open(sys.argv[1], encoding="ascii") as handle:
    value = json.load(handle)
print(value.get("stage_fingerprint", ""))
' "$stage_dir/$marker")
    if [[ $recorded != "$stage_fingerprint" ]]; then
      echo "HGP screen marker identity conflict: $stage_dir/$marker" >&2
      exit 74
    fi
  fi
done

if [[ -e "$stage_dir/SUCCESS" ]]; then
  echo "HGP screen SUCCESS marker is immutable" >&2
  exit 75
fi
if [[ -e "$stage_dir/RUNNING" ]]; then
  echo "stale HGP screen RUNNING marker requires a fresh deployment" >&2
  exit 76
fi
if [[ -e "$stage_dir/FAILED" ]]; then
  echo "HGP screen FAILED marker requires a fresh deployment" >&2
  exit 77
fi

prerequisite_json=$(python -c '
import json, sys
print(json.dumps(sys.argv[1:], separators=(",", ":")))
' "${prerequisite_sha256[@]}")
printf '{"pid":%d,"stage":"%s","source_commit":"%s","stage_fingerprint":"%s","prerequisite_success_sha256":%s,"started_utc":"%s"}\n' \
  "$$" "$stage" "$EXP102_SOURCE_COMMIT" "$stage_fingerprint" \
  "$prerequisite_json" "$(date -u +%FT%TZ)" >"$stage_dir/RUNNING.tmp.$$"
mv "$stage_dir/RUNNING.tmp.$$" "$stage_dir/RUNNING"

mark_failed() {
  status=$?
  printf '{"exit_code":%d,"stage":"%s","source_commit":"%s","stage_fingerprint":"%s","prerequisite_success_sha256":%s,"failed_utc":"%s"}\n' \
    "$status" "$stage" "$EXP102_SOURCE_COMMIT" "$stage_fingerprint" \
    "$prerequisite_json" "$(date -u +%FT%TZ)" >"$stage_dir/FAILED.tmp.$$"
  mv "$stage_dir/FAILED.tmp.$$" "$stage_dir/FAILED"
  rm -f "$stage_dir/RUNNING"
  exit "$status"
}
trap mark_failed ERR INT TERM HUP

"$@" >"$log_file" 2>&1
printf '{"stage":"%s","source_commit":"%s","stage_fingerprint":"%s","prerequisite_success_sha256":%s,"completed_utc":"%s"}\n' \
  "$stage" "$EXP102_SOURCE_COMMIT" "$stage_fingerprint" \
  "$prerequisite_json" "$(date -u +%FT%TZ)" >"$stage_dir/SUCCESS.tmp.$$"
mv "$stage_dir/SUCCESS.tmp.$$" "$stage_dir/SUCCESS"
rm -f "$stage_dir/RUNNING"
trap - ERR INT TERM HUP
