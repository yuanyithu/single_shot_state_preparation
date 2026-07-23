#!/usr/bin/env bash
set -euo pipefail

module=data.expander_code.exp102.exp102_pipeline.q0_logical_stratified_v0

usage() {
  cat >&2 <<'EOF'
usage: run_v0_stage.sh STAGE STAGE_DIR LOG_FILE [--require-success PATH]... \
  -- python -m data.expander_code.exp102.exp102_pipeline.q0_logical_stratified_v0 ACTION [ARGS...]

Stages/actions are artifacts/prepare-artifacts, manifest/build-manifest,
audit/audit-artifacts, preflight/preflight, node/run-node, and analysis/analyze.
This wrapper must execute under run_verified_source.sh.
EOF
  exit 64
}

[[ $# -ge 7 ]] || usage
stage=$1
stage_dir=$2
log_file=$3
shift 3

case "$stage" in
  artifacts|manifest|audit|preflight|node|analysis) ;;
  *) usage ;;
esac
[[ ${EXP102_SOURCE_COMMIT:-} =~ ^[0-9a-f]{40}$ ]] || {
  echo "V0v2 stage must run through run_verified_source.sh" >&2
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
[[ $1 == python && $2 == -m && $3 == "$module" ]] || {
  echo "V0v2 stage command must invoke the frozen module directly" >&2
  exit 67
}
action=$4
case "$stage:$action" in
  artifacts:prepare-artifacts|manifest:build-manifest|audit:audit-artifacts|\
  preflight:preflight|node:run-node|analysis:analyze) ;;
  *)
    echo "V0v2 action $action is invalid for stage $stage" >&2
    exit 67
    ;;
esac

case "$stage" in
  artifacts)
    [[ ${#prerequisites[@]} -eq 0 ]] || exit 68
    expected_prerequisite=''
    ;;
  manifest)
    [[ ${#prerequisites[@]} -eq 1 ]]
    expected_prerequisite='artifacts'
    ;;
  audit)
    [[ ${#prerequisites[@]} -eq 1 ]]
    expected_prerequisite='manifest'
    ;;
  preflight)
    [[ ${#prerequisites[@]} -ge 1 ]]
    expected_prerequisite='audit'
    ;;
  node)
    [[ ${#prerequisites[@]} -ge 1 ]]
    expected_prerequisite='preflight'
    ;;
  analysis)
    [[ ${#prerequisites[@]} -ge 2 ]]
    expected_prerequisite='node'
    ;;
esac

prerequisite_hashes=()
for marker in "${prerequisites[@]}"; do
  [[ -f $marker && ${marker##*/} == SUCCESS ]] || {
    echo "missing V0v2 prerequisite SUCCESS marker: $marker" >&2
    exit 68
  }
  marker_stage=$(sed -n 's/.*"stage":"\([^"]*\)".*/\1/p' "$marker")
  marker_source=$(sed -n 's/.*"source_commit":"\([0-9a-f]*\)".*/\1/p' "$marker")
  [[ $marker_stage == "$expected_prerequisite" && $marker_source == "$EXP102_SOURCE_COMMIT" ]] || {
    echo "invalid V0v2 prerequisite marker: $marker" >&2
    exit 68
  }
  prerequisite_hashes+=("$(sha256sum "$marker" | awk '{print $1}')")
done

[[ ! -e $log_file ]] || {
  echo "V0v2 stage log already exists: $log_file" >&2
  exit 69
}
mkdir -p "$stage_dir" "$(dirname "$log_file")"
exec 9>"$stage_dir/stage.lock"
flock -n 9 || exit 73
for marker in RUNNING SUCCESS FAILED; do
  [[ ! -e "$stage_dir/$marker" ]] || {
    echo "V0v2 stage marker already exists: $stage_dir/$marker" >&2
    exit 74
  }
done

command_sha256=$(printf '%q\0' "$@" | sha256sum | awk '{print $1}')
stage_fingerprint=$(printf '%s\0' "$stage" "$EXP102_SOURCE_COMMIT" \
  "$command_sha256" "${prerequisite_hashes[@]}" | sha256sum | awk '{print $1}')
printf '{"stage":"%s","source_commit":"%s","stage_fingerprint":"%s"}\n' \
  "$stage" "$EXP102_SOURCE_COMMIT" "$stage_fingerprint" >"$stage_dir/RUNNING"

mark_failed() {
  status=$?
  printf '{"exit_code":%d,"stage":"%s","source_commit":"%s","stage_fingerprint":"%s"}\n' \
    "$status" "$stage" "$EXP102_SOURCE_COMMIT" "$stage_fingerprint" >"$stage_dir/FAILED"
  rm -f "$stage_dir/RUNNING"
  exit "$status"
}
trap mark_failed ERR INT TERM HUP

"$@" >"$log_file" 2>&1
log_sha256=$(sha256sum "$log_file" | awk '{print $1}')
printf '{"log_sha256":"%s","stage":"%s","source_commit":"%s","stage_fingerprint":"%s"}\n' \
  "$log_sha256" "$stage" "$EXP102_SOURCE_COMMIT" "$stage_fingerprint" >"$stage_dir/SUCCESS"
rm -f "$stage_dir/RUNNING"
trap - ERR INT TERM HUP
