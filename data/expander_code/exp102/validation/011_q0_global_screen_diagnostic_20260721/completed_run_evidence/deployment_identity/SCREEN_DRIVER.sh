#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: SCREEN_DRIVER.sh RUN_ID COMMIT ARCHIVE_SHA256 MANIFEST_SHA256" >&2
  exit 64
fi

RUN_ID=$1
COMMIT=$2
ARCHIVE_SHA=$3
MANIFEST_SHA=$4
ROOT=$HOME/.single_shot
RUN_ROOT=$ROOT/runs/$RUN_ID
DEPLOY=$ROOT/repos/$RUN_ID
MOD=data.expander_code.exp102.validation.011_q0_global_screen_diagnostic_20260721
REG=data/expander_code/exp102/registry/registry.json
CFG=data/expander_code/exp102/config/q0_global.screen_diagnostic.v1.json
VERIFY_REL=data/expander_code/exp102/validation/002_numba_smoke_20260719/run_verified_source.sh
SCHEDULE=$RUN_ROOT/control/SCREEN_DIAGNOSTIC_24H_SCHEDULE.json
RUNTIME=$RUN_ROOT/control/screen_runtime_consensus.json
DIGEST=$RUN_ROOT/control/screen_digest_consensus.json
PREFLIGHT=$RUN_ROOT/control/screen_preflight_report.json
BIAS_CONTROL=$RUN_ROOT/control/screen_bias_input.json
MEASUREMENT_CONTROL=$RUN_ROOT/control/screen_measurement_input.json

finish() {
  rc=$?
  printf 'SCREEN_DRIVER_EXIT status=%s unix=%s\n' "$rc" "$(date +%s)"
}
trap finish EXIT

run_verified() {
  printf '%s  %s\n' "$ARCHIVE_SHA" "$DEPLOY/SOURCE.tar" \
    | sha256sum -c - >/dev/null
  tar -xOf "$DEPLOY/SOURCE.tar" "$VERIFY_REL" | bash -s -- \
    "$DEPLOY" "$COMMIT" "$ARCHIVE_SHA" "$MANIFEST_SHA" \
    conda run -n 11 --no-capture-output "$@"
}

run_verified_on_node() {
  local node=$1 remote_command
  shift
  printf -v remote_command '%q ' bash -s -- \
    "$DEPLOY" "$COMMIT" "$ARCHIVE_SHA" "$MANIFEST_SHA" "$@"
  ssh -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=4 \
    "$node" "$remote_command" <<'REMOTE_VERIFY'
set -euo pipefail
deployment_root=$1
expected_commit=$2
expected_archive_sha256=$3
expected_manifest_sha256=$4
shift 4
verify_rel=data/expander_code/exp102/validation/002_numba_smoke_20260719/run_verified_source.sh
printf '%s  %s\n' "$expected_archive_sha256" "$deployment_root/SOURCE.tar" \
  | sha256sum -c - >/dev/null
tar -xOf "$deployment_root/SOURCE.tar" "$verify_rel" | bash -s -- \
  "$deployment_root" "$expected_commit" "$expected_archive_sha256" \
  "$expected_manifest_sha256" conda run -n 11 --no-capture-output "$@"
REMOTE_VERIFY
}

[[ $RUN_ID =~ ^[A-Za-z0-9._-]+$ ]]
[[ $COMMIT =~ ^[0-9a-f]{40}$ ]]
[[ $ARCHIVE_SHA =~ ^[0-9a-f]{64}$ ]]
[[ $MANIFEST_SHA =~ ^[0-9a-f]{64}$ ]]
[[ -d $DEPLOY && ! -e $RUN_ROOT ]]
printf '%s  %s\n' "$ARCHIVE_SHA" "$DEPLOY/SOURCE.tar" | sha256sum -c - >/dev/null
printf '%s  %s\n' "$MANIFEST_SHA" "$DEPLOY/SOURCE_MANIFEST.json" | sha256sum -c - >/dev/null
[[ $(tr -d '\r\n' < "$DEPLOY/SOURCE_COMMIT") == "$COMMIT" ]]

run_verified python -m "$MOD.prepare_screen_schedule" \
  --registry "$REG" --config "$CFG" --source-commit "$COMMIT" \
  --archive-sha256 "$ARCHIVE_SHA" --manifest-sha256 "$MANIFEST_SHA" \
  --output "$SCHEDULE"
SCHEDULE_SHA=$(sha256sum "$SCHEDULE" | awk '{print $1}')

run_verified python -m "$MOD.orchestrate_screen_preflight" \
  --run-id "$RUN_ID" --source-commit "$COMMIT" \
  --archive-sha256 "$ARCHIVE_SHA" --manifest-sha256 "$MANIFEST_SHA" \
  --schedule "$SCHEDULE" --schedule-file-sha256 "$SCHEDULE_SHA"

run_verified python -m "$MOD.prepare_screen_controls" bias \
  --registry "$REG" --config "$CFG" --source-commit "$COMMIT" \
  --schedule "$SCHEDULE" --runtime-report "$RUNTIME" \
  --output "$BIAS_CONTROL"
BIAS_SHA=$(sha256sum "$BIAS_CONTROL" | awk '{print $1}')
BIAS_OWNERSHIP=$RUN_ROOT/control/screen_ownership_${BIAS_SHA:0:12}.json

run_verified python -m "$MOD.orchestrate_screen_stage" \
  --run-id "$RUN_ID" --source-commit "$COMMIT" \
  --archive-sha256 "$ARCHIVE_SHA" --manifest-sha256 "$MANIFEST_SHA" \
  --schedule "$SCHEDULE" --schedule-file-sha256 "$SCHEDULE_SHA" \
  --preflight-report "$PREFLIGHT" --runtime-report "$RUNTIME" \
  --control "$BIAS_CONTROL"

run_verified_on_node nd-3 python -m "$MOD.prepare_screen_controls" measurement \
  --registry "$REG" --config "$CFG" --source-commit "$COMMIT" \
  --schedule "$SCHEDULE" --runtime-report "$RUNTIME" --run-root "$RUN_ROOT" \
  --raw-root "$RUN_ROOT/screen_diagnostic/raw" --deployment-root "$DEPLOY" \
  --bias-control "$BIAS_CONTROL" --bias-ownership "$BIAS_OWNERSHIP" \
  --output "$MEASUREMENT_CONTROL"
MEASUREMENT_SHA=$(sha256sum "$MEASUREMENT_CONTROL" | awk '{print $1}')
MEASUREMENT_OWNERSHIP=$RUN_ROOT/control/screen_ownership_${MEASUREMENT_SHA:0:12}.json

run_verified python -m "$MOD.orchestrate_screen_stage" \
  --run-id "$RUN_ID" --source-commit "$COMMIT" \
  --archive-sha256 "$ARCHIVE_SHA" --manifest-sha256 "$MANIFEST_SHA" \
  --schedule "$SCHEDULE" --schedule-file-sha256 "$SCHEDULE_SHA" \
  --preflight-report "$PREFLIGHT" --runtime-report "$RUNTIME" \
  --control "$MEASUREMENT_CONTROL"

run_verified_on_node nd-3 python -m "$MOD.analyze_screen" \
  --run-root "$RUN_ROOT" --raw-root "$RUN_ROOT/screen_diagnostic/raw" \
  --deployment-root "$DEPLOY" --schedule "$SCHEDULE" \
  --runtime-report "$RUNTIME" --digest-report "$DIGEST" \
  --preflight-report "$PREFLIGHT" --bias-control "$BIAS_CONTROL" \
  --bias-ownership "$BIAS_OWNERSHIP" \
  --measurement-control "$MEASUREMENT_CONTROL" \
  --measurement-ownership "$MEASUREMENT_OWNERSHIP" --registry "$REG" \
  --config "$CFG" --report-output "$RUN_ROOT/control/screen_report.json" \
  --decision-output "$RUN_ROOT/control/screen_decision.json" \
  --package-output "$RUN_ROOT/control/screen_terminal_package.json" \
  --num-workers 32

echo SCREEN_DRIVER_COMPLETE
