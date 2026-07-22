#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 || $# -gt 7 ]]; then
  echo "usage: launch_hgp_orchestrator.sh RUN_ID COMMIT ARCHIVE_SHA256 MANIFEST_SHA256 preflight" >&2
  echo "   or: launch_hgp_orchestrator.sh RUN_ID COMMIT ARCHIVE_SHA256 MANIFEST_SHA256 measurement ATTESTATION_PATH ATTESTATION_SHA256" >&2
  exit 64
fi

run_id=$1
source_commit=$2
archive_sha256=$3
manifest_sha256=$4
phase=$5

[[ $run_id =~ ^[A-Za-z0-9._-]+$ ]] || exit 65
[[ $source_commit =~ ^[0-9a-f]{40}$ ]] || exit 65
[[ $archive_sha256 =~ ^[0-9a-f]{64}$ ]] || exit 65
[[ $manifest_sha256 =~ ^[0-9a-f]{64}$ ]] || exit 65
[[ $phase == preflight || $phase == measurement ]] || exit 65
if [[ $phase == preflight && $# -ne 5 ]]; then
  exit 65
fi
if [[ $phase == measurement ]]; then
  [[ $# -eq 7 && $7 =~ ^[0-9a-f]{64}$ ]] || exit 65
  local_attestation=$6
  local_attestation_sha256=$7
fi
[[ ${HOSTNAME%%.*} == nd-0 ]] || {
  echo "HGP orchestrator launcher must run on nd-0" >&2
  exit 66
}

server_root=$HOME/.single_shot
deployment_root=$server_root/repos/$run_id
run_root=$server_root/runs/$run_id
archive=$deployment_root/SOURCE.tar
manifest=$deployment_root/SOURCE_MANIFEST.json
commit_marker=$deployment_root/SOURCE_COMMIT
verify_relative=data/expander_code/exp102/validation/002_numba_smoke_20260719/run_verified_source.sh
module=data.expander_code.exp102.validation.013_q0_hgp_global_screen_20260722.orchestrate_hgp

[[ -d $deployment_root && -d $deployment_root/source ]] || exit 67
[[ -f $archive && -f $manifest && -f $commit_marker ]] || exit 67
[[ -d $server_root/logs ]] || exit 67
printf '%s  %s\n' "$archive_sha256" "$archive" | sha256sum -c - >/dev/null
printf '%s  %s\n' "$manifest_sha256" "$manifest" | sha256sum -c - >/dev/null
[[ $(tr -d '\r\n' < "$commit_marker") == "$source_commit" ]] || exit 67
if [[ $phase == preflight ]]; then
  [[ ! -e $run_root ]] || {
    echo "HGP preflight requires a fresh run root" >&2
    exit 68
  }
else
  [[ -d $run_root ]] || {
    echo "HGP measurement requires a completed preflight run root" >&2
    exit 68
  }
fi

token=$(printf '%s' "$run_id" | sha256sum | cut -c1-8)
session=e102h_orchestrator_${token}_${phase}
log=$server_root/logs/${run_id}_hgp_orchestrator_${phase}.log
[[ ! -e $log ]] || {
  echo "HGP orchestrator log already exists: $log" >&2
  exit 69
}
if screen -S "$session" -Q select . >/dev/null 2>&1; then
  echo "HGP orchestrator screen already exists: $session" >&2
  exit 69
fi

printf -v verified_arguments '%q ' \
  "$deployment_root" "$source_commit" "$archive_sha256" "$manifest_sha256" \
  conda run -n 11 --no-capture-output python -m "$module" \
  --run-id "$run_id" --source-commit "$source_commit" \
  --archive-sha256 "$archive_sha256" \
  --source-manifest-sha256 "$manifest_sha256" --phase "$phase"
if [[ $phase == measurement ]]; then
  printf -v attestation_arguments '%q ' \
    --local-attestation "$local_attestation" \
    --local-attestation-sha256 "$local_attestation_sha256"
  verified_arguments+=$attestation_arguments
fi
printf -v inner \
  'set -euo pipefail\nprintf '\''%%s  %%s\\n'\'' %q %q | sha256sum -c - >/dev/null\ntar -xOf %q %q | bash -s -- %s' \
  "$archive_sha256" "$archive" "$archive" "$verify_relative" \
  "$verified_arguments"

screen -dmS "$session" bash -lc "$inner > $(printf '%q' "$log") 2>&1"
printf '{"log":"%s","phase":"%s","screen":"%s","status":"LAUNCHED"}\n' \
  "$log" "$phase" "$session"
