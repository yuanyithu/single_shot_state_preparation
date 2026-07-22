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
archive_marker=$deployment_root/ARCHIVE_SHA256
manifest=$deployment_root/SOURCE_MANIFEST.json
commit_marker=$deployment_root/SOURCE_COMMIT
verify_relative=data/expander_code/exp102/validation/002_numba_smoke_20260719/run_verified_source.sh
module=data.expander_code.exp102.validation.013_q0_hgp_global_screen_20260722.orchestrate_hgp

[[ -d $deployment_root && -d $deployment_root/source ]] || exit 67
[[ -f $archive && -f $archive_marker && -f $manifest && -f $commit_marker ]] \
  || exit 67
[[ -d $server_root/logs ]] || exit 67
printf '%s  %s\n' "$archive_sha256" "$archive" | sha256sum -c - >/dev/null
printf '%s  %s\n' "$manifest_sha256" "$manifest" | sha256sum -c - >/dev/null
[[ $(tr -d '\r\n' < "$commit_marker") == "$source_commit" ]] || exit 67
[[ $(tr -d '\r\n' < "$archive_marker") == "$archive_sha256" ]] || exit 67
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
  expected_attestation=$run_root/control/HGP_LOCAL_PREFLIGHT_ATTESTATION.json
  [[ $local_attestation == "$expected_attestation" ]] || {
    echo "HGP measurement attestation path is not canonical" >&2
    exit 65
  }
  [[ -f $local_attestation && ! -L $local_attestation ]] || {
    echo "HGP measurement attestation is absent or is a symlink" >&2
    exit 67
  }
  printf '%s  %s\n' "$local_attestation_sha256" "$local_attestation" \
    | sha256sum -c - >/dev/null
fi

token=$(printf '%s' "$run_id" | sha256sum | cut -c1-8)
log=$server_root/logs/${run_id}_hgp_orchestrator_${phase}.log
launch_guard=$server_root/logs/.${run_id}_hgp_orchestrator_${token}_${phase}.launch
[[ ! -e $log ]] || {
  echo "HGP orchestrator log already exists: $log" >&2
  exit 69
}

printf -v verified_arguments '%q ' \
  "$deployment_root" "$source_commit" "$archive_sha256" "$manifest_sha256" \
  conda run -n 11 --no-capture-output /usr/bin/setsid python -m "$module" \
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

if ! mkdir -m 700 -- "$launch_guard"; then
  echo "HGP orchestrator launch guard already exists: $launch_guard" >&2
  exit 69
fi

# The guard is intentionally retained after exit: either phase is immutable
# and a failed detached bootstrap requires a fresh run/deployment.
command_sha256=$(printf '%s' "$inner" | sha256sum | awk '{print $1}')
if [[ $phase == measurement ]]; then
  attestation_binding=\"$local_attestation_sha256\"
else
  attestation_binding=null
fi
launch_metadata_tmp=$launch_guard/LAUNCH.json.tmp.$$
printf '{"archive_sha256":"%s","command_sha256":"%s","launcher_version":"exp102.q0_hgp.nd0_nohup_setsid.v1","local_attestation_sha256":%s,"manifest_sha256":"%s","phase":"%s","run_id":"%s","source_commit":"%s"}\n' \
  "$archive_sha256" "$command_sha256" "$attestation_binding" \
  "$manifest_sha256" "$phase" "$run_id" "$source_commit" \
  >"$launch_metadata_tmp"
mv -- "$launch_metadata_tmp" "$launch_guard/LAUNCH.json"

(set -o noclobber; : >"$log") || {
  echo "HGP orchestrator log was created concurrently: $log" >&2
  exit 69
}
persistence_token=exp102_q0_hgp_nd0_nohup_setsid_v1
EXP102_HGP_ORCHESTRATOR_PERSISTENCE=$persistence_token \
EXP102_HGP_ORCHESTRATOR_GUARD=$launch_guard \
  /usr/bin/nohup /usr/bin/setsid /bin/bash -lc "$inner" \
  </dev/null >>"$log" 2>&1 &
bootstrap_pid=$!
printf '%s\n' "$bootstrap_pid" >"$launch_guard/BOOTSTRAP_PID.tmp.$$"
mv -- "$launch_guard/BOOTSTRAP_PID.tmp.$$" "$launch_guard/BOOTSTRAP_PID"

orchestrator_pid_file=$launch_guard/ORCHESTRATOR_PID
for _ in {1..600}; do
  [[ -s $orchestrator_pid_file ]] && break
  kill -0 "$bootstrap_pid" 2>/dev/null || break
  sleep 0.1
done
if [[ ! -s $orchestrator_pid_file ]]; then
  kill -TERM -- "-$bootstrap_pid" 2>/dev/null || true
  echo "HGP detached orchestrator did not publish its PID: $log" >&2
  exit 70
fi
orchestrator_pid=$(tr -d '\r\n' <"$orchestrator_pid_file")
[[ $orchestrator_pid =~ ^[1-9][0-9]*$ ]] || exit 70
kill -0 "$orchestrator_pid" 2>/dev/null || {
  echo "HGP detached orchestrator exited during launch: $log" >&2
  exit 70
}
printf '{"guard":"%s","log":"%s","orchestrator_pid":%s,"phase":"%s","status":"LAUNCHED"}\n' \
  "$launch_guard" "$log" "$orchestrator_pid" "$phase"
