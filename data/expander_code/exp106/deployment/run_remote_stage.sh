#!/usr/bin/env bash
set -euo pipefail

# exp106 runs as a single stage. Each step is skipped if its immutable artifact
# already exists, so an interrupted screen session resumes without redoing work
# and without ever overwriting evidence.

if [[ $# -ne 3 ]]; then
  echo "usage: run_remote_stage.sh DEPLOYMENT_ROOT MANIFEST_SHA256 RUN_ROOT" >&2
  exit 64
fi

deployment_root=$1
deployment_manifest_sha256=$2
run_root=$3
config=data/expander_code/exp106/config/noisy_mc.remote.v1.json
module=data.expander_code.exp106.exp106_pipeline.remote_cli
common=(
  --config "$config"
  --run-root "$run_root"
  --deployment-root "$deployment_root"
  --deployment-manifest-sha256 "$deployment_manifest_sha256"
)

if [[ ! -f $run_root/validation/scan.json ]]; then
  python -B -m "$module" scan "${common[@]}" --num-workers 72
fi
if [[ ! -f $run_root/validation/replay.json ]]; then
  python -B -m "$module" replay "${common[@]}" --num-workers 72
fi
if [[ ! -f $run_root/aggregate/ensemble_crossing.npz ]]; then
  python -B -m "$module" aggregate "${common[@]}"
fi

echo "exp106 remote stage complete"
