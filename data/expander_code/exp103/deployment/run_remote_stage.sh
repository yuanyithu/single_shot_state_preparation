#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: run_remote_stage.sh STAGE DEPLOYMENT_ROOT MANIFEST_SHA256 RUN_ROOT" >&2
  exit 64
fi

stage=$1
deployment_root=$2
deployment_manifest_sha256=$3
run_root=$4
config=data/expander_code/exp103/config/decoder_mc.remote.v1.json
preflight=$run_root/validation/remote_resource_preflight.json
common=(
  --config "$config"
  --run-root "$run_root"
  --deployment-root "$deployment_root"
  --deployment-manifest-sha256 "$deployment_manifest_sha256"
)

case "$stage" in
  stage1)
    if [[ ! -f $run_root/control/SCAN_STAGE1.json ]]; then
      python -B -m data.expander_code.exp103.exp103_pipeline.remote_cli scan \
        "${common[@]}" --stage stage1 --preflight-report "$preflight" \
        --num-workers 64
    fi
    if [[ ! -f $run_root/raw/stage1/REPLAY_STAGE1.json ]]; then
      python -B -m data.expander_code.exp103.exp103_pipeline.remote_cli replay \
        "${common[@]}" --stage stage1 --preflight-report "$preflight" \
        --num-workers 64
    fi
    if [[ ! -f $run_root/final_results/stage1_aggregate.npz ]]; then
      python -B -m data.expander_code.exp103.exp103_pipeline.remote_cli aggregate \
        "${common[@]}" --scope stage1 --preflight-report "$preflight"
    fi
    if [[ ! -f $run_root/validation/stage1_technical_report.json ]]; then
      python -B -m data.expander_code.exp103.exp103_pipeline.remote_cli \
        stage1-technical "${common[@]}" --preflight-report "$preflight"
    fi
    ;;
  stage2)
    if [[ ! -f $run_root/control/SCAN_STAGE2.json ]]; then
      python -B -m data.expander_code.exp103.exp103_pipeline.remote_cli scan \
        "${common[@]}" --stage stage2 --preflight-report "$preflight" \
        --num-workers 64
    fi
    if [[ ! -f $run_root/raw/stage2/REPLAY_STAGE2.json ]]; then
      python -B -m data.expander_code.exp103.exp103_pipeline.remote_cli replay \
        "${common[@]}" --stage stage2 --preflight-report "$preflight" \
        --num-workers 64
    fi
    if [[ ! -f $run_root/final_results/decoder_crossing.npz ]]; then
      python -B -m data.expander_code.exp103.exp103_pipeline.remote_cli aggregate \
        "${common[@]}" --scope final --preflight-report "$preflight"
    fi
    ;;
  *)
    echo "stage must be stage1 or stage2" >&2
    exit 64
    ;;
esac
