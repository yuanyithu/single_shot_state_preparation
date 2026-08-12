#!/usr/bin/env bash
set -euo pipefail

# The one nd-3 command that runs before the production plan exists.
#
# It measures kappa_m and c_m for every m on the machine that will spend the
# budget, so that the section 6 allocation rule can be evaluated on real costs.
# exp105 had no such command -- its only path to nd-3 costs was `preflight`,
# which already requires the frozen plan -- so it evaluated the rule on macmini
# numbers and its resource gate blocked at 5,368 core-hours against a cap of 800.
#
# Outcome-blind: it times the decode and its replay, compares them, and discards
# the comparison. Nothing about which trials failed is returned or stored.
#
# One worker, deliberately. The projection this feeds assumes a single-process
# per-trial cost, and timing under contention would measure the machine's load
# rather than the decoder.

if [[ $# -ne 3 ]]; then
  echo "usage: run_cost_benchmark.sh DEPLOYMENT_ROOT MANIFEST_SHA256 RUN_ROOT" >&2
  exit 64
fi

deployment_root=$1
deployment_manifest_sha256=$2
run_root=$3
config=data/expander_code/exp106/config/noisy_mc.pilot.remote.v1.json
module=data.expander_code.exp106.exp106_pipeline.remote_cli

if [[ -f $run_root/validation/cost_benchmark.json ]]; then
  echo "cost benchmark already exists and is immutable: $run_root" >&2
  exit 0
fi

python -B -m "$module" cost-benchmark \
  --config "$config" \
  --run-root "$run_root" \
  --deployment-root "$deployment_root" \
  --deployment-manifest-sha256 "$deployment_manifest_sha256"

echo "exp106 cost benchmark complete"
