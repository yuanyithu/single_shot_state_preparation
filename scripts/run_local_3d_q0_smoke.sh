#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="q0_smoke_local_$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="$PROJECT_ROOT/data/3d_toric_code/without_measurement_noise/$RUN_ID"
OUTPUT_STEM="scan_result_multi_L_3d_toric_q0_smoke"

conda run -n 12 python "$PROJECT_ROOT/src/production_chunked_scan.py" submit \
  --run-root "$RUN_ROOT" \
  --code-family 3d_toric \
  --workers 4 \
  --chunk-size 16 \
  --num-disorder-samples-total 16 \
  --data-error-probabilities "0.0200,0.0500,0.0800" \
  --lattice-sizes "2,3" \
  --syndrome-error-probability 0.0 \
  --num-burn-in-sweeps 200 \
  --num-sweeps-between-measurements 4 \
  --num-measurements-per-disorder 80 \
  --q0-num-start-chains 8 \
  --seed-base 20260429 \
  --output-stem "$OUTPUT_STEM"

echo "LOCAL_SMOKE_RUN_ROOT=$RUN_ROOT"
