#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
MASTER_RUN_SUFFIX="${MASTER_RUN_SUFFIX:-}"
MASTER_RUN_ID="${MASTER_RUN_ID:-3d_toric_measurement_noise_threshold_search_${RUN_TIMESTAMP}${MASTER_RUN_SUFFIX}}"
REMOTE_COMPUTE_HOST="${REMOTE_COMPUTE_HOST:-nd-1}"
REMOTE_BASE='/home/DATA1/users/yuany/.single_shot'
REMOTE_REPO_ROOT="$REMOTE_BASE/repos"
REMOTE_REPO_DIR="$REMOTE_REPO_ROOT/$MASTER_RUN_ID"
REMOTE_RUN_ROOT="$REMOTE_BASE/runs/$MASTER_RUN_ID"
REMOTE_LOG_PATH="$REMOTE_BASE/logs/${MASTER_RUN_ID}.log"
REMOTE_RUNNER_PATH="$REMOTE_RUN_ROOT/run_3d_measurement_noise_threshold_search.sh"
REMOTE_SCREEN_NAME="ssprep_${MASTER_RUN_ID}"
COMMIT_SHA="$(git -C "$PROJECT_ROOT" rev-parse HEAD)"

LATTICE_SIZES="${LATTICE_SIZES:-3,4,5}"
NUM_DISORDER_SAMPLES_TOTAL="${NUM_DISORDER_SAMPLES_TOTAL:-256}"
CHUNK_SIZE="${CHUNK_SIZE:-16}"
REQUESTED_WORKERS="${REQUESTED_WORKERS:-48}"
NUM_BURN_IN_SWEEPS="${NUM_BURN_IN_SWEEPS:-1200}"
NUM_SWEEPS_BETWEEN_MEASUREMENTS="${NUM_SWEEPS_BETWEEN_MEASUREMENTS:-6}"
NUM_MEASUREMENTS_PER_DISORDER="${NUM_MEASUREMENTS_PER_DISORDER:-240}"
Q0_NUM_START_CHAINS="${Q0_NUM_START_CHAINS:-8}"
NUM_START_CHAINS="${NUM_START_CHAINS:-8}"
NUM_REPLICAS_PER_START="${NUM_REPLICAS_PER_START:-2}"
PT_P_HOT="${PT_P_HOT:-0.44}"
PT_NUM_TEMPERATURES="${PT_NUM_TEMPERATURES:-9}"
PT_SWAP_ATTEMPT_EVERY_NUM_SWEEPS="${PT_SWAP_ATTEMPT_EVERY_NUM_SWEEPS:-1}"
SEED_BASE="${SEED_BASE:-20260423}"
BURN_IN_SCALING_REFERENCE_NUM_QUBITS="${BURN_IN_SCALING_REFERENCE_NUM_QUBITS:-18}"
DRY_RUN="${DRY_RUN:-0}"


quote_arg() {
  printf '%q' "$1"
}


require_clean_worktree() {
  local branch
  branch="$(git -C "$PROJECT_ROOT" branch --show-current)"
  if [[ "$branch" != "main" ]]; then
    echo "Expected current branch to be main, found: $branch" >&2
    exit 1
  fi
  if [[ -n "$(git -C "$PROJECT_ROOT" status --porcelain)" ]]; then
    echo "Working tree must be clean before remote deployment." >&2
    exit 1
  fi
}


default_stage_a_p_values() {
  printf '%s' '0.0400,0.0600,0.0800,0.1000,0.1200,0.1400,0.1600,0.1800,0.2000,0.2200,0.2400,0.2600,0.2800'
}


default_q_and_p_windows_for_host() {
  local p_values
  p_values="$(default_stage_a_p_values)"
  case "$REMOTE_COMPUTE_HOST" in
    nd-1)
      printf '0.0010|%s\n0.0025|%s' "$p_values" "$p_values"
      ;;
    nd-2)
      printf '0.0050|%s\n0.0100|%s' "$p_values" "$p_values"
      ;;
    nd-3)
      printf '0.0200|%s\n0.0400|%s' "$p_values" "$p_values"
      ;;
    *)
      printf '0.0010|%s\n0.0025|%s' "$p_values" "$p_values"
      ;;
  esac
}


resolve_q_and_p_windows() {
  if [[ -n "${Q_AND_P_WINDOWS:-}" ]]; then
    printf '%s' "$Q_AND_P_WINDOWS"
  else
    default_q_and_p_windows_for_host
  fi
}


fallback_archive_sync() {
  echo "Syncing local HEAD to ${REMOTE_COMPUTE_HOST} via git archive." >&2
  git -C "$PROJECT_ROOT" archive --format=tar HEAD \
    | ssh yuany "ssh ${REMOTE_COMPUTE_HOST} 'mkdir -p $(quote_arg "$REMOTE_REPO_DIR") && tar -xf - -C $(quote_arg "$REMOTE_REPO_DIR")'"
}


verify_remote_env() {
  ssh yuany "ssh ${REMOTE_COMPUTE_HOST} 'set -e; command -v screen >/dev/null 2>&1; command -v conda >/dev/null 2>&1; export CONDA_NO_PLUGINS=true; conda run -n 11 python -c \"import numpy, matplotlib\" >/dev/null'"
}


build_remote_runner_script() {
  local q_and_p_windows_base64="$1"
  cat <<EOF
#!/usr/bin/env bash
set -euo pipefail

export MPLCONFIGDIR=\$HOME/.single_shot/mpl-cache
export CONDA_NO_PLUGINS=true
cd $(quote_arg "$REMOTE_REPO_DIR")

if command -v python3 >/dev/null 2>&1; then
  q_and_p_windows="\$(python3 -c 'import base64, sys; print(base64.b64decode(sys.argv[1]).decode(), end="")' $(quote_arg "$q_and_p_windows_base64"))"
else
  echo "python3 is required on ${REMOTE_COMPUTE_HOST} to decode q/p window payload." >&2
  exit 23
fi

if command -v nproc >/dev/null 2>&1; then
  cpu_count="\$(nproc)"
else
  cpu_count="\$(python3 - <<'PY'
import os
print(os.cpu_count() or 1)
PY
)"
fi

workers=$(quote_arg "$REQUESTED_WORKERS")
if (( workers > cpu_count )); then
  workers="\$cpu_count"
fi

master_run_root=$(quote_arg "$REMOTE_RUN_ROOT")
mkdir -p "\$master_run_root" $(quote_arg "$REMOTE_BASE/logs") \$HOME/.single_shot/mpl-cache

q_index=0
while IFS='|' read -r syndrome_error_probability data_error_probabilities; do
  [[ -z "\$syndrome_error_probability" ]] && continue
  q_tag="\${syndrome_error_probability/./p}"
  q_tag="\${q_tag//-/m}"
  run_root="\$master_run_root/q_\$q_tag"
  current_seed_base=\$(( $(quote_arg "$SEED_BASE") + q_index * 1000000000 ))
  output_stem="scan_result_multi_L_3d_toric_q\${q_tag}_measurement_noise_threshold_search_common_random"
  final_npz="\$run_root/\${output_stem}.npz"
  q_is_zero="\$(python3 -c 'import sys; print("1" if float(sys.argv[1]) == 0.0 else "0")' "\$syndrome_error_probability")"
  submit_extra_args=(
    --q0-num-start-chains $(quote_arg "$Q0_NUM_START_CHAINS")
    --num-start-chains $(quote_arg "$NUM_START_CHAINS")
    --num-replicas-per-start $(quote_arg "$NUM_REPLICAS_PER_START")
  )
  if [[ "\$q_is_zero" != "1" ]]; then
    submit_extra_args+=(
      --pt-p-hot $(quote_arg "$PT_P_HOT")
      --pt-num-temperatures $(quote_arg "$PT_NUM_TEMPERATURES")
      --pt-swap-attempt-every-num-sweeps $(quote_arg "$PT_SWAP_ATTEMPT_EVERY_NUM_SWEEPS")
    )
  fi
  echo "[launcher] starting q=\$syndrome_error_probability host=$(quote_arg "$REMOTE_COMPUTE_HOST") run_root=\$run_root seed_base=\$current_seed_base workers=\$workers"
  conda run -n 11 python src/production_chunked_scan.py submit \
    --run-root "\$run_root" \
    --code-family 3d_toric \
    --workers "\$workers" \
    --chunk-size $(quote_arg "$CHUNK_SIZE") \
    --num-disorder-samples-total $(quote_arg "$NUM_DISORDER_SAMPLES_TOTAL") \
    --data-error-probabilities "\$data_error_probabilities" \
    --lattice-sizes $(quote_arg "$LATTICE_SIZES") \
    --syndrome-error-probability "\$syndrome_error_probability" \
    --num-burn-in-sweeps $(quote_arg "$NUM_BURN_IN_SWEEPS") \
    --num-sweeps-between-measurements $(quote_arg "$NUM_SWEEPS_BETWEEN_MEASUREMENTS") \
    --num-measurements-per-disorder $(quote_arg "$NUM_MEASUREMENTS_PER_DISORDER") \
    "\${submit_extra_args[@]}" \
    --seed-base "\$current_seed_base" \
    --burn-in-scaling-reference-num-qubits $(quote_arg "$BURN_IN_SCALING_REFERENCE_NUM_QUBITS") \
    --output-stem "\$output_stem" \
    --common-random-disorder-across-p \
    --git-commit-sha $(quote_arg "$COMMIT_SHA")
  conda run -n 11 python src/analyze_threshold_crossing.py \
    "\$final_npz" \
    --output-dir "\$run_root" \
    --output-stem "\$output_stem" \
    --summary-path "\$run_root/threshold_summary.json"
  q_index=\$((q_index + 1))
done <<< "\$q_and_p_windows"

conda run -n 11 python src/plot_threshold_search_overview.py "\$master_run_root"

conda run -n 11 python - <<'PY' "\$master_run_root"
import json
import sys
from pathlib import Path

import numpy as np

run_root = Path(sys.argv[1]).resolve()
rows = []
for summary_path in sorted(run_root.glob("q_*/threshold_summary.json")):
    q_dir = summary_path.parent
    npz_path = next(q_dir.glob("*.npz"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with np.load(npz_path, allow_pickle=True) as loaded_result:
        q_value = float(loaded_result["syndrome_error_probability"])
    rows.append({
        "q": q_value,
        "q_dir": str(q_dir),
        "summary_path": str(summary_path),
        "boundary_saturation_artifact": bool(
            summary.get("boundary_saturation_artifact", False)
        ),
        "primary_crossing_window_hit": bool(
            summary.get("primary_crossing_window_hit", False)
        ),
        "interior_crossing_window": summary.get("interior_crossing_window"),
        "recommended_server_window": summary.get("recommended_server_window"),
        "right_edge_gap_signs": summary.get("right_edge_gap_signs"),
    })
output_path = run_root / "measurement_noise_threshold_search_summary.json"
with output_path.open("w", encoding="utf-8") as handle:
    json.dump({"rows": rows}, handle, indent=2, sort_keys=True)
print(json.dumps({
    "summary_path": str(output_path),
    "num_q_runs": len(rows),
}, indent=2, sort_keys=True))
PY

echo "[launcher] all 3D measurement-noise threshold-search runs completed"
EOF
}


launch_remote_runner() {
  local q_and_p_windows_base64="$1"
  local runner_tmp
  runner_tmp="$(mktemp)"
  build_remote_runner_script "$q_and_p_windows_base64" > "$runner_tmp"

  ssh yuany "ssh ${REMOTE_COMPUTE_HOST} 'mkdir -p $(quote_arg "$REMOTE_REPO_DIR") $(quote_arg "$REMOTE_RUN_ROOT") $(quote_arg "$REMOTE_BASE/logs") $(quote_arg "$REMOTE_BASE/mpl-cache")'"
  ssh yuany "ssh ${REMOTE_COMPUTE_HOST} 'cat > $(quote_arg "$REMOTE_RUNNER_PATH")'" < "$runner_tmp"
  ssh yuany "ssh ${REMOTE_COMPUTE_HOST} 'chmod +x $(quote_arg "$REMOTE_RUNNER_PATH") && screen -dmS $(quote_arg "$REMOTE_SCREEN_NAME") bash -lc \"exec $(quote_arg "$REMOTE_RUNNER_PATH") >> $(quote_arg "$REMOTE_LOG_PATH") 2>&1\" && printf \"MASTER_RUN_ID=%s\nREMOTE_COMPUTE_HOST=%s\nSCREEN_NAME=%s\nLOG_PATH=%s\nRUN_ROOT=%s\n\" $(quote_arg "$MASTER_RUN_ID") $(quote_arg "$REMOTE_COMPUTE_HOST") $(quote_arg "$REMOTE_SCREEN_NAME") $(quote_arg "$REMOTE_LOG_PATH") $(quote_arg "$REMOTE_RUN_ROOT")'"

  rm -f "$runner_tmp"
}


main() {
  local q_and_p_windows
  local q_and_p_windows_base64

  q_and_p_windows="$(resolve_q_and_p_windows)"
  q_and_p_windows_base64="$(printf '%s' "$q_and_p_windows" | base64 | tr -d '\n')"

  if [[ "$DRY_RUN" == "1" ]]; then
    printf "MASTER_RUN_ID=%s\nREMOTE_COMPUTE_HOST=%s\nRUN_ROOT=%s\nLOG_PATH=%s\nLATTICE_SIZES=%s\nQ_AND_P_WINDOWS=%s\nNUM_DISORDER_SAMPLES_TOTAL=%s\nCHUNK_SIZE=%s\nREQUESTED_WORKERS=%s\nNUM_BURN_IN_SWEEPS=%s\nNUM_SWEEPS_BETWEEN_MEASUREMENTS=%s\nNUM_MEASUREMENTS_PER_DISORDER=%s\nNUM_START_CHAINS=%s\nNUM_REPLICAS_PER_START=%s\nPT_P_HOT=%s\nPT_NUM_TEMPERATURES=%s\nPT_SWAP_ATTEMPT_EVERY_NUM_SWEEPS=%s\nSEED_BASE=%s\n" \
      "$MASTER_RUN_ID" "$REMOTE_COMPUTE_HOST" "$REMOTE_RUN_ROOT" "$REMOTE_LOG_PATH" \
      "$LATTICE_SIZES" "$q_and_p_windows" "$NUM_DISORDER_SAMPLES_TOTAL" "$CHUNK_SIZE" \
      "$REQUESTED_WORKERS" "$NUM_BURN_IN_SWEEPS" "$NUM_SWEEPS_BETWEEN_MEASUREMENTS" \
      "$NUM_MEASUREMENTS_PER_DISORDER" "$NUM_START_CHAINS" "$NUM_REPLICAS_PER_START" \
      "$PT_P_HOT" "$PT_NUM_TEMPERATURES" "$PT_SWAP_ATTEMPT_EVERY_NUM_SWEEPS" "$SEED_BASE"
    exit 0
  fi

  require_clean_worktree
  fallback_archive_sync
  verify_remote_env
  launch_remote_runner "$q_and_p_windows_base64"
}


main "$@"
