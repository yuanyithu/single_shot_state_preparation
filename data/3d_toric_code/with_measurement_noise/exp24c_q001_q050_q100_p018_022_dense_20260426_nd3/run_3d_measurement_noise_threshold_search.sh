#!/usr/bin/env bash
set -euo pipefail

export MPLCONFIGDIR=$HOME/.single_shot/mpl-cache
export CONDA_NO_PLUGINS=true
cd /home/DATA1/users/yuany/.single_shot/repos/3d_toric_exp24c_q001_q050_q100_p018_022_dense_20260426_nd3

if command -v python3 >/dev/null 2>&1; then
  q_and_p_windows="$(python3 -c 'import base64, sys; print(base64.b64decode(sys.argv[1]).decode(), end="")' MC4xMDAwfDAuMTgwLDAuMTg1LDAuMTkwLDAuMTk1LDAuMjAwLDAuMjA1LDAuMjEwLDAuMjE1LDAuMjIwCjAuMDEwMHwwLjE4MCwwLjE4NSwwLjE5MCwwLjE5NSwwLjIwMCwwLjIwNSwwLjIxMCwwLjIxNSwwLjIyMAowLjA1MDB8MC4xODAsMC4xODUsMC4xOTAsMC4xOTUsMC4yMDAsMC4yMDUsMC4yMTAsMC4yMTUsMC4yMjA=)"
else
  echo "python3 is required on nd-3 to decode q/p window payload." >&2
  exit 23
fi

if command -v nproc >/dev/null 2>&1; then
  cpu_count="$(nproc)"
else
  cpu_count="$(python3 - <<'PY'
import os
print(os.cpu_count() or 1)
PY
)"
fi

workers=48
if (( workers > cpu_count )); then
  workers="$cpu_count"
fi

master_run_root=/home/DATA1/users/yuany/.single_shot/runs/3d_toric_exp24c_q001_q050_q100_p018_022_dense_20260426_nd3
mkdir -p "$master_run_root" /home/DATA1/users/yuany/.single_shot/logs $HOME/.single_shot/mpl-cache

q_index=0
while IFS='|' read -r syndrome_error_probability data_error_probabilities; do
  [[ -z "$syndrome_error_probability" ]] && continue
  q_tag="${syndrome_error_probability/./p}"
  q_tag="${q_tag//-/m}"
  run_root="$master_run_root/q_$q_tag"
  current_seed_base=$(( 2026042603 + q_index * 1000000000 ))
  output_stem="scan_result_multi_L_3d_toric_q${q_tag}_measurement_noise_threshold_search_common_random"
  final_npz="$run_root/${output_stem}.npz"
  q_is_zero="$(python3 -c 'import sys; print("1" if float(sys.argv[1]) == 0.0 else "0")' "$syndrome_error_probability")"
  submit_extra_args=(
    --q0-num-start-chains 8
    --num-start-chains 8
    --num-replicas-per-start 1
  )
  if [[ "$q_is_zero" != "1" ]]; then
    submit_extra_args+=(
      --pt-p-hot 0.44
      --pt-num-temperatures 7
      --pt-swap-attempt-every-num-sweeps 1
    )
  fi
  burn_in_cap_args=()
  if [[ -n 3000 ]]; then
    burn_in_cap_args+=(
      --max-effective-num-burn-in-sweeps 3000
    )
  fi
  echo "[launcher] starting q=$syndrome_error_probability host=nd-3 run_root=$run_root seed_base=$current_seed_base workers=$workers"
  conda run -n 11 python src/production_chunked_scan.py submit     --run-root "$run_root"     --code-family 3d_toric     --workers "$workers"     --chunk-size 4     --num-disorder-samples-total 384     --data-error-probabilities "$data_error_probabilities"     --lattice-sizes 3\,4\,5     --syndrome-error-probability "$syndrome_error_probability"     --num-burn-in-sweeps 1000     --num-sweeps-between-measurements 6     --num-measurements-per-disorder 2048     "${submit_extra_args[@]}"     --seed-base "$current_seed_base"     --burn-in-scaling-reference-num-qubits 18     "${burn_in_cap_args[@]}"     --output-stem "$output_stem"     --common-random-disorder-across-p     --git-commit-sha 8ac6157d5465af754265ef71d475ae8d566b9db0
  conda run -n 11 python src/analyze_threshold_crossing.py     "$final_npz"     --output-dir "$run_root"     --output-stem "$output_stem"     --summary-path "$run_root/threshold_summary.json"
  q_index=$((q_index + 1))
done <<< "$q_and_p_windows"

conda run -n 11 python src/plot_threshold_search_overview.py "$master_run_root"

conda run -n 11 python - <<'PY' "$master_run_root"
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
