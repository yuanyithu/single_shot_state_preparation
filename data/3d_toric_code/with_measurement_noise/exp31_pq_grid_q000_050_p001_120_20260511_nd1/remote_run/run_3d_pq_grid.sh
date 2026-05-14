#!/usr/bin/env bash

set -euo pipefail

export MPLCONFIGDIR="$HOME/.single_shot/mpl-cache"
export CONDA_NO_PLUGINS=true

repo_dir=/home/DATA1/users/yuany/.single_shot/repos/3d_toric_pq_grid_q000_050_nd1_20260511_175054
master_run_root=/home/DATA1/users/yuany/.single_shot/runs/3d_toric_pq_grid_q000_050_nd1_20260511_175054
lattice_sizes=3\,4\,5
p_values=0.0100\,0.0300\,0.0500\,0.0700\,0.1000\,0.1200
num_disorder_samples_total=256
chunk_size=16
requested_workers=64
num_burn_in_sweeps=1200
num_sweeps_between_measurements=6
num_measurements_per_disorder=240
q0_num_start_chains=8
num_start_chains=8
num_replicas_per_start=2
pt_p_hot=0.44
pt_num_temperatures=9
pt_swap_attempt_every_num_sweeps=1
seed_base=20260511
burn_in_scaling_reference_num_qubits=18
commit_sha=7e4b3b1a6a72e20334d042af18d31d59f7d64ce5
q_values=(0.0000 0.0050 0.0100 0.0150 0.0200 0.0250 0.0300 0.0350 0.0400 0.0450 0.0500 )

cd "$repo_dir"

if command -v nproc >/dev/null 2>&1; then
  cpu_count="$(nproc)"
else
  cpu_count="$(conda run -n 11 python -c 'import os; print(os.cpu_count() or 1)')"
fi
workers="$requested_workers"
if (( workers > cpu_count )); then
  workers="$cpu_count"
fi

conda run -n 11 python -c "import sys; sys.path.insert(0, 'src'); import numpy, matplotlib; import production_chunked_scan" >/dev/null

echo "[runner] host=$(hostname) started_at=$(date -Is)"
echo "[runner] run_root=$master_run_root"
echo "[runner] lattice_sizes=$lattice_sizes"
echo "[runner] p_values=$p_values"
echo "[runner] q_values=${q_values[*]}"
echo "[runner] samples=$num_disorder_samples_total chunk=$chunk_size workers=$workers"
echo "[runner] starts=$num_start_chains replicas=$num_replicas_per_start pt_temperatures=$pt_num_temperatures"

q_index=0
for syndrome_error_probability in "${q_values[@]}"; do
  q_tag="${syndrome_error_probability/./p}"
  q_tag="${q_tag//-/m}"
  run_root="$master_run_root/q_$q_tag"
  current_seed_base=$((seed_base + q_index * 1000000000))
  output_stem="scan_result_multi_L_3d_toric_q${q_tag}_pq_grid_common_random"
  final_npz="$run_root/${output_stem}.npz"
  echo "[runner] q=$syndrome_error_probability begin run_root=$run_root seed_base=$current_seed_base at $(date -Is)"

  common_args=(
    submit
    --resume
    --run-root "$run_root"
    --code-family 3d_toric
    --workers "$workers"
    --chunk-size "$chunk_size"
    --num-disorder-samples-total "$num_disorder_samples_total"
    --data-error-probabilities "$p_values"
    --lattice-sizes "$lattice_sizes"
    --syndrome-error-probability "$syndrome_error_probability"
    --num-burn-in-sweeps "$num_burn_in_sweeps"
    --num-sweeps-between-measurements "$num_sweeps_between_measurements"
    --num-measurements-per-disorder "$num_measurements_per_disorder"
    --q0-num-start-chains "$q0_num_start_chains"
    --seed-base "$current_seed_base"
    --burn-in-scaling-reference-num-qubits "$burn_in_scaling_reference_num_qubits"
    --output-stem "$output_stem"
    --common-random-disorder-across-p
    --git-commit-sha "$commit_sha"
  )

  if [[ "$syndrome_error_probability" == "0.0000" ]]; then
    conda run -n 11 python src/production_chunked_scan.py "${common_args[@]}"
  else
    conda run -n 11 python src/production_chunked_scan.py "${common_args[@]}" \
      --num-start-chains "$num_start_chains" \
      --num-replicas-per-start "$num_replicas_per_start" \
      --pt-p-hot "$pt_p_hot" \
      --pt-num-temperatures "$pt_num_temperatures" \
      --pt-swap-attempt-every-num-sweeps "$pt_swap_attempt_every_num_sweeps"
  fi

  conda run -n 11 python src/analyze_threshold_crossing.py \
    "$final_npz" \
    --output-dir "$run_root" \
    --output-stem "$output_stem" \
    --summary-path "$run_root/threshold_summary.json"

  echo "[runner] q=$syndrome_error_probability complete at $(date -Is)"
  q_index=$((q_index + 1))
done

conda run -n 11 python src/plot_threshold_search_overview.py "$master_run_root"

conda run -n 11 python -c '
import json
from pathlib import Path
run_root = Path("'"$master_run_root"'")
rows = []
for manifest_path in sorted(run_root.glob("q_*/manifest.json")):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows.append({
        "q_dir": manifest_path.parent.name,
        "summary": manifest.get("summary"),
        "final_outputs": manifest.get("final_outputs"),
    })
out = run_root / "pq_grid_manifest_summary.json"
out.write_text(json.dumps({"rows": rows}, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps({"summary_path": str(out), "num_q_runs": len(rows)}, indent=2))
'

echo "[runner] all q runs complete at $(date -Is)"
