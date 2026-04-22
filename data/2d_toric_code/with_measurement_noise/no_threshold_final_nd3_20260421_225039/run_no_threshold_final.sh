#!/usr/bin/env bash
set -euo pipefail
export MPLCONFIGDIR=/home/DATA1/users/yuany/.single_shot/mpl-cache
cd /home/DATA1/users/yuany/.single_shot/repo
python_cmd=(/home/DATA1/users/yuany/miniconda3/bin/conda run -n 11 python )
master_run_root=/home/DATA1/users/yuany/.single_shot/runs/no_threshold_final_20260421_225039
q_and_p_windows=$'0.0010|0.0750,0.0775,0.0800,0.0825,0.0850,0.0875,0.0900,0.0925,0.0950,0.0975,0.1000\n0.0025|0.0600,0.0625,0.0650,0.0675,0.0700,0.0725,0.0750,0.0775,0.0800,0.0825,0.0850,0.0875\n0.0050|0.0500,0.0525,0.0550,0.0575,0.0600,0.0625,0.0650,0.0675,0.0700,0.0725,0.0750,0.0775,0.0800,0.0825\n0.0100|0.0300,0.0325,0.0350,0.0375,0.0400,0.0425,0.0450,0.0475,0.0500,0.0525,0.0550,0.0575,0.0600,0.0625,0.0650,0.0675,0.0700'
q_index=0
while IFS='|' read -r syndrome_error_probability data_error_probabilities; do
  [[ -z "$syndrome_error_probability" ]] && continue
  q_tag="${syndrome_error_probability/./p}"
  q_tag="${q_tag//-/m}"
  run_root="$master_run_root/q_$q_tag"
  current_seed_base=$((20260428 + q_index * 1000000000))
  output_stem="scan_result_multi_L_q${q_tag}_no_threshold_final_common_random"
  echo "[launcher] starting q=$syndrome_error_probability run_root=$run_root seed_base=$current_seed_base"
  "${python_cmd[@]}" "/home/DATA1/users/yuany/.single_shot/repo/production_chunked_scan.py" submit     --run-root "$run_root"     --workers 96     --chunk-size 32     --num-disorder-samples-total 1024     --data-error-probabilities "$data_error_probabilities"     --lattice-sizes 3\,5\,7\,9\,11     --syndrome-error-probability "$syndrome_error_probability"     --num-burn-in-sweeps 2200     --num-sweeps-between-measurements 12     --num-measurements-per-disorder 600     --q0-num-start-chains 4     --seed-base "$current_seed_base"     --burn-in-scaling-reference-num-qubits 18     --output-stem "$output_stem"     --common-random-disorder-across-p     --git-commit-sha fa3750ded9109d64d275fc103cb19e1963a62bf0
  "${python_cmd[@]}" "/home/DATA1/users/yuany/.single_shot/repo/analyze_threshold_crossing.py"     "$run_root/$output_stem.npz"     --output-dir "$run_root"     --output-stem "$output_stem"     --summary-path "$run_root/threshold_summary.json"
  q_index=$((q_index + 1))
done <<< "$q_and_p_windows"
echo "[launcher] all no-threshold final q>0 runs completed"
