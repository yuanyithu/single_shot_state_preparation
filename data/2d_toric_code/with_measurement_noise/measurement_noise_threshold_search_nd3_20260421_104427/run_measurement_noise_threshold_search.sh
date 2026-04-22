#!/usr/bin/env bash
set -euo pipefail

export MPLCONFIGDIR=/home/DATA1/users/yuany/.single_shot/mpl-cache
cd /home/DATA1/users/yuany/.single_shot/repo
python_cmd=(/home/DATA1/users/yuany/miniconda3/bin/conda run -n 11 python )
lattice_sizes=3\,5\,7
num_disorder_samples_total=2048
chunk_size=64
workers=96
num_burn_in_sweeps=2000
num_sweeps_between_measurements=10
num_measurements_per_disorder=800
q0_num_start_chains=4
seed_base=20260426
burn_in_scaling_reference_num_qubits=18
master_run_root=/home/DATA1/users/yuany/.single_shot/runs/measurement_noise_threshold_search_20260421_104427
commit_sha=cfd76e861cba70cec506e2c5f3bbcb681a004b09
q_and_p_windows=$'0.0025|0.0850,0.0875,0.0900,0.0925,0.0950,0.0975,0.1000,0.1025,0.1050,0.1075,0.1100\n0.0050|0.0800,0.0825,0.0850,0.0875,0.0900,0.0925,0.0950,0.0975,0.1000,0.1025,0.1050,0.1075\n0.0075|0.0725,0.0750,0.0775,0.0800,0.0825,0.0850,0.0875,0.0900,0.0925,0.0950,0.0975,0.1000\n0.0100|0.0600,0.0625,0.0650,0.0675,0.0700,0.0725,0.0750,0.0775,0.0800,0.0825,0.0850,0.0875,0.0900\n0.0150|0.0450,0.0475,0.0500,0.0525,0.0550,0.0575,0.0600,0.0625,0.0650,0.0675,0.0700,0.0725,0.0750,0.0775,0.0800\n0.0200|0.0350,0.0375,0.0400,0.0425,0.0450,0.0475,0.0500,0.0525,0.0550,0.0575,0.0600,0.0625,0.0650,0.0675,0.0700'

q_index=0
while IFS='|' read -r syndrome_error_probability data_error_probabilities; do
  [[ -z "$syndrome_error_probability" ]] && continue
  q_tag="${syndrome_error_probability/./p}"
  q_tag="${q_tag//-/m}"
  run_root="$master_run_root/q_$q_tag"
  current_seed_base=$((seed_base + q_index * 1000000000))
  output_stem="scan_result_multi_L_q${q_tag}_measurement_noise_threshold_search_common_random"
  final_npz="$run_root/${output_stem}.npz"
  echo "[launcher] starting q=$syndrome_error_probability run_root=$run_root seed_base=$current_seed_base"
  "${python_cmd[@]}" "/home/DATA1/users/yuany/.single_shot/repo/production_chunked_scan.py" submit     --run-root "$run_root"     --workers "$workers"     --chunk-size "$chunk_size"     --num-disorder-samples-total "$num_disorder_samples_total"     --data-error-probabilities "$data_error_probabilities"     --lattice-sizes "$lattice_sizes"     --syndrome-error-probability "$syndrome_error_probability"     --num-burn-in-sweeps "$num_burn_in_sweeps"     --num-sweeps-between-measurements "$num_sweeps_between_measurements"     --num-measurements-per-disorder "$num_measurements_per_disorder"     --q0-num-start-chains "$q0_num_start_chains"     --seed-base "$current_seed_base"     --burn-in-scaling-reference-num-qubits "$burn_in_scaling_reference_num_qubits"     --output-stem "$output_stem"     --common-random-disorder-across-p     --git-commit-sha "$commit_sha"
  "${python_cmd[@]}" "/home/DATA1/users/yuany/.single_shot/repo/analyze_threshold_crossing.py"     "$final_npz"     --output-dir "$run_root"     --output-stem "$output_stem"     --summary-path "$run_root/threshold_summary.json"
  q_index=$((q_index + 1))
done <<< "$q_and_p_windows"

echo "[launcher] all measurement-noise threshold-search runs completed"
