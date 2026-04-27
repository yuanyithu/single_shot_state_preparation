#!/usr/bin/env bash
set -euo pipefail
export MPLCONFIGDIR=$HOME/.single_shot/mpl-cache
export CONDA_NO_PLUGINS=true
cd '/home/DATA1/users/yuany/.single_shot/repos/3d_toric_exp28c_fixed_p010_q000_100_L6_20260427_nd3'
master_run_root='/home/DATA1/users/yuany/.single_shot/runs/3d_toric_exp28c_fixed_p010_q000_100_L6_20260427_nd3'
mkdir -p "$master_run_root" /home/DATA1/users/yuany/.single_shot/logs $HOME/.single_shot/mpl-cache
q_values=(0.0100 0.0200 0.0300 0.0400 0.0500 0.0600 0.0700 0.0800 0.0900 0.1000)
q_index=1
for syndrome_error_probability in "${q_values[@]}"; do
  q_tag="${syndrome_error_probability/./p}"
  run_root="$master_run_root/q_$q_tag"
  current_seed_base=$(( 2026042831 + q_index * 1000000000 ))
  output_stem="scan_result_multi_L_3d_toric_q${q_tag}_measurement_noise_threshold_search_common_random"
  final_npz="$run_root/${output_stem}.npz"
  if [[ -f "$final_npz" ]]; then
    echo "[L6-resume] skipping existing q=$syndrome_error_probability final_npz=$final_npz"
    q_index=$((q_index + 1))
    continue
  fi
  echo "[L6-resume] starting q=$syndrome_error_probability host=nd-3 run_root=$run_root seed_base=$current_seed_base workers=48"
  conda run -n 11 python src/production_chunked_scan.py submit     --run-root "$run_root"     --code-family 3d_toric     --workers 48     --chunk-size 4     --num-disorder-samples-total 512     --data-error-probabilities 0.1000     --lattice-sizes 6     --syndrome-error-probability "$syndrome_error_probability"     --num-burn-in-sweeps 1000     --num-sweeps-between-measurements 6     --num-measurements-per-disorder 2048     --q0-num-start-chains 8     --num-start-chains 8     --num-replicas-per-start 1     --pt-p-hot 0.44     --pt-num-temperatures 7     --pt-swap-attempt-every-num-sweeps 1     --seed-base "$current_seed_base"     --burn-in-scaling-reference-num-qubits 18     --max-effective-num-burn-in-sweeps 3000     --output-stem "$output_stem"     --common-random-disorder-across-p     --git-commit-sha 41c10bc0
  q_index=$((q_index + 1))
done
echo "[L6-resume] all remaining L6 q runs completed"
