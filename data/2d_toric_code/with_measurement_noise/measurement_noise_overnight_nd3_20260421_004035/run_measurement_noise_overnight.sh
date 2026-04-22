#!/usr/bin/env bash
set -euo pipefail

export MPLCONFIGDIR=/home/DATA1/users/yuany/.single_shot/mpl-cache
cd /home/DATA1/users/yuany/.single_shot/repo
python_cmd=(/home/DATA1/users/yuany/.single_shot/.venv/bin/python )
lattice_sizes=3\,5\,7
data_error_probabilities=0.0900\,0.0925\,0.0950\,0.0975\,0.1000\,0.1025\,0.1050\,0.1075\,0.1100\,0.1125\,0.1150\,0.1175\,0.1200\,0.1225\,0.1250
syndrome_error_probabilities=0.0100\,0.0200\,0.0300
num_disorder_samples_total=2048
chunk_size=64
workers=96
num_burn_in_sweeps=2000
num_sweeps_between_measurements=10
num_measurements_per_disorder=800
q0_num_start_chains=4
seed_base=20260424
burn_in_scaling_reference_num_qubits=18
master_run_root=/home/DATA1/users/yuany/.single_shot/runs/measurement_noise_overnight_20260421_004035
commit_sha=5cfb0832b1fec7c017a6673b4d74967722d69ea0

IFS=',' read -r -a q_values <<< "$syndrome_error_probabilities"
q_index=0
for syndrome_error_probability in "${q_values[@]}"; do
  q_tag="${syndrome_error_probability/./p}"
  q_tag="${q_tag//-/m}"
  run_root="$master_run_root/q_$q_tag"
  current_seed_base=$((seed_base + q_index * 1000000000))
  echo "[launcher] starting q=$syndrome_error_probability run_root=$run_root seed_base=$current_seed_base"
  "${python_cmd[@]}" "/home/DATA1/users/yuany/.single_shot/repo/production_chunked_scan.py" submit     --run-root "$run_root"     --workers "$workers"     --chunk-size "$chunk_size"     --num-disorder-samples-total "$num_disorder_samples_total"     --data-error-probabilities "$data_error_probabilities"     --lattice-sizes "$lattice_sizes"     --syndrome-error-probability "$syndrome_error_probability"     --num-burn-in-sweeps "$num_burn_in_sweeps"     --num-sweeps-between-measurements "$num_sweeps_between_measurements"     --num-measurements-per-disorder "$num_measurements_per_disorder"     --q0-num-start-chains "$q0_num_start_chains"     --seed-base "$current_seed_base"     --burn-in-scaling-reference-num-qubits "$burn_in_scaling_reference_num_qubits"     --common-random-disorder-across-p     --git-commit-sha "$commit_sha"
  q_index=$((q_index + 1))
done

echo "[launcher] all measurement-noise overnight runs completed"
