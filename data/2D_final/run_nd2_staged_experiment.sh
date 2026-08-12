#!/usr/bin/env bash

set -euo pipefail

if [[ "$#" -ne 5 ]]; then
  echo "usage: $0 REPO_ROOT STAGE_ROOT SOURCE_COMMIT CONDA_BIN CONDA_ENV" >&2
  exit 2
fi

REPO_ROOT="$1"
STAGE_ROOT="$2"
SOURCE_COMMIT="$3"
CONDA_BIN="$4"
CONDA_ENV="$5"
CONTROL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="$CONTROL_DIR/nd2_staged_experiment_config.json"
AUDIT_SCRIPT="$CONTROL_DIR/audit_nd2_staged_experiment.py"
SENTINEL_SCRIPT="$CONTROL_DIR/nd2_qpositive_sentinel.py"
CPU_LIST="0-31,40-71"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export MPLCONFIGDIR="$STAGE_ROOT/cache/matplotlib"
export NUMBA_CACHE_DIR="$STAGE_ROOT/cache/numba"
export TMPDIR="$STAGE_ROOT/cache/tmp"
export CONDA_NO_PLUGINS=true

mkdir -p \
  "$STAGE_ROOT/cache/matplotlib" \
  "$STAGE_ROOT/cache/numba" \
  "$STAGE_ROOT/cache/tmp" \
  "$STAGE_ROOT/logs" \
  "$STAGE_ROOT/control/pids"

log_msg() {
  printf '[%(%Y-%m-%dT%H:%M:%S%z)T] %s\n' -1 "$*"
}

write_phase() {
  printf '%s\n' "$1" > "$STAGE_ROOT/control/phase"
}

record_pid() {
  local name="$1"
  local pid="$2"
  printf '%s\n' "$pid" > "$STAGE_ROOT/control/pids/${name}.pid"
}

safe_stop_track() {
  local name="$1"
  local pid_path="$STAGE_ROOT/control/pids/${name}.pid"
  local pid
  local pgid
  local owner_marker
  [[ -f "$pid_path" ]] || return 0
  pid="$(<"$pid_path")"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 0
  kill -0 "$pid" 2>/dev/null || return 0
  pgid="$(ps -o pgid= -p "$pid" | tr -d ' ')"
  owner_marker="$(tr '\0' '\n' < "/proc/$pid/environ" | grep -F -x "STAGE_ROOT=$STAGE_ROOT" || true)"
  if [[ "$pgid" != "$pid" || -z "$owner_marker" ]]; then
    log_msg "refusing to signal unverified track name=$name pid=$pid pgid=$pgid"
    return 0
  fi
  log_msg "memory emergency: stopping task-owned process group name=$name pgid=$pgid"
  kill -TERM -- "-$pgid"
}

manifest_progress() {
  "$CONDA_BIN" run -n "$CONDA_ENV" python -c '
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
for path in sorted(root.glob("runs/**/manifest.json")):
    try:
        data = json.loads(path.read_text())
        summary = data.get("summary", {})
        completed = summary.get("completed_chunks")
        failed = summary.get("failed_chunks")
        pending = summary.get("pending_chunks")
        total = summary.get("total_chunks")
        print(
            f"manifest={path.relative_to(root)} "
            f"completed={completed} failed={failed} "
            f"pending={pending} total={total}"
        )
    except Exception as exc:
        print(f"manifest={path.relative_to(root)} error={exc}")
' "$STAGE_ROOT" 2>/dev/null || true
}

monitor_loop() {
  local emergency_count=0
  local mem_available_kib
  local other_cpu
  while [[ ! -f "$STAGE_ROOT/control/stage_done" ]]; do
    {
      printf '[monitor] timestamp=%s phase=%s\n' \
        "$(date -Is)" "$(<"$STAGE_ROOT/control/phase")"
      uptime
      free -h
      other_cpu="$(ps -eo user=,pcpu= | awk -v self="$USER" '$1 != self {sum += $2} END {printf "%.1f", sum + 0}')"
      printf '[monitor] other_user_cpu_percent_sum=%s competition_authorized=1\n' "$other_cpu"
      ps -eo user:20,pid,ppid,pgid,psr,pcpu,pmem,etime,comm,args --sort=-pcpu | sed -n '1,25p'
      manifest_progress
    } >> "$STAGE_ROOT/logs/monitor.log" 2>&1

    mem_available_kib="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
    if (( mem_available_kib < 64 * 1024 * 1024 )); then
      emergency_count=$((emergency_count + 1))
      if (( emergency_count == 1 )); then
        safe_stop_track qpositive
      elif (( emergency_count >= 2 )); then
        safe_stop_track q0_B
        safe_stop_track formal_L11
      fi
    else
      emergency_count=0
    fi
    sleep 300
  done
}

launch_track() {
  local name="$1"
  local log_path="$2"
  shift 2
  setsid taskset -c "$CPU_LIST" nice -n 10 "$@" > "$log_path" 2>&1 &
  local pid=$!
  record_pid "$name" "$pid"
  log_msg "launched name=$name pid=$pid log=$log_path"
}

wait_track() {
  local name="$1"
  local pid
  pid="$(<"$STAGE_ROOT/control/pids/${name}.pid")"
  if wait "$pid"; then
    log_msg "completed name=$name rc=0"
    return 0
  fi
  local rc=$?
  log_msg "failed name=$name rc=$rc"
  return "$rc"
}

run_q0_pilot_arm() {
  local arm="$1"
  local burn_in="$2"
  local measurements="$3"
  local run_root="$STAGE_ROOT/runs/q0_pilot_${arm}"
  local output_stem="q0_pilot_${arm}"
  local resume_args=()
  if [[ -f "$run_root/manifest.json" ]]; then
    resume_args+=(--resume)
  fi
  "$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV" \
    python "$REPO_ROOT/src/production_chunked_scan.py" submit \
    --run-root "$run_root" \
    --code-family 2d_toric \
    --workers 16 \
    --chunk-size 16 \
    --num-disorder-samples-total 128 \
    --data-error-probabilities 0.1000,0.1100 \
    --lattice-sizes 11 \
    --syndrome-error-probability 0.0 \
    --num-burn-in-sweeps "$burn_in" \
    --num-sweeps-between-measurements 10 \
    --num-measurements-per-disorder "$measurements" \
    --q0-num-start-chains 4 \
    --seed-base 620260811 \
    --burn-in-scaling-reference-num-qubits 18 \
    --output-stem "$output_stem" \
    --common-random-disorder-across-p \
    --git-commit-sha "$SOURCE_COMMIT" \
    "${resume_args[@]}"
}

run_qpositive_sentinel() {
  "$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV" \
    python "$SENTINEL_SCRIPT" \
    --config "$CONFIG_PATH" \
    --repo-root "$REPO_ROOT" \
    --run-root "$STAGE_ROOT/runs/qpositive_sentinel" \
    --workers 32 \
    --source-commit "$SOURCE_COMMIT" \
    --resume
}

run_formal_lattice() {
  local lattice_size="$1"
  local seed_base="$2"
  local burn_in="$3"
  local between="$4"
  local measurements="$5"
  local run_root="$STAGE_ROOT/runs/q0_formal_L${lattice_size}"
  local output_stem="q0_formal_L${lattice_size}"
  local resume_args=()
  if [[ -f "$run_root/manifest.json" ]]; then
    resume_args+=(--resume)
  fi
  "$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV" \
    python "$REPO_ROOT/src/production_chunked_scan.py" submit \
    --run-root "$run_root" \
    --code-family 2d_toric \
    --workers 32 \
    --chunk-size 16 \
    --num-disorder-samples-total 1024 \
    --data-error-probabilities 0.1125,0.1150,0.1175,0.1200,0.1225,0.1250 \
    --lattice-sizes "$lattice_size" \
    --syndrome-error-probability 0.0 \
    --num-burn-in-sweeps "$burn_in" \
    --num-sweeps-between-measurements "$between" \
    --num-measurements-per-disorder "$measurements" \
    --q0-num-start-chains 4 \
    --seed-base "$seed_base" \
    --burn-in-scaling-reference-num-qubits 18 \
    --output-stem "$output_stem" \
    --common-random-disorder-across-p \
    --git-commit-sha "$SOURCE_COMMIT" \
    "${resume_args[@]}"
}

finish_stage() {
  local rc="$1"
  write_phase "done_rc_${rc}"
  touch "$STAGE_ROOT/control/stage_done"
  if [[ -n "${MONITOR_PID:-}" ]]; then
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
  fi
  log_msg "stage finished rc=$rc"
  exit "$rc"
}

if [[ "$(<"$REPO_ROOT/SOURCE_COMMIT")" != "$SOURCE_COMMIT" ]]; then
  echo "immutable source marker mismatch" >&2
  exit 3
fi

write_phase pilot
monitor_loop &
MONITOR_PID=$!

export REPO_ROOT STAGE_ROOT SOURCE_COMMIT CONDA_BIN CONDA_ENV CONTROL_DIR
export CONFIG_PATH AUDIT_SCRIPT SENTINEL_SCRIPT CPU_LIST
export -f run_q0_pilot_arm run_qpositive_sentinel run_formal_lattice

launch_track q0_A "$STAGE_ROOT/logs/q0_pilot_A.log" \
  bash -c 'run_q0_pilot_arm A 2000 600'
launch_track q0_B "$STAGE_ROOT/logs/q0_pilot_B.log" \
  bash -c 'run_q0_pilot_arm B 4000 1200'
launch_track qpositive "$STAGE_ROOT/logs/qpositive_sentinel.log" \
  bash -c 'run_qpositive_sentinel'

pilot_rc=0
wait_track q0_A || pilot_rc=1
wait_track q0_B || pilot_rc=1
wait_track qpositive || true
if (( pilot_rc != 0 )); then
  log_msg "q0 pilot incomplete; formal extension will not start"
  finish_stage 20
fi

qpositive_args=()
if [[ -f "$STAGE_ROOT/runs/qpositive_sentinel/qpositive_sentinel_A.npz" \
      && -f "$STAGE_ROOT/runs/qpositive_sentinel/qpositive_sentinel_B.npz" ]]; then
  qpositive_args+=(
    --qpositive-a "$STAGE_ROOT/runs/qpositive_sentinel/qpositive_sentinel_A.npz"
    --qpositive-b "$STAGE_ROOT/runs/qpositive_sentinel/qpositive_sentinel_B.npz"
  )
fi

"$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV" \
  python "$AUDIT_SCRIPT" pilot \
  --config "$CONFIG_PATH" \
  --q0-a "$STAGE_ROOT/runs/q0_pilot_A/q0_pilot_A.npz" \
  --q0-b "$STAGE_ROOT/runs/q0_pilot_B/q0_pilot_B.npz" \
  --q0-a-manifest "$STAGE_ROOT/runs/q0_pilot_A/manifest.json" \
  --q0-b-manifest "$STAGE_ROOT/runs/q0_pilot_B/manifest.json" \
  "${qpositive_args[@]}" \
  --output-json "$STAGE_ROOT/pilot_audit.json" \
  --output-csv "$STAGE_ROOT/pilot_summary.csv" \
  > "$STAGE_ROOT/logs/pilot_audit.log" 2>&1

selected_schedule="$($CONDA_BIN run -n "$CONDA_ENV" python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["q0"]["gate"]["selected_schedule"])' \
  "$STAGE_ROOT/pilot_audit.json")"
log_msg "q0 pilot selected_schedule=$selected_schedule"
if [[ "$selected_schedule" == "STOP" ]]; then
  finish_stage 21
fi

if [[ "$selected_schedule" == "A" ]]; then
  l11_burn_in=2000
  l11_between=10
  l11_measurements=600
else
  l11_burn_in=4000
  l11_between=10
  l11_measurements=1200
fi

write_phase formal
launch_track formal_L9 "$STAGE_ROOT/logs/q0_formal_L9.log" \
  bash -c 'run_formal_lattice 9 620260909 2000 10 600'
launch_track formal_L11 "$STAGE_ROOT/logs/q0_formal_L11.log" \
  bash -c "run_formal_lattice 11 620260911 $l11_burn_in $l11_between $l11_measurements"

formal_rc=0
wait_track formal_L9 || formal_rc=1
wait_track formal_L11 || formal_rc=1
if (( formal_rc != 0 )); then
  log_msg "formal q0 run incomplete; chunks remain resumable"
  finish_stage 30
fi

"$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV" \
  python "$AUDIT_SCRIPT" formal \
  --config "$CONFIG_PATH" \
  --l9-npz "$STAGE_ROOT/runs/q0_formal_L9/q0_formal_L9.npz" \
  --l9-manifest "$STAGE_ROOT/runs/q0_formal_L9/manifest.json" \
  --l11-npz "$STAGE_ROOT/runs/q0_formal_L11/q0_formal_L11.npz" \
  --l11-manifest "$STAGE_ROOT/runs/q0_formal_L11/manifest.json" \
  --selected-l11-schedule "$selected_schedule" \
  --output-json "$STAGE_ROOT/formal_audit.json" \
  --output-csv "$STAGE_ROOT/formal_summary.csv" \
  > "$STAGE_ROOT/logs/formal_audit.log" 2>&1

finish_stage 0
