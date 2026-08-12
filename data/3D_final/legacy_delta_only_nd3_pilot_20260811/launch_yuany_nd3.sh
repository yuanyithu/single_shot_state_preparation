#!/usr/bin/env bash
# Create-only controller for the shared yuany filesystem and nd-3 compute node.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PILOT_REL="data/3D_final/legacy_delta_only_nd3_pilot_20260811"
PILOT_DIR="$PROJECT_ROOT/$PILOT_REL"
RUNNER_REL="$PILOT_REL/run_legacy_pilot.py"
REMOTE_BASE="/home/DATA1/users/yuany/.single_shot"
REMOTE_CONDA="/home/DATA1/users/yuany/miniconda3/bin/conda"
SSH_GATEWAY=(ssh -o BatchMode=yes -o ConnectTimeout=20 yuany)

usage() {
  cat <<'EOF'
Usage:
  launch_yuany_nd3.sh preflight
  launch_yuany_nd3.sh seed-check SEED_BASE
  launch_yuany_nd3.sh stage RUN_ID
  launch_yuany_nd3.sh launch RUN_ID WAVE_ID PAIRS SEED_BASE
  launch_yuany_nd3.sh status RUN_ID WAVE_ID
  launch_yuany_nd3.sh collect RUN_ID WAVE_ID DESTINATION

PAIRS is a comma-separated list such as 0.230:0.012,0.230:0.022,0.230:0.030.
EOF
}

validate_name() {
  local value="$1"
  [[ "$value" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$ ]] || {
    echo "unsafe name: $value" >&2
    exit 64
  }
}

validate_pairs() {
  local value="$1"
  [[ "$value" =~ ^0\.(225|230|235|240):0\.(012|022|030)(,0\.(225|230|235|240):0\.(012|022|030))*$ ]] || {
    echo "pairs are outside the approved adaptive pool: $value" >&2
    exit 64
  }
}

remote_nd3() {
  local command="$1"
  "${SSH_GATEWAY[@]}" "ssh -o BatchMode=yes -o ConnectTimeout=20 nd-3 $(printf '%q' "$command")"
}

run_preflight() {
  local output
  output="$({ remote_nd3 "export LC_ALL=C; $REMOTE_CONDA run --no-capture-output -n 11 python - preflight" < "$PILOT_DIR/run_legacy_pilot.py"; } 2>/dev/null)"
  printf '%s\n' "$output"
  PREFLIGHT_JSON="$output" conda run --no-capture-output -n 12 python - <<'PY'
import json, os
payload = json.loads(os.environ["PREFLIGHT_JSON"])
if not payload.get("passed"):
    raise SystemExit("nd-3 preflight failed")
workers = int(payload["recommended_workers"])
if not 8 <= workers <= 70:
    raise SystemExit(f"unsafe recommended worker count: {workers}")
print(workers)
PY
}

seed_check() {
  local seed_base="$1"
  [[ "$seed_base" =~ ^[0-9]+$ ]] || { echo "invalid seed base" >&2; exit 64; }
  local seed_stop=$((seed_base + 383))
  "${SSH_GATEWAY[@]}" "SEED_START=$seed_base SEED_STOP=$seed_stop python3 -" <<'PY'
import json, os
from pathlib import Path

start = int(os.environ["SEED_START"])
stop = int(os.environ["SEED_STOP"])
root = Path("/home/DATA1/users/yuany/.single_shot/runs")
hits = []
for path in sorted(root.glob("**/*.json")):
    if path.name not in {"manifest.json", "run_manifest.json", "remote_runs_manifest.json"}:
        continue
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        continue
    candidates = []
    if isinstance(payload, dict) and "seed_base" in payload:
        candidates.append((int(payload["seed_base"]), int(payload.get("num_disorder_samples", 1))))
    for node in payload.get("nodes", []) if isinstance(payload, dict) else []:
        for cell in node.get("cells", []):
            if "seed_base" in cell:
                candidates.append((int(cell["seed_base"]), int(payload.get("num_disorder_samples", 1))))
    for existing, count in candidates:
        existing_stop = existing + max(1, count) - 1
        if not (stop < existing or existing_stop < start):
            hits.append({"path": str(path), "seed_base": existing, "count": count})
print(json.dumps({"seed_start": start, "seed_stop": stop, "hits": hits}, indent=2))
if hits:
    raise SystemExit(2)
PY
}

stage_run() {
  local run_id="$1"
  validate_name "$run_id"
  local remote_repo="$REMOTE_BASE/repos/$run_id"
  local remote_run="$REMOTE_BASE/runs/$run_id"
  local remote_log_prefix="$REMOTE_BASE/logs/${run_id}_nd3"

  "${SSH_GATEWAY[@]}" "set -euo pipefail; umask 077; test ! -e $(printf '%q' "$remote_repo"); test ! -e $(printf '%q' "$remote_run"); compgen -G $(printf '%q' "${remote_log_prefix}*") >/dev/null && exit 73 || true; mkdir $(printf '%q' "$remote_repo") $(printf '%q' "$remote_run")"

  tar -C "$PROJECT_ROOT" \
    --exclude='__pycache__' --exclude='*.pyc' \
    -cf - src \
    "$PILOT_REL/README.md" \
    "$PILOT_REL/experiment_spec.json" \
    "$PILOT_REL/run_legacy_pilot.py" \
    "$PILOT_REL/analyze_legacy_pilot.py" \
    "$PILOT_REL/source_manifest.json" \
    | "${SSH_GATEWAY[@]}" "tar -C $(printf '%q' "$remote_repo") -xf -"

  local expected_sha
  expected_sha="$(shasum -a 256 "$PROJECT_ROOT/src/exp37_sector_ti.py" | awk '{print $1}')"
  local remote_sha
  remote_sha="$("${SSH_GATEWAY[@]}" "sha256sum $(printf '%q' "$remote_repo/src/exp37_sector_ti.py")" | awk '{print $1}')"
  [[ "$remote_sha" == "$expected_sha" ]] || {
    echo "remote source SHA mismatch" >&2
    exit 74
  }
  printf 'STAGED RUN_ID=%s REPO=%s RUN_ROOT=%s SOURCE_SHA=%s\n' \
    "$run_id" "$remote_repo" "$remote_run" "$remote_sha"
}

launch_wave() {
  local run_id="$1" wave_id="$2" pairs="$3" seed_base="$4"
  validate_name "$run_id"
  validate_name "$wave_id"
  validate_pairs "$pairs"
  [[ "$seed_base" =~ ^[0-9]+$ ]] || { echo "invalid seed base" >&2; exit 64; }

  local preflight_output workers
  preflight_output="$(run_preflight)"
  printf '%s\n' "$preflight_output"
  workers="$(printf '%s\n' "$preflight_output" | tail -n 1)"
  [[ "$workers" =~ ^[0-9]+$ ]] && (( workers >= 8 && workers <= 70 )) || {
    echo "could not obtain safe worker count" >&2
    exit 75
  }

  local remote_repo="$REMOTE_BASE/repos/$run_id"
  local remote_run="$REMOTE_BASE/runs/$run_id"
  local wave_root="$remote_run/nd3/waves/$wave_id"
  local log_path="$REMOTE_BASE/logs/${run_id}_nd3_${wave_id}.log"
  local screen_name="${run_id}_${wave_id}"
  local runner="$remote_repo/$RUNNER_REL"

  local remote_command
  printf -v remote_command '%s' \
    "set -euo pipefail; test -d $(printf '%q' "$remote_repo"); test -d $(printf '%q' "$remote_run"); test ! -e $(printf '%q' "$wave_root"); test ! -e $(printf '%q' "$log_path"); if screen -ls 2>/dev/null | grep -Fq $(printf '%q' ".$screen_name"); then exit 76; fi; mkdir -p $(printf '%q' "$remote_run/nd3/waves"); priority='nice -n 10'; command -v ionice >/dev/null && priority='nice -n 10 ionice -c2 -n7'; cd $(printf '%q' "$remote_repo"); export LC_ALL=C CONDA_NO_PLUGINS=true NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1; command_line=\"\$priority $REMOTE_CONDA run --no-capture-output -n 11 python $(printf '%q' "$runner") run-wave --run-root $(printf '%q' "$wave_root") --pairs $(printf '%q' "$pairs") --seed-base $(printf '%q' "$seed_base") --workers $(printf '%q' "$workers")\"; screen -dmS $(printf '%q' "$screen_name") bash -lc \"exec setsid \$command_line >> $(printf '%q' "$log_path") 2>&1\"; printf 'RUN_ID=%s\\nWAVE_ID=%s\\nSCREEN=%s\\nLOG=%s\\nWORKERS=%s\\n' $(printf '%q' "$run_id") $(printf '%q' "$wave_id") $(printf '%q' "$screen_name") $(printf '%q' "$log_path") $(printf '%q' "$workers")"
  remote_nd3 "$remote_command"
}

status_wave() {
  local run_id="$1" wave_id="$2"
  validate_name "$run_id"
  validate_name "$wave_id"
  local remote_run="$REMOTE_BASE/runs/$run_id"
  local wave_root="$remote_run/nd3/waves/$wave_id"
  local log_path="$REMOTE_BASE/logs/${run_id}_nd3_${wave_id}.log"
  remote_nd3 "screen -ls 2>&1 || true; if test -f $(printf '%q' "$wave_root/status.json"); then cat $(printf '%q' "$wave_root/status.json"); fi; if test -f $(printf '%q' "$log_path"); then tail -n 30 $(printf '%q' "$log_path"); fi"
}

collect_wave() {
  local run_id="$1" wave_id="$2" destination="$3"
  validate_name "$run_id"
  validate_name "$wave_id"
  test ! -e "$destination"
  mkdir -p "$destination"
  local relative="runs/$run_id/nd3/waves/$wave_id"
  "${SSH_GATEWAY[@]}" "set -euo pipefail; base=$(printf '%q' "$REMOTE_BASE"); wave=\"\$base/$relative\"; test -f \"\$wave/SUCCESS.json\"; tar -C \"\$base\" --exclude='tasks' -cf - $(printf '%q' "$relative") logs/$(printf '%q' "${run_id}_nd3_${wave_id}.log")" \
    | tar -C "$destination" -xf -
  find "$destination" -type f -exec shasum -a 256 {} \; | sort
}

main() {
  [[ $# -ge 1 ]] || { usage; exit 64; }
  local command="$1"
  shift
  case "$command" in
    preflight)
      [[ $# -eq 0 ]] || { usage; exit 64; }
      run_preflight
      ;;
    seed-check)
      [[ $# -eq 1 ]] || { usage; exit 64; }
      seed_check "$1"
      ;;
    stage)
      [[ $# -eq 1 ]] || { usage; exit 64; }
      stage_run "$1"
      ;;
    launch)
      [[ $# -eq 4 ]] || { usage; exit 64; }
      launch_wave "$1" "$2" "$3" "$4"
      ;;
    status)
      [[ $# -eq 2 ]] || { usage; exit 64; }
      status_wave "$1" "$2"
      ;;
    collect)
      [[ $# -eq 3 ]] || { usage; exit 64; }
      collect_wave "$1" "$2" "$3"
      ;;
    *)
      usage
      exit 64
      ;;
  esac
}

main "$@"
