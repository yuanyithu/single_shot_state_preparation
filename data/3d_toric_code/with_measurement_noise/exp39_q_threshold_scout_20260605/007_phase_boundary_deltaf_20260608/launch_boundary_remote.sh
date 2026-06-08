#!/usr/bin/env bash
#
# exp39/007 phase-boundary launcher (node<->p).
#
# Unlike exp38/003 (node<->L sharding, one L per node), this runs the FULL
# L=3,4,5 sweep for each p in a single `run` call (the worker pool balances
# L x q x disorder across cores) and assigns whole p-values to nodes. Each node
# runs its assigned p-cells SEQUENTIALLY in one screen, using all its cores.
#
# Path: TI / projection_mode=linear (CORRECT observable). Never ais/decoder_reject.
#
# Override the cell matrix with env CELLS (semicolon-separated "host|p|seed_base|q_csv").
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
STAGE_DIR="$PROJECT_ROOT/data/3d_toric_code/with_measurement_noise/exp39_q_threshold_scout_20260605/007_phase_boundary_deltaf_20260608"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
MASTER_RUN_ID="${MASTER_RUN_ID:-exp39_boundary_${RUN_TIMESTAMP}}"
REMOTE_BASE="${REMOTE_BASE:-/home/DATA1/users/yuany/.single_shot}"
REMOTE_MASTER_ROOT="${REMOTE_MASTER_ROOT:-$REMOTE_BASE/runs/$MASTER_RUN_ID}"
REMOTE_REPO_BASE="${REMOTE_REPO_BASE:-$REMOTE_BASE/repos/$MASTER_RUN_ID}"

# Default cell matrix: 6 new p, 2 per node, balanced. (done p=0.06/0.12 reused from 004/006.)
DEFAULT_CELLS=(
  "nd-1|0.02|830000|0.03,0.05,0.062,0.072,0.082,0.095,0.115,0.15"
  "nd-1|0.14|833000|0.008,0.014,0.02,0.026,0.033,0.043,0.058,0.08"
  "nd-2|0.04|831000|0.025,0.04,0.053,0.063,0.073,0.087,0.105,0.135"
  "nd-2|0.16|834000|0.005,0.01,0.015,0.02,0.026,0.034,0.046,0.066"
  "nd-3|0.09|832000|0.015,0.026,0.036,0.043,0.052,0.065,0.085,0.115"
  "nd-3|0.20|835000|0.002,0.005,0.008,0.012,0.017,0.024,0.034,0.05"
)
if [[ -n "${CELLS:-}" ]]; then
  IFS=';' read -r -a CELL_LIST <<< "$CELLS"
else
  CELL_LIST=("${DEFAULT_CELLS[@]}")
fi

# Fixed run parameters (aligned with validated 004/006 production runs).
FIXED_LATTICE_SIZES="${FIXED_LATTICE_SIZES:-3,4,5}"
NUM_DISORDER_SAMPLES="${NUM_DISORDER_SAMPLES:-48}"
NUM_KP_GRID_POINTS="${NUM_KP_GRID_POINTS:-129}"
NUM_BURN_IN_SWEEPS="${NUM_BURN_IN_SWEEPS:-512}"
MAX_EFFECTIVE_NUM_BURN_IN_SWEEPS="${MAX_EFFECTIVE_NUM_BURN_IN_SWEEPS:-512}"
NUM_MEASUREMENTS="${NUM_MEASUREMENTS:-8192}"
NUM_SWEEPS_BETWEEN_MEASUREMENTS="${NUM_SWEEPS_BETWEEN_MEASUREMENTS:-2}"
BLOCK_COUNT="${BLOCK_COUNT:-128}"
NUM_BOOTSTRAP="${NUM_BOOTSTRAP:-800}"
WINDING_HEATBATH_SWEEPS="${WINDING_HEATBATH_SWEEPS:-1}"
PROJECTION_MODE="${PROJECTION_MODE:-linear}"
DISORDER_SEED_SCOPE="${DISORDER_SEED_SCOPE:-disorder_index}"
DISORDER_REALIZATION_MODE="${DISORDER_REALIZATION_MODE:-rng_stream}"
# Worker count per node: default = nproc (computed on the node). Override with NUM_WORKERS.
NUM_WORKERS="${NUM_WORKERS:-0}"
SKIP_SYNC="${SKIP_SYNC:-0}"
DRY_RUN="${DRY_RUN:-0}"

MANIFEST_PATH="${MANIFEST_PATH:-$STAGE_DIR/remote_runs_manifest.json}"


quote_arg() { printf '%q' "$1"; }
host_tag() { printf '%s' "${1//-/}"; }
p_tag() { printf 'p%s' "${1//./p}"; }


sync_remote_repo() {
  local host="$1" repo_dir="$2"
  echo "[boundary] syncing selected working tree to $host:$repo_dir"
  tar -C "$PROJECT_ROOT" --exclude='__pycache__' --exclude='*.pyc' -cf - src \
    | ssh yuany "ssh ${host} 'rm -rf $(quote_arg "$repo_dir") && mkdir -p $(quote_arg "$repo_dir") && tar -xf - -C $(quote_arg "$repo_dir")'"
}


verify_remote_env() {
  local host="$1"
  ssh yuany "ssh ${host} 'set -euo pipefail; hostname; echo nproc=\$(nproc); command -v screen >/dev/null; command -v conda >/dev/null; export CONDA_NO_PLUGINS=true; conda run --no-capture-output -n 11 python -c \"import importlib.util, numpy; print(\\\"python_env_ok=1\\\"); print(\\\"numba_available=\\\" + str(importlib.util.find_spec(\\\"numba\\\") is not None))\"'"
}


# Build a runner that loops over this host's (p, seed_base, q_csv) cells.
build_remote_runner() {
  local host="$1" repo_dir="$2" run_root="$3" cells_blob="$4" effective_workers="$5"

  {
    printf '#!/usr/bin/env bash\n'
    printf 'set -uo pipefail\n\n'
    printf 'host=%q\n' "$host"
    printf 'repo_dir=%q\n' "$repo_dir"
    printf 'run_root=%q\n' "$run_root"
    printf 'cells_blob=%q\n' "$cells_blob"
    printf 'lattice_sizes=%q\n' "$FIXED_LATTICE_SIZES"
    printf 'num_disorder_samples=%q\n' "$NUM_DISORDER_SAMPLES"
    printf 'num_kp_grid_points=%q\n' "$NUM_KP_GRID_POINTS"
    printf 'num_burn_in_sweeps=%q\n' "$NUM_BURN_IN_SWEEPS"
    printf 'max_effective_num_burn_in_sweeps=%q\n' "$MAX_EFFECTIVE_NUM_BURN_IN_SWEEPS"
    printf 'num_measurements=%q\n' "$NUM_MEASUREMENTS"
    printf 'num_sweeps_between_measurements=%q\n' "$NUM_SWEEPS_BETWEEN_MEASUREMENTS"
    printf 'block_count=%q\n' "$BLOCK_COUNT"
    printf 'num_bootstrap=%q\n' "$NUM_BOOTSTRAP"
    printf 'winding_heatbath_sweeps=%q\n' "$WINDING_HEATBATH_SWEEPS"
    printf 'projection_mode=%q\n' "$PROJECTION_MODE"
    printf 'disorder_seed_scope=%q\n' "$DISORDER_SEED_SCOPE"
    printf 'disorder_realization_mode=%q\n' "$DISORDER_REALIZATION_MODE"
    printf 'workers=%q\n' "$effective_workers"
    cat <<'EOF_RUNNER'

export CONDA_NO_PLUGINS=true
export MPLCONFIGDIR="$HOME/.single_shot/mpl-cache"
# Each (L,q,disorder) task is its own process; with 80-96 workers prevent nested
# BLAS/numba threading from oversubscribing the node.
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

log_msg() { printf '[%(%Y-%m-%dT%H:%M:%S%z)T] %s\n' -1 "$*"; }

write_json_status() {
  local path="$1" status="$2" exit_code="$3"
  python -c 'import json,sys; from datetime import datetime, timezone; path,status,exit_code=sys.argv[1],sys.argv[2],int(sys.argv[3]); open(path,"w",encoding="utf-8").write(json.dumps({"status":status,"exit_code":exit_code,"finished_at":datetime.now(timezone.utc).isoformat()},indent=2,sort_keys=True))' "$path" "$status" "$exit_code"
}

# workers was baked in at launch time (nproc detected OUTSIDE the screen, where it
# is reliable; in-screen $(nproc) has mis-reported 1). Defensive fallback only.
[[ "$workers" =~ ^[0-9]+$ ]] && [[ "$workers" -ge 1 ]] || workers="$(nproc)"
mkdir -p "$run_root" "$run_root/collected" "$HOME/.single_shot/mpl-cache"
cd "$repo_dir"
rm -f "$run_root/_SUCCESS.json" "$run_root/_FAILED.json"
log_msg "exp39 boundary host=$host workers=$workers cells=[$cells_blob]"

overall_rc=0
IFS=';' read -r -a cells <<< "$cells_blob"
for cell in "${cells[@]}"; do
  IFS='|' read -r p seed_base q_csv <<< "$cell"
  ptag="p${p//./p}"
  out_dir="$run_root/collected/$ptag"
  mkdir -p "$out_dir"
  rm -f "$out_dir/_CELL_SUCCESS.json" "$out_dir/_CELL_FAILED.json"
  log_msg "BEGIN cell p=$p seed_base=$seed_base out=$out_dir q=$q_csv"
  set +e
  conda run --no-capture-output -n 11 python src/exp37_sector_ti.py run \
    --output-dir "$out_dir" \
    --code-family 3d_toric \
    --projection-mode "$projection_mode" \
    --lattice-sizes "$lattice_sizes" \
    --p "$p" \
    --q-values "$q_csv" \
    --num-disorder-samples "$num_disorder_samples" \
    --seed-base "$seed_base" \
    --common-disorder-across-q \
    --disorder-seed-scope "$disorder_seed_scope" \
    --disorder-realization-mode "$disorder_realization_mode" \
    --num-kp-grid-points "$num_kp_grid_points" \
    --num-burn-in-sweeps "$num_burn_in_sweeps" \
    --max-effective-num-burn-in-sweeps "$max_effective_num_burn_in_sweeps" \
    --num-measurements "$num_measurements" \
    --num-sweeps-between-measurements "$num_sweeps_between_measurements" \
    --block-count "$block_count" \
    --num-bootstrap "$num_bootstrap" \
    --winding-heatbath-sweeps "$winding_heatbath_sweeps" \
    --use-numba \
    --grid-tv-warning 0.02 \
    --grid-q-top-warning 0.02 \
    --num-workers "$workers"
  rc="$?"
  set -e
  if [[ "$rc" == "0" ]]; then
    write_json_status "$out_dir/_CELL_SUCCESS.json" success 0
    log_msg "END   cell p=$p rc=0"
  else
    write_json_status "$out_dir/_CELL_FAILED.json" failed "$rc"
    log_msg "END   cell p=$p rc=$rc (continuing to next cell)"
    overall_rc="$rc"
  fi
done

if [[ "$overall_rc" == "0" ]]; then
  write_json_status "$run_root/_SUCCESS.json" success 0
  log_msg "exp39 boundary host=$host ALL CELLS OK"
else
  write_json_status "$run_root/_FAILED.json" failed "$overall_rc"
  log_msg "exp39 boundary host=$host had failures rc=$overall_rc"
  exit "$overall_rc"
fi
EOF_RUNNER
  }
}


write_manifest() {
  python - "$MANIFEST_PATH" "$MASTER_RUN_ID" "$REMOTE_MASTER_ROOT" \
    "$FIXED_LATTICE_SIZES" "$NUM_DISORDER_SAMPLES" "$NUM_KP_GRID_POINTS" \
    "$NUM_BURN_IN_SWEEPS" "$MAX_EFFECTIVE_NUM_BURN_IN_SWEEPS" "$NUM_MEASUREMENTS" \
    "$NUM_SWEEPS_BETWEEN_MEASUREMENTS" "$BLOCK_COUNT" "$NUM_BOOTSTRAP" \
    "$WINDING_HEATBATH_SWEEPS" "$PROJECTION_MODE" "$DISORDER_SEED_SCOPE" \
    "$DISORDER_REALIZATION_MODE" <<'PY'
import json, sys
from pathlib import Path
(manifest_path, master_run_id, remote_master_root, lattice_sizes, num_disorder,
 grid, burn, max_burn, meas, stride, blocks, boot, winding, projection_mode,
 seed_scope, realization_mode) = sys.argv[1:]
Path(manifest_path).write_text(json.dumps({
    "stage": "exp39_boundary",
    "master_run_id": master_run_id,
    "remote_master_root": remote_master_root,
    "lattice_sizes": lattice_sizes,
    "num_disorder_samples": int(num_disorder),
    "num_kp_grid_points": int(grid),
    "num_burn_in_sweeps": int(burn),
    "max_effective_num_burn_in_sweeps": int(max_burn),
    "num_measurements": int(meas),
    "num_sweeps_between_measurements": int(stride),
    "block_count": int(blocks),
    "num_bootstrap": int(boot),
    "winding_heatbath_sweeps": int(winding),
    "projection_mode": projection_mode,
    "disorder_seed_scope": seed_scope,
    "disorder_realization_mode": realization_mode,
    "nodes": [],
}, indent=2, sort_keys=True), encoding="utf-8")
PY
}


append_manifest_node() {
  local host="$1" run_root="$2" screen_name="$3" log_path="$4" cells_blob="$5"
  python - "$MANIFEST_PATH" "$host" "$run_root" "$screen_name" "$log_path" "$cells_blob" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
cells = []
for entry in sys.argv[6].split(";"):
    if not entry.strip():
        continue
    p, seed_base, q_csv = entry.split("|")
    cells.append({"p": float(p), "seed_base": int(seed_base),
                  "q_values": [float(x) for x in q_csv.split(",")],
                  "ptag": "p" + p.replace(".", "p")})
payload["nodes"].append({"host": sys.argv[2], "run_root": sys.argv[3],
                         "screen_name": sys.argv[4], "log_path": sys.argv[5],
                         "cells": cells})
path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
PY
}


launch_host() {
  local host="$1" cells_blob="$2"
  local tag repo_dir run_root runner_path log_path screen_name runner_tmp remote_command eff_workers
  tag="$(host_tag "$host")"
  repo_dir="$REMOTE_REPO_BASE/$tag"
  run_root="$REMOTE_MASTER_ROOT/$tag"
  runner_path="$run_root/run_boundary.sh"
  log_path="$REMOTE_BASE/logs/${MASTER_RUN_ID}_${tag}.log"
  screen_name="exp39B_${RUN_TIMESTAMP}_${tag}"

  append_manifest_node "$host" "$run_root" "$screen_name" "$log_path" "$cells_blob"

  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'HOST=%s SCREEN=%s LOG=%s\n  CELLS=%s\n' "$host" "$screen_name" "$log_path" "$cells_blob"
    return 0
  fi

  verify_remote_env "$host"
  if [[ "$SKIP_SYNC" != "1" ]]; then sync_remote_repo "$host" "$repo_dir"; fi

  # Effective workers: explicit NUM_WORKERS wins; else detect nproc at launch time
  # (outside the screen, where it is reliable) and leave a little headroom.
  if [[ "$NUM_WORKERS" != "0" ]]; then
    eff_workers="$NUM_WORKERS"
  else
    eff_workers="$(ssh yuany "ssh ${host} nproc" 2>/dev/null | tr -dc '0-9')"
    if [[ -z "$eff_workers" || "$eff_workers" -lt 1 ]]; then eff_workers=8; fi
    if [[ "$eff_workers" -gt 8 ]]; then eff_workers=$(( eff_workers - 4 )); fi
  fi
  echo "[boundary] $host effective workers=$eff_workers"

  runner_tmp="$(mktemp)"
  build_remote_runner "$host" "$repo_dir" "$run_root" "$cells_blob" "$eff_workers" > "$runner_tmp"
  ssh yuany "ssh ${host} 'mkdir -p $(quote_arg "$run_root") $(quote_arg "$REMOTE_BASE/logs")'"
  ssh yuany "ssh ${host} 'cat > $(quote_arg "$runner_path")'" < "$runner_tmp"
  rm -f "$runner_tmp"

  printf -v remote_command 'chmod +x %q && if screen -ls | grep -q %q; then echo %q >&2; exit 24; fi && screen -dmS %q bash -lc %q && printf "HOST=%%s\nSCREEN_NAME=%%s\nLOG_PATH=%%s\nRUN_ROOT=%%s\n" %q %q %q %q' \
    "$runner_path" "[.]${screen_name}[[:space:]]" "screen session already exists: $screen_name" \
    "$screen_name" "exec $(quote_arg "$runner_path") >> $(quote_arg "$log_path") 2>&1" \
    "$host" "$screen_name" "$log_path" "$run_root"
  ssh yuany "ssh ${host} $(quote_arg "$remote_command")"
}


main() {
  mkdir -p "$STAGE_DIR"
  write_manifest

  # Unique hosts in first-seen order (bash 3.2 safe: no associative arrays).
  local hosts_order=() seen=" " host p seed_base q_csv cell blob ch cp cs cq
  for cell in "${CELL_LIST[@]}"; do
    IFS='|' read -r host p seed_base q_csv <<< "$cell"
    case "$seen" in
      *" $host "*) ;;
      *) hosts_order+=("$host"); seen="$seen$host ";;
    esac
  done

  printf 'MASTER_RUN_ID=%s\nMANIFEST_PATH=%s\nREMOTE_MASTER_ROOT=%s\n' \
    "$MASTER_RUN_ID" "$MANIFEST_PATH" "$REMOTE_MASTER_ROOT"
  for host in "${hosts_order[@]}"; do
    blob=""
    for cell in "${CELL_LIST[@]}"; do
      IFS='|' read -r ch cp cs cq <<< "$cell"
      if [[ "$ch" == "$host" ]]; then
        blob="${blob:+$blob;}${cp}|${cs}|${cq}"
      fi
    done
    launch_host "$host" "$blob"
  done
}

main "$@"
