#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONTROL_SOURCE_DIR="$PROJECT_ROOT/data/2D_final"
SOURCE_COMMIT="70ea84cd5fe800948a619e4a070c693e684e5b4b"
REMOTE_HOST="nd-2"
REMOTE_BASE="/home/DATA1/users/yuany/.single_shot"
CONDA_BIN="/home/DATA1/users/yuany/miniconda3/bin/conda"
CONDA_ENV="11"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ID="${RUN_ID:-2d_final_nd2_staged_${RUN_TIMESTAMP}}"
REMOTE_REPO="$REMOTE_BASE/repos/${RUN_ID}_${SOURCE_COMMIT:0:12}"
REMOTE_STAGE_ROOT="$REMOTE_BASE/runs/$RUN_ID"
REMOTE_CONTROL="$REMOTE_STAGE_ROOT/control"
REMOTE_LOG="$REMOTE_STAGE_ROOT/logs/stage.log"
SCREEN_NAME="ssprep_${RUN_ID}"
DRY_RUN="${DRY_RUN:-0}"

CONTROL_FILES=(
  nd2_staged_experiment_config.json
  nd2_qpositive_sentinel.py
  audit_nd2_staged_experiment.py
  run_nd2_staged_experiment.sh
)

quote_arg() {
  printf '%q' "$1"
}

relay() {
  ssh -o BatchMode=yes yuany "ssh -o BatchMode=yes $REMOTE_HOST $(quote_arg "$1")"
}

if [[ "$(git -C "$PROJECT_ROOT" rev-parse HEAD)" != "$SOURCE_COMMIT" ]]; then
  echo "HEAD must remain at the approved source commit $SOURCE_COMMIT" >&2
  exit 10
fi
git -C "$PROJECT_ROOT" cat-file -e "${SOURCE_COMMIT}^{commit}"
for filename in "${CONTROL_FILES[@]}"; do
  if [[ ! -f "$CONTROL_SOURCE_DIR/$filename" ]]; then
    echo "missing control file: $CONTROL_SOURCE_DIR/$filename" >&2
    exit 11
  fi
done

printf 'RUN_ID=%s\nREMOTE_HOST=%s\nREMOTE_REPO=%s\nREMOTE_STAGE_ROOT=%s\nSCREEN_NAME=%s\nSOURCE_COMMIT=%s\n' \
  "$RUN_ID" "$REMOTE_HOST" "$REMOTE_REPO" "$REMOTE_STAGE_ROOT" \
  "$SCREEN_NAME" "$SOURCE_COMMIT"
if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi

relay "set -euo pipefail; \
  test ! -e $(quote_arg "$REMOTE_REPO"); \
  test ! -e $(quote_arg "$REMOTE_STAGE_ROOT"); \
  mkdir -p $(quote_arg "$REMOTE_REPO") $(quote_arg "$REMOTE_CONTROL") $(quote_arg "$REMOTE_STAGE_ROOT/logs")"

git -C "$PROJECT_ROOT" archive --format=tar "$SOURCE_COMMIT" \
  | ssh -o BatchMode=yes yuany \
      "ssh -o BatchMode=yes $REMOTE_HOST 'tar -xf - -C $(quote_arg "$REMOTE_REPO")'"

COPYFILE_DISABLE=1 tar --no-xattrs -C "$CONTROL_SOURCE_DIR" -cf - "${CONTROL_FILES[@]}" \
  | ssh -o BatchMode=yes yuany \
      "ssh -o BatchMode=yes $REMOTE_HOST 'tar -xf - -C $(quote_arg "$REMOTE_CONTROL")'"

relay "set -euo pipefail; \
  printf '%s\\n' $(quote_arg "$SOURCE_COMMIT") > $(quote_arg "$REMOTE_REPO/SOURCE_COMMIT"); \
  chmod 0555 \
    $(quote_arg "$REMOTE_CONTROL/nd2_qpositive_sentinel.py") \
    $(quote_arg "$REMOTE_CONTROL/audit_nd2_staged_experiment.py") \
    $(quote_arg "$REMOTE_CONTROL/run_nd2_staged_experiment.sh"); \
  sha256sum $(printf '%q ' "${CONTROL_FILES[@]/#/$REMOTE_CONTROL/}") \
    > $(quote_arg "$REMOTE_CONTROL/control_sha256s.txt"); \
  chmod -R a-w $(quote_arg "$REMOTE_REPO")"

relay "set -euo pipefail; \
  date -Is; hostname; nproc; uptime; free -h; df -h / /home/DATA1; \
  command -v screen; command -v taskset; command -v setsid; \
  test -x $(quote_arg "$CONDA_BIN"); \
  test \"\$(hostname)\" = nd-2; \
  test \"\$(nproc)\" -eq 80; \
  mem_kib=\$(awk '/MemAvailable:/ {print \$2}' /proc/meminfo); \
  test \"\$mem_kib\" -ge \$((192 * 1024 * 1024)); \
  test -f $(quote_arg "$REMOTE_REPO/src/production_chunked_scan.py"); \
  test \"\$(<$(quote_arg "$REMOTE_REPO/SOURCE_COMMIT"))\" = $(quote_arg "$SOURCE_COMMIT"); \
  ps -eo user=,pcpu= | awk -v self=\"\$USER\" '\$1 != self {sum += \$2} END {printf \"other_user_cpu_percent_sum=%.1f competition_authorized=1\\n\", sum + 0}'; \
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1; \
  export NUMBA_CACHE_DIR=$(quote_arg "$REMOTE_STAGE_ROOT/cache/numba"); \
  export MPLCONFIGDIR=$(quote_arg "$REMOTE_STAGE_ROOT/cache/matplotlib"); \
  export TMPDIR=$(quote_arg "$REMOTE_STAGE_ROOT/cache/tmp"); \
  mkdir -p \"\$NUMBA_CACHE_DIR\" \"\$MPLCONFIGDIR\" \"\$TMPDIR\"; \
  $(quote_arg "$CONDA_BIN") run --no-capture-output -n $(quote_arg "$CONDA_ENV") \
    python $(quote_arg "$REMOTE_CONTROL/audit_nd2_staged_experiment.py") self-test \
    --config $(quote_arg "$REMOTE_CONTROL/nd2_staged_experiment_config.json") \
    --repo-root $(quote_arg "$REMOTE_REPO")"

remote_runner_command="exec $(quote_arg "$REMOTE_CONTROL/run_nd2_staged_experiment.sh") \
$(quote_arg "$REMOTE_REPO") \
$(quote_arg "$REMOTE_STAGE_ROOT") \
$(quote_arg "$SOURCE_COMMIT") \
$(quote_arg "$CONDA_BIN") \
$(quote_arg "$CONDA_ENV") >> $(quote_arg "$REMOTE_LOG") 2>&1"

relay "set -euo pipefail; \
  screen_listing=\$(screen -ls 2>/dev/null || true); \
  if grep -q $(quote_arg "[.]${SCREEN_NAME}[[:space:]]") <<< \"\$screen_listing\"; then \
    echo $(quote_arg "screen already exists: $SCREEN_NAME") >&2; exit 24; \
  fi; \
  screen -dmS $(quote_arg "$SCREEN_NAME") bash -lc $(quote_arg "$remote_runner_command"); \
  sleep 2; \
  screen_listing=\$(screen -ls 2>/dev/null || true); \
  grep $(quote_arg "[.]${SCREEN_NAME}[[:space:]]") <<< \"\$screen_listing\"; \
  printf 'REMOTE_LOG=%s\\n' $(quote_arg "$REMOTE_LOG")"
