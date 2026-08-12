#!/usr/bin/env bash

set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "usage: $0 RUN_ID" >&2
  exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ID="$1"
REMOTE_HOST="nd-2"
REMOTE_BASE="/home/DATA1/users/yuany/.single_shot"
REMOTE_STAGE_ROOT="$REMOTE_BASE/runs/$RUN_ID"
LOCAL_STAGE_ROOT="$PROJECT_ROOT/data/2D_final/nd2_runs/$RUN_ID"
ALLOW_INCOMPLETE="${ALLOW_INCOMPLETE:-0}"

quote_arg() {
  printf '%q' "$1"
}

relay() {
  ssh -o BatchMode=yes yuany "ssh -o BatchMode=yes $REMOTE_HOST $(quote_arg "$1")"
}

if [[ -e "$LOCAL_STAGE_ROOT" ]]; then
  echo "refusing to overwrite existing local collection: $LOCAL_STAGE_ROOT" >&2
  exit 10
fi

phase="$(relay "test -f $(quote_arg "$REMOTE_STAGE_ROOT/control/phase") && cat $(quote_arg "$REMOTE_STAGE_ROOT/control/phase")")"
if [[ "$ALLOW_INCOMPLETE" != "1" && "$phase" != "done_rc_0" ]]; then
  echo "remote stage is not complete (phase=$phase); set ALLOW_INCOMPLETE=1 for a diagnostic-only collection" >&2
  exit 11
fi

mkdir -p "$LOCAL_STAGE_ROOT"

remote_file_list="$(relay "set -euo pipefail; cd $(quote_arg "$REMOTE_STAGE_ROOT"); \
  find . -type f \
    ! -path './cache/*' \
    ! -path './runs/*/chunks/*' \
    ! -path './runs/*/preflight/*' \
    ! -path './runs/qpositive_sentinel/parts/*' \
    ! -path './control/pids/*' \
    ! -path '*/__pycache__/*' \
    ! -name '*.png' \
    -print | LC_ALL=C sort")"
if [[ -z "$remote_file_list" ]]; then
  echo "no deliverable files found under $REMOTE_STAGE_ROOT" >&2
  exit 12
fi

printf '%s\n' "$remote_file_list" \
  | ssh -o BatchMode=yes yuany \
      "ssh -o BatchMode=yes $REMOTE_HOST 'cd $(quote_arg "$REMOTE_STAGE_ROOT") && tar -cf - -T -'" \
  | tar -xf - -C "$LOCAL_STAGE_ROOT"

RUN_ID="$RUN_ID" REMOTE_STAGE_ROOT="$REMOTE_STAGE_ROOT" \
LOCAL_STAGE_ROOT="$LOCAL_STAGE_ROOT" PHASE="$phase" \
  python3 -c '
import hashlib, json, os
from datetime import datetime
from pathlib import Path

root = Path(os.environ["LOCAL_STAGE_ROOT"])
files = []
for file_path in sorted(path for path in root.rglob("*") if path.is_file()):
    digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
    files.append({
        "path": file_path.relative_to(root).as_posix(),
        "size_bytes": file_path.stat().st_size,
        "sha256": digest,
    })
report = {
    "schema_version": 1,
    "collected_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    "run_id": os.environ["RUN_ID"],
    "remote_stage_root": os.environ["REMOTE_STAGE_ROOT"],
    "remote_phase": os.environ["PHASE"],
    "excluded": ["remote chunks", "q-positive task parts", "preflight scratch assets", "cache and __pycache__", "automatic preview PNG"],
    "files": files,
}
(root / "collection_audit.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(f"collected_files={len(files)}")
print(f"local_stage_root={root}")
'
