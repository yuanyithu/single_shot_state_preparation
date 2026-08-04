#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: run_verified_source.sh DEPLOYMENT_ROOT MANIFEST_SHA256 COMMAND [ARGS...]" >&2
  exit 64
fi

deployment_root=$1
expected_manifest_sha256=$2
shift 2

[[ -d $deployment_root ]] || {
  echo "deployment root is not a directory" >&2
  exit 66
}
deployment_root=$(cd "$deployment_root" && pwd -P)
source_dir=$deployment_root/source
archive=$deployment_root/SOURCE.tar
archive_marker=$deployment_root/ARCHIVE_SHA256
source_manifest=$deployment_root/SOURCE_MANIFEST.json
deployment_manifest=$deployment_root/DEPLOYMENT_MANIFEST.json
deployment_manifest_marker=$deployment_root/DEPLOYMENT_MANIFEST_SHA256
commit_marker=$deployment_root/SOURCE_COMMIT

[[ $expected_manifest_sha256 =~ ^[0-9a-f]{64}$ ]] || {
  echo "expected deployment manifest SHA256 is invalid" >&2
  exit 65
}
[[ -d $source_dir && -f $archive && -f $archive_marker \
  && -f $source_manifest && -f $deployment_manifest \
  && -f $deployment_manifest_marker && -f $commit_marker ]] || {
  echo "deployed source evidence is incomplete" >&2
  exit 66
}
[[ $(tr -d '\r\n' <"$deployment_manifest_marker") == "$expected_manifest_sha256" ]] || {
  echo "deployment manifest marker mismatch" >&2
  exit 66
}
printf '%s  %s\n' "$expected_manifest_sha256" "$deployment_manifest" \
  | sha256sum -c - >/dev/null
archive_sha256=$(tr -d '\r\n' <"$archive_marker")
[[ $archive_sha256 =~ ^[0-9a-f]{64}$ ]] || {
  echo "source archive marker is invalid" >&2
  exit 66
}
printf '%s  %s\n' "$archive_sha256" "$archive" | sha256sum -c - >/dev/null
source_manifest_sha256=$(sha256sum "$source_manifest" | awk '{print $1}')
source_commit=$(tr -d '\r\n' <"$commit_marker")

python3 - "$deployment_manifest" "$source_manifest" \
  "$archive_sha256" "$source_manifest_sha256" "$source_commit" <<'PY'
import json
import re
import sys

deployment_path, source_path, archive_sha, source_sha, source_commit = sys.argv[1:]
with open(deployment_path, encoding="ascii") as handle:
    deployment = json.load(handle)
with open(source_path, encoding="ascii") as handle:
    source = json.load(handle)
if deployment.get("archive_sha256") != archive_sha:
    raise SystemExit("deployment manifest is not bound to the source archive")
if deployment.get("source_manifest_sha256") != source_sha:
    raise SystemExit("deployment manifest is not bound to the source manifest")
if deployment.get("source_commit") != source_commit:
    raise SystemExit("deployment manifest is not bound to the source commit marker")
if source.get("archive_sha256") != archive_sha:
    raise SystemExit("source manifest is not bound to the source archive")
if source.get("source_commit") != source_commit:
    raise SystemExit("source manifest is not bound to the source commit marker")
if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
    raise SystemExit("source commit marker is invalid")
PY

verify_tree() {
  local archive_files source_files archive_tree
  archive_files=$(mktemp "$deployment_root/.archive-files.XXXXXX")
  source_files=$(mktemp "$deployment_root/.source-files.XXXXXX")
  archive_tree=$(mktemp -d "$deployment_root/.archive-tree.XXXXXX")
  tar --quoting-style=literal -tf "$archive" \
    | sed -e '/\/$/d' -e 's#^\./##' | LC_ALL=C sort >"$archive_files"
  (
    cd "$source_dir"
    find . \( -type f -o -type l \) -print \
      | sed 's#^\./##' | LC_ALL=C sort
  ) >"$source_files"
  if ! cmp -s "$archive_files" "$source_files"; then
    rm -f "$archive_files" "$source_files"
    rm -rf -- "$archive_tree"
    return 1
  fi
  tar -xf "$archive" -C "$archive_tree"
  if ! diff -qr "$archive_tree" "$source_dir" >/dev/null; then
    rm -f "$archive_files" "$source_files"
    rm -rf -- "$archive_tree"
    return 1
  fi
  rm -f "$archive_files" "$source_files"
  rm -rf -- "$archive_tree"
}

verify_tree || {
  echo "deployed source differs from its verified archive" >&2
  exit 67
}

cd "$source_dir"
export PYTHONPATH=.
export PYTHONDONTWRITEBYTECODE=1
export PYTEST_ADDOPTS="-p no:cacheprovider"
export NUMBA_CACHE_DIR="$deployment_root/runtime-cache/numba"
export NUMBA_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
unset PYTHONOPTIMIZE
mkdir -p "$NUMBA_CACHE_DIR"

set +e
"$@"
command_status=$?
set -e
if ! verify_tree; then
  echo "verified source changed or emitted bytecode during execution" >&2
  exit 67
fi
exit "$command_status"
