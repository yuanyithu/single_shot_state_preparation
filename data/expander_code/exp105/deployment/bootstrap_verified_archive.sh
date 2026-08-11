#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: bootstrap_verified_archive.sh DEPLOYMENT_ROOT MANIFEST_SHA256 ARCHIVE_SHA256 COMMAND [ARGS...]" >&2
  exit 64
fi

deployment_root=$1
expected_manifest_sha256=$2
expected_archive_sha256=$3
shift 3

[[ -d $deployment_root ]] || {
  echo "deployment root is not a directory" >&2
  exit 66
}
deployment_root=$(cd "$deployment_root" && pwd -P)
archive=$deployment_root/SOURCE.tar
manifest=$deployment_root/DEPLOYMENT_MANIFEST.json
archive_marker=$deployment_root/ARCHIVE_SHA256
manifest_marker=$deployment_root/DEPLOYMENT_MANIFEST_SHA256

[[ $expected_manifest_sha256 =~ ^[0-9a-f]{64}$ ]] || {
  echo "expected deployment manifest SHA256 is invalid" >&2
  exit 65
}
[[ $expected_archive_sha256 =~ ^[0-9a-f]{64}$ ]] || {
  echo "expected source archive SHA256 is invalid" >&2
  exit 65
}
[[ -f $archive && -f $manifest && -f $archive_marker && -f $manifest_marker ]] || {
  echo "deployment bootstrap evidence is incomplete" >&2
  exit 66
}
[[ $(tr -d '\r\n' <"$archive_marker") == "$expected_archive_sha256" ]] || {
  echo "source archive marker mismatch" >&2
  exit 66
}
[[ $(tr -d '\r\n' <"$manifest_marker") == "$expected_manifest_sha256" ]] || {
  echo "deployment manifest marker mismatch" >&2
  exit 66
}
printf '%s  %s\n' "$expected_archive_sha256" "$archive" | sha256sum -c - >/dev/null
printf '%s  %s\n' "$expected_manifest_sha256" "$manifest" | sha256sum -c - >/dev/null

python3 - "$manifest" "$expected_archive_sha256" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="ascii") as handle:
    manifest = json.load(handle)
if manifest.get("archive_sha256") != sys.argv[2]:
    raise SystemExit("deployment manifest is not bound to the expected source archive")
PY

source_dir=$deployment_root/source
if [[ ! -e $source_dir ]]; then
  temporary_source=$(mktemp -d "$deployment_root/.source.partial.XXXXXX")
  tar -xf "$archive" -C "$temporary_source"
  mv "$temporary_source" "$source_dir"
elif [[ ! -d $source_dir ]]; then
  echo "deployment source path exists but is not a directory" >&2
  exit 67
fi

wrapper=data/expander_code/exp105/deployment/run_verified_source.sh
tar -xOf "$archive" "$wrapper" \
  | bash -s -- "$deployment_root" "$expected_manifest_sha256" "$@"
