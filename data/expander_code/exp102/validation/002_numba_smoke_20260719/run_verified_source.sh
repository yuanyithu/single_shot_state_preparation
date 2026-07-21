#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "usage: run_verified_source.sh DEPLOYMENT_ROOT COMMIT ARCHIVE_SHA256 MANIFEST_SHA256 COMMAND [ARGS...]" >&2
  exit 64
fi

deployment_root=$1
expected_commit=$2
expected_archive_sha256=$3
expected_manifest_sha256=$4
shift 4

[[ -d $deployment_root ]] || {
  echo "deployment root is not a directory" >&2
  exit 66
}
deployment_root=$(cd "$deployment_root" && pwd -P)

source_dir=$deployment_root/source
archive=$deployment_root/SOURCE.tar
manifest=$deployment_root/SOURCE_MANIFEST.json
commit_marker=$deployment_root/SOURCE_COMMIT

[[ $expected_commit =~ ^[0-9a-f]{40}$ ]] || {
  echo "expected commit is not a full lowercase Git SHA" >&2
  exit 65
}
[[ $expected_archive_sha256 =~ ^[0-9a-f]{64}$ ]] || {
  echo "expected archive SHA256 is invalid" >&2
  exit 65
}
[[ $expected_manifest_sha256 =~ ^[0-9a-f]{64}$ ]] || {
  echo "expected manifest SHA256 is invalid" >&2
  exit 65
}
[[ -d $source_dir && -f $archive && -f $manifest && -f $commit_marker ]] || {
  echo "deployed source evidence is incomplete" >&2
  exit 66
}
[[ $(tr -d '\r\n' <"$commit_marker") == "$expected_commit" ]] || {
  echo "deployed source commit marker mismatch" >&2
  exit 66
}

# These expected digests come from the clean local launcher. No project Python
# or source-tree shell code has run when these checks execute.
printf '%s  %s\n' "$expected_archive_sha256" "$archive" | sha256sum -c - >/dev/null
printf '%s  %s\n' "$expected_manifest_sha256" "$manifest" | sha256sum -c - >/dev/null

archive_files=$(mktemp "$deployment_root/.archive-files.XXXXXX")
source_files=$(mktemp "$deployment_root/.source-files.XXXXXX")
archive_tree=$(mktemp -d "$deployment_root/.archive-tree.XXXXXX")
cleanup() {
  rm -f "$archive_files" "$source_files"
  rm -rf -- "$archive_tree"
}
trap cleanup EXIT
trap 'exit 70' INT TERM

# Reject both changed archived files and untracked shadow files before Python
# can import sitecustomize or any project module.
tar_list_args=(-tf "$archive")
if tar --help 2>&1 | grep -q -- '--quoting-style'; then
  # GNU tar escapes non-ASCII names by default, unlike bsdtar and find.
  tar_list_args=(--quoting-style=literal -tf "$archive")
fi
tar "${tar_list_args[@]}" | sed -e '/\/$/d' -e 's#^\./##' | LC_ALL=C sort >"$archive_files"
(
  cd "$source_dir"
  find . \( -type f -o -type l \) -print | sed 's#^\./##' | LC_ALL=C sort
) >"$source_files"
cmp -s "$archive_files" "$source_files" || {
  echo "deployed source tree contains missing or unexpected files" >&2
  exit 67
}
tar -xf "$archive" -C "$archive_tree"
diff -qr "$archive_tree" "$source_dir" >/dev/null || {
  echo "deployed source tree differs from the verified archive" >&2
  exit 67
}

trap - EXIT INT TERM
cleanup
cd "$source_dir"
export PYTHONPATH=.
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONOPTIMIZE
export EXP102_SOURCE_COMMIT=$expected_commit
export PYTEST_ADDOPTS="-p no:cacheprovider"
export NUMBA_CACHE_DIR="$deployment_root/numba-cache-${HOSTNAME:-unknown}"
export NUMBA_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
exec "$@"
