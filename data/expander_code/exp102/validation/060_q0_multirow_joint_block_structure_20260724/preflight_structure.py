"""Fail-closed source and input provenance checks for validation 060."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[5]
ROOT = Path(__file__).resolve().parent
EXP102_ROOT = ROOT.parents[1]
CONFIG_PATH = ROOT / "structure_config.json"
OUTPUT_PATHS = (
    ROOT / "structure_report.json",
    ROOT / "independent_structure_audit.json",
)
COMMIT_RE = re.compile(r"[0-9a-f]{40}")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def load_json_strict(path):
    def reject_constant(value):
        raise ValueError(f"non-finite JSON constant: {value}")

    return json.loads(
        Path(path).read_text(encoding="ascii"), parse_constant=reject_constant,
    )


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(args, *, text=True):
    return subprocess.run(
        ["git", *args], cwd=PROJECT_ROOT, check=True,
        capture_output=True, text=text,
    ).stdout


def _exp102_path(relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError("configured artifact path escapes exp102 root")
    path = (EXP102_ROOT / relative).resolve()
    try:
        path.relative_to(EXP102_ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError("configured artifact path escapes exp102 root") from exc
    if path.is_symlink():
        raise RuntimeError(f"frozen artifact may not be a symlink: {path}")
    return path


def _configured_artifacts(config):
    rows = []
    for section in ("implementation", "documentation"):
        values = config[section]
        for key in sorted(value[:-7] for value in values if value.endswith("_sha256")):
            path_key = key
            sha_key = f"{key}_sha256"
            if path_key not in values:
                raise RuntimeError(f"missing path for {section}.{sha_key}")
            rows.append((_exp102_path(values[path_key]), values[sha_key]))
    inputs = config["inputs"]
    rows.extend((
        (_exp102_path(inputs["control"]), inputs["control_file_sha256"]),
        (
            _exp102_path(inputs["predecessor_report"]),
            inputs["predecessor_report_file_sha256"],
        ),
    ))
    unique = {}
    for path, expected in rows:
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise RuntimeError(f"invalid configured SHA256 for {path}")
        if path in unique and unique[path] != expected:
            raise RuntimeError(f"conflicting configured SHA256 for {path}")
        unique[path] = expected
    return sorted(unique.items(), key=lambda row: str(row[0]))


def _repo_relative(path):
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise RuntimeError(f"path is outside repository: {path}") from exc


def verify_for_launch(config=None):
    """Verify a clean committed source tree before the one-shot analyzer runs."""
    if config is None:
        config = load_json_strict(CONFIG_PATH)
    repository = Path(_git(["rev-parse", "--show-toplevel"]).strip()).resolve()
    if repository != PROJECT_ROOT.resolve():
        raise RuntimeError("validation is not running from its bound repository")
    source_commit = _git(["rev-parse", "HEAD"]).strip()
    if COMMIT_RE.fullmatch(source_commit) is None:
        raise RuntimeError("HEAD is not a full commit identity")
    if any(path.exists() for path in OUTPUT_PATHS):
        raise RuntimeError("validation 060 output already exists")

    # The official one-shot launch starts from a completely clean worktree.
    dirty = _git(["status", "--porcelain=v1", "--untracked-files=all"])
    if dirty:
        raise RuntimeError("validation 060 requires a clean committed worktree")

    artifacts = [(CONFIG_PATH.resolve(), sha256_file(CONFIG_PATH))]
    artifacts.extend(_configured_artifacts(config))
    rows = []
    for path, expected in artifacts:
        if not path.is_file():
            raise RuntimeError(f"missing frozen artifact: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"configured artifact SHA256 changed: {path}")
        relative = _repo_relative(path)
        _git(["ls-files", "--error-unmatch", "--", relative])
        committed = _git(["show", f"{source_commit}:{relative}"], text=False)
        if sha256_bytes(committed) != actual:
            raise RuntimeError(f"working bytes differ from source commit: {relative}")
        rows.append([relative, actual])
    rows.sort()
    source_tree_sha256 = sha256_bytes(canonical(rows).encode("ascii"))
    return {
        "config_sha256": sha256_file(CONFIG_PATH),
        "source_commit": source_commit,
        "source_file_count": len(rows),
        "source_tree_sha256": source_tree_sha256,
    }


def main():
    payload = {
        **verify_for_launch(),
        "status": "PRE_RUN_COMMITTED_SOURCE_PASS",
        "version": "exp102.q0_multirow_joint_block.preflight.v0",
    }
    print(json.dumps(payload, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
