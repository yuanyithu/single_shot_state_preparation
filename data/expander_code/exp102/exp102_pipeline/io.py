import hashlib
import json
import os
import re
import subprocess
import tarfile
import tempfile
from pathlib import Path

import numpy as np


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value):
    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="ascii") as handle:
            handle.write(canonical_json(value) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_npz_no_pickle(path):
    return np.load(path, allow_pickle=False)


def verify_source_identity(source_dir, expected_commit):
    source_dir = Path(source_dir).resolve()
    if re.fullmatch(r"[0-9a-f]{40}", str(expected_commit)) is None:
        raise ValueError("source commit must be a full lowercase Git SHA")
    if (source_dir / ".git").exists():
        head = subprocess.run(
            ("git", "-C", str(source_dir), "rev-parse", "HEAD"),
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ("git", "-C", str(source_dir), "status", "--porcelain", "--untracked-files=all"),
            check=True, capture_output=True, text=True,
        ).stdout
        if head != expected_commit or dirty:
            raise ValueError("local source tree is not the requested clean Git commit")
        return {"source_commit": head, "mode": "git"}

    deployment_root = source_dir.parent
    marker = deployment_root / "SOURCE_COMMIT"
    archive_path = deployment_root / "SOURCE.tar"
    archive_digest_path = deployment_root / "ARCHIVE_SHA256"
    manifest_path = deployment_root / "SOURCE_MANIFEST.json"
    if not marker.is_file() or marker.read_text(encoding="ascii").strip() != expected_commit:
        raise ValueError("deployed source commit marker mismatch")
    if not manifest_path.is_file():
        raise ValueError("deployed source manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    if set(manifest) != {"source_identity_version", "source_commit", "archive_sha256", "files"}:
        raise ValueError("deployed source manifest fields are invalid")
    if manifest["source_identity_version"] != "exp102.source.v1":
        raise ValueError("deployed source manifest version mismatch")
    if manifest["source_commit"] != expected_commit:
        raise ValueError("deployed source manifest commit mismatch")
    if re.fullmatch(r"[0-9a-f]{64}", str(manifest["archive_sha256"])) is None:
        raise ValueError("deployed archive digest is invalid")
    if (not archive_path.is_file() or not archive_digest_path.is_file()
            or archive_digest_path.read_text(encoding="ascii").strip() != manifest["archive_sha256"]
            or sha256_file(archive_path) != manifest["archive_sha256"]):
        raise ValueError("deployed source archive digest mismatch")
    files = manifest["files"]
    if not isinstance(files, list) or not files:
        raise ValueError("deployed source manifest has no files")
    expected_files = {}
    for item in files:
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise ValueError("deployed source manifest file entry is invalid")
        relative = Path(item["path"])
        normalized = relative.as_posix()
        if relative.is_absolute() or ".." in relative.parts or normalized in expected_files:
            raise ValueError("deployed source manifest path is invalid or duplicated")
        expected_files[normalized] = item["sha256"]
        path = (source_dir / relative).resolve()
        if source_dir not in path.parents or not path.is_file():
            raise ValueError(f"deployed source file is missing: {item['path']}")
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"deployed source file hash mismatch: {item['path']}")
    with tarfile.open(archive_path, "r:") as archive:
        archived_files = {}
        for member in archive.getmembers():
            if member.isdir():
                continue
            relative = Path(member.name)
            if (not member.isfile() or relative.is_absolute() or ".." in relative.parts
                    or relative.as_posix() in archived_files):
                raise ValueError("deployed source archive contains an invalid member")
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError("deployed source archive member cannot be read")
            archived_files[relative.as_posix()] = hashlib.sha256(handle.read()).hexdigest()
    if archived_files != expected_files:
        raise ValueError("deployed source manifest does not match the verified archive")
    actual_files = {
        path.relative_to(source_dir).as_posix()
        for path in source_dir.rglob("*")
        if path.is_file()
    }
    if actual_files != set(expected_files):
        raise ValueError("deployed source tree contains missing or unexpected files")
    return {
        "source_commit": expected_commit,
        "mode": "archive",
        "archive_sha256": manifest["archive_sha256"],
        "manifest_sha256": sha256_file(manifest_path),
        "file_count": len(files),
    }
