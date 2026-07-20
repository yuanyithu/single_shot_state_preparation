"""Build a self-verifying exp102 deployment bundle from a clean Git commit."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile
import tempfile


FULL_GIT_SHA = re.compile(r"[0-9a-f]{40}")


def _git(repo, *arguments):
    return subprocess.run(
        ("git", "-C", str(repo), *arguments),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with open(temporary, "x", encoding="ascii") as handle:
            handle.write(value)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _safe_member_path(name):
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError("source archive contains an invalid member path")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("source archive contains an invalid member path")
    normalized = path.as_posix()
    if normalized != name.rstrip("/"):
        raise ValueError("source archive contains a non-canonical member path")
    return normalized


def _extract_and_hash_regular_files(archive_path, source_dir):
    records = []
    seen = set()
    with tarfile.open(archive_path, "r:") as archive:
        for member in archive.getmembers():
            relative = _safe_member_path(member.name)
            if relative in seen:
                raise ValueError("source archive contains a duplicate member")
            seen.add(relative)
            target = source_dir.joinpath(*PurePosixPath(relative).parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ValueError("source archive contains a non-regular member")
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError("source archive member cannot be read")
            digest = hashlib.sha256()
            with source, open(target, "xb") as output:
                for block in iter(lambda: source.read(1 << 20), b""):
                    digest.update(block)
                    output.write(block)
            os.chmod(target, member.mode & 0o777)
            records.append({"path": relative, "sha256": digest.hexdigest()})
    if not records:
        raise ValueError("source archive contains no regular files")
    return sorted(records, key=lambda item: item["path"])


def build_source_package(repo, commit, output_root):
    repo = Path(repo).expanduser().resolve()
    if not repo.is_dir():
        raise ValueError("repository path is not a directory")
    if FULL_GIT_SHA.fullmatch(str(commit)) is None:
        raise ValueError("commit must be a full lowercase Git SHA")
    repository_root = Path(_git(repo, "rev-parse", "--show-toplevel")).resolve()
    head = _git(repository_root, "rev-parse", "HEAD")
    if head != commit:
        raise ValueError("requested commit is not the repository HEAD")
    if _git(repository_root, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("repository worktree is not clean")

    requested_output = Path(output_root).expanduser()
    output_parent = requested_output.parent.resolve()
    output_parent.mkdir(parents=True, exist_ok=True)
    output_root = output_parent / requested_output.name
    if os.path.lexists(output_root):
        raise FileExistsError(f"deployment root already exists: {output_root}")

    temporary_root = Path(tempfile.mkdtemp(
        prefix=f".{output_root.name}.building.", dir=output_parent,
    ))
    try:
        archive_path = temporary_root / "SOURCE.tar"
        subprocess.run(
            ("git", "-C", str(repository_root), "archive", "--format=tar",
             f"--output={archive_path}", commit),
            check=True,
        )
        archive_sha256 = _sha256_file(archive_path)
        source_dir = temporary_root / "source"
        source_dir.mkdir()
        files = _extract_and_hash_regular_files(archive_path, source_dir)

        _atomic_text(temporary_root / "SOURCE_COMMIT", f"{commit}\n")
        _atomic_text(temporary_root / "ARCHIVE_SHA256", f"{archive_sha256}\n")
        manifest = {
            "source_identity_version": "exp102.source.v1",
            "source_commit": commit,
            "archive_sha256": archive_sha256,
            "files": files,
        }
        manifest_text = json.dumps(
            manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        ) + "\n"
        manifest_path = temporary_root / "SOURCE_MANIFEST.json"
        _atomic_text(manifest_path, manifest_text)
        manifest_sha256 = _sha256_file(manifest_path)

        os.replace(temporary_root, output_root)
        return {
            "source_commit": commit,
            "archive_sha256": archive_sha256,
            "manifest_sha256": manifest_sha256,
            "file_count": len(files),
        }
    except BaseException:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("repo")
    parser.add_argument("commit")
    parser.add_argument("output_root")
    args = parser.parse_args(argv)
    result = build_source_package(args.repo, args.commit, args.output_root)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
