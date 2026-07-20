import hashlib
from importlib import import_module
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from data.expander_code.exp102.exp102_pipeline.io import verify_source_identity


pytestmark = pytest.mark.skipif(
    shutil.which("git") is None,
    reason="source-package builder tests require the build host's Git CLI",
)


builder = import_module(
    "data.expander_code.exp102.validation.002_numba_smoke_20260719.build_source_package"
).build_source_package


def _git(repo, *arguments):
    return subprocess.run(
        ("git", "-C", str(repo), *arguments),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _repository(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(("git", "init", "-q", str(repo)), check=True)
    _git(repo, "config", "user.email", "exp102-test@example.invalid")
    _git(repo, "config", "user.name", "exp102 test")
    (repo / "nested").mkdir()
    (repo / "nested/data.txt").write_text("payload\n", encoding="ascii")
    executable = repo / "runner.sh"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="ascii")
    executable.chmod(0o755)
    _git(repo, "add", "nested/data.txt", "runner.sh")
    _git(repo, "commit", "-q", "-m", "fixture")
    return repo, _git(repo, "rev-parse", "HEAD")


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_builder_creates_verified_archive_manifest_and_fresh_source(tmp_path):
    repo, commit = _repository(tmp_path)
    deployment = tmp_path / "deployment"

    result = builder(repo, commit, deployment)

    assert set(result) == {
        "source_commit", "archive_sha256", "manifest_sha256", "file_count",
    }
    assert result["source_commit"] == commit
    assert result["archive_sha256"] == _sha256(deployment / "SOURCE.tar")
    assert result["manifest_sha256"] == _sha256(deployment / "SOURCE_MANIFEST.json")
    assert (deployment / "SOURCE_COMMIT").read_text(encoding="ascii") == commit + "\n"
    assert (deployment / "ARCHIVE_SHA256").read_text(encoding="ascii") == result["archive_sha256"] + "\n"
    assert (deployment / "source/nested/data.txt").read_text(encoding="ascii") == "payload\n"
    assert os.access(deployment / "source/runner.sh", os.X_OK)

    manifest = json.loads((deployment / "SOURCE_MANIFEST.json").read_text(encoding="ascii"))
    assert result["file_count"] == len(manifest["files"]) == 2
    assert manifest["files"] == sorted(manifest["files"], key=lambda item: item["path"])
    identity = verify_source_identity(deployment / "source", commit)
    assert identity["archive_sha256"] == result["archive_sha256"]
    assert identity["manifest_sha256"] == result["manifest_sha256"]

    with pytest.raises(FileExistsError):
        builder(repo, commit, deployment)


def test_builder_rejects_dirty_worktree_and_non_head_commit(tmp_path):
    repo, first_commit = _repository(tmp_path)
    (repo / "nested/data.txt").write_text("changed\n", encoding="ascii")
    with pytest.raises(ValueError, match="not clean"):
        builder(repo, first_commit, tmp_path / "dirty")
    assert not (tmp_path / "dirty").exists()

    _git(repo, "add", "nested/data.txt")
    _git(repo, "commit", "-q", "-m", "second")
    with pytest.raises(ValueError, match="not the repository HEAD"):
        builder(repo, first_commit, tmp_path / "old-head")
    with pytest.raises(ValueError, match="full lowercase Git SHA"):
        builder(repo, "abcdef0", tmp_path / "short-sha")


def test_builder_rejects_non_regular_archive_members(tmp_path):
    repo, _ = _repository(tmp_path)
    (repo / "link").symlink_to("nested/data.txt")
    _git(repo, "add", "link")
    _git(repo, "commit", "-q", "-m", "symlink")
    commit = _git(repo, "rev-parse", "HEAD")

    with pytest.raises(ValueError, match="non-regular member"):
        builder(repo, commit, tmp_path / "unsafe")
    assert not (tmp_path / "unsafe").exists()
