import io
import subprocess
import tarfile
from importlib import import_module
from pathlib import Path

import pytest

from data.expander_code.exp102.exp102_pipeline.io import (
    atomic_json, sha256_file, verify_source_identity,
)


ORCHESTRATOR = import_module(
    "data.expander_code.exp102.validation.002_numba_smoke_20260719.orchestrate_ladder"
)


def test_deployed_source_manifest_binds_commit_and_file_hashes(tmp_path):
    deployment = tmp_path / "repo"
    source = deployment / "source"
    source.mkdir(parents=True)
    tracked = source / "tracked.py"
    tracked.write_text("VALUE = 1\n", encoding="ascii")
    commit = "a" * 40
    archive = deployment / "SOURCE.tar"
    with tarfile.open(archive, "w") as handle:
        content = tracked.read_bytes()
        info = tarfile.TarInfo("tracked.py")
        info.size = len(content)
        handle.addfile(info, io.BytesIO(content))
    archive_sha256 = sha256_file(archive)
    (deployment / "SOURCE_COMMIT").write_text(commit + "\n", encoding="ascii")
    (deployment / "ARCHIVE_SHA256").write_text(archive_sha256 + "\n", encoding="ascii")
    atomic_json(deployment / "SOURCE_MANIFEST.json", {
        "source_identity_version": "exp102.source.v1",
        "source_commit": commit,
        "archive_sha256": archive_sha256,
        "files": [{"path": "tracked.py", "sha256": sha256_file(tracked)}],
    })

    identity = verify_source_identity(source, commit)
    assert identity["mode"] == "archive" and identity["file_count"] == 1
    assert identity["manifest_sha256"] == sha256_file(deployment / "SOURCE_MANIFEST.json")

    verifier = (
        Path(__file__).resolve().parents[1]
        / "validation/002_numba_smoke_20260719/run_verified_source.sh"
    )
    command = (
        "bash", str(verifier), str(deployment), commit, archive_sha256,
        sha256_file(deployment / "SOURCE_MANIFEST.json"),
        "sh", "-c", "printf verified",
    )
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    assert completed.stdout == "verified"

    (source / "sitecustomize.py").write_text("raise RuntimeError('shadowed')\n", encoding="ascii")
    rejected = subprocess.run(command, capture_output=True, text=True)
    assert rejected.returncode != 0
    assert "missing or unexpected files" in rejected.stderr
    (source / "sitecustomize.py").unlink()

    tracked.write_text("VALUE = 2\n", encoding="ascii")
    with pytest.raises(ValueError, match="file hash mismatch"):
        verify_source_identity(source, commit)
    with pytest.raises(ValueError, match="full lowercase Git SHA"):
        verify_source_identity(source, "abcdef0")


def test_ladder_bootstrap_verifies_before_project_code_and_wraps_failures():
    shell = ORCHESTRATOR.verified_bootstrap(
        Path("/deployment"), "1" * 40, "2" * 64, "3" * 64,
        Path("/stage"), Path("/log"), ("false",),
    )
    assert shell.index("sha256sum") < shell.index("run_stage_wrapper.sh")
    assert shell.index("run_stage_wrapper.sh") < shell.index("run_verified_source.sh")
    assert "bash -s -- /stage /log" in shell

    with pytest.raises(ValueError, match="archive SHA256"):
        ORCHESTRATOR.verified_bootstrap(
            Path("/deployment"), "1" * 40, "short", "3" * 64,
            Path("/stage"), Path("/log"), ("false",),
        )
