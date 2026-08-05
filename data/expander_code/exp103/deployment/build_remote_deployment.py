"""Build an immutable, selective exp103 remote source deployment."""

import argparse
import hashlib
import json
import re
import subprocess
import tarfile
from pathlib import Path

from data.expander_code.exp103.exp103_pipeline.config import (
    REMOTE_CONFIG_SCHEMA,
    REMOTE_EXECUTION_PROFILE,
    load_config,
)
from data.expander_code.exp103.exp103_pipeline.identity import (
    REMOTE_DEPLOYMENT_SCHEMA,
    source_tree_sha256,
)
from data.expander_code.exp103.exp103_pipeline.io import (
    atomic_json,
    canonical_json,
    sha256_file,
)


SOURCE_PATHS = (
    "AGENTS.md",
    "src/build_toric_code_examples.py",
    "data/expander_code/exp101/AGENTS.md",
    "data/expander_code/exp101/PHYSICS_CONTRACT.md",
    "data/expander_code/exp101/src",
    "data/expander_code/exp101/tests",
    "data/expander_code/exp102/exp102_pipeline",
    "data/expander_code/exp102/config/production.v1.json",
    "data/expander_code/exp102/registry",
    "data/expander_code/exp102/tests/conftest.py",
    "data/expander_code/exp102/tests/test_core.py",
    "data/expander_code/exp102/tests/test_scan_results_strict.py",
    "data/expander_code/exp102/tests/test_source_identity.py",
    "data/expander_code/exp102/validation/002_numba_smoke_20260719/orchestrate_ladder.py",
    "data/expander_code/exp102/validation/002_numba_smoke_20260719/run_stage_wrapper.sh",
    "data/expander_code/exp102/validation/002_numba_smoke_20260719/run_verified_source.sh",
    "data/expander_code/exp103/EXPERIMENT_CONTRACT.md",
    "data/expander_code/exp103/REMOTE_EXECUTION_AMENDMENT.md",
    "data/expander_code/exp103/REMOTE_EXECUTION_AMENDMENT_V2.md",
    "data/expander_code/exp103/config",
    "data/expander_code/exp103/deployment/build_remote_deployment.py",
    "data/expander_code/exp103/deployment/bootstrap_verified_archive.sh",
    "data/expander_code/exp103/deployment/run_remote_stage.sh",
    "data/expander_code/exp103/deployment/run_verified_source.sh",
    "data/expander_code/exp103/exp103_pipeline",
    "data/expander_code/exp103/final_results/README.md",
    "data/expander_code/exp103/raw/README.md",
    "data/expander_code/exp103/status.md",
    "data/expander_code/exp103/tests",
    "data/expander_code/exp103/validation",
)

FROZEN_EXECUTION_PATHS = (
    "src/build_toric_code_examples.py",
    "data/expander_code/exp101/PHYSICS_CONTRACT.md",
    "data/expander_code/exp101/src",
    "data/expander_code/exp101/tests/test_gf2.py",
    "data/expander_code/exp101/tests/test_hgp.py",
    "data/expander_code/exp101/tests/test_logicals.py",
    "data/expander_code/exp102/config/production.v1.json",
    "data/expander_code/exp102/exp102_pipeline",
    "data/expander_code/exp102/registry",
    "data/expander_code/exp102/tests/conftest.py",
    "data/expander_code/exp102/tests/test_core.py",
    "data/expander_code/exp102/tests/test_scan_results_strict.py",
    "data/expander_code/exp102/tests/test_source_identity.py",
    "data/expander_code/exp102/validation/002_numba_smoke_20260719/orchestrate_ladder.py",
    "data/expander_code/exp102/validation/002_numba_smoke_20260719/run_stage_wrapper.sh",
    "data/expander_code/exp102/validation/002_numba_smoke_20260719/run_verified_source.sh",
    "data/expander_code/exp103/EXPERIMENT_CONTRACT.md",
    "data/expander_code/exp103/REMOTE_EXECUTION_AMENDMENT.md",
    "data/expander_code/exp103/REMOTE_EXECUTION_AMENDMENT_V2.md",
    "data/expander_code/exp103/deployment",
    "data/expander_code/exp103/exp103_pipeline",
    "data/expander_code/exp103/tests",
)


def _git(root, *args, check=True):
    return subprocess.run(
        ("git", "-C", str(root), *args), check=check,
        capture_output=True, text=True,
    )


def _archive_files(archive_path):
    files = []
    with tarfile.open(archive_path, "r:") as archive:
        seen = set()
        for member in archive.getmembers():
            if member.isdir():
                continue
            relative = Path(member.name)
            normalized = relative.as_posix()
            if (
                not member.isfile()
                or relative.is_absolute()
                or ".." in relative.parts
                or normalized in seen
            ):
                raise ValueError("source archive contains an unsafe or duplicate member")
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError("source archive member cannot be read")
            files.append({
                "path": normalized,
                "sha256": hashlib.sha256(handle.read()).hexdigest(),
            })
            seen.add(normalized)
    if not files:
        raise ValueError("source archive is empty")
    return sorted(files, key=lambda item: item["path"])


def build_remote_deployment(repo_root, output_dir, commit, config_path):
    root = Path(repo_root).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"deployment evidence is immutable: {output}")
    if re.fullmatch(r"[0-9a-f]{40}", str(commit)) is None:
        raise ValueError("deployment commit must be a full lowercase Git SHA")
    head = _git(root, "rev-parse", "HEAD").stdout.strip()
    if head != commit:
        raise ValueError("deployment must be built from the checked-out HEAD")
    if _git(root, "status", "--porcelain", "--untracked-files=all").stdout:
        raise ValueError("deployment must be built from a completely clean worktree")
    upstream = _git(root, "rev-parse", "@{upstream}").stdout.strip()
    if _git(root, "merge-base", "--is-ancestor", commit, upstream, check=False).returncode:
        raise ValueError("deployment commit is not present in the configured upstream")

    config = load_config(config_path)
    if config["schema_version"] != REMOTE_CONFIG_SCHEMA:
        raise ValueError("remote deployment requires the canonical remote config")
    package_path = root / "data/expander_code/exp103/exp103_pipeline"
    if source_tree_sha256(package_path) != config["source_tree_sha256"]:
        raise ValueError("checked-out exp103 package differs from the frozen source tree")
    source_diff = _git(
        root, "diff", "--quiet", config["source_commit"], commit, "--",
        *FROZEN_EXECUTION_PATHS, check=False,
    )
    if source_diff.returncode:
        raise ValueError("frozen remote execution dependencies changed after source freeze")

    output.mkdir(parents=True)
    archive_path = output / "SOURCE.tar"
    subprocess.run(
        (
            "git", "-C", str(root), "archive", "--format=tar",
            f"--output={archive_path}", commit, "--", *SOURCE_PATHS,
        ),
        check=True,
    )
    archive_sha256 = sha256_file(archive_path)
    files = _archive_files(archive_path)
    source_manifest = {
        "source_identity_version": "exp102.source.v1",
        "source_commit": commit,
        "archive_sha256": archive_sha256,
        "files": files,
    }
    source_manifest_path = output / "SOURCE_MANIFEST.json"
    atomic_json(source_manifest_path, source_manifest)
    source_manifest_sha256 = sha256_file(source_manifest_path)
    deployment_manifest = {
        "schema_version": REMOTE_DEPLOYMENT_SCHEMA,
        "experiment_id": config["experiment_id"],
        "execution_profile_id": REMOTE_EXECUTION_PROFILE,
        "source_commit": commit,
        "frozen_source_commit": config["source_commit"],
        "source_tree_sha256": config["source_tree_sha256"],
        "config_sha256": config["config_sha256"],
        "registry_sha256": config["registry_sha256"],
        "archive_sha256": archive_sha256,
        "source_manifest_sha256": source_manifest_sha256,
    }
    deployment_manifest_path = output / "DEPLOYMENT_MANIFEST.json"
    atomic_json(deployment_manifest_path, deployment_manifest)
    (output / "SOURCE_COMMIT").write_text(commit + "\n", encoding="ascii")
    (output / "ARCHIVE_SHA256").write_text(archive_sha256 + "\n", encoding="ascii")
    deployment_sha256 = sha256_file(deployment_manifest_path)
    (output / "DEPLOYMENT_MANIFEST_SHA256").write_text(
        deployment_sha256 + "\n", encoding="ascii",
    )
    return {
        "deployment_manifest_sha256": deployment_sha256,
        "archive_sha256": archive_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "source_files": len(files),
        "source_commit": commit,
        "config_sha256": config["config_sha256"],
        "manifest": json.loads(canonical_json(deployment_manifest)),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    result = build_remote_deployment(
        args.repo_root, args.output, args.commit, args.config,
    )
    print(canonical_json(result))


if __name__ == "__main__":
    main()
