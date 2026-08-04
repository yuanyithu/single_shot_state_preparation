import hashlib
import importlib
import importlib.metadata
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy

from .config import ensure_config
from .io import sha256_file


DEVICE_BY_HOSTNAME = {"ymini.local": "macmini"}


def source_manifest(package_dir=None):
    package_dir = Path(package_dir or Path(__file__).resolve().parent)
    files = sorted(path for path in package_dir.glob("*.py") if path.is_file())
    return [{"path": path.name, "sha256": sha256_file(path)} for path in files]


def source_tree_sha256(package_dir=None):
    digest = hashlib.sha256()
    for item in source_manifest(package_dir):
        digest.update(item["path"].encode("ascii") + b"\0")
        digest.update(item["sha256"].encode("ascii") + b"\n")
    return digest.hexdigest()


def bplsd_binary_path():
    module = importlib.import_module("ldpc.bplsd_decoder._bplsd_decoder")
    return Path(module.__file__).resolve()


def runtime_identity(config, verify_source=False, repo_root=None):
    config = ensure_config(config)
    hostname = socket.gethostname()
    conda_environment = os.environ.get("CONDA_DEFAULT_ENV", "")
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    conda_prefix_matches_python = bool(conda_prefix) and (
        Path(conda_prefix).resolve() == Path(sys.prefix).resolve()
        and Path(conda_prefix).name == conda_environment
    )
    actual = {
        "device_name": DEVICE_BY_HOSTNAME.get(hostname, ""),
        "hostname": hostname,
        "conda_environment": conda_environment,
        "conda_prefix_matches_python": conda_prefix_matches_python,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "ldpc_version": importlib.metadata.version("ldpc"),
        "bplsd_binary_sha256": sha256_file(bplsd_binary_path()),
        "source_tree_sha256": source_tree_sha256(),
        "source_commit": config["source_commit"],
    }
    expected_env = config["environment"]
    for key, expected in (
        ("device_name", expected_env["device_name"]),
        ("hostname", expected_env["hostname"]),
        ("conda_environment", expected_env["conda_environment"]),
        ("conda_prefix_matches_python", expected_env["conda_prefix_matches_python"]),
        ("python_version", expected_env["python"]),
        ("numpy_version", expected_env["numpy"]),
        ("scipy_version", expected_env["scipy"]),
        ("ldpc_version", expected_env["ldpc"]),
        ("bplsd_binary_sha256", config["bplsd_binary"]["sha256"]),
        ("source_tree_sha256", config["source_tree_sha256"]),
    ):
        if actual[key] != expected:
            raise ValueError(f"runtime identity mismatch for {key}")
    if not bplsd_binary_path().name.endswith(config["bplsd_binary"]["filename_suffix"]):
        raise ValueError("runtime identity mismatch for BpLSD binary filename")
    if verify_source:
        root = Path(repo_root or Path(__file__).resolve().parents[4]).resolve()
        source_rel = Path(__file__).resolve().parent.relative_to(root)
        commit = config["source_commit"]
        subprocess.run(
            ("git", "-C", str(root), "cat-file", "-e", f"{commit}^{{commit}}"), check=True,
            capture_output=True,
        )
        diff = subprocess.run(
            ("git", "-C", str(root), "diff", "--quiet", commit, "HEAD", "--", str(source_rel)),
        )
        dirty = subprocess.run(
            ("git", "-C", str(root), "status", "--porcelain", "--", str(source_rel)),
            check=True, capture_output=True, text=True,
        ).stdout
        if diff.returncode != 0 or dirty:
            raise ValueError("exp103 source differs from the frozen source commit")
    return actual


def require_canonical_python():
    if sys.version_info[:3] != (3, 12, 12):
        raise ValueError("exp103 formal statistics require Python 3.12.12")


def _require_file_matches_commit(root, path, commit):
    """Require the checked-out bytes to equal the bytes frozen at commit."""
    root = Path(root).resolve()
    path = Path(path).resolve()
    relative = path.relative_to(root).as_posix()
    committed = subprocess.run(
        ("git", "-C", str(root), "show", f"{commit}:{relative}"),
        check=True, capture_output=True,
    ).stdout
    if path.read_bytes() != committed:
        raise ValueError(f"formal exp103 artifact differs from source commit: {relative}")


def verify_frozen_repository(config_path, repo_root=None):
    """Require the formal contract/config/source to be clean and pushed."""
    root = Path(repo_root or Path(__file__).resolve().parents[4]).resolve()
    config_path = Path(config_path).resolve()
    config = ensure_config(config_path)
    contract_path = Path(__file__).resolve().parents[1] / "EXPERIMENT_CONTRACT.md"
    scoped = [
        Path(__file__).resolve().parent.relative_to(root),
        config_path.relative_to(root),
        contract_path.relative_to(root),
    ]
    for path in scoped:
        subprocess.run(
            ("git", "-C", str(root), "ls-files", "--error-unmatch", str(path)),
            check=True, capture_output=True,
        )
    dirty = subprocess.run(
        ("git", "-C", str(root), "status", "--porcelain", "--", *(str(path) for path in scoped)),
        check=True, capture_output=True, text=True,
    ).stdout
    if dirty:
        raise ValueError("formal exp103 contract/config/source has uncommitted changes")
    subprocess.run(
        ("git", "-C", str(root), "cat-file", "-e", f"{config['source_commit']}^{{commit}}"),
        check=True, capture_output=True,
    )
    _require_file_matches_commit(root, contract_path, config["source_commit"])
    head = subprocess.run(
        ("git", "-C", str(root), "rev-parse", "HEAD"), check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    upstream = subprocess.run(
        ("git", "-C", str(root), "rev-parse", "@{upstream}"), check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    pushed = subprocess.run(
        ("git", "-C", str(root), "merge-base", "--is-ancestor", head, upstream),
    )
    if pushed.returncode != 0:
        raise ValueError("formal exp103 HEAD is not present in the configured upstream")
    return {"head": head, "upstream": upstream}


def require_tracked_clean_evidence(path, repo_root=None):
    root = Path(repo_root or Path(__file__).resolve().parents[4]).resolve()
    path = Path(path).resolve()
    relative = path.relative_to(root)
    subprocess.run(
        ("git", "-C", str(root), "ls-files", "--error-unmatch", str(relative)),
        check=True, capture_output=True,
    )
    dirty = subprocess.run(
        ("git", "-C", str(root), "status", "--porcelain", "--", str(relative)),
        check=True, capture_output=True, text=True,
    ).stdout
    if dirty:
        raise ValueError(f"evidence is not clean in Git: {relative}")
    return relative
