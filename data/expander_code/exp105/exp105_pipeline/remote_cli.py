"""Single-stage remote execution on nd-3, with the same gates exp103 used.

exp105 has no stage split: there is no preliminary curve to authorize a second
stage on, because the whole point is one fair draw from one ensemble. The gates
that remain are the ones that caught real defects: archive identity, a
bytecode-clean source tree, a decoder-determinism regression run in the remote
interpreter, a resource projection compared against frozen caps, no unplanned
raw evidence, and a fail-closed replay of the committed subsample.
"""

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .aggregate import ARRAY_FIELDS, SCALAR_FIELDS, aggregate_scan
from .config import (
    REMOTE_CONFIG_SCHEMA,
    tasks_per_m,
    load_config,
)
from .ensemble import load_registry, registry_index
from .identity import (
    decoder_binary_path,
    runtime_identity,
    verify_remote_deployment,
)
from .io import atomic_json, sha256_file, sha256_json
from .preflight import benchmark_task, estimate_resources
from .raw import load_raw, raw_filename, save_raw
from .replay import (
    build_replay_report,
    committed_replay_blocks,
    expected_replay_keys,
    replay_task,
    validate_replay_against_raw,
)
from .report import write_report
from .worker import run_code_block


REMOTE_PREFLIGHT_SCHEMA = "exp105.remote_resource_preflight.v1"
REMOTE_QUALIFICATION_SCHEMA = "exp105.remote_environment_qualification.v1"
REMOTE_SCAN_SCHEMA = "exp105.remote_scan.v1"
REMOTE_QUALIFICATION_RELATIVE = Path("validation/environment_qualification.json")
REMOTE_PREFLIGHT_RELATIVE = Path("validation/remote_resource_preflight.json")
COMMITTED_QUALIFICATION_RELATIVE = Path(
    "data/expander_code/exp105/validation/004_remote_gate_20260811/"
    "environment_qualification.json"
)
COMMITTED_PREFLIGHT_RELATIVE = Path(
    "data/expander_code/exp105/validation/004_remote_gate_20260811/"
    "remote_resource_preflight.json"
)
QUALIFICATION_GROUPS = (
    ("exp105", ("data/expander_code/exp105/tests",)),
    # The exp101 and exp102 groups are the same certified subsets exp103
    # qualified against, so a change in either shows up as a count mismatch.
    ("exp101", (
        "data/expander_code/exp101/tests/test_gf2.py",
        "data/expander_code/exp101/tests/test_hgp.py",
        "data/expander_code/exp101/tests/test_logicals.py",
    )),
    ("exp104", ("data/expander_code/exp104/tests",)),
    ("exp102", (
        "data/expander_code/exp102/tests/test_core.py",
        "data/expander_code/exp102/tests/test_scan_results_strict.py",
        "data/expander_code/exp102/tests/test_source_identity.py",
    )),
)
# exp104 is included because exp105 reuses its ensemble rule, its decoder
# identity and its comparison codes; a change there would silently change
# what exp105 means. The exp101 and exp102 groups are the same certified
# subsets exp103 and exp104 qualified against.
QUALIFICATION_EXPECTED_PASSES = {
    "exp105": 152, "exp104": 131, "exp101": 58, "exp102": 17,
}


def _require_remote_config(config):
    if config["schema_version"] != REMOTE_CONFIG_SCHEMA:
        raise ValueError("this command requires the frozen remote config")
    return config


def resolve_remote_run_root(run_root, config):
    """Confine every run to one directory under the frozen run root.

    A bare name is placed under the frozen root; anything with a path is taken
    literally and refused unless it already sits directly there. Silently
    rewriting an out-of-tree path would let a typo write evidence somewhere the
    contract does not cover.
    """
    config = _require_remote_config(config)
    allowed = Path(config["execution_profile"]["run_root"]).expanduser().resolve()
    given = Path(run_root).expanduser()
    resolved = (allowed / given.name) if len(given.parts) == 1 else given.resolve()
    if resolved.parent != allowed:
        raise ValueError("remote run root must sit directly under the frozen run root")
    return resolved


def _raw_root(run_root):
    return Path(run_root) / "raw"


def _qualification_path(run_root):
    return Path(run_root) / REMOTE_QUALIFICATION_RELATIVE


def _preflight_path(run_root):
    return Path(run_root) / REMOTE_PREFLIGHT_RELATIVE


def _scan_report_path(run_root):
    return Path(run_root) / "validation" / "scan.json"


def _replay_path(run_root):
    return Path(run_root) / "validation" / "replay.json"


def _aggregate_path(run_root):
    return Path(run_root) / "aggregate" / "ensemble_crossing.npz"


@contextlib.contextmanager
def _stage_lock(run_root, name):
    lock_path = Path(run_root) / "locks" / f"{name}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"remote stage is already running: {name}") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _require_single_thread_environment():
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        if os.environ.get(name) != "1":
            raise ValueError(f"remote formal execution requires {name}=1")


def _verify_remote_runtime(config, deployment_root, deployment_manifest_sha256):
    config = _require_remote_config(config)
    _require_single_thread_environment()
    identity = runtime_identity(config)
    deployment = verify_remote_deployment(
        config, deployment_root, deployment_manifest_sha256,
    )
    return identity, deployment


def _require_deployed_evidence_bytes(run_path, deployment_root, relative, purpose):
    run_path = Path(run_path).resolve()
    source_root = Path(deployment_root).expanduser().resolve() / "source"
    deployed = (source_root / relative).resolve()
    if source_root not in deployed.parents or not run_path.is_file() or not deployed.is_file():
        raise ValueError(f"{purpose} is not present in the deployed pushed source")
    if run_path.read_bytes() != deployed.read_bytes():
        raise ValueError(f"{purpose} differs from the deployed pushed evidence")
    return sha256_file(deployed)


def _bytecode_clean(source_root):
    source_root = Path(source_root)
    return not any(
        path.name == "__pycache__"
        or (path.is_file() and path.suffix in {".pyc", ".nbi", ".nbc"})
        for path in source_root.rglob("*")
    )


def _ldpc_source_provenance(config):
    expected = dict(config["ldpc_source"])
    archive_path = Path(expected["archive_path"])
    if not archive_path.is_file() or sha256_file(archive_path) != expected["archive_sha256"]:
        raise ValueError("frozen ldpc source archive is missing or has drifted")
    prefix = f"ldpc-{expected['commit']}"
    with tarfile.open(archive_path, "r:gz") as archive:
        rng_handle = archive.extractfile(f"{prefix}/src_cpp/rng.hpp")
        project_handle = archive.extractfile(f"{prefix}/pyproject.toml")
        if rng_handle is None or project_handle is None:
            raise ValueError("frozen ldpc source archive is incomplete")
        rng_sha256 = hashlib.sha256(rng_handle.read()).hexdigest()
        project = project_handle.read().decode("utf-8")
    if rng_sha256 != expected["rng_hpp_sha256"]:
        raise ValueError("frozen ldpc rng.hpp identity mismatch")
    if re.search(r'^version\s*=\s*"2\.4\.1"\s*$', project, re.MULTILINE) is None:
        raise ValueError("frozen ldpc source does not declare version 2.4.1")
    return {**expected, "project_version": "2.4.1"}


def _host_resources(config):
    logical = int(os.cpu_count() or 0)
    sockets = set()
    physical_cores = set()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        record = {}
        for line in cpuinfo.read_text(encoding="ascii", errors="replace").splitlines() + [""]:
            if not line:
                if "physical id" in record and "core id" in record:
                    sockets.add(record["physical id"])
                    physical_cores.add((record["physical id"], record["core id"]))
                record = {}
            elif ":" in line:
                key, value = line.split(":", 1)
                record[key.strip()] = value.strip()
    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    memory_total = int(os.sysconf("SC_PHYS_PAGES")) * page_size
    available_pages = os.sysconf_names.get("SC_AVPHYS_PAGES")
    memory_available = (
        int(os.sysconf(available_pages)) * page_size
        if available_pages is not None else 0
    )
    run_disk_path = Path(config["execution_profile"]["run_root"]).expanduser()
    disk = shutil.disk_usage(run_disk_path)
    libc_name, libc_version = platform.libc_ver()
    return {
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
        "libc_name": libc_name,
        "libc_version": libc_version,
        "logical_cpu_count": logical,
        "physical_core_count": len(physical_cores),
        "cpu_socket_count": len(sockets),
        "memory_total_bytes": memory_total,
        "memory_available_bytes": memory_available,
        "run_disk_path": str(run_disk_path.resolve()),
        "run_disk_total_bytes": int(disk.total),
        "run_disk_free_bytes": int(disk.free),
    }


def _pytest_outcome_count(output, label):
    matches = re.findall(rb"(?:^|\s)(\d+) " + label.encode("ascii") + rb"(?:\s|,|$)", output)
    return int(matches[-1]) if matches else 0


def run_environment_qualification(config, source_root):
    config = _require_remote_config(config)
    source_root = Path(source_root).resolve()
    clean_before = _bytecode_clean(source_root)
    environment = os.environ.copy()
    environment.update({
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_ADDOPTS": "-p no:cacheprovider",
        "PYTHONPATH": str(source_root),
        "EXP105_TEST_CONFIG_PATH": str(
            source_root / "data/expander_code/exp105/config/ensemble_mc.remote.v1.json"
        ),
    })
    environment.pop("PYTHONOPTIMIZE", None)
    results = []
    for name, paths in QUALIFICATION_GROUPS:
        argv = (str(Path(sys.executable).resolve()), "-B", "-m", "pytest", "-q", *paths)
        completed = subprocess.run(
            argv, cwd=source_root, env=environment, capture_output=True,
        )
        stdout = completed.stdout
        stderr = completed.stderr
        output = stdout + b"\n" + stderr
        passed = _pytest_outcome_count(output, "passed")
        nonpass = {
            label: _pytest_outcome_count(output, label)
            for label in ("skipped", "xfailed", "xpassed", "deselected")
        }
        expected_passed = QUALIFICATION_EXPECTED_PASSES[name]
        status = "PASS" if (
            completed.returncode == 0
            and passed == expected_passed
            and not any(nonpass.values())
        ) else "FAIL"
        results.append({
            "name": name,
            "argv": list(argv),
            "exit_code": int(completed.returncode),
            "status": status,
            "passed_count": passed,
            "expected_passed_count": expected_passed,
            "skipped_count": nonpass["skipped"],
            "xfailed_count": nonpass["xfailed"],
            "xpassed_count": nonpass["xpassed"],
            "deselected_count": nonpass["deselected"],
            "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
            "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
        })
    clean_after = _bytecode_clean(source_root)
    passed = clean_before and clean_after and all(
        result["status"] == "PASS" for result in results
    )
    binary_path = decoder_binary_path()
    if Path(sys.prefix).resolve() not in binary_path.parents:
        raise ValueError("decoder binary is outside the frozen Python prefix")
    identity = runtime_identity(config)
    return {
        "schema_version": REMOTE_QUALIFICATION_SCHEMA,
        "status": "PASS" if passed else "FAIL",
        "experiment_id": config["experiment_id"],
        "execution_profile_id": config["execution_profile"]["profile_id"],
        "config_sha256": config["config_sha256"],
        "registry_sha256": config["registry_sha256"],
        "source_commit": config["source_commit"],
        "source_tree_sha256": config["source_tree_sha256"],
        "decoder_binary_sha256": config["decoder_binary"]["sha256"],
        "decoder_binary_path_suffix": binary_path.name,
        "device_name": identity["device_name"],
        "hostname": identity["hostname"],
        "conda_environment": identity["conda_environment"],
        "conda_prefix_matches_python": identity["conda_prefix_matches_python"],
        "python_executable": str(Path(sys.executable).resolve()),
        "bytecode_clean_before": clean_before,
        "bytecode_clean_after": clean_after,
        "ldpc_source": _ldpc_source_provenance(config),
        "support_packages": identity["support_packages"],
        "host": _host_resources(config),
        "groups": results,
    }


def validate_environment_qualification(report, config):
    config = _require_remote_config(config)
    if report["schema_version"] != REMOTE_QUALIFICATION_SCHEMA:
        raise ValueError("qualification schema mismatch")
    for field, expected in (
        ("status", "PASS"),
        ("experiment_id", config["experiment_id"]),
        ("execution_profile_id", config["execution_profile"]["profile_id"]),
        ("config_sha256", config["config_sha256"]),
        ("registry_sha256", config["registry_sha256"]),
        ("source_commit", config["source_commit"]),
        ("source_tree_sha256", config["source_tree_sha256"]),
        ("decoder_binary_sha256", config["decoder_binary"]["sha256"]),
        ("device_name", config["environment"]["device_name"]),
        ("hostname", config["environment"]["hostname"]),
        ("conda_environment", config["environment"]["conda_environment"]),
        ("conda_prefix_matches_python", True),
        ("bytecode_clean_before", True),
        ("bytecode_clean_after", True),
    ):
        if report[field] != expected:
            raise ValueError(f"qualification identity mismatch for {field}")
    if report["ldpc_source"]["archive_sha256"] != config["ldpc_source"]["archive_sha256"]:
        raise ValueError("qualification ldpc provenance mismatch")
    if report["support_packages"] != config["support_packages"]:
        raise ValueError("qualification support package mismatch")
    names = {group["name"] for group in report["groups"]}
    if names != {name for name, _ in QUALIFICATION_GROUPS}:
        raise ValueError("qualification group set mismatch")
    for group in report["groups"]:
        if group["status"] != "PASS" or group["exit_code"] != 0:
            raise ValueError(f"qualification group failed: {group['name']}")
        if group["passed_count"] != QUALIFICATION_EXPECTED_PASSES[group["name"]]:
            raise ValueError(f"qualification pass count mismatch: {group['name']}")
        for label in ("skipped_count", "xfailed_count", "xpassed_count", "deselected_count"):
            if group[label]:
                raise ValueError(f"qualification group is not fully executed: {group['name']}")
    host = report["host"]
    profile = config["execution_profile"]
    if host["logical_cpu_count"] < profile["num_workers"]:
        raise ValueError("remote host has fewer logical CPUs than the frozen worker count")
    return report


def run_remote_resource_preflight(config, registry_rows, qualification_sha256):
    config = _require_remote_config(config)
    spec = config["preflight"]
    profile = config["execution_profile"]
    tasks = [
        benchmark_task(m, code_index, token, config, registry_rows)
        for m in spec["m_values"]
        for code_index in spec["code_indices"]
        for token in spec["p_tokens"]
    ]
    estimate = estimate_resources(tasks, config, profile["num_workers"], profile)
    blocks = committed_replay_blocks(config)
    return {
        "schema_version": REMOTE_PREFLIGHT_SCHEMA,
        "status": estimate["status"],
        "experiment_id": config["experiment_id"],
        "execution_profile_id": profile["profile_id"],
        "config_sha256": config["config_sha256"],
        "registry_sha256": config["registry_sha256"],
        "source_commit": config["source_commit"],
        "source_tree_sha256": config["source_tree_sha256"],
        "decoder_binary_sha256": config["decoder_binary"]["sha256"],
        "device_name": config["environment"]["device_name"],
        "hostname": config["environment"]["hostname"],
        "conda_environment": config["environment"]["conda_environment"],
        "num_workers": profile["num_workers"],
        "omp_thread_count": profile["omp_thread_count"],
        "qualification_report_sha256": qualification_sha256,
        "outcome_blind": True,
        "caps": {
            "reserve_multiplier": profile["reserve_multiplier"],
            "stage_core_hour_cap": profile["stage_core_hour_cap"],
            "stage_wall_hour_cap": profile["stage_wall_hour_cap"],
            "peak_rss_gib_cap": profile["peak_rss_gib_cap"],
        },
        # The replay subsample is fixed here, before any production task runs.
        "committed_replay_blocks": {str(m): blocks[m] for m in config["m_values"]},
        "estimate": estimate,
        "tasks": tasks,
    }


def validate_remote_resource_preflight(report, config):
    config = _require_remote_config(config)
    profile = config["execution_profile"]
    if report["schema_version"] != REMOTE_PREFLIGHT_SCHEMA:
        raise ValueError("remote preflight schema mismatch")
    for field, expected in (
        ("status", "PASS"),
        ("experiment_id", config["experiment_id"]),
        ("execution_profile_id", profile["profile_id"]),
        ("config_sha256", config["config_sha256"]),
        ("registry_sha256", config["registry_sha256"]),
        ("source_commit", config["source_commit"]),
        ("source_tree_sha256", config["source_tree_sha256"]),
        ("decoder_binary_sha256", config["decoder_binary"]["sha256"]),
        ("hostname", config["environment"]["hostname"]),
        ("num_workers", profile["num_workers"]),
        ("outcome_blind", True),
    ):
        if report[field] != expected:
            raise ValueError(f"remote preflight identity mismatch for {field}")
    blocks = committed_replay_blocks(config)
    if report["committed_replay_blocks"] != {
        str(m): blocks[m] for m in config["m_values"]
    }:
        raise ValueError("remote preflight replay subsample is not the committed one")
    estimate = report["estimate"]
    if not all(estimate["checks"].values()):
        raise ValueError("remote preflight resource checks did not all pass")
    if estimate["reserved_core_hours"] > profile["stage_core_hour_cap"]:
        raise ValueError("remote preflight reserved core hours exceed the cap")
    if estimate["predicted_wall_hours"] > profile["stage_wall_hour_cap"]:
        raise ValueError("remote preflight predicted wall hours exceed the cap")
    if estimate["projected_peak_rss_gib"] > profile["peak_rss_gib_cap"]:
        raise ValueError("remote preflight projected RSS exceeds the cap")
    return report


_REGISTRY_CACHE = {}


def _cached_rows(config_path):
    key = str(config_path)
    if key not in _REGISTRY_CACHE:
        config = load_config(config_path)
        root = Path(__file__).resolve().parents[4]
        _REGISTRY_CACHE[key] = registry_index(
            load_registry(root / config["registry_path"])
        )
    return _REGISTRY_CACHE[key]


def _run_one_task(payload):
    m, block_index, config_path, raw_root = payload
    config = load_config(config_path)
    rows = _cached_rows(config_path)
    output = Path(raw_root) / raw_filename(config, m, block_index)
    if Path(raw_root).is_symlink() or output.is_symlink():
        raise ValueError("remote raw paths must not be symlinks")
    if output.exists():
        existing = load_raw(output)
        if (
            int(existing["m"]) != m
            or int(existing["block_index"]) != block_index
            or existing["config_sha256"] != config["config_sha256"]
            or existing["status"] != "VALID"
        ):
            raise ValueError(f"existing immutable remote raw does not match its key: {output}")
        return m, block_index, "RESUMED"
    raw = run_code_block(m, block_index, config, rows)
    save_raw(output, raw)
    return m, block_index, raw["status"]


def _planned_tasks(config):
    counts = {int(k): int(v) for k, v in config["codes_per_m"].items()}
    sizes = {int(k): int(v) for k, v in config["codes_per_task"].items()}
    per_m = tasks_per_m(counts, sizes)
    return [
        (int(m), block)
        for m in config["m_values"]
        for block in range(per_m[int(m)])
    ]


def _assert_no_unplanned_npz(raw_root, config, require_complete=False):
    raw_root = Path(raw_root)
    if raw_root.is_symlink():
        raise ValueError("remote raw root must not be a symlink")
    expected = {
        raw_root / raw_filename(config, m, block)
        for m, block in _planned_tasks(config)
    }
    actual = set(raw_root.glob("*.npz")) if raw_root.is_dir() else set()
    if any(path.is_symlink() for path in actual):
        raise ValueError("remote raw evidence must not be a symlink")
    if actual - expected:
        raise ValueError("remote raw root contains unplanned NPZ evidence")
    if require_complete and actual != expected:
        raise ValueError("remote raw evidence is incomplete")
    return expected


def _require_preflight(path, run_root, config, deployment_root):
    path = Path(path)
    report = json.loads(path.read_text(encoding="ascii"))
    validate_remote_resource_preflight(report, config)
    canonical = _preflight_path(run_root)
    if path.resolve() != canonical.resolve():
        raise ValueError("remote preflight must be the immutable run-root artifact")
    _require_deployed_evidence_bytes(
        path, deployment_root, COMMITTED_PREFLIGHT_RELATIVE,
        "remote resource preflight",
    )
    return report


def _require_qualification(run_root, config, deployment_root):
    path = _qualification_path(run_root)
    if not path.is_file():
        raise ValueError("remote run requires the immutable qualification report")
    report = json.loads(path.read_text(encoding="ascii"))
    validate_environment_qualification(report, config)
    _require_deployed_evidence_bytes(
        path, deployment_root, COMMITTED_QUALIFICATION_RELATIVE,
        "remote environment qualification",
    )
    return report


def run_remote_scan(config_path, run_root, num_workers, deployment_root,
                    deployment_manifest_sha256):
    config = load_config(config_path)
    run_root = resolve_remote_run_root(run_root, config)
    if num_workers != config["execution_profile"]["num_workers"]:
        raise ValueError("formal remote scans require the frozen worker count")
    identity, deployment = _verify_remote_runtime(
        config, deployment_root, deployment_manifest_sha256,
    )
    _require_qualification(run_root, config, deployment_root)
    preflight = _require_preflight(
        _preflight_path(run_root), run_root, config, deployment_root,
    )
    output = _scan_report_path(run_root)
    if output.exists():
        raise FileExistsError(f"remote scan report is immutable: {output}")
    raw_root = _raw_root(run_root)
    raw_root.mkdir(parents=True, exist_ok=True)
    _assert_no_unplanned_npz(raw_root, config)
    # Longest tasks first so the tail does not decide the wall time.
    tasks = sorted(
        _planned_tasks(config), key=lambda item: item[0], reverse=True,
    )
    payloads = [(m, block, str(config_path), str(raw_root)) for m, block in tasks]
    statuses = []
    started = time.time()
    with _stage_lock(run_root, "scan"):
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_run_one_task, payload) for payload in payloads]
            for future in as_completed(futures):
                statuses.append(future.result())
    if any(status not in {"VALID", "RESUMED"} for _, _, status in statuses):
        raise RuntimeError("one or more remote formal tasks saved INVALID evidence")
    if len(statuses) != len(payloads):
        raise RuntimeError("remote scan did not produce every planned task")
    _assert_no_unplanned_npz(raw_root, config, require_complete=True)
    result = {
        "schema_version": REMOTE_SCAN_SCHEMA,
        "status": "PASS",
        "config_sha256": config["config_sha256"],
        "registry_sha256": config["registry_sha256"],
        "source_tree_sha256": config["source_tree_sha256"],
        "decoder_binary_sha256": config["decoder_binary"]["sha256"],
        "hostname": identity["hostname"],
        "num_workers": num_workers,
        "planned_tasks": len(payloads),
        "fresh_tasks": sum(status == "VALID" for _, _, status in statuses),
        "resumed_tasks": sum(status == "RESUMED" for _, _, status in statuses),
        "wall_seconds": time.time() - started,
        "preflight_sha256": sha256_json(preflight),
        "deployment_manifest_sha256": deployment["deployment_manifest_sha256"],
    }
    atomic_json(output, result)
    return result


def _replay_one(payload):
    path, config_path = payload
    config = load_config(config_path)
    rows = _cached_rows(config_path)
    return replay_task(path, config, rows)


def run_remote_replay(config_path, run_root, num_workers, deployment_root,
                      deployment_manifest_sha256):
    config = load_config(config_path)
    run_root = resolve_remote_run_root(run_root, config)
    _verify_remote_runtime(config, deployment_root, deployment_manifest_sha256)
    _require_qualification(run_root, config, deployment_root)
    scan_path = _scan_report_path(run_root)
    if not scan_path.is_file():
        raise ValueError("replay requires a completed scan report")
    scan = json.loads(scan_path.read_text(encoding="ascii"))
    if scan["status"] != "PASS" or scan["config_sha256"] != config["config_sha256"]:
        raise ValueError("replay requires a PASS scan for this exact config")
    raw_root = _raw_root(run_root)
    _assert_no_unplanned_npz(raw_root, config, require_complete=True)
    output = _replay_path(run_root)
    if output.exists():
        raise FileExistsError(f"remote replay report is immutable: {output}")
    keys = sorted(expected_replay_keys(config), key=lambda item: (-item[0], item[1]))
    payloads = [
        (str(raw_root / raw_filename(config, m, block)), str(config_path))
        for m, block in keys
    ]
    results = []
    with _stage_lock(run_root, "replay"):
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_replay_one, payload) for payload in payloads]
            for future in as_completed(futures):
                results.append(future.result())
    report = build_replay_report(results, config)
    atomic_json(output, report)
    if report["status"] != "PASS":
        raise RuntimeError("committed replay subsample did not reproduce bit for bit")
    return report


def run_remote_aggregate(config_path, run_root, deployment_root,
                         deployment_manifest_sha256):
    config = load_config(config_path)
    run_root = resolve_remote_run_root(run_root, config)
    _verify_remote_runtime(config, deployment_root, deployment_manifest_sha256)
    raw_root = _raw_root(run_root)
    _assert_no_unplanned_npz(raw_root, config, require_complete=True)
    replay_path = _replay_path(run_root)
    replay_report = json.loads(replay_path.read_text(encoding="ascii"))
    validate_replay_against_raw(replay_report, raw_root, config)
    replay_report = dict(replay_report, report_sha256=sha256_json(replay_report))
    aggregate = aggregate_scan(raw_root, config, replay_report)
    output = _aggregate_path(run_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"remote aggregate is immutable: {output}")
    np.savez_compressed(output, **{
        key: np.asarray(aggregate[key]) for key in ARRAY_FIELDS + SCALAR_FIELDS
    })
    report = write_report(Path(run_root) / "aggregate", aggregate, config)
    return {
        "overall_status": aggregate["overall_status"],
        "terminal_status": aggregate["terminal_status"],
        "aggregate_sha256": sha256_file(output),
        "report": report,
    }


def build_parser():
    parser = argparse.ArgumentParser(description="exp105 remote execution on nd-3")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("qualify", "preflight", "scan", "replay", "aggregate"):
        item = sub.add_parser(name)
        item.add_argument("--config", required=True)
        item.add_argument("--run-root", required=True)
        item.add_argument("--deployment-root", required=True)
        item.add_argument("--deployment-manifest-sha256", required=True)
        if name in {"scan", "replay"}:
            item.add_argument("--num-workers", type=int, required=True)
        item.set_defaults(name=name)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    run_root = resolve_remote_run_root(args.run_root, config)
    run_root.mkdir(parents=True, exist_ok=True)
    source_root = Path(args.deployment_root).expanduser().resolve() / "source"

    if args.name == "qualify":
        verify_remote_deployment(
            config, args.deployment_root, args.deployment_manifest_sha256,
        )
        report = run_environment_qualification(config, source_root)
        path = _qualification_path(run_root)
        if path.exists():
            raise FileExistsError(f"qualification report is immutable: {path}")
        atomic_json(path, report)
        print(report["status"], sha256_file(path))
        return 0 if report["status"] == "PASS" else 1

    if args.name == "preflight":
        _verify_remote_runtime(
            config, args.deployment_root, args.deployment_manifest_sha256,
        )
        qualification = _require_qualification(run_root, config, args.deployment_root)
        rows = _cached_rows(args.config)
        report = run_remote_resource_preflight(
            config, rows, sha256_json(qualification),
        )
        path = _preflight_path(run_root)
        if path.exists():
            raise FileExistsError(f"preflight report is immutable: {path}")
        atomic_json(path, report)
        estimate = report["estimate"]
        print(
            report["status"],
            f"reserved={estimate['reserved_core_hours']:.1f}core-h",
            f"wall={estimate['predicted_wall_hours']:.2f}h",
            f"rss={estimate['projected_peak_rss_gib']:.1f}GiB",
            sha256_file(path),
        )
        return 0 if report["status"] == "PASS" else 1

    if args.name == "scan":
        result = run_remote_scan(
            args.config, run_root, args.num_workers, args.deployment_root,
            args.deployment_manifest_sha256,
        )
        print(result["status"], result["planned_tasks"], f"{result['wall_seconds']:.0f}s")
        return 0

    if args.name == "replay":
        report = run_remote_replay(
            args.config, run_root, args.num_workers, args.deployment_root,
            args.deployment_manifest_sha256,
        )
        print(report["status"], report["tasks"])
        return 0

    result = run_remote_aggregate(
        args.config, run_root, args.deployment_root, args.deployment_manifest_sha256,
    )
    print(result["overall_status"], result["terminal_status"], result["aggregate_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
