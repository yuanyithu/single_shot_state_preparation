"""Fail-closed single-node remote execution for exp103 decoder Monte Carlo."""

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import fcntl
import numpy as np

from .aggregate import (
    ARRAY_FIELDS, SCALAR_FIELDS, _registry, _validate_raw,
    aggregate_decoder_scan, save_aggregate,
)
from .config import REMOTE_CONFIG_SCHEMA, ensure_config, load_config
from .identity import decoder_binary_path, runtime_identity, verify_remote_deployment
from .io import arrays_sha256, atomic_json, canonical_json, sha256_file, sha256_json
from .preflight import _stage_estimate, benchmark_task
from .raw import load_raw, raw_filename, save_raw
from .replay import (
    build_replay_report, replay_decoder_shard, validate_replay_report,
)
from .report import (
    FINAL_REPORT_FILENAMES, STAGE1_REPORT_FILENAMES,
    generate_final_report, generate_stage1_preliminary_report,
)
from .worker import run_decoder_shard


RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
REMOTE_PREFLIGHT_SCHEMA = "exp103.remote_resource_preflight.v1"
REMOTE_QUALIFICATION_SCHEMA = "exp103.remote_environment_qualification.v1"
REMOTE_SCAN_SCHEMA = "exp103.remote_scan.v1"
REMOTE_TECHNICAL_SCHEMA = "exp103.remote_stage1_technical.v1"
REMOTE_QUALIFICATION_RELATIVE = Path("validation/environment_qualification.json")
REMOTE_PREFLIGHT_RELATIVE = Path("validation/remote_resource_preflight.json")
REMOTE_TECHNICAL_RELATIVE = Path("validation/stage1_technical_report.json")
COMMITTED_PREFLIGHT_RELATIVE = Path(
    "data/expander_code/exp103/validation/"
    "007_remote_gate_v3_20260806/remote_resource_preflight.json"
)
COMMITTED_QUALIFICATION_RELATIVE = Path(
    "data/expander_code/exp103/validation/"
    "007_remote_gate_v3_20260806/environment_qualification.json"
)
COMMITTED_TECHNICAL_RELATIVE = Path(
    "data/expander_code/exp103/validation/"
    "008_remote_m3_m5_scan_20260806/technical_report.json"
)
QUALIFICATION_GROUPS = (
    ("exp103", ("data/expander_code/exp103/tests",)),
    ("exp101", (
        "data/expander_code/exp101/tests/test_gf2.py",
        "data/expander_code/exp101/tests/test_hgp.py",
        "data/expander_code/exp101/tests/test_logicals.py",
    )),
    ("exp102", (
        "data/expander_code/exp102/tests/test_core.py",
        "data/expander_code/exp102/tests/test_scan_results_strict.py",
        "data/expander_code/exp102/tests/test_source_identity.py",
    )),
)
QUALIFICATION_EXPECTED_PASSES = {
    "exp103": 131,
    "exp101": 58,
    "exp102": 17,
}


def _require_remote_config(config):
    config = ensure_config(config)
    if config["schema_version"] != REMOTE_CONFIG_SCHEMA:
        raise ValueError("remote execution requires exp103.config.remote.v3")
    return config


def resolve_remote_run_root(run_root, config):
    """Resolve one direct, non-symlinked child of the frozen remote run base."""
    config = _require_remote_config(config)
    base = Path(config["execution_profile"]["run_root"]).expanduser().resolve()
    requested = Path(run_root).expanduser()
    if not requested.is_absolute():
        raise ValueError("remote run root must be an absolute path")
    resolved = requested.resolve()
    if resolved == base or resolved.parent != base:
        raise ValueError("remote run root is outside the frozen run-root base")
    if RUN_ID_PATTERN.fullmatch(resolved.name) is None:
        raise ValueError("remote run ID contains unsafe characters")
    if requested.exists():
        if requested.is_symlink() or not requested.is_dir():
            raise ValueError("remote run root must be a real directory")
        if any(path.is_symlink() for path in requested.rglob("*")):
            raise ValueError("remote run root must not contain symlinks")
    return resolved


def _raw_root(run_root):
    return Path(run_root) / "raw"


def _stage_root(run_root, stage):
    return _raw_root(run_root) / stage


def _preflight_path(run_root):
    return Path(run_root) / REMOTE_PREFLIGHT_RELATIVE


def _qualification_path(run_root):
    return Path(run_root) / REMOTE_QUALIFICATION_RELATIVE


def _scan_report_path(run_root, stage):
    return Path(run_root) / "control" / f"SCAN_{stage.upper()}.json"


def _replay_path(run_root, stage):
    return _stage_root(run_root, stage) / f"REPLAY_{stage.upper()}.json"


def _aggregate_path(run_root, scope):
    filename = "stage1_aggregate.npz" if scope == "stage1" else "decoder_crossing.npz"
    return Path(run_root) / "final_results" / filename


def _require_scope_reportable(aggregate, scope):
    """Reject a saved aggregate that cannot serve its formal stage."""
    unexpected = json.loads(aggregate["unexpected_raw_errors_json"])
    if scope == "stage1":
        valid = (
            aggregate["overall_status"] == "INCOMPLETE"
            and aggregate["terminal_status"] == "EXP103_INCOMPLETE"
            and aggregate["replay_status"] == "PASS"
            and aggregate["replay_scope"] == "stage1"
            and np.all(aggregate["code_status"][:24] == "REPORTABLE")
            and np.all(aggregate["code_status"][24:] == "INCOMPLETE")
            and np.all(aggregate["m_status"][:3] == "REPORTABLE")
            and np.all(aggregate["m_status"][3:] == "INCOMPLETE")
            and not unexpected
        )
    elif scope == "final":
        valid = (
            aggregate["overall_status"] == "COMPLETE"
            and aggregate["replay_status"] == "PASS"
            and aggregate["replay_scope"] == "final_combined"
            and np.all(aggregate["code_status"] == "REPORTABLE")
            and np.all(aggregate["m_status"] == "REPORTABLE")
            and aggregate["terminal_status"] in {
                "EXP103_DECODER_CROSSING_RESOLVED",
                "EXP103_PAIRWISE_BRACKET_ONLY",
                "EXP103_NO_CORRECT_CROSSING_IN_WINDOW",
                "EXP103_DECODER_CROSSING_INCONCLUSIVE",
            }
            and not unexpected
        )
    else:
        raise ValueError("aggregate scope must be stage1 or final")
    if not valid:
        raise ValueError(f"remote {scope} aggregate is not formally reportable")
    return aggregate


def _arrays_equal(left, right):
    if left.dtype.kind in "fc":
        return np.array_equal(left, right, equal_nan=True)
    return np.array_equal(left, right)


def _scalars_equal(left, right):
    if isinstance(left, (float, np.floating)) and isinstance(
        right, (float, np.floating)
    ):
        return bool((np.isnan(left) and np.isnan(right)) or left == right)
    return left == right


def _require_live_aggregate_matches(path, raw_root, config, scope):
    """Rebuild from live raw/replay bytes before trusting a saved aggregate."""
    stored = _load_aggregate(path)
    live = aggregate_decoder_scan(raw_root, config)
    _require_scope_reportable(stored, scope)
    _require_scope_reportable(live, scope)
    for field in ARRAY_FIELDS:
        if not _arrays_equal(stored[field], live[field]):
            raise ValueError(f"saved aggregate is stale for live field {field}")
    for field in SCALAR_FIELDS:
        if not _scalars_equal(stored[field], live[field]):
            raise ValueError(f"saved aggregate is stale for live field {field}")
    return stored


def _require_hashed_report(
    report_dir, expected_filenames, expected,
    reference_result_path=None, reference_generator=None,
):
    report_dir = Path(report_dir)
    entries = (
        list(report_dir.iterdir())
        if report_dir.is_dir() and not report_dir.is_symlink() else []
    )
    if any(not path.is_file() or path.is_symlink() for path in entries):
        raise ValueError("remote report contains a non-file or symlink entry")
    actual_files = {path.name for path in entries}
    if actual_files != set(expected_filenames):
        raise ValueError("remote report file set is incomplete or unexpected")
    report = json.loads((report_dir / "report.json").read_text(encoding="ascii"))
    for field, value in expected.items():
        if report.get(field) != value:
            raise ValueError(f"remote report mismatch for {field}")
    files = report.get("files")
    file_sha256 = report.get("file_sha256")
    if (
        not isinstance(files, list)
        or set(files) != set(expected_filenames) - {"report.json"}
        or not isinstance(file_sha256, dict)
        or set(file_sha256) != set(files)
    ):
        raise ValueError("remote report hash manifest is invalid")
    for filename in files:
        if sha256_file(report_dir / filename) != file_sha256[filename]:
            raise ValueError(f"remote report file hash mismatch: {filename}")
    if (reference_result_path is None) != (reference_generator is None):
        raise ValueError("remote report reference validation is incomplete")
    if reference_generator is not None:
        with tempfile.TemporaryDirectory(
            prefix=".report-audit-", dir=report_dir.parent,
        ) as reference_dir:
            reference_generator(reference_result_path, reference_dir)
            for filename in expected_filenames:
                if sha256_file(report_dir / filename) != sha256_file(
                    Path(reference_dir) / filename
                ):
                    raise ValueError(
                        f"remote report differs from the live aggregate: {filename}"
                    )
    return report


def _technical_path(run_root):
    return Path(run_root) / REMOTE_TECHNICAL_RELATIVE


@contextmanager
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
    """Require generated evidence to be present byte-for-byte in this archive."""
    run_path = Path(run_path).resolve()
    source_root = Path(deployment_root).expanduser().resolve() / "source"
    deployed = (source_root / relative).resolve()
    if source_root not in deployed.parents or not run_path.is_file() or not deployed.is_file():
        raise ValueError(f"{purpose} is not present in the deployed pushed source")
    if run_path.read_bytes() != deployed.read_bytes():
        raise ValueError(f"{purpose} differs from the deployed pushed evidence")
    return sha256_file(deployed)


def _qualification_argv():
    return tuple(
        (
            name,
            (str(Path(sys.executable).resolve()), "-B", "-m", "pytest", "-q", *paths),
        )
        for name, paths in QUALIFICATION_GROUPS
    )


def _bytecode_clean(source_root):
    source_root = Path(source_root)
    return not any(
        path.name == "__pycache__"
        or (path.is_file() and path.suffix in {".pyc", ".nbi", ".nbc"})
        for path in source_root.rglob("*")
    )


def _ldpc_source_provenance(config):
    """Verify the exact official source archive used for the Linux build."""
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


def run_environment_qualification(config, deployment, source_root):
    """Run the frozen oracle/regression groups in the active remote Python."""
    config = _require_remote_config(config)
    source_root = Path(source_root).resolve()
    clean_before = _bytecode_clean(source_root)
    environment = os.environ.copy()
    environment.update({
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_ADDOPTS": "-p no:cacheprovider",
        "PYTHONPATH": str(source_root),
        "EXP103_TEST_CONFIG_PATH": str(
            source_root / "data/expander_code/exp103/config/decoder_mc.remote.v3.json"
        ),
    })
    environment.pop("PYTHONOPTIMIZE", None)
    results = []
    for name, argv in _qualification_argv():
        completed = subprocess.run(
            argv, cwd=source_root, env=environment, capture_output=True,
        )
        stdout = completed.stdout.encode() if isinstance(completed.stdout, str) else completed.stdout
        stderr = completed.stderr.encode() if isinstance(completed.stderr, str) else completed.stderr
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
    identity = runtime_identity(config)
    binary_path = decoder_binary_path()
    resolved_prefix = Path(sys.prefix).resolve()
    if resolved_prefix not in binary_path.parents:
        raise ValueError("BpLSD binary is outside the frozen Python prefix")
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
        "device_name": config["environment"]["device_name"],
        "hostname": config["environment"]["hostname"],
        "conda_environment": config["environment"]["conda_environment"],
        "conda_prefix": str(Path(os.environ.get("CONDA_PREFIX", "")).resolve()),
        "python_executable": str(Path(sys.executable).resolve()),
        "python_prefix": str(resolved_prefix),
        "python_version": identity["python_version"],
        "numpy_version": identity["numpy_version"],
        "scipy_version": identity["scipy_version"],
        "ldpc_version": identity["ldpc_version"],
        "support_packages": identity["support_packages"],
        "decoder_binary_path": str(binary_path),
        "decoder_binary_filename": binary_path.name,
        "python_no_bytecode_flag": True,
        "pythondontwritebytecode": True,
        "pytest_cache_disabled": True,
        "bytecode_clean_before": clean_before,
        "bytecode_clean_after": clean_after,
        "deployment": dict(deployment),
        "ldpc_source_provenance": _ldpc_source_provenance(config),
        "host_resources": _host_resources(config),
        "groups": results,
        "total_passed": sum(result["passed_count"] for result in results),
    }


def validate_environment_qualification(report, config):
    config = _require_remote_config(config)
    required = {
        "schema_version", "status", "experiment_id", "execution_profile_id",
        "config_sha256", "registry_sha256", "source_commit",
        "source_tree_sha256", "decoder_binary_sha256", "device_name", "hostname",
        "conda_environment", "conda_prefix", "python_executable", "python_prefix",
        "python_version", "numpy_version", "scipy_version", "ldpc_version",
        "support_packages",
        "decoder_binary_path", "decoder_binary_filename", "python_no_bytecode_flag",
        "pythondontwritebytecode", "pytest_cache_disabled",
        "bytecode_clean_before", "bytecode_clean_after", "deployment",
        "ldpc_source_provenance", "host_resources", "groups", "total_passed",
    }
    if set(report) != required or report["schema_version"] != REMOTE_QUALIFICATION_SCHEMA:
        raise ValueError("remote environment qualification schema mismatch")
    for field, expected in (
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
        ("conda_prefix", str(Path(sys.prefix).resolve())),
        ("python_executable", str(Path(sys.executable).resolve())),
        ("python_prefix", str(Path(sys.prefix).resolve())),
        ("python_version", config["environment"]["python"]),
        ("numpy_version", config["environment"]["numpy"]),
        ("scipy_version", config["environment"]["scipy"]),
        ("ldpc_version", config["environment"]["ldpc"]),
        ("support_packages", config["support_packages"]),
        ("decoder_binary_path", str(decoder_binary_path())),
        ("decoder_binary_filename", decoder_binary_path().name),
        ("python_no_bytecode_flag", True),
        ("pythondontwritebytecode", True),
        ("pytest_cache_disabled", True),
        ("bytecode_clean_before", True),
        ("bytecode_clean_after", True),
    ):
        if report[field] != expected:
            raise ValueError(f"remote environment qualification mismatch for {field}")
    expected_commands = _qualification_argv()
    groups = report["groups"]
    if len(groups) != len(expected_commands):
        raise ValueError("remote environment qualification group count mismatch")
    for result, (name, argv) in zip(groups, expected_commands):
        if set(result) != {
            "name", "argv", "exit_code", "status", "passed_count",
            "expected_passed_count", "skipped_count", "xfailed_count",
            "xpassed_count", "deselected_count", "stdout_sha256", "stderr_sha256",
        }:
            raise ValueError("remote environment qualification result fields mismatch")
        if (
            result["name"] != name
            or result["argv"] != list(argv)
            or result["exit_code"] != 0
            or result["status"] != "PASS"
            or isinstance(result["passed_count"], bool)
            or not isinstance(result["passed_count"], int)
            or result["passed_count"] != QUALIFICATION_EXPECTED_PASSES[name]
            or result["expected_passed_count"] != QUALIFICATION_EXPECTED_PASSES[name]
            or any(result[field] != 0 for field in (
                "skipped_count", "xfailed_count", "xpassed_count", "deselected_count",
            ))
        ):
            raise ValueError(f"remote environment qualification failed for {name}")
        for field in ("stdout_sha256", "stderr_sha256"):
            value = str(result[field])
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"remote environment qualification has an invalid {field}")
    total = sum(result["passed_count"] for result in groups)
    if report["total_passed"] != total or report["status"] != "PASS":
        raise ValueError("remote environment qualification total status mismatch")
    expected_provenance = {**config["ldpc_source"], "project_version": "2.4.1"}
    if report["ldpc_source_provenance"] != expected_provenance:
        raise ValueError("remote ldpc source provenance mismatch")
    resources = report["host_resources"]
    resource_fields = {
        "platform_system", "platform_release", "platform_machine", "libc_name",
        "libc_version", "logical_cpu_count", "physical_core_count",
        "cpu_socket_count", "memory_total_bytes", "memory_available_bytes",
        "run_disk_path", "run_disk_total_bytes", "run_disk_free_bytes",
    }
    if (
        not isinstance(resources, dict)
        or set(resources) != resource_fields
        or resources["platform_system"] != "Linux"
        or resources["platform_machine"] != "x86_64"
        or resources["logical_cpu_count"] < config["execution_profile"]["num_workers"]
        or resources["physical_core_count"] <= 0
        or resources["cpu_socket_count"] <= 0
        or resources["memory_total_bytes"] < 128 * 1024 ** 3
        or not 0 <= resources["memory_available_bytes"] <= resources["memory_total_bytes"]
        or resources["run_disk_path"] != str(
            Path(config["execution_profile"]["run_root"]).expanduser().resolve()
        )
        or not 0 < resources["run_disk_free_bytes"] <= resources["run_disk_total_bytes"]
    ):
        raise ValueError("remote host resource identity is invalid")
    deployment = report["deployment"]
    if (
        not isinstance(deployment, dict)
        or deployment.get("schema_version") != "exp103.remote_deployment.v1"
        or deployment.get("config_sha256") != config["config_sha256"]
        or deployment.get("source_tree_sha256") != config["source_tree_sha256"]
    ):
        raise ValueError("remote environment qualification deployment mismatch")
    return report


def _require_qualification(run_root, config, deployment_root):
    path = _qualification_path(run_root)
    if not path.is_file():
        raise ValueError("remote environment qualification is missing")
    report = json.loads(path.read_text(encoding="ascii"))
    validate_environment_qualification(report, config)
    _require_deployed_evidence_bytes(
        path, deployment_root, COMMITTED_QUALIFICATION_RELATIVE,
        "remote environment qualification",
    )
    return report


def _remote_preflight_view(config):
    """Use the frozen benchmark panel with the separately frozen remote caps."""
    config = _require_remote_config(config)
    profile = config["execution_profile"]
    view = dict(config)
    view["preflight"] = {
        **config["preflight"],
        "num_workers": profile["num_workers"],
        "reserve_multiplier": profile["reserve_multiplier"],
        "stage_core_hour_cap": profile["stage_core_hour_cap"],
        "stage_wall_hour_cap": profile["stage_wall_hour_cap"],
        "peak_rss_gib_cap": profile["peak_rss_gib_cap"],
    }
    return view


def _remote_stage_estimates(tasks, config):
    view = _remote_preflight_view(config)
    stages = {}
    for name, values in config["stage_m_values"].items():
        stage = _stage_estimate(name, values, tasks, view)
        stage["status"] = (
            "PASS" if all(stage["checks"].values())
            else "BLOCKED_REMOTE_RESOURCE_PREFLIGHT"
        )
        stages[name] = stage
    return stages


def run_remote_resource_preflight(config, deployment, qualification_report_sha256):
    config = _require_remote_config(config)
    identity = runtime_identity(config)
    tasks = [
        benchmark_task(code_id, p_token, config)
        for code_id in config["preflight"]["code_ids"]
        for p_token in config["preflight"]["p_tokens"]
    ]
    stages = _remote_stage_estimates(tasks, config)
    return {
        "schema_version": REMOTE_PREFLIGHT_SCHEMA,
        "status": (
            "PASS_ALL_STAGES"
            if all(stage["status"] == "PASS" for stage in stages.values())
            else "BLOCKED_REMOTE_RESOURCE_PREFLIGHT"
        ),
        "experiment_id": config["experiment_id"],
        "execution_profile_id": config["execution_profile"]["profile_id"],
        "config_sha256": config["config_sha256"],
        "registry_sha256": config["registry_sha256"],
        "source_commit": config["source_commit"],
        "source_tree_sha256": identity["source_tree_sha256"],
        "decoder_binary_sha256": identity["decoder_binary_sha256"],
        "device_name": identity["device_name"],
        "hostname": identity["hostname"],
        "conda_environment": identity["conda_environment"],
        "conda_prefix_matches_python": identity["conda_prefix_matches_python"],
        "num_workers": config["execution_profile"]["num_workers"],
        "omp_thread_count": config["execution_profile"]["omp_thread_count"],
        "outcome_blind": True,
        "logical_outcomes_saved": False,
        "qualification_report_sha256": qualification_report_sha256,
        "deployment": dict(deployment),
        "tasks": tasks,
        "stages": stages,
    }


def validate_remote_resource_preflight(report, config):
    config = _require_remote_config(config)
    required = {
        "schema_version", "status", "experiment_id", "execution_profile_id",
        "config_sha256", "registry_sha256", "source_commit",
        "source_tree_sha256", "decoder_binary_sha256", "device_name", "hostname",
        "conda_environment", "conda_prefix_matches_python", "num_workers",
        "omp_thread_count", "outcome_blind", "logical_outcomes_saved",
        "qualification_report_sha256", "deployment", "tasks", "stages",
    }
    if set(report) != required or report["schema_version"] != REMOTE_PREFLIGHT_SCHEMA:
        raise ValueError("remote resource preflight schema mismatch")
    expected = {
        "experiment_id": config["experiment_id"],
        "execution_profile_id": config["execution_profile"]["profile_id"],
        "config_sha256": config["config_sha256"],
        "registry_sha256": config["registry_sha256"],
        "source_commit": config["source_commit"],
        "source_tree_sha256": config["source_tree_sha256"],
        "decoder_binary_sha256": config["decoder_binary"]["sha256"],
        "device_name": config["environment"]["device_name"],
        "hostname": config["environment"]["hostname"],
        "conda_environment": config["environment"]["conda_environment"],
        "conda_prefix_matches_python": True,
        "num_workers": 64,
        "omp_thread_count": 1,
        "outcome_blind": True,
        "logical_outcomes_saved": False,
    }
    for field, value in expected.items():
        if report[field] != value:
            raise ValueError(f"remote resource preflight identity mismatch for {field}")
    qualification_sha256 = str(report["qualification_report_sha256"])
    if (
        len(qualification_sha256) != 64
        or any(character not in "0123456789abcdef" for character in qualification_sha256)
    ):
        raise ValueError("remote resource preflight qualification SHA mismatch")
    tasks = report["tasks"]
    expected_task_ids = {
        (code_id, p_token)
        for code_id in config["preflight"]["code_ids"]
        for p_token in config["preflight"]["p_tokens"]
    }
    if len(tasks) != 9 or {
        (task.get("code_id"), task.get("p_token")) for task in tasks
    } != expected_task_ids:
        raise ValueError("remote resource preflight task panel mismatch")
    task_fields = {
        "code_id", "m", "p_token", "trials", "model_seconds",
        "measurement_identity_seconds", "decoder_setup_seconds",
        "raw_serialization_seconds", "replay_identity_seconds",
        "replay_setup_seconds", "raw_load_seconds",
        "replay_raw_sha256_seconds", "manifest_seconds",
        "measurement_seconds", "replay_seconds",
        "measurement_seconds_per_trial", "replay_seconds_per_trial",
        "peak_rss_gib", "seed_namespace",
    }
    for task in tasks:
        if set(task) != task_fields:
            raise ValueError("remote resource preflight task fields mismatch")
        if task["trials"] != config["preflight"]["trials_per_task"]:
            raise ValueError("remote resource preflight trial count mismatch")
        if task["seed_namespace"] != config["namespaces"]["benchmark"]:
            raise ValueError("measurement namespace leaked into remote benchmark")
        if task["measurement_seconds_per_trial"] != task["measurement_seconds"] / task["trials"]:
            raise ValueError("remote measurement timing arithmetic mismatch")
        if task["replay_seconds_per_trial"] != task["replay_seconds"] / task["trials"]:
            raise ValueError("remote replay timing arithmetic mismatch")
        if min(
            task["model_seconds"], task["measurement_identity_seconds"],
            task["decoder_setup_seconds"], task["raw_serialization_seconds"],
            task["replay_identity_seconds"], task["replay_setup_seconds"],
            task["raw_load_seconds"], task["replay_raw_sha256_seconds"],
            task["manifest_seconds"], task["measurement_seconds"],
            task["replay_seconds"], task["peak_rss_gib"],
        ) < 0:
            raise ValueError("remote resource preflight contains a negative measurement")
    expected_stages = _remote_stage_estimates(tasks, config)
    if report["stages"] != expected_stages:
        raise ValueError("remote resource preflight stage arithmetic mismatch")
    status = (
        "PASS_ALL_STAGES"
        if all(stage["status"] == "PASS" for stage in expected_stages.values())
        else "BLOCKED_REMOTE_RESOURCE_PREFLIGHT"
    )
    if report["status"] != status:
        raise ValueError("remote resource preflight terminal status mismatch")
    deployment = report["deployment"]
    if (
        not isinstance(deployment, dict)
        or set(deployment) != {
            "schema_version", "deployment_manifest_sha256", "source_commit",
            "archive_sha256", "source_manifest_sha256", "source_tree_sha256",
            "config_sha256",
        }
        or deployment.get("schema_version") != "exp103.remote_deployment.v1"
        or deployment.get("config_sha256") != config["config_sha256"]
        or deployment.get("source_tree_sha256") != config["source_tree_sha256"]
    ):
        raise ValueError("remote resource preflight deployment identity mismatch")
    for field in (
        "deployment_manifest_sha256", "archive_sha256", "source_manifest_sha256",
        "source_tree_sha256", "config_sha256",
    ):
        value = str(deployment[field])
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"remote resource preflight deployment SHA mismatch for {field}")
    if re.fullmatch(r"[0-9a-f]{40}", str(deployment["source_commit"])) is None:
        raise ValueError("remote resource preflight deployment commit is invalid")
    return report


def _require_preflight(path, run_root, config, stage, deployment_root):
    expected = _preflight_path(run_root)
    if Path(path).resolve() != expected.resolve():
        raise ValueError("formal remote scan requires its canonical remote preflight")
    report = json.loads(expected.read_text(encoding="ascii"))
    validate_remote_resource_preflight(report, config)
    _require_qualification(run_root, config, deployment_root)
    if report["qualification_report_sha256"] != sha256_file(_qualification_path(run_root)):
        raise ValueError("remote preflight is not bound to the frozen qualification")
    _require_deployed_evidence_bytes(
        expected, deployment_root, COMMITTED_PREFLIGHT_RELATIVE,
        "remote resource preflight",
    )
    if report["stages"][stage]["status"] != "PASS":
        raise ValueError(f"{stage} is blocked by the remote resource preflight")
    return report


@lru_cache(maxsize=8)
def _cached_registry(config_path):
    config = load_config(config_path)
    return _registry(config)


def _save_code_p_task(task):
    code_id, p_token, config_path, stage_root = task
    config = load_config(config_path)
    rows = _cached_registry(str(config_path))
    row = rows[code_id]
    results = []
    for shard_index in range(config["shards_per_code_p"]):
        output = Path(stage_root) / raw_filename(code_id, p_token, shard_index)
        if Path(stage_root).is_symlink() or output.is_symlink():
            raise ValueError("remote raw paths must not be symlinks")
        if output.exists():
            existing = load_raw(output)
            reason = _validate_raw(
                existing, config, row, code_id, p_token, shard_index,
            )
            if reason is not None:
                raise ValueError(
                    "existing immutable remote raw does not match its canonical key "
                    f"({reason}): {output}"
                )
            results.append((str(output), "RESUMED"))
            continue
        raw = run_decoder_shard(code_id, p_token, shard_index, config)
        save_raw(output, raw)
        results.append((str(output), raw["status"]))
        if raw["status"] != "VALID":
            break
    return code_id, p_token, results


def _planned_code_p(config, stage):
    return [
        (f"m{m:02d}_c{code:02d}", p_token)
        for m in config["stage_m_values"][stage]
        for code in range(8)
        for p_token in config["p_tokens"]
    ]


def _assert_no_unplanned_npz(stage_root, config, stage, require_complete=False):
    stage_root = Path(stage_root)
    if stage_root.is_symlink():
        raise ValueError("remote raw stage root must not be a symlink")
    expected = {
        stage_root / raw_filename(code_id, p_token, shard)
        for code_id, p_token in _planned_code_p(config, stage)
        for shard in range(config["shards_per_code_p"])
    }
    actual = set(stage_root.glob("*.npz")) if stage_root.is_dir() else set()
    unexpected = actual - expected
    if any(path.is_symlink() for path in actual):
        raise ValueError("remote raw evidence must not be a symlink")
    if unexpected:
        raise ValueError("remote raw root contains unplanned NPZ evidence")
    if require_complete and actual != expected:
        raise ValueError(f"remote {stage} raw is incomplete")
    return expected


def _require_stage1_technical(run_root, config, deployment_root):
    path = _technical_path(run_root)
    if not path.is_file():
        raise ValueError("Stage 2 requires the immutable remote Stage 1 technical report")
    report = json.loads(path.read_text(encoding="ascii"))
    expected = build_remote_stage1_technical(run_root, config)
    if report != expected:
        raise ValueError("remote Stage 1 technical authorization is stale or tampered")
    _require_deployed_evidence_bytes(
        path, deployment_root, COMMITTED_TECHNICAL_RELATIVE,
        "remote Stage 1 technical authorization",
    )
    return report


def run_remote_scan(
    config_path, stage, run_root, preflight_report, num_workers,
    deployment_root, deployment_manifest_sha256,
):
    config = load_config(config_path)
    run_root = resolve_remote_run_root(run_root, config)
    if num_workers != config["execution_profile"]["num_workers"] or num_workers != 64:
        raise ValueError("formal remote scans require --num-workers 64")
    identity, deployment = _verify_remote_runtime(
        config, deployment_root, deployment_manifest_sha256,
    )
    preflight = _require_preflight(
        preflight_report, run_root, config, stage, deployment_root,
    )
    if stage == "stage2":
        _require_stage1_technical(run_root, config, deployment_root)
    output = _scan_report_path(run_root, stage)
    if output.exists():
        raise FileExistsError(f"remote scan report is immutable: {output}")
    stage_root = _stage_root(run_root, stage)
    stage_root.mkdir(parents=True, exist_ok=True)
    _assert_no_unplanned_npz(stage_root, config, stage)
    tasks = [
        (code_id, p_token, str(config_path), str(stage_root))
        for code_id, p_token in _planned_code_p(config, stage)
    ]
    statuses = []
    with _stage_lock(run_root, f"scan-{stage}"):
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_save_code_p_task, task) for task in tasks]
            for future in as_completed(futures):
                _, _, results = future.result()
                statuses.extend(results)
    if any(status not in {"VALID", "RESUMED"} for _, status in statuses):
        raise RuntimeError("one or more remote formal shards saved INVALID evidence")
    expected_shards = len(tasks) * config["shards_per_code_p"]
    if len(statuses) != expected_shards:
        raise RuntimeError("remote scan did not produce every planned shard")
    _assert_no_unplanned_npz(stage_root, config, stage, require_complete=True)
    result = {
        "schema_version": REMOTE_SCAN_SCHEMA,
        "status": "PASS",
        "stage": stage,
        "config_sha256": config["config_sha256"],
        "source_tree_sha256": config["source_tree_sha256"],
        "decoder_binary_sha256": config["decoder_binary"]["sha256"],
        "hostname": identity["hostname"],
        "num_workers": num_workers,
        "scheduled_code_p": len(tasks),
        "measurement_shards": len(statuses),
        "fresh_shards": sum(status == "VALID" for _, status in statuses),
        "resumed_shards": sum(status == "RESUMED" for _, status in statuses),
        "preflight_sha256": sha256_json(preflight),
        "deployment_manifest_sha256": deployment["deployment_manifest_sha256"],
    }
    if output.exists():
        raise FileExistsError(f"remote scan report is immutable: {output}")
    atomic_json(output, result)
    return result


def _replay_code_p_task(task):
    paths, config_path = task
    return [replay_decoder_shard(path, config_path) for path in paths]


def run_remote_replay(
    config_path, stage, run_root, preflight_report, num_workers,
    deployment_root, deployment_manifest_sha256,
):
    config = load_config(config_path)
    run_root = resolve_remote_run_root(run_root, config)
    if num_workers != config["execution_profile"]["num_workers"] or num_workers != 64:
        raise ValueError("formal remote replay requires --num-workers 64")
    _verify_remote_runtime(config, deployment_root, deployment_manifest_sha256)
    _require_preflight(
        preflight_report, run_root, config, stage, deployment_root,
    )
    if stage == "stage2":
        _require_stage1_technical(run_root, config, deployment_root)
    stage_root = _stage_root(run_root, stage)
    expected = _assert_no_unplanned_npz(
        stage_root, config, stage, require_complete=True,
    )
    output = _replay_path(run_root, stage)
    if output.exists():
        raise FileExistsError(f"remote replay evidence is immutable: {output}")
    tasks = []
    for code_id, p_token in _planned_code_p(config, stage):
        paths = sorted(
            stage_root / raw_filename(code_id, p_token, shard)
            for shard in range(config["shards_per_code_p"])
        )
        tasks.append((paths, str(config_path)))
    if set(path for paths, _ in tasks for path in paths) != expected:
        raise AssertionError("internal remote replay plan mismatch")
    with _stage_lock(run_root, f"replay-{stage}"):
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            nested = list(executor.map(_replay_code_p_task, tasks))
    results = [item for group in nested for item in group]
    report = build_replay_report(stage_root, results, config)
    if report["scope"] != stage:
        raise ValueError("remote replay report has the wrong stage scope")
    if output.exists():
        raise FileExistsError(f"remote replay evidence is immutable: {output}")
    atomic_json(output, report)
    return report


def run_remote_aggregate(
    config_path, scope, run_root, preflight_report,
    deployment_root, deployment_manifest_sha256,
):
    config = load_config(config_path)
    run_root = resolve_remote_run_root(run_root, config)
    _verify_remote_runtime(config, deployment_root, deployment_manifest_sha256)
    for stage in (("stage1",) if scope == "stage1" else ("stage1", "stage2")):
        _require_preflight(
            preflight_report, run_root, config, stage, deployment_root,
        )
    if scope == "final":
        _require_stage1_technical(run_root, config, deployment_root)
    with _stage_lock(run_root, f"aggregate-{scope}"):
        result = aggregate_decoder_scan(_raw_root(run_root), config)
        expected_scope = "stage1" if scope == "stage1" else "final_combined"
        if result["replay_scope"] != expected_scope:
            raise ValueError("remote aggregate does not have the required replay scope")
        output = _aggregate_path(run_root, scope)
        save_aggregate(output, result)
        # Preserve an invalid aggregate as evidence, but never return success for it.
        _require_scope_reportable(result, scope)
    return result


def run_remote_publication(
    config_path, run_root, preflight_report,
    deployment_root, deployment_manifest_sha256,
):
    """Loader-verify the complete panel and atomically publish final reports."""
    config = load_config(config_path)
    run_root = resolve_remote_run_root(run_root, config)
    _verify_remote_runtime(config, deployment_root, deployment_manifest_sha256)
    for stage in ("stage1", "stage2"):
        _require_preflight(
            preflight_report, run_root, config, stage, deployment_root,
        )
    _require_stage1_technical(run_root, config, deployment_root)
    aggregate_path = _aggregate_path(run_root, "final")
    if not aggregate_path.is_file():
        raise ValueError("remote publication requires the final aggregate")
    _require_live_aggregate_matches(
        aggregate_path, _raw_root(run_root), config, "final",
    )
    parent = aggregate_path.parent
    output = parent / "publication"
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"remote publication evidence is immutable: {output}")
    with _stage_lock(run_root, "publication"):
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"remote publication evidence is immutable: {output}")
        staging = Path(tempfile.mkdtemp(prefix=".publication.partial-", dir=parent))
        summary = generate_final_report(aggregate_path, staging)
        allowed = {
            "EXP103_DECODER_CROSSING_RESOLVED",
            "EXP103_PAIRWISE_BRACKET_ONLY",
            "EXP103_NO_CORRECT_CROSSING_IN_WINDOW",
            "EXP103_DECODER_CROSSING_INCONCLUSIVE",
        }
        if (
            summary.get("terminal_status") not in allowed
            or summary.get("num_code_p") != 624
            or summary.get("total_trials") != 6_240_000
        ):
            raise ValueError("remote publication did not validate the complete frozen panel")
        os.replace(staging, output)
    return summary


def run_remote_stage1_preliminary(
    config_path, run_root, preflight_report,
    deployment_root, deployment_manifest_sha256,
):
    config = load_config(config_path)
    run_root = resolve_remote_run_root(run_root, config)
    _verify_remote_runtime(config, deployment_root, deployment_manifest_sha256)
    _require_preflight(
        preflight_report, run_root, config, "stage1", deployment_root,
    )
    _require_stage1_technical(run_root, config, deployment_root)
    aggregate_path = _aggregate_path(run_root, "stage1")
    if not aggregate_path.is_file():
        raise ValueError("Stage 1 preliminary publication requires its aggregate")
    parent = aggregate_path.parent
    output = parent / "stage1_preliminary"
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Stage 1 preliminary evidence is immutable: {output}")
    with _stage_lock(run_root, "stage1-preliminary"):
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"Stage 1 preliminary evidence is immutable: {output}")
        staging = Path(tempfile.mkdtemp(prefix=".stage1-preliminary.partial-", dir=parent))
        summary = generate_stage1_preliminary_report(aggregate_path, staging)
        if (
            summary.get("reportable_code_p") != 312
            or summary.get("total_trials") != 3_120_000
            or summary.get("stage2_decision_uses_curves") is not False
        ):
            raise ValueError("Stage 1 preliminary report did not validate the frozen panel")
        os.replace(staging, output)
    return summary


def _load_aggregate(path):
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != set(ARRAY_FIELDS) | set(SCALAR_FIELDS):
            raise ValueError("remote Stage 1 aggregate fields mismatch")
        return {
            key: data[key].copy() if key in ARRAY_FIELDS else data[key].item()
            for key in data.files
        }


def build_remote_stage1_technical(run_root, config):
    config = _require_remote_config(config)
    raw_root = _stage_root(run_root, "stage1")
    replay_path = _replay_path(run_root, "stage1")
    aggregate_path = _aggregate_path(run_root, "stage1")
    aggregate = _load_aggregate(aggregate_path)
    if aggregate["payload_sha256"] != arrays_sha256(aggregate, ARRAY_FIELDS):
        raise ValueError("remote Stage 1 aggregate payload hash mismatch")
    for field, expected in (
        ("schema_version", "exp103.aggregate.v2"),
        ("experiment_id", config["experiment_id"]),
        ("config_sha256", config["config_sha256"]),
        ("registry_sha256", config["registry_sha256"]),
        ("source_commit", config["source_commit"]),
        ("source_tree_sha256", config["source_tree_sha256"]),
        ("decoder_binary_sha256", config["decoder_binary"]["sha256"]),
        ("overall_status", "INCOMPLETE"),
        ("terminal_status", "EXP103_INCOMPLETE"),
        ("replay_status", "PASS"),
        ("replay_scope", "stage1"),
    ):
        if aggregate[field] != expected:
            raise ValueError(f"remote Stage 1 aggregate mismatch for {field}")
    if (
        not np.all(aggregate["code_status"][:24] == "REPORTABLE")
        or not np.all(aggregate["code_status"][24:] == "INCOMPLETE")
        or not np.all(aggregate["m_status"][:3] == "REPORTABLE")
        or not np.all(aggregate["m_status"][3:] == "INCOMPLETE")
        or json.loads(aggregate["unexpected_raw_errors_json"])
    ):
        raise ValueError("remote Stage 1 aggregate is not exactly the frozen 312-cell panel")
    replay = json.loads(replay_path.read_text(encoding="ascii"))
    validate_replay_report(replay, raw_root, config, "stage1")
    replay_sha256 = sha256_json(replay)
    if (
        aggregate["replay_report_sha256"] != replay_sha256
        or aggregate["raw_manifest_sha256"] != replay["raw_manifest_sha256"]
        or aggregate["replay_report_json"] != canonical_json(replay)
    ):
        raise ValueError("remote Stage 1 aggregate is not bound to replay evidence")
    return {
        "schema_version": REMOTE_TECHNICAL_SCHEMA,
        "status": "TECHNICAL_PASS",
        "config_sha256": config["config_sha256"],
        "registry_sha256": config["registry_sha256"],
        "source_tree_sha256": config["source_tree_sha256"],
        "decoder_binary_sha256": config["decoder_binary"]["sha256"],
        "reportable_code_p": int(np.sum(aggregate["code_status"][:24] == "REPORTABLE")),
        "measurement_shards": replay["shards"],
        "replay_status": replay["status"],
        "outcome_blind_stage2_authorization": True,
        "aggregate_sha256": sha256_file(aggregate_path),
        "replay_report_sha256": replay_sha256,
        "raw_manifest_sha256": replay["raw_manifest_sha256"],
    }


def save_remote_stage1_technical(run_root, config):
    output = _technical_path(run_root)
    with _stage_lock(run_root, "stage1-technical"):
        if output.exists():
            raise FileExistsError(f"remote Stage 1 technical evidence is immutable: {output}")
        report = build_remote_stage1_technical(run_root, config)
        atomic_json(output, report)
    return report


def _require_scan_report(run_root, stage, config, preflight):
    path = _scan_report_path(run_root, stage)
    report = json.loads(path.read_text(encoding="ascii"))
    required = {
        "schema_version", "status", "stage", "config_sha256",
        "source_tree_sha256", "decoder_binary_sha256", "hostname",
        "num_workers", "scheduled_code_p", "measurement_shards",
        "fresh_shards", "resumed_shards", "preflight_sha256",
        "deployment_manifest_sha256",
    }
    expected_shards = len(_planned_code_p(config, stage)) * config["shards_per_code_p"]
    if (
        set(report) != required
        or report["schema_version"] != REMOTE_SCAN_SCHEMA
        or report["status"] != "PASS"
        or report["stage"] != stage
        or report["config_sha256"] != config["config_sha256"]
        or report["source_tree_sha256"] != config["source_tree_sha256"]
        or report["decoder_binary_sha256"] != config["decoder_binary"]["sha256"]
        or report["hostname"] != config["environment"]["hostname"]
        or report["num_workers"] != 64
        or report["scheduled_code_p"] != len(_planned_code_p(config, stage))
        or report["measurement_shards"] != expected_shards
        or report["fresh_shards"] + report["resumed_shards"] != expected_shards
        or report["preflight_sha256"] != sha256_json(preflight)
        or re.fullmatch(
            r"[0-9a-f]{64}", str(report["deployment_manifest_sha256"]),
        ) is None
    ):
        raise ValueError(f"remote {stage} scan report is stale or invalid")
    return report


def run_remote_verify_stage(
    config_path, stage, run_root, preflight_report,
    deployment_root, deployment_manifest_sha256,
):
    """Revalidate every live artifact before a resumable stage exits zero."""
    config = load_config(config_path)
    run_root = resolve_remote_run_root(run_root, config)
    _verify_remote_runtime(config, deployment_root, deployment_manifest_sha256)
    stages = ("stage1",) if stage == "stage1" else ("stage1", "stage2")
    preflights = {
        item: _require_preflight(
            preflight_report, run_root, config, item, deployment_root,
        )
        for item in stages
    }
    if stage == "stage2":
        _require_stage1_technical(run_root, config, deployment_root)
        stage1_aggregate = _load_aggregate(_aggregate_path(run_root, "stage1"))
        _require_scope_reportable(stage1_aggregate, "stage1")
        stage1_bracket = (
            None if not np.isfinite(stage1_aggregate["stage1_bracket_low"])
            else [
                stage1_aggregate["stage1_bracket_low"],
                stage1_aggregate["stage1_bracket_high"],
            ]
        )
        _require_hashed_report(
            Path(run_root) / "final_results" / "stage1_preliminary",
            STAGE1_REPORT_FILENAMES,
            {
                "schema_version": "exp103.stage1_preliminary_report.v1",
                "aggregate_sha256": sha256_file(_aggregate_path(run_root, "stage1")),
                "stage1_status": stage1_aggregate["stage1_status"],
                "crossing_bracket": stage1_bracket,
                "reportable_code_p": 312,
                "total_trials": 3_120_000,
                "stage2_decision_uses_curves": False,
                "exp102_blockers_cleared": [],
            },
            _aggregate_path(run_root, "stage1"),
            generate_stage1_preliminary_report,
        )
    _require_scan_report(run_root, stage, config, preflights[stage])
    replay_path = _replay_path(run_root, stage)
    replay = json.loads(replay_path.read_text(encoding="ascii"))
    validate_replay_report(replay, _stage_root(run_root, stage), config, stage)
    scope = "stage1" if stage == "stage1" else "final"
    aggregate_path = _aggregate_path(run_root, scope)
    aggregate = _require_live_aggregate_matches(
        aggregate_path, _raw_root(run_root), config, scope,
    )
    if stage == "stage1":
        technical_path = _technical_path(run_root)
        technical = json.loads(technical_path.read_text(encoding="ascii"))
        if technical != build_remote_stage1_technical(run_root, config):
            raise ValueError("remote Stage 1 technical report is stale or invalid")
    else:
        bracket = (
            None if not np.isfinite(aggregate["crossing_bracket_low"])
            else [aggregate["crossing_bracket_low"], aggregate["crossing_bracket_high"]]
        )
        _require_hashed_report(
            Path(run_root) / "final_results" / "publication",
            FINAL_REPORT_FILENAMES,
            {
                "schema_version": "exp103.final_report.v1",
                "aggregate_sha256": sha256_file(aggregate_path),
                "terminal_status": aggregate["terminal_status"],
                "crossing_bracket": bracket,
                "num_code_p": 624,
                "total_trials": 6_240_000,
                "authority": "finite_grid_bposd_decoder_crossing_only",
                "exp102_blockers_cleared": [],
            },
            aggregate_path,
            generate_final_report,
        )
    return {"status": "PASS", "stage": stage}


def _add_common(parser, include_workers=False):
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--deployment-root", required=True)
    parser.add_argument("--deployment-manifest-sha256", required=True)
    if include_workers:
        parser.add_argument("--num-workers", type=int, required=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    qualification = commands.add_parser("qualification")
    _add_common(qualification)
    preflight = commands.add_parser("preflight")
    _add_common(preflight)
    scan = commands.add_parser("scan")
    _add_common(scan, include_workers=True)
    scan.add_argument("--stage", choices=("stage1", "stage2"), required=True)
    scan.add_argument("--preflight-report", required=True)
    replay = commands.add_parser("replay")
    _add_common(replay, include_workers=True)
    replay.add_argument("--stage", choices=("stage1", "stage2"), required=True)
    replay.add_argument("--preflight-report", required=True)
    aggregate = commands.add_parser("aggregate")
    _add_common(aggregate)
    aggregate.add_argument("--scope", choices=("stage1", "final"), required=True)
    aggregate.add_argument("--preflight-report", required=True)
    publication = commands.add_parser("publication")
    _add_common(publication)
    publication.add_argument("--preflight-report", required=True)
    preliminary = commands.add_parser("stage1-preliminary")
    _add_common(preliminary)
    preliminary.add_argument("--preflight-report", required=True)
    technical = commands.add_parser("stage1-technical")
    _add_common(technical)
    technical.add_argument("--preflight-report", required=True)
    verify_stage = commands.add_parser("verify-stage")
    _add_common(verify_stage)
    verify_stage.add_argument("--stage", choices=("stage1", "stage2"), required=True)
    verify_stage.add_argument("--preflight-report", required=True)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    run_root = resolve_remote_run_root(args.run_root, config)
    if args.command == "qualification":
        _, deployment = _verify_remote_runtime(
            config, args.deployment_root, args.deployment_manifest_sha256,
        )
        output = _qualification_path(run_root)
        if output.exists():
            raise FileExistsError(f"remote qualification evidence is immutable: {output}")
        source_root = Path(args.deployment_root).expanduser().resolve() / "source"
        report = run_environment_qualification(config, deployment, source_root)
        atomic_json(output, report)
        print(report["status"])
        if report["status"] != "PASS":
            raise SystemExit(1)
    elif args.command == "preflight":
        _, deployment = _verify_remote_runtime(
            config, args.deployment_root, args.deployment_manifest_sha256,
        )
        _require_qualification(run_root, config, args.deployment_root)
        output = _preflight_path(run_root)
        if output.exists():
            raise FileExistsError(f"remote preflight evidence is immutable: {output}")
        report = run_remote_resource_preflight(
            config, deployment, sha256_file(_qualification_path(run_root)),
        )
        atomic_json(output, report)
        print(report["status"])
        if report["status"] != "PASS_ALL_STAGES":
            raise SystemExit(1)
    elif args.command == "scan":
        result = run_remote_scan(
            args.config, args.stage, run_root, args.preflight_report,
            args.num_workers, args.deployment_root,
            args.deployment_manifest_sha256,
        )
        print(json.dumps(result, sort_keys=True))
    elif args.command == "replay":
        report = run_remote_replay(
            args.config, args.stage, run_root, args.preflight_report,
            args.num_workers, args.deployment_root,
            args.deployment_manifest_sha256,
        )
        print(report["status"])
        if report["status"] != "PASS":
            raise SystemExit(1)
    elif args.command == "aggregate":
        result = run_remote_aggregate(
            args.config, args.scope, run_root, args.preflight_report,
            args.deployment_root, args.deployment_manifest_sha256,
        )
        print(result["terminal_status"])
    elif args.command == "publication":
        result = run_remote_publication(
            args.config, run_root, args.preflight_report,
            args.deployment_root, args.deployment_manifest_sha256,
        )
        print(result["terminal_status"])
    elif args.command == "stage1-preliminary":
        result = run_remote_stage1_preliminary(
            args.config, run_root, args.preflight_report,
            args.deployment_root, args.deployment_manifest_sha256,
        )
        print(result["stage1_status"])
    elif args.command == "verify-stage":
        result = run_remote_verify_stage(
            args.config, args.stage, run_root, args.preflight_report,
            args.deployment_root, args.deployment_manifest_sha256,
        )
        print(result["status"])
    else:
        _verify_remote_runtime(
            config, args.deployment_root, args.deployment_manifest_sha256,
        )
        _require_preflight(
            args.preflight_report, run_root, config, "stage1",
            args.deployment_root,
        )
        report = save_remote_stage1_technical(run_root, config)
        print(report["status"])


if __name__ == "__main__":
    main()
