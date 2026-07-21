"""Shared fail-closed machinery for the isolated q=0 diagnostic screen."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import time


CONTRACT_VERSION = "exp102.q0_global.screen_diagnostic.v1"
SCHEDULE_VERSION = "exp102.q0_global.screen_diagnostic.schedule.v1"
OWNERSHIP_VERSION = "exp102.q0_global.screen_diagnostic.ownership.v1"
RUNTIME_NODE_VERSION = "exp102.q0_global.screen_diagnostic.runtime_node.v1"
RUNTIME_CONSENSUS_VERSION = (
    "exp102.q0_global.screen_diagnostic.runtime_consensus.v1"
)
DIGEST_NODE_VERSION = "exp102.q0_global.screen_diagnostic.digest_node.v2"
DIGEST_CONSENSUS_VERSION = (
    "exp102.q0_global.screen_diagnostic.digest_consensus.v2"
)
PREFLIGHT_NODE_VERSION = (
    "exp102.q0_global.screen_diagnostic.preflight_node.v1"
)
PREFLIGHT_VERSION = "exp102.q0_global.screen_diagnostic.preflight.v1"
RAW_MANIFEST_VERSION = (
    "exp102.q0_global.screen_diagnostic.remote_raw_manifest.v1"
)
DECISION_VERSION = "exp102.q0_global.screen_diagnostic.decision.v1"

EXPECTED_PREFLIGHT_NODES = ("nd-1", "nd-2", "nd-3")
EXECUTION_NODES = ("nd-1", "nd-3")
NODE_CAPACITY_FALLBACK = {"nd-1": 75, "nd-2": 75, "nd-3": 91}
PREFLIGHT_SECONDS = 8 * 3600.0
BIAS_DEADLINE_SECONDS = 12 * 3600.0
MEASUREMENT_DEADLINE_SECONDS = 22 * 3600.0
SCREEN_DEADLINE_SECONDS = 24 * 3600.0
SCREEN_WINDOW_SECONDS = 14 * 3600.0
TRAJECTORY_LIMIT_SECONDS = 2 * 3600.0
SAFETY_FACTOR = 2.0

VALIDATION_RELATIVE = Path(
    "data/expander_code/exp102/validation/"
    "011_q0_global_screen_diagnostic_20260721"
)
DEFAULT_REGISTRY_RELATIVE = Path(
    "data/expander_code/exp102/registry/registry.json"
)
DEFAULT_CONFIG_RELATIVE = Path(
    "data/expander_code/exp102/config/"
    "q0_global.screen_diagnostic.v1.json"
)
FULL_SHA_RE = re.compile(r"[0-9a-f]{40}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def canonical_json(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    )


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
    payload = (canonical_json(value) + "\n").encode("ascii")
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with open(temporary, "xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_exclusive(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (canonical_json(value) + "\n").encode("ascii")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(payload).hexdigest()


def validate_source_and_hashes(source_commit, *hashes):
    if FULL_SHA_RE.fullmatch(str(source_commit)) is None:
        raise ValueError("diagnostic source commit must be a full lowercase Git SHA")
    if any(SHA256_RE.fullmatch(str(value)) is None for value in hashes):
        raise ValueError("diagnostic identity hashes must be lowercase SHA256")


def pipeline_module():
    return importlib.import_module(
        "data.expander_code.exp102.exp102_pipeline.screen_diagnostic"
    )


def pipeline_attr(*names, required=True):
    module = pipeline_module()
    for name in names:
        if hasattr(module, name):
            return getattr(module, name)
    if required:
        raise AttributeError(
            "screen_diagnostic API lacks all supported names: "
            + ", ".join(names)
        )
    return None


def _call_adapted(function, values):
    """Call a pipeline API by matching its declared keyword names."""
    signature = inspect.signature(function)
    kwargs = {
        name: values[name]
        for name, parameter in signature.parameters.items()
        if name in values
        and parameter.kind
        in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY)
    }
    missing = [
        name for name, parameter in signature.parameters.items()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind
        in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD,
            parameter.KEYWORD_ONLY)
        and name not in kwargs
    ]
    if missing:
        raise TypeError(
            f"unsupported screen_diagnostic API {function.__name__}; "
            f"missing adapter values {missing}"
        )
    return function(**kwargs)


def load_registry(path):
    from data.expander_code.exp102.exp102_pipeline.registry import load_registry
    return load_registry(path)


def load_config(path, registry=None):
    function = pipeline_attr(
        "load_screen_diagnostic_config",
        "load_q0_screen_diagnostic_config",
        "load_diagnostic_config",
    )
    return function(path, registry) if registry is not None else function(path)


def config_sha256(config):
    for name in (
        "screen_config_sha256", "diagnostic_config_sha256",
        "discovery_config_sha256", "config_sha256",
    ):
        value = config.get(name)
        if isinstance(value, str) and SHA256_RE.fullmatch(value):
            return value
    raw = {
        key: value for key, value in config.items()
        if key not in {
            "screen_config_sha256", "diagnostic_config_sha256",
            "discovery_config_sha256", "config_sha256", "config_path",
        }
    }
    return sha256_json(raw)


def hard_methods():
    return tuple(pipeline_attr("HARD_METHODS", "SCREEN_HARD_METHODS"))


def defect_methods():
    return tuple(pipeline_attr("DEFECT_METHODS", "SCREEN_DEFECT_METHODS"))


def all_methods():
    return (*hard_methods(), *defect_methods())


def resource_tiers(config=None):
    value = pipeline_attr("RESOURCE_TIERS", "SCREEN_RESOURCE_TIERS", required=False)
    if value is not None:
        return dict(value)
    if config is not None and isinstance(config.get("resource_tiers"), dict):
        return dict(config["resource_tiers"])
    raise AttributeError("screen_diagnostic does not expose resource tiers")


def node_capacity():
    value = pipeline_attr("NODE_CAPACITY", required=False)
    return NODE_CAPACITY_FALLBACK if value is None else dict(value)


def uniform_seed_for_cell(registry, code, cell):
    function = pipeline_attr(
        "uniform_seed_for_cell", "screen_uniform_seed_for_cell",
        "diagnostic_uniform_seed_for_cell",
    )
    return int(function(registry, code, cell))


def build_bias_manifest(registry_path, config_path, source_commit, tier,
                        output_path):
    function = pipeline_attr(
        "build_screen_bias_manifest", "build_bias_manifest",
    )
    return _call_adapted(function, {
        "registry_path": registry_path, "config_path": config_path,
        "source_commit": source_commit, "stage": "screen",
        "method_tiers": [(method, tier) for method in defect_methods()],
        "resource_tier": tier, "tier": tier, "output_path": output_path,
    })


def build_measurement_manifest(registry_path, config_path, source_commit, tier,
                               output_path, bias_manifest_path, bias_raw_root):
    function = pipeline_attr(
        "build_screen_measurement_manifest", "build_measurement_manifest",
    )
    return _call_adapted(function, {
        "registry_path": registry_path, "config_path": config_path,
        "source_commit": source_commit, "stage": "screen",
        "method_tiers": [(method, tier) for method in all_methods()],
        "resource_tier": tier, "tier": tier, "output_path": output_path,
        "bias_manifest_path": bias_manifest_path,
        "bias_raw_root": bias_raw_root,
    })


def validate_control(manifest, registry, config):
    function = pipeline_attr(
        "validate_screen_control_manifest", "validate_control_manifest",
        "validate_diagnostic_control_manifest",
    )
    return function(manifest, registry, config)


def run_bias_task(registry_path, config_path, source_commit, task, output_path):
    function = pipeline_attr("run_screen_bias_task", "run_bias_task")
    return function(registry_path, config_path, source_commit, task, output_path)


def run_hard_task(registry_path, config_path, source_commit, task, output_path):
    function = pipeline_attr("run_screen_hard_task", "run_hard_task")
    return function(registry_path, config_path, source_commit, task, output_path)


def run_defect_task(registry_path, config_path, source_commit, task, bias_path,
                    output_path, *, _validated_bias_record=None):
    function = pipeline_attr("run_screen_defect_task", "run_defect_task")
    if _validated_bias_record is None:
        return function(
            registry_path, config_path, source_commit, task, bias_path,
            output_path,
        )
    return function(
        registry_path, config_path, source_commit, task, bias_path, output_path,
        _validated_bias_record=_validated_bias_record,
    )


def validate_raw(path, kind, method_id, registry, config, source_commit,
                 bias_path=None):
    if kind == "defect_bias":
        function = pipeline_attr("validate_screen_bias_raw", "validate_bias_raw")
        return function(path, registry, config, source_commit)
    if method_id in defect_methods():
        function = pipeline_attr(
            "validate_screen_defect_raw", "validate_defect_raw",
        )
    else:
        function = pipeline_attr("validate_screen_hard_raw", "validate_hard_raw")
    if method_id in defect_methods():
        return function(path, registry, config, source_commit, bias_path)
    return function(path, registry, config, source_commit)


def analyze_measurement(raw_root, manifest_path, registry_path, config_path,
                        output_path, num_workers):
    function = pipeline_attr(
        "analyze_screen_measurement_stage", "analyze_measurement_stage",
        "analyze_screen_measurement",
    )
    return _call_adapted(function, {
        "raw_root": raw_root, "manifest_path": manifest_path,
        "registry_path": registry_path, "config_path": config_path,
        "output_path": output_path, "num_workers": num_workers,
    })


def task_version():
    return str(pipeline_attr(
        "SCREEN_TASKS_VERSION", "DIAGNOSTIC_TASKS_VERSION",
        "TASKS_VERSION",
    ))


def resolve_source_path(source, relative):
    source = Path(source).resolve()
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("diagnostic source path must be a safe relative path")
    resolved = (source / relative).resolve(strict=True)
    try:
        resolved.relative_to(source)
    except ValueError as error:
        raise ValueError("diagnostic source path escapes verified source") from error
    return resolved


def freeze_schedule(registry_path, config_path, source_commit, archive_sha256,
                    source_manifest_sha256, output_path, *, started_unix=None):
    validate_source_and_hashes(source_commit, archive_sha256,
                               source_manifest_sha256)
    function = pipeline_attr("freeze_screen_schedule")
    values = {
        "registry_path": registry_path, "config_path": config_path,
        "source_commit": source_commit, "archive_sha256": archive_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "output_path": output_path,
    }
    if started_unix is not None:
        values["started_unix"] = started_unix
    return _call_adapted(function, values)


def validate_schedule(path, registry, config, source_commit=None):
    function = pipeline_attr("validate_screen_schedule")
    return function(path, registry, config, source_commit)


def _task_cost(task):
    cell = task.get("cell", {})
    code_id = str(cell.get("code_id", "m01"))
    try:
        m = int(code_id[1:3])
    except (ValueError, IndexError):
        m = 1
    sampler = task.get("sampler_config", {})
    work = int(sampler.get("burn_sweeps", 1)) + int(
        sampler.get("measurement_sweeps", 1)
    )
    method = str(task.get("method_id", ""))
    if method.startswith("RC8-QC"):
        multiplier = 1.0 + float(sampler.get("cluster_repeats", 1))
    elif method.startswith("RC8-J"):
        multiplier = 1.0 + (1 << int(sampler.get("joint_block_size", 8))) / 256.0
    else:
        multiplier = 1.0
    if task.get("kind") == "defect_bias" or task.get("stage_kind") == "defect_bias":
        multiplier *= float(sampler.get("tuning_chains", 8))
        work = int(sampler.get("tuning_sweeps", 4096))
    return max(1.0, float(m * m * work) * multiplier)


def fixed_ownership(tasks, source_commit, registry_sha256,
                    diagnostic_config_sha256, schedule_file_sha256,
                    schedule_sha256, control_sha256, runtime_report_sha256,
                    stage, kind):
    nodes = list(EXECUTION_NODES)
    capacities = node_capacity()
    loads = {node: 0.0 for node in nodes}
    owners = {}
    task_rows = [(task, sha256_json(task), _task_cost(task)) for task in tasks]
    if len({fingerprint for _, fingerprint, _ in task_rows}) != len(task_rows):
        raise ValueError("diagnostic ownership received duplicate tasks")
    if kind == "diagnostic_measurement":
        grouped = {}
        for task, fingerprint, cost in task_rows:
            group_key = (
                task["method_id"], sha256_json(task["cell"]),
                task.get("resource_tier"),
            )
            grouped.setdefault(group_key, []).append((task, fingerprint, cost))
        groups = sorted(
            grouped.values(),
            key=lambda values: (
                -sum(value[2] for value in values),
                tuple(sorted(value[1] for value in values)),
            ),
        )
    else:
        groups = [[value] for value in sorted(
            task_rows, key=lambda value: (-value[2], value[1]),
        )]
    for group in groups:
        owner = min(nodes, key=lambda node: (loads[node] / capacities[node], node))
        group_cost = 0.0
        for _, fingerprint, cost in group:
            owners[fingerprint] = owner
            group_cost += cost
        loads[owner] += group_cost
    identity = {
        "ownership_version": OWNERSHIP_VERSION,
        "contract_version": CONTRACT_VERSION,
        "source_commit": source_commit,
        "registry_sha256": registry_sha256,
        "diagnostic_config_sha256": diagnostic_config_sha256,
        "schedule_file_sha256": schedule_file_sha256,
        "schedule_sha256": schedule_sha256,
        "control_sha256": control_sha256,
        "runtime_report_sha256": runtime_report_sha256,
        "stage": stage,
        "kind": kind,
        "nodes": nodes,
        "task_owner": owners,
    }
    return {
        **identity,
        "stage_fingerprint": sha256_json(identity),
        "weighted_load": loads,
        "capacity": {node: capacities[node] for node in nodes},
    }


def validate_ownership(ownership, tasks, source_commit, registry_sha256,
                       diagnostic_config_sha256, schedule_file_sha256,
                       schedule_sha256, control_sha256,
                       runtime_report_sha256, stage, kind):
    expected = fixed_ownership(
        tasks, source_commit, registry_sha256, diagnostic_config_sha256,
        schedule_file_sha256, schedule_sha256, control_sha256,
        runtime_report_sha256, stage, kind,
    )
    if ownership != expected:
        raise ValueError("diagnostic ownership is not canonical")
    return ownership


def remote_command(arguments):
    return " ".join(shlex.quote(str(value)) for value in arguments)


def verified_bootstrap(deployment_root, source_commit, archive_sha256,
                       manifest_sha256, stage_dir, log_file,
                       stage_fingerprint, command):
    archive = Path(deployment_root) / "SOURCE.tar"
    verifier = Path(
        "data/expander_code/exp102/validation/002_numba_smoke_20260719/"
        "run_verified_source.sh"
    )
    wrapper = VALIDATION_RELATIVE / "run_screen_wrapper.sh"
    guarded = (
        "set -euo pipefail; "
        f"tar -xOf {shlex.quote(str(archive))} "
        f"{shlex.quote(verifier.as_posix())} | bash -s -- "
        f"{shlex.quote(str(deployment_root))} {source_commit} "
        f"{archive_sha256} {manifest_sha256} {remote_command(command)}"
    )
    return (
        "set -euo pipefail; "
        f"printf '%s  %s\\n' {archive_sha256} "
        f"{shlex.quote(str(archive))} | sha256sum -c - >/dev/null; "
        f"tar -xOf {shlex.quote(str(archive))} "
        f"{shlex.quote(wrapper.as_posix())} | bash -s -- "
        f"{shlex.quote(str(stage_dir))} {shlex.quote(str(log_file))} "
        f"{stage_fingerprint} bash -c {shlex.quote(guarded)}"
    )


def wait_for_markers(stage_dirs, stage_fingerprint, timeout_seconds):
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
        raise ValueError("diagnostic marker timeout is invalid")
    deadline = time.monotonic() + timeout_seconds
    previous = None
    while True:
        states = {}
        for node, root in stage_dirs.items():
            found = [
                name for name in ("RUNNING", "SUCCESS", "FAILED")
                if (Path(root) / name).exists()
            ]
            states[node] = "+".join(found) if found else "MISSING"
            for name in found:
                marker = json.loads(
                    (Path(root) / name).read_text(encoding="ascii")
                )
                if marker.get("stage_fingerprint") != stage_fingerprint:
                    raise ValueError("diagnostic marker fingerprint mismatch")
        if states != previous:
            print(" ".join(
                f"{node}={state}" for node, state in sorted(states.items())
            ), flush=True)
            previous = states
        if any("FAILED" in value for value in states.values()):
            raise RuntimeError("diagnostic remote stage failed")
        if all(value == "SUCCESS" for value in states.values()):
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("diagnostic remote stage exceeded its deadline")
        time.sleep(2.0)


def read_linux_loads(nodes):
    loads = {}
    for node in nodes:
        output = subprocess.run(
            ("ssh", node, "cat /proc/loadavg"), check=True,
            capture_output=True, text=True,
        ).stdout.split()
        if len(output) < 3:
            raise ValueError(f"cannot parse load average from {node}")
        loads[node] = tuple(float(value) for value in output[:3])
    return loads


def validate_runtime_consensus(path, source_commit, registry_sha256,
                               diagnostic_config_sha256, archive_sha256,
                               manifest_sha256):
    report = json.loads(Path(path).read_text(encoding="ascii"))
    identity = report.get("source_identity")
    expected_fields = {
        "benchmark_version", "contract_version", "source_commit",
        "source_identity", "registry_sha256", "diagnostic_config_sha256",
        "environment", "node_report_sha256", "completed_unix_max",
        "projections", "selected_resource_tier",
        "selected_eligible_methods", "excluded_work", "status",
    }
    source_identity_fields = {
        "source_commit", "mode", "archive_sha256", "manifest_sha256",
        "file_count",
    }
    if (set(report) != expected_fields
            or report.get("benchmark_version") != RUNTIME_CONSENSUS_VERSION
            or report.get("contract_version") != CONTRACT_VERSION
            or report.get("source_commit") != source_commit
            or report.get("registry_sha256") != registry_sha256
            or report.get("diagnostic_config_sha256")
            != diagnostic_config_sha256
            or report.get("status") != "PASS"
            or report.get("environment", {}).get("system") != "Linux"
            or not isinstance(identity, dict)
            or set(identity) != source_identity_fields
            or identity.get("source_commit") != source_commit
            or identity.get("mode") != "archive"
            or identity.get("archive_sha256") != archive_sha256
            or identity.get("manifest_sha256") != manifest_sha256
            or isinstance(identity.get("file_count"), bool)
            or not isinstance(identity.get("file_count"), int)
            or identity["file_count"] <= 0
            or report.get("environment") != {
                "system": "Linux", "nodes": list(EXPECTED_PREFLIGHT_NODES),
            }
            or set(report.get("node_report_sha256", {}))
            != set(EXPECTED_PREFLIGHT_NODES)
            or any(SHA256_RE.fullmatch(str(value)) is None
                   for value in report.get("node_report_sha256", {}).values())
            or not math.isfinite(float(
                report.get("completed_unix_max", math.nan)
            ))
            or report.get("excluded_work") != ["full_sector_ti", "wmc"]):
        raise ValueError("diagnostic runtime consensus identity/status mismatch")
    methods = list(all_methods())
    defects = list(defect_methods())
    tiers = list(resource_tiers().keys())
    capacities = node_capacity()
    capacity = sum(capacities[node] for node in EXECUTION_NODES)
    projection_fields = {
        "resource_tier", "eligible_methods", "per_node",
        "trajectory_seconds_m8", "bias_tuning_seconds_m8",
        "projected_core_seconds", "projected_screen_wall_seconds",
        "safety_factor", "execution_nodes", "execution_capacity", "pass",
    }
    node_fields = {
        "projected_screen_wall_seconds", "eligible_methods", "pass",
    }
    projections = report.get("projections")
    if (not isinstance(projections, list)
            or [value.get("resource_tier") for value in projections
                if isinstance(value, dict)] != tiers
            or len(projections) != len(tiers)):
        raise ValueError("diagnostic runtime consensus tier set is malformed")
    passing = []
    for projection in projections:
        if (set(projection) != projection_fields
                or set(projection.get("trajectory_seconds_m8", {}))
                != set(methods)
                or set(projection.get("bias_tuning_seconds_m8", {}))
                != set(defects)
                or set(projection.get("per_node", {}))
                != set(EXPECTED_PREFLIGHT_NODES)
                or projection.get("execution_nodes") != list(EXECUTION_NODES)
                or projection.get("execution_capacity") != capacity
                or projection.get("safety_factor") != SAFETY_FACTOR
                or not isinstance(projection.get("pass"), bool)):
            raise ValueError("diagnostic runtime consensus projection is malformed")
        trajectory = projection["trajectory_seconds_m8"]
        bias = projection["bias_tuning_seconds_m8"]
        numeric = [*trajectory.values(), *bias.values()]
        if any(not math.isfinite(float(value)) or float(value) < 0.0
               for value in numeric):
            raise ValueError("diagnostic runtime consensus timing is invalid")
        node_eligible = []
        for node in EXPECTED_PREFLIGHT_NODES:
            value = projection["per_node"][node]
            eligible = value.get("eligible_methods") if isinstance(value, dict) else None
            if (not isinstance(value, dict) or set(value) != node_fields
                    or not isinstance(eligible, list)
                    or eligible != [method for method in methods
                                    if method in set(eligible)]
                    or len(eligible) != len(set(eligible))
                    or not isinstance(value.get("pass"), bool)
                    or not math.isfinite(float(
                        value.get("projected_screen_wall_seconds", math.nan)
                    ))
                    or float(value["projected_screen_wall_seconds"]) < 0.0):
                raise ValueError(
                    "diagnostic runtime consensus per-node projection is malformed"
                )
            node_eligible.append(set(eligible))
        intersection = set(methods).intersection(*node_eligible)
        expected_eligible = [method for method in methods
                             if method in intersection]
        projected_core = (
            5 * 2 * 16 * sum(float(trajectory[method]) for method in methods)
            + 2.0 * 5 * sum(float(bias[method]) for method in defects)
        )
        projected_wall = SAFETY_FACTOR * projected_core / capacity
        expected_pass = (
            expected_eligible == methods
            and projected_wall <= SCREEN_WINDOW_SECONDS
            and all(float(trajectory[method]) <= TRAJECTORY_LIMIT_SECONDS
                    for method in methods)
        )
        if (projection.get("eligible_methods") != expected_eligible
                or not math.isclose(
                    float(projection.get("projected_core_seconds", math.nan)),
                    projected_core, rel_tol=1e-12, abs_tol=1e-9,
                )
                or not math.isclose(
                    float(projection.get(
                        "projected_screen_wall_seconds", math.nan,
                    )), projected_wall, rel_tol=1e-12, abs_tol=1e-9,
                )
                or projection["pass"] is not expected_pass):
            raise ValueError("diagnostic runtime consensus projection is inconsistent")
        if expected_pass:
            passing.append(projection)
    selected = passing[-1]["resource_tier"] if passing else None
    selected_projection = passing[-1] if passing else None
    if (selected is None
            or report.get("selected_resource_tier") != selected
            or report.get("selected_eligible_methods") != methods
            or not all(value["pass"]
                       for value in selected_projection["per_node"].values())):
        raise ValueError("diagnostic selected runtime tier is not feasible")
    return report


def verify_remote_stage(run_root, raw_root, control_path, ownership_path,
                        deployment_root, schedule_path, runtime_report_path,
                        registry_path, config_path):
    """Verify one complete bias or measurement stage before analysis."""
    run_root = Path(run_root).resolve(strict=True)
    raw_root = Path(raw_root).resolve(strict=True)
    deployment_root = Path(deployment_root).resolve(strict=True)
    control_path = Path(control_path).resolve(strict=True)
    ownership_path = Path(ownership_path).resolve(strict=True)
    schedule_path = Path(schedule_path).resolve(strict=True)
    runtime_report_path = Path(runtime_report_path).resolve(strict=True)
    expected_raw_root = (run_root / "screen_diagnostic/raw").resolve(strict=True)
    control_root = (run_root / "control").resolve(strict=True)
    if raw_root != expected_raw_root:
        raise ValueError("diagnostic raw root is outside the frozen run")
    for name, path in (
            ("control", control_path), ("ownership", ownership_path),
            ("schedule", schedule_path), ("runtime", runtime_report_path)):
        if path.parent != control_root:
            raise ValueError(
                f"diagnostic {name} evidence is outside run_root/control"
            )
    registry = load_registry(registry_path)
    config = load_config(config_path, registry)
    control = json.loads(control_path.read_text(encoding="ascii"))
    ownership = json.loads(ownership_path.read_text(encoding="ascii"))
    validate_control(control, registry, config)
    source_commit = control["source_commit"]
    schedule = validate_schedule(
        schedule_path, registry, config, source_commit,
    )
    control_sha = sha256_file(control_path)
    ownership_sha = sha256_file(ownership_path)
    schedule_file_sha = sha256_file(schedule_path)
    runtime_sha = sha256_file(runtime_report_path)
    validate_runtime_consensus(
        runtime_report_path, source_commit, registry["registry_sha256"],
        config_sha256(config), schedule["archive_sha256"],
        schedule["source_manifest_sha256"],
    )
    tasks = [entry["task"] for entry in control["tasks"]]
    validate_ownership(
        ownership, tasks, source_commit, registry["registry_sha256"],
        config_sha256(config), schedule_file_sha, schedule["schedule_sha256"],
        control_sha, runtime_sha, control["stage"], control["kind"],
    )
    if (sha256_file(deployment_root / "SOURCE.tar")
            != schedule["archive_sha256"]
            or sha256_file(deployment_root / "SOURCE_MANIFEST.json")
            != schedule["source_manifest_sha256"]):
        raise ValueError("diagnostic evidence deployment hash mismatch")
    entry_by_fingerprint = {
        entry["task_fingerprint"]: entry for entry in control["tasks"]
    }
    if (len(entry_by_fingerprint) != len(control["tasks"])
            or any(fingerprint != sha256_json(entry["task"])
                   for fingerprint, entry in entry_by_fingerprint.items())):
        raise ValueError("diagnostic control task fingerprints are malformed")
    expected_by_node = {
        node: {
            fingerprint for fingerprint, owner in ownership["task_owner"].items()
            if owner == node
        }
        for node in EXECUTION_NODES
    }
    stage_key = "bias" if control["kind"] == "diagnostic_defect_bias" else "measurement"
    deadline = float(schedule["deadlines_unix"][stage_key])
    seen = set()
    node_evidence = []
    for node in EXECUTION_NODES:
        marker_root = (
            run_root / "screen_diagnostic/stages" / stage_key / "markers"
            / control_sha[:12] / node
        )
        manifest_root = (
            run_root / "screen_diagnostic/stages" / stage_key
            / "node_manifests" / control_sha[:12] / node
        )
        success_path = marker_root / "SUCCESS"
        status_path = manifest_root / "stage_status.json"
        raw_manifest_path = manifest_root / "raw_manifest.json"
        if (not success_path.is_file() or not status_path.is_file()
                or not raw_manifest_path.is_file()
                or (marker_root / "RUNNING").exists()
                or (marker_root / "FAILED").exists()):
            raise ValueError(f"diagnostic {stage_key} markers incomplete: {node}")
        success = json.loads(success_path.read_text(encoding="ascii"))
        status = json.loads(status_path.read_text(encoding="ascii"))
        raw_manifest = json.loads(raw_manifest_path.read_text(encoding="ascii"))
        completed = float(status.get("completed_unix", math.nan))
        source_identity = raw_manifest.get("source_identity")
        files = raw_manifest.get("files")
        if (success.get("stage_fingerprint") != ownership["stage_fingerprint"]
                or status.get("status") != "SUCCESS"
                or status.get("node") != node
                or status.get("stage_fingerprint")
                != ownership["stage_fingerprint"]
                or status.get("raw_manifest_sha256")
                != sha256_file(raw_manifest_path)
                or not math.isfinite(completed) or completed > deadline
                or raw_manifest.get("raw_manifest_version")
                != RAW_MANIFEST_VERSION
                or raw_manifest.get("contract_version") != CONTRACT_VERSION
                or raw_manifest.get("node") != node
                or raw_manifest.get("stage") != control["stage"]
                or raw_manifest.get("kind") != control["kind"]
                or raw_manifest.get("stage_fingerprint")
                != ownership["stage_fingerprint"]
                or raw_manifest.get("source_commit") != source_commit
                or raw_manifest.get("registry_sha256")
                != registry["registry_sha256"]
                or raw_manifest.get("diagnostic_config_sha256")
                != config_sha256(config)
                or raw_manifest.get("control_sha256") != control_sha
                or raw_manifest.get("ownership_sha256") != ownership_sha
                or raw_manifest.get("schedule_file_sha256") != schedule_file_sha
                or raw_manifest.get("schedule_sha256")
                != schedule["schedule_sha256"]
                or raw_manifest.get("runtime_report_sha256") != runtime_sha
                or not isinstance(source_identity, dict)
                or source_identity.get("mode") != "archive"
                or source_identity.get("source_commit") != source_commit
                or source_identity.get("archive_sha256")
                != schedule["archive_sha256"]
                or source_identity.get("manifest_sha256")
                != schedule["source_manifest_sha256"]
                or not isinstance(files, list)):
            raise ValueError(f"diagnostic {stage_key} evidence invalid: {node}")
        fingerprints = {value.get("task_fingerprint") for value in files}
        if fingerprints != expected_by_node[node] or len(files) != len(fingerprints):
            raise ValueError(f"diagnostic {stage_key} raw task set mismatch: {node}")
        for value in files:
            fingerprint = value["task_fingerprint"]
            if fingerprint in seen:
                raise ValueError("diagnostic task appears on multiple nodes")
            seen.add(fingerprint)
            entry = entry_by_fingerprint[fingerprint]
            if value.get("path") != entry["output_relpath"]:
                raise ValueError("diagnostic raw path differs from control")
            raw_path = raw_root / value["path"]
            if (not raw_path.is_file()
                    or sha256_file(raw_path) != value.get("sha256")):
                raise ValueError("diagnostic raw file hash mismatch")
        node_evidence.append({
            "node": node, "completed_unix": completed,
            "success_sha256": sha256_file(success_path),
            "status_sha256": sha256_file(status_path),
            "raw_manifest_sha256": sha256_file(raw_manifest_path),
        })
    if seen != set(entry_by_fingerprint):
        raise ValueError("diagnostic remote evidence is incomplete")
    expected_paths = {entry["output_relpath"] for entry in control["tasks"]}
    subdirectory = "bias" if stage_key == "bias" else "trajectories"
    actual_paths = {
        path.relative_to(raw_root).as_posix()
        for path in (raw_root / subdirectory).glob("*.npz")
    }
    if actual_paths != expected_paths:
        raise ValueError("diagnostic raw directory has missing or extra files")
    return {
        "contract_version": CONTRACT_VERSION,
        "stage": stage_key,
        "source_commit": source_commit,
        "archive_sha256": schedule["archive_sha256"],
        "source_manifest_sha256": schedule["source_manifest_sha256"],
        "registry_sha256": registry["registry_sha256"],
        "diagnostic_config_sha256": config_sha256(config),
        "control_sha256": control_sha,
        "ownership_sha256": ownership_sha,
        "schedule_file_sha256": schedule_file_sha,
        "schedule_sha256": schedule["schedule_sha256"],
        "runtime_report_sha256": runtime_sha,
        "stage_fingerprint": ownership["stage_fingerprint"],
        "raw_count": len(seen),
        "completed_unix_max": max(
            value["completed_unix"] for value in node_evidence
        ),
        "nodes": node_evidence,
    }
