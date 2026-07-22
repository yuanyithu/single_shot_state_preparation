"""Fail-closed local entry points for the immutable HGP screen run.

This module deliberately contains no sampler mathematics.  The small adapter
section is the only place coupled to ``q0_hgp_screen``; the rest handles source
identity, three-node consensus, ownership, exclusive raw files, and replay.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from typing import Mapping

import numpy as np


CONTRACT_VERSION = "exp102.q0_hgp_global.screen.v1"
SCHEDULE_VERSION = "exp102.q0_hgp_global.screen.schedule.v2"
CLOCK_AUTHORITY_VERSION = "exp102.q0_hgp.nd0_boottime.v1"
PREFLIGHT_NODE_VERSION = "exp102.q0_hgp_global.screen.preflight_node.v1"
PREFLIGHT_VERSION = "exp102.q0_hgp_global.screen.preflight.v1"
ARTIFACT_MANIFEST_VERSION = "exp102.q0_hgp_global.screen.artifact_manifest.v1"
NODE_RAW_VERSION = "exp102.q0_hgp_global.screen.node_raw.v1"
DECISION_VERSION = "exp102.q0_hgp_global.screen.decision.v1"
TERMINAL_PACKAGE_VERSION = "exp102.q0_hgp_global.screen.terminal_package.v1"
EXPECTED_PREFLIGHT_NODES = ("nd-1", "nd-2", "nd-3")
ARTIFACT_BUILDER_NODE = "nd-1"
RESOURCE_TIER_ORDER = ("T1", "T2", "T3")
DEFAULT_REGISTRY = "data/expander_code/exp102/registry/registry.json"
DEFAULT_CONFIG = (
    "data/expander_code/exp102/config/q0_hgp_global.screen.v1.json"
)
SHA1_RE = re.compile(r"[0-9a-f]{40}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
RUN_ID_RE = re.compile(r"[A-Za-z0-9._-]+")
BOOT_ID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12}"
)
NANOSECONDS_PER_SECOND = 1_000_000_000
MAX_CLOCK_CAPTURE_SPAN_NS = NANOSECONDS_PER_SECOND


# Pipeline adapter. Keep all API name and argument assumptions in this block.
def _pipeline():
    return importlib.import_module(
        "data.expander_code.exp102.exp102_pipeline.q0_hgp_screen"
    )


def _load_registry(path):
    from data.expander_code.exp102.exp102_pipeline.registry import load_registry

    return load_registry(path)


def _load_config(path, registry=None):
    return _pipeline().load_hgp_screen_config(path, registry=registry)


def _build_map_artifacts(registry_path, config_path, source_commit,
                         archive_sha256, source_manifest_sha256,
                         artifact_root):
    return _pipeline().build_hgp_map_artifacts(
        registry_path, config_path, source_commit, archive_sha256,
        source_manifest_sha256, artifact_root,
    )


def _load_map_artifact_descriptors(
        registry_path, config_path, source_commit, archive_sha256,
        source_manifest_sha256, artifact_root):
    return _pipeline().load_hgp_map_artifact_descriptors(
        registry_path, config_path, source_commit, archive_sha256,
        source_manifest_sha256, artifact_root,
    )


def _build_manifest(registry_path, config_path, source_commit, archive_sha256,
                    source_manifest_sha256, resource_tier, artifact_root,
                    output_path):
    return _pipeline().build_hgp_screen_manifest(
        registry_path, config_path, source_commit, archive_sha256,
        source_manifest_sha256, resource_tier, artifact_root, output_path,
    )


def _validate_manifest(manifest, registry, config, artifact_root):
    return _pipeline().validate_hgp_screen_manifest(
        manifest, registry, config, artifact_root,
    )


def _run_task(registry_path, config_path, source_commit, archive_sha256,
              source_manifest_sha256, task, artifact_root, output_path):
    return _pipeline().run_hgp_screen_task(
        registry_path, config_path, source_commit, archive_sha256,
        source_manifest_sha256, task, artifact_root, output_path,
    )


def _validate_raw(path, registry, config, source_commit, archive_sha256,
                  source_manifest_sha256, artifact_root):
    return _pipeline().validate_hgp_screen_raw(
        path, registry, config, source_commit, archive_sha256,
        source_manifest_sha256, artifact_root,
    )


def _run_is(registry_path, config_path, source_commit, archive_sha256,
            source_manifest_sha256, cell, artifact_root, output_path,
            seed_namespace=None):
    if seed_namespace is None:
        seed_namespace = _pipeline().HGP_SCREEN_IS_ROOT
    return _pipeline().run_hgp_map_is_diagnostic(
        registry_path, config_path, source_commit, archive_sha256,
        source_manifest_sha256, cell, artifact_root, output_path,
        seed_namespace=seed_namespace,
    )


def _validate_is(path, registry_path, config_path, source_commit,
                 archive_sha256, source_manifest_sha256, cell,
                 artifact_root, seed_namespace=None):
    if seed_namespace is None:
        seed_namespace = _pipeline().HGP_SCREEN_IS_ROOT
    return _pipeline().validate_hgp_map_is_diagnostic(
        path, registry_path, config_path, source_commit, archive_sha256,
        source_manifest_sha256, cell, artifact_root,
        seed_namespace=seed_namespace,
    )


def _analyze(raw_root, manifest_path, registry_path, config_path, output_path,
             artifact_root, num_workers):
    return _pipeline().analyze_hgp_screen(
        raw_root, manifest_path, registry_path, config_path, artifact_root,
        output_path, num_workers=num_workers,
    )


def _preflight_digest(registry_path, config_path, source_commit,
                      archive_sha256, source_manifest_sha256, artifact_root):
    return _pipeline().hgp_screen_preflight_digest(
        registry_path, config_path, source_commit, archive_sha256,
        source_manifest_sha256, artifact_root,
    )


def _benchmark(registry_path, config_path, source_commit, archive_sha256,
               source_manifest_sha256, artifact_root):
    return _pipeline().benchmark_hgp_screen(
        registry_path, config_path, source_commit, archive_sha256,
        source_manifest_sha256, artifact_root,
    )


def _canonical_json(value):
    return json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    )


def _jsonable(value):
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        return _jsonable(value.item())
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"value is not canonical-JSON serializable: {type(value)!r}")


def _sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def _sha256_json(value):
    return _sha256_bytes(_canonical_json(value).encode("ascii"))


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_exclusive_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (_canonical_json(value) + "\n").encode("ascii")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return _sha256_bytes(payload)


def _write_exclusive_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="ascii"))


def _verify_commit(source_commit):
    if SHA1_RE.fullmatch(str(source_commit)) is None:
        raise ValueError("source commit must be a full lowercase Git SHA")


def _verify_source(source_commit):
    from data.expander_code.exp102.exp102_pipeline.io import verify_source_identity

    _verify_commit(source_commit)
    return verify_source_identity(Path.cwd(), source_commit)


def _verify_provenance(source_commit, archive_sha256,
                       source_manifest_sha256):
    """Bind every stage to the archive verified before project code runs."""
    _verify_commit(source_commit)
    for name, value in (
            ("archive_sha256", archive_sha256),
            ("source_manifest_sha256", source_manifest_sha256)):
        if SHA256_RE.fullmatch(str(value)) is None:
            raise ValueError(f"{name} must be a full lowercase SHA256")
    identity = _verify_source(source_commit)
    if (identity.get("mode") != "archive"
            or identity.get("archive_sha256") != archive_sha256
            or identity.get("manifest_sha256")
            != source_manifest_sha256):
        raise ValueError(
            "HGP remote workflow requires the verified source archive"
        )
    return identity


def _require_node(expected):
    actual = platform.node().split(".", 1)[0]
    if actual != expected:
        raise ValueError(
            f"HGP stage ownership requires host {expected}, found {actual}"
        )


def _config_sha(path):
    return _sha256_file(path)


def _registry_sha(path):
    return _sha256_file(path)


def _validate_clock_authority(value):
    if not isinstance(value, Mapping) or set(value) != {
            "clock_authority_version", "clock_authority_node",
            "clock_authority_boot_id", "boottime_before_ns",
            "authority_unix_ns", "boottime_after_ns"}:
        raise ValueError("HGP clock authority schema is invalid")
    integer_fields = (
        "boottime_before_ns", "authority_unix_ns", "boottime_after_ns",
    )
    if (value.get("clock_authority_version") != CLOCK_AUTHORITY_VERSION
            or value.get("clock_authority_node") != "nd-0"
            or BOOT_ID_RE.fullmatch(str(
                value.get("clock_authority_boot_id", ""),
            )) is None
            or any(isinstance(value.get(name), bool)
                   or not isinstance(value.get(name), int)
                   or value[name] <= 0 for name in integer_fields)
            or value["boottime_after_ns"] < value["boottime_before_ns"]
            or value["boottime_after_ns"] - value["boottime_before_ns"]
            > MAX_CLOCK_CAPTURE_SPAN_NS):
        raise ValueError("HGP clock authority identity is invalid")
    return dict(value)


def _schedule_deadlines(config, started_unix, started_boottime_ns):
    frozen = config.get("schedule", {})
    expected = {
        "preflight_deadline_hour": 6,
        "control_freeze_deadline_hour": 8,
        "screen_deadline_hour": 22,
        "analysis_deadline_hour": 24,
        "wall_limit_hours": 24,
    }
    if frozen != expected:
        raise ValueError("HGP screen schedule protocol changed")
    result = {}
    for name, hours in (
            ("preflight", 6), ("control_freeze", 8),
            ("screen", 22), ("analysis", 24)):
        seconds = hours * 3600
        result[f"{name}_deadline_unix"] = started_unix + seconds
        result[f"{name}_deadline_boottime_ns"] = (
            started_boottime_ns + seconds * NANOSECONDS_PER_SECOND
        )
    return result


def _build_schedule(args):
    _require_node(ARTIFACT_BUILDER_NODE)
    source_identity = _verify_provenance(
        args.source_commit, args.archive_sha256,
        args.source_manifest_sha256,
    )
    if RUN_ID_RE.fullmatch(str(args.run_id)) is None:
        raise ValueError("HGP schedule run ID is invalid")
    registry_path = Path(args.registry).resolve(strict=True)
    config_path = Path(args.config).resolve(strict=True)
    registry = _load_registry(registry_path)
    config = _load_config(config_path, registry)
    try:
        clock_authority = _validate_clock_authority(
            json.loads(args.clock_authority_json)
        )
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("HGP clock authority JSON is invalid") from exc
    started_unix = (
        clock_authority["authority_unix_ns"] // NANOSECONDS_PER_SECOND
    )
    started_boottime_ns = clock_authority["boottime_before_ns"]
    identity = {
        "schedule_version": SCHEDULE_VERSION,
        "contract_version": CONTRACT_VERSION,
        "run_id": args.run_id,
        "source_commit": args.source_commit,
        "archive_sha256": args.archive_sha256,
        "source_manifest_sha256": args.source_manifest_sha256,
        "source_identity": source_identity,
        "registry_file_sha256": _registry_sha(registry_path),
        "config_file_sha256": _config_sha(config_path),
        "clock_authority": clock_authority,
        "started_unix": started_unix,
        "started_boottime_ns": started_boottime_ns,
        **_schedule_deadlines(config, started_unix, started_boottime_ns),
    }
    schedule = {**identity, "schedule_sha256": _sha256_json(identity)}
    if _verify_source(args.source_commit) != source_identity:
        raise RuntimeError("schedule construction changed the verified source")
    _write_exclusive_json(args.output, schedule)
    print(_canonical_json({
        "status": "SUCCESS", "run_id": args.run_id,
        "schedule": str(Path(args.output).resolve()),
        "schedule_sha256": schedule["schedule_sha256"],
    }))


def _validate_schedule(
        path, registry_path, config_path, source_commit, archive_sha256,
        source_manifest_sha256, config):
    schedule_path = Path(path).resolve(strict=True)
    schedule = _read_json(schedule_path)
    clock_authority = _validate_clock_authority(
        schedule.get("clock_authority")
    )
    started = schedule.get("started_unix")
    started_boottime_ns = schedule.get("started_boottime_ns")
    if (isinstance(started, bool) or not isinstance(started, int)
            or started <= 0
            or started != clock_authority["authority_unix_ns"]
            // NANOSECONDS_PER_SECOND
            or isinstance(started_boottime_ns, bool)
            or not isinstance(started_boottime_ns, int)
            or started_boottime_ns != clock_authority["boottime_before_ns"]
            or RUN_ID_RE.fullmatch(str(schedule.get("run_id", ""))) is None):
        raise ValueError("HGP schedule start identity is invalid")
    identity = {
        "schedule_version": SCHEDULE_VERSION,
        "contract_version": CONTRACT_VERSION,
        "run_id": schedule["run_id"],
        "source_commit": source_commit,
        "archive_sha256": archive_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "source_identity": schedule.get("source_identity"),
        "registry_file_sha256": _registry_sha(registry_path),
        "config_file_sha256": _config_sha(config_path),
        "clock_authority": clock_authority,
        "started_unix": started,
        "started_boottime_ns": started_boottime_ns,
        **_schedule_deadlines(config, started, started_boottime_ns),
    }
    expected = {**identity, "schedule_sha256": _sha256_json(identity)}
    if schedule != expected:
        raise ValueError("HGP schedule is noncanonical or stale")
    local_identity = _verify_provenance(
        source_commit, archive_sha256, source_manifest_sha256,
    )
    if (local_identity.get("mode") == "archive"
            and schedule["source_identity"] != local_identity):
        raise ValueError("HGP schedule source archive differs from this stage")
    return schedule, schedule_path


def _enforce_deadline(schedule, field):
    # Compute-node epochs are unsynchronized and never grant deadline
    # authority. The nd-0 orchestrator accepts markers against CLOCK_BOOTTIME.
    boottime_field = field.removesuffix("_unix") + "_boottime_ns"
    deadline = schedule.get(boottime_field)
    if isinstance(deadline, bool) or not isinstance(deadline, int):
        raise ValueError("HGP schedule boottime deadline is invalid")


def _forbidden_runtime_outcome(value):
    forbidden = {
        "q_top", "posterior_purity", "posterior_mass_on_planted_class",
        "map_success_probability", "d2_norm", "label", "labels", "weight",
        "weights", "characters", "measurement_weights", "physical_result",
        "sampler_pass", "sampler_fail", "sampling_pass", "sampling_fail",
    }
    if isinstance(value, Mapping):
        return any(
            str(key).lower() in forbidden or _forbidden_runtime_outcome(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_forbidden_runtime_outcome(item) for item in value)
    return False


def _validate_auxiliary_seed_catalog(runtime):
    catalog = runtime.get("auxiliary_seed_catalog")
    if (not isinstance(catalog, list) or len(catalog) != 8
            or runtime.get("auxiliary_seed_catalog_sha256")
            != _sha256_json(catalog)):
        raise ValueError("runtime auxiliary seed catalog is invalid")
    purposes = [value.get("purpose") for value in catalog]
    if purposes.count("runtime_warmup") != 3:
        raise ValueError("runtime warmup seed catalog changed")
    if purposes.count("runtime_timed") != 3:
        raise ValueError("runtime timed seed catalog changed")
    if purposes.count("importance_sampling_runtime") != 2:
        raise ValueError("runtime IS seed catalog changed")
    pipeline = _pipeline()
    formal_namespaces = {
        pipeline.HGP_SCREEN_HP_TRAJECTORY_ROOT,
        pipeline.HGP_SCREEN_MAP_TRAJECTORY_ROOT,
        pipeline.HGP_SCREEN_IS_ROOT,
    }
    rows = set()
    for value in catalog:
        if not isinstance(value, Mapping):
            raise ValueError("runtime auxiliary seed row is invalid")
        if value["purpose"] in {"runtime_warmup", "runtime_timed"}:
            identity = value.get("seed_identity")
            expected_namespace = (
                pipeline.HGP_SCREEN_RUNTIME_WARMUP_ROOT
                if value["purpose"] == "runtime_warmup"
                else pipeline.HGP_SCREEN_RUNTIME_TIMED_ROOT
            )
            if (not isinstance(identity, Mapping)
                    or identity.get("trajectory_namespace")
                    != expected_namespace):
                raise ValueError("runtime trajectory seed namespace changed")
        else:
            if (value.get("seed_namespace")
                    != pipeline.HGP_SCREEN_RUNTIME_IS_ROOT
                    or isinstance(value.get("seed"), bool)
                    or not isinstance(value.get("seed"), int)):
                raise ValueError("runtime IS seed namespace changed")
        if any(namespace in _canonical_json(value)
               for namespace in formal_namespaces):
            raise ValueError("runtime seed overlaps a formal namespace")
        rows.add(_canonical_json(value))
    if len(rows) != len(catalog):
        raise ValueError("runtime auxiliary seed catalog contains duplicates")
    return catalog


def _resource_tiers(config):
    tiers = config.get("resource_tiers")
    if not isinstance(tiers, Mapping):
        tiers = config.get("runtime", {}).get("resource_tiers")
    if not isinstance(tiers, Mapping):
        raise ValueError("HGP screen config does not define resource_tiers")
    if not all(tier in tiers for tier in RESOURCE_TIER_ORDER):
        raise ValueError("HGP screen requires frozen T1/T2/T3 resource tiers")
    return tiers


def _screen_budget_seconds(config):
    candidates = (
        config.get("screen_budget_seconds"),
        config.get("resource_selection", {}).get("screen_window_seconds"),
        config.get("resource_selection", {}).get("screen_budget_seconds"),
        config.get("runtime", {}).get("screen_budget_seconds"),
        config.get("runtime_gate", {}).get("screen_budget_seconds"),
        config.get("schedule", {}).get("screen_budget_seconds"),
    )
    for candidate in candidates:
        if candidate is not None:
            value = float(candidate)
            if math.isfinite(value) and value > 0.0:
                return value
            raise ValueError("screen_budget_seconds must be finite and positive")
    raise ValueError("HGP screen config must freeze screen_budget_seconds")


def _runtime_accounting(config):
    selection = config.get("resource_selection", {})
    values = {
        "full_sampler_passes_per_task": selection.get(
            "full_sampler_passes_per_task",
        ),
        "full_is_passes_per_cell": selection.get("full_is_passes_per_cell"),
        "safety_factor": selection.get("safety_factor"),
        "staging_validation_mode": selection.get("staging_validation_mode"),
        "final_analysis_validation_mode": selection.get(
            "final_analysis_validation_mode",
        ),
    }
    if (values["full_sampler_passes_per_task"] != 2
            or values["full_is_passes_per_cell"] != 2
            or float(values["safety_factor"] or 0.0) != 2.0
            or values["staging_validation_mode"] != "structure_only"
            or values["final_analysis_validation_mode"]
            != "full_bit_exact_replay"):
        raise ValueError("HGP runtime accounting protocol changed")
    return values


def _projection_rows(runtime):
    for key in ("projections", "tier_projections", "tiers"):
        rows = runtime.get(key) if isinstance(runtime, Mapping) else None
        if isinstance(rows, list):
            return rows
        if isinstance(rows, Mapping):
            return [dict(value, resource_tier=tier)
                    for tier, value in rows.items()]
    mapping = runtime.get("projected_wall_seconds_by_tier", {})
    if isinstance(mapping, Mapping) and mapping:
        return [
            {"resource_tier": tier, "projected_screen_wall_seconds": value}
            for tier, value in mapping.items()
        ]
    raise ValueError(
        "benchmark must expose projections/tier_projections or "
        "projected_wall_seconds_by_tier"
    )


def _tier_projection(runtime, tier):
    matches = [
        row for row in _projection_rows(runtime)
        if row.get("resource_tier", row.get("tier")) == tier
    ]
    if len(matches) != 1:
        raise ValueError(f"benchmark must contain exactly one {tier} projection")
    row = matches[0]
    seconds = None
    for key in (
        "projected_screen_wall_seconds_with_safety_factor",
        "projected_complete_schedule_seconds",
        "projected_screen_wall_seconds", "projected_wall_seconds",
        "wall_seconds_with_safety_factor", "screen_wall_seconds",
        "wall_seconds",
    ):
        if key in row:
            seconds = float(row[key])
            break
    if seconds is None or not math.isfinite(seconds) or seconds < 0.0:
        raise ValueError(f"benchmark {tier} projection has invalid wall time")
    gate_key = "eligible" if "eligible" in row else "pass"
    if gate_key in row and not isinstance(row[gate_key], bool):
        raise ValueError(f"benchmark {tier} pass flag is not boolean")
    return seconds, row.get(gate_key, True), _jsonable(row)


def _lpt_makespan(durations, capacity):
    durations = sorted((float(value) for value in durations), reverse=True)
    if (not durations or isinstance(capacity, bool) or int(capacity) <= 0
            or any(not math.isfinite(value) or value < 0.0
                   for value in durations)):
        if durations:
            raise ValueError("invalid HGP runtime workload")
        return 0.0
    lanes = [0.0] * min(int(capacity), len(durations))
    for duration in durations:
        lane = min(range(len(lanes)), key=lanes.__getitem__)
        lanes[lane] += duration
    return float(max(lanes))


def _expected_owner_counts(config):
    nodes = tuple(config["execution"]["execution_nodes"])
    result = {
        node: {method: 0 for method in config["method_panels"]}
        for node in nodes
    }
    count = int(config["trajectory_count_per_init_family"])
    for method, panels in config["method_panels"].items():
        cells = sum(len(config["panels"][name]["cells"]) for name in panels)
        for _cell in range(cells):
            for _family in config["init_families"]:
                for trajectory in range(count):
                    result[nodes[trajectory % len(nodes)]][method] += 1
    return result


def _importance_cells(config):
    return [
        cell
        for panel in config["importance_sampling"]["panels"]
        for cell in config["panels"][panel]["cells"]
    ]


def _validate_projection_accounting(
        row, accounting, config, runtime_context, artifact_manifest):
    sampler_passes = accounting["full_sampler_passes_per_task"]
    is_passes = accounting["full_is_passes_per_cell"]
    safety = float(accounting["safety_factor"])
    nodes = row.get("per_node_generation_workload")
    analysis = row.get("analysis_workload")
    frozen_analysis = config.get("execution", {}).get("analysis")
    tier = row.get("resource_tier")
    timings = runtime_context.get("timings")
    is_seconds = runtime_context.get("is_seconds_by_cell")
    b_timings = runtime_context.get("b_analysis_timings")
    if tier not in RESOURCE_TIER_ORDER:
        raise ValueError("runtime projection resource tier is invalid")
    if (row.get("full_sampler_passes_per_task") != sampler_passes
            or row.get("full_is_passes_per_cell") != is_passes
            or float(row.get("safety_factor", -1.0)) != safety
            or not isinstance(nodes, Mapping) or not nodes
            or set(nodes) != set(config["execution"]["execution_nodes"])
            or not isinstance(analysis, Mapping)
            or not isinstance(frozen_analysis, Mapping)
            or not isinstance(timings, Mapping)
            or not isinstance(is_seconds, Mapping)
            or not isinstance(b_timings, Mapping)
            or row.get("artifact_generation_mode")
            != "single_serial_stage"):
        raise ValueError("runtime projection omitted frozen replay accounting")
    expected_b_timing_fields = {
        "benchmark_measurement_rounds", "trace_benchmark_seconds",
        "trace_seconds_per_round", "family_benchmark_seconds",
        "comparison_benchmark_seconds",
    }
    if set(b_timings) != expected_b_timing_fields:
        raise ValueError("runtime B-analysis benchmark was omitted")
    benchmark_rounds = int(b_timings["benchmark_measurement_rounds"])
    expected_benchmark_rounds = max(
        int(value["measurement"]) for value in config["resource_tiers"].values()
    )
    b_values = [
        float(b_timings["trace_benchmark_seconds"]),
        float(b_timings["trace_seconds_per_round"]),
        float(b_timings["family_benchmark_seconds"]),
        float(b_timings["comparison_benchmark_seconds"]),
    ]
    if (benchmark_rounds != expected_benchmark_rounds
            or any(not math.isfinite(value) or value < 0.0 for value in b_values)
            or not math.isclose(
                float(b_timings["trace_seconds_per_round"]),
                float(b_timings["trace_benchmark_seconds"]) / benchmark_rounds,
                rel_tol=1e-12, abs_tol=1e-12,
            )):
        raise ValueError("runtime B-analysis benchmark arithmetic changed")
    methods = tuple(config["method_panels"])
    if set(timings) != set(methods):
        raise ValueError("runtime benchmark method timing set changed")
    resource = config["resource_tiers"][tier]
    total_steps = int(resource["burn"]) + int(resource["measurement"])
    projected = {}
    for method in methods:
        timing = timings[method]
        benchmark_seconds = float(timing["benchmark_seconds"])
        benchmark_steps = int(timing["benchmark_steps"])
        seconds_per_step = float(timing["seconds_per_step"])
        setup = float(timing["setup_seconds_per_task"])
        if (benchmark_steps <= 0 or min(
                benchmark_seconds, seconds_per_step, setup) < 0.0
                or not all(math.isfinite(value) for value in (
                    benchmark_seconds, seconds_per_step, setup,
                ))
                or not math.isclose(
                    seconds_per_step, benchmark_seconds / benchmark_steps,
                    rel_tol=1e-12, abs_tol=1e-12,
                )):
            raise ValueError("runtime benchmark timing arithmetic changed")
        projected[method] = seconds_per_step * total_steps + setup
    stored_projected = row.get("per_method_projected_seconds")
    if (not isinstance(stored_projected, Mapping)
            or set(stored_projected) != set(methods)
            or any(not math.isclose(
                float(stored_projected[method]), projected[method],
                rel_tol=1e-12, abs_tol=1e-9,
            ) for method in methods)):
        raise ValueError("runtime projected method durations changed")
    worst_trajectory = max(projected.values())
    if not math.isclose(
            float(row.get("projected_worst_trajectory_seconds", -1.0)),
            worst_trajectory, rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError("runtime worst trajectory projection changed")

    owner_counts = _expected_owner_counts(config)
    if runtime_context.get("owner_task_counts") != owner_counts:
        raise ValueError("runtime owner task counts changed")
    map_cells = _importance_cells(config)
    expected_is_keys = {_sha256_json(cell) for cell in map_cells}
    if (set(is_seconds) != expected_is_keys
            or any(not math.isfinite(float(value)) or float(value) < 0.0
                   for value in is_seconds.values())):
        raise ValueError("runtime IS timing set changed")
    artifact_seconds = float(sum(
        float(value["generation_wall_seconds"])
        for value in artifact_manifest["map_artifacts"]
    ))
    if (not math.isfinite(artifact_seconds) or artifact_seconds < 0.0
            or not math.isclose(
                float(runtime_context["artifact_generation_wall_seconds"]),
                artifact_seconds, rel_tol=1e-12, abs_tol=1e-9,
            )
            or not math.isclose(
                float(row["artifact_generation_wall_seconds"]),
                artifact_seconds, rel_tol=1e-12, abs_tol=1e-9,
            )):
        raise ValueError("runtime artifact generation timing changed")

    generation = []
    execution_nodes = tuple(config["execution"]["execution_nodes"])
    for node_index, node in enumerate(execution_nodes):
        value = nodes[node]
        if (value.get("sampler_generation_passes_per_task") != 1
                or value.get("is_generation_passes_per_cell") != 1
                or int(value.get("capacity", -1))
                != int(config["execution"]["capacities"][node])):
            raise ValueError(f"runtime generation passes changed for {node}")
        expected_counts = owner_counts[node]
        expected_is_cells = [
            cell for index, cell in enumerate(map_cells)
            if index % len(execution_nodes) == node_index
        ]
        expected_is_fingerprints = [_sha256_json(cell)
                                    for cell in expected_is_cells]
        if (value.get("owned_task_counts") != expected_counts
                or value.get("owned_is_cell_fingerprints")
                != expected_is_fingerprints):
            raise ValueError(f"runtime frozen ownership changed for {node}")
        sampler_lpt = _lpt_makespan([
            projected[method]
            for method in methods
            for _ in range(expected_counts[method])
        ], int(config["execution"]["capacities"][node]))
        is_lpt = _lpt_makespan([
            float(is_seconds[fingerprint])
            for fingerprint in expected_is_fingerprints
        ], int(config["execution"]["capacities"][node]))
        if (not math.isclose(
                float(value["sampler_generation_lpt_seconds"]), sampler_lpt,
                rel_tol=1e-12, abs_tol=1e-9)
                or not math.isclose(
                    float(value["is_generation_lpt_seconds"]), is_lpt,
                    rel_tol=1e-12, abs_tol=1e-9,
                )):
            raise ValueError(f"runtime LPT projection changed for {node}")
        expected = (
            sampler_lpt + is_lpt
        )
        stored = float(value["projected_generation_wall_seconds"])
        if (min(expected, stored) < 0.0
                or not math.isclose(
                    stored, expected, rel_tol=1e-12, abs_tol=1e-9,
                )):
            raise ValueError(f"runtime accounting arithmetic changed for {node}")
        generation.append(stored)
    if ({
            "node": analysis.get("node"),
            "capacity": analysis.get("capacity"),
            "num_workers": analysis.get("num_workers"),
            } != dict(frozen_analysis)
            or analysis.get("sampler_replay_mode") != "process_pool_lpt"
            or analysis.get("sampler_replay_passes_per_task") != 1
            or analysis.get("b_trace_postprocess_included_in_sampler_replay")
            is not True
            or analysis.get("is_replay_mode") != "serial"
            or analysis.get("is_replay_passes_per_cell") != 1):
        raise ValueError("runtime analysis placement or replay mode changed")
    configured_is_cells = len(map_cells)
    if (sampler_passes != 1 + int(
            analysis["sampler_replay_passes_per_task"])
            or is_passes != 1 + int(
                analysis["is_replay_passes_per_cell"])
            or int(analysis.get("sampler_task_count", -1))
            != int(config["task_counts"]["total_measurement"])
            or int(analysis.get("is_cell_count", -1))
            != configured_is_cells):
        raise ValueError("runtime full-pass or replay task count changed")
    b_trace_seconds = (
        float(b_timings["trace_seconds_per_round"])
        * int(resource["measurement"])
    )
    all_sampler_durations = [
        projected[method] + b_trace_seconds
        for node in execution_nodes
        for method in methods
        for _ in range(owner_counts[node][method])
    ]
    analysis_sampler_lpt = _lpt_makespan(
        all_sampler_durations, int(frozen_analysis["capacity"]),
    )
    analysis_is_seconds = float(sum(float(value)
                                    for value in is_seconds.values()))
    cell_methods = sum(
        sum(len(config["panels"][panel]["cells"]) for panel in panels)
        for panels in config["method_panels"].values()
    )
    b_family_count = cell_methods * len(config["init_families"])
    b_cross_count = (
        2 * sum(
            len(config["panels"][panel]["cells"])
            for panel in config["selection"]["cross_mechanism_panels"]
        ) * len(config["init_families"])
    )
    b_comparison_count = cell_methods + b_cross_count
    diagnostic_scale = (
        int(resource["measurement"])
        * math.log2(max(int(resource["measurement"]), 2))
        / (benchmark_rounds * math.log2(max(benchmark_rounds, 2)))
    )
    b_family_seconds = (
        float(b_timings["family_benchmark_seconds"])
        * diagnostic_scale * b_family_count
    )
    b_comparison_seconds = (
        float(b_timings["comparison_benchmark_seconds"])
        * diagnostic_scale * b_comparison_count
    )
    b_diagnostic_seconds = b_family_seconds + b_comparison_seconds
    if (not math.isclose(
            float(analysis["sampler_replay_lpt_seconds"]),
            analysis_sampler_lpt, rel_tol=1e-12, abs_tol=1e-9)
            or not math.isclose(
                float(analysis["is_replay_seconds"]), analysis_is_seconds,
                rel_tol=1e-12, abs_tol=1e-9,
            ) or not math.isclose(
                float(analysis.get("b_trace_seconds_per_task", -1.0)),
                b_trace_seconds, rel_tol=1e-12, abs_tol=1e-9,
            ) or analysis.get("b_statistical_diagnostics_mode")
            != "single_node_serial"
            or int(analysis.get("b_family_count", -1)) != b_family_count
            or int(analysis.get("b_comparison_count", -1))
            != b_comparison_count
            or not math.isclose(
                float(analysis.get("b_diagnostic_scale", -1.0)),
                diagnostic_scale, rel_tol=1e-12, abs_tol=1e-12,
            ) or not math.isclose(
                float(analysis.get("b_family_diagnostics_seconds", -1.0)),
                b_family_seconds, rel_tol=1e-12, abs_tol=1e-9,
            ) or not math.isclose(
                float(analysis.get("b_comparison_diagnostics_seconds", -1.0)),
                b_comparison_seconds, rel_tol=1e-12, abs_tol=1e-9,
            ) or not math.isclose(
                float(analysis.get("b_statistical_diagnostics_seconds", -1.0)),
                b_diagnostic_seconds, rel_tol=1e-12, abs_tol=1e-9,
            )):
        raise ValueError("runtime analysis replay LPT changed")
    analysis_expected = (
        analysis_sampler_lpt + analysis_is_seconds + b_diagnostic_seconds
    )
    if (analysis_expected < 0.0
            or not math.isclose(
                float(analysis["projected_analysis_wall_seconds"]),
                analysis_expected, rel_tol=1e-12, abs_tol=1e-9,
            )):
        raise ValueError("runtime analysis arithmetic changed")
    generation_expected = max(generation)
    if not math.isclose(
            float(row["screen_generation_wall_seconds"]),
            generation_expected, rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError("runtime generation critical path changed")
    expected_unsafetied = (
        artifact_seconds + generation_expected + analysis_expected
    )
    if not math.isclose(
            float(row["projected_unsafetied_schedule_seconds"]),
            expected_unsafetied, rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError("runtime unsafetied schedule arithmetic changed")
    expected_schedule = safety * expected_unsafetied
    if not math.isclose(
            float(row["projected_complete_schedule_seconds"]),
            expected_schedule, rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError("runtime safety/pass factors were replaced or double-counted")
    expected_eligible = (
        worst_trajectory
        <= float(config["resource_selection"]["max_trajectory_seconds"])
        and expected_schedule <= _screen_budget_seconds(config)
    )
    if row.get("eligible") is not bool(expected_eligible):
        raise ValueError("runtime eligibility or per-trajectory gate changed")


def _screen_cells(config):
    panels = config.get("panels", {})
    cells = []
    for panel_name in ("HARD2", "EASY3"):
        panel = panels.get(panel_name, {})
        values = panel.get("cells", [])
        if not isinstance(values, list):
            raise ValueError(f"HGP screen {panel_name} cells must be a list")
        cells.extend(values)
    if not cells:
        raise ValueError("HGP screen config has no cells")
    return cells


def _safe_relative_path(value, *, field):
    relative = Path(str(value))
    if relative.is_absolute() or ".." in relative.parts or not relative.name:
        raise ValueError(f"unsafe HGP screen {field}")
    return relative.as_posix()


def _validate_artifact_file_set(artifact_root, descriptors):
    root = Path(artifact_root).resolve()
    expected = {
        _safe_relative_path(value["artifact_relpath"], field="artifact path")
        for value in descriptors
    }
    if len(expected) != len(descriptors):
        raise ValueError("MAP artifact descriptors contain duplicate paths")
    map_root = root / "map_artifacts"
    if map_root.is_symlink():
        raise ValueError("MAP artifact root cannot be a symlink")
    if map_root.exists() and root not in map_root.resolve().parents:
        raise ValueError("MAP artifact root escaped its immutable directory")
    if map_root.is_dir() and any(path.is_symlink()
                                 for path in map_root.rglob("*")):
        raise ValueError("MAP artifact file set cannot contain symlinks")
    actual = {
        path.relative_to(root).as_posix()
        for path in map_root.rglob("*") if path.is_file()
    } if map_root.is_dir() else set()
    if actual != expected:
        raise ValueError("MAP artifact file set is incomplete or has extras")
    for relative in expected:
        path = (root / relative).resolve(strict=True)
        if root not in path.parents or not path.is_file():
            raise ValueError("MAP artifact file escaped its immutable directory")
    return expected


def _artifact_manifest_identity(
        registry_path, config_path, source_commit, archive_sha256,
        source_manifest_sha256, schedule, schedule_path, descriptors):
    return {
        "manifest_version": ARTIFACT_MANIFEST_VERSION,
        "contract_version": CONTRACT_VERSION,
        "source_commit": source_commit,
        "archive_sha256": archive_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "schedule_sha256": schedule["schedule_sha256"],
        "schedule_file_sha256": _sha256_file(schedule_path),
        "builder_node": ARTIFACT_BUILDER_NODE,
        "registry_file_sha256": _registry_sha(registry_path),
        "config_file_sha256": _config_sha(config_path),
        "artifact_count": len(descriptors),
        "map_artifacts": _jsonable(descriptors),
    }


def _validate_artifact_manifest(
        manifest_path, registry_path, config_path, source_commit,
        archive_sha256, source_manifest_sha256, artifact_root, schedule,
        schedule_path):
    path = Path(manifest_path).resolve(strict=True)
    manifest = _read_json(path)
    descriptors = _load_map_artifact_descriptors(
        registry_path, config_path, source_commit, archive_sha256,
        source_manifest_sha256, artifact_root,
    )
    identity = _artifact_manifest_identity(
        registry_path, config_path, source_commit, archive_sha256,
        source_manifest_sha256, schedule, schedule_path, descriptors,
    )
    expected = {
        **identity, "artifact_manifest_sha256": _sha256_json(identity),
    }
    if manifest != expected:
        raise ValueError("MAP artifact manifest is noncanonical or stale")
    _validate_artifact_file_set(artifact_root, descriptors)
    return manifest, path


def _artifact_cells(config, artifact_manifest):
    by_fingerprint = {
        _sha256_json(cell): cell for cell in _screen_cells(config)
    }
    cells = []
    for descriptor in artifact_manifest["map_artifacts"]:
        fingerprint = descriptor.get("cell_fingerprint")
        if fingerprint not in by_fingerprint:
            raise ValueError("MAP artifact cell is outside the configured panels")
        cells.append(by_fingerprint[fingerprint])
    if len({_canonical_json(cell) for cell in cells}) != len(cells):
        raise ValueError("MAP artifact cells are duplicated")
    return cells


def _build_artifacts_command(args):
    _require_node(ARTIFACT_BUILDER_NODE)
    source_identity = _verify_provenance(
        args.source_commit, args.archive_sha256,
        args.source_manifest_sha256,
    )
    registry_path = Path(args.registry).resolve(strict=True)
    config_path = Path(args.config).resolve(strict=True)
    registry = _load_registry(registry_path)
    config = _load_config(config_path, registry)
    schedule, schedule_path = _validate_schedule(
        args.schedule, registry_path, config_path, args.source_commit,
        args.archive_sha256, args.source_manifest_sha256, config,
    )
    _enforce_deadline(schedule, "preflight_deadline_unix")
    artifact_root = Path(args.artifact_root).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("MAP artifact manifest already exists")
    map_root = artifact_root / "map_artifacts"
    if map_root.exists():
        raise FileExistsError("fresh artifact stage found an existing map_artifacts root")
    artifact_root.mkdir(parents=True, exist_ok=True)
    descriptors = _build_map_artifacts(
        registry_path, config_path, args.source_commit,
        args.archive_sha256, args.source_manifest_sha256, artifact_root,
    )
    replayed = _load_map_artifact_descriptors(
        registry_path, config_path, args.source_commit,
        args.archive_sha256, args.source_manifest_sha256, artifact_root,
    )
    if _jsonable(descriptors) != _jsonable(replayed):
        raise RuntimeError("MAP artifact descriptors changed on immediate replay")
    if not replayed:
        raise RuntimeError("MAP artifact stage produced no frozen artifacts")
    _validate_artifact_file_set(artifact_root, replayed)
    identity = _artifact_manifest_identity(
        registry_path, config_path, args.source_commit,
        args.archive_sha256, args.source_manifest_sha256, schedule,
        schedule_path, replayed,
    )
    manifest = {
        **identity, "artifact_manifest_sha256": _sha256_json(identity),
    }
    _enforce_deadline(schedule, "preflight_deadline_unix")
    if _verify_source(args.source_commit) != source_identity:
        raise RuntimeError("artifact construction changed the verified source")
    _write_exclusive_json(output, manifest)
    _validate_artifact_manifest(
        output, registry_path, config_path, args.source_commit,
        args.archive_sha256, args.source_manifest_sha256, artifact_root,
        schedule, schedule_path,
    )
    print(_canonical_json({
        "status": "SUCCESS", "artifact_count": len(replayed),
        "artifact_manifest": str(output),
        "artifact_manifest_file_sha256": _sha256_file(output),
    }))


def _manifest_records(manifest):
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("HGP screen manifest has no tasks")
    records = []
    seen_fingerprints = set()
    seen_outputs = set()
    for index, entry in enumerate(tasks):
        if not isinstance(entry, Mapping):
            raise ValueError("HGP screen task entry must be an object")
        task = entry.get("task", entry)
        if not isinstance(task, Mapping):
            raise ValueError("HGP screen task payload must be an object")
        fingerprint = entry.get("task_fingerprint", _sha256_json(task))
        if not isinstance(fingerprint, str) or not SHA256_RE.fullmatch(fingerprint):
            raise ValueError("HGP screen task fingerprint is invalid")
        if "task_fingerprint" in entry and fingerprint != _sha256_json(task):
            raise ValueError("HGP screen task fingerprint does not match task")
        output = entry.get("output_relpath")
        if output is None:
            output = task.get("output_relpath", f"measurement/{fingerprint}.npz")
        relative = _safe_relative_path(output, field="raw output path")
        if fingerprint in seen_fingerprints or relative in seen_outputs:
            raise ValueError("HGP screen task fingerprint/output is duplicated")
        seen_fingerprints.add(fingerprint)
        seen_outputs.add(relative)
        records.append({
            "task_index": index,
            "task_fingerprint": fingerprint,
            "output_relpath": relative,
            "owner": entry.get("owner", task.get("owner")),
        })
    return records


def _importance_records(manifest, config):
    spec = manifest.get("importance_sampling")
    if not isinstance(spec, Mapping):
        raise ValueError("HGP screen manifest has no IS specification")
    outputs = spec.get("outputs")
    cells = []
    descriptors_by_cell = {}
    seen_cells = set()
    for entry in manifest.get("tasks", []):
        task = entry.get("task", entry)
        descriptor = task.get("map_artifact") if isinstance(task, Mapping) else None
        if not isinstance(descriptor, Mapping):
            continue
        cell = task.get("cell")
        cell_key = _canonical_json(cell)
        if cell_key not in seen_cells:
            seen_cells.add(cell_key)
            cells.append(cell)
        fingerprint = _sha256_json(cell)
        old = descriptors_by_cell.setdefault(fingerprint, dict(descriptor))
        if old != descriptor:
            raise ValueError("MAP tasks disagree on their artifact descriptor")
    owners = tuple(manifest.get("execution_nodes", ()))
    configured = config.get("importance_sampling", {})
    if (not isinstance(outputs, list) or len(outputs) != len(cells)
            or not owners
            or spec.get("num_samples_per_cell")
            != configured.get("num_samples_per_cell")
            or spec.get("used_for_gate_or_selection") is not False):
        raise ValueError("HGP screen IS manifest changed")
    records = []
    seen = set()
    for index, (cell, output) in enumerate(zip(cells, outputs)):
        relative = _safe_relative_path(output, field="IS output path")
        if relative in seen:
            raise ValueError("HGP screen IS output path is duplicated")
        seen.add(relative)
        identity = {
            "contract_version": CONTRACT_VERSION,
            "manifest_sha256": manifest.get("manifest_sha256"),
            "archive_sha256": manifest.get("archive_sha256"),
            "source_manifest_sha256": manifest.get(
                "source_manifest_sha256",
            ),
            "cell": cell,
            "output_relpath": relative,
            "raw_version": spec.get("raw_version"),
            "num_samples": spec.get("num_samples_per_cell"),
        }
        records.append({
            "is_index": index,
            "is_fingerprint": _sha256_json(identity),
            "output_relpath": relative,
            "owner": owners[index % len(owners)],
            "cell": cell,
            "artifact_descriptor": descriptors_by_cell[_sha256_json(cell)],
        })
    return records


def _task_payload(manifest, record):
    entry = manifest["tasks"][record["task_index"]]
    task = entry.get("task", entry)
    if _sha256_json(task) != record["task_fingerprint"]:
        raise ValueError("control task no longer matches frozen manifest")
    return task


def _validate_control(
        control_path, registry_path, config_path, source_commit,
        archive_sha256, source_manifest_sha256, artifact_root,
        artifact_manifest_path, schedule, schedule_path):
    manifest_path = Path(control_path).resolve(strict=True)
    manifest = _read_json(manifest_path)
    registry = _load_registry(registry_path)
    config = _load_config(config_path, registry)
    _validate_artifact_manifest(
        artifact_manifest_path, registry_path, config_path, source_commit,
        archive_sha256, source_manifest_sha256, artifact_root,
        schedule, schedule_path,
    )
    _validate_manifest(manifest, registry, config, artifact_root)
    if (manifest.get("contract_version") != CONTRACT_VERSION
            or manifest.get("source_commit") != source_commit
            or manifest.get("archive_sha256") != archive_sha256
            or manifest.get("source_manifest_sha256")
            != source_manifest_sha256
            or manifest.get("resource_tier") not in RESOURCE_TIER_ORDER):
        raise ValueError("HGP screen control identity is invalid")
    records = _manifest_records(manifest)
    is_records = _importance_records(manifest, config)
    owners = tuple(manifest.get("execution_nodes", ()))
    configured_owners = tuple(
        config.get("execution", {}).get("execution_nodes", ()),
    )
    if (not owners or owners != configured_owners
            or int(manifest.get("task_count", -1)) != len(records)
            or manifest.get("analysis")
            != config.get("execution", {}).get("analysis")
            or any(record["owner"] not in owners for record in records)):
        raise ValueError("HGP screen control task count is invalid")
    return (
        manifest, manifest_path, records, is_records, registry, config,
    )


def _preflight_node(args):
    if args.node not in EXPECTED_PREFLIGHT_NODES:
        raise ValueError("preflight-node requires nd-1, nd-2, or nd-3")
    _require_node(args.node)
    source_identity = _verify_provenance(
        args.source_commit, args.archive_sha256,
        args.source_manifest_sha256,
    )
    registry_path = Path(args.registry).resolve(strict=True)
    config_path = Path(args.config).resolve(strict=True)
    registry = _load_registry(registry_path)
    config = _load_config(config_path, registry)
    schedule, schedule_path = _validate_schedule(
        args.schedule, registry_path, config_path, args.source_commit,
        args.archive_sha256, args.source_manifest_sha256, config,
    )
    _enforce_deadline(schedule, "preflight_deadline_unix")
    artifact_root = Path(args.artifact_root).resolve(strict=True)
    artifact_manifest, artifact_manifest_path = _validate_artifact_manifest(
        args.artifact_manifest, registry_path, config_path,
        args.source_commit, args.archive_sha256,
        args.source_manifest_sha256, artifact_root, schedule, schedule_path,
    )
    output_root = Path(args.output_root).resolve()
    node_root = output_root / "nodes" / args.node
    if node_root.exists():
        raise FileExistsError("preflight node output already exists")
    node_root.mkdir(parents=True)
    started_local_unix = time.time()
    started_monotonic = time.monotonic()

    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [
        sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
        "data/expander_code/exp102/tests",
        "data/expander_code/exp101/tests",
    ]
    completed = subprocess.run(
        command, check=False, capture_output=True, text=True,
        cwd=Path.cwd(), env=environment,
    )
    test_log = completed.stdout + completed.stderr
    _write_exclusive_text(node_root / "pytest.log", test_log)
    if completed.returncode:
        raise RuntimeError(f"full exp102+exp101 tests failed on {args.node}")
    if _verify_source(args.source_commit) != source_identity:
        raise RuntimeError("pytest changed the verified source tree")

    # Exercise the complete frozen IS transcript on every preflight node.
    # These files are preflight-only evidence and are never reusable screen raw.
    is_digest_root = node_root / "is_digest"
    is_digest_root.mkdir()
    is_digests = []
    for cell in _artifact_cells(config, artifact_manifest):
        fingerprint = _sha256_json(cell)
        is_path = is_digest_root / f"{fingerprint}.npz"
        generated = _run_is(
            registry_path, config_path, args.source_commit,
            args.archive_sha256, args.source_manifest_sha256, cell,
            artifact_root, is_path,
            seed_namespace=_pipeline().HGP_SCREEN_PREFLIGHT_IS_ROOT,
        )
        validated = _validate_is(
            is_path, registry_path, config_path, args.source_commit,
            args.archive_sha256, args.source_manifest_sha256, cell,
            artifact_root,
            seed_namespace=_pipeline().HGP_SCREEN_PREFLIGHT_IS_ROOT,
        )
        if (generated["transcript_sha256"]
                != validated["transcript_sha256"]):
            raise RuntimeError("preflight IS generation/replay digest changed")
        is_digests.append({
            "cell_fingerprint": fingerprint,
            "transcript_sha256": validated["transcript_sha256"],
            "file_sha256": _sha256_file(is_path),
        })

    digest_payload = _jsonable(_preflight_digest(
        registry_path, config_path, args.source_commit,
        args.archive_sha256, args.source_manifest_sha256, artifact_root,
    ))
    digest_payload["importance_sampling_transcript_sha256"] = [
        {
            "cell_fingerprint": value["cell_fingerprint"],
            "transcript_sha256": value["transcript_sha256"],
        }
        for value in is_digests
    ]
    runtime_payload = _jsonable(_benchmark(
        registry_path, config_path, args.source_commit,
        args.archive_sha256, args.source_manifest_sha256, artifact_root,
    ))
    if _forbidden_runtime_outcome(runtime_payload):
        raise ValueError("runtime benchmark contains a forbidden physics outcome")
    digest_report = {
        "contract_version": CONTRACT_VERSION,
        "source_commit": args.source_commit,
        "archive_sha256": args.archive_sha256,
        "source_manifest_sha256": args.source_manifest_sha256,
        "schedule_sha256": schedule["schedule_sha256"],
        "schedule_file_sha256": _sha256_file(schedule_path),
        "artifact_manifest_sha256": artifact_manifest[
            "artifact_manifest_sha256"
        ],
        "artifact_manifest_file_sha256": _sha256_file(
            artifact_manifest_path,
        ),
        "registry_file_sha256": _registry_sha(registry_path),
        "config_file_sha256": _config_sha(config_path),
        "payload": digest_payload,
    }
    runtime_report = {
        "contract_version": CONTRACT_VERSION,
        "source_commit": args.source_commit,
        "archive_sha256": args.archive_sha256,
        "source_manifest_sha256": args.source_manifest_sha256,
        "schedule_sha256": schedule["schedule_sha256"],
        "schedule_file_sha256": _sha256_file(schedule_path),
        "artifact_manifest_sha256": artifact_manifest[
            "artifact_manifest_sha256"
        ],
        "artifact_manifest_file_sha256": _sha256_file(
            artifact_manifest_path,
        ),
        "registry_file_sha256": _registry_sha(registry_path),
        "config_file_sha256": _config_sha(config_path),
        "payload": runtime_payload,
    }
    digest_sha = _write_exclusive_json(node_root / "digest.json", digest_report)
    runtime_sha = _write_exclusive_json(node_root / "runtime.json", runtime_report)
    if _verify_source(args.source_commit) != source_identity:
        raise RuntimeError("preflight changed the verified source tree")
    report = {
        "report_version": PREFLIGHT_NODE_VERSION,
        "contract_version": CONTRACT_VERSION,
        "status": "PASS",
        "node": args.node,
        "source_commit": args.source_commit,
        "archive_sha256": args.archive_sha256,
        "source_manifest_sha256": args.source_manifest_sha256,
        "schedule_sha256": schedule["schedule_sha256"],
        "schedule_file_sha256": _sha256_file(schedule_path),
        "artifact_manifest_sha256": artifact_manifest[
            "artifact_manifest_sha256"
        ],
        "artifact_manifest_file_sha256": _sha256_file(
            artifact_manifest_path,
        ),
        "source_identity": source_identity,
        "registry_file_sha256": _registry_sha(registry_path),
        "config_file_sha256": _config_sha(config_path),
        "pytest_returncode": completed.returncode,
        "pytest_log_sha256": _sha256_file(node_root / "pytest.log"),
        "digest_sha256": digest_sha,
        "runtime_sha256": runtime_sha,
        "preflight_is_file_sha256": {
            value["cell_fingerprint"]: value["file_sha256"]
            for value in is_digests
        },
        "environment": {
            "system": platform.system(), "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "clock_domain": "unsynchronized_local_diagnostic",
        "started_local_unix": started_local_unix,
        "completed_local_unix": time.time(),
        "elapsed_monotonic_seconds": time.monotonic() - started_monotonic,
    }
    _enforce_deadline(schedule, "preflight_deadline_unix")
    _write_exclusive_json(node_root / "preflight.json", report)
    print(_canonical_json({
        "node": args.node, "status": "PASS",
        "wall_seconds": report["elapsed_monotonic_seconds"],
    }))


def _parse_node_paths(values):
    result = {}
    for value in values:
        node, separator, path = value.partition("=")
        if not separator or node in result:
            raise ValueError("node report must be unique NODE=PATH")
        result[node] = Path(path).resolve(strict=True)
    if set(result) != set(EXPECTED_PREFLIGHT_NODES):
        raise ValueError("three-node consensus requires nd-1, nd-2, and nd-3")
    return result


def _combine_preflight(args):
    _require_node(ARTIFACT_BUILDER_NODE)
    local_source_identity = _verify_provenance(
        args.source_commit, args.archive_sha256,
        args.source_manifest_sha256,
    )
    registry_path = Path(args.registry).resolve(strict=True)
    config_path = Path(args.config).resolve(strict=True)
    registry = _load_registry(registry_path)
    config = _load_config(config_path, registry)
    schedule, schedule_path = _validate_schedule(
        args.schedule, registry_path, config_path, args.source_commit,
        args.archive_sha256, args.source_manifest_sha256, config,
    )
    _enforce_deadline(schedule, "preflight_deadline_unix")
    artifact_root = Path(args.artifact_root).resolve(strict=True)
    artifact_manifest, artifact_manifest_path = _validate_artifact_manifest(
        args.artifact_manifest, registry_path, config_path,
        args.source_commit, args.archive_sha256,
        args.source_manifest_sha256, artifact_root, schedule, schedule_path,
    )
    _resource_tiers(config)
    accounting = _runtime_accounting(config)
    budget = _screen_budget_seconds(config)
    node_paths = _parse_node_paths(args.node_report)
    node_reports = {}
    digest_reports = {}
    runtime_reports = {}
    for node in EXPECTED_PREFLIGHT_NODES:
        path = node_paths[node]
        report = _read_json(path)
        if (report.get("report_version") != PREFLIGHT_NODE_VERSION
                or report.get("contract_version") != CONTRACT_VERSION
                or report.get("status") != "PASS"
                or report.get("node") != node
                or report.get("source_commit") != args.source_commit
                or report.get("archive_sha256") != args.archive_sha256
                or report.get("source_manifest_sha256")
                != args.source_manifest_sha256
                or report.get("schedule_sha256")
                != schedule["schedule_sha256"]
                or report.get("schedule_file_sha256")
                != _sha256_file(schedule_path)
                or report.get("artifact_manifest_sha256")
                != artifact_manifest["artifact_manifest_sha256"]
                or report.get("artifact_manifest_file_sha256")
                != _sha256_file(artifact_manifest_path)
                or report.get("registry_file_sha256") != _registry_sha(registry_path)
                or report.get("config_file_sha256") != _config_sha(config_path)
                or report.get("pytest_returncode") != 0
                or report.get("environment", {}).get("system") != "Linux"
                or report.get("clock_domain")
                != "unsynchronized_local_diagnostic"
                or not math.isfinite(float(report.get(
                    "started_local_unix", math.nan,
                )))
                or not math.isfinite(float(report.get(
                    "completed_local_unix", math.nan,
                )))
                or not math.isfinite(float(report.get(
                    "elapsed_monotonic_seconds", math.nan,
                )))
                or float(report.get("elapsed_monotonic_seconds", -1.0))
                < 0.0):
            raise ValueError(f"invalid HGP preflight report for {node}")
        digest_path = path.parent / "digest.json"
        runtime_path = path.parent / "runtime.json"
        if (_sha256_file(digest_path) != report.get("digest_sha256")
                or _sha256_file(runtime_path) != report.get("runtime_sha256")
                or _sha256_file(path.parent / "pytest.log")
                != report.get("pytest_log_sha256")):
            raise ValueError(f"HGP preflight evidence hash mismatch for {node}")
        expected_is = report.get("preflight_is_file_sha256")
        expected_fingerprints = {
            _sha256_json(cell)
            for cell in _artifact_cells(config, artifact_manifest)
        }
        if (not isinstance(expected_is, Mapping)
                or set(expected_is) != expected_fingerprints):
            raise ValueError(f"HGP preflight IS evidence is absent for {node}")
        for fingerprint, expected_sha in expected_is.items():
            is_path = path.parent / "is_digest" / f"{fingerprint}.npz"
            if not is_path.is_file() or _sha256_file(is_path) != expected_sha:
                raise ValueError(f"HGP preflight IS evidence hash mismatch for {node}")
        digest_reports[node] = _read_json(digest_path)
        runtime_reports[node] = _read_json(runtime_path)
        node_reports[node] = report

    source_identity = node_reports[EXPECTED_PREFLIGHT_NODES[0]]["source_identity"]
    if (not isinstance(source_identity, Mapping)
            or source_identity.get("mode") != "archive"
            or source_identity.get("source_commit") != args.source_commit
            or source_identity.get("archive_sha256") != args.archive_sha256
            or source_identity.get("manifest_sha256")
            != args.source_manifest_sha256
            or any(node_reports[node].get("source_identity") != source_identity
                   for node in EXPECTED_PREFLIGHT_NODES[1:])):
        raise ValueError("three-node preflight source archives are not identical")
    if (local_source_identity.get("mode") == "archive"
            and local_source_identity != source_identity):
        raise ValueError("combiner source archive differs from preflight nodes")

    identity = {
        "contract_version": CONTRACT_VERSION,
        "source_commit": args.source_commit,
        "archive_sha256": args.archive_sha256,
        "source_manifest_sha256": args.source_manifest_sha256,
        "schedule_sha256": schedule["schedule_sha256"],
        "schedule_file_sha256": _sha256_file(schedule_path),
        "artifact_manifest_sha256": artifact_manifest[
            "artifact_manifest_sha256"
        ],
        "artifact_manifest_file_sha256": _sha256_file(
            artifact_manifest_path,
        ),
        "registry_file_sha256": _registry_sha(registry_path),
        "config_file_sha256": _config_sha(config_path),
    }
    for node in EXPECTED_PREFLIGHT_NODES:
        for report in (digest_reports[node], runtime_reports[node]):
            if {key: report.get(key) for key in identity} != identity:
                raise ValueError(f"preflight identity mismatch on {node}")
    digest_payload = digest_reports[EXPECTED_PREFLIGHT_NODES[0]]["payload"]
    digest_transcript = _canonical_json(digest_payload)
    if any(_canonical_json(digest_reports[node]["payload"])
           != digest_transcript
           for node in EXPECTED_PREFLIGHT_NODES[1:]):
        raise ValueError("HGP canonical digest payload differs across nodes")
    seed_catalog = runtime_reports[EXPECTED_PREFLIGHT_NODES[0]][
        "payload"
    ].get("auxiliary_seed_catalog")
    if any(runtime_reports[node]["payload"].get("auxiliary_seed_catalog")
           != seed_catalog for node in EXPECTED_PREFLIGHT_NODES[1:]):
        raise ValueError("HGP runtime auxiliary seed catalog differs across nodes")

    control_origin_elapsed_seconds = 8 * 3600.0
    screen_deadline_elapsed_seconds = 22 * 3600.0
    analysis_deadline_elapsed_seconds = 24 * 3600.0
    tier_consensus = []
    for tier in RESOURCE_TIER_ORDER:
        node_rows = {}
        for node in EXPECTED_PREFLIGHT_NODES:
            runtime = runtime_reports[node]["payload"]
            if runtime.get("version") != "exp102.q0_hgp_global.screen.runtime_node.v3":
                raise ValueError("runtime benchmark schema version changed")
            if _forbidden_runtime_outcome(runtime):
                raise ValueError("runtime consensus found a physics outcome")
            _validate_auxiliary_seed_catalog(runtime)
            seconds, local_pass, row = _tier_projection(runtime, tier)
            _validate_projection_accounting(
                row, accounting, config, runtime, artifact_manifest,
            )
            runtime_context = {
                key: runtime[key] for key in (
                    "timings", "is_seconds_by_cell",
                    "artifact_generation_wall_seconds", "owner_task_counts",
                    "b_analysis_timings", "auxiliary_seed_catalog",
                    "auxiliary_seed_catalog_sha256",
                )
            }
            safe_generation = accounting["safety_factor"] * float(
                row["screen_generation_wall_seconds"]
            )
            safe_analysis = accounting["safety_factor"] * float(
                row["analysis_workload"]["projected_analysis_wall_seconds"]
            )
            projected_screen_completion = (
                control_origin_elapsed_seconds + safe_generation
            )
            projected_analysis_completion = (
                projected_screen_completion + safe_analysis
            )
            deadline_pass = (
                projected_screen_completion
                <= screen_deadline_elapsed_seconds
                and projected_analysis_completion
                <= analysis_deadline_elapsed_seconds
            )
            node_rows[node] = {
                "projected_wall_seconds": seconds,
                "runtime_gate_pass": local_pass,
                "projection": row,
                "runtime_context": runtime_context,
                "projected_screen_completion_elapsed_seconds": (
                    projected_screen_completion
                ),
                "projected_analysis_completion_elapsed_seconds": (
                    projected_analysis_completion
                ),
                "schedule_deadline_pass": bool(deadline_pass),
            }
        worst = max(value["projected_wall_seconds"]
                    for value in node_rows.values())
        passed = (
            all(value["runtime_gate_pass"] for value in node_rows.values())
            and all(value["schedule_deadline_pass"]
                    for value in node_rows.values())
            and worst <= budget
        )
        tier_consensus.append({
            "resource_tier": tier,
            "screen_budget_seconds": budget,
            "worst_node_projected_wall_seconds": worst,
            "nodes": node_rows,
            "pass": bool(passed),
        })
    passing = [row for row in tier_consensus if row["pass"]]
    selected = passing[-1]["resource_tier"] if passing else None
    result = {
        "report_version": PREFLIGHT_VERSION,
        **identity,
        "status": "PASS" if selected is not None else "RUNTIME_EXHAUSTED",
        "nodes": list(EXPECTED_PREFLIGHT_NODES),
        "source_identity": source_identity,
        "node_report_sha256": {
            node: _sha256_file(node_paths[node])
            for node in EXPECTED_PREFLIGHT_NODES
        },
        "canonical_digest": digest_payload,
        "canonical_digest_sha256": _sha256_json(digest_payload),
        "runtime_consensus": tier_consensus,
        "selected_resource_tier": selected,
        "selection_basis": (
            "runtime_only_worst_node_and_frozen_elapsed_deadlines"
        ),
        "control_origin_elapsed_seconds": control_origin_elapsed_seconds,
        "screen_budget_seconds": budget,
        "clock_domain": "unsynchronized_local_diagnostic",
        "completed_local_unix": time.time(),
    }
    _enforce_deadline(schedule, "preflight_deadline_unix")
    if _verify_source(args.source_commit) != local_source_identity:
        raise RuntimeError("preflight combination changed the verified source")
    _write_exclusive_json(args.output, result)
    print(_canonical_json({
        "status": result["status"], "selected_resource_tier": selected,
        "output": str(Path(args.output).resolve()),
    }))


def _validate_preflight(
        path, registry_path, config_path, source_commit, archive_sha256,
        source_manifest_sha256, artifact_manifest_path, artifact_root,
        config, schedule, schedule_path):
    artifact_manifest, artifact_path = _validate_artifact_manifest(
        artifact_manifest_path, registry_path, config_path, source_commit,
        archive_sha256, source_manifest_sha256, artifact_root,
        schedule, schedule_path,
    )
    report = _read_json(path)
    source_identity = report.get("source_identity")
    if (report.get("report_version") != PREFLIGHT_VERSION
            or report.get("contract_version") != CONTRACT_VERSION
            or report.get("status") != "PASS"
            or report.get("source_commit") != source_commit
            or report.get("archive_sha256") != archive_sha256
            or report.get("source_manifest_sha256")
            != source_manifest_sha256
            or report.get("schedule_sha256") != schedule["schedule_sha256"]
            or report.get("schedule_file_sha256")
            != _sha256_file(schedule_path)
            or report.get("artifact_manifest_sha256")
            != artifact_manifest["artifact_manifest_sha256"]
            or report.get("artifact_manifest_file_sha256")
            != _sha256_file(artifact_path)
            or report.get("registry_file_sha256") != _registry_sha(registry_path)
            or report.get("config_file_sha256") != _config_sha(config_path)
            or report.get("selected_resource_tier") not in RESOURCE_TIER_ORDER
            or report.get("selection_basis")
            != "runtime_only_worst_node_and_frozen_elapsed_deadlines"
            or not isinstance(source_identity, Mapping)
            or source_identity.get("mode") != "archive"
            or source_identity.get("source_commit") != source_commit
            or source_identity.get("archive_sha256") != archive_sha256
            or source_identity.get("manifest_sha256")
            != source_manifest_sha256
            or report.get("clock_domain")
            != "unsynchronized_local_diagnostic"
            or not math.isfinite(float(report.get(
                "completed_local_unix", math.nan,
            )))):
        raise ValueError("HGP screen preflight does not grant run authority")
    rows = report.get("runtime_consensus")
    if (not isinstance(rows, list)
            or [row.get("resource_tier") for row in rows]
            != list(RESOURCE_TIER_ORDER)
            or any(not isinstance(row.get("pass"), bool) for row in rows)):
        raise ValueError("HGP screen preflight runtime consensus is invalid")
    accounting = _runtime_accounting(config)
    budget = _screen_budget_seconds(config)
    control_origin_elapsed_seconds = 8 * 3600.0
    screen_deadline_elapsed_seconds = 22 * 3600.0
    analysis_deadline_elapsed_seconds = 24 * 3600.0
    if float(report.get("control_origin_elapsed_seconds", -1.0)) != (
            control_origin_elapsed_seconds):
        raise ValueError("HGP preflight runtime selection clock changed")
    for consensus in rows:
        node_rows = consensus.get("nodes")
        if (not isinstance(node_rows, Mapping)
                or set(node_rows) != set(EXPECTED_PREFLIGHT_NODES)
                or float(consensus.get("screen_budget_seconds", -1.0))
                != budget):
            raise ValueError("HGP preflight node runtime evidence is invalid")
        projected = []
        local_passes = []
        for node in EXPECTED_PREFLIGHT_NODES:
            value = node_rows[node]
            projection = value.get("projection")
            _validate_auxiliary_seed_catalog(value.get("runtime_context", {}))
            _validate_projection_accounting(
                projection, accounting, config,
                value.get("runtime_context"), artifact_manifest,
            )
            seconds, local_pass, canonical = _tier_projection(
                {"tiers": [projection]}, consensus["resource_tier"],
            )
            if (canonical != projection
                    or float(value.get("projected_wall_seconds", -1.0))
                    != seconds
                    or value.get("runtime_gate_pass") is not local_pass):
                raise ValueError("HGP preflight runtime row changed")
            safe_generation = accounting["safety_factor"] * float(
                projection["screen_generation_wall_seconds"]
            )
            safe_analysis = accounting["safety_factor"] * float(
                projection["analysis_workload"][
                    "projected_analysis_wall_seconds"
                ]
            )
            screen_completion = control_origin_elapsed_seconds + safe_generation
            analysis_completion = screen_completion + safe_analysis
            deadline_pass = (
                screen_completion <= screen_deadline_elapsed_seconds
                and analysis_completion
                <= analysis_deadline_elapsed_seconds
            )
            if (not math.isclose(
                    float(value.get(
                        "projected_screen_completion_elapsed_seconds",
                        math.inf,
                    )), screen_completion, rel_tol=1e-12, abs_tol=1e-6)
                    or not math.isclose(
                        float(value.get(
                            "projected_analysis_completion_elapsed_seconds",
                            math.inf,
                        )), analysis_completion,
                        rel_tol=1e-12, abs_tol=1e-6,
                    )
                    or value.get("schedule_deadline_pass")
                    is not bool(deadline_pass)):
                raise ValueError("HGP preflight deadline projection changed")
            projected.append(seconds)
            local_passes.append(local_pass and deadline_pass)
        worst = max(projected)
        expected_pass = all(local_passes) and worst <= budget
        if (not math.isclose(
                float(consensus.get("worst_node_projected_wall_seconds", -1.0)),
                worst, rel_tol=1e-12, abs_tol=1e-9)
                or consensus["pass"] is not bool(expected_pass)):
            raise ValueError("HGP preflight runtime consensus changed")
    passing = [row["resource_tier"] for row in rows if row["pass"]]
    flags = [row["pass"] for row in rows]
    if (flags != sorted(flags, reverse=True)
            or not passing
            or report["selected_resource_tier"] != passing[-1]):
        raise ValueError("HGP screen preflight did not select the largest tier")
    digest = report.get("canonical_digest")
    if report.get("canonical_digest_sha256") != _sha256_json(digest):
        raise ValueError("HGP screen preflight canonical digest SHA is invalid")
    return report


def _build_control(args):
    _require_node(ARTIFACT_BUILDER_NODE)
    source_identity = _verify_provenance(
        args.source_commit, args.archive_sha256,
        args.source_manifest_sha256,
    )
    registry_path = Path(args.registry).resolve(strict=True)
    config_path = Path(args.config).resolve(strict=True)
    registry = _load_registry(registry_path)
    config = _load_config(config_path, registry)
    schedule, schedule_path = _validate_schedule(
        args.schedule, registry_path, config_path, args.source_commit,
        args.archive_sha256, args.source_manifest_sha256, config,
    )
    _enforce_deadline(schedule, "control_freeze_deadline_unix")
    artifact_root = Path(args.artifact_root).resolve(strict=True)
    artifact_manifest_path = Path(args.artifact_manifest).resolve(strict=True)
    preflight_path = Path(args.preflight).resolve(strict=True)
    preflight = _validate_preflight(
        preflight_path, registry_path, config_path, args.source_commit,
        args.archive_sha256, args.source_manifest_sha256,
        artifact_manifest_path, artifact_root, config, schedule,
        schedule_path,
    )
    if (source_identity.get("mode") == "archive"
            and preflight.get("source_identity") != source_identity):
        raise ValueError("control source archive differs from preflight")
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("HGP screen control already exists")
    manifest = _build_manifest(
        registry_path, config_path, args.source_commit,
        args.archive_sha256, args.source_manifest_sha256,
        preflight["selected_resource_tier"], artifact_root, None,
    )
    _validate_manifest(manifest, registry, config, artifact_root)
    records = _manifest_records(manifest)
    is_records = _importance_records(manifest, config)
    _enforce_deadline(schedule, "control_freeze_deadline_unix")
    if _verify_source(args.source_commit) != source_identity:
        raise RuntimeError("control construction changed the verified source")
    _write_exclusive_json(output, manifest)
    # Re-read through the same path used by remote workers.
    _validate_control(
        output, registry_path, config_path, args.source_commit,
        args.archive_sha256, args.source_manifest_sha256, artifact_root,
        artifact_manifest_path, schedule, schedule_path,
    )
    owner_counts = {
        node: sum(record["owner"] == node for record in records)
        for node in manifest["execution_nodes"]
    }
    print(_canonical_json({
        "control": str(output), "task_count": len(records),
        "importance_sampling_count": len(is_records),
        "resource_tier": preflight["selected_resource_tier"],
        "owner_counts": owner_counts,
        "preflight_sha256": _sha256_file(preflight_path),
    }))


def _claim_raw(claim_path, fingerprint, node, manifest_sha256, kind):
    return _write_exclusive_json(claim_path, {
        "contract_version": CONTRACT_VERSION,
        "kind": kind,
        "fingerprint": fingerprint,
        "manifest_sha256": manifest_sha256,
        "node": node,
        "pid": os.getpid(),
        "claimed_unix": time.time(),
    })


def _npz_scalar(data, name):
    if name not in data.files:
        raise ValueError(f"staging raw is missing scalar {name}")
    value = np.asarray(data[name])
    if value.shape != ():
        raise ValueError(f"staging raw scalar changed shape: {name}")
    return value.item()


def _stored_sampler_digest(data):
    values = {
        name.removeprefix("sampler_"): np.asarray(data[name])
        for name in data.files if name.startswith("sampler_")
    }
    if not values:
        raise ValueError("staging raw contains no sampler payload")
    digest = hashlib.sha256(
        b"exp102.q0_hgp_global.screen.trajectory.v1\0"
    )
    for name in sorted(values):
        value = np.ascontiguousarray(values[name])
        digest.update(name.encode("ascii") + b"\0")
        digest.update(value.dtype.str.encode("ascii") + b"\0")
        digest.update(np.asarray(value.shape, dtype=">u8").tobytes())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _unpack_state_rows(packed, num_qubits, *, name):
    packed = np.asarray(packed)
    if packed.dtype != np.uint8 or packed.ndim not in (1, 2):
        raise ValueError(f"{name} is not a packed uint8 state array")
    rows = packed[None, :] if packed.ndim == 1 else packed
    all_bits = np.unpackbits(rows, axis=1, bitorder="little")
    if all_bits.shape[1] < num_qubits or all_bits[:, num_qubits:].any():
        raise ValueError(f"{name} has a noncanonical packed tail")
    return np.ascontiguousarray(all_bits[:, :num_qubits], dtype=np.uint8)


def _hard_residual_weights(model, syndrome, states):
    states = np.asarray(states, dtype=np.uint8)
    syndrome = np.asarray(syndrome, dtype=np.uint8)
    if (states.ndim != 2 or states.shape[1] != model.num_qubits
            or syndrome.shape != (model.num_checks,)):
        raise ValueError("staging hard-coset algebra dimensions changed")
    residual = np.zeros(states.shape[0], dtype=np.int32)
    checks = np.asarray(model.H_check, dtype=np.uint8)
    for row in range(checks.shape[0]):
        support = np.flatnonzero(checks[row])
        parity = np.bitwise_xor.reduce(states[:, support], axis=1)
        residual += (parity ^ syndrome[row]).astype(np.int32)
    return residual


def _state_labels(frame, states):
    states = np.asarray(states, dtype=np.uint8)
    if int(frame.k) > 64:
        raise ValueError("staging label replay supports at most 64 logical bits")
    signatures = np.zeros(frame.num_qubits, dtype=np.uint64)
    for qubit in range(frame.num_qubits):
        value = np.uint64(0)
        for bit in np.flatnonzero(frame.W_basis[:, qubit]):
            value |= np.uint64(1) << np.uint64(bit)
        signatures[qubit] = value
    labels = np.zeros(states.shape[0], dtype=np.uint64)
    for qubit, signature in enumerate(signatures):
        if signature:
            labels ^= states[:, qubit].astype(np.uint64) * signature
    return labels


def _load_model_for_cell(registry_path, cell):
    from data.expander_code.exp102.exp102_pipeline.registry import load_frozen_code
    from data.expander_code.exp102.exp102_pipeline.worker import build_model

    _, _, matrix = load_frozen_code(registry_path, cell["code_id"])
    return build_model(matrix)


def _check_state_field(data, field, model, syndrome):
    states = _unpack_state_rows(
        data[field], model.num_qubits, name=field,
    )
    if _hard_residual_weights(model, syndrome, states).any():
        raise ValueError(f"staging state left the hard coset: {field}")
    return states


def _validate_staging_measurement(
        path, registry_path, config, expected_task, source_commit,
        archive_sha256, source_manifest_sha256, artifact_root):
    """Check immutable structure/algebra without rerunning the sampler."""
    registry = _load_registry(registry_path)
    method = expected_task["method_id"]
    has_map = "map_artifact" in expected_task
    common_outer = {
        "raw_version", "sampler_raw_version", "contract_version",
        "task_json", "task_fingerprint", "source_commit",
        "archive_sha256", "source_manifest_sha256", "registry_sha256",
        "hgp_screen_config_sha256", "cell_json", "uniform_seed",
        "syndrome_packed", "syndrome_sha256", "model_fingerprint",
        "section_fingerprint", "logical_frame_fingerprint",
        "character_masks", "character_sha256", "num_qubits", "k",
        "b_character_masks_packed", "b_character_sha256",
        "b_character_count", "b_dimension", "b_dense_character_count",
        "trajectory_digest", "core_seconds", "wall_seconds",
    }
    map_outer = {
        "map_artifact_descriptor_json", "map_artifact_file_sha256",
        "map_artifact_content_sha256",
    } if has_map else set()
    with np.load(path, allow_pickle=False) as data:
        if any(np.asarray(data[name]).dtype.hasobject for name in data.files):
            raise ValueError("staging raw contains an object array")
        outer = {
            name for name in data.files
            if not name.startswith("sampler_") or name == "sampler_raw_version"
        }
        if outer != common_outer | map_outer:
            raise ValueError("staging raw outer schema changed")
        task_json = str(_npz_scalar(data, "task_json"))
        if (task_json != _canonical_json(expected_task)
                or json.loads(task_json) != expected_task
                or str(_npz_scalar(data, "task_fingerprint"))
                != _sha256_json(expected_task)):
            raise ValueError("staging raw task identity changed")
        expected_scalars = {
            "contract_version": CONTRACT_VERSION,
            "source_commit": source_commit,
            "archive_sha256": archive_sha256,
            "source_manifest_sha256": source_manifest_sha256,
            "registry_sha256": registry["registry_sha256"],
            "hgp_screen_config_sha256": config["hgp_screen_config_sha256"],
            "cell_json": _canonical_json(expected_task["cell"]),
            "raw_version": config["raw_versions"][
                "map" if has_map else "hp"
            ],
            "sampler_raw_version": (
                _pipeline().MAP_RAW_VERSION if has_map
                else _pipeline().COLLAPSED_RAW_VERSION
            ),
        }
        for name, expected in expected_scalars.items():
            if str(_npz_scalar(data, name)) != str(expected):
                raise ValueError(f"staging raw identity mismatch: {name}")
        for name in ("core_seconds", "wall_seconds"):
            value = float(_npz_scalar(data, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"staging raw timing is invalid: {name}")
        for name in data.files:
            value = np.asarray(data[name])
            if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
                raise ValueError(f"staging raw contains nonfinite values: {name}")
        if str(_npz_scalar(data, "trajectory_digest")) != _stored_sampler_digest(data):
            raise ValueError("staging sampler payload digest changed")
        model, frame = _load_model_for_cell(registry_path, expected_task["cell"])
        if (int(_npz_scalar(data, "num_qubits")) != model.num_qubits
                or int(_npz_scalar(data, "k")) != model.k):
            raise ValueError("staging model dimensions changed")
        b_masks = np.asarray(data["b_character_masks_packed"], dtype=np.uint8)
        b_count = int(_npz_scalar(data, "b_character_count"))
        b_dimension = int(_npz_scalar(data, "b_dimension"))
        b_dense = int(_npz_scalar(data, "b_dense_character_count"))
        b_sha = str(_npz_scalar(data, "b_character_sha256"))
        if (b_dimension <= 0 or b_dense < 64 or b_count <= b_dense
                or b_masks.shape != (b_count, (b_dimension + 7) // 8)
                or len(b_sha) != 64
                or any(value not in "0123456789abcdef" for value in b_sha)):
            raise ValueError("staging B-character catalog structure changed")
        syndrome_bits = np.unpackbits(
            np.asarray(data["syndrome_packed"], dtype=np.uint8),
            bitorder="little",
        )
        if (syndrome_bits.size < model.num_checks
                or syndrome_bits[model.num_checks:].any()):
            raise ValueError("staging syndrome packing changed")
        syndrome = np.ascontiguousarray(
            syndrome_bits[:model.num_checks], dtype=np.uint8,
        )
        state_fields = [
            "sampler_initial_state_packed", "sampler_burn_state_packed",
            "sampler_final_state_packed", "sampler_measurement_states_packed",
        ]
        for optional in (
                "sampler_burn_states_packed",
                "sampler_burn_proposal_states_packed",
                "sampler_measurement_proposal_states_packed"):
            if optional in data.files:
                state_fields.append(optional)
        state_cache = {
            name: _check_state_field(data, name, model, syndrome)
            for name in state_fields
        }
        pairs = (
            ("sampler_measurement_states_packed", "sampler_measurement_weights"),
            ("sampler_burn_states_packed", "sampler_burn_weights"),
            ("sampler_burn_proposal_states_packed", "sampler_burn_proposal_weights"),
            ("sampler_measurement_proposal_states_packed",
             "sampler_measurement_proposal_weights"),
        )
        for state_name, weight_name in pairs:
            if state_name in state_cache:
                expected = state_cache[state_name].sum(axis=1)
                if not np.array_equal(np.asarray(data[weight_name]), expected):
                    raise ValueError(f"staging cached weights changed: {weight_name}")
        label_pairs = (
            ("sampler_measurement_states_packed", "sampler_measurement_labels"),
            ("sampler_burn_states_packed", "sampler_burn_labels"),
        )
        for state_name, label_name in label_pairs:
            if state_name in state_cache:
                expected = _state_labels(frame, state_cache[state_name])
                if not np.array_equal(np.asarray(data[label_name]), expected):
                    raise ValueError(f"staging cached labels changed: {label_name}")
        residual = np.asarray(data["sampler_measurement_residual_weights"])
        if residual.shape != (state_cache[
                "sampler_measurement_states_packed"].shape[0],) or residual.any():
            raise ValueError("staging cached hard residual changed")
        blocks = np.asarray(data["sampler_measurement_block"])
        if (blocks.shape != residual.shape or blocks.size % 8
                or not np.array_equal(
                    blocks,
                    np.repeat(np.arange(8, dtype=blocks.dtype), blocks.size // 8),
                )):
            raise ValueError("staging fixed-clock block labels changed")
        if has_map:
            descriptor = expected_task["map_artifact"]
            if (str(_npz_scalar(data, "map_artifact_descriptor_json"))
                    != _canonical_json(descriptor)
                    or str(_npz_scalar(data, "map_artifact_file_sha256"))
                    != descriptor["artifact_file_sha256"]
                    or str(_npz_scalar(data, "map_artifact_content_sha256"))
                    != descriptor["artifact_content_sha256"]):
                raise ValueError("staging MAP artifact binding changed")
            artifact_path = Path(artifact_root) / descriptor["artifact_relpath"]
            if _sha256_file(artifact_path) != descriptor["artifact_file_sha256"]:
                raise ValueError("staging MAP artifact file changed")
    return True


def _content_sha256(metadata, arrays, version):
    digest = hashlib.sha256(str(version).encode("ascii") + b"\0")
    digest.update(_canonical_json(metadata).encode("ascii") + b"\0")
    for name in sorted(arrays):
        value = np.ascontiguousarray(np.asarray(arrays[name]))
        digest.update(name.encode("ascii") + b"\0")
        digest.update(value.dtype.str.encode("ascii") + b"\0")
        digest.update(np.asarray(value.shape, dtype=">u8").tobytes())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _validate_staging_is(
        path, registry_path, config, record, source_commit, archive_sha256,
        source_manifest_sha256, artifact_root):
    """Validate the frozen IS transcript without redrawing its 50k samples."""
    array_names = {
        "sample_states_packed", "sample_coordinates_packed",
        "sample_physical_weights", "sample_log_q",
        "sample_log_importance_weight", "sample_anchor_index",
        "sample_component_index",
    }
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {"identity_json", "transcript_sha256", *array_names}:
            raise ValueError("staging IS schema changed")
        if any(np.asarray(data[name]).dtype.hasobject for name in data.files):
            raise ValueError("staging IS contains an object array")
        identity_json = str(_npz_scalar(data, "identity_json"))
        identity = json.loads(identity_json)
        if identity_json != _canonical_json(identity):
            raise ValueError("staging IS identity is noncanonical")
        expected_scalars = {
            "contract_version": CONTRACT_VERSION,
            "source_commit": source_commit,
            "archive_sha256": archive_sha256,
            "source_manifest_sha256": source_manifest_sha256,
            "hgp_screen_config_sha256": config["hgp_screen_config_sha256"],
            "cell": record["cell"],
            "artifact_descriptor": record["artifact_descriptor"],
            "num_samples": config["importance_sampling"][
                "num_samples_per_cell"
            ],
            "used_for_gate_or_selection": False,
        }
        for name, expected in expected_scalars.items():
            if identity.get(name) != expected:
                raise ValueError(f"staging IS identity mismatch: {name}")
        arrays = {name: np.asarray(data[name]) for name in array_names}
        count = int(identity["num_samples"])
        if any(value.shape[0] != count for value in arrays.values()):
            raise ValueError("staging IS sample count changed")
        for name in ("sample_log_q", "sample_log_importance_weight"):
            if not np.isfinite(arrays[name]).all():
                raise ValueError(f"staging IS contains nonfinite {name}")
        model, _ = _load_model_for_cell(registry_path, record["cell"])
        states = _unpack_state_rows(
            arrays["sample_states_packed"], model.num_qubits,
            name="sample_states_packed",
        )
        descriptor = record["artifact_descriptor"]
        artifact_path = Path(artifact_root) / descriptor["artifact_relpath"]
        if _sha256_file(artifact_path) != descriptor["artifact_file_sha256"]:
            raise ValueError("staging IS MAP artifact file changed")
        with np.load(artifact_path, allow_pickle=False) as artifact:
            metadata = json.loads(str(_npz_scalar(artifact, "metadata_json")))
            syndrome_bits = np.unpackbits(
                np.asarray(artifact["syndrome_packed"], dtype=np.uint8),
                bitorder="little",
            )
        syndrome = syndrome_bits[:int(metadata["num_checks"])]
        if _hard_residual_weights(model, syndrome, states).any():
            raise ValueError("staging IS proposal left the hard coset")
        if not np.array_equal(
                arrays["sample_physical_weights"], states.sum(axis=1)):
            raise ValueError("staging IS cached physical weights changed")
        expected_sha = _content_sha256(
            identity, arrays, config["raw_versions"]["importance_sampling"],
        )
        if str(_npz_scalar(data, "transcript_sha256")) != expected_sha:
            raise ValueError("staging IS transcript SHA changed")
    return True


def _execute_task(argument):
    (registry_path, config_path, source_commit, archive_sha256,
     source_manifest_sha256, artifact_root, node, raw_root,
     manifest_sha256, record, task) = argument
    raw_path = Path(raw_root) / record["output_relpath"]
    if raw_path.exists():
        raise FileExistsError(f"fresh HGP screen refuses existing raw {raw_path}")
    claim_path = Path(raw_root) / ".claims" / (
        record["task_fingerprint"] + ".json"
    )
    _claim_raw(
        claim_path, record["task_fingerprint"], node, manifest_sha256,
        "measurement",
    )
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    staging_path = Path(raw_root) / ".staging" / (
        record["task_fingerprint"] + f".{os.getpid()}.npz"
    )
    staging_path.parent.mkdir(parents=True, exist_ok=True)
    if staging_path.exists():
        raise FileExistsError("fresh HGP task found an existing staging file")
    result = _run_task(
        registry_path, config_path, source_commit, archive_sha256,
        source_manifest_sha256, task, artifact_root, str(staging_path),
    )
    if not staging_path.is_file():
        raise RuntimeError("HGP screen task did not create its raw output")
    registry = _load_registry(registry_path)
    config = _load_config(config_path, registry)
    _validate_staging_measurement(
        staging_path, registry_path, config, task, source_commit,
        archive_sha256, source_manifest_sha256, artifact_root,
    )
    staging_sha256 = _sha256_file(staging_path)
    # link(2) provides an atomic no-overwrite installation of the final raw.
    os.link(staging_path, raw_path)
    staging_path.unlink()
    if _sha256_file(raw_path) != staging_sha256:
        raise RuntimeError("installed HGP raw differs from validated staging raw")
    return {
        "task_fingerprint": record["task_fingerprint"],
        "output_relpath": record["output_relpath"],
        "sha256": _sha256_file(raw_path),
        "size_bytes": raw_path.stat().st_size,
        "pipeline_result_sha256": _sha256_json(_jsonable(result)),
        "claim_sha256": _sha256_file(claim_path),
    }


def _execute_is(argument):
    (registry_path, config_path, source_commit, archive_sha256,
     source_manifest_sha256, artifact_root, node, raw_root,
     manifest_sha256, record) = argument
    raw_path = Path(raw_root) / record["output_relpath"]
    if raw_path.exists():
        raise FileExistsError(f"fresh HGP screen refuses existing IS raw {raw_path}")
    claim_path = Path(raw_root) / ".claims_is" / (
        record["is_fingerprint"] + ".json"
    )
    _claim_raw(
        claim_path, record["is_fingerprint"], node, manifest_sha256,
        "importance_sampling",
    )
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    staging_path = Path(raw_root) / ".staging_is" / (
        record["is_fingerprint"] + f".{os.getpid()}.npz"
    )
    staging_path.parent.mkdir(parents=True, exist_ok=True)
    if staging_path.exists():
        raise FileExistsError("fresh HGP IS task found an existing staging file")
    result = _run_is(
        registry_path, config_path, source_commit, archive_sha256,
        source_manifest_sha256, record["cell"], artifact_root,
        str(staging_path),
    )
    if not staging_path.is_file():
        raise RuntimeError("HGP screen IS task did not create its raw output")
    registry = _load_registry(registry_path)
    config = _load_config(config_path, registry)
    _validate_staging_is(
        staging_path, registry_path, config, record, source_commit,
        archive_sha256, source_manifest_sha256, artifact_root,
    )
    staging_sha256 = _sha256_file(staging_path)
    os.link(staging_path, raw_path)
    staging_path.unlink()
    if _sha256_file(raw_path) != staging_sha256:
        raise RuntimeError("installed HGP IS raw differs from validated staging raw")
    return {
        "is_fingerprint": record["is_fingerprint"],
        "output_relpath": record["output_relpath"],
        "sha256": _sha256_file(raw_path),
        "size_bytes": raw_path.stat().st_size,
        "pipeline_result_sha256": _sha256_json(_jsonable(result)),
        "claim_sha256": _sha256_file(claim_path),
    }


def _run_node(args):
    if args.num_workers <= 0:
        raise ValueError("num-workers must be positive")
    source_identity = _verify_provenance(
        args.source_commit, args.archive_sha256,
        args.source_manifest_sha256,
    )
    _require_node(args.node)
    registry_path = Path(args.registry).resolve(strict=True)
    config_path = Path(args.config).resolve(strict=True)
    control_path = Path(args.control).resolve(strict=True)
    artifact_root = Path(args.artifact_root).resolve(strict=True)
    artifact_manifest_path = Path(args.artifact_manifest).resolve(strict=True)
    registry = _load_registry(registry_path)
    config = _load_config(config_path, registry)
    schedule, schedule_path = _validate_schedule(
        args.schedule, registry_path, config_path, args.source_commit,
        args.archive_sha256, args.source_manifest_sha256, config,
    )
    _enforce_deadline(schedule, "screen_deadline_unix")
    (manifest, _, records, is_records, _, config) = _validate_control(
        control_path, registry_path, config_path, args.source_commit,
        args.archive_sha256, args.source_manifest_sha256, artifact_root,
        artifact_manifest_path, schedule, schedule_path,
    )
    preflight_path = Path(args.preflight).resolve(strict=True)
    preflight = _validate_preflight(
        preflight_path, registry_path, config_path, args.source_commit,
        args.archive_sha256, args.source_manifest_sha256,
        artifact_manifest_path, artifact_root, config, schedule,
        schedule_path,
    )
    if (preflight["selected_resource_tier"] != manifest["resource_tier"]
            or preflight.get("source_identity") != source_identity):
        raise ValueError("HGP node control differs from authorized preflight tier")
    if args.node not in manifest["execution_nodes"]:
        raise ValueError("node is not a frozen owner in the HGP control")
    capacities = config.get("execution", {}).get("capacities", {})
    if (args.node not in capacities
            or args.num_workers != int(capacities[args.node])):
        raise ValueError("num-workers differs from the frozen node capacity")
    raw_root = Path(args.raw_root).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("HGP node report already exists")
    frozen = [record for record in records
              if record["owner"] == args.node]
    frozen_is = [record for record in is_records
                 if record["owner"] == args.node]
    work = [(
        str(registry_path), str(config_path), args.source_commit,
        args.archive_sha256, args.source_manifest_sha256,
        str(artifact_root), args.node, str(raw_root),
        manifest["manifest_sha256"], record,
        _task_payload(manifest, record),
    ) for record in frozen]
    is_work = [(
        str(registry_path), str(config_path), args.source_commit,
        args.archive_sha256, args.source_manifest_sha256,
        str(artifact_root), args.node, str(raw_root),
        manifest["manifest_sha256"], record,
    ) for record in frozen_is]
    if any((raw_root / record["output_relpath"]).exists()
           for record in frozen):
        raise FileExistsError("fresh HGP node stage found existing raw")
    if any((raw_root / ".claims" /
            (record["task_fingerprint"] + ".json")).exists()
           for record in frozen):
        raise FileExistsError("fresh HGP node stage found an existing claim")
    if any((raw_root / record["output_relpath"]).exists()
           for record in frozen_is):
        raise FileExistsError("fresh HGP node stage found existing IS raw")
    if any((raw_root / ".claims_is" /
            (record["is_fingerprint"] + ".json")).exists()
           for record in frozen_is):
        raise FileExistsError("fresh HGP node stage found an existing IS claim")

    started_local_unix = time.time()
    started_monotonic = time.monotonic()
    results = []
    is_results = []
    if is_work:
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=min(args.num_workers, len(is_work))) as pool:
            for result in pool.map(_execute_is, is_work, chunksize=1):
                is_results.append(result)
    if work:
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=min(args.num_workers, len(work))) as pool:
            for result in pool.map(_execute_task, work, chunksize=1):
                results.append(result)
    results.sort(key=lambda value: value["task_fingerprint"])
    is_results.sort(key=lambda value: value["is_fingerprint"])
    if len(results) != len(frozen):
        raise RuntimeError("HGP node did not complete every owned task")
    if len(is_results) != len(frozen_is):
        raise RuntimeError("HGP node did not complete every owned IS task")
    if _verify_source(args.source_commit) != source_identity:
        raise RuntimeError("HGP node stage changed the verified source")
    report = {
        "report_version": NODE_RAW_VERSION,
        "contract_version": CONTRACT_VERSION,
        "status": "SUCCESS",
        "node": args.node,
        "source_commit": args.source_commit,
        "archive_sha256": args.archive_sha256,
        "source_manifest_sha256": args.source_manifest_sha256,
        "schedule_sha256": schedule["schedule_sha256"],
        "schedule_file_sha256": _sha256_file(schedule_path),
        "source_identity": source_identity,
        "control_sha256": _sha256_file(control_path),
        "preflight_file_sha256": _sha256_file(preflight_path),
        "artifact_manifest_file_sha256": _sha256_file(
            artifact_manifest_path,
        ),
        "resource_tier": manifest["resource_tier"],
        "expected_count": len(frozen),
        "computed_count": len(results),
        "expected_is_count": len(frozen_is),
        "computed_is_count": len(is_results),
        "reused_count": 0,
        "files": results,
        "importance_sampling_files": is_results,
        "clock_domain": "unsynchronized_local_diagnostic",
        "started_local_unix": started_local_unix,
        "completed_local_unix": time.time(),
        "elapsed_monotonic_seconds": time.monotonic() - started_monotonic,
    }
    _enforce_deadline(schedule, "screen_deadline_unix")
    _write_exclusive_json(output, report)
    print(_canonical_json({
        "node": args.node, "status": "SUCCESS",
        "computed_count": len(results), "reused_count": 0,
        "computed_is_count": len(is_results),
    }))


def _parse_execution_report_paths(values):
    result = {}
    for value in values:
        node, separator, path = value.partition("=")
        if not separator or node in result:
            raise ValueError("execution report must be unique NODE=PATH")
        result[node] = Path(path).resolve(strict=True)
    if set(result) != {"nd-2", "nd-3"}:
        raise ValueError("analysis requires nd-2 and nd-3 execution reports")
    return result


def _validate_execution_reports(
        paths, manifest, records, is_records, raw_root, control_path,
        artifact_manifest_path, source_identity, schedule, schedule_path,
        preflight_path):
    by_node = {
        node: [record for record in records if record["owner"] == node]
        for node in manifest["execution_nodes"]
    }
    is_by_node = {
        node: [record for record in is_records if record["owner"] == node]
        for node in manifest["execution_nodes"]
    }
    reports = {}
    for node in manifest["execution_nodes"]:
        report = _read_json(paths[node])
        expected = by_node[node]
        expected_is = is_by_node[node]
        if (report.get("report_version") != NODE_RAW_VERSION
                or report.get("contract_version") != CONTRACT_VERSION
                or report.get("status") != "SUCCESS"
                or report.get("node") != node
                or report.get("source_commit") != manifest["source_commit"]
                or report.get("archive_sha256") != manifest["archive_sha256"]
                or report.get("source_manifest_sha256")
                != manifest["source_manifest_sha256"]
                or report.get("schedule_sha256")
                != schedule["schedule_sha256"]
                or report.get("schedule_file_sha256")
                != _sha256_file(schedule_path)
                or report.get("source_identity") != source_identity
                or report.get("control_sha256") != _sha256_file(control_path)
                or report.get("preflight_file_sha256")
                != _sha256_file(preflight_path)
                or report.get("artifact_manifest_file_sha256")
                != _sha256_file(artifact_manifest_path)
                or report.get("resource_tier") != manifest["resource_tier"]
                or report.get("expected_count") != len(expected)
                or report.get("computed_count") != len(expected)
                or report.get("expected_is_count") != len(expected_is)
                or report.get("computed_is_count") != len(expected_is)
                or report.get("reused_count") != 0
                or report.get("clock_domain")
                != "unsynchronized_local_diagnostic"
                or not math.isfinite(float(report.get(
                    "started_local_unix", math.nan,
                )))
                or not math.isfinite(float(report.get(
                    "completed_local_unix", math.nan,
                )))
                or not math.isfinite(float(report.get(
                    "elapsed_monotonic_seconds", math.nan,
                )))
                or float(report.get("elapsed_monotonic_seconds", -1.0))
                < 0.0):
            raise ValueError(f"invalid HGP execution report for {node}")
        expected_files = {
            record["task_fingerprint"]: record for record in expected
        }
        files = report.get("files")
        if (not isinstance(files, list) or len(files) != len(expected_files)
                or {value.get("task_fingerprint") for value in files}
                != set(expected_files)):
            raise ValueError(f"HGP execution file set changed for {node}")
        for value in files:
            record = expected_files[value["task_fingerprint"]]
            raw_path = raw_root / record["output_relpath"]
            claim_path = raw_root / ".claims" / (
                record["task_fingerprint"] + ".json"
            )
            if (value.get("output_relpath") != record["output_relpath"]
                    or value.get("sha256") != _sha256_file(raw_path)
                    or int(value.get("size_bytes", -1)) != raw_path.stat().st_size
                    or value.get("claim_sha256") != _sha256_file(claim_path)
                    or SHA256_RE.fullmatch(str(
                        value.get("pipeline_result_sha256", ""),
                    )) is None):
                raise ValueError(f"HGP execution raw evidence changed for {node}")
        expected_is_files = {
            record["is_fingerprint"]: record for record in expected_is
        }
        is_files = report.get("importance_sampling_files")
        if (not isinstance(is_files, list)
                or len(is_files) != len(expected_is_files)
                or {value.get("is_fingerprint") for value in is_files}
                != set(expected_is_files)):
            raise ValueError(f"HGP execution IS set changed for {node}")
        for value in is_files:
            record = expected_is_files[value["is_fingerprint"]]
            raw_path = raw_root / record["output_relpath"]
            claim_path = raw_root / ".claims_is" / (
                record["is_fingerprint"] + ".json"
            )
            if (value.get("output_relpath") != record["output_relpath"]
                    or value.get("sha256") != _sha256_file(raw_path)
                    or int(value.get("size_bytes", -1)) != raw_path.stat().st_size
                    or value.get("claim_sha256") != _sha256_file(claim_path)
                    or SHA256_RE.fullmatch(str(
                        value.get("pipeline_result_sha256", ""),
                    )) is None):
                raise ValueError(f"HGP execution IS evidence changed for {node}")
        reports[node] = report
    return reports


def _analyze_command(args):
    if args.num_workers <= 0:
        raise ValueError("num-workers must be positive")
    source_identity = _verify_provenance(
        args.source_commit, args.archive_sha256,
        args.source_manifest_sha256,
    )
    _require_node(args.node)
    registry_path = Path(args.registry).resolve(strict=True)
    config_path = Path(args.config).resolve(strict=True)
    control_path = Path(args.control).resolve(strict=True)
    artifact_root = Path(args.artifact_root).resolve(strict=True)
    artifact_manifest_path = Path(args.artifact_manifest).resolve(strict=True)
    registry = _load_registry(registry_path)
    config = _load_config(config_path, registry)
    schedule, schedule_path = _validate_schedule(
        args.schedule, registry_path, config_path, args.source_commit,
        args.archive_sha256, args.source_manifest_sha256, config,
    )
    _enforce_deadline(schedule, "analysis_deadline_unix")
    (manifest, manifest_path, records, is_records,
     registry, config) = _validate_control(
        control_path, registry_path, config_path, args.source_commit,
        args.archive_sha256, args.source_manifest_sha256, artifact_root,
        artifact_manifest_path, schedule, schedule_path,
    )
    frozen_analysis = config["execution"]["analysis"]
    if (args.node != frozen_analysis["node"]
            or args.num_workers != int(frozen_analysis["num_workers"])
            or int(frozen_analysis["capacity"]) != args.num_workers
            or manifest.get("analysis") != frozen_analysis):
        raise ValueError("HGP analysis placement differs from frozen runtime")
    preflight_path = Path(args.preflight).resolve(strict=True)
    preflight = _validate_preflight(
        preflight_path, registry_path, config_path, args.source_commit,
        args.archive_sha256, args.source_manifest_sha256,
        artifact_manifest_path, artifact_root, config, schedule,
        schedule_path,
    )
    if (preflight["selected_resource_tier"] != manifest["resource_tier"]
            or preflight.get("source_identity") != source_identity):
        raise ValueError("HGP analysis control differs from preflight tier")
    raw_root = Path(args.raw_root).resolve(strict=True)
    output = Path(args.output).resolve()
    decision_output = Path(args.decision_output).resolve()
    package_output = Path(args.package_output).resolve()
    for path in (output, decision_output, package_output):
        if path.exists():
            raise FileExistsError(f"HGP terminal output already exists: {path}")
    expected_claims = {
        record["task_fingerprint"] + ".json" for record in records
    }
    claim_root = raw_root / ".claims"
    actual_claims = (
        {path.name for path in claim_root.glob("*.json")}
        if claim_root.is_dir() else set()
    )
    if actual_claims != expected_claims:
        raise ValueError("HGP screen raw claim set is incomplete or has extras")
    expected_is_claims = {
        record["is_fingerprint"] + ".json" for record in is_records
    }
    is_claim_root = raw_root / ".claims_is"
    actual_is_claims = (
        {path.name for path in is_claim_root.glob("*.json")}
        if is_claim_root.is_dir() else set()
    )
    if actual_is_claims != expected_is_claims:
        raise ValueError("HGP screen IS claim set is incomplete or has extras")
    staging_root = raw_root / ".staging"
    if staging_root.exists() and any(staging_root.iterdir()):
        raise ValueError("HGP screen has an incomplete staging raw")
    is_staging_root = raw_root / ".staging_is"
    if is_staging_root.exists() and any(is_staging_root.iterdir()):
        raise ValueError("HGP screen has an incomplete staging IS raw")
    for record in records:
        raw_path = raw_root / record["output_relpath"]
        claim_path = raw_root / ".claims" / (
            record["task_fingerprint"] + ".json"
        )
        if not raw_path.is_file() or not claim_path.is_file():
            raise ValueError("HGP screen analysis is missing raw or claim evidence")
        claim = _read_json(claim_path)
        if (claim.get("contract_version") != CONTRACT_VERSION
                or claim.get("kind") != "measurement"
                or claim.get("fingerprint") != record["task_fingerprint"]
                or claim.get("manifest_sha256")
                != manifest["manifest_sha256"]
                or claim.get("node") != record["owner"]):
            raise ValueError("HGP screen raw claim identity mismatch")
        # The pipeline analyzer below independently replays every raw file.
    for record in is_records:
        raw_path = raw_root / record["output_relpath"]
        claim_path = raw_root / ".claims_is" / (
            record["is_fingerprint"] + ".json"
        )
        if not raw_path.is_file() or not claim_path.is_file():
            raise ValueError("HGP screen analysis is missing IS raw or claim")
        claim = _read_json(claim_path)
        if (claim.get("contract_version") != CONTRACT_VERSION
                or claim.get("kind") != "importance_sampling"
                or claim.get("fingerprint") != record["is_fingerprint"]
                or claim.get("manifest_sha256")
                != manifest["manifest_sha256"]
                or claim.get("node") != record["owner"]):
            raise ValueError("HGP screen IS claim identity mismatch")
    execution_paths = _parse_execution_report_paths(args.node_report)
    execution_reports = _validate_execution_reports(
        execution_paths, manifest, records, is_records, raw_root,
        control_path, artifact_manifest_path, source_identity, schedule,
        schedule_path, preflight_path,
    )
    staging_output = output.with_name(
        "." + output.name + f".staging.{os.getpid()}"
    )
    staging_decision = decision_output.with_name(
        "." + decision_output.name + f".staging.{os.getpid()}"
    )
    staging_package = package_output.with_name(
        "." + package_output.name + f".staging.{os.getpid()}"
    )
    if any(path.exists() for path in (
            staging_output, staging_decision, staging_package)):
        raise FileExistsError("HGP screen terminal staging output exists")
    result = _analyze(
        raw_root, manifest_path, registry_path, config_path, staging_output,
        artifact_root, args.num_workers,
    )
    if not staging_output.is_file():
        raise RuntimeError("HGP screen analyzer did not create its output")
    report = _read_json(staging_output)
    report_identity = dict(report)
    report_sha256 = report_identity.pop("report_sha256", None)
    if (report != _jsonable(result)
            or report_sha256 != _sha256_json(report_identity)
            or report.get("source_commit") != args.source_commit
            or report.get("archive_sha256") != args.archive_sha256
            or report.get("source_manifest_sha256")
            != args.source_manifest_sha256
            or report.get("manifest_sha256") != manifest["manifest_sha256"]
            or int(report.get("raw_count", -1)) != len(records)
            or report.get("formal_authorization") is not False
            or report.get("production_authorization") is not False):
        raise ValueError("HGP analysis report identity or self-hash changed")
    allowed_statuses = {
        "DIAGNOSTIC_HARD_PAIR_FOUND", "UNRESOLVED_NO_HP_PASS",
        "UNRESOLVED_MAP_MIXTURE_FAIL",
        "UNRESOLVED_NO_CROSS_MECHANISM_AGREEMENT",
    }
    if report["status"] not in allowed_statuses:
        raise ValueError("HGP analysis produced an unauthorized terminal status")
    _enforce_deadline(schedule, "analysis_deadline_unix")
    report_file_sha256 = _sha256_file(staging_output)
    decision_identity = {
        "decision_version": DECISION_VERSION,
        "contract_version": CONTRACT_VERSION,
        "source_commit": args.source_commit,
        "archive_sha256": args.archive_sha256,
        "source_manifest_sha256": args.source_manifest_sha256,
        "schedule_sha256": schedule["schedule_sha256"],
        "schedule_file_sha256": _sha256_file(schedule_path),
        "artifact_manifest_sha256": _read_json(
            artifact_manifest_path,
        )["artifact_manifest_sha256"],
        "artifact_manifest_file_sha256": _sha256_file(
            artifact_manifest_path,
        ),
        "preflight_file_sha256": _sha256_file(preflight_path),
        "control_file_sha256": _sha256_file(control_path),
        "manifest_sha256": manifest["manifest_sha256"],
        "report_sha256": report["report_sha256"],
        "report_file_sha256": report_file_sha256,
        "status": report["status"],
        "selected_pair": report["selected_pair"],
        "formal_authorization": False,
        "production_authorization": False,
    }
    decision = {
        **decision_identity,
        "decision_sha256": _sha256_json(decision_identity),
    }
    _write_exclusive_json(staging_decision, decision)
    raw_files = sorted(
        [
            {
                "kind": "measurement",
                "fingerprint": value["task_fingerprint"],
                "output_relpath": value["output_relpath"],
                "sha256": value["sha256"],
                "claim_sha256": value["claim_sha256"],
            }
            for node in manifest["execution_nodes"]
            for value in execution_reports[node]["files"]
        ] + [
            {
                "kind": "importance_sampling",
                "fingerprint": value["is_fingerprint"],
                "output_relpath": value["output_relpath"],
                "sha256": value["sha256"],
                "claim_sha256": value["claim_sha256"],
            }
            for node in manifest["execution_nodes"]
            for value in execution_reports[node]["importance_sampling_files"]
        ],
        key=lambda value: (value["kind"], value["fingerprint"]),
    )
    package_identity = {
        "package_version": TERMINAL_PACKAGE_VERSION,
        "contract_version": CONTRACT_VERSION,
        "source_identity": source_identity,
        "schedule_sha256": schedule["schedule_sha256"],
        "schedule_file_sha256": _sha256_file(schedule_path),
        "artifact_manifest_file_sha256": _sha256_file(
            artifact_manifest_path,
        ),
        "preflight_file_sha256": _sha256_file(preflight_path),
        "control_file_sha256": _sha256_file(control_path),
        "execution_report_file_sha256": {
            node: _sha256_file(execution_paths[node])
            for node in manifest["execution_nodes"]
        },
        "report_file_sha256": report_file_sha256,
        "decision_file_sha256": _sha256_file(staging_decision),
        "decision_sha256": decision["decision_sha256"],
        "status": report["status"],
        "raw_file_count": len(raw_files),
        "raw_files": raw_files,
        "formal_authorization": False,
        "production_authorization": False,
    }
    package = {
        **package_identity,
        "package_sha256": _sha256_json(package_identity),
    }
    _write_exclusive_json(staging_package, package)
    _enforce_deadline(schedule, "analysis_deadline_unix")
    if _verify_source(args.source_commit) != source_identity:
        raise RuntimeError("HGP screen analysis changed the verified source")
    for staging, final in (
            (staging_output, output),
            (staging_decision, decision_output),
            (staging_package, package_output)):
        os.link(staging, final)
        staging.unlink()
    print(_canonical_json({
        "status": "SUCCESS", "output": str(output),
        "output_sha256": _sha256_file(output),
        "decision": str(decision_output),
        "decision_sha256": decision["decision_sha256"],
        "terminal_package": str(package_output),
        "package_sha256": package["package_sha256"],
        "terminal_status": report["status"],
        "task_count": manifest["task_count"],
    }))


def _add_provenance_arguments(
        parser, *, artifact_root=True, artifact_manifest=False,
        schedule=True):
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    if artifact_root:
        parser.add_argument("--artifact-root", required=True)
    if artifact_manifest:
        parser.add_argument("--artifact-manifest", required=True)
    if schedule:
        parser.add_argument("--schedule", required=True)
    parser.add_argument("--registry", default=DEFAULT_REGISTRY)
    parser.add_argument("--config", default=DEFAULT_CONFIG)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    schedule = subparsers.add_parser("build-schedule")
    _add_provenance_arguments(
        schedule, artifact_root=False, artifact_manifest=False,
        schedule=False,
    )
    schedule.add_argument("--run-id", required=True)
    schedule.add_argument("--clock-authority-json", required=True)
    schedule.add_argument("--output", required=True)
    schedule.set_defaults(function=_build_schedule)

    artifacts = subparsers.add_parser("build-artifacts")
    _add_provenance_arguments(artifacts, artifact_manifest=False)
    artifacts.add_argument("--output", required=True)
    artifacts.set_defaults(function=_build_artifacts_command)

    preflight = subparsers.add_parser("preflight-node")
    preflight.add_argument("node", choices=EXPECTED_PREFLIGHT_NODES)
    _add_provenance_arguments(preflight, artifact_manifest=True)
    preflight.add_argument("--output-root", required=True)
    preflight.set_defaults(function=_preflight_node)

    combine = subparsers.add_parser("combine-preflight")
    _add_provenance_arguments(combine, artifact_manifest=True)
    combine.add_argument("--node-report", action="append", default=[],
                         metavar="NODE=PATH")
    combine.add_argument("--output", required=True)
    combine.set_defaults(function=_combine_preflight)

    control = subparsers.add_parser("build-control")
    _add_provenance_arguments(control, artifact_manifest=True)
    control.add_argument("--preflight", required=True)
    control.add_argument("--output", required=True)
    control.set_defaults(function=_build_control)

    run_node = subparsers.add_parser("run-node")
    run_node.add_argument("node")
    _add_provenance_arguments(run_node, artifact_manifest=True)
    run_node.add_argument("--control", required=True)
    run_node.add_argument("--preflight", required=True)
    run_node.add_argument("--raw-root", required=True)
    run_node.add_argument("--output", required=True)
    run_node.add_argument("--num-workers", type=int, required=True)
    run_node.set_defaults(function=_run_node)

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("node", choices=("nd-3",))
    _add_provenance_arguments(analyze, artifact_manifest=True)
    analyze.add_argument("--control", required=True)
    analyze.add_argument("--preflight", required=True)
    analyze.add_argument("--node-report", action="append", default=[],
                         metavar="NODE=PATH")
    analyze.add_argument("--raw-root", required=True)
    analyze.add_argument("--output", required=True)
    analyze.add_argument("--decision-output", required=True)
    analyze.add_argument("--package-output", required=True)
    analyze.add_argument("--num-workers", type=int, required=True)
    analyze.set_defaults(function=_analyze_command)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    args.function(args)


if __name__ == "__main__":
    main()
