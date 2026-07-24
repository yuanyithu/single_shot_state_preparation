"""Fresh direct-block T1 workflow with a two-length cold runtime estimator."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import hashlib
from importlib import import_module
import math
import multiprocessing
from pathlib import Path
import time

import numpy as np


_base = import_module(
    "data.expander_code.exp102.validation."
    "055_q0_random_full_column_direct_block_t1_m8_20260724.workflow"
)
_legacy = _base._legacy
_direct = _base._direct
_conditional = _base._conditional

CONTRACT_VERSION = "exp102.q0_random_full_column_direct_block.t1_m8.v2"
SCHEDULE_VERSION = "exp102.q0_random_full_column_direct_block.t1_m8.schedule.v2"
PREFLIGHT_VERSION = "exp102.q0_random_full_column_direct_block.t1_m8.preflight.v2"
RAW_VERSION = "exp102.q0_random_full_column_direct_block.t1_m8.raw.v2"
NODE_REPORT_VERSION = (
    "exp102.q0_random_full_column_direct_block.t1_m8.node_report.v2"
)
CONTROL_VERSION = "exp102.q0_random_full_column_direct_block.t1_m8.control.v2"
REPORT_VERSION = "exp102.q0_random_full_column_direct_block.t1_m8.report.v2"
ROOT = Path(__file__).resolve().parent
EXP102_ROOT = ROOT.parents[1]
CONFIG_PATH = (
    EXP102_ROOT / "config/q0_random_full_column_direct_block.t1_m8.v2.json"
)
SOURCE_CONTROL_DIR = ROOT / "control"
FAMILIES = ("P", "U", "M0", "M1", "S")

_ORIGINAL_LOAD_CONTROL = _base._ORIGINAL_LOAD_CONTROL
_ORIGINAL_BUILD_MASS = _base._ORIGINAL_BUILD_MASS
_MASS_BY_KEY = {}

_load_canonical_json = _base._load_canonical_json
_verify_self_hash = _base._verify_self_hash
_source_identity = _base._source_identity
_require = _base._require
canonical_json = _base.canonical_json
sha256_file = _base.sha256_file
sha256_json = _base.sha256_json


EXPECTED_RESOURCE = {
    "allowed_nodes": ["nd-1", "nd-2", "nd-3"],
    "burn_updates": 2048,
    "fixed_workers_per_node": 4,
    "measurement_updates": 8192,
    "runtime_estimator": "two_length_cold_process_sampling_replay_v1",
    "runtime_probe_concurrency": 4,
    "runtime_probe_families": ["P", "M0", "S", "U"],
    "runtime_probe_long_burn_updates": 16,
    "runtime_probe_long_measurement_updates": 256,
    "runtime_probe_short_burn_updates": 8,
    "runtime_probe_short_measurement_updates": 128,
    "safety_factor": 2.0,
    "trajectory_wall_cap_seconds": 7200.0,
}


def _load_config():
    config = _load_canonical_json(CONFIG_PATH)
    _require(
        config["version"] == config["contract_version"] == CONTRACT_VERSION,
        "direct-block T1 v2 contract version changed",
    )
    _require(
        config["config_version"]
        == "exp102.q0_random_full_column_direct_block.t1_m8.config.v2",
        "direct-block T1 v2 config version changed",
    )
    _require(
        tuple(config["initialization"]["families"]) == FAMILIES
        and config["initialization"]["trajectories_per_family"] == 8,
        "initialization panel changed",
    )
    _require(config["resource"] == EXPECTED_RESOURCE, "resource design changed")
    implementation = config["implementation"]
    _require(
        implementation["method_id"]
        == _direct.RANDOM_FULL_COLUMN_DIRECT_BLOCK_METHOD_ID,
        "direct-block method identity changed",
    )
    for path_field, hash_field in (
        ("full_column_relpath", "full_column_file_sha256"),
        ("random_full_column_relpath", "random_full_column_file_sha256"),
        ("portable_reference_relpath", "portable_reference_file_sha256"),
    ):
        path = EXP102_ROOT / implementation[path_field]
        _require(
            sha256_file(path) == implementation[hash_field],
            f"implementation binding changed: {path_field}",
        )
    reference = _load_canonical_json(
        EXP102_ROOT / implementation["portable_reference_relpath"]
    )
    _verify_self_hash(reference, "reference_sha256")
    _require(
        reference["reference_sha256"]
        == implementation["portable_reference_sha256"],
        "portable reference identity changed",
    )
    aggregate = _load_canonical_json(
        EXP102_ROOT
        / "validation/054_q0_random_full_column_direct_block_preflight_20260724"
        / "remote_evidence/preflight/aggregate.json"
    )
    _verify_self_hash(aggregate, "aggregate_sha256")
    _require(
        aggregate["status"] == "PASS"
        and aggregate["aggregate_sha256"]
        == implementation["validation_054_aggregate_sha256"],
        "validation-054 terminal authorization changed",
    )
    audit = _load_canonical_json(
        EXP102_ROOT
        / "validation/055_q0_random_full_column_direct_block_t1_m8_20260724"
        / "independent_preflight_audit.json"
    )
    _verify_self_hash(audit, "audit_sha256")
    _require(
        audit["status"]
        == "INDEPENDENT_AUDIT_PASS_PORTABLE_PASS_T1_RUNTIME_EXHAUSTED_CONFIRMED"
        and audit["audit_sha256"]
        == implementation["validation_055_terminal_audit_sha256"],
        "validation-055 terminal audit changed",
    )
    return config, sha256_file(CONFIG_PATH)


def _load_control(control_dir, config, config_sha):
    previous = _legacy.CONTROL_VERSION
    _legacy.CONTROL_VERSION = CONTROL_VERSION
    try:
        return _ORIGINAL_LOAD_CONTROL(control_dir, config, config_sha)
    finally:
        _legacy.CONTROL_VERSION = previous


def _build_mass(H, p, engine="numba"):
    mass = np.ascontiguousarray(
        _ORIGINAL_BUILD_MASS(H, p, engine=engine), dtype=np.float64,
    )
    _MASS_BY_KEY[(int(H.shape[0]), float(p))] = mass
    return mass


def _build_cache(rows, p):
    key = (int(rows), float(p))
    _require(key in _MASS_BY_KEY, "direct-block mass must precede its cache")
    return _conditional.build_full_column_direct_block_cache(
        rows, p, _MASS_BY_KEY[key],
    )


def _fit_runtime_component(short_seconds, long_seconds, short_updates,
                           long_updates, target_updates):
    values = (
        short_seconds, long_seconds, short_updates, long_updates, target_updates,
    )
    if not all(math.isfinite(float(value)) for value in values):
        return {"stable": False}
    if not (short_seconds > 0.0 and long_seconds > short_seconds
            and 0 < short_updates < long_updates <= target_updates):
        return {"stable": False}
    slope = (long_seconds - short_seconds) / (long_updates - short_updates)
    if not math.isfinite(slope) or slope <= 0.0:
        return {"stable": False}
    raw_intercept = short_seconds - slope * short_updates
    intercept = max(0.0, raw_intercept)
    projection = intercept + slope * target_updates
    return {
        "intercept_seconds": float(intercept),
        "raw_intercept_seconds": float(raw_intercept),
        "slope_seconds_per_update": float(slope),
        "stable": True,
        "target_seconds_before_safety": float(projection),
    }


_PROBE_CONTEXT = None
_PROBE_SPEC = None


def _probe_initial_state(context, family, index):
    if family == "P":
        return context["fixed_states"][0].copy()
    if family == "M0":
        return context["fixed_states"][1].copy()
    if family == "S":
        return context["fixed_states"][3].copy()
    if family == "U":
        config = context["config"]
        seed = _legacy.derive_seed(
            config["seed_namespace"], context["config_sha"],
            context["metadata"]["control_content_sha256"],
            "runtime_preflight", "initialization", family, index,
        )
        return _legacy.uniform_hard_coset_state(
            context["model"], context["syndrome"], seed,
        )
    raise RuntimeError("unknown runtime probe family")


def _probe_worker(index):
    context = _PROBE_CONTEXT
    spec = _PROBE_SPEC
    config = context["config"]
    family = config["resource"]["runtime_probe_families"][index]
    sampler = _direct.RandomFullColumnDirectBlockConfig(
        p=0.04,
        burn_updates=spec["burn_updates"],
        measurement_updates=spec["measurement_updates"],
    )
    prefix = (
        config["seed_namespace"], context["config_sha"],
        context["metadata"]["control_content_sha256"],
        "runtime_preflight", family, index,
    )
    initial = _probe_initial_state(context, family, index)
    workspace = _conditional.build_full_column_direct_block_workspace(
        context["cache"]
    )
    sampling_start = time.perf_counter()
    raw = _direct.run_random_full_column_direct_block_trajectory(
        context["model"], context["frame"], context["H"], context["syndrome"],
        sampler, initial, _legacy.derive_seed(*prefix, "burn"),
        _legacy.derive_seed(*prefix, "measurement"),
        _legacy.derive_seed(*prefix, "observation"), mass=context["mass"],
        cache=context["cache"], workspace=workspace,
    )
    sampling_seconds = time.perf_counter() - sampling_start
    replay_start = time.perf_counter()
    replay_ok = _direct.replay_random_full_column_direct_block_trajectory(
        context["model"], context["frame"], context["H"], context["syndrome"],
        sampler, initial, _legacy.derive_seed(*prefix, "burn"),
        _legacy.derive_seed(*prefix, "measurement"),
        _legacy.derive_seed(*prefix, "observation"), raw, mass=context["mass"],
        cache=context["cache"], workspace=workspace,
    )
    replay_seconds = time.perf_counter() - replay_start
    _require(replay_ok is True, "runtime probe replay failed")
    discrete_names = (
        "burn__counters", "burn__final_b_columns", "burn__selected_columns",
        "burn__old_columns", "burn__new_columns", "measurement__counters",
        "measurement__selected_columns", "measurement__old_columns",
        "measurement__new_columns", "measurement__b_columns",
        "measurement__labels", "measurement__states_packed",
        "measurement__weights", "final_b_columns", "final_state_packed",
    )
    digest = hashlib.sha256()
    for name in discrete_names:
        value = np.ascontiguousarray(raw[name])
        digest.update(name.encode("ascii") + b"\0")
        digest.update(value.dtype.str.encode("ascii") + b"\0")
        digest.update(np.asarray(value.shape, dtype=">u8").tobytes())
        digest.update(value.tobytes(order="C"))
    return {
        "family": family,
        "index": index,
        "portable_transcript_sha256": digest.hexdigest(),
        "replay_seconds": float(replay_seconds),
        "sampling_seconds": float(sampling_seconds),
        "total_probe_updates": int(
            sampler.burn_updates + sampler.measurement_updates
        ),
    }


def _run_probe_batch(context, spec):
    global _PROBE_CONTEXT, _PROBE_SPEC
    _PROBE_CONTEXT = context
    _PROBE_SPEC = spec
    workers = context["config"]["resource"]["runtime_probe_concurrency"]
    try:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("fork"),
        ) as executor:
            rows = list(executor.map(_probe_worker, range(workers)))
    finally:
        _PROBE_CONTEXT = None
        _PROBE_SPEC = None
    rows.sort(key=lambda row: row["index"])
    return rows


def preflight_node(args):
    run_root, schedule, context = _legacy._load_schedule(args.run_root, args)
    _require(
        args.node in context["config"]["resource"]["allowed_nodes"],
        "preflight node is not allowed",
    )
    context["mass"] = _build_mass(context["H"], 0.04, engine="numba")
    context["cache"] = _build_cache(context["H"].shape[0], 0.04)
    resource = context["config"]["resource"]
    short_spec = {
        "burn_updates": resource["runtime_probe_short_burn_updates"],
        "measurement_updates": resource["runtime_probe_short_measurement_updates"],
        "name": "short",
    }
    long_spec = {
        "burn_updates": resource["runtime_probe_long_burn_updates"],
        "measurement_updates": resource["runtime_probe_long_measurement_updates"],
        "name": "long",
    }
    short_rows = _run_probe_batch(context, short_spec)
    long_rows = _run_probe_batch(context, long_spec)
    short_updates = short_spec["burn_updates"] + short_spec["measurement_updates"]
    long_updates = long_spec["burn_updates"] + long_spec["measurement_updates"]
    target_updates = resource["burn_updates"] + resource["measurement_updates"]
    fits = []
    for short, long in zip(short_rows, long_rows, strict=True):
        _require(
            short["index"] == long["index"]
            and short["family"] == long["family"]
            and short["total_probe_updates"] == short_updates
            and long["total_probe_updates"] == long_updates,
            "runtime probe pairing changed",
        )
        sampling_fit = _fit_runtime_component(
            short["sampling_seconds"], long["sampling_seconds"],
            short_updates, long_updates, target_updates,
        )
        replay_fit = _fit_runtime_component(
            short["replay_seconds"], long["replay_seconds"],
            short_updates, long_updates, target_updates,
        )
        stable = sampling_fit["stable"] and replay_fit["stable"]
        projected = None
        if stable:
            projected = resource["safety_factor"] * (
                sampling_fit["target_seconds_before_safety"]
                + replay_fit["target_seconds_before_safety"]
            )
        fits.append({
            "family": short["family"],
            "index": short["index"],
            "long": long,
            "projected_replay_inclusive_trajectory_seconds": (
                None if projected is None else float(projected)
            ),
            "replay_fit": replay_fit,
            "sampling_fit": sampling_fit,
            "short": short,
            "stable": stable,
        })
    estimator_stable = all(row["stable"] for row in fits)
    worst_projection = (
        max(row["projected_replay_inclusive_trajectory_seconds"] for row in fits)
        if estimator_stable else None
    )
    pass_runtime = (
        estimator_stable
        and worst_projection <= resource["trajectory_wall_cap_seconds"]
    )
    if not estimator_stable:
        status = "RUNTIME_ESTIMATOR_UNSTABLE"
    elif not pass_runtime:
        status = "RUNTIME_EXHAUSTED"
    else:
        status = "PASS"
    core = {
        "config_sha256": context["config_sha"],
        "control_content_sha256": context["metadata"]["control_content_sha256"],
        "estimator_stable": estimator_stable,
        "fits": fits,
        "mass_sha256": hashlib.sha256(
            np.ascontiguousarray(context["mass"], dtype=">f8").tobytes()
        ).hexdigest(),
        "node": args.node,
        "pass_runtime": pass_runtime,
        "preflight_version": PREFLIGHT_VERSION,
        "probe_design": {
            "concurrency": resource["runtime_probe_concurrency"],
            "estimator": resource["runtime_estimator"],
            "families": resource["runtime_probe_families"],
            "long": long_spec,
            "safety_factor": resource["safety_factor"],
            "short": short_spec,
            "target_updates": target_updates,
        },
        "schedule_sha256": schedule["schedule_sha256"],
        "source_identity": schedule["source_identity"],
        "status": status,
        "worst_projected_replay_inclusive_trajectory_seconds": worst_projection,
    }
    report = {**core, "preflight_sha256": sha256_json(core)}
    output = run_root / f"preflight/{args.node}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    _legacy._exclusive_json(output, report)
    print(canonical_json(report))


def combine_preflight(args):
    run_root, schedule, context = _legacy._load_schedule(args.run_root, args)
    reports = []
    for node in context["config"]["resource"]["allowed_nodes"]:
        report = _load_canonical_json(run_root / f"preflight/{node}.json")
        _verify_self_hash(report, "preflight_sha256")
        _require(
            report["node"] == node
            and report["schedule_sha256"] == schedule["schedule_sha256"]
            and report["source_identity"] == schedule["source_identity"],
            "node preflight identity mismatch",
        )
        reports.append(report)
    mass_hashes = {report["mass_sha256"] for report in reports}
    transcript_catalogs = {
        tuple(
            (
                fit["family"],
                fit["short"]["portable_transcript_sha256"],
                fit["long"]["portable_transcript_sha256"],
            )
            for fit in report["fits"]
        )
        for report in reports
    }
    designs = {canonical_json(report["probe_design"]) for report in reports}
    exact_consensus = (
        len(mass_hashes) == 1
        and len(transcript_catalogs) == 1
        and len(designs) == 1
    )
    if not exact_consensus:
        status = "CONFLICT"
    elif not all(report["estimator_stable"] for report in reports):
        status = "RUNTIME_ESTIMATOR_UNSTABLE"
    elif not all(report["pass_runtime"] for report in reports):
        status = "RUNTIME_EXHAUSTED"
    else:
        status = "PASS"
    finite_projections = [
        report["worst_projected_replay_inclusive_trajectory_seconds"]
        for report in reports
        if report["worst_projected_replay_inclusive_trajectory_seconds"] is not None
    ]
    core = {
        "config_sha256": context["config_sha"],
        "exact_consensus": exact_consensus,
        "node_preflight_sha256": {
            report["node"]: report["preflight_sha256"] for report in reports
        },
        "node_status": {report["node"]: report["status"] for report in reports},
        "preflight_version": PREFLIGHT_VERSION,
        "schedule_sha256": schedule["schedule_sha256"],
        "source_identity": schedule["source_identity"],
        "status": status,
        "worst_projected_replay_inclusive_trajectory_seconds": (
            max(finite_projections) if finite_projections else None
        ),
    }
    aggregate = {**core, "preflight_sha256": sha256_json(core)}
    _legacy._exclusive_json(run_root / "preflight/aggregate.json", aggregate)
    print(canonical_json(aggregate))


def _configure_legacy():
    bindings = {
        "CONTRACT_VERSION": CONTRACT_VERSION,
        "SCHEDULE_VERSION": SCHEDULE_VERSION,
        "PREFLIGHT_VERSION": PREFLIGHT_VERSION,
        "RAW_VERSION": RAW_VERSION,
        "NODE_REPORT_VERSION": NODE_REPORT_VERSION,
        "CONTROL_VERSION": CONTROL_VERSION,
        "ROOT": ROOT,
        "EXP102_ROOT": EXP102_ROOT,
        "CONFIG_PATH": CONFIG_PATH,
        "SOURCE_CONTROL_DIR": SOURCE_CONTROL_DIR,
        "FAMILIES": FAMILIES,
        "RANDOM_FULL_COLUMN_METHOD_ID": (
            _direct.RANDOM_FULL_COLUMN_DIRECT_BLOCK_METHOD_ID
        ),
        "RANDOM_FULL_COLUMN_VERSION": (
            _direct.RANDOM_FULL_COLUMN_DIRECT_BLOCK_VERSION
        ),
        "RandomFullColumnConfig": _direct.RandomFullColumnDirectBlockConfig,
        "build_classical_coset_mass": _build_mass,
        "build_full_column_candidate_cache": _build_cache,
        "build_full_column_workspace": (
            _conditional.build_full_column_direct_block_workspace
        ),
        "run_random_full_column_trajectory": (
            _direct.run_random_full_column_direct_block_trajectory
        ),
        "replay_random_full_column_trajectory": (
            _direct.replay_random_full_column_direct_block_trajectory
        ),
        "_load_config": _load_config,
        "_load_control": _load_control,
        "preflight_node": preflight_node,
        "combine_preflight": combine_preflight,
    }
    for name, value in bindings.items():
        setattr(_legacy, name, value)


def main():
    _configure_legacy()
    _legacy.main()


if __name__ == "__main__":
    main()
