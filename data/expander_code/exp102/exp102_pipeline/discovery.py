"""Isolated PT-v2 discovery runner and fail-closed analyzer.

Discovery evidence is deliberately incompatible with the formal pilot raw
schema.  It can nominate two fixed configurations, but it can never certify a
production freezer.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import time

import numpy as np

from .diagnostics import evaluate_gate, validate_transport_counters
from .io import atomic_json, atomic_npz, canonical_json, sha256_file, sha256_json
from .ladders import (
    PT_CANDIDATE_FIELDS,
    ladder_fingerprint,
    make_piecewise_density_ladder,
    make_pt_candidate,
    q0_config_from_candidate,
    validate_ladder_record,
    validate_pt_candidate,
)
from .labels import initial_labels
from .q0_pt import expected_swap_attempts, run_q0_pt_instance
from .registry import load_frozen_code, load_registry
from .seeds import derive_seed
from .worker import build_model


DISCOVERY_VERSION = "exp102.discovery.v2"
DISCOVERY_RAW_VERSION = "exp102.discovery.raw.v2"
DISCOVERY_REPORT_VERSION = "exp102.discovery.report.v2"
DISCOVERY_PT_VERSION = "exp102.q0_pt.v2"
DISCOVERY_STAGES = ("screen", "transport", "confirmation")
OLD_RUN_ID = "exp102_pilot_20260720_2b01d9d"
OLD_SOURCE_COMMIT = "2b01d9dcb463ec47a1b30202fc9105430b95e18c"
OLD_ATTEMPT = 22
OLD_FAILED_CELLS = (
    ("m05_c00", 0.04, 2),
    ("m05_c05", 0.04, 3),
    ("m06_c00", 0.04, 0),
    ("m06_c05", 0.04, 2),
    ("m07_c00", 0.04, 0),
    ("m07_c00", 0.04, 1),
    ("m07_c05", 0.04, 1),
    ("m08_c06", 0.04, 0),
    ("m08_c07", 0.04, 3),
)
TRANSPORT_CELLS = (
    ("m06_c00", 0.04, 0),
    ("m08_c06", 0.04, 0),
)
TUNING_P_VALUES = (0.04, 0.07, 0.10)
ROUND_TIERS = ((2000, 8000), (4000, 16000), (8000, 32000))
BASE_GATE = {
    "min_swap_rate": 0.05,
    "min_swap_accepts": 20,
    "min_round_trips": 0,
    "min_sector_changing_round_trips": 0,
    "min_hot_logical_rate": 0.01,
    "min_hot_logical_accepts_per_basis": 20,
    "max_rhat": np.inf,
    "min_ess": 0,
    "max_instance_mean_spread": np.inf,
}
DISCOVERY_RAW_FIELDS = {
    "raw_version", "task_fingerprint", "namespace", "stage", "cell_json",
    "code_id", "m", "p", "disorder_index", "candidate_json",
    "ladder_fingerprint", "swap_sweeps_per_round", "valid", "failure_reason",
    "labels", "swap_attempts", "swap_accepts", "swap_rates",
    "logical_attempts", "logical_accepts", "logical_rates",
    "hot_touches", "hot_updated_visits", "uncertified_round_trips",
    "round_trips", "sector_changing_round_trips",
    "hot_touches_per_replica", "hot_updated_visits_per_replica",
    "uncertified_round_trips_per_replica", "round_trips_per_replica",
    "sector_changing_round_trips_per_replica", "residual", "rhat", "ess",
    "constant_status", "uniform_seed", "instance_seeds", "core_seconds",
    "wall_seconds", "engine", "source_commit", "model_fingerprint",
    "registry_sha256", "discovery_config_sha256", "section_fingerprint",
    "logical_frame_fingerprint",
}


def default_discovery_config(registry):
    registry_sha256 = registry["registry_sha256"]
    ladders = [
        make_piecewise_density_ladder("D0", 0.45, 88, 6),
        make_piecewise_density_ladder("D1", 0.45, 104, 4),
        make_piecewise_density_ladder("D2", 0.475, 96, 5),
        make_piecewise_density_ladder("D3", 0.49, 112, 4),
        make_piecewise_density_ladder("D4", 0.49, 128, 4),
    ]
    old_cells = [_cell(code_id, p, disorder, "attempt022")
                 for code_id, p, disorder in OLD_FAILED_CELLS]
    transport_cells = [_cell(code_id, p, disorder, "attempt022")
                       for code_id, p, disorder in TRANSPORT_CELLS]
    fresh_cells = _fresh_confirmation_cells(registry)
    return {
        "discovery_version": DISCOVERY_VERSION,
        "pt_contract_version": DISCOVERY_PT_VERSION,
        "registry_sha256": registry_sha256,
        "old_evidence": {
            "run_id": OLD_RUN_ID,
            "source_commit": OLD_SOURCE_COMMIT,
            "attempt": OLD_ATTEMPT,
        },
        "ladders": ladders,
        "screen": {
            "cells": old_cells,
            "burn_rounds": 500,
            "measurement_rounds": 2000,
            "swap_sweeps": [1],
            "min_swap_rate": 0.20,
        },
        "transport": {
            "cells": transport_cells,
            "burn_rounds": 2000,
            "measurement_rounds": 8000,
            "swap_sweeps": [4, 16, 64],
            "conditional_swap_sweeps": 128,
            "min_swap_rate": 0.05,
            "min_round_trips": 1,
            "min_sector_changing_round_trips": 1,
        },
        "confirmation": {
            "cells": old_cells + fresh_cells,
            "fresh_cells": fresh_cells,
            "round_tiers": [list(tier) for tier in ROUND_TIERS],
            "min_swap_rate": 0.05,
            "min_round_trips": 8,
            "min_sector_changing_round_trips": 4,
            "max_rhat": 1.02,
            "min_ess": 400,
        },
    }


def write_default_discovery_config(registry_path, output_path):
    config = default_discovery_config(load_registry(registry_path))
    atomic_json(output_path, config)
    return config


def load_discovery_config(path, registry=None):
    raw = json.loads(Path(path).read_text(encoding="ascii"))
    expected_top = {
        "discovery_version", "pt_contract_version", "registry_sha256",
        "old_evidence", "ladders", "screen", "transport", "confirmation",
    }
    if set(raw) != expected_top or raw["discovery_version"] != DISCOVERY_VERSION:
        raise ValueError("discovery config schema/version mismatch")
    if raw["pt_contract_version"] != DISCOVERY_PT_VERSION:
        raise ValueError("discovery config PT contract mismatch")
    if registry is not None and raw["registry_sha256"] != registry["registry_sha256"]:
        raise ValueError("discovery config registry mismatch")
    if raw["old_evidence"] != {
        "run_id": OLD_RUN_ID, "source_commit": OLD_SOURCE_COMMIT, "attempt": OLD_ATTEMPT,
    }:
        raise ValueError("discovery old-evidence identity mismatch")
    ladders = [validate_ladder_record(record) for record in raw["ladders"]]
    if [record["ladder_id"] for record in ladders] != [f"D{i}" for i in range(5)]:
        raise ValueError("discovery ladder order is not frozen")
    # Normal runners always supply the registry so the prospective fresh-cell
    # panel is independently reconstructed rather than trusted from JSON.
    if registry is not None and raw != default_discovery_config(registry):
        raise ValueError("discovery config differs from the frozen protocol")
    normalized = dict(raw)
    normalized["ladders"] = ladders
    normalized["discovery_config_sha256"] = sha256_json(raw)
    normalized["config_path"] = str(Path(path).resolve())
    return normalized


def discovery_task_identity(registry, config, source_commit, stage, cell, candidate):
    if re.fullmatch(r"[0-9a-f]{40}", str(source_commit)) is None:
        raise ValueError("discovery source commit must be a full lowercase Git SHA")
    if stage not in DISCOVERY_STAGES:
        raise ValueError("unknown discovery stage")
    cell = _validate_cell(cell, config[stage]["cells"])
    candidate = validate_pt_candidate(candidate)
    _validate_stage_candidate(config, stage, candidate)
    identity = {
        "namespace": f"discovery_v2_{stage}",
        "raw_version": DISCOVERY_RAW_VERSION,
        "stage": stage,
        "cell": cell,
        "candidate": candidate,
        "ladder_fingerprint": ladder_fingerprint(candidate),
        "registry_sha256": registry["registry_sha256"],
        "discovery_config_sha256": config["discovery_config_sha256"],
        "source_commit": source_commit,
        "engine": "numba",
    }
    return identity


def screen_candidates(config):
    section = config["screen"]
    return [make_pt_candidate(
        ladder, section["burn_rounds"], section["measurement_rounds"], 1,
    ) for ladder in config["ladders"]]


def transport_candidates(config, ladder_ids, include_conditional=False):
    section = config["transport"]
    swap_sweeps = list(section["swap_sweeps"])
    if include_conditional:
        swap_sweeps.append(section["conditional_swap_sweeps"])
    ladders = {record["ladder_id"]: record for record in config["ladders"]}
    if not set(ladder_ids) <= set(ladders):
        raise ValueError("unknown transport ladder id")
    return [
        make_pt_candidate(
            ladders[ladder_id], section["burn_rounds"],
            section["measurement_rounds"], swap_sweep,
        )
        for ladder_id in ladder_ids for swap_sweep in swap_sweeps
    ]


def confirmation_candidate(config, ladder_id, swap_sweeps, tier):
    ladders = {record["ladder_id"]: record for record in config["ladders"]}
    tiers = [tuple(value) for value in config["confirmation"]["round_tiers"]]
    if ladder_id not in ladders or tuple(tier) not in tiers:
        raise ValueError("unknown confirmation candidate")
    return make_pt_candidate(ladders[ladder_id], tier[0], tier[1], swap_sweeps)


def task_manifest(registry_path, config_path, source_commit, stage, candidates, output_path):
    registry = load_registry(registry_path)
    config = load_discovery_config(config_path, registry)
    normalized_candidates = [validate_pt_candidate(candidate) for candidate in candidates]
    tasks = [
        discovery_task_identity(registry, config, source_commit, stage, cell, candidate)
        for candidate in normalized_candidates for cell in config[stage]["cells"]
    ]
    if len({sha256_json(task) for task in tasks}) != len(tasks):
        raise ValueError("discovery task manifest contains duplicate identities")
    manifest = {
        "manifest_version": "exp102.discovery.tasks.v2",
        "stage": stage,
        "source_commit": source_commit,
        "registry_sha256": registry["registry_sha256"],
        "discovery_config_sha256": config["discovery_config_sha256"],
        "tasks": tasks,
    }
    atomic_json(output_path, manifest)
    return manifest


def run_discovery_cell(registry_path, config_path, source_commit, task, output_path):
    registry = load_registry(registry_path)
    config = load_discovery_config(config_path, registry)
    stage = task.get("stage")
    expected = discovery_task_identity(
        registry, config, source_commit, stage, task.get("cell"), task.get("candidate"),
    )
    if task != expected:
        raise ValueError("discovery task identity is noncanonical or tampered")
    fingerprint = sha256_json(expected)
    output_path = Path(output_path)
    if output_path.exists():
        record = validate_discovery_raw(
            output_path, registry, config, expected_source_commit=source_commit,
        )
        if record["task_fingerprint"] != fingerprint:
            raise ValueError("existing discovery output conflicts with the task")
        return "reused"

    cell = expected["cell"]
    registry_loaded, code, H = load_frozen_code(registry_path, cell["code_id"])
    if registry_loaded["registry_sha256"] != registry["registry_sha256"]:
        raise ValueError("discovery registry changed while loading a code")
    model, frame = build_model(H)
    candidate = expected["candidate"]
    pt_config = q0_config_from_candidate(candidate)
    uniform_seed = _uniform_seed(registry, code, cell)
    uniforms = np.random.Generator(np.random.PCG64(uniform_seed)).random(model.num_qubits)
    epsilon = (uniforms < float(cell["p"])).astype(np.uint8)
    syndrome = (model.H_check.astype(np.int64) @ epsilon.astype(np.int64) % 2).astype(np.uint8)

    results = []
    instance_seeds = []
    wall_start, core_start = time.monotonic(), time.process_time()
    for instance, initial_label in enumerate(initial_labels(model.k)):
        seed = _trajectory_seed(
            registry, config, source_commit, stage, code, cell, candidate, instance,
        )
        result = run_q0_pt_instance(
            model, frame, syndrome, float(cell["p"]), pt_config, seed,
            initial_label, engine="numba",
        )
        result["seed"] = seed
        results.append(result)
        instance_seeds.append(seed)
    core_seconds = time.process_time() - core_start
    wall_seconds = time.monotonic() - wall_start
    valid, failures, rhats, esses, statuses = evaluate_gate(
        results, _discovery_gate(config, stage), model.k,
        require_trace_gate=stage == "confirmation",
    )
    arrays = _result_arrays(results)
    atomic_npz(
        output_path,
        raw_version=np.array(DISCOVERY_RAW_VERSION),
        task_fingerprint=np.array(fingerprint),
        namespace=np.array(expected["namespace"]),
        stage=np.array(stage),
        cell_json=np.array(canonical_json(cell)),
        code_id=np.array(code["code_id"]),
        m=np.array(code["m"], dtype=np.int8),
        p=np.array(cell["p"]),
        disorder_index=np.array(cell["disorder_index"], dtype=np.int16),
        candidate_json=np.array(canonical_json(candidate)),
        ladder_fingerprint=np.array(expected["ladder_fingerprint"]),
        swap_sweeps_per_round=np.array(candidate["swap_sweeps_per_round"], dtype=np.int16),
        valid=np.array(valid),
        failure_reason=np.array(";".join(failures), dtype="U4096"),
        **arrays,
        rhat=rhats,
        ess=esses,
        constant_status=statuses,
        uniform_seed=np.array(uniform_seed, dtype=np.int64),
        instance_seeds=np.asarray(instance_seeds, dtype=np.int64),
        core_seconds=np.array(core_seconds),
        wall_seconds=np.array(wall_seconds),
        engine=np.array("numba"),
        source_commit=np.array(source_commit),
        model_fingerprint=np.array(sha256_json({"n": model.num_qubits, "k": model.k})),
        registry_sha256=np.array(registry["registry_sha256"]),
        discovery_config_sha256=np.array(config["discovery_config_sha256"]),
        section_fingerprint=np.array(code["section_fingerprint"]),
        logical_frame_fingerprint=np.array(code["logical_frame_fingerprint"]),
    )
    return "computed"


def validate_discovery_raw(path, registry, config, expected_source_commit=None):
    path = Path(path)
    code_by_id = {row["code_id"]: row for row in registry["codes"]}
    try:
        context = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise ValueError(f"cannot read discovery raw {path}: {exc}") from exc
    with context as data:
        if set(data.files) != DISCOVERY_RAW_FIELDS:
            raise ValueError("discovery raw schema mismatch")
        if str(_scalar(data, "raw_version")) != DISCOVERY_RAW_VERSION:
            raise ValueError("discovery raw version mismatch")
        stage = str(_scalar(data, "stage"))
        if stage not in DISCOVERY_STAGES:
            raise ValueError("unknown discovery raw stage")
        code_id = str(_scalar(data, "code_id"))
        if code_id not in code_by_id:
            raise ValueError("unknown discovery code")
        code = code_by_id[code_id]
        cell = json.loads(str(_scalar(data, "cell_json")))
        cell = _validate_cell(cell, config[stage]["cells"])
        if cell["code_id"] != code_id or int(_scalar(data, "m")) != int(code["m"]):
            raise ValueError("discovery raw cell/code identity mismatch")
        if (float(_scalar(data, "p")) != cell["p"]
                or int(_scalar(data, "disorder_index")) != cell["disorder_index"]):
            raise ValueError("discovery raw cell scalar mismatch")
        candidate = validate_pt_candidate(json.loads(str(_scalar(data, "candidate_json"))))
        if str(_scalar(data, "candidate_json")) != canonical_json(candidate):
            raise ValueError("discovery candidate JSON is noncanonical")
        source_commit = str(_scalar(data, "source_commit"))
        if expected_source_commit is not None and source_commit != expected_source_commit:
            raise ValueError("discovery raw source commit mismatch")
        identity = discovery_task_identity(
            registry, config, source_commit, stage, cell, candidate,
        )
        fingerprint = sha256_json(identity)
        expected_scalars = {
            "task_fingerprint": fingerprint,
            "namespace": identity["namespace"],
            "ladder_fingerprint": identity["ladder_fingerprint"],
            "swap_sweeps_per_round": candidate["swap_sweeps_per_round"],
            "engine": "numba",
            "registry_sha256": registry["registry_sha256"],
            "discovery_config_sha256": config["discovery_config_sha256"],
            "section_fingerprint": code["section_fingerprint"],
            "logical_frame_fingerprint": code["logical_frame_fingerprint"],
            "model_fingerprint": sha256_json({"n": code["n"], "k": code["k"]}),
        }
        for field, expected in expected_scalars.items():
            if str(_scalar(data, field)) != str(expected):
                raise ValueError(f"discovery raw identity mismatch: {field}")

        instances, temperatures, k = 4, candidate["num_temperatures"], int(code["k"])
        measurements = candidate["measurement_rounds"]
        labels = _array(data, "labels", (instances, measurements), np.dtype(np.uint64))
        swap_attempts = _array(data, "swap_attempts", (instances, temperatures - 1))
        swap_accepts = _array(data, "swap_accepts", swap_attempts.shape)
        logical_attempts = _array(data, "logical_attempts", (instances, temperatures, k))
        logical_accepts = _array(data, "logical_accepts", logical_attempts.shape)
        _validate_attempt_counters(candidate, swap_attempts, swap_accepts,
                                   logical_attempts, logical_accepts)
        if not np.array_equal(
                _array(data, "swap_rates", swap_attempts.shape),
                swap_accepts / np.maximum(swap_attempts, 1)):
            raise ValueError("discovery swap rates disagree with counters")
        if not np.array_equal(
                _array(data, "logical_rates", logical_attempts.shape),
                logical_accepts / np.maximum(logical_attempts, 1)):
            raise ValueError("discovery logical rates disagree with counters")

        total_names = (
            "hot_touches", "hot_updated_visits", "uncertified_round_trips",
            "round_trips", "sector_changing_round_trips",
        )
        vector_names = tuple(f"{name}_per_replica" for name in total_names)
        totals = {name: _array(data, name, (instances,)) for name in total_names}
        vectors = {name: _array(data, name, (instances, temperatures)) for name in vector_names}
        residual = _array(data, "residual", (instances,))
        if residual.dtype.kind not in "iu" or np.any(residual < 0):
            raise ValueError("discovery residual counters are invalid")
        seeds = _array(data, "instance_seeds", (instances,), np.dtype(np.int64))
        uniform_seed = _uniform_seed(registry, code, cell)
        if int(_scalar(data, "uniform_seed")) != uniform_seed:
            raise ValueError("discovery disorder seed mismatch")

        results = []
        for instance in range(instances):
            expected_seed = _trajectory_seed(
                registry, config, source_commit, stage, code, cell, candidate, instance,
            )
            if int(seeds[instance]) != expected_seed:
                raise ValueError("discovery trajectory seed mismatch")
            result = {
                "labels": labels[instance],
                "swap_attempts": swap_attempts[instance],
                "swap_accepts": swap_accepts[instance],
                "logical_attempts": logical_attempts[instance],
                "logical_accepts": logical_accepts[instance],
                "max_hard_coset_residual": int(residual[instance]),
                "seed": expected_seed,
            }
            for name in total_names:
                result[name] = int(totals[name][instance])
            for name in vector_names:
                result[name] = vectors[name][instance]
            validate_transport_counters(result, temperatures)
            results.append(result)
        gate_result = evaluate_gate(
            results, _discovery_gate(config, stage), k,
            require_trace_gate=stage == "confirmation",
        )
        stored = (
            bool(_scalar(data, "valid")), str(_scalar(data, "failure_reason")),
            _array(data, "rhat", (k,)), _array(data, "ess", (k,)),
            _array(data, "constant_status", (k,)),
        )
        recomputed = (
            gate_result[0], ";".join(gate_result[1]), gate_result[2],
            gate_result[3], gate_result[4],
        )
        for index, (left, right) in enumerate(zip(stored, recomputed)):
            if isinstance(left, np.ndarray):
                equal_nan = left.dtype.kind in "fc"
                if not np.array_equal(left, right, equal_nan=equal_nan):
                    raise ValueError(f"discovery stored gate array {index} was tampered")
            elif left != right:
                raise ValueError("discovery stored validity/failures were tampered")
        for field in ("core_seconds", "wall_seconds"):
            value = float(_scalar(data, field))
            if not np.isfinite(value) or value < 0:
                raise ValueError("discovery timing is invalid")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "task_fingerprint": fingerprint,
        "stage": stage,
        "cell": cell,
        "candidate": candidate,
        "candidate_key": canonical_json(candidate),
        "ladder_id": candidate["ladder_id"],
        "swap_sweeps_per_round": candidate["swap_sweeps_per_round"],
        "valid": gate_result[0],
        "failure_reason": ";".join(gate_result[1]),
        "core_seconds": float(_scalar_value(path, "core_seconds")),
        "wall_seconds": float(_scalar_value(path, "wall_seconds")),
        "min_hot_updated_visits": int(np.min(totals["hot_updated_visits"])),
    }


def analyze_discovery(raw_dir, registry_path, config_path, source_commit, output_path=None):
    registry = load_registry(registry_path)
    config = load_discovery_config(config_path, registry)
    paths = sorted(Path(raw_dir).rglob("*.npz"))
    records = [validate_discovery_raw(path, registry, config, source_commit) for path in paths]
    fingerprints = [record["task_fingerprint"] for record in records]
    if len(fingerprints) != len(set(fingerprints)):
        raise ValueError("discovery evidence contains duplicate task fingerprints")
    groups = _discovery_groups(records, config)
    screen = _screen_analysis(groups, config)
    transport = _transport_analysis(groups, config, screen["passing_ladder_ids"])
    confirmation = _confirmation_analysis(groups, config, transport["ranked_candidates"])
    report = {
        "report_version": DISCOVERY_REPORT_VERSION,
        "generated_by": "discovery.analyze.v2",
        "registry_sha256": registry["registry_sha256"],
        "discovery_config_sha256": config["discovery_config_sha256"],
        "source_commit": source_commit,
        "raw_evidence": [{"path": record["path"], "sha256": record["sha256"]}
                         for record in records],
        "screen": screen,
        "transport": transport,
        "confirmation": confirmation,
        "formal_config_ready": confirmation["primary"] is not None
        and confirmation["backup"] is not None,
    }
    report["analysis_sha256"] = sha256_json({
        "screen": screen, "transport": transport, "confirmation": confirmation,
    })
    if output_path is not None:
        atomic_json(output_path, report)
    return report


def _screen_analysis(groups, config):
    trials, passed, missing = [], [], []
    for candidate in screen_candidates(config):
        group = groups.get(("screen", canonical_json(candidate)))
        public = _public_group(group, len(config["screen"]["cells"]), candidate)
        trials.append(public)
        if not public["complete"]:
            missing.append(candidate["ladder_id"])
        elif public["all_pass"]:
            passed.append(candidate["ladder_id"])
    return {
        "trials": trials,
        "passing_ladder_ids": passed,
        "missing_ladder_ids": missing,
        "complete": not missing,
    }


def _transport_analysis(groups, config, passing_ladder_ids):
    trials = []
    passing = []
    needs = []
    conditional = []
    base_candidates = transport_candidates(config, passing_ladder_ids)
    for candidate in base_candidates:
        group = groups.get(("transport", canonical_json(candidate)))
        public = _public_group(group, len(config["transport"]["cells"]), candidate)
        trials.append(public)
        if not public["complete"]:
            needs.append({"ladder_id": candidate["ladder_id"],
                          "swap_sweeps_per_round": candidate["swap_sweeps_per_round"]})
        elif public["all_pass"]:
            passing.append(public)

    for ladder_id in passing_ladder_ids:
        s64 = next((trial for trial in trials
                    if trial["candidate"]["ladder_id"] == ladder_id
                    and trial["candidate"]["swap_sweeps_per_round"] == 64), None)
        if (s64 is not None and s64["complete"] and not s64["all_pass"]
                and s64["min_hot_updated_visits"] >= 1):
            candidate = transport_candidates(config, [ladder_id], include_conditional=True)[-1]
            group = groups.get(("transport", canonical_json(candidate)))
            public = _public_group(group, len(config["transport"]["cells"]), candidate)
            trials.append(public)
            if not public["complete"]:
                conditional.append({"ladder_id": ladder_id, "swap_sweeps_per_round": 128})
            elif public["all_pass"]:
                passing.append(public)

    best_by_ladder = {}
    for trial in passing:
        ladder_id = trial["candidate"]["ladder_id"]
        key = (trial["core_seconds"], trial["candidate"]["swap_sweeps_per_round"])
        if ladder_id not in best_by_ladder or key < best_by_ladder[ladder_id][0]:
            best_by_ladder[ladder_id] = (key, trial)
    ranked = [value[1]["candidate"] for value in best_by_ladder.values()]
    ranked.sort(key=lambda candidate: (
        next(value[1]["core_seconds"] for value in best_by_ladder.values()
             if value[1]["candidate"]["ladder_id"] == candidate["ladder_id"]),
        candidate["num_temperatures"], candidate["swap_sweeps_per_round"], candidate["p_hot"],
    ))
    return {
        "trials": trials,
        "needs_base_tasks": needs,
        "needs_conditional_tasks": conditional,
        "ranked_candidates": ranked,
        "route_stopped": bool(passing_ladder_ids) and not ranked and not needs and not conditional,
    }


def _confirmation_analysis(groups, config, ranked_transport_candidates):
    trials = []
    passed = []
    next_action = None
    tiers = [tuple(value) for value in config["confirmation"]["round_tiers"]]
    for transport_candidate in ranked_transport_candidates:
        ladder_id = transport_candidate["ladder_id"]
        swap_sweeps = transport_candidate["swap_sweeps_per_round"]
        candidate_pass = None
        for tier in tiers:
            candidate = confirmation_candidate(config, ladder_id, swap_sweeps, tier)
            group = groups.get(("confirmation", canonical_json(candidate)))
            public = _public_group(group, len(config["confirmation"]["cells"]), candidate)
            trials.append(public)
            if not public["complete"]:
                next_action = {"ladder_id": ladder_id, "swap_sweeps_per_round": swap_sweeps,
                               "tier": list(tier)}
                break
            if public["all_pass"]:
                candidate_pass = public
                passed.append(public)
                break
        if next_action is not None:
            break
        if candidate_pass is None:
            continue
        if len({trial["candidate"]["ladder_id"] for trial in passed}) >= 2:
            break

    ranked_passed = sorted(passed, key=lambda trial: (
        trial["core_seconds"], trial["candidate"]["num_temperatures"],
        trial["candidate"]["swap_sweeps_per_round"], trial["candidate"]["p_hot"],
    ))
    primary = ranked_passed[0]["candidate"] if ranked_passed else None
    backup = next((trial["candidate"] for trial in ranked_passed
                   if primary is not None
                   and trial["candidate"]["ladder_id"] != primary["ladder_id"]), None)
    return {
        "trials": trials,
        "passing_candidates": [trial["candidate"] for trial in ranked_passed],
        "next_action": next_action,
        "primary": primary,
        "backup": backup,
        "complete": primary is not None and backup is not None,
    }


def _discovery_groups(records, config):
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["stage"], record["candidate_key"])].append(record)
    result = {}
    for key, rows in grouped.items():
        stage = key[0]
        expected = {canonical_json(cell) for cell in config[stage]["cells"]}
        actual = {canonical_json(row["cell"]) for row in rows}
        if len(actual) != len(rows):
            raise ValueError("discovery group has duplicate logical cells")
        result[key] = {
            "candidate": rows[0]["candidate"],
            "present": len(actual),
            "missing": len(expected - actual),
            "unexpected": len(actual - expected),
            "valid": sum(row["valid"] for row in rows),
            "all_pass": actual == expected and all(row["valid"] for row in rows),
            "core_seconds": float(sum(row["core_seconds"] for row in rows)),
            "wall_seconds_sum": float(sum(row["wall_seconds"] for row in rows)),
            "min_hot_updated_visits": min(
                (row["min_hot_updated_visits"] for row in rows), default=0,
            ),
            "failure_counts": _failure_counts(rows),
        }
    return result


def _public_group(group, expected, candidate):
    if group is None:
        return {
            "candidate": candidate, "expected": expected, "present": 0,
            "missing": expected, "unexpected": 0, "valid": 0,
            "complete": False, "all_pass": False, "core_seconds": 0.0,
            "wall_seconds_sum": 0.0, "min_hot_updated_visits": 0,
            "failure_counts": {},
        }
    return {**group, "expected": expected,
            "complete": group["missing"] == 0 and group["unexpected"] == 0}


def _result_arrays(results):
    names = (
        "labels", "swap_attempts", "swap_accepts", "logical_attempts",
        "logical_accepts", "hot_touches", "hot_updated_visits",
        "uncertified_round_trips", "round_trips", "sector_changing_round_trips",
        "hot_touches_per_replica", "hot_updated_visits_per_replica",
        "uncertified_round_trips_per_replica", "round_trips_per_replica",
        "sector_changing_round_trips_per_replica",
    )
    arrays = {name: np.asarray([result[name] for result in results]) for name in names}
    arrays["swap_rates"] = arrays["swap_accepts"] / np.maximum(arrays["swap_attempts"], 1)
    arrays["logical_rates"] = (
        arrays["logical_accepts"] / np.maximum(arrays["logical_attempts"], 1)
    )
    arrays["residual"] = np.asarray([
        result["max_hard_coset_residual"] for result in results
    ])
    return arrays


def _validate_attempt_counters(candidate, swap_attempts, swap_accepts,
                               logical_attempts, logical_accepts):
    for attempts, accepts in ((swap_attempts, swap_accepts),
                              (logical_attempts, logical_accepts)):
        if (attempts.dtype.kind not in "iu" or accepts.dtype.kind not in "iu"
                or np.any(attempts < 0) or np.any(accepts < 0)
                or np.any(accepts > attempts)):
            raise ValueError("discovery attempt counters are invalid")
    total_rounds = candidate["burn_rounds"] + candidate["measurement_rounds"]
    expected_swap = expected_swap_attempts(
        candidate["num_temperatures"], total_rounds,
        candidate["swap_sweeps_per_round"],
    )
    if not np.array_equal(swap_attempts, np.broadcast_to(expected_swap, swap_attempts.shape)):
        raise ValueError("discovery swap attempt counts are not exact")
    expected_logical = (
        total_rounds * candidate["sweeps_per_round"] * candidate["logical_move_repeat"]
    )
    if np.any(logical_attempts != expected_logical):
        raise ValueError("discovery logical attempt counts are not exact")


def _discovery_gate(config, stage):
    gate = dict(BASE_GATE)
    section = config[stage]
    gate["min_swap_rate"] = float(section["min_swap_rate"])
    if stage in {"transport", "confirmation"}:
        gate["min_round_trips"] = int(section["min_round_trips"])
        gate["min_sector_changing_round_trips"] = int(
            section["min_sector_changing_round_trips"]
        )
    if stage == "confirmation":
        gate["max_rhat"] = float(section["max_rhat"])
        gate["min_ess"] = int(section["min_ess"])
        gate["max_instance_mean_spread"] = 0.10
    return gate


def _uniform_seed(registry, code, cell):
    if cell["disorder_source"] == "attempt022":
        namespace = f"pilot_ladder_m{int(code['m'])}_attempt{OLD_ATTEMPT}"
    elif cell["disorder_source"] == "fresh_v2":
        namespace = "discovery_v2_fresh_disorder"
    else:  # pragma: no cover - cell validation rejects this first
        raise ValueError("unknown discovery disorder source")
    return derive_seed(
        namespace, registry["registry_sha256"], code["code_id"],
        cell["disorder_index"], "uniforms",
    )


def _trajectory_seed(registry, config, source_commit, stage, code, cell,
                     candidate, instance):
    if stage == "screen":
        namespace = f"pilot_ladder_m{int(code['m'])}_attempt{OLD_ATTEMPT}"
        return derive_seed(
            namespace, registry["registry_sha256"], code["code_id"],
            cell["disorder_index"], f"p={float(cell['p']):.8f}", instance,
        )
    return derive_seed(
        "discovery_v2_trajectory", config["discovery_config_sha256"], source_commit,
        stage, ladder_fingerprint(candidate), candidate["swap_sweeps_per_round"],
        candidate["burn_rounds"], candidate["measurement_rounds"],
        registry["registry_sha256"], code["code_id"], cell["disorder_index"],
        f"p={float(cell['p']):.8f}", instance,
    )


def _validate_stage_candidate(config, stage, candidate):
    ladder_ids = {record["ladder_id"] for record in config["ladders"]}
    if candidate["ladder_id"] not in ladder_ids:
        raise ValueError("candidate ladder is outside discovery config")
    section = config[stage]
    if stage == "screen":
        allowed_s = {1}
        allowed_tiers = {(section["burn_rounds"], section["measurement_rounds"])}
    elif stage == "transport":
        allowed_s = set(section["swap_sweeps"]) | {section["conditional_swap_sweeps"]}
        allowed_tiers = {(section["burn_rounds"], section["measurement_rounds"])}
    else:
        allowed_s = {1, 4, 16, 64, 128}
        allowed_tiers = {tuple(tier) for tier in section["round_tiers"]}
    if candidate["swap_sweeps_per_round"] not in allowed_s:
        raise ValueError("candidate swap count is outside the discovery stage")
    if (candidate["burn_rounds"], candidate["measurement_rounds"]) not in allowed_tiers:
        raise ValueError("candidate round tier is outside the discovery stage")


def _fresh_confirmation_cells(registry):
    old = set(OLD_FAILED_CELLS)
    selected = []
    codes_by_m = defaultdict(list)
    for code in registry["codes"]:
        codes_by_m[int(code["m"])].append(code["code_id"])
    for m in range(5, 9):
        candidates = []
        for code_id in sorted(codes_by_m[m]):
            for p in TUNING_P_VALUES:
                for disorder in range(4):
                    if (code_id, p, disorder) in old:
                        continue
                    identity = f"exp102:discovery_fresh_selection_v2:{m}:{code_id}:{p:.8f}:{disorder}"
                    candidates.append((sha256_json(identity), code_id, p, disorder))
        for _, code_id, p, disorder in sorted(candidates)[:2]:
            selected.append(_cell(code_id, p, disorder, "fresh_v2"))
    if len(selected) != 8:
        raise ValueError("could not freeze two fresh discovery cells per m")
    return selected


def _cell(code_id, p, disorder_index, disorder_source):
    return {
        "code_id": str(code_id),
        "p": float(p),
        "disorder_index": int(disorder_index),
        "disorder_source": str(disorder_source),
    }


def _validate_cell(cell, allowed):
    if not isinstance(cell, dict) or set(cell) != {
            "code_id", "p", "disorder_index", "disorder_source"}:
        raise ValueError("discovery cell fields are invalid")
    normalized = _cell(
        cell["code_id"], cell["p"], cell["disorder_index"], cell["disorder_source"],
    )
    if normalized not in allowed:
        raise ValueError("cell is outside the frozen discovery panel")
    return normalized


def _failure_counts(rows):
    counts = defaultdict(int)
    for row in rows:
        for reason in row["failure_reason"].split(";"):
            if reason:
                counts[reason] += 1
    return dict(sorted(counts.items()))


def _scalar(data, field):
    if field not in data or data[field].shape != ():
        raise ValueError(f"discovery raw scalar is missing: {field}")
    return data[field].item()


def _array(data, field, shape, dtype=None):
    if field not in data or data[field].shape != shape:
        raise ValueError(f"discovery raw array has wrong shape: {field}")
    value = data[field].copy()
    if dtype is not None and value.dtype != dtype:
        raise ValueError(f"discovery raw array has wrong dtype: {field}")
    return value


def _scalar_value(path, field):
    with np.load(path, allow_pickle=False) as data:
        return _scalar(data, field)


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("make-config")
    create.add_argument("registry"); create.add_argument("output")
    plan = sub.add_parser("plan")
    plan.add_argument("registry"); plan.add_argument("config"); plan.add_argument("source_commit")
    plan.add_argument("stage", choices=DISCOVERY_STAGES); plan.add_argument("candidates_json")
    plan.add_argument("output")
    run = sub.add_parser("run-cell")
    run.add_argument("registry"); run.add_argument("config"); run.add_argument("source_commit")
    run.add_argument("task_json"); run.add_argument("output")
    analyze = sub.add_parser("analyze")
    analyze.add_argument("raw_dir"); analyze.add_argument("registry"); analyze.add_argument("config")
    analyze.add_argument("source_commit"); analyze.add_argument("output")
    args = parser.parse_args(argv)
    if args.command == "make-config":
        result = write_default_discovery_config(args.registry, args.output)
    elif args.command == "plan":
        candidates = json.loads(Path(args.candidates_json).read_text(encoding="ascii"))
        result = task_manifest(
            args.registry, args.config, args.source_commit, args.stage, candidates, args.output,
        )
    elif args.command == "run-cell":
        task = json.loads(Path(args.task_json).read_text(encoding="ascii"))
        result = run_discovery_cell(
            args.registry, args.config, args.source_commit, task, args.output,
        )
    else:
        result = analyze_discovery(
            args.raw_dir, args.registry, args.config, args.source_commit, args.output,
        )
    print(sha256_json(result) if isinstance(result, dict) else result)


if __name__ == "__main__":
    main()
