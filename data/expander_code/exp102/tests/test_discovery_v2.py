import json
from pathlib import Path

import numpy as np
import pytest

from data.expander_code.exp102.exp102_pipeline import discovery
from data.expander_code.exp102.exp102_pipeline.discovery import (
    DISCOVERY_RAW_FIELDS,
    _confirmation_analysis,
    _screen_analysis,
    _trajectory_seed,
    _transport_analysis,
    _uniform_seed,
    _verified_discovery_paths,
    confirmation_candidate,
    default_discovery_config,
    discovery_task_identity,
    load_discovery_config,
    run_discovery_cell,
    screen_candidates,
    transport_candidates,
    validate_discovery_raw,
)
from data.expander_code.exp102.exp102_pipeline.io import (
    atomic_json,
    atomic_npz,
    canonical_json,
    sha256_file,
    sha256_json,
)
from data.expander_code.exp102.exp102_pipeline.pilot import _validate_raw as validate_formal_pilot_raw
from data.expander_code.exp102.exp102_pipeline.q0_pt import expected_swap_attempts
from data.expander_code.exp102.exp102_pipeline.registry import load_registry


EXP102_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = EXP102_ROOT / "registry/registry.json"
DISCOVERY_CONFIG_PATH = EXP102_ROOT / "config/discovery.v2.json"
SOURCE_COMMIT = "1" * 40


def test_discovery_config_freezes_ladders_and_prospective_fresh_panel(tmp_path):
    registry = load_registry(REGISTRY_PATH)
    config = load_discovery_config(DISCOVERY_CONFIG_PATH, registry)
    assert [item["ladder_id"] for item in config["ladders"]] == [
        "D0", "D1", "D2", "D3", "D4",
    ]
    assert len(config["screen"]["cells"]) == 9
    assert len(config["confirmation"]["cells"]) == 17
    assert len(config["confirmation"]["fresh_cells"]) == 8
    assert config["pt_contract_version"] == "exp102.q0_pt.v2"

    tampered = json.loads(DISCOVERY_CONFIG_PATH.read_text(encoding="ascii"))
    tampered["confirmation"]["fresh_cells"][0]["disorder_index"] += 1
    path = tmp_path / "tampered.json"
    atomic_json(path, tampered)
    with pytest.raises(ValueError, match="frozen protocol"):
        load_discovery_config(path, registry)


def test_disorder_seed_is_candidate_independent_but_transport_trajectory_is_bound():
    registry = load_registry(REGISTRY_PATH)
    config = load_discovery_config(DISCOVERY_CONFIG_PATH, registry)
    code = next(item for item in registry["codes"] if item["code_id"] == "m06_c00")
    cell = config["transport"]["cells"][0]
    candidates = transport_candidates(config, ["D0"])
    assert _uniform_seed(registry, code, cell) == _uniform_seed(registry, code, cell)
    seeds = {
        _trajectory_seed(
            registry, config, SOURCE_COMMIT, "transport", code, cell, candidate, 0,
        )
        for candidate in candidates
    }
    assert len(seeds) == len(candidates)

    screen = screen_candidates(config)
    screen_seeds = {
        _trajectory_seed(
            registry, config, SOURCE_COMMIT, "screen", code,
            config["screen"]["cells"][2], candidate, 0,
        )
        for candidate in screen
    }
    assert len(screen_seeds) == 1


def _fake_result(candidate, k, seed):
    temperatures = candidate["num_temperatures"]
    measurements = candidate["measurement_rounds"]
    total_rounds = candidate["burn_rounds"] + measurements
    swap_attempts = expected_swap_attempts(
        temperatures, total_rounds, candidate["swap_sweeps_per_round"],
    )
    logical_attempts = np.full(
        (temperatures, k),
        total_rounds * candidate["sweeps_per_round"] * candidate["logical_move_repeat"],
        dtype=np.int64,
    )
    zero = np.zeros(temperatures, dtype=np.int64)
    return {
        "labels": np.full(measurements, np.uint64(seed & 1), dtype=np.uint64),
        "swap_attempts": swap_attempts,
        "swap_accepts": swap_attempts // 2,
        "logical_attempts": logical_attempts,
        "logical_accepts": logical_attempts // 2,
        "hot_touches": 0,
        "hot_updated_visits": 0,
        "uncertified_round_trips": 0,
        "round_trips": 0,
        "sector_changing_round_trips": 0,
        "hot_touches_per_replica": zero.copy(),
        "hot_updated_visits_per_replica": zero.copy(),
        "uncertified_round_trips_per_replica": zero.copy(),
        "round_trips_per_replica": zero.copy(),
        "sector_changing_round_trips_per_replica": zero.copy(),
        "max_hard_coset_residual": 0,
    }


def test_discovery_raw_is_self_validating_and_formal_pilot_rejects_it(monkeypatch, tmp_path):
    registry = load_registry(REGISTRY_PATH)
    config = load_discovery_config(DISCOVERY_CONFIG_PATH, registry)
    candidate = screen_candidates(config)[0]
    cell = config["screen"]["cells"][0]
    task = discovery_task_identity(
        registry, config, SOURCE_COMMIT, "screen", cell, candidate,
    )

    def fake_engine(model, frame, syndrome, p, pt_config, seed, initial_label, engine):
        assert engine == "numba"
        return _fake_result(candidate, model.k, seed)

    monkeypatch.setattr(discovery, "run_q0_pt_instance", fake_engine)
    output = tmp_path / "screen.npz"
    assert run_discovery_cell(
        REGISTRY_PATH, DISCOVERY_CONFIG_PATH, SOURCE_COMMIT, task, output,
    ) == "computed"
    record = validate_discovery_raw(output, registry, config, SOURCE_COMMIT)
    assert record["valid"]
    with np.load(output, allow_pickle=False) as data:
        assert set(data.files) == DISCOVERY_RAW_FIELDS

    formal_config = json.loads((EXP102_ROOT / "config/production.v1.json").read_text())
    formal_config["config_sha256"] = "not-used"
    with pytest.raises(ValueError, match="unknown pilot stage"):
        validate_formal_pilot_raw(output, registry, formal_config, SOURCE_COMMIT)

    with np.load(output, allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files}
    arrays["swap_sweeps_per_round"] = np.array(4, dtype=np.int16)
    atomic_npz(output, **arrays)
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_discovery_raw(output, registry, config, SOURCE_COMMIT)


def _group(candidate, expected, core, all_pass=True, min_hot=2):
    return {
        "candidate": candidate,
        "present": expected,
        "missing": 0,
        "unexpected": 0,
        "valid": expected if all_pass else expected - 1,
        "all_pass": all_pass,
        "core_seconds": float(core),
        "wall_seconds_sum": float(core),
        "min_hot_updated_visits": min_hot,
        "failure_counts": {} if all_pass else {"instance_0:round_trips": 1},
    }


def test_discovery_selection_requires_two_distinct_confirmed_ladders():
    registry = load_registry(REGISTRY_PATH)
    config = load_discovery_config(DISCOVERY_CONFIG_PATH, registry)
    groups = {}
    screen = screen_candidates(config)
    for candidate in screen:
        groups[("screen", canonical_json(candidate))] = _group(candidate, 9, 10)
    screen_result = _screen_analysis(groups, config)
    assert screen_result["passing_ladder_ids"] == ["D0", "D1", "D2", "D3", "D4"]

    transport = transport_candidates(config, ["D0", "D1"])
    for candidate in transport:
        passed = candidate["swap_sweeps_per_round"] == 16
        groups[("transport", canonical_json(candidate))] = _group(
            candidate, 2, 2 if candidate["ladder_id"] == "D1" else 3,
            all_pass=passed,
        )
    transport_result = _transport_analysis(groups, config, ["D0", "D1"])
    assert {item["ladder_id"] for item in transport_result["ranked_candidates"]} == {
        "D0", "D1",
    }

    for transport_candidate in transport_result["ranked_candidates"]:
        candidate = confirmation_candidate(
            config, transport_candidate["ladder_id"], 16, (2000, 8000),
        )
        core = 100 if candidate["ladder_id"] == "D0" else 90
        groups[("confirmation", canonical_json(candidate))] = _group(candidate, 17, core)
    confirmation = _confirmation_analysis(
        groups, config, transport_result["ranked_candidates"],
    )
    assert confirmation["primary"]["ladder_id"] == "D1"
    assert confirmation["backup"]["ladder_id"] == "D0"
    assert confirmation["complete"]


def _write_stage_evidence(root):
    registry = load_registry(REGISTRY_PATH)
    config = load_discovery_config(DISCOVERY_CONFIG_PATH, registry)
    candidate = screen_candidates(config)[0]
    tasks = [
        discovery_task_identity(
            registry, config, SOURCE_COMMIT, "screen", cell, candidate,
        )
        for cell in config["screen"]["cells"][:2]
    ]
    control = {
        "manifest_version": "exp102.discovery.tasks.v2",
        "stage": "screen",
        "source_commit": SOURCE_COMMIT,
        "registry_sha256": registry["registry_sha256"],
        "discovery_config_sha256": config["discovery_config_sha256"],
        "tasks": tasks,
    }
    control_dir = root / "control"
    control_path = control_dir / "screen.json"
    atomic_json(control_path, control)
    control_sha256 = sha256_file(control_path)
    task_by_fingerprint = {sha256_json(task): task for task in tasks}
    ordered = sorted(task_by_fingerprint)
    task_owner = {ordered[0]: "nd-1", ordered[1]: "nd-2"}
    task_cost = float(5 * 5 * candidate["num_temperatures"] * (
        candidate["burn_rounds"] + candidate["measurement_rounds"]
    ))
    fingerprint_identity = {
        "source_commit": SOURCE_COMMIT,
        "control_sha256": control_sha256,
        "stage": "screen",
        "nodes": ["nd-1", "nd-2"],
        "task_owner": task_owner,
        "candidate_transport": [[discovery.ladder_fingerprint(candidate), 1]],
        "m_values": [5],
    }
    ownership = {
        "ownership_version": "exp102.discovery.ownership.v2",
        **fingerprint_identity,
        "stage_fingerprint": sha256_json(fingerprint_identity),
        "weighted_load": {"nd-1": task_cost, "nd-2": task_cost},
        "capacity": {"nd-1": 75, "nd-2": 75},
    }
    ownership_path = control_dir / "ownership.json"
    atomic_json(ownership_path, ownership)
    ownership_sha256 = sha256_file(ownership_path)
    source_identity = {
        "source_commit": SOURCE_COMMIT,
        "mode": "archive",
        "archive_sha256": "a" * 64,
        "manifest_sha256": "b" * 64,
        "file_count": 10,
    }
    manifest_paths = []
    for fingerprint, node in sorted(task_owner.items()):
        node_dir = root / "screen" / control_sha256[:12] / node
        raw_path = node_dir / "raw.npz"
        atomic_npz(raw_path, value=np.array(1, dtype=np.int8))
        raw_manifest = {
            "raw_manifest_version": discovery.DISCOVERY_RAW_VERSION,
            "node": node,
            "stage": "screen",
            "stage_fingerprint": ownership["stage_fingerprint"],
            "source_commit": SOURCE_COMMIT,
            "control_sha256": control_sha256,
            "ownership_sha256": ownership_sha256,
            "source_identity": source_identity,
            "files": [{
                "task_fingerprint": fingerprint,
                "path": "raw.npz",
                "sha256": sha256_file(raw_path),
            }],
        }
        manifest_path = node_dir / "raw_manifest.json"
        atomic_json(manifest_path, raw_manifest)
        atomic_json(node_dir / "stage_status.json", {
            "status": "SUCCESS",
            "node": node,
            "stage_fingerprint": ownership["stage_fingerprint"],
            "expected": 1,
            "computed": 1,
            "reused": 0,
            "raw_manifest_sha256": sha256_file(manifest_path),
        })
        atomic_json(node_dir / "SUCCESS", {
            "stage_fingerprint": ownership["stage_fingerprint"],
            "completed_utc": "2026-07-20T00:00:00Z",
        })
        manifest_paths.append(manifest_path)
    return registry, config, manifest_paths


def test_verified_discovery_paths_bind_manifests_controls_and_markers(tmp_path):
    registry, config, manifests = _write_stage_evidence(tmp_path)
    verified = _verified_discovery_paths(
        tmp_path, registry, config, SOURCE_COMMIT,
    )
    assert len(verified["paths"]) == 2
    assert len(verified["manifest_evidence"]) == 2
    assert verified["source_identity"]["archive_sha256"] == "a" * 64

    marker = manifests[0].parent / "SUCCESS"
    value = json.loads(marker.read_text(encoding="ascii"))
    value["stage_fingerprint"] = "c" * 64
    atomic_json(marker, value)
    with pytest.raises(ValueError, match="SUCCESS marker identity mismatch"):
        _verified_discovery_paths(tmp_path, registry, config, SOURCE_COMMIT)


def test_verified_discovery_paths_reject_tampered_hash_and_duplicate_task(tmp_path):
    hash_root = tmp_path / "hash"
    registry, config, manifests = _write_stage_evidence(hash_root)
    manifest = json.loads(manifests[0].read_text(encoding="ascii"))
    manifest["files"][0]["sha256"] = "d" * 64
    atomic_json(manifests[0], manifest)
    status_path = manifests[0].parent / "stage_status.json"
    status = json.loads(status_path.read_text(encoding="ascii"))
    status["raw_manifest_sha256"] = sha256_file(manifests[0])
    atomic_json(status_path, status)
    with pytest.raises(ValueError, match="raw hash mismatch"):
        _verified_discovery_paths(hash_root, registry, config, SOURCE_COMMIT)

    duplicate_root = tmp_path / "duplicate"
    registry, config, manifests = _write_stage_evidence(duplicate_root)
    manifest = json.loads(manifests[0].read_text(encoding="ascii"))
    manifest["files"].append(dict(manifest["files"][0]))
    atomic_json(manifests[0], manifest)
    status_path = manifests[0].parent / "stage_status.json"
    status = json.loads(status_path.read_text(encoding="ascii"))
    status["expected"] = 2
    status["computed"] = 2
    status["raw_manifest_sha256"] = sha256_file(manifests[0])
    atomic_json(status_path, status)
    with pytest.raises(ValueError, match="task coverage is invalid"):
        _verified_discovery_paths(duplicate_root, registry, config, SOURCE_COMMIT)
