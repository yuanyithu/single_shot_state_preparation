import copy
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from data.expander_code.exp102.exp102_pipeline import global_discovery as gd
from data.expander_code.exp102.exp102_pipeline import screen_diagnostic as sd
from data.expander_code.exp102.exp102_pipeline.io import (
    atomic_json,
    atomic_npz,
    sha256_file,
    sha256_json,
)
from data.expander_code.exp102.exp102_pipeline.registry import load_registry


EXP102_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = EXP102_ROOT / "registry/registry.json"
GLOBAL_CONFIG_PATH = EXP102_ROOT / "config/q0_global.discovery.v1.json"
SCREEN_CONFIG_PATH = EXP102_ROOT / "config/q0_global.screen_diagnostic.v1.json"
SOURCE_COMMIT = "1" * 40
ALL_METHOD_TIERS = (
    [(method, "T3") for method in sd.HARD_METHODS]
    + [(method, "T3") for method in sd.DEFECT_METHODS]
)
DEFECT_METHOD_TIERS = [(method, "T3") for method in sd.DEFECT_METHODS]
RUNTIME_PACKAGE = (
    "data.expander_code.exp102.validation."
    "011_q0_global_screen_diagnostic_20260721"
)


@pytest.fixture
def diagnostic_protocol(tmp_path):
    config_path = tmp_path / "screen_diagnostic.json"
    sd.write_default_screen_diagnostic_config(REGISTRY_PATH, config_path)
    registry = sd._registry_with_path(load_registry(REGISTRY_PATH), REGISTRY_PATH)
    config = sd.load_screen_diagnostic_config(config_path, registry)
    return registry, config, config_path


def _fake_bias_index(config):
    index = {}
    for method in sd.DEFECT_METHODS:
        for cell in sd._screen_cells(config):
            fingerprint = f"{len(index) + 1:064x}"
            index[(method, "T3", sha256_json(cell))] = {
                "binding": {
                    "bias_task_fingerprint": fingerprint,
                    "bias_raw_sha256": ("a" if method == "DT16" else "b") * 64,
                    "bias_sha256": ("c" if method == "DT64" else "d") * 64,
                },
                "relpath": f"bias/{fingerprint}.npz",
            }
    assert len(index) == 15
    return index


def _build_controls(tmp_path, monkeypatch, diagnostic_protocol):
    registry, config, config_path = diagnostic_protocol
    bias_path = tmp_path / "bias_manifest.json"
    bias = sd.build_bias_manifest(
        REGISTRY_PATH,
        config_path,
        SOURCE_COMMIT,
        sd.SCREEN_STAGE,
        DEFECT_METHOD_TIERS,
        bias_path,
    )
    bias_index = _fake_bias_index(config)

    def fake_index(raw_root, manifest, actual_registry, actual_config, source):
        assert Path(raw_root) == tmp_path / "bias_raw"
        assert manifest == bias
        assert actual_registry["registry_sha256"] == registry["registry_sha256"]
        assert actual_config["screen_config_sha256"] == config["screen_config_sha256"]
        assert source == SOURCE_COMMIT
        return bias_index

    monkeypatch.setattr(sd, "_bias_index_from_manifest", fake_index)
    measurement_path = tmp_path / "measurement_manifest.json"
    measurement = sd.build_measurement_manifest(
        REGISTRY_PATH,
        config_path,
        SOURCE_COMMIT,
        sd.SCREEN_STAGE,
        ALL_METHOD_TIERS,
        measurement_path,
        bias_manifest_path=bias_path,
        bias_raw_root=tmp_path / "bias_raw",
    )
    return registry, config, bias, measurement, bias_path, measurement_path


def test_default_config_freezes_scope_panel_methods_resources_and_counts(
        diagnostic_protocol):
    registry, config, _ = diagnostic_protocol
    expected = sd.default_screen_diagnostic_config(registry)
    assert {key: config[key] for key in expected} == expected
    assert config["config_version"] == sd.SCREEN_CONFIG_VERSION
    assert config["contract_version"] == sd.SCREEN_DIAGNOSTIC_VERSION
    assert config["task_version"] == sd.SCREEN_TASKS_VERSION
    assert config["panels"]["HARD2"]["cells"] == list(sd.HARD_CELLS)
    assert config["panels"]["EASY3"]["cells"] == list(sd.EASY_CELLS)
    assert len(sd._screen_cells(config)) == 5
    assert [row["method_id"] for row in config["hard_methods"]] == list(
        sd.HARD_METHODS
    )
    assert [row["method_id"] for row in config["defect_methods"]] == list(
        sd.DEFECT_METHODS
    )
    assert config["resource_tiers"]["T3"] == {
        "burn_sweeps": 8192,
        "measurement_sweeps": 32768,
    }
    assert config["resource_selection"]["capacity_nodes"] == ["nd-1", "nd-3"]
    assert config["resource_selection"]["capacity"] == 166
    assert config["trajectory_count_per_init_family"] == 16
    assert config["init_families"] == ["P", "U"]
    scope = config["scope"]
    assert scope["purpose"] == "diagnostic_sampler_screen_only"
    assert scope["maximum_terminal_status"] == "DIAGNOSTIC_SCREEN_PAIR_FOUND"
    assert scope["formal_authorization"] is False
    assert scope["formal_readiness_authorized"] is False
    assert scope["ti_in_scope"] is False
    assert scope["held_out_in_scope"] is False
    assert scope["production_authorization"] is False
    assert scope["production_authorized"] is False
    assert scope["excluded_work"] == [
        "full_sector_ti",
        "method_selection",
        "confirmation",
        "resolution",
        "held_out",
        "production",
    ]
    frozen = sd.load_screen_diagnostic_config(SCREEN_CONFIG_PATH, registry)
    assert {key: frozen[key] for key in expected} == expected


def test_config_loader_rejects_global_or_tampered_protocol(
        tmp_path, diagnostic_protocol):
    registry, config, _ = diagnostic_protocol
    with pytest.raises(ValueError, match="config version|frozen protocol"):
        sd.load_screen_diagnostic_config(GLOBAL_CONFIG_PATH, registry)

    tampered = {
        key: copy.deepcopy(value)
        for key, value in config.items()
        if key not in {"screen_config_sha256", "config_path"}
    }
    tampered["scope"]["formal_readiness_authorized"] = True
    path = tmp_path / "tampered_config.json"
    atomic_json(path, tampered)
    with pytest.raises(ValueError, match="frozen protocol"):
        sd.load_screen_diagnostic_config(path, registry)


def test_diagnostic_task_and_seed_namespaces_are_isolated_from_global(
        diagnostic_protocol):
    registry, config, _ = diagnostic_protocol
    global_config = gd.load_global_discovery_config(GLOBAL_CONFIG_PATH, registry)
    cell = config["panels"]["HARD2"]["cells"][0]
    diagnostic = sd.diagnostic_task_identity(
        registry, config, SOURCE_COMMIT, sd.SCREEN_STAGE, "RC8-QC1", "T3",
        cell, "P", 0,
    )
    historical = gd.global_task_identity(
        registry, global_config, SOURCE_COMMIT, "screen", "RC8-QC1", "T3",
        cell, "P", 0,
    )
    assert diagnostic["task_version"] != historical["task_version"]
    assert diagnostic["raw_version"] != historical["raw_version"]
    assert diagnostic["seed_identity"] != historical["seed_identity"]
    assert diagnostic["seed_identity"]["seed_root"] == sd.SCREEN_SEED_ROOT
    assert sha256_json(diagnostic) != sha256_json(historical)

    diagnostic_bias = sd.diagnostic_bias_task_identity(
        registry, config, SOURCE_COMMIT, sd.SCREEN_STAGE, "DT16", "T3", cell,
    )
    historical_bias = gd.bias_task_identity(
        registry, global_config, SOURCE_COMMIT, "screen", "DT16", "T3", cell,
    )
    assert diagnostic_bias["raw_version"] != historical_bias["raw_version"]
    assert diagnostic_bias["tuning_seed_identities"] != historical_bias[
        "tuning_seed_identities"
    ]
    assert all(
        seed["seed_root"] == sd.SCREEN_SEED_ROOT
        for seed in diagnostic_bias["tuning_seed_identities"]
    )


def test_bias_and_measurement_controls_are_complete_and_cross_version_closed(
        tmp_path, monkeypatch, diagnostic_protocol):
    registry, config, bias, measurement, _, _ = _build_controls(
        tmp_path, monkeypatch, diagnostic_protocol,
    )
    assert len(bias["tasks"]) == 15
    assert len({row["task_fingerprint"] for row in bias["tasks"]}) == 15
    assert len(measurement["tasks"]) == 1280
    assert len({row["task_fingerprint"] for row in measurement["tasks"]}) == 1280
    assert sd.validate_control_manifest(bias, registry, config)
    assert sd.validate_control_manifest(measurement, registry, config)

    global_config = gd.load_global_discovery_config(GLOBAL_CONFIG_PATH, registry)
    with pytest.raises(gd.GlobalConflictError):
        gd.validate_global_control_manifest(bias, registry, global_config)
    old_path = tmp_path / "old_bias.json"
    old = gd.build_bias_manifest(
        REGISTRY_PATH,
        GLOBAL_CONFIG_PATH,
        SOURCE_COMMIT,
        "screen",
        DEFECT_METHOD_TIERS,
        old_path,
    )
    with pytest.raises(gd.GlobalConflictError):
        sd.validate_control_manifest(old, registry, config)


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate"])
def test_bias_manifest_rejects_missing_extra_and_duplicate_tasks(
        mutation, tmp_path, diagnostic_protocol):
    registry, config, config_path = diagnostic_protocol
    manifest = sd.build_bias_manifest(
        REGISTRY_PATH,
        config_path,
        SOURCE_COMMIT,
        sd.SCREEN_STAGE,
        DEFECT_METHOD_TIERS,
        tmp_path / "bias.json",
    )
    broken = copy.deepcopy(manifest)
    if mutation == "missing":
        broken["tasks"].pop()
    elif mutation == "extra":
        broken["tasks"].append(copy.deepcopy(broken["tasks"][-1]))
    else:
        broken["tasks"][1] = copy.deepcopy(broken["tasks"][0])
    with pytest.raises(gd.GlobalConflictError, match="task order"):
        sd.validate_control_manifest(broken, registry, config)


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate"])
def test_measurement_manifest_rejects_missing_extra_and_duplicate_tasks(
        mutation, tmp_path, monkeypatch, diagnostic_protocol):
    registry, config, _, manifest, _, _ = _build_controls(
        tmp_path, monkeypatch, diagnostic_protocol,
    )
    broken = copy.deepcopy(manifest)
    if mutation == "missing":
        broken["tasks"].pop()
    elif mutation == "extra":
        broken["tasks"].append(copy.deepcopy(broken["tasks"][-1]))
    else:
        broken["tasks"][1] = copy.deepcopy(broken["tasks"][0])
    with pytest.raises(gd.GlobalConflictError, match="task count|task/order"):
        sd.validate_control_manifest(broken, registry, config)


def test_measurement_manifest_rejects_inconsistent_bias_binding(
        tmp_path, monkeypatch, diagnostic_protocol):
    registry, config, _, manifest, _, _ = _build_controls(
        tmp_path, monkeypatch, diagnostic_protocol,
    )
    broken = copy.deepcopy(manifest)
    first_defect = len(sd.HARD_METHODS) * 5 * 2 * 16
    entry = broken["tasks"][first_defect + 1]
    entry["task"]["bias_binding"]["bias_sha256"] = "e" * 64
    fingerprint = sha256_json(entry["task"])
    entry["task_fingerprint"] = fingerprint
    entry["output_relpath"] = f"trajectories/{fingerprint}.npz"
    with pytest.raises(
        gd.GlobalConflictError,
        match="task/order changed|different biases",
    ):
        sd.validate_control_manifest(broken, registry, config)


def test_frozen_schedule_is_exactly_24h_and_never_authorizes_production(
        tmp_path, diagnostic_protocol):
    registry, config, config_path = diagnostic_protocol
    schedule_path = tmp_path / "schedule.json"
    started = 1_800_000_000.0
    schedule = sd.freeze_screen_schedule(
        REGISTRY_PATH,
        config_path,
        SOURCE_COMMIT,
        "a" * 64,
        "b" * 64,
        schedule_path,
        started_unix=started,
    )
    assert schedule["status"] == "FROZEN_24H_DIAGNOSTIC"
    assert schedule["wall_limit_hours"] == 24
    assert schedule["production_authorized"] is False
    assert schedule["deadlines_unix"] == {
        "preflight": started + 8 * 3600,
        "bias": started + 12 * 3600,
        "measurement": started + 22 * 3600,
        "analysis": started + 24 * 3600,
    }
    assert sd.validate_screen_schedule(
        schedule_path, registry, config, SOURCE_COMMIT,
    ) == schedule

    tampered = copy.deepcopy(schedule)
    tampered["production_authorized"] = True
    identity = {key: value for key, value in tampered.items()
                if key != "schedule_sha256"}
    tampered["schedule_sha256"] = sha256_json(identity)
    atomic_json(schedule_path, tampered)
    with pytest.raises(gd.GlobalConflictError, match="schedule identity"):
        sd.validate_screen_schedule(schedule_path, registry, config, SOURCE_COMMIT)

    for field, value in (
        ("deadlines_unix", {**schedule["deadlines_unix"],
                            "analysis": started + 25 * 3600}),
        ("wall_limit_hours", 25),
    ):
        tampered = copy.deepcopy(schedule)
        tampered[field] = value
        identity = {key: item for key, item in tampered.items()
                    if key != "schedule_sha256"}
        tampered["schedule_sha256"] = sha256_json(identity)
        atomic_json(schedule_path, tampered)
        with pytest.raises(gd.GlobalConflictError, match="deadlines"):
            sd.validate_screen_schedule(
                schedule_path, registry, config, SOURCE_COMMIT,
            )


def test_ownership_is_deterministic_complete_two_node_and_fail_closed(
        tmp_path, monkeypatch, diagnostic_protocol):
    _, _, _, measurement, _, _ = _build_controls(
        tmp_path, monkeypatch, diagnostic_protocol,
    )
    tasks = [row["task"] for row in measurement["tasks"]]
    nodes = ["nd-1", "nd-3"]
    first = sd.fixed_screen_ownership(
        tasks, nodes, SOURCE_COMMIT, "a" * 64,
    )
    second = sd.fixed_screen_ownership(
        tasks, nodes, SOURCE_COMMIT, "a" * 64,
    )
    assert first == second
    assert set(first["task_owner"]) == {sha256_json(task) for task in tasks}
    assert set(first["task_owner"].values()) == set(nodes)
    assert sd.validate_screen_ownership(
        first, tasks, nodes, SOURCE_COMMIT, "a" * 64,
    )
    normalized = {
        node: first["weighted_load"][node] / sd.NODE_CAPACITY[node]
        for node in nodes
    }
    assert max(normalized.values()) / min(normalized.values()) < 1.01

    with pytest.raises(gd.GlobalConflictError, match="duplicate tasks"):
        sd.fixed_screen_ownership(
            [tasks[0], tasks[0]], nodes, SOURCE_COMMIT, "b" * 64,
        )
    with pytest.raises(ValueError, match="nodes/stage"):
        sd.fixed_screen_ownership(
            tasks[:1], ["nd-1"], SOURCE_COMMIT, "c" * 64,
        )


def test_diagnostic_raw_roundtrip_replay_and_cross_version_rejection(
        tmp_path, monkeypatch, diagnostic_protocol):
    registry, config, config_path = diagnostic_protocol

    def tiny_sampler(actual_config, method, p, tier):
        assert tier == "T1"
        if method in sd.HARD_METHODS:
            return sd.HardCosetConfig(method, p, 8, 8)
        return sd.DefectTraceConfig(
            method, p, 8, 8, tuning_chains=8, tuning_sweeps=4096,
        )

    monkeypatch.setattr(sd, "resolved_sampler_config", tiny_sampler)

    def fake_tune(model, syndrome, sampler, identities, *, engine):
        assert engine == "numba"
        assert len(identities) == 8
        bias = np.zeros(sampler.dmax + 1, dtype=np.float64)
        return {
            "bias": bias,
            "bias_trace": bias[None, :],
            "tuning_histogram": np.zeros(
                (1, sampler.dmax + 1), dtype=np.uint8,
            ),
            "tuning_final_states_packed": np.zeros(
                (8, (model.num_qubits + 7) // 8), dtype=np.uint8,
            ),
            "tuning_final_residuals": np.zeros(
                (8, model.num_checks), dtype=np.uint8,
            ),
            "tuning_final_defects": np.zeros(8, dtype=np.int32),
            "gammas": np.zeros(1, dtype=np.float64),
            "bias_sha256": "e" * 64,
        }

    monkeypatch.setattr(sd, "tune_defect_bias", fake_tune)
    cell = config["panels"]["EASY3"]["cells"][0]

    hard_task = sd.diagnostic_task_identity(
        registry, config, SOURCE_COMMIT, sd.SCREEN_STAGE, "RC8-QC1", "T1",
        cell, "P", 0,
    )
    hard_path = tmp_path / "hard.npz"
    assert sd.run_hard_task(
        REGISTRY_PATH, config_path, SOURCE_COMMIT, hard_task, hard_path,
    ) == "computed"
    hard_record = sd.validate_hard_raw(
        hard_path, registry, config, SOURCE_COMMIT,
    )
    assert hard_record["task"] == hard_task
    assert sd.run_hard_task(
        REGISTRY_PATH, config_path, SOURCE_COMMIT, hard_task, hard_path,
    ) == "reused"

    bias_task = sd.diagnostic_bias_task_identity(
        registry, config, SOURCE_COMMIT, sd.SCREEN_STAGE, "DT16", "T1", cell,
    )
    bias_path = tmp_path / "bias.npz"
    assert sd.run_bias_task(
        REGISTRY_PATH, config_path, SOURCE_COMMIT, bias_task, bias_path,
    ) == "computed"
    bias_record = sd.validate_bias_raw(
        bias_path, registry, config, SOURCE_COMMIT,
    )
    binding = sd.screen_bias_binding_from_raw(
        bias_path, registry, config, SOURCE_COMMIT,
        _validated_bias_record=bias_record,
    )
    assert binding["bias_task_fingerprint"] == bias_record["task_fingerprint"]

    def reject_repeated_tuning(*args, **kwargs):
        raise AssertionError("cached bias must not replay tuning")

    monkeypatch.setattr(sd, "tune_defect_bias", reject_repeated_tuning)
    stale_bias_record = dict(bias_record)
    stale_bias_record["sha256"] = "0" * 64
    with pytest.raises(gd.GlobalConflictError, match="stale or mismatched"):
        sd.screen_bias_binding_from_raw(
            bias_path, registry, config, SOURCE_COMMIT,
            _validated_bias_record=stale_bias_record,
        )

    defect_task = sd.diagnostic_task_identity(
        registry, config, SOURCE_COMMIT, sd.SCREEN_STAGE, "DT16", "T1",
        cell, "P", 0, bias_binding=binding,
    )
    defect_path = tmp_path / "defect.npz"
    assert sd.run_defect_task(
        REGISTRY_PATH,
        config_path,
        SOURCE_COMMIT,
        defect_task,
        bias_path,
        defect_path,
        _validated_bias_record=bias_record,
    ) == "computed"
    defect_record = sd.validate_defect_raw(
        defect_path, registry, config, SOURCE_COMMIT, bias_path,
        _validated_bias_record=bias_record,
    )
    assert defect_record["task"] == defect_task

    with np.load(hard_path, allow_pickle=False) as data:
        assert set(data.files) == sd.HARD_RAW_FIELDS
        assert str(data["contract_version"].item()) == sd.SCREEN_DIAGNOSTIC_VERSION
        assert str(data["raw_version"].item()) == sd.SCREEN_HARD_RAW_VERSION
    with np.load(bias_path, allow_pickle=False) as data:
        assert set(data.files) == sd.BIAS_RAW_FIELDS
    with np.load(defect_path, allow_pickle=False) as data:
        assert set(data.files) == sd.DEFECT_RAW_FIELDS

    global_config = gd.load_global_discovery_config(GLOBAL_CONFIG_PATH, registry)
    with pytest.raises(gd.GlobalConflictError):
        gd.validate_hard_raw(hard_path, registry, global_config, SOURCE_COMMIT)

    with np.load(defect_path, allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files}
    arrays["bias_sha256"] = np.array("f" * 64)
    tampered = tmp_path / "tampered_defect.npz"
    atomic_npz(tampered, **arrays)
    with pytest.raises(gd.GlobalConflictError, match="bias mismatch"):
        sd.validate_defect_raw(
            tampered, registry, config, SOURCE_COMMIT, bias_path,
            _validated_bias_record=bias_record,
        )


def test_analyzer_prevalidates_each_unique_bias_exactly_once(
        tmp_path, monkeypatch, diagnostic_protocol):
    registry, config, _ = diagnostic_protocol
    raw_root = tmp_path / "raw"
    tasks = []
    for method in sd.DEFECT_METHODS:
        for index, cell in enumerate(sd._screen_cells(config)):
            relative = f"bias/{method}_{index}.npz"
            path = raw_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"placeholder")
            tasks.append({
                "task": {"method_id": method, "cell": cell},
                "bias_relpath": relative,
            })
    manifest = {"source_commit": SOURCE_COMMIT, "tasks": tasks}
    calls = []

    def fake_validate(path, actual_registry, actual_config, source_commit):
        calls.append(str(Path(path).resolve()))
        return {"path": str(Path(path).resolve())}

    monkeypatch.setattr(sd, "validate_screen_bias_raw", fake_validate)
    cache = sd._validated_bias_cache(raw_root, manifest, registry, config)
    assert len(cache) == 15
    assert len(calls) == 15
    assert len(set(calls)) == 15


def _install_synthetic_screen_analysis(
        monkeypatch, manifest, valid_methods, agreeing_pairs, *,
        core_seconds=None, defect_ess=None):
    core_seconds = {} if core_seconds is None else core_seconds
    defect_ess = {} if defect_ess is None else defect_ess
    records = [
        {
            "cell": entry["task"]["cell"],
            "method_id": entry["task"]["method_id"],
            "resource_tier": entry["task"]["resource_tier"],
        }
        for entry in manifest["tasks"]
    ]
    monkeypatch.setattr(
        sd,
        "_load_measurement_records",
        lambda *args, **kwargs: records,
    )

    def fake_summary(values, config):
        record = values[0]
        method = record["method_id"]
        method_valid = method in valid_methods
        total_core = float(core_seconds.get(method, 10.0))
        q_se = 0.01 if method_valid else 0.5
        q_trajectory_se = 0.008 if method_valid else 0.4
        q_character_se = 0.006 if method_valid else 0.3
        aggregate_ess = float(defect_ess.get(method, 1000.0))
        worm = None
        if method in sd.DEFECT_METHODS:
            per_chain_ess = aggregate_ess / 16.0
            worm = {
                "d0_counts": [300] * 16,
                "excursions": [60] * 16,
                "per_chain_d0_ess": [per_chain_ess] * 16,
                "median_d0_ess": per_chain_ess,
                "aggregate_d0_ess": aggregate_ess,
                "boundary_occupancy": [0.0] * 16,
            }
        families = {}
        for family_name in sd.INIT_FAMILIES:
            families[family_name] = {
                "init_family": family_name,
                "q_top": 0.5,
                "q_top_total_se": q_se,
                "q_top_trajectory_se": q_trajectory_se,
                "q_top_character_se": q_character_se,
                "label_collision_mass_diagnostic": 0.5,
                "label_collision_q_top_diagnostic": 0.5,
                "normalized_mean_weight": 0.1,
                "normalized_mean_weight_se": 0.001,
                "max_rhat": 1.0,
                "min_nondegenerate_bulk_ess": 500.0,
                "constant_failures": [],
                "worm": copy.deepcopy(worm),
                "core_seconds": total_core / 2.0,
                "valid": method_valid,
                "failures": [] if method_valid else ["q_top_se"],
            }
        initialization_se = float(np.hypot(q_se, q_se))
        return {
            "cell": record["cell"],
            "method_id": method,
            "resource_tier": record["resource_tier"],
            "num_qubits": 100,
            "families": families,
            "q_top": 0.5,
            "q_top_total_se": 0.01,
            "label_collision_mass_diagnostic": 0.5,
            "label_collision_q_top_diagnostic": 0.5,
            "initialization_delta": {
                "delta_q_top": 0.0,
                "se_delta_q_top": initialization_se,
                "absolute_pass": True,
                "sigma_pass": True,
            },
            "d2": {
                "d2_norm": 0.0,
                "d2_trajectory_se": 0.0,
                "d2_character_se": 0.0,
                "d2_total_se": 0.0,
            },
            "normalized_weight_delta": 0.0,
            "normalized_weight_delta_se": float(np.hypot(0.001, 0.001)),
            "ti_anchor_payload": None,
            "core_seconds": total_core,
            "valid": method_valid,
            "failures": [] if method_valid else ["family_gate"],
        }

    def fake_compare(left, right, config):
        agrees = (
            left["method_id"], right["method_id"]
        ) in agreeing_pairs
        return {
            "left": {
                "method_id": left["method_id"],
                "resource_tier": left["resource_tier"],
            },
            "right": {
                "method_id": right["method_id"],
                "resource_tier": right["resource_tier"],
            },
            "q_top": {
                "delta_q_top": 0.0,
                "se_delta_q_top": float(np.hypot(
                    left["q_top_total_se"], right["q_top_total_se"],
                )),
                "absolute_pass": True,
                "sigma_pass": True,
            },
            "d2_norm": 0.0 if agrees else 0.1,
            "d2_total_se": 0.0,
            "d2_pass": agrees,
            "normalized_weight_delta": 0.0,
            "normalized_weight_delta_se": 0.0,
            "weight_pass": True,
            "valid": agrees,
        }

    monkeypatch.setattr(sd._global, "_cell_method_summary", fake_summary)
    monkeypatch.setattr(sd._global, "compare_cell_summaries", fake_compare)


@pytest.mark.parametrize(
    "valid_methods,agreeing_pairs,report_status,decision_status",
    [
        (
            {"RC8-QC1", "DT16"},
            {("RC8-QC1", "DT16")},
            "PAIR_FOUND",
            "DIAGNOSTIC_SCREEN_PAIR_FOUND",
        ),
        (
            {"DT16"},
            set(),
            "NO_HARD_COSET_PASS",
            "UNRESOLVED_NO_HARD_COSET_PASS",
        ),
        (
            {"RC8-QC1"},
            set(),
            "NO_DEFECT_TRACE_PASS",
            "UNRESOLVED_NO_DEFECT_TRACE_PASS",
        ),
        (
            {"RC8-QC1", "DT16"},
            set(),
            "NO_CROSS_MECHANISM_AGREEMENT",
            "UNRESOLVED_NO_CROSS_MECHANISM_AGREEMENT",
        ),
    ],
)
def test_analysis_and_terminal_decision_have_four_diagnostic_only_branches(
        valid_methods, agreeing_pairs, report_status, decision_status,
        tmp_path, monkeypatch, diagnostic_protocol):
    _, _, _, manifest, _, measurement_path = _build_controls(
        tmp_path, monkeypatch, diagnostic_protocol,
    )
    _install_synthetic_screen_analysis(
        monkeypatch, manifest, valid_methods, agreeing_pairs,
    )
    report_path = tmp_path / "screen_report.json"
    report = sd.analyze_screen(
        tmp_path / "raw",
        measurement_path,
        REGISTRY_PATH,
        diagnostic_protocol[2],
        report_path,
    )
    assert report["status"] == report_status
    assert report["raw_count"] == 1280
    assert len(report["method_status"]) == 8
    assert len(report["pair_status"]) == 15
    assert len(report["comparisons"]) == 75
    assert report["formal_authorization"] is False
    assert report["production_authorization"] is False

    decision_path = tmp_path / "decision.json"
    decision = sd.terminal_decision(
        report_path,
        REGISTRY_PATH,
        diagnostic_protocol[2],
        decision_path,
    )
    assert decision["status"] == decision_status
    assert decision["maximum_possible_status"] == (
        "DIAGNOSTIC_SCREEN_PAIR_FOUND"
    )
    assert decision["formal_authorization"] is False
    assert decision["production_authorization"] is False
    assert decision["formal_blockers"] == sd.FORMAL_BLOCKERS
    assert "READY_FOR_FORMAL" not in json.dumps(decision, sort_keys=True)
    assert "FROZEN_HELD_OUT_PASS" not in json.dumps(decision, sort_keys=True)
    if report_status == "PAIR_FOUND":
        assert decision["selected_pair"] is not None
    else:
        assert decision["selected_pair"] is None


def test_analyzer_does_not_rescue_a_failed_primary_pair_with_runner_up(
        tmp_path, monkeypatch, diagnostic_protocol):
    _, _, _, manifest, _, measurement_path = _build_controls(
        tmp_path, monkeypatch, diagnostic_protocol,
    )
    valid_methods = set(sd.HARD_METHODS) | set(sd.DEFECT_METHODS)
    # QC1 is the fastest hard method and DT16 is the most efficient defect
    # method.  Only a slower pair agrees, which must not rescue the primary.
    _install_synthetic_screen_analysis(
        monkeypatch,
        manifest,
        valid_methods,
        {("RC8-QC4", "DT32")},
        core_seconds={
            "RC8-QC1": 1.0,
            "RC8-QC4": 2.0,
            "RC8-J08": 3.0,
            "RC8-J12": 4.0,
            "RC8-J16": 5.0,
            "DT16": 1.0,
            "DT32": 2.0,
            "DT64": 3.0,
        },
        defect_ess={"DT16": 3000.0, "DT32": 2000.0, "DT64": 1000.0},
    )
    report = sd.analyze_screen(
        tmp_path / "raw",
        measurement_path,
        REGISTRY_PATH,
        diagnostic_protocol[2],
    )
    assert report["status"] == "NO_CROSS_MECHANISM_AGREEMENT"
    assert report["selected_pair"] is None


def test_terminal_decision_rejects_formal_status_or_authorization_tampering(
        tmp_path, monkeypatch, diagnostic_protocol):
    registry, config, _, manifest, _, measurement_path = _build_controls(
        tmp_path, monkeypatch, diagnostic_protocol,
    )
    _install_synthetic_screen_analysis(
        monkeypatch,
        manifest,
        {"RC8-QC1", "DT16"},
        {("RC8-QC1", "DT16")},
    )
    report_path = tmp_path / "screen_report.json"
    report = sd.analyze_screen(
        tmp_path / "raw",
        measurement_path,
        REGISTRY_PATH,
        diagnostic_protocol[2],
        report_path,
    )
    assert sd.validate_screen_report(report, registry, config)

    for collection, index in (("method_status", 1), ("pair_status", 1)):
        tampered = copy.deepcopy(report)
        tampered[collection][index]["valid"] = not tampered[collection][index][
            "valid"
        ]
        identity = {
            key: item for key, item in tampered.items()
            if key != "report_sha256"
        }
        tampered["report_sha256"] = sha256_json(identity)
        atomic_json(report_path, tampered)
        with pytest.raises(gd.GlobalConflictError, match="status is inconsistent"):
            sd.terminal_decision(
                report_path, REGISTRY_PATH, diagnostic_protocol[2],
            )

    for field, value in (
            ("status", "READY_FOR_FORMAL"),
            ("formal_authorization", True),
            ("production_authorization", True)):
        tampered = copy.deepcopy(report)
        tampered[field] = value
        identity = {
            key: item for key, item in tampered.items()
            if key != "report_sha256"
        }
        tampered["report_sha256"] = sha256_json(identity)
        atomic_json(report_path, tampered)
        with pytest.raises(gd.GlobalConflictError):
            sd.terminal_decision(
                report_path,
                REGISTRY_PATH,
                diagnostic_protocol[2],
            )


def _runtime_modules():
    benchmark = importlib.import_module(RUNTIME_PACKAGE + ".benchmark_screen")
    common = importlib.import_module(RUNTIME_PACKAGE + ".common")
    return benchmark, common


def _runtime_node_report(node, benchmark, common, *, multiplier=1.0):
    methods = list(common.all_methods())
    rows = []
    for code_id, m in (("m06_c00", 6), ("m08_c06", 8)):
        for index, method in enumerate(methods):
            per_sweep = multiplier * (2e-5 + index * 1e-6)
            if node == "nd-3" and method == "RC8-J16":
                per_sweep = 0.18
            row = {
                "code_id": code_id,
                "m": m,
                "method_id": method,
                "catalog_seconds": 0.0,
                "joint_build_seconds": 0.0,
                "warmup_seconds": 0.1,
                "timed_sweeps": 160,
                "wall_seconds": per_sweep * 160.0,
                "core_seconds": per_sweep * 160.0,
                "core_seconds_per_sweep": per_sweep,
            }
            if method in common.defect_methods():
                row["bias_tuning_wall_seconds"] = multiplier * 0.5
            rows.append(row)
    projections, selected, eligible, checks = (
        benchmark._reconstruct_node_projection(rows)
    )
    source_identity = {
        "source_commit": SOURCE_COMMIT,
        "mode": "archive",
        "archive_sha256": "a" * 64,
        "manifest_sha256": "b" * 64,
        "file_count": 10,
    }
    return {
        "benchmark_version": common.RUNTIME_NODE_VERSION,
        "contract_version": common.CONTRACT_VERSION,
        "source_commit": SOURCE_COMMIT,
        "source_identity": source_identity,
        "registry_sha256": "c" * 64,
        "diagnostic_config_sha256": "d" * 64,
        "node": node,
        "environment": {
            "system": "Linux", "machine": "x86_64", "hostname": node,
            "python": "3.11.0", "numpy": "2.3.0",
        },
        "completed_unix": 1_800_000_000.0,
        "rows": rows,
        "projections": projections,
        "selected_resource_tier": selected,
        "selected_eligible_methods": eligible,
        "checks": checks,
        "status": "PASS",
    }


def _write_runtime_nodes(tmp_path):
    benchmark, common = _runtime_modules()
    paths = {}
    for multiplier, node in enumerate(common.EXPECTED_PREFLIGHT_NODES, start=1):
        path = tmp_path / f"runtime_{node}.json"
        atomic_json(
            path,
            _runtime_node_report(
                node, benchmark, common, multiplier=float(multiplier),
            ),
        )
        paths[node] = path
    return common, paths


def test_runtime_consensus_uses_three_node_worst_case_and_largest_common_tier(
        tmp_path):
    benchmark, common = _runtime_modules()
    _, paths = _write_runtime_nodes(tmp_path)
    output = tmp_path / "runtime_consensus.json"
    consensus = benchmark.combine_runtime_reports(paths, output)
    assert consensus["status"] == "PASS"
    assert consensus["environment"] == {
        "system": "Linux",
        "nodes": list(common.EXPECTED_PREFLIGHT_NODES),
    }
    assert consensus["selected_resource_tier"] == "T2"
    assert consensus["selected_eligible_methods"] == list(common.all_methods())
    projections = {
        value["resource_tier"]: value for value in consensus["projections"]
    }
    assert projections["T1"]["pass"] is True
    assert projections["T2"]["pass"] is True
    assert projections["T3"]["pass"] is False
    expected_worst = max(
        next(value for value in json.loads(path.read_text(encoding="ascii"))[
            "projections"
        ] if value["resource_tier"] == "T2")["trajectory_seconds_m8"][
            "RC8-QC1"
        ]
        for path in paths.values()
    )
    assert projections["T2"]["trajectory_seconds_m8"][
        "RC8-QC1"
    ] == expected_worst
    assert projections["T3"]["trajectory_seconds_m8"]["RC8-J16"] > 7200.0
    assert json.loads(output.read_text(encoding="ascii")) == consensus


@pytest.mark.parametrize("excluded_field", [
    "ti_anchor_projection",
    "wmc_report",
    "q_top",
])
def test_runtime_rejects_physics_or_excluded_work_fields(
        excluded_field, tmp_path):
    benchmark, _ = _runtime_modules()
    _, paths = _write_runtime_nodes(tmp_path)
    node = "nd-2"
    report = json.loads(paths[node].read_text(encoding="ascii"))
    report[excluded_field] = 0.0
    atomic_json(paths[node], report)
    with pytest.raises(ValueError, match="excluded work|verified evidence"):
        benchmark.combine_runtime_reports(paths)


def test_runtime_rejects_missing_timing_rows(tmp_path):
    benchmark, _ = _runtime_modules()
    _, paths = _write_runtime_nodes(tmp_path)
    report = json.loads(paths["nd-2"].read_text(encoding="ascii"))
    report["rows"] = []
    atomic_json(paths["nd-2"], report)
    with pytest.raises(ValueError, match="rows are missing"):
        benchmark.combine_runtime_reports(paths)


def test_runtime_source_config_and_consensus_report_tampering_is_rejected(
        tmp_path):
    benchmark, common = _runtime_modules()
    _, paths = _write_runtime_nodes(tmp_path)

    for field, value in (
            ("source_commit", "2" * 40),
            ("diagnostic_config_sha256", "e" * 64)):
        original = json.loads(paths["nd-3"].read_text(encoding="ascii"))
        tampered = copy.deepcopy(original)
        tampered[field] = value
        atomic_json(paths["nd-3"], tampered)
        with pytest.raises(ValueError, match="verified evidence"):
            benchmark.combine_runtime_reports(paths)
        atomic_json(paths["nd-3"], original)

    output = tmp_path / "runtime_consensus.json"
    consensus = benchmark.combine_runtime_reports(paths, output)
    assert common.validate_runtime_consensus(
        output,
        SOURCE_COMMIT,
        "c" * 64,
        "d" * 64,
        "a" * 64,
        "b" * 64,
    ) == consensus
    for field, value in (
            ("source_commit", "2" * 40),
            ("diagnostic_config_sha256", "e" * 64),
            ("selected_resource_tier", "T3")):
        tampered = copy.deepcopy(consensus)
        tampered[field] = value
        atomic_json(output, tampered)
        with pytest.raises(
            ValueError, match="identity/status|not feasible|runtime consensus",
        ):
            common.validate_runtime_consensus(
                output,
                SOURCE_COMMIT,
                "c" * 64,
                "d" * 64,
                "a" * 64,
                "b" * 64,
            )

    tampered = copy.deepcopy(consensus)
    tampered["projections"][1]["projected_core_seconds"] += 1.0
    atomic_json(output, tampered)
    with pytest.raises(ValueError, match="projection is inconsistent"):
        common.validate_runtime_consensus(
            output, SOURCE_COMMIT, "c" * 64, "d" * 64, "a" * 64, "b" * 64,
        )


def test_final_analyzer_reconstructs_all_three_preflight_nodes(
        tmp_path, diagnostic_protocol):
    benchmark, common = _runtime_modules()
    cross = importlib.import_module(RUNTIME_PACKAGE + ".cross_node_screen")
    analyzer = importlib.import_module(RUNTIME_PACKAGE + ".analyze_screen")
    registry, config, _ = diagnostic_protocol
    archive_sha = "a" * 64
    manifest_sha = "b" * 64
    source_identity = {
        "source_commit": SOURCE_COMMIT,
        "mode": "archive",
        "archive_sha256": archive_sha,
        "manifest_sha256": manifest_sha,
        "file_count": 10,
    }
    run_root = tmp_path / "run"
    output_root = run_root / "screen_diagnostic/preflight"
    runtime_paths = {}
    digest_paths = {}
    node_report_paths = {}
    deadline = 1_800_028_800.0
    for multiplier, node in enumerate(common.EXPECTED_PREFLIGHT_NODES, start=1):
        node_root = output_root / "nodes" / node
        node_root.mkdir(parents=True)
        runtime = _runtime_node_report(
            node, benchmark, common, multiplier=float(multiplier),
        )
        runtime["registry_sha256"] = registry["registry_sha256"]
        runtime["diagnostic_config_sha256"] = config[
            "screen_config_sha256"
        ]
        runtime["source_identity"] = source_identity
        runtime_path = node_root / "runtime.json"
        atomic_json(runtime_path, runtime)
        runtime_paths[node] = runtime_path

        digest = {
            "digest_version": common.DIGEST_NODE_VERSION,
            "contract_version": common.CONTRACT_VERSION,
            "records": [{"kind": "synthetic", "digest": "f" * 64}],
            "canonical_digest": "e" * 64,
            "registry_sha256": registry["registry_sha256"],
            "diagnostic_config_sha256": config["screen_config_sha256"],
            "source_commit": SOURCE_COMMIT,
            "source_identity": source_identity,
            "node": node,
            "completed_unix": 1_800_000_100.0 + multiplier,
            "environment": {"system": "Linux", "hostname": node},
        }
        digest_path = node_root / "digest.json"
        atomic_json(digest_path, digest)
        digest_paths[node] = digest_path

        pytest_log = node_root / "pytest.log"
        pytest_log.write_text("pass\n", encoding="utf-8")
        node_report = {
            "report_version": common.PREFLIGHT_NODE_VERSION,
            "contract_version": common.CONTRACT_VERSION,
            "status": "PASS",
            "node": node,
            "source_commit": SOURCE_COMMIT,
            "source_identity": source_identity,
            "environment": {
                "system": "Linux", "machine": "x86_64", "python": "3.11",
            },
            "pytest_returncode": 0,
            "pytest_log_sha256": sha256_file(pytest_log),
            "digest_path": f"nodes/{node}/digest.json",
            "digest_sha256": sha256_file(digest_path),
            "runtime_path": f"nodes/{node}/runtime.json",
            "runtime_sha256": sha256_file(runtime_path),
            "excluded_work": ["full_sector_ti", "wmc"],
            "started_unix": 1_800_000_000.0 + multiplier,
            "completed_unix": 1_800_000_200.0 + multiplier,
        }
        node_report_path = node_root / "preflight.json"
        atomic_json(node_report_path, node_report)
        node_report_paths[node] = node_report_path

    control = run_root / "control"
    control.mkdir(parents=True)
    runtime_path = control / "runtime.json"
    digest_path = control / "digest.json"
    runtime = benchmark.combine_runtime_reports(runtime_paths, runtime_path)
    digest = cross.combine_digest_reports(digest_paths, digest_path)
    schedule = {
        "archive_sha256": archive_sha,
        "source_manifest_sha256": manifest_sha,
        "schedule_sha256": "9" * 64,
        "deadlines_unix": {"preflight": deadline},
    }
    schedule_file_sha = "8" * 64
    identity = {
        "contract_version": common.CONTRACT_VERSION,
        "stage": "preflight",
        "source_commit": SOURCE_COMMIT,
        "archive_sha256": archive_sha,
        "source_manifest_sha256": manifest_sha,
        "schedule_file_sha256": schedule_file_sha,
        "schedule_sha256": schedule["schedule_sha256"],
        "registry_sha256": registry["registry_sha256"],
        "diagnostic_config_sha256": config["screen_config_sha256"],
        "registry_relative": str(common.DEFAULT_REGISTRY_RELATIVE),
        "config_relative": str(common.DEFAULT_CONFIG_RELATIVE),
        "nodes": list(common.EXPECTED_PREFLIGHT_NODES),
    }
    preflight = {
        "report_version": common.PREFLIGHT_VERSION,
        "status": "PASS",
        **identity,
        "stage_fingerprint": sha256_json(identity),
        "node_report_sha256": {
            node: sha256_file(path) for node, path in node_report_paths.items()
        },
        "runtime_consensus_sha256": sha256_file(runtime_path),
        "digest_consensus_sha256": sha256_file(digest_path),
        "selected_resource_tier": runtime["selected_resource_tier"],
        "selected_eligible_methods": runtime["selected_eligible_methods"],
        "canonical_digest": digest["canonical_digest"],
        "excluded_work": ["full_sector_ti", "wmc"],
        "maximum_terminal_status": "DIAGNOSTIC_SCREEN_PAIR_FOUND",
        "completed_unix": 1_800_000_300.0,
    }
    preflight_path = control / "preflight.json"
    atomic_json(preflight_path, preflight)
    assert analyzer._validate_preflight(
        preflight_path, digest_path, runtime_path, run_root, schedule,
        registry, config, SOURCE_COMMIT, schedule_file_sha,
    ) == (preflight, digest)

    runtime_node = json.loads(runtime_paths["nd-2"].read_text(encoding="ascii"))
    runtime_node["rows"] = []
    atomic_json(runtime_paths["nd-2"], runtime_node)
    with pytest.raises(ValueError, match="node evidence|rows are missing"):
        analyzer._validate_preflight(
            preflight_path, digest_path, runtime_path, run_root, schedule,
            registry, config, SOURCE_COMMIT, schedule_file_sha,
        )
