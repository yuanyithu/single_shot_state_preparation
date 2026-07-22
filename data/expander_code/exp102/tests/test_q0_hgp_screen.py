import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from data.expander_code.exp102.exp102_pipeline import q0_hgp_screen as hs
from data.expander_code.exp102.exp102_pipeline.io import (
    atomic_json,
    atomic_npz,
    sha256_file,
    sha256_json,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_collapsed import (
    CollapsedPowerPtConfig,
)
from data.expander_code.exp102.exp102_pipeline.q0_map_mixture import (
    MAP_METHOD_ID,
    MapMixtureConfig,
)
from data.expander_code.exp102.exp102_pipeline.registry import (
    load_frozen_code,
    load_registry,
)
from data.expander_code.exp102.exp102_pipeline.worker import build_model
from data.expander_code.exp102.exp102_pipeline.seeds import derive_seed


EXP102_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = EXP102_ROOT / "registry/registry.json"
CONFIG_PATH = EXP102_ROOT / "config/q0_hgp_global.screen.v2.json"
SOURCE_COMMIT = "1" * 40
ARCHIVE_SHA256 = "a" * 64
SOURCE_MANIFEST_SHA256 = "b" * 64


@pytest.fixture
def protocol():
    registry = hs._registry_with_path(load_registry(REGISTRY_PATH), REGISTRY_PATH)
    config = hs.load_hgp_screen_config(CONFIG_PATH, registry)
    return registry, config


def _tiny_sampler_config(method, p, _tier, *, max_anchors=8):
    if method in hs.HP_METHODS:
        return CollapsedPowerPtConfig(method, p, 8, 8)
    if method == MAP_METHOD_ID:
        return MapMixtureConfig(p, 8, 8, max_anchors=max_anchors)
    raise AssertionError(f"unexpected test method: {method}")


def _frozen_b_records(character_set, *, n=12, steps=64,
                      opposite_without_crossing=False):
    r = character_set.r
    total = n * n + r * r
    rng = np.random.default_rng(9301)
    records = []
    for trajectory in range(hs.TRAJECTORIES_PER_FAMILY):
        full_bits = np.zeros((steps, total), dtype=np.uint8)
        # Conditional A noise is deliberately abundant while every B clock is
        # frozen, reproducing the blind spot this gate is meant to catch.
        full_bits[:, :n * n] = rng.integers(
            0, 2, size=(steps, n * n), dtype=np.uint8,
        )
        packed = np.packbits(full_bits, axis=1, bitorder="little")
        b_states = hs._extract_b_states_packed(packed, n, r)
        initial_b = np.zeros((1, (r * r + 7) // 8), dtype=np.uint8)
        burn_b = initial_b.copy()
        if opposite_without_crossing and trajectory == 0:
            initial_b[0, 0] = burn_b[0, 0] = np.uint8(1)
        records.append({
            "trajectory_index": trajectory,
            "init_family": "P",
            "b_character_set": character_set,
            "b_measurement_states_packed": b_states,
            "b_measurement_weights": hs._b_weights(b_states),
            "b_measurement_log_likelihood": np.zeros(steps, dtype=np.float64),
            "b_initial_character_bits": hs._b_character_bits(
                initial_b, character_set.masks_packed,
            )[0],
            "b_burn_character_bits": hs._b_character_bits(
                burn_b, character_set.masks_packed,
            )[0],
            "b_a_factor_count": n,
        })
    return records


@pytest.fixture(scope="module")
def artifact_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("hgp_screen_artifacts")
    descriptors = hs.build_hgp_map_artifacts(
        REGISTRY_PATH, CONFIG_PATH, SOURCE_COMMIT, ARCHIVE_SHA256,
        SOURCE_MANIFEST_SHA256, root,
    )
    assert len(descriptors) == 2
    return root


def test_config_is_canonical_and_all_mutations_fail_closed(tmp_path, protocol):
    registry, config = protocol
    assert sha256_file(CONFIG_PATH) == hs.HGP_SCREEN_CONFIG_SHA256
    assert config["hgp_screen_config_sha256"] == hs.HGP_SCREEN_CONFIG_SHA256
    assert config["scope"]["maximum_terminal_status"] == (
        "DIAGNOSTIC_HARD_PAIR_FOUND"
    )
    assert config["scope"]["formal_authorization"] is False
    assert config["scope"]["production_authorization"] is False
    assert config["scope"]["held_out_in_scope"] is False
    assert config["execution"]["analysis"] == {
        "node": "nd-3", "capacity": 91, "num_workers": 91,
    }
    assert config["screen_panel_sha256"] == sha256_json(
        hs._method_schedule_identity(config),
    )
    assert config["raw_versions"]["map"] == "exp102.q0_map_mixture.raw.v4"
    assert config["portable_evidence"] == {
        "version": "exp102.q0_hgp_global.screen.portable_evidence.v1",
        "remote_linux_policy": "full_transcript_bit_exact_three_node_consensus",
        "local_policy": "portable_transcript_and_acceptance_decision_bit_exact",
        "float_tolerance": None,
        "map_preflight_burn": 256,
        "map_preflight_measurement": 2048,
        "map_preflight_init_families": ["P", "U"],
        "map_preflight_panels": ["HARD2"],
    }
    assert config["gates"]["map_min_burn_state_changes"] == 1
    assert config["gates"]["map_min_measurement_state_change_rate"] == 0.05
    assert config["gates"]["map_min_measurement_state_changes"] == 400

    original = json.loads(CONFIG_PATH.read_text(encoding="ascii"))
    noncanonical = tmp_path / "noncanonical_config.json"
    noncanonical.write_text(json.dumps(original, indent=2) + "\n", encoding="ascii")
    with pytest.raises(ValueError, match="not canonical JSON"):
        hs.load_hgp_screen_config(noncanonical, registry)
    mutations = []
    changed = copy.deepcopy(original)
    changed["scope"]["formal_authorization"] = True
    mutations.append(changed)
    changed = copy.deepcopy(original)
    changed["resource_tiers"]["T1"]["measurement"] += 8
    mutations.append(changed)
    changed = copy.deepcopy(original)
    changed["hp_methods"][0]["block_size"] = 12
    mutations.append(changed)
    changed = copy.deepcopy(original)
    changed["seed_namespaces"]["root"] = "q0_global_discovery_v1"
    mutations.append(changed)
    changed = copy.deepcopy(original)
    changed["panels"]["HARD2"]["cells"][0]["p"] = 0.05
    mutations.append(changed)
    changed = copy.deepcopy(original)
    changed["method_panels"][MAP_METHOD_ID].append("EASY3")
    mutations.append(changed)

    for index, changed in enumerate(mutations):
        path = tmp_path / f"tampered_config_{index}.json"
        atomic_json(path, changed)
        with pytest.raises(ValueError, match="canonical config SHA"):
            hs.load_hgp_screen_config(path, registry)

    wrong_registry = dict(registry)
    wrong_registry["registry_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="registry SHA"):
        hs.load_hgp_screen_config(CONFIG_PATH, wrong_registry)


def _synthetic_sampler_result(method):
    fields = hs._HP_RESULT_FIELDS if method in hs.HP_METHODS else hs._MAP_RESULT_FIELDS
    result = {
        name: np.asarray([0], dtype=np.uint8) for name in fields
    }
    result["method_id"] = method
    result["engine"] = "reference"
    for name in hs._HP_NONPORTABLE_RESULT_FIELDS | hs._MAP_NONPORTABLE_RESULT_FIELDS:
        if name in result and name not in {"engine", "lambda_sha256", "mass_sha256"}:
            result[name] = np.asarray([0.25], dtype=np.float64)
    if method == MAP_METHOD_ID:
        result["burn_accept_uniform"] = np.asarray([0.125], dtype=np.float64)
        result["measurement_accept_uniform"] = np.asarray(
            [0.75], dtype=np.float64,
        )
    return result


def test_sampler_evidence_has_explicit_exact_portable_partitions():
    mam = _synthetic_sampler_result(MAP_METHOD_ID)
    baseline = hs._sampler_evidence(mam, MAP_METHOD_ID)
    manifest = baseline["field_manifest"]
    assert set(manifest["nonportable_fields"]) == hs._MAP_NONPORTABLE_RESULT_FIELDS
    assert "burn_accept_uniform" in manifest["portable_fields"]
    assert "measurement_accept_uniform" in manifest["portable_fields"]
    assert manifest["float_tolerance"] is None

    derived_float_changed = dict(mam)
    derived_float_changed["burn_log_acceptance"] = np.asarray(
        [1.0e100], dtype=np.float64,
    )
    changed = hs._sampler_evidence(derived_float_changed, MAP_METHOD_ID)
    assert changed["full_transcript_sha256"] != baseline["full_transcript_sha256"]
    assert changed["nonportable_float_sha256"] != baseline[
        "nonportable_float_sha256"
    ]
    assert changed["portable_transcript_sha256"] == baseline[
        "portable_transcript_sha256"
    ]
    assert changed["acceptance_decision_sha256"] == baseline[
        "acceptance_decision_sha256"
    ]

    uniform_changed = dict(mam)
    uniform_changed["burn_accept_uniform"] = np.asarray(
        [0.5], dtype=np.float64,
    )
    changed = hs._sampler_evidence(uniform_changed, MAP_METHOD_ID)
    assert changed["portable_transcript_sha256"] != baseline[
        "portable_transcript_sha256"
    ]
    assert changed["acceptance_decision_sha256"] != baseline[
        "acceptance_decision_sha256"
    ]

    hp = _synthetic_sampler_result("HP32")
    baseline = hs._sampler_evidence(hp, "HP32")
    assert set(baseline["field_manifest"]["nonportable_fields"]) == (
        hs._HP_NONPORTABLE_RESULT_FIELDS
    )
    likelihood_changed = dict(hp)
    likelihood_changed["cold_log_likelihood"] = np.asarray(
        [-1234.5], dtype=np.float64,
    )
    changed = hs._sampler_evidence(likelihood_changed, "HP32")
    assert changed["full_transcript_sha256"] != baseline["full_transcript_sha256"]
    assert changed["portable_transcript_sha256"] == baseline[
        "portable_transcript_sha256"
    ]
    counter_changed = dict(hp)
    counter_changed["swap_accepts"] = np.asarray([1], dtype=np.int64)
    changed = hs._sampler_evidence(counter_changed, "HP32")
    assert changed["portable_transcript_sha256"] != baseline[
        "portable_transcript_sha256"
    ]


def test_is_evidence_splits_identity_diagnostics_and_float_arrays():
    identity = {
        "raw_version": hs.HGP_MAP_IS_RAW_VERSION, "seed": 7,
        "num_samples": hs.HGP_MAP_IS_SAMPLES,
        "num_qubits": 10, "coordinate_dimension": 7,
        "artifact_descriptor": {"anchor_count": 2},
    }
    diagnostics = {name: 1.0 for name in hs._MAP_IS_DIAGNOSTIC_FIELDS}
    diagnostics["num_samples"] = hs.HGP_MAP_IS_SAMPLES
    diagnostics["minimum_sampled_physical_weight"] = 1
    count = hs.HGP_MAP_IS_SAMPLES
    arrays = {
        "sample_states_packed": np.zeros((count, 2), dtype=np.uint8),
        "sample_coordinates_packed": np.zeros((count, 1), dtype=np.uint8),
        "sample_physical_weights": np.zeros(count, dtype=np.int32),
        "sample_anchor_index": np.zeros(count, dtype=np.int16),
        "sample_component_index": np.zeros(count, dtype=np.int8),
    }
    arrays.update({
        name: np.full(count, 0.25, dtype=np.float64)
        for name in hs._MAP_IS_NONPORTABLE_ARRAY_FIELDS
    })
    baseline = hs._map_is_evidence(identity, diagnostics, arrays)
    assert set(baseline["field_manifest"]["portable_array_fields"]) == (
        hs._MAP_IS_PORTABLE_ARRAY_FIELDS
    )
    changed_arrays = dict(arrays)
    changed_arrays["sample_log_q"] = np.full(count, 9.0, dtype=np.float64)
    changed = hs._map_is_evidence(identity, diagnostics, changed_arrays)
    assert changed["full_transcript_sha256"] != baseline["full_transcript_sha256"]
    assert changed["nonportable_float_sha256"] != baseline[
        "nonportable_float_sha256"
    ]
    assert changed["portable_transcript_sha256"] == baseline[
        "portable_transcript_sha256"
    ]


def test_preflight_bundle_exposes_only_two_independent_canonical_projections():
    full = {
        "evidence_schema_version": hs.HGP_PREFLIGHT_EVIDENCE_VERSION,
        "evidence_projection": "full",
        "acceptance_decision_catalog": [],
        "acceptance_decision_catalog_sha256": sha256_json([]),
    }
    portable = {
        "evidence_schema_version": hs.HGP_PREFLIGHT_EVIDENCE_VERSION,
        "evidence_projection": "portable",
        "acceptance_decision_catalog": [],
        "acceptance_decision_catalog_sha256": sha256_json([]),
    }
    bundle = hs.build_hgp_preflight_evidence_bundle(full, portable)
    assert set(bundle) == {
        "canonical_full_payload", "canonical_full_payload_sha256",
        "canonical_portable_payload", "canonical_portable_payload_sha256",
    }
    assert bundle["canonical_full_payload_sha256"] == sha256_json(full)
    assert bundle["canonical_portable_payload_sha256"] == sha256_json(portable)


def test_preflight_digest_freezes_hard2_p_u_decision_catalog(artifact_root):
    bundle = hs.hgp_screen_preflight_digest(
        REGISTRY_PATH, CONFIG_PATH, SOURCE_COMMIT, ARCHIVE_SHA256,
        SOURCE_MANIFEST_SHA256, artifact_root,
    )
    assert set(bundle) == {
        "canonical_full_payload", "canonical_full_payload_sha256",
        "canonical_portable_payload", "canonical_portable_payload_sha256",
    }
    full = bundle["canonical_full_payload"]
    portable = bundle["canonical_portable_payload"]
    assert bundle["canonical_full_payload_sha256"] == sha256_json(full)
    assert bundle["canonical_portable_payload_sha256"] == sha256_json(portable)
    assert len(full["cells"]) == len(portable["cells"]) == 5
    expected_catalog = [
        (cell["code_id"], family)
        for cell in hs.HARD_CELLS for family in hs.INIT_FAMILIES
    ]
    assert [
        (
            next(
                cell["code_id"] for cell in hs.HARD_CELLS
                if hs._cell_fingerprint(cell) == row["cell_fingerprint"]
            ),
            row["init_family"],
        )
        for row in portable["acceptance_decision_catalog"]
    ] == expected_catalog
    assert portable["acceptance_decision_catalog_sha256"] == sha256_json(
        portable["acceptance_decision_catalog"],
    )
    assert full["acceptance_decision_catalog"] == portable[
        "acceptance_decision_catalog"
    ]
    for full_row, portable_row in zip(full["cells"], portable["cells"]):
        assert "mass_sha256" in full_row
        assert "mass_sha256" not in portable_row
        if full_row["cell"] in hs.HARD_CELLS:
            assert [
                row["init_family"] for row in full_row["map_portability_probes"]
            ] == list(hs.INIT_FAMILIES)
            for row in portable_row["map_portability_probes"]:
                evidence = row["transcript_evidence"]
                assert "full_transcript_sha256" not in evidence
                assert "nonportable_float_sha256" not in evidence
                assert "burn_accept_uniform" in evidence[
                    "field_manifest"
                ]["portable_fields"]


def test_map_transport_gate_uses_state_changes_not_accepted_self_loops(protocol):
    _, config = protocol
    record = {
        "algorithm_metrics": {
            "burn_accepts": 8192,
            "burn_state_changes": 0,
            "measurement_accepts": 32768,
            "measurement_acceptance": 1.0,
            "measurement_state_changes": 0,
            "measurement_state_change_rate": 0.0,
        },
    }
    assert hs._algorithm_failures(
        [record], MAP_METHOD_ID, config["gates"],
    ) == [
        "map_burn_state_changes",
        "map_measurement_state_change_rate",
        "map_measurement_state_changes",
    ]

    record["algorithm_metrics"].update({
        "burn_state_changes": 1,
        "measurement_state_changes": 400,
        "measurement_state_change_rate": 0.05,
    })
    assert not hs._algorithm_failures(
        [record], MAP_METHOD_ID, config["gates"],
    )


def test_map_transition_replay_distinguishes_accepted_self_loop():
    data = {
        "sampler_initial_state_packed": np.asarray([0], dtype=np.uint8),
        "sampler_burn_state_packed": np.asarray([0], dtype=np.uint8),
        "sampler_burn_proposal_states_packed": np.asarray(
            [[0], [1]], dtype=np.uint8,
        ),
        "sampler_burn_states_packed": np.asarray(
            [[0], [0]], dtype=np.uint8,
        ),
        "sampler_burn_accepted": np.asarray([1, 0], dtype=np.uint8),
        "sampler_burn_state_changed": np.asarray([0, 0], dtype=np.uint8),
        "sampler_burn_attempts": np.asarray(2, dtype=np.int64),
        "sampler_burn_accepts": np.asarray(1, dtype=np.int64),
        "sampler_burn_state_changes": np.asarray(0, dtype=np.int64),
        "sampler_measurement_proposal_states_packed": np.asarray(
            [[1]], dtype=np.uint8,
        ),
        "sampler_measurement_states_packed": np.asarray(
            [[1]], dtype=np.uint8,
        ),
        "sampler_measurement_accepted": np.asarray([1], dtype=np.uint8),
        "sampler_measurement_state_changed": np.asarray([1], dtype=np.uint8),
        "sampler_measurement_attempts": np.asarray(1, dtype=np.int64),
        "sampler_measurement_accepts": np.asarray(1, dtype=np.int64),
        "sampler_measurement_state_changes": np.asarray(1, dtype=np.int64),
    }
    hs._validate_map_transition_counters(data)
    data["sampler_burn_state_changed"][0] = np.uint8(1)
    with pytest.raises(hs.HgpScreenConflictError, match="transition counters"):
        hs._validate_map_transition_counters(data)


def test_frozen_b_character_catalog_is_complete_unique_and_dense():
    characters = hs.frozen_b_character_set(9, 123456789)
    assert characters.single_count == 81
    assert characters.row_column_count == 18
    assert characters.dense_count == 64
    assert characters.size == 163
    assert len({row.tobytes() for row in characters.masks_packed}) == 163
    dense = characters.masks_packed[characters.dense_start:]
    weights = np.bitwise_count(dense).sum(axis=1)
    assert np.all(weights >= 27)
    assert np.all(weights <= 54)
    assert characters.character_sha256 == hs._b_character_digest(
        characters.masks_packed, 9, 64, 123456789,
    )


def test_b_gate_rejects_frozen_b_despite_fresh_conditional_a_noise(protocol):
    _, config = protocol
    characters = hs.frozen_b_character_set(9, 987654321)
    records = _frozen_b_records(characters)
    assert all(
        not np.any(record["b_measurement_states_packed"])
        for record in records
    )
    summary = hs._b_family_summary(records, config)
    assert summary["valid"] is False
    assert "b_weight_constant" in summary["failures"]
    assert "b_likelihood_constant" in summary["failures"]
    assert "b_dense_characters_uninformative" in summary["failures"]


def test_b_constant_character_requires_every_opposite_chain_to_cross(protocol):
    _, config = protocol
    characters = hs.frozen_b_character_set(9, 11223344)
    records = _frozen_b_records(
        characters, opposite_without_crossing=True,
    )
    summary = hs._b_family_summary(records, config)
    assert "b_constant_character_no_burn_crossing" in summary["failures"]
    assert "single_r00_c00" in summary["constant_character_failures"]


def test_single_b_bit_disagreement_cannot_be_diluted_by_catalog_average(protocol):
    _, config = protocol
    characters = hs.frozen_b_character_set(9, 55667788)
    left = np.ones((16, characters.size), dtype=np.float64)
    right = left.copy()
    right[:, 0] = -1.0
    d2 = hs._b_d2_estimate(left, right)
    assert d2["mean_square_character_delta"] < config["gates"][
        "max_b_character_d2_upper"
    ]
    gate = hs._b_character_delta_gate(left, right, characters, config)
    assert gate["absolute_pass"] is False
    assert gate["sigma_pass"] is False
    assert gate["failed_character_count"] == 1
    assert gate["failed_characters"] == ["single_r00_c00"]


def test_hp_cold_likelihood_tail_must_match_reconstructed_b(protocol):
    registry, config = protocol
    _, code, H = load_frozen_code(REGISTRY_PATH, "m03_c00")
    model, frame = build_model(H)
    syndrome = np.zeros(model.num_checks, dtype=np.uint8)
    initial = np.zeros(model.num_qubits, dtype=np.uint8)
    cell = dict(hs.EASY_CELLS[0])
    seed = hs._seed_identity(
        config, registry, SOURCE_COMMIT, ARCHIVE_SHA256,
        SOURCE_MANIFEST_SHA256, "HP32", "T1", cell, "P", 0,
    )
    sampler = CollapsedPowerPtConfig("HP32", cell["p"], 1, 8)
    replay = hs.run_collapsed_power_pt_trajectory(
        model, frame, H, syndrome, sampler, seed, initial, engine="numba",
    )
    characters = hs.frozen_b_character_set(
        H.shape[0],
        hs._b_character_seed(registry["registry_sha256"], code["code_id"]),
    )
    hs._b_record_from_replay(
        replay, H, syndrome, cell["p"], characters, "HP32", 1,
    )
    tampered = dict(replay)
    tampered["cold_log_likelihood"] = replay["cold_log_likelihood"].copy()
    tampered["cold_log_likelihood"][1] += 1.0
    with pytest.raises(hs.HgpScreenConflictError, match="cold likelihood"):
        hs._b_record_from_replay(
            tampered, H, syndrome, cell["p"], characters, "HP32", 1,
        )


def test_runtime_projection_models_owner_generation_and_single_node_replay(
    protocol,
):
    _, config = protocol
    timings = {
        "HP32": {"seconds_per_step": 0.0, "setup_seconds_per_task": 10.0},
        "HP64": {"seconds_per_step": 0.0, "setup_seconds_per_task": 20.0},
        MAP_METHOD_ID: {
            "seconds_per_step": 0.0, "setup_seconds_per_task": 30.0,
        },
    }
    hard = hs._map_cells(config)
    is_seconds = {
        hs._cell_fingerprint(hard[0]): 40.0,
        hs._cell_fingerprint(hard[1]): 50.0,
    }
    b_timings = {
        "benchmark_measurement_rounds": 32768,
        "trace_benchmark_seconds": 32768.0,
        "trace_seconds_per_round": 1.0,
        "family_benchmark_seconds": 10.0,
        "comparison_benchmark_seconds": 20.0,
    }
    tiers, counts = hs._runtime_tier_projections(
        config, timings, is_seconds, 7.0, b_timings,
    )
    assert counts == {
        "nd-2": {"HP32": 80, "HP64": 80, MAP_METHOD_ID: 32},
        "nd-3": {"HP32": 80, "HP64": 80, MAP_METHOD_ID: 32},
    }
    row = tiers["T1"]
    assert row["full_sampler_passes_per_task"] == 2
    assert row["full_is_passes_per_cell"] == 2
    assert row["safety_factor"] == 2.0
    generation_walls = []
    for node, is_value in (("nd-2", 40.0), ("nd-3", 50.0)):
        durations = [10.0] * 80 + [20.0] * 80 + [30.0] * 32
        lpt = hs._lpt_makespan(
            durations, config["execution"]["capacities"][node],
        )
        expected = lpt + is_value
        workload = row["per_node_generation_workload"][node]
        assert workload["sampler_generation_lpt_seconds"] == lpt
        assert workload["is_generation_lpt_seconds"] == is_value
        assert workload["projected_generation_wall_seconds"] == expected
        generation_walls.append(expected)

    all_durations = (
        [10.0 + 8192.0] * 160
        + [20.0 + 8192.0] * 160
        + [30.0 + 8192.0] * 64
    )
    analysis_sampler_lpt = hs._lpt_makespan(all_durations, 91)
    analysis = row["analysis_workload"]
    assert analysis["node"] == "nd-3"
    assert analysis["capacity"] == analysis["num_workers"] == 91
    assert analysis["sampler_task_count"] == 384
    assert analysis["sampler_replay_lpt_seconds"] == analysis_sampler_lpt
    assert analysis["b_trace_seconds_per_task"] == 8192.0
    assert analysis["b_family_count"] == 24
    assert analysis["b_comparison_count"] == 20
    expected_scale = (8192 * 13) / (32768 * 15)
    expected_b_diagnostics = expected_scale * (24 * 10.0 + 20 * 20.0)
    assert analysis["b_statistical_diagnostics_seconds"] == (
        expected_b_diagnostics
    )
    assert analysis["is_replay_mode"] == "serial"
    assert analysis["is_replay_seconds"] == 90.0
    expected_unsafetied = (
        7.0 + max(generation_walls) + analysis_sampler_lpt + 90.0
        + expected_b_diagnostics
    )
    assert row["projected_unsafetied_schedule_seconds"] == expected_unsafetied
    assert row["projected_complete_schedule_seconds"] == (
        2.0 * expected_unsafetied
    )


def test_manifest_freezes_384_method_scoped_tasks_and_new_seeds(
    protocol, artifact_root,
):
    registry, config = protocol
    manifest = hs.build_hgp_screen_manifest(
        REGISTRY_PATH, CONFIG_PATH, SOURCE_COMMIT, ARCHIVE_SHA256,
        SOURCE_MANIFEST_SHA256, "T1", artifact_root,
    )
    assert hs.validate_hgp_screen_manifest(
        manifest, registry, config, artifact_root,
    )
    assert manifest["task_count"] == 384
    assert manifest["analysis"] == {
        "node": "nd-3", "capacity": 91, "num_workers": 91,
    }
    assert len(manifest["tasks"]) == 384
    assert Counter(row["owner"] for row in manifest["tasks"]) == {
        "nd-2": 192,
        "nd-3": 192,
    }
    assert Counter(row["task"]["method_id"] for row in manifest["tasks"]) == {
        "HP32": 160,
        "HP64": 160,
        MAP_METHOD_ID: 64,
    }
    assert Counter(row["task"]["init_family"] for row in manifest["tasks"]) == {
        "P": 192,
        "U": 192,
    }
    assert len({row["task_fingerprint"] for row in manifest["tasks"]}) == 384
    assert len({row["output_relpath"] for row in manifest["tasks"]}) == 384
    assert len(manifest["map_artifacts"]) == 2
    assert len(manifest["importance_sampling"]["outputs"]) == 2
    assert {
        row["task"]["cell"]["code_id"] for row in manifest["tasks"]
        if row["task"]["method_id"] == MAP_METHOD_ID
    } == {"m06_c00", "m08_c06"}

    identities = [row["task"]["seed_identity"] for row in manifest["tasks"]]
    assert Counter(
        identity["trajectory_namespace"] for identity in identities
    ) == {
        hs.HGP_SCREEN_HP_TRAJECTORY_ROOT: 320,
        hs.HGP_SCREEN_MAP_TRAJECTORY_ROOT: 64,
    }
    seeds = [
        hs.HgpScreenSeedIdentity(**identity).seed("measurement")
        for identity in identities
    ]
    assert len(set(seeds)) == 384
    first = identities[0]
    legacy_seed = derive_seed(
        "q0_global_discovery_v1",
        first["source_commit"],
        first["config_sha256"],
        first["registry_sha256"],
        first["cell_fingerprint"],
        first["method_id"],
        first["resource_tier"],
        first["init_family"],
        first["trajectory_index"],
        "measurement",
        "stream",
        0,
    )
    assert seeds[0] != legacy_seed
    assert hs.HGP_SCREEN_TASK_VERSION != "exp102.q0_global.tasks.v1"
    assert hs.HGP_POWER_RAW_VERSION != "exp102.q0_hardcoset.raw.v1"


def test_hp_and_map_short_raw_are_pickle_free_replayable_and_tamper_closed(
    tmp_path, monkeypatch, protocol, artifact_root,
):
    registry, config = protocol
    monkeypatch.setattr(hs, "_sampler_config", _tiny_sampler_config)
    manifest = hs.build_hgp_screen_manifest(
        REGISTRY_PATH, CONFIG_PATH, SOURCE_COMMIT, ARCHIVE_SHA256,
        SOURCE_MANIFEST_SHA256, "T1", artifact_root,
    )
    selected = []
    for method in ("HP32", MAP_METHOD_ID):
        selected.append(next(
            row for row in manifest["tasks"]
            if row["task"]["method_id"] == method
            and row["task"]["cell"]["code_id"] == "m06_c00"
            and row["task"]["init_family"] == "P"
            and row["task"]["trajectory_index"] == 0
        ))

    for entry in selected:
        method = entry["task"]["method_id"]
        path = tmp_path / f"{method}.npz"
        hs.run_hgp_screen_task(
            REGISTRY_PATH, CONFIG_PATH, SOURCE_COMMIT, ARCHIVE_SHA256,
            SOURCE_MANIFEST_SHA256, entry["task"], artifact_root, path,
        )
        record = hs.validate_hgp_screen_raw(
            path, registry, config, SOURCE_COMMIT, ARCHIVE_SHA256,
            SOURCE_MANIFEST_SHA256, artifact_root,
        )
        assert record["method_id"] == method
        assert record["labels"].shape == (8,)
        assert record["weights"].shape == (8,)
        assert record["valid_mask"].all()
        stored_evidence = hs.validate_hgp_screen_stored_evidence(path)
        assert stored_evidence["task"] == entry["task"]
        for name in (
                "full_transcript_sha256", "portable_transcript_sha256",
                "nonportable_float_sha256", "field_manifest_sha256",
                "acceptance_decision_sha256"):
            assert stored_evidence[name] == record[name]
        if method == MAP_METHOD_ID:
            metrics = record["algorithm_metrics"]
            assert metrics["burn_accepts"] >= metrics["burn_state_changes"]
            assert (
                metrics["measurement_accepts"]
                >= metrics["measurement_state_changes"]
            )
            assert metrics["measurement_state_change_rate"] == pytest.approx(
                metrics["measurement_state_changes"] / 8.0,
            )
            original_run_sampler = hs._run_sampler

            def replay_with_changed_derived_logs(*args, **kwargs):
                replay = original_run_sampler(*args, **kwargs)
                replay = dict(replay)
                for name in hs._MAP_NONPORTABLE_RESULT_FIELDS:
                    replay[name] = np.asarray(replay[name]).copy() + 1000.0
                return replay

            monkeypatch.setattr(
                hs, "_run_sampler", replay_with_changed_derived_logs,
            )
            with pytest.raises(
                    hs.HgpScreenConflictError, match="sampler replay mismatch"):
                hs.validate_hgp_screen_raw(
                    path, registry, config, SOURCE_COMMIT, ARCHIVE_SHA256,
                    SOURCE_MANIFEST_SHA256, artifact_root,
                    replay_evidence="full",
                )
            portable = hs.validate_hgp_screen_raw(
                path, registry, config, SOURCE_COMMIT, ARCHIVE_SHA256,
                SOURCE_MANIFEST_SHA256, artifact_root,
                replay_evidence="portable",
            )
            assert portable["replay_evidence"] == "portable"

            def replay_with_changed_uniform(*args, **kwargs):
                replay = original_run_sampler(*args, **kwargs)
                replay = dict(replay)
                replay["burn_accept_uniform"] = np.asarray(
                    replay["burn_accept_uniform"],
                ).copy()
                replay["burn_accept_uniform"].flat[0] += 0.25
                return replay

            monkeypatch.setattr(hs, "_run_sampler", replay_with_changed_uniform)
            with pytest.raises(
                    hs.HgpScreenConflictError, match="sampler replay mismatch"):
                hs.validate_hgp_screen_raw(
                    path, registry, config, SOURCE_COMMIT, ARCHIVE_SHA256,
                    SOURCE_MANIFEST_SHA256, artifact_root,
                    replay_evidence="portable",
                )
            monkeypatch.setattr(hs, "_run_sampler", original_run_sampler)

        with np.load(path, allow_pickle=False) as data:
            arrays = {name: data[name].copy() for name in data.files}
            assert all(value.dtype != np.dtype("O") for value in arrays.values())
            assert np.asarray(arrays["sampler_measurement_states_packed"]).shape[0] == 8
            assert str(arrays["source_commit"].item()) == SOURCE_COMMIT

        tamper_cases = {}
        changed = {name: value.copy() for name, value in arrays.items()}
        changed["source_commit"] = np.array("2" * 40)
        tamper_cases["identity"] = changed
        changed = {name: value.copy() for name, value in arrays.items()}
        changed["syndrome_packed"].flat[0] ^= np.uint8(1)
        tamper_cases["syndrome"] = changed
        changed = {name: value.copy() for name, value in arrays.items()}
        changed["character_masks"].flat[0] ^= np.uint64(1)
        tamper_cases["character"] = changed
        changed = {name: value.copy() for name, value in arrays.items()}
        changed["b_character_masks_packed"].flat[0] ^= np.uint8(1)
        tamper_cases["b_character_mask"] = changed
        changed = {name: value.copy() for name, value in arrays.items()}
        changed["b_character_sha256"] = np.array("0" * 64)
        tamper_cases["b_character_sha"] = changed
        changed = {name: value.copy() for name, value in arrays.items()}
        changed["sampler_measurement_weights"].flat[0] += 1
        tamper_cases["trajectory"] = changed
        changed = {name: value.copy() for name, value in arrays.items()}
        changed["full_transcript_sha256"] = np.array("0" * 64)
        tamper_cases["digest"] = changed

        for kind, changed in tamper_cases.items():
            tampered = tmp_path / f"{method}_{kind}_tampered.npz"
            atomic_npz(tampered, **changed)
            with pytest.raises(hs.HgpScreenConflictError):
                hs.validate_hgp_screen_raw(
                    tampered, registry, config, SOURCE_COMMIT,
                    ARCHIVE_SHA256, SOURCE_MANIFEST_SHA256, artifact_root,
                )


def test_is_v2_stored_full_and_portable_replay_are_separate(
        tmp_path, monkeypatch, artifact_root):
    cell = dict(hs.HARD_CELLS[0])
    path = tmp_path / "is_v2.npz"
    generated = hs.run_hgp_map_is_diagnostic(
        REGISTRY_PATH, CONFIG_PATH, SOURCE_COMMIT, ARCHIVE_SHA256,
        SOURCE_MANIFEST_SHA256, cell, artifact_root, path,
        seed_namespace=hs.HGP_SCREEN_PREFLIGHT_IS_ROOT,
    )
    stored = hs.validate_hgp_map_is_stored_evidence(path)
    full = hs.validate_hgp_map_is_diagnostic(
        path, REGISTRY_PATH, CONFIG_PATH, SOURCE_COMMIT, ARCHIVE_SHA256,
        SOURCE_MANIFEST_SHA256, cell, artifact_root,
        seed_namespace=hs.HGP_SCREEN_PREFLIGHT_IS_ROOT,
    )
    for name in (
            "full_transcript_sha256", "portable_transcript_sha256",
            "nonportable_float_sha256", "field_manifest_sha256"):
        assert generated[name] == stored[name] == full[name]

    original_transcript = hs._map_is_transcript

    def changed_float_transcript(*args, **kwargs):
        arrays, diagnostics = original_transcript(*args, **kwargs)
        arrays = dict(arrays)
        arrays["sample_log_q"] = arrays["sample_log_q"].copy() + 1000.0
        arrays["sample_log_importance_weight"] = (
            arrays["sample_log_importance_weight"].copy() - 1000.0
        )
        return arrays, diagnostics

    monkeypatch.setattr(hs, "_map_is_transcript", changed_float_transcript)
    with pytest.raises(hs.HgpScreenConflictError, match="replay mismatch"):
        hs.validate_hgp_map_is_diagnostic(
            path, REGISTRY_PATH, CONFIG_PATH, SOURCE_COMMIT, ARCHIVE_SHA256,
            SOURCE_MANIFEST_SHA256, cell, artifact_root,
            seed_namespace=hs.HGP_SCREEN_PREFLIGHT_IS_ROOT,
            replay_evidence="full",
        )
    portable = hs.validate_hgp_map_is_diagnostic(
        path, REGISTRY_PATH, CONFIG_PATH, SOURCE_COMMIT, ARCHIVE_SHA256,
        SOURCE_MANIFEST_SHA256, cell, artifact_root,
        seed_namespace=hs.HGP_SCREEN_PREFLIGHT_IS_ROOT,
        replay_evidence="portable",
    )
    assert portable["replay_evidence"] == "portable"

    monkeypatch.setattr(hs, "_map_is_transcript", original_transcript)
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files}
    arrays["sample_log_q"].flat[0] += 1.0
    tampered = tmp_path / "is_v2_tampered.npz"
    atomic_npz(tampered, **arrays)
    with pytest.raises(
            hs.HgpScreenConflictError, match="stored evidence mismatch"):
        hs.validate_hgp_map_is_stored_evidence(tampered)


def test_manifest_tamper_is_rejected_and_report_cannot_authorize_formal_work(
    tmp_path, monkeypatch, protocol, artifact_root,
):
    registry, config = protocol
    manifest_path = tmp_path / "manifest.json"
    manifest = hs.build_hgp_screen_manifest(
        REGISTRY_PATH, CONFIG_PATH, SOURCE_COMMIT, ARCHIVE_SHA256,
        SOURCE_MANIFEST_SHA256, "T1", artifact_root, manifest_path,
    )
    tampered = copy.deepcopy(manifest)
    tampered["tasks"][0]["owner"] = "nd-3"
    identity = {
        key: value for key, value in tampered.items()
        if key != "manifest_sha256"
    }
    tampered["manifest_sha256"] = sha256_json(identity)
    with pytest.raises(hs.HgpScreenConflictError, match="noncanonical"):
        hs.validate_hgp_screen_manifest(
            tampered, registry, config, artifact_root,
        )

    raw_root = tmp_path / "raw"
    by_path = {}
    for entry in manifest["tasks"]:
        path = raw_root / entry["output_relpath"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        by_path[str(path)] = entry["task"]
    for relative in manifest["importance_sampling"]["outputs"]:
        path = raw_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    def fake_validate(path):
        task = by_path[str(Path(path))]
        if task["method_id"] in hs.HP_METHODS:
            metrics = {
                "min_edge_swap_rate": 1.0,
                "min_edge_swap_accepts": 1000,
                "round_trips": 100,
                "cold_origin_fraction": 1.0,
            }
        else:
            metrics = {
                "burn_accepts": 8,
                "burn_state_changes": 8,
                "measurement_accepts": 1000,
                "measurement_acceptance": 1.0,
                "measurement_state_changes": 1000,
                "measurement_state_change_rate": 1.0,
            }
        return {
            "cell": task["cell"],
            "method_id": task["method_id"],
            "resource_tier": task["resource_tier"],
            "init_family": task["init_family"],
            "trajectory_index": task["trajectory_index"],
            "algorithm_metrics": metrics,
            "core_seconds": 1.0,
        }

    def fake_family(records, _config):
        return {"init_family": records[0]["init_family"], "valid": True}

    branch = {"invalid_methods": set(), "agreement_valid": True}

    def fake_cell(records, _config):
        method = records[0]["method_id"]
        valid = method not in branch["invalid_methods"]
        return {
            "cell": records[0]["cell"],
            "method_id": method,
            "resource_tier": records[0]["resource_tier"],
            "num_qubits": 10,
            "core_seconds": 1.0 if method == "HP32" else 2.0,
            "valid": valid,
            "failures": [] if valid else ["forced_test_failure"],
        }

    def fake_compare(_left, _right, _config, family, _num_qubits):
        return {
            "init_family": family,
            "q_top": {"absolute_pass": True, "sigma_pass": True},
            "d2": {"d2_norm": 0.0, "d2_total_se": 0.0},
            "normalized_weight_delta": 0.0,
            "normalized_weight_delta_se": 0.0,
            "valid": branch["agreement_valid"],
            "failures": (
                [] if branch["agreement_valid"] else ["forced_disagreement"]
            ),
        }

    monkeypatch.setattr(hs, "_validate_raw_worker", fake_validate)
    monkeypatch.setattr(
        hs, "validate_hgp_map_is_diagnostic",
        lambda path, registry_path, config_path, source_commit, archive_sha,
        source_manifest_sha, cell, artifacts: {
            "sha256": "d" * 64, "cell": cell,
            "diagnostics": {"importance_ess": 1.0},
            "full_transcript_sha256": "e" * 64,
            "portable_transcript_sha256": "f" * 64,
            "nonportable_float_sha256": "a" * 64,
            "field_manifest_sha256": "b" * 64,
            "used_for_gate_or_selection": False,
        },
    )
    monkeypatch.setattr(hs._statistics, "_family_summary", fake_family)
    monkeypatch.setattr(hs._statistics, "_cell_method_summary", fake_cell)
    monkeypatch.setattr(
        hs, "_b_family_summary",
        lambda records, _config: {
            "init_family": records[0]["init_family"], "valid": True,
        },
    )
    monkeypatch.setattr(
        hs, "_b_cell_summary",
        lambda families, _config: {
            "families": families, "initialization_comparison": {},
            "valid": True, "failures": [],
        },
    )
    monkeypatch.setattr(hs, "_compare_family_summaries", fake_compare)
    report_path = tmp_path / "report.json"
    report = hs.analyze_hgp_screen(
        raw_root, manifest_path, REGISTRY_PATH, CONFIG_PATH,
        artifact_root, output_path=report_path, num_workers=1,
    )
    assert report["status"] == "DIAGNOSTIC_HARD_PAIR_FOUND"
    assert report["selected_pair"] == {
        "hp_method_id": "HP32",
        "map_method_id": MAP_METHOD_ID,
        "resource_tier": "T1",
        "agreement_valid": True,
        "agreement_panels": ["HARD2"],
        "easy3_scope": "hp_runtime_and_false_negative_control_only",
    }
    assert report["formal_authorization"] is False
    assert report["production_authorization"] is False
    assert report["raw_count"] == 384
    assert len(report["importance_sampling_diagnostics"]) == 2
    assert len(report["comparisons"]) == 8
    assert Counter(
        (
            row["hp_method_id"], row["cell"]["code_id"],
            row["init_family"],
        )
        for row in report["comparisons"]
    ) == {
        (method, cell["code_id"], family): 1
        for method in hs.HP_METHODS
        for cell in hs._cross_mechanism_cells(config)
        for family in hs.INIT_FAMILIES
    }
    assert all(row["family_cells_total"] == 4 for row in report["pair_status"])
    method_status = {row["method_id"]: row for row in report["method_status"]}
    assert method_status["HP32"]["cells_total"] == 5
    assert method_status["HP64"]["cells_total"] == 5
    assert method_status[MAP_METHOD_ID]["cells_total"] == 2
    assert set(report["remaining_required_stages"]) >= {
        "FRESH_T_AND_2T_HARD2",
        "FORMAL_TUNING",
        "HELD_OUT",
    }
    report_identity = {
        key: value for key, value in report.items() if key != "report_sha256"
    }
    assert report["report_sha256"] == sha256_json(report_identity)
    assert json.loads(report_path.read_text(encoding="ascii")) == report

    branch["invalid_methods"] = set(hs.HP_METHODS)
    assert hs.analyze_hgp_screen(
        raw_root, manifest_path, REGISTRY_PATH, CONFIG_PATH, artifact_root,
    )["status"] == "UNRESOLVED_NO_HP_PASS"
    branch["invalid_methods"] = {MAP_METHOD_ID}
    assert hs.analyze_hgp_screen(
        raw_root, manifest_path, REGISTRY_PATH, CONFIG_PATH, artifact_root,
    )["status"] == "UNRESOLVED_MAP_MIXTURE_FAIL"
    branch["invalid_methods"] = set()
    branch["agreement_valid"] = False
    assert hs.analyze_hgp_screen(
        raw_root, manifest_path, REGISTRY_PATH, CONFIG_PATH, artifact_root,
    )["status"] == "UNRESOLVED_NO_CROSS_MECHANISM_AGREEMENT"
