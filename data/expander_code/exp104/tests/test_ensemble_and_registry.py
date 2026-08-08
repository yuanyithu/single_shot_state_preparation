"""The ensemble is a rule; the registry only records what the rule produced."""

import collections

import numpy as np
import pytest

from data.expander_code.exp104.exp104_pipeline.config import (
    CODES_PER_M,
    M_VALUES,
    code_id,
)
from data.expander_code.exp104.exp104_pipeline.ensemble import (
    build_candidate,
    census,
    generate_codes,
    load_registry,
    matrix_sha256,
    rebuild_code,
)
from data.expander_code.exp104.exp104_pipeline.model import load_model


def test_registry_covers_the_whole_panel_exactly_once(registry):
    assert len(registry["codes"]) == len(M_VALUES) * CODES_PER_M
    ids = [row["code_id"] for row in registry["codes"]]
    assert len(set(ids)) == len(ids)
    assert set(ids) == {
        code_id(m, index) for m in M_VALUES for index in range(CODES_PER_M)
    }


def test_every_accepted_code_has_the_frozen_family_parameters(registry):
    for row in registry["codes"]:
        m = row["m"]
        assert row["classical_rank"] == 3 * m
        assert row["n"] == 25 * m ** 2
        assert row["k"] == m ** 2
        assert row["classical_distance"] >= 2


def test_parity_check_matrices_are_pairwise_distinct_within_each_m(registry):
    by_m = collections.defaultdict(set)
    for row in registry["codes"]:
        by_m[row["m"]].add(row["classical_H_sha256"])
    for m in M_VALUES:
        assert len(by_m[m]) == CODES_PER_M


def test_codes_rebuild_from_their_seeds(registry):
    # Spot check across the panel: reconstruction is the reproducibility claim.
    rows = registry["codes"]
    for row in rows[::997]:
        H = rebuild_code(row)
        assert matrix_sha256(H) == row["classical_H_sha256"]
        assert H.shape == (3 * row["m"], 4 * row["m"])


def test_tampering_with_a_registry_row_is_detected(registry):
    row = dict(registry["codes"][0])
    row["graph_seed"] = int(row["graph_seed"]) + 1
    with pytest.raises(ValueError):
        rebuild_code(row)


def test_registry_hash_binds_the_whole_file(tmp_path, registry):
    import json

    from data.expander_code.exp104.exp104_pipeline.io import atomic_json

    tampered = json.loads(json.dumps(registry))
    tampered["codes"][5]["classical_distance"] = 999
    path = tmp_path / "registry.json"
    atomic_json(path, tampered)
    with pytest.raises(ValueError):
        load_registry(path)


def test_acceptance_rule_rejects_rank_deficient_candidates():
    """The only rejections are algebraic, and they really do occur."""
    from data.expander_code.exp104.exp104_pipeline.config import (
        MASTER_SEED_HEX,
        NAMESPACES,
    )

    rejected = 0
    for candidate_index in range(60):
        candidate = build_candidate(
            MASTER_SEED_HEX, NAMESPACES["ensemble"], 3, candidate_index,
        )
        if not candidate["full_row_rank"]:
            rejected += 1
            assert candidate["classical_rank"] < 9
    assert rejected > 0


def test_generation_is_deterministic_for_a_given_seed():
    first, first_stats = generate_codes(3, 12)
    second, second_stats = generate_codes(3, 12)
    assert first == second
    assert first_stats == second_stats


def test_registry_prefix_is_stable_under_extension():
    """Accepting more codes must not renumber the ones already accepted."""
    short, _ = generate_codes(4, 8)
    longer, _ = generate_codes(4, 20)
    assert longer[:8] == short


def test_census_reports_a_normalized_distance_distribution():
    report = census(3, 120, "ab" * 32, "exp104.test.census")
    assert report["accepted"] == 120
    assert 0.0 < report["acceptance_rate"] <= 1.0
    assert abs(sum(report["distance_fractions"].values()) - 1.0) < 1e-12
    assert sum(report["distance_counts"].values()) == 120
    assert all(int(key) % 2 == 0 for key in report["distance_counts"])


def test_model_build_agrees_with_the_registry_row(registry_rows):
    for identifier in ("m03_c00000", "m05_c00007", "m08_c00003"):
        row = registry_rows[identifier]
        model = load_model(row)
        assert model.code_id == identifier
        assert model.n == row["n"] == model.H_Z.shape[1]
        assert model.k == row["k"] == model.logical_Z.shape[0]
        assert model.classical_H_sha256 == row["classical_H_sha256"]
        assert len(model.logical_frame_sha256) == 64
        # Logical operators must commute with the stabilizers they are paired to.
        product = np.asarray(model.H_Z @ model.H_X.T, dtype=np.uint8) & np.uint8(1)
        assert not product.any()


def test_model_rejects_a_row_with_unexpected_fields(registry_rows):
    row = dict(registry_rows["m03_c00000"])
    row["extra"] = 1
    with pytest.raises(ValueError):
        load_model(row)
