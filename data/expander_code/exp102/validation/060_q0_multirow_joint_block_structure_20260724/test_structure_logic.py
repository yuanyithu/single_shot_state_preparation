"""Focused pre-run tests for the validation-060 structural screen."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import analyze_structure as primary
import audit_structure as independent
import preflight_structure as preflight


TOY_H = np.asarray([
    [1, 1, 0, 0, 1],
    [1, 0, 1, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 1, 1],
], dtype=np.uint8)


def test_multirow_scopes_match_direct_perturbations_and_row_symmetry():
    rank = TOY_H.shape[0]
    for row_count in (1, 2, 3):
        expected = primary.factor_scopes_multirow(TOY_H, row_count)
        for rows in combinations(range(rank), row_count):
            coordinates = independent._coordinates_multirow(rank, rows)
            assert independent.semantic_scopes(TOY_H, coordinates) == expected


def test_row_column_scopes_match_direct_perturbations_and_row_symmetry():
    rank = TOY_H.shape[0]
    for selected_column in range(rank):
        expected = primary.factor_scopes_row_column(TOY_H, selected_column)
        for selected_row in range(rank):
            coordinates = independent._coordinates_row_column(
                rank, selected_row, selected_column,
            )
            assert independent.semantic_scopes(TOY_H, coordinates) == expected


def test_min_fill_set_and_bitset_implementations_are_identical():
    cases = [
        (4, ((0,), (1,), (2,), (3,), (0, 1), (1, 2), (0, 2, 3))),
        (8, primary.factor_scopes_multirow(TOY_H, 2)),
        (7, primary.factor_scopes_row_column(TOY_H, 2)),
    ]
    for variable_count, scopes in cases:
        assert primary.min_fill_plan(
            variable_count, scopes,
        ) == independent.min_fill_bitset(variable_count, scopes)


def test_primary_semantic_audit_rejects_a_tampered_scope():
    scopes = list(primary.factor_scopes_multirow(TOY_H, 2))
    first_factor = len(scopes) - TOY_H.shape[1]
    scopes[first_factor] = tuple(
        value for value in scopes[first_factor] if value != scopes[first_factor][0]
    )
    with pytest.raises(RuntimeError, match="factor scope"):
        primary.scope_semantic_audit(TOY_H, "multirow", 2, tuple(scopes))


def test_config_binds_every_local_source_artifact():
    config = preflight.load_json_strict(preflight.CONFIG_PATH)
    artifacts = preflight._configured_artifacts(config)
    assert len(artifacts) >= 8
    for path, expected in artifacts:
        assert path.is_file()
        assert preflight.sha256_file(path) == expected
    assert config["selection_policy"]["successor_authority"] == (
        "CONTINGENCY_ONLY_IF_HP64_STAGE3_OR_STAGE4_FAILS"
    )
    assert [row["id"] for row in config["candidates"]] == [
        "MR2", "MR3", "MR4", "RC1",
    ]
    assert config["elimination"] == {
        "algorithm": "deterministic_min_fill",
        "tie_break": [
            "missing_fill_edges", "live_degree", "variable_index",
        ],
    }
    assert config["gates"] == {
        "max_induced_width": 25,
        "max_initial_factor_scope": 26,
        "max_largest_factor_entries": 1 << 26,
        "max_single_table_bytes": 512 << 20,
    }


def test_launch_guard_rejects_a_dirty_or_untracked_source(monkeypatch):
    config = preflight.load_json_strict(preflight.CONFIG_PATH)

    def fake_git(args, *, text=True):
        if args == ["rev-parse", "--show-toplevel"]:
            return str(preflight.PROJECT_ROOT) + "\n"
        if args == ["rev-parse", "HEAD"]:
            return "a" * 40 + "\n"
        if args[:3] == ["status", "--porcelain=v1", "--untracked-files=all"]:
            return "?? data/expander_code/exp102/validation/060/source.py\n"
        raise AssertionError(f"unexpected git call: {args}")

    monkeypatch.setattr(preflight, "_git", fake_git)
    with pytest.raises(RuntimeError, match="clean committed worktree"):
        preflight.verify_for_launch(config)


def test_paths_cannot_escape_the_exp102_root():
    with pytest.raises(RuntimeError, match="escapes exp102 root"):
        preflight._exp102_path("../outside")
    with pytest.raises(RuntimeError, match="escapes exp102 root"):
        independent._exp102_path("../outside")
