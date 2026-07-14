"""Publication-loader tests for fail-closed exp101 scan v3 results."""

import json

import numpy as np
import pytest

from src.scan_results import (
    ESTIMATED_MAP_PURITY_BOUNDS_LABEL,
    MAP_SUCCESS_BOUND_PLOT_LABELS,
    NonPublicationEnsembleError,
    PublicationPointNotReportableError,
    ScanResultsSchemaError,
    UnsupportedScanContractError,
    load_publication_q_top,
    map_success_bound_plot_label,
)


AGGREGATION_POLICY = {
    "point_eligibility": "all_planned_disorders_valid",
    "fraction_denominator": "planned_disorders",
    "maximum_invalid_disorders": 0,
    "maximum_missing_disorders": 0,
    "conditional_statistics_purpose": "diagnostics_only",
    "conditional_statistics_are_publication_eligible": False,
    "crossing_input_policy": "whole_point_nan_unless_reportable",
}


def _fixture_arrays():
    sizes = np.asarray([2, 3], dtype=np.int64)
    q_values = np.asarray([0.05, 0.1], dtype=np.float64)
    crossing = np.asarray([
        [[0.20, 0.40], [0.35, 0.55]],
        [[0.50, 0.70], [0.65, 0.85]],
    ])
    mean = crossing.mean(axis=2)
    sem = crossing.std(axis=2, ddof=1) / np.sqrt(2)
    point_shape = mean.shape
    disorder_shape = crossing.shape
    arrays = {
        "scan_contract_version": np.asarray("exp101.scan.v3"),
        "physics_contract_version": np.asarray("exp101.physics.v2"),
        "canonical_ensemble": np.asarray("true_posterior"),
        "code_size_list": sizes,
        "q_values": q_values,
        "q_top_estimate_per_disorder": crossing.copy(),
        "q_top_crossing_input_per_disorder": crossing.copy(),
        "mean_q_top_estimate": mean.copy(),
        "disorder_sem_q_top_estimate": sem.copy(),
        "mean_q_top": mean.copy(),
        "disorder_sem_q_top": sem.copy(),
        "conditional_mean_q_top_estimate_valid_only": mean.copy(),
        "conditional_disorder_sem_q_top_estimate_valid_only": sem.copy(),
        "aggregation_status_per_point": np.full(
            point_shape, "REPORTABLE", dtype="U24"
        ),
        "aggregation_failure_reasons_per_point": np.full(
            point_shape, "", dtype="U64"
        ),
        "reportable_for_crossing_fss": np.ones(point_shape, dtype=bool),
        "valid_for_aggregation": np.ones(disorder_shape, dtype=bool),
        "planned_disorder_count": np.full(point_shape, 2, dtype=np.int64),
        "present_disorder_count": np.full(point_shape, 2, dtype=np.int64),
        "valid_disorder_count": np.full(point_shape, 2, dtype=np.int64),
        "invalid_disorder_count": np.zeros(point_shape, dtype=np.int64),
        "missing_disorder_count": np.zeros(point_shape, dtype=np.int64),
        "map_success_bound_kind_per_disorder": np.full(
            disorder_shape,
            "sampled_u_statistic_plugin_no_coverage",
            dtype="U48",
        ),
        "map_success_bound_has_confidence_coverage_per_disorder": np.zeros(
            disorder_shape, dtype=bool
        ),
    }
    return arrays


def _set_nonreportable(arrays, index, status):
    i, j = index
    arrays["aggregation_status_per_point"][i, j] = status
    arrays["reportable_for_crossing_fss"][i, j] = False
    arrays["mean_q_top_estimate"][i, j] = np.nan
    arrays["disorder_sem_q_top_estimate"][i, j] = np.nan
    arrays["mean_q_top"][i, j] = np.nan
    arrays["disorder_sem_q_top"][i, j] = np.nan
    arrays["q_top_crossing_input_per_disorder"][i, j] = np.nan
    arrays["valid_for_aggregation"][i, j, 1] = False
    arrays["valid_disorder_count"][i, j] = 1
    arrays["conditional_mean_q_top_estimate_valid_only"][i, j] = arrays[
        "q_top_estimate_per_disorder"
    ][i, j, 0]
    arrays[
        "conditional_disorder_sem_q_top_estimate_valid_only"
    ][i, j] = np.nan
    if status == "INCOMPLETE":
        arrays["aggregation_failure_reasons_per_point"][i, j] = (
            "missing_disorders_present"
        )
        arrays["present_disorder_count"][i, j] = 1
        arrays["missing_disorder_count"][i, j] = 1
        arrays["q_top_estimate_per_disorder"][i, j, 1] = np.nan
    else:
        arrays["aggregation_failure_reasons_per_point"][i, j] = (
            "invalid_disorders_present"
        )
        arrays["invalid_disorder_count"][i, j] = 1


def _write_fixture(path, arrays):
    point_manifest_fields = (
        "planned_disorder_count",
        "present_disorder_count",
        "valid_disorder_count",
        "invalid_disorder_count",
        "missing_disorder_count",
        "aggregation_status_per_point",
        "aggregation_failure_reasons_per_point",
        "reportable_for_crossing_fss",
    )
    manifest = {
        "protocol": str(arrays["scan_contract_version"]),
        "scan_contract_version": str(arrays["scan_contract_version"]),
        "physics_contract_version": str(arrays["physics_contract_version"]),
        "canonical_ensemble": str(arrays["canonical_ensemble"]),
        "code_size_list": arrays["code_size_list"].tolist(),
        "q_values": arrays["q_values"].tolist(),
        "num_disorder_samples": int(
            arrays["q_top_estimate_per_disorder"].shape[2]
        ),
        "aggregation_policy": dict(AGGREGATION_POLICY),
    }
    manifest.update({name: arrays[name].tolist() for name in point_manifest_fields})
    payload = dict(arrays)
    payload["manifest_json"] = np.asarray(json.dumps(manifest))
    np.savez_compressed(path, **payload)
    return path


def test_loads_reportable_v3_and_provides_canonical_bound_labels(tmp_path):
    arrays = _fixture_arrays()
    path = _write_fixture(tmp_path / "reportable.npz", arrays)

    result = load_publication_q_top(path)

    assert result.scan_contract_version == "exp101.scan.v3"
    assert result.canonical_ensemble == "true_posterior"
    assert result.selected_point_count == 4
    assert np.array_equal(result.point_mask, np.ones((2, 2), dtype=bool))
    assert np.allclose(result.mean_q_top_estimate, [[0.3, 0.45], [0.6, 0.75]])
    assert np.allclose(
        result.q_top_crossing_input_per_disorder,
        arrays["q_top_crossing_input_per_disorder"],
    )
    assert np.all(
        result.map_success_bound_plot_label_per_disorder
        == ESTIMATED_MAP_PURITY_BOUNDS_LABEL
    )
    assert not result.mean_q_top_estimate.flags.writeable


def test_point_mask_excludes_predeclared_nonreportable_region(tmp_path):
    arrays = _fixture_arrays()
    _set_nonreportable(arrays, (1, 1), "INCOMPLETE")
    path = _write_fixture(tmp_path / "masked.npz", arrays)
    mask = np.asarray([[True, False], [True, False]])

    result = load_publication_q_top(path, point_mask=mask)

    assert result.selected_point_count == 2
    assert np.isfinite(result.mean_q_top_estimate[0, 0])
    assert np.isfinite(result.mean_q_top_estimate[1, 0])
    assert np.isnan(result.mean_q_top_estimate[:, 1]).all()
    assert np.isnan(
        result.q_top_crossing_input_per_disorder[:, 1]
    ).all()
    assert np.all(
        result.map_success_bound_plot_label_per_disorder[:, 1] == ""
    )


@pytest.mark.parametrize(
    ("status", "reason", "count_fragment"),
    [
        (
            "SAMPLING_INSUFFICIENT",
            "invalid_disorders_present",
            "invalid=1, missing=0",
        ),
        ("INCOMPLETE", "missing_disorders_present", "invalid=0, missing=1"),
    ],
)
def test_rejects_selected_sampling_failure_with_coordinates_and_reason(
    tmp_path, status, reason, count_fragment
):
    arrays = _fixture_arrays()
    _set_nonreportable(arrays, (1, 0), status)
    path = _write_fixture(tmp_path / f"{status}.npz", arrays)

    with pytest.raises(PublicationPointNotReportableError) as caught:
        load_publication_q_top(path)

    message = str(caught.value)
    assert "size=3, q=0.05" in message
    assert f"status={status}" in message
    assert count_fragment in message
    assert f"reason={reason}" in message


def test_rejects_v2_without_trying_to_infer_eligibility(tmp_path):
    path = tmp_path / "v2.npz"
    np.savez_compressed(
        path,
        scan_contract_version=np.asarray("exp101.scan.v2"),
        canonical_ensemble=np.asarray("true_posterior"),
        mean_q_top_estimate=np.asarray([[0.2]]),
    )

    with pytest.raises(UnsupportedScanContractError, match="audit-only"):
        load_publication_q_top(path)


def test_rejects_legacy_before_interpreting_formal_fields(tmp_path):
    path = tmp_path / "legacy.npz"
    np.savez_compressed(
        path,
        scan_contract_version=np.asarray("exp101.scan.v3"),
        canonical_ensemble=np.asarray("legacy_delta_only"),
        formal_q_top_per_disorder=np.asarray([[[0.8]]]),
    )

    with pytest.raises(NonPublicationEnsembleError, match="formal-only"):
        load_publication_q_top(path)


@pytest.mark.parametrize("corruption", ["mean", "crossing"])
def test_rejects_malicious_reportable_schema_inconsistency(
    tmp_path, corruption
):
    arrays = _fixture_arrays()
    if corruption == "mean":
        arrays["mean_q_top_estimate"][0, 1] = 0.99
        expected = "mean_q_top_estimate"
    else:
        arrays["q_top_crossing_input_per_disorder"][0, 1, 0] = 0.99
        expected = "crossing input disagrees"
    path = _write_fixture(tmp_path / f"bad_{corruption}.npz", arrays)

    with pytest.raises(ScanResultsSchemaError) as caught:
        load_publication_q_top(path)

    message = str(caught.value)
    assert "size=2, q=0.1" in message
    assert expected in message


def test_point_mask_must_be_boolean_nonempty_and_match_grid(tmp_path):
    path = _write_fixture(tmp_path / "mask.npz", _fixture_arrays())

    with pytest.raises(ScanResultsSchemaError, match="boolean"):
        load_publication_q_top(path, point_mask=np.ones((2, 2), dtype=np.int8))
    with pytest.raises(ScanResultsSchemaError, match="shape"):
        load_publication_q_top(path, point_mask=np.ones((1, 2), dtype=bool))
    with pytest.raises(ScanResultsSchemaError, match="at least one"):
        load_publication_q_top(path, point_mask=np.zeros((2, 2), dtype=bool))


def test_bound_kind_mapping_uses_one_required_plugin_label():
    assert MAP_SUCCESS_BOUND_PLOT_LABELS[
        "full_sector_ti_plugin_no_coverage"
    ] == ESTIMATED_MAP_PURITY_BOUNDS_LABEL
    assert MAP_SUCCESS_BOUND_PLOT_LABELS[
        "sampled_u_statistic_plugin_no_coverage"
    ] == ESTIMATED_MAP_PURITY_BOUNDS_LABEL
    assert map_success_bound_plot_label(
        "sampled_u_statistic_plugin_no_coverage"
    ) == "Estimated MAP-purity bounds (plug-in; no confidence coverage)"
    with pytest.raises(ValueError, match="unknown map_success_bound_kind"):
        map_success_bound_plot_label("invented")
