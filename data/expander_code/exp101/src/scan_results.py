"""Fail-closed access to publication-eligible exp101 q_top scan data."""

from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


SCAN_CONTRACT_VERSION = "exp101.scan.v3"
PHYSICS_CONTRACT_VERSION = "exp101.physics.v2"
PUBLICATION_ENSEMBLE = "true_posterior"

ESTIMATED_MAP_PURITY_BOUNDS_LABEL = (
    "Estimated MAP-purity bounds (plug-in; no confidence coverage)"
)
MAP_SUCCESS_BOUND_PLOT_LABELS = MappingProxyType({
    "exact_posterior_algebraic": (
        "Algebraic MAP-purity bounds (exact posterior)"
    ),
    "analytic_endpoint_algebraic": (
        "Algebraic MAP-purity bounds (analytic endpoint)"
    ),
    "full_sector_ti_plugin_no_coverage": (
        ESTIMATED_MAP_PURITY_BOUNDS_LABEL
    ),
    "sampled_u_statistic_plugin_no_coverage": (
        ESTIMATED_MAP_PURITY_BOUNDS_LABEL
    ),
    "unavailable": "MAP-purity bounds unavailable",
})
# Descriptive alias for callers that prefer the schema field's terminology.
MAP_SUCCESS_BOUND_KIND_TO_PLOT_LABEL = MAP_SUCCESS_BOUND_PLOT_LABELS

_EXPECTED_AGGREGATION_POLICY = {
    "point_eligibility": "all_planned_disorders_valid",
    "fraction_denominator": "planned_disorders",
    "maximum_invalid_disorders": 0,
    "maximum_missing_disorders": 0,
    "conditional_statistics_purpose": "diagnostics_only",
    "conditional_statistics_are_publication_eligible": False,
    "crossing_input_policy": "whole_point_nan_unless_reportable",
}


class PublicationQTopError(ValueError):
    """Base class for publication-loader rejections."""


class UnsupportedScanContractError(PublicationQTopError):
    """The file uses a scan contract that is not publication eligible."""


class NonPublicationEnsembleError(PublicationQTopError):
    """The file contains a formal/legacy rather than paper posterior."""


class PublicationPointNotReportableError(PublicationQTopError):
    """At least one selected parameter point failed the publication gate."""


class ScanResultsSchemaError(PublicationQTopError):
    """The v3 file is incomplete or internally inconsistent."""


@dataclass(frozen=True)
class PublicationQTopData:
    """Validated q_top arrays; unselected parameter points are masked by NaN."""

    source_path: Path
    scan_contract_version: str
    physics_contract_version: str
    canonical_ensemble: str
    manifest: Mapping[str, Any]
    code_size_list: np.ndarray
    q_values: np.ndarray
    point_mask: np.ndarray
    mean_q_top_estimate: np.ndarray
    disorder_sem_q_top_estimate: np.ndarray
    q_top_crossing_input_per_disorder: np.ndarray
    map_success_bound_kind_per_disorder: np.ndarray
    map_success_bound_plot_label_per_disorder: np.ndarray

    @property
    def selected_point_count(self):
        return int(np.count_nonzero(self.point_mask))

    @property
    def mean_q_top(self):
        """Safe compatibility alias for the validated publication mean."""
        return self.mean_q_top_estimate


def map_success_bound_plot_label(kind):
    """Return the canonical plotting label for a v3 MAP-bound kind."""
    try:
        return MAP_SUCCESS_BOUND_PLOT_LABELS[str(kind)]
    except KeyError as error:
        raise ValueError(f"unknown map_success_bound_kind {kind!r}") from error


def _readonly(array, dtype=None):
    result = np.array(array, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _float_array(name, value):
    try:
        return np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ScanResultsSchemaError(
            f"scan v3 field {name!r} must be numeric"
        ) from error


def _required(data, name):
    if name not in data.files:
        raise ScanResultsSchemaError(
            f"scan v3 schema is missing required field {name!r}"
        )
    return np.asarray(data[name])


def _scalar_text(data, name):
    value = _required(data, name)
    if value.ndim != 0:
        raise ScanResultsSchemaError(
            f"scan v3 field {name!r} must be a scalar string; "
            f"got shape {value.shape}"
        )
    return str(value.item())


def _point_description(size, q_value):
    return f"size={int(size)}, q={float(q_value)!r}"


def _point_schema_error(message, mask, sizes, q_values):
    locations = []
    for i, j in np.argwhere(mask):
        locations.append(
            f"{_point_description(sizes[i], q_values[j])}: {message(i, j)}"
        )
    raise ScanResultsSchemaError(
        "scan v3 publication schema is inconsistent at selected point(s): "
        + " | ".join(locations)
    )


def _require_shape(name, array, shape):
    if array.shape != shape:
        raise ScanResultsSchemaError(
            f"scan v3 field {name!r} has shape {array.shape}; expected {shape}"
        )


def _manifest_array(manifest, name, shape):
    if name not in manifest:
        raise ScanResultsSchemaError(
            f"scan v3 manifest is missing required field {name!r}"
        )
    result = np.asarray(manifest[name])
    if result.shape != shape:
        raise ScanResultsSchemaError(
            f"scan v3 manifest field {name!r} has shape {result.shape}; "
            f"expected {shape}"
        )
    return result


def _validate_manifest(
    manifest, sizes, q_values, num_disorders, point_mask, point_arrays,
):
    for key, expected in (
        ("protocol", SCAN_CONTRACT_VERSION),
        ("scan_contract_version", SCAN_CONTRACT_VERSION),
        ("physics_contract_version", PHYSICS_CONTRACT_VERSION),
        ("canonical_ensemble", PUBLICATION_ENSEMBLE),
    ):
        if manifest.get(key) != expected:
            raise ScanResultsSchemaError(
                f"scan v3 manifest field {key!r} is "
                f"{manifest.get(key)!r}; expected {expected!r}"
            )
    if manifest.get("num_disorder_samples") != num_disorders:
        raise ScanResultsSchemaError(
            "scan v3 manifest num_disorder_samples does not match the "
            "per-disorder array width"
        )
    if not np.array_equal(np.asarray(manifest.get("code_size_list")), sizes):
        raise ScanResultsSchemaError(
            "scan v3 manifest code_size_list disagrees with the NPZ field"
        )
    try:
        manifest_q = np.asarray(manifest["q_values"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise ScanResultsSchemaError(
            "scan v3 manifest has no valid q_values"
        ) from error
    if manifest_q.shape != q_values.shape or not np.array_equal(
        manifest_q, q_values
    ):
        raise ScanResultsSchemaError(
            "scan v3 manifest q_values disagree with the NPZ field"
        )

    policy = manifest.get("aggregation_policy")
    if not isinstance(policy, dict):
        raise ScanResultsSchemaError(
            "scan v3 manifest has no aggregation_policy object"
        )
    for key, expected in _EXPECTED_AGGREGATION_POLICY.items():
        if policy.get(key) != expected:
            raise ScanResultsSchemaError(
                f"scan v3 aggregation_policy[{key!r}] is "
                f"{policy.get(key)!r}; expected {expected!r}"
            )

    point_shape = point_mask.shape
    for name, npz_array in point_arrays.items():
        manifest_array = _manifest_array(manifest, name, point_shape)
        if not np.array_equal(
            manifest_array[point_mask], npz_array[point_mask]
        ):
            raise ScanResultsSchemaError(
                f"scan v3 manifest field {name!r} disagrees with the NPZ "
                "field on selected point(s)"
            )


def load_publication_q_top(path, point_mask=None):
    """Load publication-safe q_top values from a fail-closed scan v3 NPZ.

    A mask selects a predeclared analysis region.  Every selected point must
    have all planned disorders present and valid.  Unselected output cells are
    replaced by NaN/empty labels so callers cannot accidentally publish them.
    """
    source_path = Path(path)
    try:
        archive = np.load(source_path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ScanResultsSchemaError(
            f"cannot read scan results from {source_path}: {error}"
        ) from error

    with archive as data:
        scan_contract = _scalar_text(data, "scan_contract_version")
        if scan_contract != SCAN_CONTRACT_VERSION:
            detail = (
                "scan v2 results are audit-only; publication eligibility "
                "cannot be inferred or migrated from conditional means"
                if scan_contract == "exp101.scan.v2"
                else "only scan v3 is publication eligible"
            )
            raise UnsupportedScanContractError(
                f"publication q_top loader requires {SCAN_CONTRACT_VERSION!r}; "
                f"got {scan_contract!r}: {detail}"
            )

        ensemble = _scalar_text(data, "canonical_ensemble")
        if ensemble != PUBLICATION_ENSEMBLE:
            detail = (
                "legacy_delta_only is formal-only and cannot be used for "
                "publication aggregation"
                if ensemble == "legacy_delta_only"
                else "the ensemble is not the paper posterior"
            )
            raise NonPublicationEnsembleError(
                f"publication q_top loader requires canonical_ensemble="
                f"{PUBLICATION_ENSEMBLE!r}; got {ensemble!r}: {detail}"
            )

        physics_contract = _scalar_text(data, "physics_contract_version")
        if physics_contract != PHYSICS_CONTRACT_VERSION:
            raise ScanResultsSchemaError(
                f"scan v3 physics_contract_version is {physics_contract!r}; "
                f"expected {PHYSICS_CONTRACT_VERSION!r}"
            )

        sizes = _required(data, "code_size_list")
        q_values = _required(data, "q_values")
        if sizes.ndim != 1 or not sizes.size:
            raise ScanResultsSchemaError(
                "scan v3 code_size_list must be a nonempty one-dimensional array"
            )
        if q_values.ndim != 1 or not q_values.size:
            raise ScanResultsSchemaError(
                "scan v3 q_values must be a nonempty one-dimensional array"
            )
        if not np.issubdtype(sizes.dtype, np.integer):
            raise ScanResultsSchemaError(
                "scan v3 code_size_list must contain integers"
            )
        try:
            q_values = np.asarray(q_values, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ScanResultsSchemaError(
                "scan v3 q_values must be numeric"
            ) from error
        if not np.isfinite(q_values).all():
            raise ScanResultsSchemaError("scan v3 q_values must all be finite")

        point_shape = (sizes.size, q_values.size)
        if point_mask is None:
            selected = np.ones(point_shape, dtype=bool)
        else:
            supplied_mask = np.asarray(point_mask)
            if supplied_mask.dtype != np.dtype(bool):
                raise ScanResultsSchemaError(
                    "point_mask must be a boolean array"
                )
            if supplied_mask.shape != point_shape:
                raise ScanResultsSchemaError(
                    f"point_mask has shape {supplied_mask.shape}; expected "
                    f"{point_shape}"
                )
            selected = supplied_mask.copy()
        if not selected.any():
            raise ScanResultsSchemaError(
                "point_mask must select at least one parameter point"
            )

        q_top_raw = _required(data, "q_top_estimate_per_disorder")
        if q_top_raw.ndim != 3 or q_top_raw.shape[:2] != point_shape:
            raise ScanResultsSchemaError(
                "scan v3 q_top_estimate_per_disorder must have shape "
                f"{point_shape} + (num_disorders,)"
            )
        num_disorders = q_top_raw.shape[2]
        if num_disorders < 1:
            raise ScanResultsSchemaError(
                "scan v3 must plan at least one disorder per point"
            )
        disorder_shape = (*point_shape, num_disorders)

        point_field_names = (
            "mean_q_top_estimate",
            "disorder_sem_q_top_estimate",
            "mean_q_top",
            "disorder_sem_q_top",
            "conditional_mean_q_top_estimate_valid_only",
            "conditional_disorder_sem_q_top_estimate_valid_only",
            "aggregation_status_per_point",
            "aggregation_failure_reasons_per_point",
            "reportable_for_crossing_fss",
            "planned_disorder_count",
            "present_disorder_count",
            "valid_disorder_count",
            "invalid_disorder_count",
            "missing_disorder_count",
        )
        point_fields = {name: _required(data, name) for name in point_field_names}
        for name, array in point_fields.items():
            _require_shape(name, array, point_shape)

        crossing = _required(data, "q_top_crossing_input_per_disorder")
        valid_mask = _required(data, "valid_for_aggregation")
        bound_kinds = _required(data, "map_success_bound_kind_per_disorder")
        bound_coverage = _required(
            data,
            "map_success_bound_has_confidence_coverage_per_disorder",
        )
        for name, array in (
            ("q_top_crossing_input_per_disorder", crossing),
            ("valid_for_aggregation", valid_mask),
            ("map_success_bound_kind_per_disorder", bound_kinds),
            (
                "map_success_bound_has_confidence_coverage_per_disorder",
                bound_coverage,
            ),
        ):
            _require_shape(name, array, disorder_shape)
        if valid_mask.dtype != np.dtype(bool):
            raise ScanResultsSchemaError(
                "scan v3 valid_for_aggregation must be boolean"
            )
        if point_fields["reportable_for_crossing_fss"].dtype != np.dtype(bool):
            raise ScanResultsSchemaError(
                "scan v3 reportable_for_crossing_fss must be boolean"
            )
        if bound_coverage.dtype != np.dtype(bool):
            raise ScanResultsSchemaError(
                "scan v3 bound confidence-coverage metadata must be boolean"
            )
        q_top_raw = _float_array("q_top_estimate_per_disorder", q_top_raw)
        crossing = _float_array(
            "q_top_crossing_input_per_disorder", crossing
        )

        count_names = (
            "planned_disorder_count",
            "present_disorder_count",
            "valid_disorder_count",
            "invalid_disorder_count",
            "missing_disorder_count",
        )
        for name in count_names:
            if not np.issubdtype(point_fields[name].dtype, np.integer):
                raise ScanResultsSchemaError(
                    f"scan v3 count field {name!r} must contain integers"
                )

        try:
            manifest_raw = _required(data, "manifest_json")
            if manifest_raw.ndim != 0:
                raise ScanResultsSchemaError(
                    "scan v3 manifest_json must be a scalar string"
                )
            manifest = json.loads(str(manifest_raw.item()))
        except json.JSONDecodeError as error:
            raise ScanResultsSchemaError(
                "scan v3 manifest_json is not valid JSON"
            ) from error
        if not isinstance(manifest, dict):
            raise ScanResultsSchemaError(
                "scan v3 manifest_json must encode an object"
            )

        manifest_point_arrays = {
            name: point_fields[name]
            for name in (
                "planned_disorder_count",
                "present_disorder_count",
                "valid_disorder_count",
                "invalid_disorder_count",
                "missing_disorder_count",
                "aggregation_status_per_point",
                "aggregation_failure_reasons_per_point",
                "reportable_for_crossing_fss",
            )
        }
        _validate_manifest(
            manifest, sizes, q_values, num_disorders, selected,
            manifest_point_arrays,
        )

        planned = point_fields["planned_disorder_count"]
        present = point_fields["present_disorder_count"]
        valid_count = point_fields["valid_disorder_count"]
        invalid = point_fields["invalid_disorder_count"]
        missing = point_fields["missing_disorder_count"]

        bad_count_identity = selected & (
            (planned != num_disorders)
            | (present + missing != planned)
            | (valid_count + invalid != present)
            | (valid_count != valid_mask.sum(axis=2))
        )
        if bad_count_identity.any():
            _point_schema_error(
                lambda i, j: (
                    "planned/present/valid/invalid/missing counts or validity "
                    "mask disagree "
                    f"(planned={planned[i, j]}, present={present[i, j]}, "
                    f"valid={valid_count[i, j]}, invalid={invalid[i, j]}, "
                    f"missing={missing[i, j]})"
                ),
                bad_count_identity,
                sizes,
                q_values,
            )

        statuses = point_fields["aggregation_status_per_point"].astype(str)
        reasons = point_fields[
            "aggregation_failure_reasons_per_point"
        ].astype(str)
        reportable = point_fields["reportable_for_crossing_fss"]
        not_reportable = selected & (
            (statuses != "REPORTABLE")
            | ~reportable
            | (invalid != 0)
            | (missing != 0)
            | (present != planned)
            | (valid_count != planned)
        )
        if not_reportable.any():
            details = []
            for i, j in np.argwhere(not_reportable):
                reason = reasons[i, j] or "publication gate not satisfied"
                details.append(
                    f"{_point_description(sizes[i], q_values[j])}: "
                    f"status={statuses[i, j]}, invalid={invalid[i, j]}, "
                    f"missing={missing[i, j]}, reason={reason}"
                )
            raise PublicationPointNotReportableError(
                "selected q_top point(s) are not publication-reportable: "
                + " | ".join(details)
            )

        nonempty_reasons = selected & (np.char.str_len(reasons) != 0)
        if nonempty_reasons.any():
            _point_schema_error(
                lambda i, j: (
                    "REPORTABLE point has aggregation failure reason "
                    f"{reasons[i, j]!r}"
                ),
                nonempty_reasons,
                sizes,
                q_values,
            )

        selected_disorders = selected[..., None]
        if not valid_mask[selected].all():
            _point_schema_error(
                lambda _i, _j: "REPORTABLE point contains an invalid disorder",
                selected,
                sizes,
                q_values,
            )
        if not np.isfinite(crossing[selected]).all():
            bad = selected & ~np.isfinite(crossing).all(axis=2)
            _point_schema_error(
                lambda _i, _j: (
                    "REPORTABLE crossing input contains a non-finite value"
                ),
                bad,
                sizes,
                q_values,
            )
        if not np.isfinite(q_top_raw[selected]).all():
            bad = selected & ~np.isfinite(q_top_raw).all(axis=2)
            _point_schema_error(
                lambda _i, _j: "REPORTABLE raw q_top contains a non-finite value",
                bad,
                sizes,
                q_values,
            )
        raw_mismatch = selected & ~np.isclose(
            crossing, q_top_raw, rtol=1e-12, atol=1e-12,
        ).all(axis=2)
        if raw_mismatch.any():
            _point_schema_error(
                lambda _i, _j: (
                    "crossing input disagrees with q_top_estimate_per_disorder"
                ),
                raw_mismatch,
                sizes,
                q_values,
            )

        expected_mean = crossing.mean(axis=2)
        expected_sem = np.full(point_shape, np.nan, dtype=np.float64)
        if num_disorders > 1:
            expected_sem = crossing.std(axis=2, ddof=1) / np.sqrt(num_disorders)
        official_mean = _float_array(
            "mean_q_top_estimate", point_fields["mean_q_top_estimate"]
        )
        official_sem = _float_array(
            "disorder_sem_q_top_estimate",
            point_fields["disorder_sem_q_top_estimate"],
        )
        mean_mismatch = selected & ~np.isclose(
            official_mean, expected_mean, rtol=1e-12, atol=1e-12,
        )
        if mean_mismatch.any():
            _point_schema_error(
                lambda i, j: (
                    f"mean_q_top_estimate={official_mean[i, j]!r} does not "
                    f"equal crossing mean {expected_mean[i, j]!r}"
                ),
                mean_mismatch,
                sizes,
                q_values,
            )
        if num_disorders == 1:
            sem_mismatch = selected & ~np.isnan(official_sem)
        else:
            sem_mismatch = selected & ~np.isclose(
                official_sem, expected_sem, rtol=1e-12, atol=1e-12,
            )
        if sem_mismatch.any():
            _point_schema_error(
                lambda i, j: (
                    f"disorder_sem_q_top_estimate={official_sem[i, j]!r} "
                    f"does not equal crossing SEM {expected_sem[i, j]!r}"
                ),
                sem_mismatch,
                sizes,
                q_values,
            )

        for name, expected in (
            ("mean_q_top", official_mean),
            ("conditional_mean_q_top_estimate_valid_only", official_mean),
            ("disorder_sem_q_top", official_sem),
            (
                "conditional_disorder_sem_q_top_estimate_valid_only",
                official_sem,
            ),
        ):
            observed = _float_array(name, point_fields[name])
            mismatch = selected & ~np.isclose(
                observed, expected, rtol=1e-12, atol=1e-12,
                equal_nan=True,
            )
            if mismatch.any():
                _point_schema_error(
                    lambda i, j, field=name: (
                        f"{field} disagrees with its publication-safe value"
                    ),
                    mismatch,
                    sizes,
                    q_values,
                )

        labels = np.full(disorder_shape, "", dtype="U72")
        unknown_kind = np.zeros(point_shape, dtype=bool)
        for i, j in np.argwhere(selected):
            for d in range(num_disorders):
                kind = str(bound_kinds[i, j, d])
                if kind not in MAP_SUCCESS_BOUND_PLOT_LABELS:
                    unknown_kind[i, j] = True
                    continue
                labels[i, j, d] = MAP_SUCCESS_BOUND_PLOT_LABELS[kind]
        if unknown_kind.any():
            _point_schema_error(
                lambda i, j: (
                    "unknown map_success_bound_kind value(s): "
                    f"{bound_kinds[i, j].astype(str).tolist()}"
                ),
                unknown_kind,
                sizes,
                q_values,
            )
        if bound_coverage[selected].any():
            _point_schema_error(
                lambda _i, _j: (
                    "map_success_bound_has_confidence_coverage must be false"
                ),
                selected & bound_coverage.any(axis=2),
                sizes,
                q_values,
            )

        masked_mean = np.where(selected, official_mean, np.nan)
        masked_sem = np.where(selected, official_sem, np.nan)
        masked_crossing = np.where(selected_disorders, crossing, np.nan)
        masked_kinds = np.where(selected_disorders, bound_kinds.astype(str), "")

    return PublicationQTopData(
        source_path=source_path,
        scan_contract_version=scan_contract,
        physics_contract_version=physics_contract,
        canonical_ensemble=ensemble,
        manifest=manifest,
        code_size_list=_readonly(sizes),
        q_values=_readonly(q_values, dtype=np.float64),
        point_mask=_readonly(selected, dtype=bool),
        mean_q_top_estimate=_readonly(masked_mean, dtype=np.float64),
        disorder_sem_q_top_estimate=_readonly(masked_sem, dtype=np.float64),
        q_top_crossing_input_per_disorder=_readonly(
            masked_crossing, dtype=np.float64
        ),
        map_success_bound_kind_per_disorder=_readonly(masked_kinds),
        map_success_bound_plot_label_per_disorder=_readonly(labels),
    )


__all__ = [
    "ESTIMATED_MAP_PURITY_BOUNDS_LABEL",
    "MAP_SUCCESS_BOUND_KIND_TO_PLOT_LABEL",
    "MAP_SUCCESS_BOUND_PLOT_LABELS",
    "NonPublicationEnsembleError",
    "PublicationPointNotReportableError",
    "PublicationQTopData",
    "PublicationQTopError",
    "ScanResultsSchemaError",
    "UnsupportedScanContractError",
    "load_publication_q_top",
    "map_success_bound_plot_label",
]
