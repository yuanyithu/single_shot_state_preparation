"""Independent re-verification of a published exp106 aggregate.

The loader recomputes everything it can from the stored per-code counts rather
than trusting the summary fields: rates, Wilson intervals, pooled means, cluster
standard errors, the strata table, the terminal decision and the crossing
location. A published aggregate that this loader will not accept is not a
result.
"""

import numpy as np

from . import AGGREGATE_SCHEMA, EXPERIMENT_ID
from .aggregate import (
    ARRAY_FIELDS,
    DISTANCE_STRATA,
    SCALAR_FIELDS,
    panel_layout,
    qtop_lower_bound,
)
from .config import ensure_config
from .crossing import (
    CERTIFIED,
    classify_crossing,
    cluster_bootstrap,
    crossing_location,
    wilson_interval,
)
from .io import arrays_sha256, canonical_json


TOLERANCE = 1e-9


def _load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        missing = (set(ARRAY_FIELDS) | set(SCALAR_FIELDS)) - set(data.files)
        extra = set(data.files) - (set(ARRAY_FIELDS) | set(SCALAR_FIELDS))
        if missing or extra:
            raise ValueError("aggregate NPZ fields do not match exp106.aggregate.v1")
        payload = {}
        for key in data.files:
            value = data[key]
            payload[key] = value.copy() if key in ARRAY_FIELDS else value.item()
    return payload


def load_exp106_crossing(path_or_payload, config, require_complete=True):
    config = ensure_config(config)
    payload = (
        path_or_payload
        if isinstance(path_or_payload, dict)
        else _load_npz(path_or_payload)
    )

    if payload["schema_version"] != AGGREGATE_SCHEMA:
        raise ValueError("aggregate schema mismatch")
    if payload["experiment_id"] != EXPERIMENT_ID:
        raise ValueError("aggregate experiment identity mismatch")
    for field, expected in (
        ("config_sha256", config["config_sha256"]),
        ("registry_sha256", config["registry_sha256"]),
        ("source_commit", config["source_commit"]),
        ("source_tree_sha256", config["source_tree_sha256"]),
        ("decoder_binary_sha256", config["decoder_binary"]["sha256"]),
        ("q_token", str(config["q_token"])),
        ("codes_per_m_json", canonical_json(
            {str(m): int(config["codes_per_m"][str(m)]) for m in config["m_values"]}
        )),
        ("trials_per_code_p_json", canonical_json(
            {str(m): int(config["trials_per_code_p"][str(m)]) for m in config["m_values"]}
        )),
    ):
        if payload[field] != expected:
            raise ValueError(f"aggregate identity mismatch for {field}")
    if arrays_sha256(payload, ARRAY_FIELDS) != payload["payload_sha256"]:
        raise ValueError("aggregate payload SHA256 mismatch")

    m_values, codes_per_m, trials_by_m, offsets, total_codes = panel_layout(config)
    n_p = len(config["p_tokens"])
    if payload["p_values"].shape != (n_p,) or not np.allclose(
        payload["p_values"], [float(token) for token in config["p_tokens"]],
    ):
        raise ValueError("aggregate p grid mismatch")
    if payload["m_values"].tolist() != m_values:
        raise ValueError("aggregate m panel mismatch")
    if payload["distance_values"].tolist() != list(DISTANCE_STRATA):
        raise ValueError("aggregate distance strata mismatch")

    if require_complete and payload["overall_status"] != "COMPLETE":
        raise ValueError("aggregate is not COMPLETE")
    if payload["unexpected_raw_errors_json"] != "[]":
        raise ValueError("aggregate carries unexpected raw errors")
    if require_complete and payload["replay_status"] != "PASS":
        raise ValueError("aggregate replay gate is not PASS")

    status = payload["code_status"]
    counts = payload["failure_counts"]
    if payload["code_m"].shape != (total_codes,):
        raise ValueError("aggregate code panel size mismatch")
    trials_per_code = np.empty(total_codes, dtype=np.int64)
    for m in m_values:
        trials_per_code[offsets[m]:offsets[m] + codes_per_m[m]] = trials_by_m[m]
    if require_complete and not np.all(status == "REPORTABLE"):
        raise ValueError("aggregate contains cells that are not REPORTABLE")
    expected_trials = np.broadcast_to(trials_per_code[:, None], counts.shape)
    if not np.all(
        payload["trial_counts"][status == "REPORTABLE"]
        == expected_trials[status == "REPORTABLE"]
    ):
        raise ValueError("aggregate trial counts disagree with the frozen allocation")
    if np.any(counts < 0) or np.any(counts > expected_trials):
        raise ValueError("aggregate failure counts are outside the legal range")

    with np.errstate(invalid="ignore"):
        expected_rates = counts / expected_trials.astype(float)
    reportable = status == "REPORTABLE"
    if not np.allclose(
        payload["code_rates"][reportable], expected_rates[reportable], atol=TOLERANCE,
    ):
        raise ValueError("aggregate per-code rates disagree with the stored counts")
    sample = np.flatnonzero(reportable.ravel())[:: max(1, reportable.sum() // 512)]
    flat_counts = counts.ravel()
    flat_low = payload["wilson_low"].ravel()
    flat_high = payload["wilson_high"].ravel()
    flat_trials = expected_trials.ravel()
    for index in sample:
        low, high = wilson_interval(int(flat_counts[index]), int(flat_trials[index]))
        if abs(low - flat_low[index]) > 1e-9 or abs(high - flat_high[index]) > 1e-9:
            raise ValueError("aggregate Wilson interval disagrees with the counts")

    failures_by_m = []
    trials_list = []
    for m_index, m in enumerate(m_values):
        trials = trials_by_m[m]
        trials_list.append(trials)
        code_slice = slice(offsets[m], offsets[m] + codes_per_m[m])
        block = counts[code_slice]
        failures_by_m.append(block)
        for p_index in range(n_p):
            if payload["m_status"][m_index, p_index] != "REPORTABLE":
                continue
            total = int(block[:, p_index].sum())
            trials_total = codes_per_m[m] * trials
            if int(payload["primary_failures"][m_index, p_index]) != total:
                raise ValueError("aggregate primary failure total mismatch")
            if int(payload["primary_trials"][m_index, p_index]) != trials_total:
                raise ValueError("aggregate primary trial total mismatch")
            mean = total / trials_total
            if abs(payload["primary_mean"][m_index, p_index] - mean) > TOLERANCE:
                raise ValueError("aggregate primary mean mismatch")
            rates = block[:, p_index] / float(trials)
            std = float(np.std(rates, ddof=1))
            if abs(payload["between_code_std"][m_index, p_index] - std) > 1e-9:
                raise ValueError("aggregate between-code standard deviation mismatch")
            if abs(
                payload["cluster_se"][m_index, p_index] - std / np.sqrt(codes_per_m[m])
            ) > 1e-9:
                raise ValueError("aggregate cluster standard error mismatch")

    distances = payload["classical_distance"]
    for m_index, m in enumerate(m_values):
        trials = trials_by_m[m]
        code_slice = slice(offsets[m], offsets[m] + codes_per_m[m])
        block_d = distances[code_slice]
        block_counts = counts[code_slice]
        block_status = status[code_slice]
        for d_index, distance in enumerate(DISTANCE_STRATA):
            member = block_d == distance
            if int(payload["strata_code_counts"][m_index, d_index]) != int(member.sum()):
                raise ValueError("aggregate strata code count mismatch")
            if not member.any():
                continue
            eligible = member[:, None] & (block_status == "REPORTABLE")
            failures = np.where(eligible, block_counts, 0).sum(axis=0)
            stratum_trials = eligible.sum(axis=0) * trials
            if not np.array_equal(payload["strata_failures"][m_index, d_index], failures):
                raise ValueError("aggregate strata failure counts mismatch")
            if not np.array_equal(payload["strata_trials"][m_index, d_index], stratum_trials):
                raise ValueError("aggregate strata trial counts mismatch")

    if require_complete:
        bootstrap = cluster_bootstrap(
            failures_by_m, trials_list, config, "final_m3_m8",
        )
        if not np.allclose(payload["delta38"], bootstrap["endpoint"], atol=TOLERANCE):
            raise ValueError("aggregate primary contrast is not reproducible")
        if abs(payload["bootstrap_half_width"] - bootstrap["half_width"]) > 1e-9:
            raise ValueError("aggregate bootstrap half-width is not reproducible")
        if not np.allclose(
            payload["delta38_band_low"], bootstrap["endpoint_low"], atol=1e-9,
        ) or not np.allclose(
            payload["delta38_band_high"], bootstrap["endpoint_high"], atol=1e-9,
        ):
            raise ValueError("aggregate simultaneous band is not reproducible")
        decision = classify_crossing(
            payload["p_values"], bootstrap["endpoint"],
            bootstrap["endpoint_low"], bootstrap["endpoint_high"],
        )
        if payload["terminal_status"] != decision["status"]:
            raise ValueError("aggregate terminal status is not reproducible")
        low, high = decision["bracket"]
        for field, expected in (
            ("crossing_bracket_low", low), ("crossing_bracket_high", high),
        ):
            stored = payload[field]
            if not (
                (np.isnan(stored) and np.isnan(expected))
                or abs(stored - expected) <= TOLERANCE
            ):
                raise ValueError(f"aggregate {field} is not reproducible")
        location = crossing_location(
            payload["p_values"], bootstrap["endpoint"],
            bootstrap["endpoint_replicates"], decision,
            confidence=float(config["bootstrap"]["confidence"]),
        )
        for field, expected in (
            ("p_cross", location["p_cross"]),
            ("p_cross_low", location["p_cross_low"]),
            ("p_cross_high", location["p_cross_high"]),
        ):
            stored = payload[field]
            if not (
                (np.isnan(stored) and np.isnan(expected))
                or abs(stored - expected) <= 1e-9
            ):
                raise ValueError(f"aggregate {field} is not reproducible")
        if payload["p_cross_reason"] != location["reason"]:
            raise ValueError("aggregate crossing location reason is not reproducible")
        if decision["status"] == CERTIFIED and not np.isfinite(payload["p_cross"]):
            raise ValueError("certified crossing without a finite location")
        bound = qtop_lower_bound(
            m_values, payload["p_values"], payload["primary_mean"],
        )
        if payload["qtop_lower_bound_json"] != canonical_json(bound):
            raise ValueError("aggregate q_top lower bound is not reproducible")

    return payload
