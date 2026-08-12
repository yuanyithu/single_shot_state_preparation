"""Certification of the ensemble crossing and its location.

Two things differ from exp103 by design.

The band is a cluster bootstrap over codes, and it is simultaneous only across
the grid points of the primary contrast. exp103 took a max-absolute deviation
simultaneously over six curves, thirteen points and five adjacent contrasts,
which is what produced a half-width of 0.2601 and made certification impossible
regardless of the point estimates.

Codes are resampled within each m separately, so unequal panel sizes are handled
without comment. exp106 uses unequal panels deliberately: a code at m=8 costs
about seventy times a code at m=3, so equal panels would spend most of the budget
on the smaller of the two variance terms in the primary contrast.

The bracket does not require adjacent grid points. Requiring them forces the
experiment to certify a contrast that vanishes at the crossing, which no finite
sample can do. exp106 instead certifies a negative point and a later positive
point, reports the bracket between them, and separately reports the interpolated
crossing location with a percentile bootstrap interval.
"""

import hashlib

import numpy as np

from .config import ensure_config


CERTIFIED = "EXP106_CERTIFIED_CROSSING"
NOT_CERTIFIED = "EXP106_NO_CERTIFIED_CROSSING"


def wilson_interval(failures, trials, z=1.959963984540054):
    failures = float(failures)
    trials = float(trials)
    if trials <= 0:
        return float("nan"), float("nan")
    rate = failures / trials
    denominator = 1.0 + z * z / trials
    centre = (rate + z * z / (2.0 * trials)) / denominator
    spread = z * np.sqrt(rate * (1.0 - rate) / trials + z * z / (4.0 * trials * trials))
    spread /= denominator
    return max(0.0, centre - spread), min(1.0, centre + spread)


def _resampled_means(failures, trials_per_code, rng, replicates, chunk=500):
    """Cluster bootstrap: resample codes, keep each code's whole p curve.

    Returns an array of shape (replicates, n_p). Codes are the sampling unit, so
    the interval covers ensemble variation rather than only shot noise; with a
    few trials per code-p the two are combined automatically because a resampled
    code carries its own trial outcomes.
    """
    n_codes, n_p = failures.shape
    out = np.empty((replicates, n_p), dtype=np.float64)
    done = 0
    while done < replicates:
        size = min(chunk, replicates - done)
        index = rng.integers(0, n_codes, size=(size, n_codes))
        for p_index in range(n_p):
            column = failures[:, p_index]
            out[done:done + size, p_index] = column[index].sum(axis=1)
        done += size
    return out / float(n_codes * trials_per_code)


def cluster_bootstrap(failures_by_m, trials_per_code_by_m, config, label):
    """Bootstrap every published quantity from one shared set of replicates.

    `trials_per_code_by_m` is a sequence parallel to `failures_by_m`, because
    the frozen allocation rule can give different m different trial counts.
    """
    config = ensure_config(config)
    spec = config["bootstrap"]
    replicates = int(spec["replicates"])
    confidence = float(spec["confidence"])
    seed_payload = ":".join([
        config["master_seed_hex"], config["namespaces"]["bootstrap"],
        config["registry_sha256"], str(label),
    ])
    seed = int.from_bytes(
        hashlib.sha256(seed_payload.encode("ascii")).digest()[:8], "big",
    ) & ((1 << 63) - 1)
    rng = np.random.Generator(np.random.PCG64(seed))

    n_m = len(failures_by_m)
    if len(trials_per_code_by_m) != n_m:
        raise ValueError("trials per code must be given for every m")
    n_p = failures_by_m[0].shape[1]
    curves = np.empty((n_m, replicates, n_p), dtype=np.float64)
    point = np.empty((n_m, n_p), dtype=np.float64)
    for m_index, failures in enumerate(failures_by_m):
        trials = int(trials_per_code_by_m[m_index])
        curves[m_index] = _resampled_means(failures, trials, rng, replicates)
        point[m_index] = failures.sum(axis=0) / float(failures.shape[0] * trials)

    tail = (1.0 - confidence) / 2.0
    point_low = np.percentile(curves, 100.0 * tail, axis=1)
    point_high = np.percentile(curves, 100.0 * (1.0 - tail), axis=1)

    endpoint = point[-1] - point[0]
    endpoint_replicates = curves[-1] - curves[0]
    deviation = np.abs(endpoint_replicates - endpoint[None, :]).max(axis=1)
    half_width = float(np.percentile(deviation, 100.0 * confidence))

    adjacent = point[1:] - point[:-1]
    adjacent_replicates = curves[1:] - curves[:-1]
    adjacent_low = np.percentile(adjacent_replicates, 100.0 * tail, axis=1)
    adjacent_high = np.percentile(adjacent_replicates, 100.0 * (1.0 - tail), axis=1)

    return {
        "point": point,
        "point_low": point_low,
        "point_high": point_high,
        "endpoint": endpoint,
        "endpoint_replicates": endpoint_replicates,
        "endpoint_low": endpoint - half_width,
        "endpoint_high": endpoint + half_width,
        "half_width": half_width,
        "adjacent": adjacent,
        "adjacent_low": adjacent_low,
        "adjacent_high": adjacent_high,
    }


def _first_sign_change(p_values, delta, low_index, high_index):
    """First negative-to-positive crossing inside the certified bracket."""
    for index in range(low_index, high_index):
        left = delta[index]
        right = delta[index + 1]
        if left <= 0.0 < right:
            span = right - left
            if span <= 0.0:
                return float("nan")
            step = p_values[index + 1] - p_values[index]
            return float(p_values[index] + (-left) * step / span)
    return float("nan")


def classify_crossing(p_values, delta, band_low, band_high):
    """Terminal decision on the primary contrast alone.

    The contract text and this function say the same thing: the status depends
    only on Delta38 and its simultaneous band. Adjacent contrasts are diagnostic
    and are never consulted here.
    """
    p_values = np.asarray(p_values, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    band_low = np.asarray(band_low, dtype=np.float64)
    band_high = np.asarray(band_high, dtype=np.float64)
    finite = np.isfinite(delta) & np.isfinite(band_low) & np.isfinite(band_high)
    negative = np.flatnonzero(finite & (band_high < 0.0))
    positive = np.flatnonzero(finite & (band_low > 0.0))
    if negative.size and positive.size:
        candidates = [
            b for b in positive if negative.min() < b
        ]
        if candidates:
            high_index = int(min(candidates))
            low_index = int(max(a for a in negative if a < high_index))
            return {
                "status": CERTIFIED,
                "bracket": (float(p_values[low_index]), float(p_values[high_index])),
                "bracket_indices": (low_index, high_index),
                "certified_negative_p": [float(p_values[i]) for i in negative],
                "certified_positive_p": [float(p_values[i]) for i in positive],
            }
    return {
        "status": NOT_CERTIFIED,
        "bracket": (float("nan"), float("nan")),
        "bracket_indices": (-1, -1),
        "certified_negative_p": [float(p_values[i]) for i in negative],
        "certified_positive_p": [float(p_values[i]) for i in positive],
    }


def crossing_location(p_values, delta, endpoint_replicates, decision, confidence=0.95,
                      minimum_defined_fraction=0.95):
    """Interpolated crossing location with a percentile bootstrap interval.

    The sub-grid is frozen by the certified bracket before the replicates are
    consulted, and a replicate that shows no sign change inside that sub-grid is
    counted as undefined rather than dropped silently. If too many replicates are
    undefined the location is reported as NaN with the reason attached.
    """
    if decision["status"] != CERTIFIED:
        return {
            "p_cross": float("nan"),
            "p_cross_low": float("nan"),
            "p_cross_high": float("nan"),
            "defined_fraction": float("nan"),
            "reason": "no_certified_bracket",
        }
    p_values = np.asarray(p_values, dtype=np.float64)
    low_index, high_index = decision["bracket_indices"]
    point = _first_sign_change(p_values, np.asarray(delta, dtype=np.float64), low_index, high_index)
    values = []
    for replicate in np.asarray(endpoint_replicates, dtype=np.float64):
        located = _first_sign_change(p_values, replicate, low_index, high_index)
        if np.isfinite(located):
            values.append(located)
    defined = len(values) / float(len(endpoint_replicates))
    if not np.isfinite(point):
        return {
            "p_cross": float("nan"), "p_cross_low": float("nan"),
            "p_cross_high": float("nan"), "defined_fraction": defined,
            "reason": "no_sign_change_in_bracket",
        }
    if defined < minimum_defined_fraction:
        return {
            "p_cross": point, "p_cross_low": float("nan"),
            "p_cross_high": float("nan"), "defined_fraction": defined,
            "reason": "insufficient_defined_replicates",
        }
    tail = (1.0 - confidence) / 2.0
    return {
        "p_cross": point,
        "p_cross_low": float(np.percentile(values, 100.0 * tail)),
        "p_cross_high": float(np.percentile(values, 100.0 * (1.0 - tail))),
        "defined_fraction": defined,
        "reason": "",
    }
