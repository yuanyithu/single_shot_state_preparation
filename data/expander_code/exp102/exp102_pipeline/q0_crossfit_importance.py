"""Cross-fitted collision diagnostics for iid hard-coset importance draws.

The routines deliberately operate on independently drawn proposal samples,
not a Markov-chain trace.  They expose the ratio estimate and its block
jackknife uncertainty without clipping either purity or ``q_top`` into a
physical interval.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


CROSSFIT_IMPORTANCE_VERSION = "exp102.q0_crossfit_importance.v0"


class CrossfitImportanceError(ValueError):
    """Raised when iid importance samples cannot support a cross-fit."""


@dataclass(frozen=True)
class CrossfitCollisionResult:
    """Collision-ratio diagnostic from mutually independent equal-size blocks."""

    purity: float
    purity_jackknife_se: float
    q_top: float
    q_top_jackknife_se: float
    block_effective_sample_sizes: np.ndarray
    leave_one_purity: np.ndarray
    leave_one_q_top: np.ndarray
    label_count: int
    top_normalized_sector_mass: float

    def as_dict(self):
        return {
            "estimator": "ordered_cross_block_importance_collision_ratio.v0",
            "purity": self.purity,
            "purity_jackknife_se": self.purity_jackknife_se,
            "q_top": self.q_top,
            "q_top_jackknife_se": self.q_top_jackknife_se,
            "block_effective_sample_sizes": self.block_effective_sample_sizes.tolist(),
            "leave_one_purity": self.leave_one_purity.tolist(),
            "leave_one_q_top": self.leave_one_q_top.tolist(),
            "label_count": self.label_count,
            "top_normalized_sector_mass": self.top_normalized_sector_mass,
        }


@dataclass(frozen=True)
class CrossfitDistributionDistanceResult:
    """Cross-fitted L2 distance between two independently sampled sector laws."""

    d2_norm: float
    d2_norm_jackknife_se: float
    overlap: float
    leave_one_d2_norm: np.ndarray

    def as_dict(self):
        return {
            "estimator": "cross_block_importance_distribution_d2.v0",
            "d2_norm": self.d2_norm,
            "d2_norm_jackknife_se": self.d2_norm_jackknife_se,
            "overlap": self.overlap,
            "leave_one_d2_norm": self.leave_one_d2_norm.tolist(),
        }


def _require(condition, message):
    if not condition:
        raise CrossfitImportanceError(message)


def _q_top_from_purity(purity, logical_dimension):
    uniform = math.ldexp(1.0, -int(logical_dimension))
    return (float(purity) - uniform) / (1.0 - uniform)


def _ratio_from_block_masses(masses):
    masses = np.asarray(masses, dtype=np.float64)
    block_totals = masses.sum(axis=1, dtype=np.float64)
    gram = masses @ masses.T
    numerator = float(gram.sum(dtype=np.float64) - np.trace(gram))
    denominator = float(
        block_totals.sum(dtype=np.float64) ** 2
        - np.dot(block_totals, block_totals)
    )
    _require(math.isfinite(numerator) and math.isfinite(denominator) and denominator > 0.0,
             "cross-fit collision ratio has a nonpositive denominator")
    return numerator / denominator


def _block_masses(labels, log_importance_weights, *, block_count, all_labels=None):
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.size == 0 or labels.dtype.kind != "u":
        raise CrossfitImportanceError("importance labels must be a nonempty unsigned integer vector")
    labels = np.ascontiguousarray(labels, dtype=np.uint64)
    log_weights = np.asarray(log_importance_weights, dtype=np.float64)
    _require(log_weights.shape == labels.shape and np.all(np.isfinite(log_weights)),
             "importance log weights must be finite and match labels")
    _require(labels.size % block_count == 0,
             "importance samples must divide evenly into frozen blocks")
    if all_labels is None:
        all_labels = np.unique(labels)
    else:
        all_labels = np.asarray(all_labels, dtype=np.uint64)
        _require(all_labels.ndim == 1 and all_labels.size > 0
                 and np.unique(all_labels).size == all_labels.size,
                 "cross-fit label support is invalid")
    inverse = np.searchsorted(all_labels, labels)
    _require(np.all(inverse < all_labels.size)
             and np.array_equal(all_labels[inverse], labels),
             "cross-fit label support excludes a sample")
    relative_weights = np.exp(log_weights - float(log_weights.max()))
    _require(np.all(np.isfinite(relative_weights)) and np.any(relative_weights > 0.0),
             "importance weights vanished after stable exponentiation")
    block_size = labels.size // block_count
    masses = np.empty((block_count, all_labels.size), dtype=np.float64)
    squared_sums = np.empty(block_count, dtype=np.float64)
    for block in range(block_count):
        start = block * block_size
        stop = start + block_size
        weights = relative_weights[start:stop]
        masses[block] = np.bincount(
            inverse[start:stop], weights=weights, minlength=all_labels.size,
        )
        squared_sums[block] = float(np.dot(weights, weights))
    block_totals = masses.sum(axis=1, dtype=np.float64)
    _require(np.all(block_totals > 0.0) and np.all(squared_sums > 0.0),
             "one importance block has zero total mass")
    return all_labels, masses, block_totals, squared_sums


def _cross_overlap_from_block_masses(left, right):
    left_totals = left.sum(axis=1, dtype=np.float64)
    right_totals = right.sum(axis=1, dtype=np.float64)
    numerator = float((left @ right.T).sum(dtype=np.float64))
    denominator = float(left_totals.sum(dtype=np.float64) * right_totals.sum(dtype=np.float64))
    _require(math.isfinite(numerator) and math.isfinite(denominator) and denominator > 0.0,
             "cross-fit overlap has a nonpositive denominator")
    return numerator / denominator


def crossfit_collision_ratio(labels, log_importance_weights, *, block_count,
                             logical_dimension):
    """Estimate sector purity using products from distinct iid sample blocks.

    Each block estimates every unnormalized sector mass.  Cross-products from
    different blocks remove the same-sample collision bias in the numerator.
    The final normalization is necessarily a ratio estimate, so its
    finite-sample behavior is reported through a delete-one-block jackknife.
    """
    if isinstance(block_count, bool) or int(block_count) < 3:
        raise CrossfitImportanceError("cross-fit needs at least three blocks")
    block_count = int(block_count)
    if isinstance(logical_dimension, bool) or not 1 <= int(logical_dimension) <= 64:
        raise CrossfitImportanceError("logical dimension must lie in [1, 64]")
    logical_dimension = int(logical_dimension)
    unique_labels, masses, block_totals, squared_sums = _block_masses(
        labels, log_importance_weights, block_count=block_count,
    )
    block_ess = block_totals ** 2 / squared_sums
    purity = _ratio_from_block_masses(masses)
    q_top = _q_top_from_purity(purity, logical_dimension)

    leave_one_purity = np.empty(block_count, dtype=np.float64)
    leave_one_q_top = np.empty(block_count, dtype=np.float64)
    for block in range(block_count):
        leave_one_purity[block] = _ratio_from_block_masses(
            np.delete(masses, block, axis=0),
        )
        leave_one_q_top[block] = _q_top_from_purity(
            leave_one_purity[block], logical_dimension,
        )
    purity_jackknife_se = math.sqrt(
        (block_count - 1.0) / block_count
        * float(np.square(leave_one_purity - leave_one_purity.mean()).sum(dtype=np.float64))
    )
    q_top_jackknife_se = math.sqrt(
        (block_count - 1.0) / block_count
        * float(np.square(leave_one_q_top - leave_one_q_top.mean()).sum(dtype=np.float64))
    )
    normalized_masses = masses.sum(axis=0, dtype=np.float64)
    normalized_masses /= float(normalized_masses.sum(dtype=np.float64))
    return CrossfitCollisionResult(
        purity=float(purity),
        purity_jackknife_se=float(purity_jackknife_se),
        q_top=float(q_top),
        q_top_jackknife_se=float(q_top_jackknife_se),
        block_effective_sample_sizes=np.ascontiguousarray(block_ess, dtype=np.float64),
        leave_one_purity=np.ascontiguousarray(leave_one_purity, dtype=np.float64),
        leave_one_q_top=np.ascontiguousarray(leave_one_q_top, dtype=np.float64),
        label_count=int(unique_labels.size),
        top_normalized_sector_mass=float(normalized_masses.max()),
    )


def crossfit_distribution_distance(left_labels, left_log_importance_weights,
                                   right_labels, right_log_importance_weights, *,
                                   block_count, logical_dimension):
    """Estimate normalized sector-distribution L2 distance without clipping.

    The two inputs must be independently generated and organized into the same
    frozen number of blocks.  Cross-family products never use a same-draw
    collision, while leave-one-*paired-block* jackknife retains the predeclared
    block structure.
    """
    if isinstance(block_count, bool) or int(block_count) < 3:
        raise CrossfitImportanceError("cross-fit distance needs at least three blocks")
    block_count = int(block_count)
    if isinstance(logical_dimension, bool) or not 1 <= int(logical_dimension) <= 64:
        raise CrossfitImportanceError("logical dimension must lie in [1, 64]")
    logical_dimension = int(logical_dimension)
    left_values = np.asarray(left_labels)
    right_values = np.asarray(right_labels)
    if (left_values.ndim != 1 or right_values.ndim != 1
            or left_values.dtype.kind != "u" or right_values.dtype.kind != "u"):
        raise CrossfitImportanceError("distance labels must be unsigned integer vectors")
    support = np.unique(np.concatenate((
        np.asarray(left_values, dtype=np.uint64), np.asarray(right_values, dtype=np.uint64),
    )))
    _, left, _, _ = _block_masses(
        left_values, left_log_importance_weights, block_count=block_count,
        all_labels=support,
    )
    _, right, _, _ = _block_masses(
        right_values, right_log_importance_weights, block_count=block_count,
        all_labels=support,
    )
    uniform = math.ldexp(1.0, -logical_dimension)
    scale = 1.0 - uniform

    def estimate(left_masses, right_masses):
        left_purity = _ratio_from_block_masses(left_masses)
        right_purity = _ratio_from_block_masses(right_masses)
        overlap = _cross_overlap_from_block_masses(left_masses, right_masses)
        return (left_purity + right_purity - 2.0 * overlap) / scale, overlap

    d2_norm, overlap = estimate(left, right)
    leave_one = np.empty(block_count, dtype=np.float64)
    for block in range(block_count):
        leave_one[block], _ = estimate(
            np.delete(left, block, axis=0), np.delete(right, block, axis=0),
        )
    jackknife_se = math.sqrt(
        (block_count - 1.0) / block_count
        * float(np.square(leave_one - leave_one.mean()).sum(dtype=np.float64))
    )
    return CrossfitDistributionDistanceResult(
        d2_norm=float(d2_norm),
        d2_norm_jackknife_se=float(jackknife_se),
        overlap=float(overlap),
        leave_one_d2_norm=np.ascontiguousarray(leave_one, dtype=np.float64),
    )
