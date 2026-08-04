"""Exact identities for fixed-sector q=0 free-energy bridges.

This module deliberately contains only algebraic bridge primitives.  It does
not claim that a particular within-sector sampler has mixed, nor does it
enumerate the exponentially many logical sectors.  Those two questions must
be established separately before any bridge estimate can become an estimator.
"""

from __future__ import annotations

import math

import numpy as np


SECTOR_BRIDGE_VERSION = "exp102.q0_sector_bridge.v0"


class SectorBridgeError(ValueError):
    """Raised when a bridge input is not a binary fixed-sector object."""


def _binary_vector(value, name):
    array = np.asarray(value)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
        raise SectorBridgeError(f"{name} must be a one-dimensional binary vector")
    if np.any((array != 0) & (array != 1)):
        raise SectorBridgeError(f"{name} must be binary")
    return np.ascontiguousarray(array, dtype=np.uint8)


def logical_bridge_prefixes(move):
    """Return the deterministic one-bit-XOR path from zero to ``move``.

    For a logical kernel vector ``d`` this produces ``d_t`` such that the
    final map ``e -> e xor d`` is a bijection between two logical sectors.
    Intermediate vectors need not be in the kernel; their partition functions
    are well-defined fixed-sector bridge ensembles.
    """
    move = _binary_vector(move, "logical move")
    support = np.flatnonzero(move).astype(np.int32)
    prefixes = np.zeros((support.size + 1, move.size), dtype=np.uint8)
    for index, bit in enumerate(support):
        prefixes[index + 1] = prefixes[index]
        prefixes[index + 1, int(bit)] = 1
    return support, prefixes


def bridge_step_ratio(p, bit_one_probability):
    """Return ``Z_{t+1}/Z_t`` from an exact bridge-bit marginal.

    If the current bridge ensemble has energy ``|e xor d_t|`` and the next
    path bit is ``i``, then

    ``Z_{t+1}/Z_t = E[lambda**(1 - 2 x_i)]``

    where ``lambda=p/(1-p)`` and ``x=e xor d_t``.  The expression is exact;
    estimating the marginal is a separate sampling problem.
    """
    p = float(p)
    probability = float(bit_one_probability)
    if not math.isfinite(p) or not 0.0 < p < 0.5:
        raise SectorBridgeError("bridge p must lie in (0, 0.5)")
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise SectorBridgeError("bridge bit probability must lie in [0, 1]")
    odds = p / (1.0 - p)
    # Keep the two probability masses explicit.  Besides being clearer, this
    # has the same rounding order as the exact-oracle calculation.
    return (1.0 - probability) * odds + probability / odds


def reverse_bridge_step_ratio(p, next_bit_one_probability):
    """Estimate ``Z_(t+1)/Z_t`` using the next bridge ensemble.

    The next ensemble stores the bit after it has been toggled.  Its
    reciprocal identity is therefore ``Z_t/Z_(t+1) =
    E_(t+1)[lambda**(1-2*x_i)]``.  Keeping this helper separate prevents a
    tempting but incorrect reversal of the bit exponent.
    """
    return 1.0 / bridge_step_ratio(p, next_bit_one_probability)


def exact_fixed_sector_bridge(states, labels, sector_label, move, p):
    """Evaluate the bridge identity exactly on a fully enumerated hard coset.

    ``states`` must contain all states in a fixed hard coset and ``labels``
    their uint64 logical labels.  This oracle is intentionally small-code
    only; it is used to test the identity before a Monte Carlo bridge is used
    on a large HGP code.
    """
    states = np.asarray(states)
    if states.ndim != 2 or not np.issubdtype(states.dtype, np.integer):
        raise SectorBridgeError("exact bridge states must be a binary matrix")
    if np.any((states != 0) & (states != 1)):
        raise SectorBridgeError("exact bridge states must be binary")
    states = np.ascontiguousarray(states, dtype=np.uint8)
    labels = np.asarray(labels, dtype=np.uint64)
    if labels.shape != (states.shape[0],):
        raise SectorBridgeError("exact bridge labels have the wrong shape")
    move = _binary_vector(move, "logical move")
    if move.shape != (states.shape[1],):
        raise SectorBridgeError("logical move length does not match exact states")
    support, prefixes = logical_bridge_prefixes(move)
    selected = labels == np.uint64(sector_label)
    if not selected.any():
        raise SectorBridgeError("requested fixed sector is absent from exact states")
    sector_states = states[selected]
    odds = float(p) / (1.0 - float(p))
    masses = np.empty(prefixes.shape[0], dtype=np.float64)
    bit_probabilities = np.empty(support.size, dtype=np.float64)
    expected_ratios = np.empty(support.size, dtype=np.float64)
    reverse_bit_probabilities = np.empty(support.size, dtype=np.float64)
    reverse_expected_ratios = np.empty(support.size, dtype=np.float64)
    actual_ratios = np.empty(support.size, dtype=np.float64)
    for stage, prefix in enumerate(prefixes):
        shifted = sector_states ^ prefix
        weights = odds ** shifted.sum(axis=1)
        masses[stage] = weights.sum(dtype=np.float64)
        if stage < support.size:
            bit = int(support[stage])
            probability = float(
                np.dot(weights, shifted[:, bit]) / masses[stage]
            )
            bit_probabilities[stage] = probability
            expected_ratios[stage] = bridge_step_ratio(p, probability)
    actual_ratios[:] = masses[1:] / masses[:-1]
    for stage, bit in enumerate(support):
        shifted = sector_states ^ prefixes[stage + 1]
        weights = odds ** shifted.sum(axis=1)
        probability = float(np.dot(weights, shifted[:, int(bit)]) / weights.sum(dtype=np.float64))
        reverse_bit_probabilities[stage] = probability
        reverse_expected_ratios[stage] = reverse_bridge_step_ratio(p, probability)
    return {
        "support": support,
        "prefixes": prefixes,
        "partition_masses": masses,
        "bit_one_probabilities": bit_probabilities,
        "expected_step_ratios": expected_ratios,
        "reverse_bit_one_probabilities": reverse_bit_probabilities,
        "reverse_expected_step_ratios": reverse_expected_ratios,
        "actual_step_ratios": actual_ratios,
        "product_ratio": float(np.prod(expected_ratios, dtype=np.float64)),
        "endpoint_ratio": float(masses[-1] / masses[0]),
    }
