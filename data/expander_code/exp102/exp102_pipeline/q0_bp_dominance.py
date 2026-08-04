"""Conservative proposal-dominance witnesses for q=0 BP proposals.

This module does not estimate a posterior.  It supplies one-way lower bounds
on the largest posterior-to-proposal density ratio, which can rule out a
finite rejection-envelope route without claiming that a small witness set
certifies global coverage.
"""

from __future__ import annotations

from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, localcontext
from fractions import Fraction
import hashlib
import itertools
import math

import numpy as np

from .q0_global import reduce_logical_basis, state_label


BP_DOMINANCE_VERSION = "exp102.q0_bp_dominance.v0"
_DECIMAL_PRECISION = 240


class BpDominanceError(ValueError):
    """Raised when a dominance witness or proposal identity is invalid."""


def _require(condition, message):
    if not condition:
        raise BpDominanceError(message)


def _bits(value, *, name):
    array = np.asarray(value)
    _require(array.ndim == 1 and np.issubdtype(array.dtype, np.integer),
             f"{name} must be a one-dimensional binary vector")
    _require(not np.any((array != 0) & (array != 1)), f"{name} is not binary")
    return np.ascontiguousarray(array, dtype=np.uint8)


def _probability_fraction(p):
    # Configured decimal probabilities are part of the scientific contract,
    # whereas proposal Bernoulli parameters remain their frozen IEEE values.
    try:
        value = Fraction(str(p))
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise BpDominanceError("target probability is not a finite decimal") from exc
    _require(0 < value < Fraction(1, 2), "target probability must lie in (0,.5)")
    return value


def _fraction_decimal_lower(value):
    _require(value >= 0, "a probability factor cannot be negative")
    with localcontext() as context:
        context.prec = _DECIMAL_PRECISION
        context.rounding = ROUND_FLOOR
        return +(Decimal(value.numerator) / Decimal(value.denominator))


def _decimal_to_text(value):
    _require(isinstance(value, Decimal) and value.is_finite() and value >= 0,
             "dominance decimal is invalid")
    return format(value, "e")


def proposal_probability_upper(coordinate, proposal):
    """Return an outward-rounded upper bound on a frozen mixture density.

    The proposal components and weights are IEEE-754 values.  ``Decimal``
    imports each of those values exactly, then rounds every positive product
    and sum upward.  This makes the later posterior/proposal ratio a
    conservative lower bound rather than a favorable floating-point estimate.
    """
    coordinate = _bits(coordinate, name="coordinate")
    dimension = int(getattr(getattr(proposal, "coordinates", None), "dimension", -1))
    _require(coordinate.shape == (dimension,), "coordinate length changed")
    probabilities = np.asarray(getattr(proposal, "component_probabilities", None))
    weights = np.asarray(getattr(proposal, "component_weights", None))
    _require(probabilities.ndim == 2 and probabilities.shape[1] == dimension,
             "proposal component probabilities have the wrong shape")
    _require(weights.shape == (probabilities.shape[0],), "proposal component weights changed")
    _require(np.all(np.isfinite(probabilities)) and np.all((probabilities > 0.0)
                                                            & (probabilities < 1.0)),
             "proposal lost full support")
    _require(np.all(np.isfinite(weights)) and np.all(weights > 0.0),
             "proposal component weights are invalid")

    with localcontext() as context:
        context.prec = _DECIMAL_PRECISION
        context.rounding = ROUND_CEILING
        total = Decimal(0)
        for weight, component in zip(weights, probabilities, strict=True):
            value = Decimal.from_float(float(weight))
            for bit, probability in zip(coordinate, component, strict=True):
                probability_decimal = Decimal.from_float(float(probability))
                factor = probability_decimal if int(bit) else Decimal(1) - probability_decimal
                value *= factor
            total += value
        _require(total.is_finite() and total > 0, "proposal probability upper bound vanished")
        return +total


def posterior_to_proposal_lower(state, proposal, p):
    """Conservatively lower-bound ``pi(state) / q(state)``.

    For the unnormalized hard-coset target ``b**|e|`` we use the universal
    normalizer upper bound ``Z <= (1-p)**(-n)``.  The result is useful only as
    a witness: it never upper-bounds the unknown global dominance constant.
    """
    state = _bits(state, name="state")
    coordinates = getattr(proposal, "coordinates", None)
    _require(coordinates is not None and int(coordinates.num_qubits) == state.size,
             "proposal and state dimensions differ")
    coordinate = coordinates.coordinates_of_state(state)
    q_upper = proposal_probability_upper(coordinate, proposal)
    probability = _probability_fraction(p)
    odds = probability / (Fraction(1) - probability)
    numerator = (Fraction(1) - probability) ** state.size * odds ** int(state.sum())
    numerator_lower = _fraction_decimal_lower(numerator)
    with localcontext() as context:
        context.prec = _DECIMAL_PRECISION
        context.rounding = ROUND_FLOOR
        lower = +(numerator_lower / q_upper)
    _require(lower.is_finite() and lower >= 0, "dominance lower bound is invalid")
    return lower, q_upper


def canonical_rank_complete_logical_witnesses(model, frame, planted, *, candidate_orders):
    """Build result-independent low-energy logical witnesses from the canonical basis."""
    planted = _bits(planted, name="planted")
    _require(planted.shape == (int(model.num_qubits),), "planted state length changed")
    candidate_orders = tuple(int(order) for order in candidate_orders)
    _require(candidate_orders == (1, 2, 3), "logical witness orders changed")
    _require(1 <= int(model.k) <= 64 and int(frame.k) == int(model.k),
             "logical witness dimensions are unsupported")
    reduced = np.ascontiguousarray(reduce_logical_basis(model.logical_move_basis), dtype=np.uint8)
    _require(reduced.shape == (int(model.k), int(model.num_qubits)),
             "reduced logical basis changed shape")
    candidates = {}
    for order in candidate_orders:
        for combination in itertools.combinations(range(int(model.k)), order):
            move = np.bitwise_xor.reduce(reduced[list(combination)], axis=0)
            packed = np.packbits(move, bitorder="little").tobytes()
            if packed in candidates:
                continue
            signature = int(state_label(frame, move))
            _require(signature != 0, "logical witness move has zero signature")
            state = np.ascontiguousarray(planted ^ move, dtype=np.uint8)
            candidates[packed] = {
                "move": move,
                "state": state,
                "move_weight": int(move.sum()),
                "state_weight": int(state.sum()),
                "signature": signature,
                "packed": packed,
            }
    _require(candidates, "logical witness catalog is empty")
    key = lambda record: (
        record["state_weight"], record["move_weight"], record["signature"], record["packed"],
    )
    per_signature = {}
    for record in sorted(candidates.values(), key=key):
        per_signature.setdefault(record["signature"], record)

    selected = []
    pivots = {}
    for record in sorted(per_signature.values(), key=key):
        residue = int(record["signature"])
        while residue:
            pivot = residue.bit_length() - 1
            previous = pivots.get(pivot)
            if previous is None:
                pivots[pivot] = residue
                selected.append(record)
                break
            residue ^= previous
        if len(selected) == int(model.k):
            break
    _require(len(selected) == int(model.k), "logical witnesses do not span every signature")
    return tuple({
        "state": np.ascontiguousarray(record["state"], dtype=np.uint8),
        "signature": int(record["signature"]),
        "move_weight": int(record["move_weight"]),
        "state_weight": int(record["state_weight"]),
    } for record in selected)


def deterministic_witness_panel(model, frame, planted, proposals, *, candidate_orders):
    """Return planted, canonical-logical, and systematic-neighbor witnesses.

    ``proposals`` must retain its caller's deterministic insertion order.  The
    same physical state is deduplicated while keeping every predetermined
    origin, so the panel cannot become result-dependent through duplicate
    removal.
    """
    planted = _bits(planted, name="planted")
    _require(planted.shape == (int(model.num_qubits),), "planted state length changed")
    _require(hasattr(proposals, "items") and proposals, "witness proposals are empty")
    records = {}

    def add(state, origin):
        state = _bits(state, name="witness state")
        packed = np.packbits(state, bitorder="little").tobytes()
        current = records.get(packed)
        if current is None:
            records[packed] = {"state": state.copy(), "origins": [str(origin)]}
        else:
            current["origins"].append(str(origin))

    add(planted, "planted")
    for index, record in enumerate(canonical_rank_complete_logical_witnesses(
            model, frame, planted, candidate_orders=candidate_orders)):
        add(record["state"], "logical_rank_%d_signature_%016x" % (
            index, int(record["signature"]),
        ))
    for order, proposal in proposals.items():
        basis = np.asarray(proposal.coordinates.basis, dtype=np.uint8)
        _require(basis.shape == (proposal.coordinates.dimension, planted.size),
                 "systematic witness basis changed shape")
        for index, move in enumerate(basis):
            add(planted ^ move, "coordinate_neighbor_%s_%d" % (order, index))

    result = []
    for packed, record in records.items():
        state = np.ascontiguousarray(record["state"], dtype=np.uint8)
        result.append({
            "state": state,
            "origins": tuple(record["origins"]),
            "state_sha256": hashlib.sha256(packed).hexdigest(),
        })
    _require(result, "deterministic witness panel is empty")
    return tuple(result)


def dominance_record(state, proposal, p):
    """Create a JSON-safe record for one legal proposal-dominance witness."""
    state = _bits(state, name="state")
    coordinate = proposal.coordinates.coordinates_of_state(state)
    lower, q_upper = posterior_to_proposal_lower(state, proposal, p)
    return {
        "coordinate_weight": int(coordinate.sum()),
        "proposal_probability_upper": _decimal_to_text(q_upper),
        "posterior_to_proposal_lower": _decimal_to_text(lower),
        "state_weight": int(state.sum()),
    }
