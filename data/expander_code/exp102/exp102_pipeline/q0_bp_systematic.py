"""BP-guided systematic proposals for the q=0 hard-coset posterior.

This module supplies proposal distributions only.  It never treats loopy BP as
an exact posterior calculation: BP produces deterministic Bernoulli parameters
for a full-support importance proposal, whose density is evaluated exactly in
the systematic hard-coset coordinates.  A later importance calculation must
still diagnose its weights and independently control unobserved target tails.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np

from .exp101_bridge import load_exp101


BP_SYSTEMATIC_VERSION = "exp102.q0_bp_systematic.v0"


class BpSystematicError(ValueError):
    """Raised when a BP-guided systematic proposal is not replayable."""


def _require(condition, message):
    if not condition:
        raise BpSystematicError(message)


def _bits(value, *, ndim, name):
    array = np.asarray(value)
    _require(array.ndim == int(ndim), f"{name} has the wrong dimension")
    _require(np.issubdtype(array.dtype, np.bool_) or np.issubdtype(array.dtype, np.integer),
             f"{name} is not binary data")
    _require(not np.any((array != 0) & (array != 1)), f"{name} is not binary")
    return np.ascontiguousarray(array, dtype=np.uint8)


def _readonly(value, dtype=None):
    result = np.array(value, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _array_digest(version, arrays, scalars=()):
    digest = hashlib.sha256(str(version).encode("ascii") + b"\0")
    for scalar in scalars:
        digest.update(str(scalar).encode("ascii") + b"\0")
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(value.dtype.str.encode("ascii") + b"\0")
        digest.update(np.asarray(value.shape, dtype=">u8").tobytes())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _gf2_product(left, right):
    return (np.asarray(left, dtype=np.int64) @ np.asarray(right, dtype=np.int64) % 2).astype(
        np.uint8,
    )


def _logsumexp(values):
    values = np.asarray(values, dtype=np.float64)
    _require(values.ndim == 1 and values.size > 0 and np.all(np.isfinite(values)),
             "log-sum-exp inputs are invalid")
    maximum = float(values.max())
    return maximum + math.log(float(np.exp(values - maximum).sum(dtype=np.float64)))


def _bernoulli_log_probability(bits, probabilities):
    bits = _bits(bits, ndim=1, name="coordinate")
    probabilities = np.asarray(probabilities, dtype=np.float64)
    _require(probabilities.shape == bits.shape and np.all(np.isfinite(probabilities)),
             "Bernoulli probabilities have the wrong shape")
    _require(np.all(probabilities > 0.0) and np.all(probabilities < 1.0),
             "Bernoulli probabilities lost full support")
    return float(
        np.where(bits != 0, np.log(probabilities), np.log1p(-probabilities)).sum(
            dtype=np.float64,
        ),
    )


def _stable_probability_one(llr_zero_over_one):
    values = np.asarray(llr_zero_over_one, dtype=np.float64)
    result = np.empty_like(values)
    nonnegative = values >= 0.0
    positive_exp = np.exp(-values[nonnegative])
    result[nonnegative] = positive_exp / (1.0 + positive_exp)
    negative_exp = np.exp(values[~nonnegative])
    result[~nonnegative] = 1.0 / (1.0 + negative_exp)
    return result


@dataclass(frozen=True)
class XorBpDiagnostics:
    """Fixed-iteration sum-product summary; it is not a convergence claim."""

    marginal_probability_one: np.ndarray
    final_max_message_delta: float
    iterations: int
    damping: float
    llr_cap: float

    def __post_init__(self):
        object.__setattr__(self, "marginal_probability_one", _readonly(
            self.marginal_probability_one, np.float64,
        ))


def xor_sum_product_marginals(H_check, syndrome, p, *, iterations, damping, llr_cap):
    """Run a fixed number of deterministic sum-product iterations on XOR checks.

    The convention is ``LLR = log P(bit=0) / P(bit=1)``.  No adaptive early
    stopping is allowed because this routine is used to freeze a proposal.
    """
    H = _bits(H_check, ndim=2, name="H_check")
    syndrome = _bits(syndrome, ndim=1, name="syndrome")
    _require(syndrome.shape == (H.shape[0],), "syndrome length changed")
    p = float(p)
    iterations = int(iterations)
    damping = float(damping)
    llr_cap = float(llr_cap)
    _require(math.isfinite(p) and 0.0 < p < 0.5, "p must lie in (0,.5)")
    _require(iterations > 0, "BP iterations must be positive")
    _require(math.isfinite(damping) and 0.0 <= damping < 1.0,
             "BP damping must lie in [0,1)")
    _require(math.isfinite(llr_cap) and llr_cap > 0.0, "BP LLR cap must be positive")

    check_edges = [np.flatnonzero(H[row]).astype(np.int32) for row in range(H.shape[0])]
    _require(all(edges.size > 0 for edges in check_edges), "an empty parity check is unsupported")
    edge_check = np.repeat(
        np.arange(H.shape[0], dtype=np.int32),
        np.asarray([edges.size for edges in check_edges], dtype=np.int32),
    )
    edge_variable = np.concatenate(check_edges).astype(np.int32, copy=False)
    edge_offsets = np.zeros(H.shape[0] + 1, dtype=np.int64)
    np.cumsum(np.asarray([edges.size for edges in check_edges], dtype=np.int64), out=edge_offsets[1:])
    _require(edge_variable.size > 0, "H_check has no Tanner-graph edges")

    prior = math.log((1.0 - p) / p)
    variable_to_check = np.full(edge_variable.size, prior, dtype=np.float64)
    check_to_variable = np.zeros(edge_variable.size, dtype=np.float64)
    final_delta = math.inf
    tanh_cap = math.tanh(llr_cap / 2.0)
    atanh_limit = math.nextafter(1.0, 0.0)

    for _iteration in range(iterations):
        previous_check = check_to_variable.copy()
        candidates = np.empty_like(check_to_variable)
        for check in range(H.shape[0]):
            start, stop = int(edge_offsets[check]), int(edge_offsets[check + 1])
            values = np.tanh(variable_to_check[start:stop] / 2.0)
            values = np.clip(values, -tanh_cap, tanh_cap)
            sign = -1.0 if int(syndrome[check]) else 1.0
            for local in range(stop - start):
                product = sign
                for other in range(stop - start):
                    if other != local:
                        product *= float(values[other])
                product = min(atanh_limit, max(-atanh_limit, product))
                candidates[start + local] = min(
                    llr_cap, max(-llr_cap, 2.0 * math.atanh(product)),
                )
        check_to_variable = damping * check_to_variable + (1.0 - damping) * candidates
        np.clip(check_to_variable, -llr_cap, llr_cap, out=check_to_variable)

        incoming = np.zeros(H.shape[1], dtype=np.float64)
        np.add.at(incoming, edge_variable, check_to_variable)
        candidate_variable = prior + incoming[edge_variable] - check_to_variable
        np.clip(candidate_variable, -llr_cap, llr_cap, out=candidate_variable)
        final_delta = max(
            float(np.max(np.abs(check_to_variable - previous_check))),
            float(np.max(np.abs(candidate_variable - variable_to_check))),
        )
        variable_to_check = damping * variable_to_check + (1.0 - damping) * candidate_variable
        np.clip(variable_to_check, -llr_cap, llr_cap, out=variable_to_check)

    incoming = np.zeros(H.shape[1], dtype=np.float64)
    np.add.at(incoming, edge_variable, check_to_variable)
    posterior_llr = np.clip(prior + incoming, -llr_cap, llr_cap)
    probabilities = _stable_probability_one(posterior_llr)
    _require(np.all(np.isfinite(probabilities))
             and np.all(probabilities > 0.0) and np.all(probabilities < 1.0),
             "BP marginals are not finite probabilities")
    return XorBpDiagnostics(
        marginal_probability_one=probabilities,
        final_max_message_delta=float(final_delta),
        iterations=iterations,
        damping=damping,
        llr_cap=llr_cap,
    )


@dataclass(frozen=True)
class SystematicHardCosetCoordinates:
    """A hard-coset bijection whose coordinates are selected physical bits."""

    H_check: np.ndarray
    syndrome: np.ndarray
    reference_anchor: np.ndarray
    basis: np.ndarray
    pivot_columns: np.ndarray
    free_columns: np.ndarray
    packed_reference: np.ndarray
    packed_basis: np.ndarray
    coordinate_sha256: str

    def __post_init__(self):
        for name, dtype in (
            ("H_check", np.uint8), ("syndrome", np.uint8),
            ("reference_anchor", np.uint8), ("basis", np.uint8),
            ("pivot_columns", np.int32), ("free_columns", np.int32),
            ("packed_reference", np.uint8), ("packed_basis", np.uint8),
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name), dtype))

    @property
    def dimension(self):
        return int(self.free_columns.size)

    @property
    def num_qubits(self):
        return int(self.reference_anchor.size)

    def state_packed_from_coordinates(self, coordinate):
        coordinate = _bits(coordinate, ndim=1, name="coordinate")
        _require(coordinate.shape == (self.dimension,), "coordinate length changed")
        selected = np.flatnonzero(coordinate)
        packed = self.packed_reference.copy()
        if selected.size:
            packed ^= np.bitwise_xor.reduce(self.packed_basis[selected], axis=0)
        return packed

    def state_from_coordinates(self, coordinate):
        return np.unpackbits(
            self.state_packed_from_coordinates(coordinate), bitorder="little",
            count=self.num_qubits,
        ).astype(np.uint8, copy=False)

    def coordinates_of_state(self, state):
        state = _bits(state, ndim=1, name="state")
        _require(state.shape == (self.num_qubits,), "state length changed")
        coordinate = np.ascontiguousarray(state[self.free_columns], dtype=np.uint8)
        _require(np.array_equal(self.state_from_coordinates(coordinate), state),
                 "state is outside this systematic hard coset")
        return coordinate


def build_systematic_hard_coset_coordinates(H_check, syndrome, *, column_order):
    """Build a deterministic physical-information-set parameterization.

    ``column_order`` affects only the predeclared information set.  It may not
    be selected from sampler output or a physical result.
    """
    load_exp101()
    from exp101_certified_src.gf2 import gf2_inverse, gf2_matmul, gf2_rank, gf2_row_echelon

    H = _bits(H_check, ndim=2, name="H_check")
    syndrome = _bits(syndrome, ndim=1, name="syndrome")
    _require(syndrome.shape == (H.shape[0],), "syndrome length changed")
    column_order = np.asarray(column_order, dtype=np.int32)
    _require(column_order.shape == (H.shape[1],)
             and np.array_equal(np.sort(column_order), np.arange(H.shape[1], dtype=np.int32)),
             "column order is not a permutation")
    _require(int(gf2_rank(H)) == H.shape[0], "H_check must have full row rank")

    _rref, permuted_pivots = gf2_row_echelon(H[:, column_order])
    _require(len(permuted_pivots) == H.shape[0], "column order lost a pivot")
    pivot_columns = column_order[np.asarray(permuted_pivots, dtype=np.int32)]
    pivot_set = set(int(value) for value in pivot_columns)
    free_columns = np.asarray(
        [int(value) for value in column_order if int(value) not in pivot_set], dtype=np.int32,
    )
    pivot_block = H[:, pivot_columns]
    pivot_inverse = gf2_inverse(pivot_block)
    anchor = gf2_matmul(pivot_inverse, syndrome[:, None])[:, 0]
    reference = np.zeros(H.shape[1], dtype=np.uint8)
    reference[pivot_columns] = anchor
    free_block = H[:, free_columns]
    pivot_values = gf2_matmul(pivot_inverse, free_block)
    basis = np.zeros((free_columns.size, H.shape[1]), dtype=np.uint8)
    basis[np.arange(free_columns.size), free_columns] = 1
    basis[:, pivot_columns] = pivot_values.T
    _require(not _gf2_product(H, basis.T).any()
             and np.array_equal(_gf2_product(H, reference[:, None])[:, 0], syndrome),
             "systematic construction escaped the hard coset")
    digest = _array_digest(
        "exp102.q0_bp_systematic.coordinates.v0",
        (np.packbits(H, axis=1, bitorder="little"), np.packbits(syndrome, bitorder="little"),
         pivot_columns.astype(">i4"), free_columns.astype(">i4"),
         np.packbits(reference, bitorder="little"), np.packbits(basis, axis=1, bitorder="little")),
    )
    return SystematicHardCosetCoordinates(
        H_check=H,
        syndrome=syndrome,
        reference_anchor=reference,
        basis=basis,
        pivot_columns=pivot_columns,
        free_columns=free_columns,
        packed_reference=np.packbits(reference, bitorder="little"),
        packed_basis=np.packbits(basis, axis=1, bitorder="little"),
        coordinate_sha256=digest,
    )


@dataclass(frozen=True)
class BpSystematicProposal:
    """A full-support mixture on an exact systematic hard-coset bijection."""

    coordinates: SystematicHardCosetCoordinates
    component_probabilities: np.ndarray
    component_weights: np.ndarray
    bp_diagnostics: XorBpDiagnostics
    proposal_sha256: str

    def __post_init__(self):
        object.__setattr__(self, "component_probabilities", _readonly(
            self.component_probabilities, np.float64,
        ))
        object.__setattr__(self, "component_weights", _readonly(
            self.component_weights, np.float64,
        ))

    @property
    def num_components(self):
        return int(self.component_weights.size)

    def log_probability_coordinates(self, coordinate):
        coordinate = _bits(coordinate, ndim=1, name="coordinate")
        _require(coordinate.shape == (self.coordinates.dimension,), "coordinate length changed")
        terms = np.asarray([
            math.log(float(weight)) + _bernoulli_log_probability(coordinate, probabilities)
            for weight, probabilities in zip(
                self.component_weights, self.component_probabilities, strict=True,
            )
        ], dtype=np.float64)
        result = _logsumexp(terms)
        _require(math.isfinite(result), "proposal log density is non-finite")
        return result

    def log_probability_state(self, state):
        return self.log_probability_coordinates(self.coordinates.coordinates_of_state(state))

    @staticmethod
    def _categorical(rng, probabilities):
        threshold = float(rng.random())
        cumulative = 0.0
        for index, probability in enumerate(probabilities):
            cumulative += float(probability)
            if threshold < cumulative:
                return index
        return len(probabilities) - 1

    def sample(self, rng):
        component = self._categorical(rng, self.component_weights)
        probabilities = self.component_probabilities[component]
        coordinate = np.zeros(self.coordinates.dimension, dtype=np.uint8)
        for bit, probability in enumerate(probabilities):
            if float(rng.random()) < float(probability):
                coordinate[bit] = 1
        state = self.coordinates.state_from_coordinates(coordinate)
        return {
            "state": state,
            "coordinate": coordinate,
            "log_q": self.log_probability_coordinates(coordinate),
            "component_index": int(component),
        }


def build_bp_systematic_proposal(model, syndrome, p, *, column_order,
                                 bp_iterations, bp_damping, bp_llr_cap,
                                 min_probability, component_weights):
    """Freeze BP, prior, and uniform components into one exact proposal."""
    H = _bits(model.H_check, ndim=2, name="model.H_check")
    _require(H.shape[1] == int(model.num_qubits), "model qubit count changed")
    syndrome = _bits(syndrome, ndim=1, name="syndrome")
    _require(syndrome.shape == (H.shape[0],), "syndrome length changed")
    p = float(p)
    min_probability = float(min_probability)
    _require(math.isfinite(p) and 0.0 < p < 0.5, "p must lie in (0,.5)")
    _require(math.isfinite(min_probability) and 0.0 < min_probability < 0.5,
             "minimum probability must lie in (0,.5)")
    diagnostics = xor_sum_product_marginals(
        H, syndrome, p, iterations=bp_iterations, damping=bp_damping,
        llr_cap=bp_llr_cap,
    )
    coordinates = build_systematic_hard_coset_coordinates(
        H, syndrome, column_order=column_order,
    )
    bp_probability = np.clip(
        diagnostics.marginal_probability_one[coordinates.free_columns],
        min_probability, 1.0 - min_probability,
    )
    weights = np.asarray(component_weights, dtype=np.float64)
    _require(weights.shape == (3,) and np.all(np.isfinite(weights))
             and np.all(weights > 0.0) and abs(float(weights.sum()) - 1.0) <= 1e-14,
             "component weights must be three positive normalized values")
    weights = weights / float(weights.sum())
    weights[-1] = 1.0 - float(weights[:-1].sum())
    probabilities = np.vstack((
        bp_probability,
        np.full(coordinates.dimension, p, dtype=np.float64),
        np.full(coordinates.dimension, 0.5, dtype=np.float64),
    ))
    digest = _array_digest(
        BP_SYSTEMATIC_VERSION,
        (np.packbits(H, axis=1, bitorder="little"), np.packbits(syndrome, bitorder="little"),
         coordinates.pivot_columns.astype(">i4"), coordinates.free_columns.astype(">i4"),
         probabilities.astype(">f8"), weights.astype(">f8"),
         diagnostics.marginal_probability_one.astype(">f8")),
        (coordinates.coordinate_sha256, int(bp_iterations), format(float(bp_damping), ".17g"),
         format(float(bp_llr_cap), ".17g"), format(min_probability, ".17g")),
    )
    return BpSystematicProposal(
        coordinates=coordinates,
        component_probabilities=probabilities,
        component_weights=weights,
        bp_diagnostics=diagnostics,
        proposal_sha256=digest,
    )
