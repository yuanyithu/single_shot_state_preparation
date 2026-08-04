"""Independent hard-coset multiple-importance sampling primitives.

Unlike a Markov-chain trace, every stored draw is generated directly from a
frozen proposal family.  Blocks use disjoint PortablePrng streams and a fixed
number of draws from every family, so cross-block products remain independent
even though each block uses deterministic stratification across proposals.

This module supplies an estimator input and a raw replay checker; it does not
claim that finite iid samples certify unobserved target tails.  That distinction
is deliberately left to the discovery contract and its analysis gates.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .exp101_bridge import load_exp101
from .q0_global import state_label


IID_IMPORTANCE_VERSION = "exp102.q0_iid_importance.v0"


class IidImportanceError(ValueError):
    """Raised when iid-MIS draws or their algebraic replay are invalid."""


def _require(condition, message):
    if not condition:
        raise IidImportanceError(message)


def _as_probability_vector(values, count):
    result = np.asarray(values, dtype=np.float64)
    _require(result.shape == (int(count),), "mixture weights have the wrong shape")
    _require(np.all(np.isfinite(result)) and np.all(result > 0.0),
             "mixture weights must be positive and finite")
    _require(abs(float(result.sum()) - 1.0) <= 1e-14,
             "mixture weights must be normalized")
    result = result / float(result.sum())
    result[-1] = 1.0 - float(result[:-1].sum())
    _require(np.all(result > 0.0), "mixture normalization lost full support")
    return result


def _logsumexp(values):
    values = np.asarray(values, dtype=np.float64)
    _require(values.ndim == 1 and values.size > 0 and np.all(np.isfinite(values)),
             "log-sum-exp inputs must be finite and nonempty")
    maximum = float(values.max())
    result = maximum + math.log(float(np.exp(values - maximum).sum(dtype=np.float64)))
    _require(math.isfinite(result), "log-sum-exp is nonfinite")
    return result


def mixture_log_probability_state(proposals, mixture_weights, state):
    """Evaluate the fixed deterministic mixture on one hard-coset state."""
    proposals = tuple(proposals)
    _require(proposals, "at least one proposal is required")
    weights = _as_probability_vector(mixture_weights, len(proposals))
    values = []
    for mixture_weight, proposal in zip(weights, proposals):
        if not hasattr(proposal, "log_probability_state"):
            raise IidImportanceError("proposal lacks log_probability_state")
        log_q = float(proposal.log_probability_state(state))
        _require(math.isfinite(log_q), "proposal returned a nonfinite log density")
        values.append(math.log(float(mixture_weight)) + log_q)
    return _logsumexp(values)


def _hard_residual(model, state, syndrome):
    return (
        model.H_check.astype(np.int64) @ state.astype(np.int64) % 2
    ).astype(np.uint8) ^ syndrome


def _validate_state(model, syndrome, state):
    state = np.asarray(state)
    _require(state.ndim == 1 and state.shape == (model.num_qubits,),
             "proposal state has the wrong shape")
    _require(np.issubdtype(state.dtype, np.bool_) or np.issubdtype(state.dtype, np.integer),
             "proposal state is not binary data")
    _require(not np.any((state != 0) & (state != 1)), "proposal state is nonbinary")
    state = np.ascontiguousarray(state, dtype=np.uint8)
    _require(not _hard_residual(model, state, syndrome).any(),
             "proposal draw escaped the hard coset")
    return state


def _validate_draw(proposal, model, syndrome, draw):
    if not isinstance(draw, dict) or {"state", "coordinate", "log_q"} - set(draw):
        raise IidImportanceError("proposal draw lacks state/coordinate/log_q")
    state = _validate_state(model, syndrome, draw["state"])
    coordinate = np.asarray(draw["coordinate"])
    _require(coordinate.ndim == 1 and not np.any((coordinate != 0) & (coordinate != 1)),
             "proposal coordinate is invalid")
    _require(hasattr(proposal, "coordinates")
             and np.array_equal(proposal.coordinates.state_from_coordinates(coordinate), state),
             "proposal coordinate does not replay its state")
    stored_log_q = float(draw["log_q"])
    state_log_q = float(proposal.log_probability_state(state))
    coordinate_log_q = float(proposal.log_probability_coordinates(coordinate))
    _require(math.isfinite(stored_log_q) and math.isfinite(state_log_q)
             and math.isfinite(coordinate_log_q), "proposal log density is nonfinite")
    _require(math.isclose(stored_log_q, state_log_q, rel_tol=0.0, abs_tol=1e-12)
             and math.isclose(stored_log_q, coordinate_log_q, rel_tol=0.0, abs_tol=1e-12),
             "proposal log density does not replay")
    return state, stored_log_q


@dataclass(frozen=True)
class IidMixtureDraws:
    """Bit-packed, block-major iid-MIS records suitable for raw storage."""

    states_packed: np.ndarray
    labels: np.ndarray
    physical_weights: np.ndarray
    block_indices: np.ndarray
    source_indices: np.ndarray
    source_log_q: np.ndarray
    mixture_log_q: np.ndarray
    source_log_importance: np.ndarray
    mixture_log_importance: np.ndarray
    block_count: int
    draws_per_proposal_per_block: int
    proposal_count: int

    def __post_init__(self):
        count = int(self.block_count) * int(self.draws_per_proposal_per_block) * int(self.proposal_count)
        _require(int(self.block_count) >= 3, "iid-MIS needs at least three blocks")
        _require(int(self.draws_per_proposal_per_block) > 0 and int(self.proposal_count) > 0,
                 "iid-MIS dimensions must be positive")
        arrays = {
            "states_packed": (self.states_packed, 2, np.uint8),
            "labels": (self.labels, 1, np.uint64),
            "physical_weights": (self.physical_weights, 1, np.int32),
            "block_indices": (self.block_indices, 1, np.int32),
            "source_indices": (self.source_indices, 1, np.int16),
            "source_log_q": (self.source_log_q, 1, np.float64),
            "mixture_log_q": (self.mixture_log_q, 1, np.float64),
            "source_log_importance": (self.source_log_importance, 1, np.float64),
            "mixture_log_importance": (self.mixture_log_importance, 1, np.float64),
        }
        for name, (value, ndim, dtype) in arrays.items():
            array = np.ascontiguousarray(value, dtype=dtype)
            _require(array.ndim == ndim and array.shape[0] == count,
                     f"iid-MIS {name} shape changed")
            if np.issubdtype(dtype, np.floating):
                _require(np.all(np.isfinite(array)), f"iid-MIS {name} is nonfinite")
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        _require(self.states_packed.shape[1] > 0, "iid-MIS packed states are empty")

    def arrays(self):
        return {
            "states_packed": self.states_packed,
            "labels": self.labels,
            "physical_weights": self.physical_weights,
            "block_indices": self.block_indices,
            "source_indices": self.source_indices,
            "source_log_q": self.source_log_q,
            "mixture_log_q": self.mixture_log_q,
            "source_log_importance": self.source_log_importance,
            "mixture_log_importance": self.mixture_log_importance,
        }


def draw_stratified_iid_mixture(model, frame, syndrome, p, proposals, mixture_weights,
                                seeds, *, block_count, draws_per_proposal_per_block):
    """Draw a fixed proposal allocation in each independent, equal-size block.

    The estimator target is ``b**|e|`` on ``H e = syndrome`` with
    ``b=p/(1-p)``.  The returned mixture log weight uses the *whole* proposal
    mixture, not the selected source component, while the source log weight is
    retained for independent-family diagnostics.
    """
    proposals = tuple(proposals)
    _require(proposals, "at least one proposal is required")
    p = float(p)
    _require(math.isfinite(p) and 0.0 < p < 0.5, "iid-MIS p must lie in (0,.5)")
    syndrome = np.asarray(syndrome, dtype=np.uint8)
    _require(syndrome.shape == (model.num_checks,)
             and not np.any((syndrome != 0) & (syndrome != 1)),
             "iid-MIS syndrome has the wrong shape")
    _require(getattr(frame, "k", None) == model.k, "iid-MIS frame/model mismatch")
    block_count = int(block_count)
    draws_per_proposal_per_block = int(draws_per_proposal_per_block)
    _require(block_count >= 3 and draws_per_proposal_per_block > 0,
             "iid-MIS block schedule is invalid")
    weights = _as_probability_vector(mixture_weights, len(proposals))
    # Every block deliberately contains the same number of draws from each
    # source.  Its expected mass is therefore the *uniform* deterministic
    # mixture; accepting a different density here would bias cross-products.
    _require(np.all(np.abs(weights - 1.0 / len(proposals)) <= 1e-14),
             "equal source allocation requires a uniform mixture density")
    seeds = np.asarray(seeds)
    _require(seeds.shape == (block_count, len(proposals))
             and np.issubdtype(seeds.dtype, np.unsignedinteger),
             "iid-MIS seeds have the wrong shape/type")
    _require(np.unique(seeds).size == seeds.size, "iid-MIS seed schedule collides")

    load_exp101()
    from exp101_certified_src.prng import PortablePrng

    total = block_count * len(proposals) * draws_per_proposal_per_block
    packed_width = (model.num_qubits + 7) // 8
    states_packed = np.empty((total, packed_width), dtype=np.uint8)
    labels = np.empty(total, dtype=np.uint64)
    physical_weights = np.empty(total, dtype=np.int32)
    block_indices = np.empty(total, dtype=np.int32)
    source_indices = np.empty(total, dtype=np.int16)
    source_log_q = np.empty(total, dtype=np.float64)
    mixture_log_q = np.empty(total, dtype=np.float64)
    source_log_importance = np.empty(total, dtype=np.float64)
    mixture_log_importance = np.empty(total, dtype=np.float64)
    log_b = math.log(p / (1.0 - p))

    index = 0
    for block in range(block_count):
        for source_index, proposal in enumerate(proposals):
            rng = PortablePrng(int(seeds[block, source_index]))
            for _ in range(draws_per_proposal_per_block):
                state, own_log_q = _validate_draw(proposal, model, syndrome, proposal.sample(rng))
                weight = int(state.sum())
                mix_log_q = mixture_log_probability_state(proposals, weights, state)
                target_log_weight = weight * log_b
                states_packed[index] = np.packbits(state, bitorder="little")
                labels[index] = state_label(frame, state)
                physical_weights[index] = weight
                block_indices[index] = block
                source_indices[index] = source_index
                source_log_q[index] = own_log_q
                mixture_log_q[index] = mix_log_q
                source_log_importance[index] = target_log_weight - own_log_q
                mixture_log_importance[index] = target_log_weight - mix_log_q
                index += 1
    return IidMixtureDraws(
        states_packed=states_packed,
        labels=labels,
        physical_weights=physical_weights,
        block_indices=block_indices,
        source_indices=source_indices,
        source_log_q=source_log_q,
        mixture_log_q=mixture_log_q,
        source_log_importance=source_log_importance,
        mixture_log_importance=mixture_log_importance,
        block_count=block_count,
        draws_per_proposal_per_block=draws_per_proposal_per_block,
        proposal_count=len(proposals),
    )


def _validate_padding(states_packed, num_qubits):
    padding = states_packed.shape[1] * 8 - int(num_qubits)
    _require(padding >= 0, "iid-MIS packed state width is too small")
    if padding:
        mask = np.uint8(~((1 << (8 - padding)) - 1) & 0xFF)
        _require(not np.any(states_packed[:, -1] & mask),
                 "iid-MIS packed states have nonzero padding")


def validate_stratified_iid_mixture(draws, model, frame, syndrome, p, proposals,
                                    mixture_weights):
    """Rebuild labels, densities, and weights from stored raw draws only."""
    if not isinstance(draws, IidMixtureDraws):
        raise TypeError("draws must be IidMixtureDraws")
    proposals = tuple(proposals)
    _require(len(proposals) == draws.proposal_count, "iid-MIS proposal count changed")
    p = float(p)
    _require(math.isfinite(p) and 0.0 < p < 0.5, "iid-MIS p must lie in (0,.5)")
    syndrome = np.asarray(syndrome, dtype=np.uint8)
    _require(syndrome.shape == (model.num_checks,), "iid-MIS syndrome has the wrong shape")
    weights = _as_probability_vector(mixture_weights, len(proposals))
    _require(np.all(np.abs(weights - 1.0 / len(proposals)) <= 1e-14),
             "stored equal source allocation requires a uniform mixture density")
    expected_blocks = np.repeat(
        np.arange(draws.block_count, dtype=np.int32),
        draws.proposal_count * draws.draws_per_proposal_per_block,
    )
    expected_sources = np.tile(
        np.repeat(np.arange(draws.proposal_count, dtype=np.int16),
                  draws.draws_per_proposal_per_block),
        draws.block_count,
    )
    _require(np.array_equal(draws.block_indices, expected_blocks)
             and np.array_equal(draws.source_indices, expected_sources),
             "iid-MIS block/source schedule changed")
    _validate_padding(draws.states_packed, model.num_qubits)
    states = np.unpackbits(draws.states_packed, axis=1, count=model.num_qubits,
                           bitorder="little").astype(np.uint8, copy=False)
    log_b = math.log(p / (1.0 - p))
    for index, state in enumerate(states):
        _require(not _hard_residual(model, state, syndrome).any(),
                 "stored iid-MIS state escaped the hard coset")
        source_index = int(draws.source_indices[index])
        own_log_q = float(proposals[source_index].log_probability_state(state))
        mix_log_q = mixture_log_probability_state(proposals, weights, state)
        weight = int(state.sum())
        _require(int(draws.physical_weights[index]) == weight
                 and draws.labels[index] == state_label(frame, state),
                 "stored iid-MIS state-derived values changed")
        _require(math.isclose(float(draws.source_log_q[index]), own_log_q,
                              rel_tol=0.0, abs_tol=1e-12)
                 and math.isclose(float(draws.mixture_log_q[index]), mix_log_q,
                                  rel_tol=0.0, abs_tol=1e-12)
                 and math.isclose(float(draws.source_log_importance[index]),
                                  weight * log_b - own_log_q, rel_tol=0.0, abs_tol=1e-12)
                 and math.isclose(float(draws.mixture_log_importance[index]),
                                  weight * log_b - mix_log_q, rel_tol=0.0, abs_tol=1e-12),
                 "stored iid-MIS log weights changed")
    return True


def weight_diagnostics(log_importance_weights, *, block_count):
    """Return non-clipped ESS and dominance diagnostics for equal-size blocks."""
    log_weights = np.asarray(log_importance_weights, dtype=np.float64)
    _require(log_weights.ndim == 1 and log_weights.size > 0
             and np.all(np.isfinite(log_weights)), "iid-MIS log weights are invalid")
    block_count = int(block_count)
    _require(block_count >= 1 and log_weights.size % block_count == 0,
             "iid-MIS weights do not divide into blocks")
    block_size = log_weights.size // block_count
    ess = np.empty(block_count, dtype=np.float64)
    maximum = np.empty(block_count, dtype=np.float64)
    for block in range(block_count):
        values = log_weights[block * block_size:(block + 1) * block_size]
        relative = np.exp(values - float(values.max()))
        normalized = relative / float(relative.sum(dtype=np.float64))
        ess[block] = 1.0 / float(np.dot(normalized, normalized))
        maximum[block] = float(normalized.max())
    return {
        "block_effective_sample_sizes": ess,
        "block_max_normalized_weights": maximum,
        "minimum_block_effective_sample_size": float(ess.min()),
        "maximum_block_normalized_weight": float(maximum.max()),
    }
