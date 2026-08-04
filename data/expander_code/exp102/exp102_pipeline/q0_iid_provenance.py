"""IID hard-coset MIS draws with proposal-component provenance.

The original IID-MIS diagnostic intentionally has a compact raw schema.  This
successor is separate so a new feasibility contract can retain each sampled
coordinate and its internal proposal component without changing the frozen
031 raw schema.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .exp101_bridge import load_exp101
from .q0_global import state_label
from .q0_iid_importance import (
    IidImportanceError,
    _as_probability_vector,
    _hard_residual,
    _validate_draw,
    _validate_padding,
    mixture_log_probability_state,
    validate_stratified_iid_mixture,
)


IID_PROVENANCE_VERSION = "exp102.q0_iid_provenance.v0"


class IidProvenanceError(ValueError):
    """Raised when a component-provenanced IID-MIS draw cannot replay."""


def _require(condition, message):
    if not condition:
        raise IidProvenanceError(message)


def _readonly(array, dtype):
    result = np.ascontiguousarray(array, dtype=dtype)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ProvenancedIidMixtureDraws:
    """Standard IID-MIS records plus exact coordinate/component provenance."""

    states_packed: np.ndarray
    coordinates_packed: np.ndarray
    labels: np.ndarray
    physical_weights: np.ndarray
    block_indices: np.ndarray
    source_indices: np.ndarray
    anchor_indices: np.ndarray
    component_indices: np.ndarray
    source_log_q: np.ndarray
    mixture_log_q: np.ndarray
    source_log_importance: np.ndarray
    mixture_log_importance: np.ndarray
    block_count: int
    draws_per_proposal_per_block: int
    proposal_count: int
    coordinate_dimension: int

    def __post_init__(self):
        count = int(self.block_count) * int(self.draws_per_proposal_per_block) * int(self.proposal_count)
        _require(int(self.block_count) >= 3 and int(self.draws_per_proposal_per_block) > 0
                 and int(self.proposal_count) > 0 and int(self.coordinate_dimension) > 0,
                 "provenanced IID-MIS dimensions are invalid")
        fields = {
            "states_packed": (self.states_packed, 2, np.uint8),
            "coordinates_packed": (self.coordinates_packed, 2, np.uint8),
            "labels": (self.labels, 1, np.uint64),
            "physical_weights": (self.physical_weights, 1, np.int32),
            "block_indices": (self.block_indices, 1, np.int32),
            "source_indices": (self.source_indices, 1, np.int16),
            "anchor_indices": (self.anchor_indices, 1, np.int16),
            "component_indices": (self.component_indices, 1, np.int16),
            "source_log_q": (self.source_log_q, 1, np.float64),
            "mixture_log_q": (self.mixture_log_q, 1, np.float64),
            "source_log_importance": (self.source_log_importance, 1, np.float64),
            "mixture_log_importance": (self.mixture_log_importance, 1, np.float64),
        }
        for name, (value, ndim, dtype) in fields.items():
            array = _readonly(value, dtype)
            _require(array.ndim == ndim and array.shape[0] == count,
                     f"provenanced IID-MIS {name} shape changed")
            if np.issubdtype(dtype, np.floating):
                _require(np.all(np.isfinite(array)), f"provenanced IID-MIS {name} is non-finite")
            object.__setattr__(self, name, array)
        _require(self.states_packed.shape[1] > 0
                 and self.coordinates_packed.shape[1] == (int(self.coordinate_dimension) + 7) // 8,
                 "provenanced IID-MIS packed widths changed")

    def standard_arrays(self):
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

    def arrays(self):
        return {
            **self.standard_arrays(),
            "coordinates_packed": self.coordinates_packed,
            "anchor_indices": self.anchor_indices,
            "component_indices": self.component_indices,
        }


def _check_component_provenance(proposal, draw):
    component = int(draw.get("component_index", -1))
    anchor = int(draw.get("anchor_index", -1))
    component_count = getattr(proposal, "num_components", None)
    _require(component_count is not None and 0 <= component < int(component_count),
             "proposal component provenance is invalid")
    anchor_count = getattr(proposal, "num_anchors", None)
    if anchor_count is None:
        _require(anchor == -1, "proposal unexpectedly supplied an anchor index")
    else:
        _require(0 <= anchor < int(anchor_count), "proposal anchor provenance is invalid")
    return anchor, component


def draw_provenanced_stratified_iid_mixture(model, frame, syndrome, p, proposals,
                                            mixture_weights, seeds, *, block_count,
                                            draws_per_proposal_per_block):
    """Draw a fixed equal-allocation mixture while retaining all proposal IDs."""
    proposals = tuple(proposals)
    _require(proposals, "at least one proposal is required")
    p = float(p)
    _require(math.isfinite(p) and 0.0 < p < 0.5, "p must lie in (0,.5)")
    syndrome = np.asarray(syndrome, dtype=np.uint8)
    _require(syndrome.shape == (model.num_checks,), "syndrome length changed")
    block_count = int(block_count)
    draws_per_proposal_per_block = int(draws_per_proposal_per_block)
    _require(block_count >= 3 and draws_per_proposal_per_block > 0,
             "provenanced IID-MIS schedule is invalid")
    mixture = _as_probability_vector(mixture_weights, len(proposals))
    _require(np.all(np.abs(mixture - 1.0 / len(proposals)) <= 1e-14),
             "equal source allocation requires a uniform mixture density")
    seeds = np.asarray(seeds)
    _require(seeds.shape == (block_count, len(proposals))
             and np.issubdtype(seeds.dtype, np.unsignedinteger)
             and np.unique(seeds).size == seeds.size,
             "provenanced IID-MIS seed schedule changed")
    coordinate_dimension = int(getattr(proposals[0].coordinates, "dimension", -1))
    _require(coordinate_dimension > 0
             and all(int(getattr(proposal.coordinates, "dimension", -1)) == coordinate_dimension
                     for proposal in proposals),
             "proposals do not share one coordinate dimension")

    load_exp101()
    from exp101_certified_src.prng import PortablePrng

    count = block_count * len(proposals) * draws_per_proposal_per_block
    packed_width = (model.num_qubits + 7) // 8
    coordinate_width = (coordinate_dimension + 7) // 8
    states_packed = np.empty((count, packed_width), dtype=np.uint8)
    coordinates_packed = np.empty((count, coordinate_width), dtype=np.uint8)
    labels = np.empty(count, dtype=np.uint64)
    physical_weights = np.empty(count, dtype=np.int32)
    block_indices = np.empty(count, dtype=np.int32)
    source_indices = np.empty(count, dtype=np.int16)
    anchor_indices = np.empty(count, dtype=np.int16)
    component_indices = np.empty(count, dtype=np.int16)
    source_log_q = np.empty(count, dtype=np.float64)
    mixture_log_q = np.empty(count, dtype=np.float64)
    source_log_importance = np.empty(count, dtype=np.float64)
    mixture_log_importance = np.empty(count, dtype=np.float64)
    log_b = math.log(p / (1.0 - p))

    index = 0
    for block in range(block_count):
        for source, proposal in enumerate(proposals):
            rng = PortablePrng(int(seeds[block, source]))
            for _ in range(draws_per_proposal_per_block):
                draw = proposal.sample(rng)
                state, own_log_q = _validate_draw(proposal, model, syndrome, draw)
                coordinate = np.asarray(draw["coordinate"], dtype=np.uint8)
                _require(coordinate.shape == (coordinate_dimension,),
                         "proposal coordinate dimension changed")
                anchor, component = _check_component_provenance(proposal, draw)
                mixed_log_q = mixture_log_probability_state(proposals, mixture, state)
                weight = int(state.sum())
                states_packed[index] = np.packbits(state, bitorder="little")
                coordinates_packed[index] = np.packbits(coordinate, bitorder="little")
                labels[index] = state_label(frame, state)
                physical_weights[index] = weight
                block_indices[index] = block
                source_indices[index] = source
                anchor_indices[index] = anchor
                component_indices[index] = component
                source_log_q[index] = own_log_q
                mixture_log_q[index] = mixed_log_q
                source_log_importance[index] = weight * log_b - own_log_q
                mixture_log_importance[index] = weight * log_b - mixed_log_q
                index += 1
    return ProvenancedIidMixtureDraws(
        states_packed=states_packed,
        coordinates_packed=coordinates_packed,
        labels=labels,
        physical_weights=physical_weights,
        block_indices=block_indices,
        source_indices=source_indices,
        anchor_indices=anchor_indices,
        component_indices=component_indices,
        source_log_q=source_log_q,
        mixture_log_q=mixture_log_q,
        source_log_importance=source_log_importance,
        mixture_log_importance=mixture_log_importance,
        block_count=block_count,
        draws_per_proposal_per_block=draws_per_proposal_per_block,
        proposal_count=len(proposals),
        coordinate_dimension=coordinate_dimension,
    )


def validate_provenanced_stratified_iid_mixture(draws, model, frame, syndrome, p,
                                                 proposals, mixture_weights):
    """Rebuild all densities plus coordinate and component provenance from raw."""
    _require(isinstance(draws, ProvenancedIidMixtureDraws),
             "draws are not provenanced IID-MIS records")
    proposals = tuple(proposals)
    _require(len(proposals) == draws.proposal_count, "proposal count changed")
    try:
        from .q0_iid_importance import IidMixtureDraws
        standard = IidMixtureDraws(
            **draws.standard_arrays(), block_count=draws.block_count,
            draws_per_proposal_per_block=draws.draws_per_proposal_per_block,
            proposal_count=draws.proposal_count,
        )
        validate_stratified_iid_mixture(
            standard, model, frame, syndrome, p, proposals, mixture_weights,
        )
    except IidImportanceError as exc:
        raise IidProvenanceError(str(exc)) from exc
    _validate_padding(draws.coordinates_packed, draws.coordinate_dimension)
    states = np.unpackbits(
        draws.states_packed, axis=1, count=model.num_qubits, bitorder="little",
    ).astype(np.uint8, copy=False)
    coordinates = np.unpackbits(
        draws.coordinates_packed, axis=1, count=draws.coordinate_dimension, bitorder="little",
    ).astype(np.uint8, copy=False)
    for index, (state, coordinate) in enumerate(zip(states, coordinates, strict=True)):
        source = int(draws.source_indices[index])
        proposal = proposals[source]
        _require(np.array_equal(proposal.coordinates.state_from_coordinates(coordinate), state)
                 and np.array_equal(proposal.coordinates.coordinates_of_state(state), coordinate),
                 "stored proposal coordinate does not replay")
        expected_anchor, expected_component = _check_component_provenance(
            proposal,
            {"anchor_index": int(draws.anchor_indices[index]),
             "component_index": int(draws.component_indices[index])},
        )
        _require(expected_anchor == int(draws.anchor_indices[index])
                 and expected_component == int(draws.component_indices[index])
                 and not _hard_residual(model, state, np.asarray(syndrome, dtype=np.uint8)).any(),
                 "stored component provenance changed")
    return True
