"""Truth-free dressed logical XOR moves for the q=0 hard coset.

The historical logical-stratified artifact contains, for every frozen logical
signature, both a code-only kernel move and (when decoding succeeded) an
absolute hard-coset representative.  This module deterministically keeps the
lower-weight of

``base_anchor xor codebook_move`` and ``decoded_representative``.

XORing the retained representative with ``base_anchor`` gives a self-inverse
zero-syndrome move.  Drawing that move with a state-independent probability
therefore needs only the physical target-weight Metropolis ratio; neither the
planted error nor a proposal-density approximation enters the kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np

from .q0_global import _signature_rank_masks
from .q0_logical_stratified import (
    LogicalStratifiedFrozenArtifact,
    _bits_to_uint64,
    validate_logical_stratified_frozen_artifact,
)


CENTER_PRESERVING_VERSION = "exp102.q0_center_preserving.discovery.v0"
CENTER_PRESERVING_CATALOG_VERSION = "exp102.q0_center_preserving.catalog.v0"


class CenterPreservingError(ValueError):
    """Raised when a dressed catalog or XOR transition loses an invariant."""


_POPCOUNT8 = np.asarray([int(value).bit_count() for value in range(256)], dtype=np.uint8)


def _require(condition, message):
    if not condition:
        raise CenterPreservingError(message)


def _bits(value, *, ndim, name):
    array = np.asarray(value)
    _require(array.ndim == int(ndim), f"{name} has the wrong dimension")
    _require(np.issubdtype(array.dtype, np.bool_)
             or np.issubdtype(array.dtype, np.integer), f"{name} is not binary data")
    _require(not np.any((array != 0) & (array != 1)), f"{name} is not binary")
    return np.ascontiguousarray(array, dtype=np.uint8)


def _integers(value, *, ndim, name, dtype):
    array = np.asarray(value)
    _require(array.ndim == int(ndim) and np.issubdtype(array.dtype, np.integer),
             f"{name} is not an integer array")
    return np.ascontiguousarray(array, dtype=dtype)


def _packed_weights(packed):
    packed = _integers(packed, ndim=2, name="packed states", dtype=np.uint8)
    return _POPCOUNT8[packed].sum(axis=1, dtype=np.int32)


def _digest(arrays, scalars):
    digest = hashlib.sha256(CENTER_PRESERVING_CATALOG_VERSION.encode("ascii") + b"\0")
    for scalar in scalars:
        digest.update(str(scalar).encode("ascii") + b"\0")
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(value.dtype.str.encode("ascii") + b"\0")
        digest.update(np.asarray(value.shape, dtype=">u8").tobytes())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _select_rank_first(signatures, anchor_weights, move_weights, anchors_packed,
                       *, logical_dimension, max_moves):
    signatures = _integers(
        signatures, ndim=1, name="candidate signatures", dtype=np.uint64,
    )
    anchor_weights = _integers(
        anchor_weights, ndim=1, name="candidate anchor weights", dtype=np.int32,
    )
    move_weights = _integers(
        move_weights, ndim=1, name="candidate move weights", dtype=np.int32,
    )
    anchors_packed = _integers(
        anchors_packed, ndim=2, name="candidate anchors", dtype=np.uint8,
    )
    count = signatures.size
    logical_dimension = int(logical_dimension)
    max_moves = int(max_moves)
    _require(count and anchor_weights.shape == move_weights.shape == (count,)
             and anchors_packed.shape[0] == count,
             "dressed candidate dimensions are inconsistent")
    _require(1 <= logical_dimension <= 64 and logical_dimension <= max_moves <= count,
             "dressed catalog size cannot retain a rank-complete set")
    _require(not np.any(signatures == np.uint64(0))
             and np.unique(signatures).size == count,
             "dressed candidate signatures are zero or duplicated")

    ordered = sorted(range(count), key=lambda index: (
        int(anchor_weights[index]), int(move_weights[index]),
        int(signatures[index]), anchors_packed[index].tobytes(),
    ))
    selected = []
    selected_set = set()
    pivots = {}
    for index in ordered:
        residue = int(signatures[index])
        while residue:
            pivot = residue.bit_length() - 1
            if pivot not in pivots:
                pivots[pivot] = residue
                selected.append(index)
                selected_set.add(index)
                break
            residue ^= pivots[pivot]
        if len(pivots) == logical_dimension:
            break
    _require(len(pivots) == logical_dimension,
             "dressed candidates do not span every logical direction")
    rank_count = len(selected)
    for index in ordered:
        if len(selected) >= max_moves:
            break
        if index not in selected_set:
            selected.append(index)
            selected_set.add(index)
    _require(len(selected) == max_moves, "dressed catalog fill is incomplete")
    roles = np.asarray(
        [1 if position < rank_count else 2 for position in range(max_moves)],
        dtype=np.uint8,
    )
    return np.asarray(selected, dtype=np.int32), roles


def choose_dressed_candidates(base_anchor, signatures, codebook_moves_packed,
                              decoded_valid, decoded_weights, decoded_packed,
                              *, logical_dimension, max_moves):
    """Choose and rank-select dressed representatives without sampler feedback."""
    base = _bits(base_anchor, ndim=1, name="dressed base anchor")
    signatures = _integers(
        signatures, ndim=1, name="dressed signatures", dtype=np.uint64,
    )
    codebook = _integers(
        codebook_moves_packed, ndim=2, name="dressed codebook moves", dtype=np.uint8,
    )
    valid = _integers(decoded_valid, ndim=1, name="decoded validity", dtype=np.uint8)
    stored_decoded_weights = _integers(
        decoded_weights, ndim=1, name="decoded weights", dtype=np.int32,
    )
    decoded = _integers(
        decoded_packed, ndim=2, name="decoded states", dtype=np.uint8,
    )
    count = signatures.size
    packed_width = (base.size + 7) // 8
    _require(codebook.shape == decoded.shape == (count, packed_width)
             and valid.shape == stored_decoded_weights.shape == (count,),
             "dressed source arrays have incompatible dimensions")
    _require(not np.any((valid != 0) & (valid != 1)),
             "decoded validity flags are not binary")
    _require(np.all((valid == 0) == (stored_decoded_weights == -1)),
             "decoded validity and stored weights disagree")

    base_packed = np.packbits(base, bitorder="little")
    code_anchors = np.bitwise_xor(codebook, base_packed[None, :])
    code_weights = _packed_weights(code_anchors)
    replayed_decoded_weights = _packed_weights(decoded)
    _require(np.array_equal(
        replayed_decoded_weights[valid.astype(bool)],
        stored_decoded_weights[valid.astype(bool)],
    ), "decoded candidate weight replay failed")

    use_decoded = (valid == 1) & (stored_decoded_weights < code_weights)
    tied = np.flatnonzero(
        (valid == 1) & (stored_decoded_weights == code_weights),
    )
    for index in tied:
        use_decoded[index] = decoded[index].tobytes() < code_anchors[index].tobytes()
    chosen = code_anchors.copy()
    chosen[use_decoded] = decoded[use_decoded]
    chosen_weights = _packed_weights(chosen)
    moves = np.bitwise_xor(chosen, base_packed[None, :])
    move_weights = _packed_weights(moves)
    selected, roles = _select_rank_first(
        signatures, chosen_weights, move_weights, chosen,
        logical_dimension=logical_dimension, max_moves=max_moves,
    )
    return {
        "selected_indices": selected,
        "selection_roles": roles,
        "anchors_packed": np.ascontiguousarray(chosen[selected], dtype=np.uint8),
        "moves_packed": np.ascontiguousarray(moves[selected], dtype=np.uint8),
        "signatures": np.ascontiguousarray(signatures[selected], dtype=np.uint64),
        "anchor_weights": np.ascontiguousarray(chosen_weights[selected], dtype=np.int32),
        "move_weights": np.ascontiguousarray(move_weights[selected], dtype=np.int32),
        "source_kind": np.ascontiguousarray(use_decoded[selected], dtype=np.uint8),
        "candidate_source_counts": np.asarray(
            [int(count - use_decoded.sum()), int(use_decoded.sum())], dtype=np.int32,
        ),
    }


@dataclass(frozen=True)
class DressedLogicalCatalog:
    base_anchor: np.ndarray
    selected_indices: np.ndarray
    selection_roles: np.ndarray
    anchors_packed: np.ndarray
    moves_packed: np.ndarray
    signatures: np.ndarray
    anchor_weights: np.ndarray
    move_weights: np.ndarray
    source_kind: np.ndarray
    candidate_source_counts: np.ndarray
    logical_dimension: int
    num_qubits: int
    codebook_sha256: str
    transcript_sha256: str
    catalog_sha256: str

    def __post_init__(self):
        base = _bits(self.base_anchor, ndim=1, name="catalog base anchor")
        selected = _integers(
            self.selected_indices, ndim=1, name="catalog selected indices", dtype=np.int32,
        )
        roles = _integers(
            self.selection_roles, ndim=1, name="catalog selection roles", dtype=np.uint8,
        )
        anchors = _integers(
            self.anchors_packed, ndim=2, name="catalog anchors", dtype=np.uint8,
        )
        moves = _integers(
            self.moves_packed, ndim=2, name="catalog moves", dtype=np.uint8,
        )
        signatures = _integers(
            self.signatures, ndim=1, name="catalog signatures", dtype=np.uint64,
        )
        anchor_weights = _integers(
            self.anchor_weights, ndim=1, name="catalog anchor weights", dtype=np.int32,
        )
        move_weights = _integers(
            self.move_weights, ndim=1, name="catalog move weights", dtype=np.int32,
        )
        source = _integers(
            self.source_kind, ndim=1, name="catalog source kinds", dtype=np.uint8,
        )
        source_counts = _integers(
            self.candidate_source_counts, ndim=1, name="candidate source counts",
            dtype=np.int32,
        )
        size = signatures.size
        width = (base.size + 7) // 8
        _require(selected.shape == roles.shape == anchor_weights.shape
                 == move_weights.shape == source.shape == (size,)
                 and anchors.shape == moves.shape == (size, width)
                 and source_counts.shape == (2,), "catalog arrays are inconsistent")
        _require(np.unique(selected).size == size and np.unique(signatures).size == size
                 and np.all((roles == 1) | (roles == 2))
                 and np.all((source == 0) | (source == 1)),
                 "catalog identities are duplicated or invalid")
        _require(int(self.logical_dimension) in range(1, 65)
                 and int(self.num_qubits) == base.size,
                 "catalog dimensions are invalid")
        for name in ("codebook_sha256", "transcript_sha256", "catalog_sha256"):
            value = getattr(self, name)
            _require(isinstance(value, str) and len(value) == 64
                     and all(character in "0123456789abcdef" for character in value),
                     f"{name} is not a SHA256")
        for name, value in (
                ("base_anchor", base), ("selected_indices", selected),
                ("selection_roles", roles), ("anchors_packed", anchors),
                ("moves_packed", moves), ("signatures", signatures),
                ("anchor_weights", anchor_weights), ("move_weights", move_weights),
                ("source_kind", source), ("candidate_source_counts", source_counts)):
            stored = np.array(value, copy=True, order="C")
            stored.setflags(write=False)
            object.__setattr__(self, name, stored)

    @property
    def size(self):
        return int(self.signatures.size)

    def unpack_moves(self):
        return np.unpackbits(
            self.moves_packed, axis=1, count=self.num_qubits, bitorder="little",
        ).astype(np.uint8, copy=False)

    def unpack_anchors(self):
        return np.unpackbits(
            self.anchors_packed, axis=1, count=self.num_qubits, bitorder="little",
        ).astype(np.uint8, copy=False)


def build_dressed_logical_catalog(model, frame, artifact, *, max_moves=127):
    """Build a frozen rank-complete catalog from a validated historical artifact."""
    _require(isinstance(artifact, LogicalStratifiedFrozenArtifact),
             "dressed source is not a logical-stratified artifact")
    validate_logical_stratified_frozen_artifact(model, frame, artifact)
    max_moves = int(max_moves)
    base = np.ascontiguousarray(artifact.catalog.anchors[0], dtype=np.uint8)
    chosen = choose_dressed_candidates(
        base, artifact.codebook.signatures, artifact.codebook.moves_packed,
        artifact.transcript.valid, artifact.transcript.decoded_weights,
        artifact.transcript.decoded_packed, logical_dimension=model.k,
        max_moves=max_moves,
    )
    digest = _digest(
        (np.packbits(base, bitorder="little"), chosen["selected_indices"].astype(">i4"),
         chosen["selection_roles"], chosen["anchors_packed"], chosen["moves_packed"],
         chosen["signatures"].astype(">u8"), chosen["anchor_weights"].astype(">i4"),
         chosen["move_weights"].astype(">i4"), chosen["source_kind"],
         chosen["candidate_source_counts"].astype(">i4")),
        (model.fingerprint(), frame.fingerprint(), artifact.codebook.codebook_sha256,
         artifact.transcript.transcript_sha256, model.k, model.num_qubits, max_moves),
    )
    result = DressedLogicalCatalog(
        base_anchor=base, **chosen, logical_dimension=model.k,
        num_qubits=model.num_qubits,
        codebook_sha256=artifact.codebook.codebook_sha256,
        transcript_sha256=artifact.transcript.transcript_sha256,
        catalog_sha256=digest,
    )
    validate_dressed_logical_catalog(model, frame, artifact.syndrome, result)
    return result


def validate_dressed_logical_catalog(model, frame, syndrome, catalog):
    """Replay hard-coset, signature, weight, rank, and catalog-hash identities."""
    _require(isinstance(catalog, DressedLogicalCatalog),
             "dressed catalog has the wrong type")
    y = _bits(syndrome, ndim=1, name="dressed syndrome")
    _require(y.shape == (model.num_checks,)
             and catalog.num_qubits == model.num_qubits
             and catalog.logical_dimension == model.k,
             "dressed catalog/model dimensions changed")
    base_residual = (
        model.H_check.astype(np.int64) @ catalog.base_anchor.astype(np.int64) % 2
    ).astype(np.uint8)
    _require(np.array_equal(base_residual, y), "dressed base leaves the hard coset")
    moves = catalog.unpack_moves()
    anchors = catalog.unpack_anchors()
    _require(np.array_equal(anchors, catalog.base_anchor[None, :] ^ moves),
             "dressed anchor/move identity changed")
    residual = (
        model.H_check.astype(np.int64) @ moves.T.astype(np.int64) % 2
    ).astype(np.uint8)
    _require(not residual.any(), "dressed logical move leaves ker(H_Z)")
    labels = (
        frame.W_basis.astype(np.int64) @ moves.T.astype(np.int64) % 2
    ).T.astype(np.uint8)
    replayed = np.asarray([_bits_to_uint64(row) for row in labels], dtype=np.uint64)
    _require(np.array_equal(replayed, catalog.signatures),
             "dressed signature does not equal W d")
    _require(np.array_equal(anchors.sum(axis=1).astype(np.int32), catalog.anchor_weights)
             and np.array_equal(moves.sum(axis=1).astype(np.int32), catalog.move_weights),
             "dressed weight replay failed")
    _require(_signature_rank_masks(catalog.signatures, model.k) == model.k
             and _signature_rank_masks(
                 catalog.signatures[catalog.selection_roles == 1], model.k,
             ) == model.k,
             "dressed catalog is not rank-complete")
    expected = _digest(
        (np.packbits(catalog.base_anchor, bitorder="little"),
         catalog.selected_indices.astype(">i4"), catalog.selection_roles,
         catalog.anchors_packed, catalog.moves_packed,
         catalog.signatures.astype(">u8"), catalog.anchor_weights.astype(">i4"),
         catalog.move_weights.astype(">i4"), catalog.source_kind,
         catalog.candidate_source_counts.astype(">i4")),
        (model.fingerprint(), frame.fingerprint(), catalog.codebook_sha256,
         catalog.transcript_sha256, model.k, model.num_qubits, catalog.size),
    )
    _require(expected == catalog.catalog_sha256, "dressed catalog SHA replay failed")
    return True


def xor_log_acceptance(current_weight, proposal_weight, p):
    """Return the untruncated exact log MH ratio for a symmetric XOR move."""
    p = float(p)
    _require(math.isfinite(p) and 0.0 < p < 0.5, "XOR move p must lie in (0,.5)")
    current_weight = int(current_weight)
    proposal_weight = int(proposal_weight)
    _require(current_weight >= 0 and proposal_weight >= 0,
             "XOR move weights must be nonnegative")
    return (proposal_weight - current_weight) * math.log(p / (1.0 - p))


def xor_metropolis_step(state, move, p, uniform):
    """Apply one exact self-inverse logical XOR proposal from an explicit uniform."""
    current = _bits(state, ndim=1, name="XOR current state")
    direction = _bits(move, ndim=1, name="XOR logical move")
    _require(current.shape == direction.shape, "XOR state/move dimensions changed")
    uniform = float(uniform)
    _require(math.isfinite(uniform) and 0.0 <= uniform < 1.0,
             "XOR acceptance uniform is invalid")
    proposal = current ^ direction
    log_ratio = xor_log_acceptance(int(current.sum()), int(proposal.sum()), p)
    log_alpha = min(0.0, log_ratio)
    accepted = uniform == 0.0 or math.log(uniform) < log_alpha
    return (proposal if accepted else current.copy()), accepted, log_ratio
