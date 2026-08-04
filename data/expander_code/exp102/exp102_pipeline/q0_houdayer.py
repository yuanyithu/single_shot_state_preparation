"""Replica-pair isoenergetic cluster primitives for the q=0 hard coset.

Coordinates use the sparse H_X stabilizer rows plus a deterministic HGP tensor
logical complement.  For two coordinate states, a Houdayer component connects
all differing coordinates that meet in one physical-bit XOR factor.  Swapping
one complete component between replicas preserves the product target exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from .exp101_bridge import load_exp101
from .q0_hgp_collapsed import join_hgp_state, validate_hgp_wiring


HOUDAYER_COORDINATE_VERSION = "exp102.q0_houdayer.coordinates.v0"
HOUDAYER_REDUCED_LOGICAL_COORDINATE_VERSION = (
    "exp102.q0_houdayer.reduced_logical_coordinates.v0"
)


class HoudayerConflictError(ValueError):
    """Raised when a replica-cluster algebraic invariant is violated."""


def _require(condition, message):
    if not condition:
        raise HoudayerConflictError(message)


def _as_bits(value, *, ndim, name):
    array = np.asarray(value)
    _require(array.ndim == int(ndim), f"{name} has the wrong dimension")
    _require(np.all((array == 0) | (array == 1)), f"{name} must be binary")
    return np.ascontiguousarray(array, dtype=np.uint8)


def _signature_mask(bits):
    bits = _as_bits(bits, ndim=1, name="logical signature")
    _require(bits.size <= 64, "Houdayer signatures support at most 64 logical bits")
    value = np.uint64(0)
    for index in np.flatnonzero(bits):
        value |= np.uint64(1) << np.uint64(index)
    return int(value)


def _gf2_rank(matrix):
    load_exp101()
    from exp101_certified_src.gf2 import gf2_rank

    return int(gf2_rank(np.asarray(matrix, dtype=np.uint8)))


def build_sparse_hgp_coordinate_basis(H, model, frame):
    """Return a code-only sparse basis of ``ker(H_Z)`` with logical masks."""
    load_exp101()
    from exp101_certified_src.gf2 import (
        gf2_extend_basis,
        gf2_matmul,
        gf2_nullspace,
        gf2_rank,
        gf2_rowspace_basis,
    )

    H = _as_bits(H, ndim=2, name="classical H")
    validate_hgp_wiring(H, model)
    rows, columns = H.shape
    side = columns - int(gf2_rank(H))
    _require(side > 0 and side * side == int(model.k),
             "Houdayer tensor logical dimensions changed")
    stabilizers = _as_bits(model.stabilizer_rows, ndim=2, name="stabilizer rows")
    _require(_gf2_rank(stabilizers) == stabilizers.shape[0],
             "Houdayer stabilizer rows are dependent")
    kernel = np.ascontiguousarray(gf2_nullspace(H), dtype=np.uint8)
    _unused, complement_indices = gf2_extend_basis(
        gf2_rowspace_basis(H), np.eye(columns, dtype=np.uint8),
    )
    quotient = np.eye(columns, dtype=np.uint8)[complement_indices]
    _require(kernel.shape == quotient.shape == (side, columns),
             "Houdayer classical quotient dimensions changed")
    zero_b = np.zeros((rows, rows), dtype=np.uint8)
    tensor_logicals = np.ascontiguousarray([
        join_hgp_state(np.outer(left, right), zero_b)
        for left in kernel for right in quotient
    ], dtype=np.uint8)
    _require(tensor_logicals.shape == (model.k, model.num_qubits),
             "Houdayer tensor logical basis has the wrong shape")
    combined = np.vstack((stabilizers, tensor_logicals)).astype(np.uint8, copy=False)
    expected_dimension = int(model.num_qubits - model.H_check.shape[0])
    _require(combined.shape[0] == expected_dimension
             and _gf2_rank(combined) == expected_dimension,
             "Houdayer coordinate rows do not form a hard-coset basis")
    _require(not gf2_matmul(model.H_check, combined.T).any(),
             "Houdayer coordinate row leaves the hard kernel")
    labels = gf2_matmul(frame.W_basis, tensor_logicals.T).T
    _require(_gf2_rank(labels) == int(model.k),
             "Houdayer tensor logical complement lacks full label rank")
    masks = np.zeros(combined.shape[0], dtype=np.uint64)
    for index, row in enumerate(labels, start=stabilizers.shape[0]):
        masks[index] = np.uint64(_signature_mask(row))
    return {
        "version": HOUDAYER_COORDINATE_VERSION,
        "generators": np.ascontiguousarray(combined, dtype=np.uint8),
        "stabilizer_count": int(stabilizers.shape[0]),
        "logical_count": int(tensor_logicals.shape[0]),
        "logical_masks": np.ascontiguousarray(masks),
        "tensor_logicals": tensor_logicals,
    }


def build_sparse_hgp_reduced_logical_coordinate_basis(H, model, frame):
    """Return H_X plus the canonical reduced logical basis of ``ker(H_Z)``.

    This is a separate, entirely code-defined coordinate choice for the same
    Houdayer identity.  It is deliberately not selected from a chain state or
    target outcome: the only reduction is the fixed row rule in q0_global.
    """
    load_exp101()
    from exp101_certified_src.gf2 import gf2_matmul, gf2_rank
    from .q0_global import reduce_logical_basis

    H = _as_bits(H, ndim=2, name="classical H")
    validate_hgp_wiring(H, model)
    stabilizers = _as_bits(model.stabilizer_rows, ndim=2, name="stabilizer rows")
    _require(_gf2_rank(stabilizers) == stabilizers.shape[0],
             "Houdayer stabilizer rows are dependent")
    reduced_logicals = _as_bits(reduce_logical_basis(model.logical_move_basis), ndim=2,
                                name="reduced logical rows")
    _require(reduced_logicals.shape == (model.k, model.num_qubits),
             "Houdayer reduced logical dimensions changed")
    combined = np.vstack((stabilizers, reduced_logicals)).astype(np.uint8, copy=False)
    expected_dimension = int(model.num_qubits - model.H_check.shape[0])
    _require(combined.shape[0] == expected_dimension
             and _gf2_rank(combined) == expected_dimension,
             "Houdayer reduced-logical rows do not form a hard-coset basis")
    _require(not gf2_matmul(model.H_check, combined.T).any(),
             "Houdayer reduced-logical row leaves the hard kernel")
    labels = gf2_matmul(frame.W_basis, reduced_logicals.T).T
    _require(_gf2_rank(labels) == int(model.k),
             "Houdayer reduced logical complement lacks full label rank")
    masks = np.zeros(combined.shape[0], dtype=np.uint64)
    for index, row in enumerate(labels, start=stabilizers.shape[0]):
        masks[index] = np.uint64(_signature_mask(row))
    return {
        "version": HOUDAYER_REDUCED_LOGICAL_COORDINATE_VERSION,
        "generators": np.ascontiguousarray(combined, dtype=np.uint8),
        "stabilizer_count": int(stabilizers.shape[0]),
        "logical_count": int(reduced_logicals.shape[0]),
        "logical_masks": np.ascontiguousarray(masks),
        "reduced_logicals": reduced_logicals,
    }


def coordinate_factor_scopes(generators):
    """Return the coordinate support of each physical-bit XOR factor."""
    generators = _as_bits(generators, ndim=2, name="Houdayer generators")
    scopes = tuple(
        tuple(int(value) for value in np.flatnonzero(generators[:, qubit]))
        for qubit in range(generators.shape[1])
    )
    _require(all(scope for scope in scopes), "a physical factor has no coordinate support")
    return scopes


def coordinates_to_state(base, generators, coordinates):
    """Map one binary coordinate vector to its affine hard-coset state."""
    base = _as_bits(base, ndim=1, name="Houdayer base state")
    generators = _as_bits(generators, ndim=2, name="Houdayer generators")
    coordinates = _as_bits(coordinates, ndim=1, name="Houdayer coordinates")
    _require(base.size == generators.shape[1] and coordinates.size == generators.shape[0],
             "Houdayer coordinate dimensions do not match")
    delta = (coordinates.astype(np.int64) @ generators.astype(np.int64) % 2).astype(np.uint8)
    return np.ascontiguousarray(base ^ delta, dtype=np.uint8)


@dataclass(frozen=True)
class CoordinateReadout:
    """A deterministic GF(2) inverse for one sparse coordinate basis."""

    generator_shape: tuple[int, int]
    generator_sha256: str
    pivot_columns: np.ndarray
    inverse: np.ndarray


def _generator_sha256(generators):
    generators = _as_bits(generators, ndim=2, name="Houdayer generators")
    header = np.asarray(generators.shape, dtype=">u8").tobytes()
    return hashlib.sha256(header + generators.tobytes(order="C")).hexdigest()


def prepare_coordinate_readout(generators):
    """Precompute the exact coordinate inverse used by repeated start states."""
    generators = _as_bits(generators, ndim=2, name="Houdayer generators")
    dimension = generators.shape[0]
    _require(dimension <= generators.shape[1],
             "Houdayer coordinate basis has more rows than physical bits")
    # Pivot columns form an invertible coordinate readout because the generator
    # rows are independent.  The retained square system is the transpose of
    # the row-coordinate map, hence its inverse directly decodes a delta.
    workspace = generators.copy()
    pivot_columns = []
    pivot_row = 0
    for column in range(generators.shape[1]):
        candidates = np.flatnonzero(workspace[pivot_row:, column])
        if not candidates.size:
            continue
        source = pivot_row + int(candidates[0])
        if source != pivot_row:
            workspace[[pivot_row, source]] = workspace[[source, pivot_row]]
        for row in np.flatnonzero(workspace[:, column]):
            if row != pivot_row:
                workspace[row] ^= workspace[pivot_row]
        pivot_columns.append(column)
        pivot_row += 1
        if pivot_row == dimension:
            break
    _require(len(pivot_columns) == dimension, "Houdayer generator rows are dependent")
    system = generators[:, pivot_columns].T.copy()
    augmented = np.concatenate((system, np.eye(dimension, dtype=np.uint8)), axis=1)
    for column in range(dimension):
        candidates = np.flatnonzero(augmented[column:, column])
        _require(candidates.size, "Houdayer coordinate system lost a pivot")
        source = column + int(candidates[0])
        if source != column:
            augmented[[column, source]] = augmented[[source, column]]
        for row in np.flatnonzero(augmented[:, column]):
            if row != column:
                augmented[row] ^= augmented[column]
    _require(np.array_equal(augmented[:, :dimension], np.eye(dimension, dtype=np.uint8)),
             "Houdayer coordinate inverse failed")
    return CoordinateReadout(
        generator_shape=(int(generators.shape[0]), int(generators.shape[1])),
        generator_sha256=_generator_sha256(generators),
        pivot_columns=np.ascontiguousarray(pivot_columns, dtype=np.int32),
        inverse=np.ascontiguousarray(augmented[:, dimension:], dtype=np.uint8),
    )


def coordinates_from_kernel_delta(delta, generators, *, readout=None):
    """Recover unique coordinates for a delta in the generator row space."""
    delta = _as_bits(delta, ndim=1, name="Houdayer kernel delta")
    generators = _as_bits(generators, ndim=2, name="Houdayer generators")
    _require(delta.size == generators.shape[1], "Houdayer delta dimensions do not match")
    if readout is None:
        readout = prepare_coordinate_readout(generators)
    _require(isinstance(readout, CoordinateReadout),
             "Houdayer coordinate readout has the wrong type")
    _require(readout.generator_shape == tuple(int(value) for value in generators.shape)
             and readout.generator_sha256 == _generator_sha256(generators),
             "Houdayer coordinate readout belongs to another generator basis")
    dimension = generators.shape[0]
    _require(readout.pivot_columns.shape == (dimension,)
             and readout.inverse.shape == (dimension, dimension),
             "Houdayer coordinate readout dimensions are invalid")
    coordinates = (
        readout.inverse.astype(np.int64) @ delta[readout.pivot_columns].astype(np.int64) % 2
    ).astype(np.uint8)
    recovered = (coordinates.astype(np.int64) @ generators.astype(np.int64) % 2).astype(np.uint8)
    _require(np.array_equal(recovered, delta), "Houdayer delta is outside the coordinate kernel")
    return np.ascontiguousarray(coordinates, dtype=np.uint8)


class _UnionFind:
    def __init__(self, values):
        self.parent = {int(value): int(value) for value in values}

    def find(self, value):
        value = int(value)
        parent = self.parent[value]
        if parent != value:
            parent = self.find(parent)
            self.parent[value] = parent
        return parent

    def union(self, left, right):
        left = self.find(left)
        right = self.find(right)
        if left != right:
            if left < right:
                self.parent[right] = left
            else:
                self.parent[left] = right


def houdayer_components(left, right, factor_scopes):
    """Partition differing coordinates into exact isoenergetic components."""
    left = _as_bits(left, ndim=1, name="left Houdayer coordinates")
    right = _as_bits(right, ndim=1, name="right Houdayer coordinates")
    _require(left.shape == right.shape, "Houdayer pair coordinate shapes differ")
    differing = np.flatnonzero(left ^ right).astype(np.int32)
    if not differing.size:
        return tuple()
    active = np.zeros(left.size, dtype=np.uint8)
    active[differing] = 1
    union_find = _UnionFind(differing)
    for scope in factor_scopes:
        entries = [int(value) for value in scope if active[int(value)]]
        for value in entries[1:]:
            union_find.union(entries[0], value)
    groups = {}
    for value in differing:
        groups.setdefault(union_find.find(value), []).append(int(value))
    return tuple(np.asarray(groups[key], dtype=np.int32) for key in sorted(groups))


def houdayer_swap_one(left, right, components, component_index):
    """Swap one complete component; this involution preserves pair energy."""
    left = _as_bits(left, ndim=1, name="left Houdayer coordinates")
    right = _as_bits(right, ndim=1, name="right Houdayer coordinates")
    _require(left.shape == right.shape, "Houdayer pair coordinate shapes differ")
    if not components:
        return left.copy(), right.copy()
    component_index = int(component_index)
    _require(0 <= component_index < len(components), "Houdayer component index is invalid")
    component = np.asarray(components[component_index], dtype=np.int32)
    output_left, output_right = left.copy(), right.copy()
    output_left[component] = right[component]
    output_right[component] = left[component]
    return output_left, output_right


def component_logical_mask(component, difference, logical_masks):
    """Return the logical label delta induced by swapping one component."""
    difference = _as_bits(difference, ndim=1, name="Houdayer coordinate difference")
    masks = np.asarray(logical_masks, dtype=np.uint64)
    _require(difference.size == masks.size, "Houdayer logical masks have the wrong length")
    value = np.uint64(0)
    for coordinate in np.asarray(component, dtype=np.int32):
        if difference[int(coordinate)]:
            value ^= masks[int(coordinate)]
    return int(value)
