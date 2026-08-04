"""Linear-code trellis structure utilities for exact q=0 feasibility checks.

The routines work directly with a binary parity-check matrix.  They measure
the conventional trellis state exponent at every variable cut, preserving XOR
linearity instead of first expanding every check into dense factor tables.
"""

from __future__ import annotations

import heapq

import numpy as np


class TrellisStructureError(ValueError):
    """Raised when a binary matrix or variable order is not well formed."""


def _require(condition, message):
    if not condition:
        raise TrellisStructureError(message)


def _as_binary_matrix(matrix):
    result = np.asarray(matrix, dtype=np.uint8)
    _require(result.ndim == 2 and result.shape[0] > 0 and result.shape[1] > 0,
             "trellis matrix must be nonempty and two-dimensional")
    _require(np.all((result == 0) | (result == 1)), "trellis matrix must be binary")
    return np.ascontiguousarray(result)


def _column_vectors(matrix):
    """Pack each GF(2) column into a Python integer for exact rank updates."""
    matrix = _as_binary_matrix(matrix)
    vectors = []
    for column in range(matrix.shape[1]):
        vector = 0
        for row in np.flatnonzero(matrix[:, column]):
            vector |= 1 << int(row)
        vectors.append(vector)
    return tuple(vectors)


def _basis_insert(basis, value):
    """Insert one GF(2) vector into a pivot-indexed echelon basis."""
    value = int(value)
    while value:
        pivot = value.bit_length() - 1
        previous = basis.get(pivot)
        if previous is None:
            basis[pivot] = value
            return 1
        value ^= previous
    return 0


def gf2_column_rank(matrix):
    """Return exact GF(2) rank using a column-oriented binary basis."""
    basis = {}
    for vector in _column_vectors(matrix):
        _basis_insert(basis, vector)
    return len(basis)


def _validate_order(order, variable_count):
    order = tuple(int(value) for value in order)
    _require(len(order) == int(variable_count), "trellis order has the wrong length")
    _require(tuple(sorted(order)) == tuple(range(int(variable_count))),
             "trellis order must be a variable permutation")
    return order


def rank_profiles(matrix, order):
    """Return exact prefix/suffix column-span ranks in the requested order."""
    matrix = _as_binary_matrix(matrix)
    order = _validate_order(order, matrix.shape[1])
    vectors = _column_vectors(matrix)
    prefix = np.zeros(matrix.shape[1] + 1, dtype=np.int32)
    suffix = np.zeros(matrix.shape[1] + 1, dtype=np.int32)
    basis = {}
    for position, variable in enumerate(order, start=1):
        _basis_insert(basis, vectors[variable])
        prefix[position] = len(basis)
    basis = {}
    for position in range(matrix.shape[1] - 1, -1, -1):
        _basis_insert(basis, vectors[order[position]])
        suffix[position] = len(basis)
    return prefix, suffix


def trellis_state_profile(matrix, order):
    """Return conventional-trellis state dimensions for a binary linear code.

    At cut ``i`` the exponent is

    ``rank(H_prefix) + rank(H_suffix) - rank(H)``.

    This equals ``dim(C) - dim(C_past) - dim(C_future)`` for
    ``C=ker(H)``.  It is a property of the code/order, not of a syndrome or
    a particular nonzero unary weight.
    """
    matrix = _as_binary_matrix(matrix)
    order = _validate_order(order, matrix.shape[1])
    prefix, suffix = rank_profiles(matrix, order)
    total_rank = int(prefix[-1])
    exponents = prefix + suffix - total_rank
    _require(np.all(exponents >= 0), "trellis state exponent became negative")
    _require(int(exponents[0]) == 0 and int(exponents[-1]) == 0,
             "trellis endpoint state is not trivial")
    return {
        "order": order,
        "rank": total_rank,
        "prefix_ranks": prefix,
        "suffix_ranks": suffix,
        "state_exponents": exponents,
    }


def tanner_min_degree_order(matrix):
    """Return a deterministic min-degree order of the check-factor primal graph."""
    matrix = _as_binary_matrix(matrix)
    variable_count = matrix.shape[1]
    adjacency = [set() for _ in range(variable_count)]
    for check in range(matrix.shape[0]):
        scope = tuple(int(value) for value in np.flatnonzero(matrix[check]))
        for index, left in enumerate(scope):
            adjacency[left].update(scope[:index])
            adjacency[left].update(scope[index + 1:])
    alive = set(range(variable_count))
    heap = [(len(adjacency[variable]), variable) for variable in alive]
    heapq.heapify(heap)
    order = []
    while alive:
        while heap:
            degree, variable = heapq.heappop(heap)
            if variable not in alive:
                continue
            neighbors = adjacency[variable] & alive
            if degree == len(neighbors):
                break
            heapq.heappush(heap, (len(neighbors), variable))
        else:  # pragma: no cover - defensive invariant
            raise TrellisStructureError("min-degree heap exhausted")
        neighbors = sorted(neighbors)
        for index, left in enumerate(neighbors):
            for right in neighbors[index + 1:]:
                adjacency[left].add(right)
                adjacency[right].add(left)
        alive.remove(variable)
        for neighbor in neighbors:
            adjacency[neighbor].discard(variable)
            heapq.heappush(heap, (len(adjacency[neighbor] & alive), neighbor))
        order.append(variable)
    return tuple(order)
