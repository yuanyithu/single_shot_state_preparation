"""Exact generalized Houdayer moves for the collapsed q=0 HGP B marginal.

The collapsed HGP sampler represents B as one uint32 mask per classical
column.  A complete component swap acts directly on these masks, so it can be
composed with a cold collapsed-B kernel without mistaking a resampled A column
for movement of the slow variable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .q0_houdayer import houdayer_components


COLLAPSED_HOUDAYER_VERSION = "exp102.q0_collapsed_houdayer.v0"
COLLAPSED_HOUDAYER_KERNEL = "collapsed_b_complete_component_swap.v0"


class CollapsedHoudayerConflictError(ValueError):
    """Raised when a collapsed-B HCA algebraic invariant is lost."""


def _require(condition, message):
    if not condition:
        raise CollapsedHoudayerConflictError(message)


def _as_binary_matrix(value, name):
    result = np.asarray(value)
    _require(result.ndim == 2, f"{name} must be a matrix")
    _require(np.all((result == 0) | (result == 1)), f"{name} must be binary")
    return np.ascontiguousarray(result, dtype=np.uint8)


def _as_masks(value, rows, name):
    result = np.asarray(value)
    _require(result.ndim == 1 and result.size == int(rows),
             f"{name} has the wrong B-column shape")
    _require(np.issubdtype(result.dtype, np.integer), f"{name} must be integral")
    result = np.ascontiguousarray(result, dtype=np.uint32)
    if int(rows) < 32:
        _require(not np.any(result >> np.uint32(int(rows))),
                 f"{name} contains bits outside the classical row space")
    return result


def _as_factor_masks(value, rows, columns, name):
    result = np.asarray(value)
    _require(result.ndim == 1 and result.size == int(columns),
             f"{name} has the wrong factor shape")
    _require(np.issubdtype(result.dtype, np.integer), f"{name} must be integral")
    result = np.ascontiguousarray(result, dtype=np.uint32)
    if int(rows) < 32:
        _require(not np.any(result >> np.uint32(int(rows))),
                 f"{name} contains bits outside the classical row space")
    return result


def b_masks_to_bits(b_columns, rows):
    """Flatten column masks into the canonical row-major B-bit order."""
    rows = int(rows)
    masks = _as_masks(b_columns, rows, "B columns")
    result = np.zeros(rows * rows, dtype=np.uint8)
    for column, mask in enumerate(masks):
        raw = int(mask)
        for row in range(rows):
            result[row * rows + column] = np.uint8((raw >> row) & 1)
    return result


def b_bits_to_masks(bits, rows):
    """Invert :func:`b_masks_to_bits` without relying on native word order."""
    rows = int(rows)
    result = np.asarray(bits)
    _require(result.ndim == 1 and result.size == rows * rows,
             "B bits have the wrong flattened shape")
    _require(np.all((result == 0) | (result == 1)), "B bits must be binary")
    masks = np.zeros(rows, dtype=np.uint32)
    for row in range(rows):
        bit = np.uint32(1 << row)
        for column in range(rows):
            if int(result[row * rows + column]):
                masks[column] |= bit
    return masks


def collapsed_b_factor_scopes(H):
    """Return the B-bit scopes of the exact collapsed column factors."""
    H = _as_binary_matrix(H, "classical H")
    rows, columns = H.shape
    _require(rows <= 32, "collapsed HCA supports at most 32 classical rows")
    scopes = []
    for factor in range(columns):
        active_columns = np.flatnonzero(H[:, factor])
        _require(active_columns.size, "collapsed HCA factor has empty support")
        scope = np.asarray(
            [row * rows + int(column)
             for row in range(rows) for column in active_columns],
            dtype=np.int32,
        )
        scopes.append(scope)
    return tuple(scopes)


def hgp_syndrome_to_columns(syndrome, H):
    """Pack an HGP syndrome matrix into the collapsed column-factor masks."""
    H = _as_binary_matrix(H, "classical H")
    rows, columns = H.shape
    syndrome = np.asarray(syndrome)
    _require(syndrome.ndim == 1 and syndrome.size == rows * columns,
             "HGP syndrome has the wrong flattened shape")
    _require(np.all((syndrome == 0) | (syndrome == 1)), "HGP syndrome must be binary")
    matrix = np.asarray(syndrome, dtype=np.uint8).reshape(rows, columns)
    result = np.zeros(columns, dtype=np.uint32)
    for column in range(columns):
        value = 0
        for row in range(rows):
            value |= int(matrix[row, column]) << row
        result[column] = np.uint32(value)
    return result


def collapsed_b_factor_masks(H, b_columns, y_columns=None):
    """Compute ``B H`` (or ``Y xor B H``) as one packed mask per factor."""
    H = _as_binary_matrix(H, "classical H")
    rows, columns = H.shape
    masks = _as_masks(b_columns, rows, "B columns")
    if y_columns is None:
        result = np.zeros(columns, dtype=np.uint32)
    else:
        result = _as_factor_masks(y_columns, rows, columns, "Y columns").copy()
    for factor in range(columns):
        value = int(result[factor])
        for column in np.flatnonzero(H[:, factor]):
            value ^= int(masks[int(column)])
        result[factor] = np.uint32(value)
    return result


def _pair_key(left, right):
    left_bytes = np.asarray(left, dtype=np.uint32).tobytes()
    right_bytes = np.asarray(right, dtype=np.uint32).tobytes()
    return left_bytes + right_bytes if left_bytes <= right_bytes else right_bytes + left_bytes


def _popcount_masks(masks):
    return sum(int(value).bit_count() for value in np.asarray(masks, dtype=np.uint32))


def _factor_pair_key(left_masks, right_masks):
    pairs = np.empty((len(left_masks), 2), dtype=np.uint32)
    for index, (left, right) in enumerate(zip(left_masks, right_masks)):
        if int(left) <= int(right):
            pairs[index] = (left, right)
        else:
            pairs[index] = (right, left)
    return pairs


@dataclass(frozen=True)
class CollapsedBHoudayerKernel:
    """Code- and syndrome-bound data for a collapsed-B complete-component swap."""

    H: np.ndarray
    y_columns: np.ndarray
    factor_scopes: tuple

    @property
    def rows(self):
        return int(self.H.shape[0])

    @property
    def columns(self):
        return int(self.H.shape[1])


@dataclass
class CollapsedBHoudayerPair:
    """Two collapsed B states targeting the product cold marginal."""

    left: np.ndarray
    right: np.ndarray

    def copy(self):
        return CollapsedBHoudayerPair(self.left.copy(), self.right.copy())


def build_collapsed_b_houdayer_kernel(H, y_columns):
    """Bind factor scopes to the collapsed posterior's fixed syndrome columns."""
    H = _as_binary_matrix(H, "classical H")
    rows, columns = H.shape
    _require(rows <= 32, "collapsed HCA supports at most 32 classical rows")
    y_columns = _as_factor_masks(y_columns, rows, columns, "Y columns")
    return CollapsedBHoudayerKernel(
        H=H,
        y_columns=y_columns,
        factor_scopes=collapsed_b_factor_scopes(H),
    )


def initialize_collapsed_b_houdayer_pair(kernel, left, right):
    """Validate and copy a pair of collapsed B states."""
    _require(isinstance(kernel, CollapsedBHoudayerKernel),
             "collapsed HCA kernel has the wrong type")
    return CollapsedBHoudayerPair(
        _as_masks(left, kernel.rows, "left B columns").copy(),
        _as_masks(right, kernel.rows, "right B columns").copy(),
    )


def collapsed_b_houdayer_components(pair, kernel):
    """Find complete disagreement components in the collapsed factor graph."""
    _require(isinstance(pair, CollapsedBHoudayerPair),
             "collapsed HCA pair has the wrong type")
    left = _as_masks(pair.left, kernel.rows, "left B columns")
    right = _as_masks(pair.right, kernel.rows, "right B columns")
    return houdayer_components(
        b_masks_to_bits(left, kernel.rows),
        b_masks_to_bits(right, kernel.rows),
        kernel.factor_scopes,
    )


def collapsed_b_component_delta(pair, kernel, component):
    """Return the B-mask XOR delta associated with one complete component."""
    _require(isinstance(pair, CollapsedBHoudayerPair),
             "collapsed HCA pair has the wrong type")
    component = np.asarray(component, dtype=np.int32)
    left_bits = b_masks_to_bits(pair.left, kernel.rows)
    difference = left_bits ^ b_masks_to_bits(pair.right, kernel.rows)
    _require(component.ndim == 1 and component.size
             and np.all((0 <= component) & (component < difference.size)),
             "collapsed HCA component is invalid")
    _require(np.all(difference[component]),
             "collapsed HCA component is not a disagreement component")
    delta_bits = np.zeros_like(difference)
    delta_bits[component] = difference[component]
    return b_bits_to_masks(delta_bits, kernel.rows)


def collapsed_b_pair_invariants(pair, kernel):
    """Return exact discrete invariants of the collapsed product pair target."""
    _require(isinstance(pair, CollapsedBHoudayerPair),
             "collapsed HCA pair has the wrong type")
    left = _as_masks(pair.left, kernel.rows, "left B columns")
    right = _as_masks(pair.right, kernel.rows, "right B columns")
    left_factors = collapsed_b_factor_masks(kernel.H, left, kernel.y_columns)
    right_factors = collapsed_b_factor_masks(kernel.H, right, kernel.y_columns)
    return {
        "pair_b_weight": _popcount_masks(left) + _popcount_masks(right),
        "factor_pairs": _factor_pair_key(left_factors, right_factors),
        "left_factor_masks": left_factors,
        "right_factor_masks": right_factors,
    }


def collapsed_b_component_swap(pair, kernel, component_index):
    """Swap one complete B component and verify factorwise pair invariance."""
    _require(isinstance(pair, CollapsedBHoudayerPair),
             "collapsed HCA pair has the wrong type")
    components = collapsed_b_houdayer_components(pair, kernel)
    if not components:
        return {
            "component_count": 0,
            "component_index": -1,
            "whole_pair_exchange": False,
            "new_unordered_b_pair": False,
            "changed_b_bits": 0,
            "pair_b_weight_before_after": [
                _popcount_masks(pair.left) + _popcount_masks(pair.right),
            ] * 2,
        }
    component_index = int(component_index)
    _require(0 <= component_index < len(components),
             "collapsed HCA component index is invalid")
    before_key = _pair_key(pair.left, pair.right)
    before = collapsed_b_pair_invariants(pair, kernel)
    component = np.asarray(components[component_index], dtype=np.int32)
    delta = collapsed_b_component_delta(pair, kernel, component)
    pair.left ^= delta
    pair.right ^= delta
    after = collapsed_b_pair_invariants(pair, kernel)
    _require(after["pair_b_weight"] == before["pair_b_weight"],
             "collapsed HCA swap changed the B unary pair weight")
    _require(np.array_equal(after["factor_pairs"], before["factor_pairs"]),
             "collapsed HCA swap changed a collapsed factor pair")
    after_key = _pair_key(pair.left, pair.right)
    return {
        "component_count": len(components),
        "component_index": component_index,
        "whole_pair_exchange": bool(after_key == before_key),
        "new_unordered_b_pair": bool(after_key != before_key),
        "changed_b_bits": _popcount_masks(delta),
        "pair_b_weight_before_after": [
            int(before["pair_b_weight"]), int(after["pair_b_weight"]),
        ],
    }


def collapsed_b_pair_key(pair):
    """Ordered key used by exhaustive small-code transition tests."""
    return np.asarray(pair.left, dtype=np.uint32).tobytes() + np.asarray(
        pair.right, dtype=np.uint32,
    ).tobytes()


def collapsed_b_houdayer_transition_distribution(pair, kernel):
    """Enumerate one exact complete-component B-HCA step for a small code."""
    components = collapsed_b_houdayer_components(pair, kernel)
    if not components:
        return {collapsed_b_pair_key(pair): 1.0}
    probability = 1.0 / len(components)
    result = {}
    for component_index in range(len(components)):
        moved = pair.copy()
        collapsed_b_component_swap(moved, kernel, component_index)
        key = collapsed_b_pair_key(moved)
        result[key] = result.get(key, 0.0) + probability
    _require(abs(sum(result.values()) - 1.0) <= 1e-13,
             "collapsed HCA transition probabilities do not sum to one")
    return result
