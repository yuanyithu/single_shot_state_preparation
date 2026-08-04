"""Exact reference kernel for reduced-coordinate Houdayer replica pairs.

The pair target is ``pi(e_left|y) * pi(e_right|y)`` with the q=0 hard
constraint.  Each clock uses independent random-scan coordinate heatbaths on
the two replicas followed by one complete-component Houdayer swap.  The
reference implementation is deliberately small and auditable; it is for
local feasibility work, not a production sampler.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math

import numpy as np

from .q0_houdayer import (
    build_sparse_hgp_reduced_logical_coordinate_basis,
    coordinate_factor_scopes,
    coordinates_from_kernel_delta,
    coordinates_to_state,
    houdayer_components,
    prepare_coordinate_readout,
)


HOUDAYER_PAIR_VERSION = "exp102.q0_houdayer_pair.v0"
HOUDAYER_PAIR_KERNEL = "reduced_coordinate_random_scan_heatbath_plus_houdayer.v0"


class HoudayerPairConflictError(ValueError):
    """Raised when the pair state loses a hard-coset or coordinate invariant."""


def _require(condition, message):
    if not condition:
        raise HoudayerPairConflictError(message)


def _as_bits(value, *, ndim, name):
    array = np.asarray(value)
    _require(array.ndim == int(ndim), f"{name} has the wrong dimension")
    _require(np.all((array == 0) | (array == 1)), f"{name} must be binary")
    return np.ascontiguousarray(array, dtype=np.uint8)


def _label(W_basis, state):
    bits = (W_basis.astype(np.int64) @ state.astype(np.int64) % 2).astype(np.uint8)
    value = 0
    for position in np.flatnonzero(bits):
        value |= 1 << int(position)
    return value


def _residual(H_check, state):
    return (H_check.astype(np.int64) @ state.astype(np.int64) % 2).astype(np.uint8)


def _unordered_pair_key(left, right):
    left_bytes = left.tobytes()
    right_bytes = right.tobytes()
    return left_bytes + right_bytes if left_bytes <= right_bytes else right_bytes + left_bytes


def deterministic_low_energy_logical_starts(model, frame, planted, *, count, orders=(1, 2, 3)):
    """Return the first label-distinct P-derived starts under a frozen rule.

    This is initialization-only infrastructure.  The planted state is never
    used by the transition energy, heatbath, or Houdayer acceptance identity.
    """
    from .q0_global import reduce_logical_basis

    planted = _as_bits(planted, ndim=1, name="planted Houdayer start")
    _require(planted.shape == (model.num_qubits,),
             "planted Houdayer start dimensions do not match")
    count = int(count)
    _require(count > 0, "Houdayer logical start count must be positive")
    orders = tuple(int(order) for order in orders)
    _require(orders and all(order > 0 for order in orders),
             "Houdayer logical start orders are invalid")
    reduced = _as_bits(reduce_logical_basis(model.logical_move_basis), ndim=2,
                       name="reduced logical basis")
    _require(reduced.shape == (model.k, model.num_qubits),
             "reduced logical basis dimensions changed")
    _require(not (model.H_check.astype(np.int64) @ reduced.T.astype(np.int64) % 2).any(),
             "reduced logical basis leaves the hard kernel")
    candidates = {}
    for order in orders:
        for combination in itertools.combinations(range(model.k), order):
            move = np.bitwise_xor.reduce(reduced[list(combination)], axis=0)
            packed = np.packbits(move, bitorder="little").tobytes()
            if packed in candidates:
                continue
            signature = _label(frame.W_basis, move)
            _require(signature, "Houdayer logical start has zero signature")
            state = np.ascontiguousarray(planted ^ move, dtype=np.uint8)
            candidates[packed] = {
                "move": move,
                "state": state,
                "move_weight": int(move.sum()),
                "state_weight": int(state.sum()),
                "signature": int(signature),
                "packed": packed,
            }
    key = lambda record: (
        record["state_weight"], record["move_weight"], record["signature"], record["packed"],
    )
    by_signature = {}
    for record in sorted(candidates.values(), key=key):
        by_signature.setdefault(record["signature"], record)
    ordered = tuple(sorted(by_signature.values(), key=key))
    _require(len(ordered) >= count, "Houdayer logical start catalog is too small")
    return tuple({
        "index": index,
        "state": record["state"].copy(),
        "move": record["move"].copy(),
        "state_weight": int(record["state_weight"]),
        "move_weight": int(record["move_weight"]),
        "signature": int(record["signature"]),
    } for index, record in enumerate(ordered[:count]))


@dataclass(frozen=True)
class HoudayerPairKernel:
    """Code/disorder-bound ingredients of one exact replica-pair kernel."""

    p: float
    log_odds: float
    base_state: np.ndarray
    syndrome: np.ndarray
    H_check: np.ndarray
    W_basis: np.ndarray
    generators: np.ndarray
    factor_scopes: tuple
    supports: tuple
    readout: object
    logical_masks: np.ndarray
    stabilizer_count: int
    logical_count: int

    @property
    def coordinate_count(self):
        return int(self.generators.shape[0])

    @property
    def num_qubits(self):
        return int(self.generators.shape[1])


@dataclass
class HoudayerPairState:
    """Mutable physical and coordinate state for two correlated replicas."""

    left: np.ndarray
    right: np.ndarray
    left_coordinates: np.ndarray
    right_coordinates: np.ndarray

    def copy(self):
        return HoudayerPairState(
            self.left.copy(), self.right.copy(),
            self.left_coordinates.copy(), self.right_coordinates.copy(),
        )


def build_reduced_houdayer_pair_kernel(H, model, frame, syndrome, p):
    """Bind the code-only reduced coordinate basis to one legal hard coset."""
    p = float(p)
    _require(math.isfinite(p) and 0.0 < p < 0.5,
             "Houdayer pair p must lie in (0, .5)")
    syndrome = _as_bits(syndrome, ndim=1, name="Houdayer pair syndrome")
    _require(syndrome.shape == (model.num_checks,),
             "Houdayer pair syndrome dimensions do not match")
    base_state = np.ascontiguousarray(
        model.logical_sector_section.apply(syndrome, strict=True), dtype=np.uint8,
    )
    _require(np.array_equal(_residual(model.H_check, base_state), syndrome),
             "Houdayer pair section state leaves the hard coset")
    basis = build_sparse_hgp_reduced_logical_coordinate_basis(H, model, frame)
    supports = tuple(
        np.ascontiguousarray(np.flatnonzero(row), dtype=np.int32)
        for row in basis["generators"]
    )
    _require(all(support.size for support in supports),
             "Houdayer pair coordinate generator has empty physical support")
    return HoudayerPairKernel(
        p=p,
        log_odds=math.log((1.0 - p) / p),
        base_state=base_state,
        syndrome=syndrome,
        H_check=np.ascontiguousarray(model.H_check, dtype=np.uint8),
        W_basis=np.ascontiguousarray(frame.W_basis, dtype=np.uint8),
        generators=np.ascontiguousarray(basis["generators"], dtype=np.uint8),
        factor_scopes=coordinate_factor_scopes(basis["generators"]),
        supports=supports,
        readout=prepare_coordinate_readout(basis["generators"]),
        logical_masks=np.ascontiguousarray(basis["logical_masks"], dtype=np.uint64),
        stabilizer_count=int(basis["stabilizer_count"]),
        logical_count=int(basis["logical_count"]),
    )


def initialize_houdayer_pair(kernel, left, right):
    """Map two arbitrary legal states into the pair coordinate representation."""
    if not isinstance(kernel, HoudayerPairKernel):
        raise TypeError("Houdayer pair kernel has the wrong type")
    left = _as_bits(left, ndim=1, name="left Houdayer state")
    right = _as_bits(right, ndim=1, name="right Houdayer state")
    _require(left.shape == right.shape == (kernel.num_qubits,),
             "Houdayer pair state dimensions do not match")
    _require(np.array_equal(_residual(kernel.H_check, left), kernel.syndrome)
             and np.array_equal(_residual(kernel.H_check, right), kernel.syndrome),
             "Houdayer pair initialization leaves the hard coset")
    left_coordinates = coordinates_from_kernel_delta(
        left ^ kernel.base_state, kernel.generators, readout=kernel.readout,
    )
    right_coordinates = coordinates_from_kernel_delta(
        right ^ kernel.base_state, kernel.generators, readout=kernel.readout,
    )
    result = HoudayerPairState(
        left.copy(), right.copy(), left_coordinates, right_coordinates,
    )
    validate_houdayer_pair_state(kernel, result)
    return result


def validate_houdayer_pair_state(kernel, pair):
    """Check all algebraic identities, intended for tests and fixed-clock audits."""
    if not isinstance(kernel, HoudayerPairKernel) or not isinstance(pair, HoudayerPairState):
        raise TypeError("Houdayer pair kernel/state types are invalid")
    for name, state, coordinates in (
            ("left", pair.left, pair.left_coordinates),
            ("right", pair.right, pair.right_coordinates)):
        state = _as_bits(state, ndim=1, name=f"{name} Houdayer state")
        coordinates = _as_bits(coordinates, ndim=1, name=f"{name} Houdayer coordinates")
        _require(state.shape == (kernel.num_qubits,)
                 and coordinates.shape == (kernel.coordinate_count,),
                 f"{name} Houdayer dimensions changed")
        _require(np.array_equal(_residual(kernel.H_check, state), kernel.syndrome),
                 f"{name} Houdayer state left the hard coset")
        _require(np.array_equal(
            coordinates_to_state(kernel.base_state, kernel.generators, coordinates), state,
        ), f"{name} Houdayer coordinate state drifted")
    return True


def coordinate_flip_probability(weight_delta, log_odds):
    """Exact probability to flip one binary coordinate under its heatbath."""
    log_ratio = -float(log_odds) * int(weight_delta)
    if log_ratio >= 0.0:
        return 1.0 / (1.0 + math.exp(-log_ratio))
    ratio = math.exp(log_ratio)
    return ratio / (1.0 + ratio)


def coordinate_heatbath(pair, kernel, replica, coordinate, uniform):
    """Apply one exact random-scan coordinate heatbath given a fixed uniform draw."""
    _require(replica in ("left", "right"), "Houdayer replica name is invalid")
    coordinate = int(coordinate)
    _require(0 <= coordinate < kernel.coordinate_count,
             "Houdayer coordinate index is invalid")
    uniform = float(uniform)
    _require(math.isfinite(uniform) and 0.0 <= uniform < 1.0,
             "Houdayer heatbath uniform is invalid")
    state = pair.left if replica == "left" else pair.right
    coordinates = pair.left_coordinates if replica == "left" else pair.right_coordinates
    support = kernel.supports[coordinate]
    weight_delta = int(support.size) - 2 * int(state[support].sum())
    probability = coordinate_flip_probability(weight_delta, kernel.log_odds)
    flipped = uniform < probability
    if flipped:
        state[support] ^= np.uint8(1)
        coordinates[coordinate] ^= np.uint8(1)
    return {
        "coordinate": coordinate,
        "weight_delta": weight_delta,
        "flip_probability": probability,
        "flipped": bool(flipped),
    }


def _component_physical_delta(kernel, component, difference):
    active = np.asarray(component, dtype=np.int32)
    _require(active.size and np.all(difference[active]),
             "Houdayer component is not a disagreement component")
    if active.size == 1:
        return kernel.generators[int(active[0])].copy()
    return np.ascontiguousarray(
        np.bitwise_xor.reduce(kernel.generators[active], axis=0), dtype=np.uint8,
    )


def houdayer_component_swap(pair, kernel, component_index):
    """Swap one complete component and report whether it created a new pair."""
    difference = np.ascontiguousarray(
        pair.left_coordinates ^ pair.right_coordinates, dtype=np.uint8,
    )
    components = houdayer_components(
        pair.left_coordinates, pair.right_coordinates, kernel.factor_scopes,
    )
    if not components:
        return {
            "component_count": 0,
            "component_index": -1,
            "pair_total_weight_before_after": [int(pair.left.sum() + pair.right.sum())] * 2,
            "whole_pair_exchange": False,
            "new_unordered_pair": False,
        }
    component_index = int(component_index)
    _require(0 <= component_index < len(components),
             "Houdayer component index is invalid")
    component = components[component_index]
    before_total = int(pair.left.sum() + pair.right.sum())
    before_key = _unordered_pair_key(pair.left, pair.right)
    physical_delta = _component_physical_delta(kernel, component, difference)
    previous_left_coordinates = pair.left_coordinates[component].copy()
    pair.left_coordinates[component] = pair.right_coordinates[component]
    pair.right_coordinates[component] = previous_left_coordinates
    pair.left ^= physical_delta
    pair.right ^= physical_delta
    after_total = int(pair.left.sum() + pair.right.sum())
    _require(after_total == before_total,
             "Houdayer component swap changed the pair energy")
    after_key = _unordered_pair_key(pair.left, pair.right)
    return {
        "component_count": len(components),
        "component_index": component_index,
        "pair_total_weight_before_after": [before_total, after_total],
        "whole_pair_exchange": bool(after_key == before_key),
        "new_unordered_pair": bool(after_key != before_key),
    }


def houdayer_pair_clock(pair, kernel, left_rng, right_rng, cluster_rng,
                        local_updates_per_clock):
    """Run independent random-scan heatbaths then one exact Houdayer move."""
    local_updates_per_clock = int(local_updates_per_clock)
    _require(local_updates_per_clock > 0,
             "Houdayer local update count must be positive")
    local_attempts = 0
    local_flips = 0
    for _ in range(local_updates_per_clock):
        left_result = coordinate_heatbath(
            pair, kernel, "left", left_rng.randbelow(kernel.coordinate_count), left_rng.random(),
        )
        right_result = coordinate_heatbath(
            pair, kernel, "right", right_rng.randbelow(kernel.coordinate_count), right_rng.random(),
        )
        local_attempts += 2
        local_flips += int(left_result["flipped"]) + int(right_result["flipped"])
    components = houdayer_components(
        pair.left_coordinates, pair.right_coordinates, kernel.factor_scopes,
    )
    component_index = cluster_rng.randbelow(len(components)) if components else -1
    cluster = houdayer_component_swap(pair, kernel, component_index)
    return {
        "local_attempts": local_attempts,
        "local_flips": local_flips,
        "houdayer_attempted": int(bool(components)),
        "houdayer_component_count": len(components),
        "houdayer_new_unordered_pair": int(cluster["new_unordered_pair"]),
        "houdayer_whole_pair_exchange": int(cluster["whole_pair_exchange"]),
    }


def pair_coordinate_key(pair):
    """Unique finite-state key used only by exact small-code transition tests."""
    return pair.left_coordinates.tobytes() + pair.right_coordinates.tobytes()


def _accumulate_clock_counters(counters, row):
    counters["local_attempts"] += int(row["local_attempts"])
    counters["local_flips"] += int(row["local_flips"])
    counters["houdayer_attempts"] += int(row["houdayer_attempted"])
    counters["houdayer_new_unordered_pair"] += int(row["houdayer_new_unordered_pair"])
    counters["houdayer_whole_pair_exchange"] += int(row["houdayer_whole_pair_exchange"])


def _stream_seed(seed, salt):
    return (int(seed) ^ int(salt)) & ((1 << 63) - 1)


def pair_clock_transition_distribution(pair, kernel):
    """Enumerate the one-local-update exact clock for small-code stationarity tests."""
    validate_houdayer_pair_state(kernel, pair)
    result = {}
    dimension = kernel.coordinate_count
    for left_coordinate in range(dimension):
        left_support = kernel.supports[left_coordinate]
        left_delta = int(left_support.size) - 2 * int(pair.left[left_support].sum())
        left_flip_probability = coordinate_flip_probability(left_delta, kernel.log_odds)
        for left_flip, left_probability in (
                (False, 1.0 - left_flip_probability), (True, left_flip_probability)):
            if not left_probability:
                continue
            left_pair = pair.copy()
            if left_flip:
                left_pair.left[left_support] ^= np.uint8(1)
                left_pair.left_coordinates[left_coordinate] ^= np.uint8(1)
            for right_coordinate in range(dimension):
                right_support = kernel.supports[right_coordinate]
                right_delta = int(right_support.size) - 2 * int(left_pair.right[right_support].sum())
                right_flip_probability = coordinate_flip_probability(right_delta, kernel.log_odds)
                for right_flip, right_probability in (
                        (False, 1.0 - right_flip_probability), (True, right_flip_probability)):
                    if not right_probability:
                        continue
                    local_pair = left_pair.copy()
                    if right_flip:
                        local_pair.right[right_support] ^= np.uint8(1)
                        local_pair.right_coordinates[right_coordinate] ^= np.uint8(1)
                    components = houdayer_components(
                        local_pair.left_coordinates, local_pair.right_coordinates,
                        kernel.factor_scopes,
                    )
                    if not components:
                        key = pair_coordinate_key(local_pair)
                        result[key] = result.get(key, 0.0) + (
                            left_probability * right_probability / (dimension * dimension)
                        )
                        continue
                    for component_index in range(len(components)):
                        final_pair = local_pair.copy()
                        houdayer_component_swap(final_pair, kernel, component_index)
                        key = pair_coordinate_key(final_pair)
                        result[key] = result.get(key, 0.0) + (
                            left_probability * right_probability
                            / (dimension * dimension * len(components))
                        )
    _require(abs(sum(result.values()) - 1.0) <= 1e-13,
             "Houdayer exact clock does not sum to one")
    return result


def run_houdayer_pair_trajectory(kernel, left_initial, right_initial, seed,
                                 burn_clocks, measurement_clocks,
                                 local_updates_per_clock):
    """Run a fixed-clock local trajectory and retain both correlated replicas."""
    from .exp101_bridge import load_exp101

    load_exp101()
    from exp101_certified_src.prng import PortablePrng

    burn_clocks = int(burn_clocks)
    measurement_clocks = int(measurement_clocks)
    _require(burn_clocks >= 0 and measurement_clocks > 0,
             "Houdayer pair clocks are invalid")
    pair = initialize_houdayer_pair(kernel, left_initial, right_initial)
    left_rng = PortablePrng(_stream_seed(seed, 0x243F6A8885A308D3))
    right_rng = PortablePrng(_stream_seed(seed, 0x13198A2E03707344))
    cluster_rng = PortablePrng(_stream_seed(seed, 0xA4093822299F31D0))
    counters = {
        "local_attempts": 0,
        "local_flips": 0,
        "houdayer_attempts": 0,
        "houdayer_new_unordered_pair": 0,
        "houdayer_whole_pair_exchange": 0,
    }
    for _ in range(burn_clocks):
        row = houdayer_pair_clock(
            pair, kernel, left_rng, right_rng, cluster_rng, local_updates_per_clock,
        )
        _accumulate_clock_counters(counters, row)
    burn_left = pair.left.copy()
    burn_right = pair.right.copy()
    bytes_per_state = (kernel.num_qubits + 7) // 8
    left_packed = np.empty((measurement_clocks, bytes_per_state), dtype=np.uint8)
    right_packed = np.empty((measurement_clocks, bytes_per_state), dtype=np.uint8)
    left_labels = np.empty(measurement_clocks, dtype=np.uint64)
    right_labels = np.empty(measurement_clocks, dtype=np.uint64)
    left_weights = np.empty(measurement_clocks, dtype=np.int32)
    right_weights = np.empty(measurement_clocks, dtype=np.int32)
    component_counts = np.empty(measurement_clocks, dtype=np.int32)
    new_pair_flags = np.empty(measurement_clocks, dtype=np.uint8)
    whole_swap_flags = np.empty(measurement_clocks, dtype=np.uint8)
    for clock in range(measurement_clocks):
        row = houdayer_pair_clock(
            pair, kernel, left_rng, right_rng, cluster_rng, local_updates_per_clock,
        )
        _accumulate_clock_counters(counters, row)
        left_packed[clock] = np.packbits(pair.left, bitorder="little")
        right_packed[clock] = np.packbits(pair.right, bitorder="little")
        left_labels[clock] = np.uint64(_label(kernel.W_basis, pair.left))
        right_labels[clock] = np.uint64(_label(kernel.W_basis, pair.right))
        left_weights[clock] = int(pair.left.sum())
        right_weights[clock] = int(pair.right.sum())
        component_counts[clock] = int(row["houdayer_component_count"])
        new_pair_flags[clock] = row["houdayer_new_unordered_pair"]
        whole_swap_flags[clock] = row["houdayer_whole_pair_exchange"]
    validate_houdayer_pair_state(kernel, pair)
    states = np.unpackbits(
        np.concatenate((left_packed, right_packed), axis=0), axis=1,
        count=kernel.num_qubits, bitorder="little",
    ).astype(np.uint8, copy=False)
    residuals = (
        kernel.H_check.astype(np.int64) @ states.T.astype(np.int64) % 2
    ).T.astype(np.uint8) ^ kernel.syndrome[None, :]
    _require(not residuals.any(), "Houdayer pair trajectory left the hard coset")
    return {
        "initial_left_packed": np.packbits(np.asarray(left_initial, dtype=np.uint8), bitorder="little"),
        "initial_right_packed": np.packbits(np.asarray(right_initial, dtype=np.uint8), bitorder="little"),
        "burn_left_packed": np.packbits(burn_left, bitorder="little"),
        "burn_right_packed": np.packbits(burn_right, bitorder="little"),
        "final_left_packed": np.packbits(pair.left, bitorder="little"),
        "final_right_packed": np.packbits(pair.right, bitorder="little"),
        "measurement_left_states_packed": left_packed,
        "measurement_right_states_packed": right_packed,
        "measurement_left_labels": left_labels,
        "measurement_right_labels": right_labels,
        "measurement_left_weights": left_weights,
        "measurement_right_weights": right_weights,
        "measurement_component_counts": component_counts,
        "measurement_new_unordered_pair": new_pair_flags,
        "measurement_whole_pair_exchange": whole_swap_flags,
        "measurement_residual_weights": residuals.sum(axis=1).astype(np.int32),
        "counters": counters,
        "kernel": HOUDAYER_PAIR_KERNEL,
    }
