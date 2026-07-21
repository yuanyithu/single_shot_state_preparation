"""Global-mixing q=0 samplers for the isolated exp102 discovery.

The algorithms in this module target the hard coset posterior

    pi(e | y) proportional to exp(-K_p |e|),  H_Z e = y.

They deliberately do not share a raw schema, seed namespace, or certification
path with the exhausted PT and PA discoveries.  Python is the reference
implementation; Numba consumes the same PortablePrng stream.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import math

import numpy as np

from .exp101_bridge import load_exp101
from .io import canonical_json, sha256_json
from .labels import bits_to_uint64
from .q0_pa import validate_hard_coset_basis
from .q0_pt import coupling
from .seeds import derive_seed

try:
    from numba import njit
except ImportError:  # pragma: no cover - Linux discovery preflight requires Numba
    njit = None


GLOBAL_DISCOVERY_VERSION = "exp102.q0_global.discovery.v1"
HARD_COSET_RAW_VERSION = "exp102.q0_hardcoset.raw.v1"
DEFECT_TRACE_RAW_VERSION = "exp102.q0_defect_trace.raw.v1"
DEFECT_BIAS_RAW_VERSION = "exp102.q0_defect_bias.raw.v1"
CATALOG_VERSION = "exp102.q0_logical_catalog.v1"
JOINT_BLOCK_VERSION = "exp102.q0_joint_blocks.v1"
CHARACTER_SET_VERSION = "exp102.q0_characters.v1"

HARD_METHODS = ("RC8-QC1", "RC8-QC4", "RC8-J08", "RC8-J12", "RC8-J16")
DEFECT_METHODS = ("DT16", "DT32", "DT64")
COUNTER_NAMES = (
    "stabilizer_attempts",
    "stabilizer_changes",
    "catalog_attempts",
    "catalog_changes",
    "cluster_attempts",
    "cluster_nonzero",
    "cluster_nullity_sum",
    "cluster_changed_bits",
    "joint_attempts",
    "joint_changes",
)
DEFECT_COUNTER_NAMES = (
    "bit_attempts", "bit_changes", "leave_events", "return_events",
)


class GlobalConflictError(ValueError):
    """An identity, algebraic, replay, or transcript conflict."""


def _sha256_arrays(version, arrays, scalars=()):
    digest = hashlib.sha256()
    digest.update(str(version).encode("ascii") + b"\0")
    for value in scalars:
        digest.update(str(value).encode("ascii") + b"\0")
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(contiguous.dtype.str.encode("ascii") + b"\0")
        digest.update(np.asarray(contiguous.shape, dtype=">u8").tobytes())
        digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _pack_rows(rows):
    return np.packbits(
        np.asarray(rows, dtype=np.uint8), axis=1, bitorder="little",
    )


def _rows_to_csr(rows):
    supports = [np.flatnonzero(row).astype(np.int32) for row in rows]
    offsets = np.zeros(len(supports) + 1, dtype=np.int64)
    for index, support in enumerate(supports):
        offsets[index + 1] = offsets[index] + support.size
    indices = np.empty(int(offsets[-1]), dtype=np.int32)
    for index, support in enumerate(supports):
        indices[offsets[index]:offsets[index + 1]] = support
    return supports, np.ascontiguousarray(indices), np.ascontiguousarray(offsets)


def _signature_from_bits(bits):
    return bits_to_uint64(np.asarray(bits, dtype=np.uint8))


def qubit_signatures(frame):
    """Return the uint64 logical signature of every physical coordinate."""
    if frame.k > 64:
        raise ValueError("global discovery supports at most 64 logical bits")
    signatures = np.zeros(frame.num_qubits, dtype=np.uint64)
    for bit in range(frame.k):
        signatures[frame.W_basis[bit].astype(bool)] ^= (
            np.uint64(1) << np.uint64(bit)
        )
    return signatures


def _signature_rank_masks(masks, k):
    """Deterministic GF(2) rank of uint64 masks, including bit 63."""
    pivots = {}
    for raw in masks:
        value = int(np.uint64(raw))
        while value:
            pivot = value.bit_length() - 1
            if pivot not in pivots:
                pivots[pivot] = value
                break
            value ^= pivots[pivot]
    rank = len(pivots)
    if rank > int(k):
        raise AssertionError("signature rank exceeds logical dimension")
    return rank


def reduce_logical_basis(logical_rows):
    """Locally reduce a logical basis with a fully deterministic row rule.

    At each step, choose the ordered pair ``(i,j)`` giving the largest strict
    decrease of the total Hamming weight under ``row_i ^= row_j``.  Ties use
    lexicographic ``(i,j)`` order.  This is code-matrix-only preprocessing.
    """
    rows = np.ascontiguousarray(logical_rows, dtype=np.uint8).copy()
    if rows.ndim != 2:
        raise ValueError("logical basis must be two-dimensional")
    while True:
        best = None
        for i in range(rows.shape[0]):
            old_weight = int(rows[i].sum())
            for j in range(rows.shape[0]):
                if i == j:
                    continue
                improvement = old_weight - int(np.count_nonzero(rows[i] ^ rows[j]))
                if improvement <= 0:
                    continue
                candidate = (-improvement, i, j)
                if best is None or candidate < best:
                    best = candidate
        if best is None:
            break
        _, i, j = best
        rows[i] ^= rows[j]
    return rows


@dataclass(frozen=True)
class LogicalProposalCatalog:
    moves: np.ndarray
    signatures: np.ndarray
    weights: np.ndarray
    reduced_basis: np.ndarray
    support_indices: np.ndarray
    support_offsets: np.ndarray
    catalog_sha256: str

    @property
    def size(self):
        return int(self.moves.shape[0])


def build_logical_proposal_catalog(model, frame, *, max_multiple=8, max_count=512):
    """Build the frozen reduced single/pair/triple logical proposal catalog."""
    load_exp101()
    from exp101_certified_src.gf2 import gf2_matmul, gf2_rank

    if not 0 <= model.k <= 64 or frame.k != model.k:
        raise ValueError("logical dimensions are incompatible with uint64 catalog")
    reduced = reduce_logical_basis(model.logical_move_basis)
    if int(gf2_rank(reduced)) != model.k:
        raise GlobalConflictError("logical local reduction changed basis rank")

    # The packed support is both the deduplication key and final sort key.
    candidates = {}
    for order in (1, 2, 3):
        for combination in itertools.combinations(range(model.k), order):
            move = np.bitwise_xor.reduce(reduced[list(combination)], axis=0)
            packed = np.packbits(move, bitorder="little").tobytes()
            if packed in candidates:
                continue
            signature_bits = gf2_matmul(frame.W_basis, move[:, None])[:, 0]
            signature = _signature_from_bits(signature_bits)
            candidates[packed] = (
                int(move.sum()), int(signature), packed, move.copy(), signature,
            )
    ordered = sorted(candidates.values(), key=lambda value: value[:3])
    target = min(int(max_multiple) * model.k, int(max_count), len(ordered))
    if model.k and target < model.k:
        raise GlobalConflictError("catalog capacity is smaller than logical rank")

    selected = []
    selected_set = set()
    pivot_rows = {}
    for index, value in enumerate(ordered):
        residue = int(value[4])
        while residue:
            pivot = residue.bit_length() - 1
            if pivot not in pivot_rows:
                pivot_rows[pivot] = residue
                selected.append(index)
                selected_set.add(index)
                break
            residue ^= pivot_rows[pivot]
        if len(selected) == model.k:
            break
    if len(selected) != model.k:
        raise GlobalConflictError("catalog signatures do not span all logical sectors")
    for index in range(len(ordered)):
        if len(selected) >= target:
            break
        if index not in selected_set:
            selected.append(index)
            selected_set.add(index)

    retained = [ordered[index] for index in selected]
    moves = np.ascontiguousarray([value[3] for value in retained], dtype=np.uint8)
    signatures = np.ascontiguousarray([value[4] for value in retained], dtype=np.uint64)
    weights = np.ascontiguousarray([value[0] for value in retained], dtype=np.int32)
    _, indices, offsets = _rows_to_csr(moves)
    if gf2_matmul(model.H_check, moves.T).any():
        raise GlobalConflictError("catalog contains a non-kernel proposal")
    recomputed = np.asarray([
        _signature_from_bits(gf2_matmul(frame.W_basis, row[:, None])[:, 0])
        for row in moves
    ], dtype=np.uint64)
    if not np.array_equal(signatures, recomputed):
        raise GlobalConflictError("catalog signature does not equal W d")
    if _signature_rank_masks(signatures, model.k) != model.k:
        raise GlobalConflictError("retained catalog signature rank is incomplete")
    if np.unique(_pack_rows(moves), axis=0).shape[0] != moves.shape[0]:
        raise GlobalConflictError("catalog contains duplicate supports")

    digest = _sha256_arrays(
        CATALOG_VERSION,
        (np.asarray([model.num_qubits, model.k], dtype=">u8"),
         _pack_rows(reduced), _pack_rows(moves), signatures.astype(">u8"), weights.astype(">i4")),
        (max_multiple, max_count),
    )
    return LogicalProposalCatalog(
        moves=moves,
        signatures=signatures,
        weights=weights,
        reduced_basis=reduced,
        support_indices=indices,
        support_offsets=offsets,
        catalog_sha256=digest,
    )


@dataclass(frozen=True)
class JointBlockCatalog:
    generators: np.ndarray
    signatures: np.ndarray
    support_indices: np.ndarray
    support_offsets: np.ndarray
    block_size: int
    joint_sha256: str

    @property
    def num_blocks(self):
        return int(self.generators.shape[0])


def build_joint_blocks(model, frame, catalog, block_size):
    """Build state-independent logical/stabilizer heatbath blocks."""
    load_exp101()
    from exp101_certified_src.gf2 import gf2_rank

    block_size = int(block_size)
    if block_size not in (8, 12, 16):
        raise ValueError("joint block size must be 8, 12, or 16")
    blocks = []
    block_signatures = []
    for logical in catalog.reduced_basis:
        overlaps = np.count_nonzero(model.stabilizer_rows & logical[None, :], axis=1)
        order = sorted(range(model.stabilizer_rows.shape[0]), key=lambda i: (-int(overlaps[i]), i))
        rows = [logical.copy()]
        for index in order:
            candidate = model.stabilizer_rows[index]
            trial = np.asarray([*rows, candidate], dtype=np.uint8)
            if int(gf2_rank(trial)) == len(rows) + 1:
                rows.append(candidate.copy())
            if len(rows) == block_size:
                break
        if len(rows) != block_size:
            raise GlobalConflictError("could not construct a full-rank joint block")
        rows = np.ascontiguousarray(rows, dtype=np.uint8)
        blocks.append(rows)
        signatures = np.zeros(block_size, dtype=np.uint64)
        for index, row in enumerate(rows):
            signatures[index] = _signature_from_bits(
                (frame.W_basis.astype(np.int64) @ row.astype(np.int64) % 2).astype(np.uint8)
            )
        block_signatures.append(signatures)
    generators = np.ascontiguousarray(blocks, dtype=np.uint8)
    signatures = np.ascontiguousarray(block_signatures, dtype=np.uint64)
    flat = generators.reshape(-1, model.num_qubits)
    _, indices, offsets = _rows_to_csr(flat)
    digest = _sha256_arrays(
        JOINT_BLOCK_VERSION,
        (_pack_rows(generators.reshape(-1, model.num_qubits)), signatures.astype(">u8")),
        (catalog.catalog_sha256, block_size),
    )
    return JointBlockCatalog(
        generators=generators,
        signatures=signatures,
        support_indices=indices,
        support_offsets=offsets,
        block_size=block_size,
        joint_sha256=digest,
    )


@dataclass(frozen=True)
class CharacterSet:
    masks: np.ndarray
    basis_positions: np.ndarray
    tier: str
    k: int
    random_seed: int | None
    character_sha256: str


def frozen_character_set(k, seed, num_nonbasis=4096):
    """Select full or basis-plus-PortablePrng nonbasis uint64 characters."""
    load_exp101()
    from exp101_certified_src.prng import PortablePrng

    k = int(k)
    if not 0 < k <= 64:
        raise ValueError("character selection requires 1<=k<=64")
    basis = [np.uint64(1) << np.uint64(bit) for bit in range(k)]
    if k <= 10:
        masks = np.arange(1, 1 << k, dtype=np.uint64)
        tier = "full"
        random_seed = None
    else:
        available = (1 << k) - 1 - k
        count = min(int(num_nonbasis), available)
        rng = PortablePrng(int(seed))
        basis_int = {int(value) for value in basis}
        chosen = set()
        limit_mask = (1 << k) - 1 if k < 64 else (1 << 64) - 1
        while len(chosen) < count:
            candidate = rng.next_uint64() & limit_mask
            if candidate and candidate not in basis_int:
                chosen.add(candidate)
        masks = np.asarray([*basis, *(np.uint64(v) for v in sorted(chosen))], dtype=np.uint64)
        tier = "sampled"
        random_seed = int(seed)
    basis_positions = np.asarray([
        int(np.flatnonzero(masks == value)[0]) for value in basis
    ], dtype=np.int32)
    digest = _sha256_arrays(
        CHARACTER_SET_VERSION,
        (masks.astype(">u8"), basis_positions.astype(">i4")),
        (k, tier, "none" if random_seed is None else random_seed, num_nonbasis),
    )
    return CharacterSet(masks, basis_positions, tier, k, random_seed, digest)


def character_values(labels, masks):
    """Evaluate (-1)^(u.label) without an int64 conversion at k=64."""
    for name, values in (("labels", labels), ("masks", masks)):
        candidate = None
        if isinstance(values, np.ndarray) and values.dtype.kind == "f":
            candidate = values
        elif isinstance(values, (list, tuple)):
            floating = [
                value for value in values
                if isinstance(value, (float, np.floating))
            ]
            if floating:
                candidate = np.asarray(floating, dtype=np.float64)
        if candidate is not None:
            if (not np.all(np.isfinite(candidate))
                    or np.any(candidate != np.floor(candidate))
                    or np.any(np.abs(candidate) > 2**53)):
                raise ValueError(
                    f"floating {name} cannot preserve exact uint64 bits"
                )
    labels = (
        np.asarray(labels, dtype=np.uint64).reshape(-1)
        if isinstance(labels, np.ndarray)
        else np.fromiter((int(value) for value in labels), dtype=np.uint64)
    )
    masks = (
        np.asarray(masks, dtype=np.uint64).reshape(-1)
        if isinstance(masks, np.ndarray)
        else np.fromiter((int(value) for value in masks), dtype=np.uint64)
    )
    result = np.empty((labels.size, masks.size), dtype=np.int8)
    # Chunking bounds the temporary uint64 outer product for T3/k=64 raw.
    for start in range(0, masks.size, 128):
        stop = min(start + 128, masks.size)
        parity = np.bitwise_count(
            labels[:, None] & masks[None, start:stop]
        ) & np.uint8(1)
        result[:, start:stop] = 1 - 2 * parity.astype(np.int8)
    return result


def character_means(label_traces, masks, valid_masks=None):
    traces = [np.asarray(trace, dtype=np.uint64) for trace in label_traces]
    if valid_masks is None:
        valid_masks = [np.ones(trace.size, dtype=bool) for trace in traces]
    means = np.empty((len(traces), len(masks)), dtype=np.float64)
    counts = np.empty(len(traces), dtype=np.int64)
    for index, (trace, valid) in enumerate(zip(traces, valid_masks)):
        valid = np.asarray(valid, dtype=bool)
        if valid.shape != trace.shape or not valid.any():
            raise ValueError("each trajectory needs at least one valid character observation")
        selected = trace[valid]
        for start in range(0, len(masks), 128):
            stop = min(start + 128, len(masks))
            parity = np.bitwise_count(
                selected[:, None] & np.asarray(masks[start:stop], dtype=np.uint64)[None, :]
            ) & np.uint8(1)
            means[index, start:stop] = 1.0 - 2.0 * parity.mean(axis=0)
        counts[index] = int(valid.sum())
    return means, counts


def _u_statistic_squares(means):
    means = np.asarray(means, dtype=np.float64)
    chains = means.shape[0]
    if means.ndim != 2 or chains < 2:
        raise ValueError("independent trajectory U-statistic needs at least two chains")
    sums = means.sum(axis=0)
    return (sums * sums - np.square(means).sum(axis=0)) / (chains * (chains - 1))


def _character_population_mean(character_set, values):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != character_set.masks.shape:
        raise ValueError("character estimate shape mismatch")
    total = (1 << character_set.k) - 1
    if character_set.tier == "full":
        return float(values.mean()), 0.0
    basis_mask = np.zeros(values.size, dtype=bool)
    basis_mask[character_set.basis_positions] = True
    sampled = values[~basis_mask]
    remaining = total - character_set.k
    estimate = (float(values[basis_mask].sum()) + remaining * float(sampled.mean())) / total
    if sampled.size <= 1 or remaining <= 1 or sampled.size == remaining:
        finite_se = 0.0
    else:
        fraction = sampled.size / remaining
        finite_se = (
            remaining / total
            * math.sqrt((1.0 - fraction) * float(sampled.var(ddof=1)) / sampled.size)
        )
    return float(estimate), float(finite_se)


def character_qtop_estimate(character_set, means):
    """Debiased q_top with delete-one-trajectory and character sampling SE."""
    means = np.asarray(means, dtype=np.float64)
    per_character = _u_statistic_squares(means)
    estimate, character_se = _character_population_mean(character_set, per_character)
    delete = []
    if means.shape[0] >= 3:
        for omitted in range(means.shape[0]):
            value, _ = _character_population_mean(
                character_set, _u_statistic_squares(np.delete(means, omitted, axis=0)),
            )
            delete.append(value)
    delete = np.asarray(delete, dtype=np.float64)
    if delete.size:
        trajectory_se = math.sqrt(
            (delete.size - 1) / delete.size * float(np.square(delete - delete.mean()).sum())
        )
    else:
        trajectory_se = float("nan")
    total_se = math.sqrt(trajectory_se**2 + character_se**2)
    return {
        "q_top": estimate,
        "q_top_trajectory_se": float(trajectory_se),
        "q_top_character_se": character_se,
        "q_top_total_se": float(total_se),
        "per_character_m2": per_character,
        "delete_one_q_top": delete,
    }


def character_d2_estimate(character_set, means_a, means_b):
    """Unbiased normalized L2 distance between two trajectory ensembles."""
    means_a = np.asarray(means_a, dtype=np.float64)
    means_b = np.asarray(means_b, dtype=np.float64)
    if means_a.shape[1:] != means_b.shape[1:]:
        raise ValueError("D2 character dimensions differ")

    def per_character(left, right):
        return (
            _u_statistic_squares(left) + _u_statistic_squares(right)
            - 2.0 * left.mean(axis=0) * right.mean(axis=0)
        )

    per_character_values = per_character(means_a, means_b)
    estimate, character_se = _character_population_mean(
        character_set, per_character_values,
    )
    delete_by_side = []
    for side in (0, 1):
        source = means_a if side == 0 else means_b
        side_delete = []
        if source.shape[0] < 3:
            delete_by_side.append(np.empty(0, dtype=np.float64))
            continue
        for omitted in range(source.shape[0]):
            left = np.delete(means_a, omitted, axis=0) if side == 0 else means_a
            right = np.delete(means_b, omitted, axis=0) if side == 1 else means_b
            side_delete.append(_character_population_mean(
                character_set, per_character(left, right),
            )[0])
        delete_by_side.append(np.asarray(side_delete, dtype=np.float64))
    delete = np.concatenate(delete_by_side)
    if any(value.size for value in delete_by_side):
        variance = 0.0
        for side_values in delete_by_side:
            if side_values.size:
                variance += (
                    (side_values.size - 1) / side_values.size
                    * float(np.square(side_values - side_values.mean()).sum())
                )
        trajectory_se = math.sqrt(variance)
    else:
        trajectory_se = float("nan")
    total_se = math.sqrt(trajectory_se**2 + character_se**2)
    return {
        "d2_norm": float(estimate),
        "d2_trajectory_se": float(trajectory_se),
        "d2_character_se": float(character_se),
        "d2_total_se": float(total_se),
        "per_character_d2": per_character_values,
        "delete_one_d2": delete,
    }


def label_collision_diagnostic(label_traces, k, valid_masks=None):
    """Cross-trajectory raw-label collision diagnostic.

    This intentionally remains secondary to the frozen character estimator.
    Each trajectory contributes a normalized empirical label distribution and
    only cross-trajectory products enter the collision mass.
    """
    traces = [np.asarray(trace, dtype=np.uint64).reshape(-1) for trace in label_traces]
    if len(traces) < 2:
        raise ValueError("label collision diagnostic needs at least two trajectories")
    if valid_masks is None:
        valid_masks = [np.ones(trace.size, dtype=bool) for trace in traces]
    label_parts = []
    probability_parts = []
    for trace, valid in zip(traces, valid_masks):
        valid = np.asarray(valid, dtype=bool)
        if valid.shape != trace.shape or not valid.any():
            raise ValueError("label collision diagnostic has no valid observations")
        labels, counts = np.unique(trace[valid], return_counts=True)
        probabilities = counts.astype(np.float64) / float(valid.sum())
        label_parts.append(labels)
        probability_parts.append(probabilities)
    labels = np.concatenate(label_parts)
    probabilities = np.concatenate(probability_parts)
    _, inverse = np.unique(labels, return_inverse=True)
    summed = np.bincount(inverse, weights=probabilities)
    squared = np.bincount(inverse, weights=probabilities * probabilities)
    trajectories = len(traces)
    collision = float(
        np.sum(summed * summed - squared) / (trajectories * (trajectories - 1))
    )
    uniform = 2.0 ** (-int(k))
    return {
        "collision_mass": collision,
        "q_top": (collision - uniform) / (1.0 - uniform),
    }


def trajectory_mean_and_se(values, normalization=1.0):
    values = np.asarray(values, dtype=np.float64) / float(normalization)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("trajectory means require at least two values")
    return float(values.mean()), float(values.std(ddof=1) / math.sqrt(values.size))


@dataclass(frozen=True)
class GlobalSeedIdentity:
    source_commit: str
    config_sha256: str
    registry_sha256: str
    cell_fingerprint: str
    method_id: str
    resource_tier: str
    init_family: str
    trajectory_index: int
    trajectory_namespace: str

    def __post_init__(self):
        if len(self.source_commit) != 40 or any(c not in "0123456789abcdef" for c in self.source_commit):
            raise ValueError("global source commit must be a full lowercase Git SHA")
        for name in ("config_sha256", "registry_sha256", "cell_fingerprint"):
            value = getattr(self, name)
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"global {name} must be a lowercase SHA256")
        if self.init_family not in ("P", "U", "TUNE"):
            raise ValueError("unknown global initialization family")
        if isinstance(self.trajectory_index, bool) or int(self.trajectory_index) < 0:
            raise ValueError("global trajectory index is invalid")

    def seed(self, stage, role="stream", index=0):
        return derive_seed(
            "q0_global_discovery_v1",
            self.source_commit,
            self.config_sha256,
            self.registry_sha256,
            self.cell_fingerprint,
            self.method_id,
            self.resource_tier,
            self.init_family,
            int(self.trajectory_index),
            self.trajectory_namespace,
            str(stage),
            str(role),
            int(index),
        )

    def as_dict(self):
        return {
            "source_commit": self.source_commit,
            "config_sha256": self.config_sha256,
            "registry_sha256": self.registry_sha256,
            "cell_fingerprint": self.cell_fingerprint,
            "method_id": self.method_id,
            "resource_tier": self.resource_tier,
            "init_family": self.init_family,
            "trajectory_index": int(self.trajectory_index),
            "trajectory_namespace": self.trajectory_namespace,
        }


@dataclass(frozen=True)
class HardCosetConfig:
    method_id: str
    p: float
    burn_sweeps: int
    measurement_sweeps: int

    def __post_init__(self):
        if self.method_id not in HARD_METHODS:
            raise ValueError("unknown hard-coset method")
        if not 0.0 < float(self.p) < 0.5:
            raise ValueError("hard-coset p must lie in (0,0.5)")
        for field in ("burn_sweeps", "measurement_sweeps"):
            value = getattr(self, field)
            if isinstance(value, bool) or int(value) <= 0:
                raise ValueError(f"{field} must be a positive integer")
        if int(self.measurement_sweeps) % 8:
            raise ValueError("measurement_sweeps must divide into eight time blocks")

    @property
    def cluster_repeats(self):
        return 1 if self.method_id == "RC8-QC1" else 4 if self.method_id == "RC8-QC4" else 0

    @property
    def joint_block_size(self):
        return int(self.method_id[-2:]) if "-J" in self.method_id else 0

    def as_dict(self):
        return {
            "method_id": self.method_id,
            "p": float(self.p),
            "burn_sweeps": int(self.burn_sweeps),
            "measurement_sweeps": int(self.measurement_sweeps),
            "cluster_repeats": self.cluster_repeats,
            "joint_block_size": self.joint_block_size,
        }


@dataclass(frozen=True)
class DefectTraceConfig:
    method_id: str
    p: float
    burn_sweeps: int
    measurement_sweeps: int
    tuning_chains: int = 8
    tuning_sweeps: int = 4096

    def __post_init__(self):
        if self.method_id not in DEFECT_METHODS:
            raise ValueError("unknown defect-trace method")
        if not 0.0 < float(self.p) < 0.5:
            raise ValueError("defect-trace p must lie in (0,0.5)")
        if int(self.tuning_chains) != 8 or int(self.tuning_sweeps) != 4096:
            raise ValueError("defect bias tuning is frozen at 8x4096")
        if int(self.burn_sweeps) <= 0 or int(self.measurement_sweeps) <= 0:
            raise ValueError("defect trace sweep counts must be positive")
        if int(self.measurement_sweeps) % 8:
            raise ValueError("measurement_sweeps must divide into eight time blocks")

    @property
    def dmax(self):
        return int(self.method_id[2:])

    def as_dict(self):
        return {
            "method_id": self.method_id,
            "p": float(self.p),
            "burn_sweeps": int(self.burn_sweeps),
            "measurement_sweeps": int(self.measurement_sweeps),
            "tuning_chains": int(self.tuning_chains),
            "tuning_sweeps": int(self.tuning_sweeps),
            "dmax": self.dmax,
            "K_q": 0.0,
        }


def uniform_hard_coset_state(model, syndrome, seed):
    """Draw exactly uniformly from the affine hard coset."""
    load_exp101()
    from exp101_certified_src.prng import PortablePrng

    validate_hard_coset_basis(model)
    state = model.logical_sector_section.apply(syndrome, strict=True).astype(np.uint8)
    rng = PortablePrng(int(seed))
    for row in model.stabilizer_rows:
        if rng.randbelow(2):
            state ^= row
    for row in model.logical_move_basis:
        if rng.randbelow(2):
            state ^= row
    return np.ascontiguousarray(state)


def state_label(frame, state):
    return _signature_from_bits(frame.label_of(np.asarray(state, dtype=np.uint8)))


def pack_state(state):
    return np.packbits(np.asarray(state, dtype=np.uint8), bitorder="little")


def unpack_states(packed, num_qubits):
    return np.unpackbits(
        np.asarray(packed, dtype=np.uint8), axis=-1, count=int(num_qubits),
        bitorder="little",
    ).astype(np.uint8, copy=False)


def _column_check_csr(model):
    offsets = np.zeros(model.num_qubits + 1, dtype=np.int64)
    for qubit, checks in enumerate(model.checks_touching_each_qubit):
        offsets[qubit + 1] = offsets[qubit] + len(checks)
    indices = np.empty(int(offsets[-1]), dtype=np.int32)
    for qubit, checks in enumerate(model.checks_touching_each_qubit):
        indices[offsets[qubit]:offsets[qubit + 1]] = checks
    return np.ascontiguousarray(indices), np.ascontiguousarray(offsets)


def _heatbath_table(K, num_qubits):
    result = np.empty(2 * int(num_qubits) + 1, dtype=np.float64)
    for delta in range(-int(num_qubits), int(num_qubits) + 1):
        x = float(K) * delta
        exp_negative = math.exp(-abs(x))
        result[delta + int(num_qubits)] = (
            exp_negative / (1.0 + exp_negative)
            if x >= 0.0 else 1.0 / (1.0 + exp_negative)
        )
    return result


def _support_delta(state, support):
    return int(support.size) - 2 * int(state[support].sum())


def _toggle_support(state, support):
    delta = _support_delta(state, support)
    state[support] ^= 1
    return delta


def _reference_cluster_move(state, label, rng, b_value, check_indices,
                            check_offsets, qubit_signatures, num_checks):
    """Reference rejection-free pin-and-kernel cluster update."""
    n = state.size
    free = []
    pin_probability = 1.0 - float(b_value)
    for qubit in range(n):
        if state[qubit] or rng.random() >= pin_probability:
            free.append(qubit)
    num_free = len(free)
    if num_free == 0:
        return label, 0, 0
    words = (num_free + 63) // 64
    rows = np.zeros((int(num_checks), words), dtype=np.uint64)
    for column, qubit in enumerate(free):
        word = column // 64
        mask = np.uint64(1) << np.uint64(column % 64)
        for position in range(check_offsets[qubit], check_offsets[qubit + 1]):
            rows[check_indices[position], word] |= mask

    rank = 0
    pivots = []
    for column in range(num_free):
        word = column // 64
        mask = np.uint64(1) << np.uint64(column % 64)
        pivot = None
        for row in range(rank, rows.shape[0]):
            if rows[row, word] & mask:
                pivot = row
                break
        if pivot is None:
            continue
        if pivot != rank:
            rows[[rank, pivot]] = rows[[pivot, rank]]
        for row in range(rows.shape[0]):
            if row != rank and rows[row, word] & mask:
                rows[row] ^= rows[rank]
        pivots.append(column)
        rank += 1
        if rank == rows.shape[0]:
            break

    pivot_mask = np.zeros(num_free, dtype=bool)
    pivot_mask[pivots] = True
    sampled = np.zeros(words, dtype=np.uint64)
    for column in range(num_free):
        if not pivot_mask[column] and rng.randbelow(2):
            sampled[column // 64] |= np.uint64(1) << np.uint64(column % 64)
    for row, pivot in enumerate(pivots):
        parity = 0
        for word in range(words):
            parity ^= (int(rows[row, word] & sampled[word]).bit_count() & 1)
        if parity:
            sampled[pivot // 64] |= np.uint64(1) << np.uint64(pivot % 64)

    changed = 0
    for column, qubit in enumerate(free):
        if sampled[column // 64] & (np.uint64(1) << np.uint64(column % 64)):
            state[qubit] ^= 1
            label ^= qubit_signatures[qubit]
            changed += 1
    return label, num_free - rank, changed


def _reference_joint_move(state, label, weight, rng, block, signatures,
                          boltzmann_delta):
    block_size = block.shape[0]
    categories = 1 << block_size
    energies = np.empty(categories, dtype=np.int32)
    energies[0] = weight
    previous_gray = 0
    scratch_weight = int(weight)
    for enumeration in range(1, categories):
        gray = enumeration ^ (enumeration >> 1)
        changed = gray ^ previous_gray
        position = (changed & -changed).bit_length() - 1
        scratch_weight += _toggle_support(state, np.flatnonzero(block[position]))
        energies[gray] = scratch_weight
        previous_gray = gray
    for position in range(block_size):
        if (previous_gray >> position) & 1:
            state[np.flatnonzero(block[position])] ^= 1
    minimum = int(energies.min())
    total = 0.0
    for category in range(categories):
        total += float(boltzmann_delta[int(energies[category]) - minimum])
    threshold = rng.random() * total
    cumulative = 0.0
    selected = categories - 1
    for category in range(categories):
        cumulative += float(boltzmann_delta[int(energies[category]) - minimum])
        if threshold < cumulative:
            selected = category
            break
    for position in range(block_size):
        if (selected >> position) & 1:
            state[np.flatnonzero(block[position])] ^= 1
            label ^= signatures[position]
    return label, int(energies[selected]), int(selected != 0)


def _update_seen(seen, label):
    for bit in range(seen.shape[0]):
        sign_index = int((np.uint64(label) >> np.uint64(bit)) & np.uint64(1))
        seen[bit, sign_index] = 1


def _run_hard_stage_reference(state, label, config, model, catalog, joint,
                              seed, sweeps, record):
    load_exp101()
    from exp101_certified_src.prng import PortablePrng

    rng = PortablePrng(int(seed))
    K = coupling(config.p)
    heatbath = _heatbath_table(K, model.num_qubits)
    boltzmann = np.exp(-K * np.arange(model.num_qubits + 1, dtype=np.float64))
    stabilizer_supports, _, _ = _rows_to_csr(model.stabilizer_rows)
    catalog_supports, _, _ = _rows_to_csr(catalog.moves)
    check_indices, check_offsets = _column_check_csr(model)
    signatures_by_qubit = qubit_signatures(
        type("FrameProxy", (), {
            "k": model.k, "num_qubits": model.num_qubits,
            "W_basis": model._global_frame_W,
        })
    )
    counters = np.zeros(len(COUNTER_NAMES), dtype=np.int64)
    seen = np.zeros((model.k, 2), dtype=np.uint8)
    _update_seen(seen, label)
    bytes_per_state = (model.num_qubits + 7) // 8
    packed = np.empty((sweeps if record else 0, bytes_per_state), dtype=np.uint8)
    labels = np.empty(sweeps if record else 0, dtype=np.uint64)
    weights = np.empty(sweeps if record else 0, dtype=np.int32)
    weight = int(state.sum())
    b_value = float(config.p) / (1.0 - float(config.p))
    for sweep in range(int(sweeps)):
        for coordinate in rng.permutation(len(stabilizer_supports)):
            support = stabilizer_supports[int(coordinate)]
            delta = _support_delta(state, support)
            counters[0] += 1
            if rng.random() < heatbath[delta + model.num_qubits]:
                state[support] ^= 1
                weight += delta
                counters[1] += 1
        for coordinate in rng.permutation(len(catalog_supports)):
            index = int(coordinate)
            support = catalog_supports[index]
            delta = _support_delta(state, support)
            counters[2] += 1
            if rng.random() < heatbath[delta + model.num_qubits]:
                state[support] ^= 1
                weight += delta
                label ^= catalog.signatures[index]
                counters[3] += 1
        for _ in range(config.cluster_repeats):
            counters[4] += 1
            label, nullity, changed = _reference_cluster_move(
                state, label, rng, b_value, check_indices, check_offsets,
                signatures_by_qubit, model.num_checks,
            )
            counters[5] += int(changed > 0)
            counters[6] += nullity
            counters[7] += changed
            if changed:
                weight = int(state.sum())
        if config.joint_block_size:
            block_index = rng.randbelow(joint.num_blocks)
            counters[8] += 1
            label, weight, changed = _reference_joint_move(
                state, label, weight, rng, joint.generators[block_index],
                joint.signatures[block_index], boltzmann,
            )
            counters[9] += changed
        _update_seen(seen, label)
        if record:
            packed[sweep] = pack_state(state)
            labels[sweep] = label
            weights[sweep] = weight
    return state, np.uint64(label), weight, packed, labels, weights, counters, seen


if njit is not None:
    @njit(cache=True, inline="always")
    def _global_nb_next(state):
        x = state[0]
        y = state[1]
        state[0] = y
        x = x ^ (x << np.uint64(23))
        x = x ^ (x >> np.uint64(17))
        x = x ^ y ^ (y >> np.uint64(26))
        state[1] = x
        return x + y


    @njit(cache=True, inline="always")
    def _global_nb_random(state):
        return float(_global_nb_next(state) >> np.uint64(11)) * (1.0 / 9007199254740992.0)


    @njit(cache=True, inline="always")
    def _global_nb_randbelow(state, n):
        return int(_global_nb_next(state) % np.uint64(n))


    @njit(cache=True, inline="always")
    def _global_nb_permutation(state, buffer):
        for index in range(buffer.size):
            buffer[index] = index
        for index in range(buffer.size - 1, 0, -1):
            selected = _global_nb_randbelow(state, index + 1)
            temporary = buffer[index]
            buffer[index] = buffer[selected]
            buffer[selected] = temporary


    @njit(cache=True, inline="always")
    def _global_nb_support_delta(state, indices, start, stop):
        ones = 0
        for position in range(start, stop):
            ones += int(state[indices[position]])
        return (stop - start) - 2 * ones


    @njit(cache=True, inline="always")
    def _global_nb_toggle(state, indices, start, stop):
        delta = _global_nb_support_delta(state, indices, start, stop)
        for position in range(start, stop):
            state[indices[position]] ^= np.uint8(1)
        return delta


    @njit(cache=True, inline="always")
    def _global_nb_parity(value):
        parity = 0
        while value:
            parity ^= 1
            value &= value - np.uint64(1)
        return parity


    @njit(cache=True)
    def _global_nb_cluster(state, label, rng_state, b_value, check_indices,
                           check_offsets, qubit_sigs, num_checks):
        n = state.size
        free = np.empty(n, dtype=np.int32)
        num_free = 0
        pin_probability = 1.0 - b_value
        for qubit in range(n):
            if state[qubit] or _global_nb_random(rng_state) >= pin_probability:
                free[num_free] = qubit
                num_free += 1
        if num_free == 0:
            return label, 0, 0
        words = (num_free + 63) // 64
        rows = np.zeros((num_checks, words), dtype=np.uint64)
        for column in range(num_free):
            qubit = free[column]
            word = column // 64
            mask = np.uint64(1) << np.uint64(column % 64)
            for position in range(check_offsets[qubit], check_offsets[qubit + 1]):
                rows[check_indices[position], word] |= mask
        pivots = np.empty(min(num_checks, num_free), dtype=np.int32)
        rank = 0
        for column in range(num_free):
            word = column // 64
            mask = np.uint64(1) << np.uint64(column % 64)
            pivot = -1
            for row in range(rank, num_checks):
                if rows[row, word] & mask:
                    pivot = row
                    break
            if pivot < 0:
                continue
            if pivot != rank:
                for w in range(words):
                    temporary = rows[rank, w]
                    rows[rank, w] = rows[pivot, w]
                    rows[pivot, w] = temporary
            for row in range(num_checks):
                if row != rank and rows[row, word] & mask:
                    for w in range(words):
                        rows[row, w] ^= rows[rank, w]
            pivots[rank] = column
            rank += 1
            if rank == num_checks:
                break
        pivot_mask = np.zeros(num_free, dtype=np.uint8)
        for row in range(rank):
            pivot_mask[pivots[row]] = 1
        sampled = np.zeros(words, dtype=np.uint64)
        for column in range(num_free):
            if not pivot_mask[column] and _global_nb_randbelow(rng_state, 2):
                sampled[column // 64] |= np.uint64(1) << np.uint64(column % 64)
        for row in range(rank):
            parity = 0
            for word in range(words):
                parity ^= _global_nb_parity(rows[row, word] & sampled[word])
            if parity:
                pivot = pivots[row]
                sampled[pivot // 64] |= np.uint64(1) << np.uint64(pivot % 64)
        changed = 0
        for column in range(num_free):
            if sampled[column // 64] & (np.uint64(1) << np.uint64(column % 64)):
                qubit = free[column]
                state[qubit] ^= np.uint8(1)
                label ^= qubit_sigs[qubit]
                changed += 1
        return label, num_free - rank, changed


    @njit(cache=True)
    def _global_nb_joint(state, label, weight, rng_state, block_index,
                         block_size, indices, offsets, signatures, boltzmann):
        categories = 1 << block_size
        energies = np.empty(categories, dtype=np.int32)
        energies[0] = weight
        previous_gray = 0
        scratch_weight = weight
        base = block_index * block_size
        for enumeration in range(1, categories):
            gray = enumeration ^ (enumeration >> 1)
            changed = gray ^ previous_gray
            position = 0
            while ((changed >> position) & 1) == 0:
                position += 1
            generator = base + position
            scratch_weight += _global_nb_toggle(
                state, indices, offsets[generator], offsets[generator + 1],
            )
            energies[gray] = scratch_weight
            previous_gray = gray
        for position in range(block_size):
            if (previous_gray >> position) & 1:
                generator = base + position
                for entry in range(offsets[generator], offsets[generator + 1]):
                    state[indices[entry]] ^= np.uint8(1)
        minimum = energies[0]
        for category in range(1, categories):
            if energies[category] < minimum:
                minimum = energies[category]
        total = 0.0
        for category in range(categories):
            total += boltzmann[energies[category] - minimum]
        threshold = _global_nb_random(rng_state) * total
        cumulative = 0.0
        selected = categories - 1
        for category in range(categories):
            cumulative += boltzmann[energies[category] - minimum]
            if threshold < cumulative:
                selected = category
                break
        for position in range(block_size):
            if (selected >> position) & 1:
                generator = base + position
                for entry in range(offsets[generator], offsets[generator + 1]):
                    state[indices[entry]] ^= np.uint8(1)
                label ^= signatures[block_index, position]
        return label, int(energies[selected]), int(selected != 0)


    @njit(cache=True, inline="always")
    def _global_nb_seen(seen, label):
        for bit in range(seen.shape[0]):
            seen[bit, int((label >> np.uint64(bit)) & np.uint64(1))] = 1


    @njit(cache=True, inline="always")
    def _global_nb_pack_row(state, output):
        for byte in range(output.size):
            value = np.uint8(0)
            start = byte * 8
            stop = min(start + 8, state.size)
            for bit in range(start, stop):
                value |= state[bit] << np.uint8(bit - start)
            output[byte] = value


    @njit(cache=True)
    def _run_hard_stage_numba_core(state, label, weight, rng_state, sweeps,
                                   record, k, heatbath, boltzmann, b_value,
                                   stabilizer_indices, stabilizer_offsets,
                                   catalog_indices, catalog_offsets, catalog_sigs,
                                   cluster_repeats, check_indices, check_offsets,
                                   qubit_sigs, num_checks, joint_block_size,
                                   joint_num_blocks, joint_indices, joint_offsets,
                                   joint_sigs):
        bytes_per_state = (state.size + 7) // 8
        record_count = sweeps if record else 0
        packed = np.empty((record_count, bytes_per_state), dtype=np.uint8)
        labels = np.empty(record_count, dtype=np.uint64)
        weights = np.empty(record_count, dtype=np.int32)
        counters = np.zeros(10, dtype=np.int64)
        seen = np.zeros((k, 2), dtype=np.uint8)
        _global_nb_seen(seen, label)
        stabilizer_order = np.empty(stabilizer_offsets.size - 1, dtype=np.int32)
        catalog_order = np.empty(catalog_offsets.size - 1, dtype=np.int32)
        n = state.size
        for sweep in range(sweeps):
            _global_nb_permutation(rng_state, stabilizer_order)
            for slot in range(stabilizer_order.size):
                coordinate = stabilizer_order[slot]
                delta = _global_nb_support_delta(
                    state, stabilizer_indices,
                    stabilizer_offsets[coordinate], stabilizer_offsets[coordinate + 1],
                )
                counters[0] += 1
                if _global_nb_random(rng_state) < heatbath[delta + n]:
                    for entry in range(stabilizer_offsets[coordinate], stabilizer_offsets[coordinate + 1]):
                        state[stabilizer_indices[entry]] ^= np.uint8(1)
                    weight += delta
                    counters[1] += 1
            _global_nb_permutation(rng_state, catalog_order)
            for slot in range(catalog_order.size):
                coordinate = catalog_order[slot]
                delta = _global_nb_support_delta(
                    state, catalog_indices,
                    catalog_offsets[coordinate], catalog_offsets[coordinate + 1],
                )
                counters[2] += 1
                if _global_nb_random(rng_state) < heatbath[delta + n]:
                    for entry in range(catalog_offsets[coordinate], catalog_offsets[coordinate + 1]):
                        state[catalog_indices[entry]] ^= np.uint8(1)
                    weight += delta
                    label ^= catalog_sigs[coordinate]
                    counters[3] += 1
            for repeat in range(cluster_repeats):
                counters[4] += 1
                label, nullity, changed = _global_nb_cluster(
                    state, label, rng_state, b_value, check_indices,
                    check_offsets, qubit_sigs, num_checks,
                )
                counters[5] += int(changed > 0)
                counters[6] += nullity
                counters[7] += changed
                if changed:
                    weight = 0
                    for qubit in range(n):
                        weight += int(state[qubit])
            if joint_block_size:
                block = _global_nb_randbelow(rng_state, joint_num_blocks)
                counters[8] += 1
                label, weight, changed = _global_nb_joint(
                    state, label, weight, rng_state, block, joint_block_size,
                    joint_indices, joint_offsets, joint_sigs, boltzmann,
                )
                counters[9] += changed
            _global_nb_seen(seen, label)
            if record:
                _global_nb_pack_row(state, packed[sweep])
                labels[sweep] = label
                weights[sweep] = weight
        return state, label, weight, packed, labels, weights, counters, seen
else:  # pragma: no cover
    _run_hard_stage_numba_core = None


def _run_hard_stage_numba(state, label, config, model, catalog, joint,
                          seed, sweeps, record):
    if _run_hard_stage_numba_core is None:
        raise RuntimeError("Numba is required for the accelerated global sampler")
    load_exp101()
    from exp101_certified_src.prng import PortablePrng

    K = coupling(config.p)
    heatbath = _heatbath_table(K, model.num_qubits)
    boltzmann = np.exp(-K * np.arange(model.num_qubits + 1, dtype=np.float64))
    _, stabilizer_indices, stabilizer_offsets = _rows_to_csr(model.stabilizer_rows)
    check_indices, check_offsets = _column_check_csr(model)
    frame_proxy = type("FrameProxy", (), {
        "k": model.k, "num_qubits": model.num_qubits,
        "W_basis": model._global_frame_W,
    })
    q_sigs = qubit_signatures(frame_proxy)
    if joint is None:
        joint_indices = np.empty(0, dtype=np.int32)
        joint_offsets = np.zeros(1, dtype=np.int64)
        joint_sigs = np.zeros((1, 1), dtype=np.uint64)
        joint_blocks = 0
    else:
        joint_indices = joint.support_indices
        joint_offsets = joint.support_offsets
        joint_sigs = joint.signatures
        joint_blocks = joint.num_blocks
    rng_state = PortablePrng(int(seed)).state_array()
    return _run_hard_stage_numba_core(
        np.ascontiguousarray(state), np.uint64(label), int(state.sum()), rng_state,
        int(sweeps), bool(record), int(model.k), heatbath, boltzmann,
        float(config.p) / (1.0 - float(config.p)),
        stabilizer_indices, stabilizer_offsets,
        catalog.support_indices, catalog.support_offsets, catalog.signatures,
        int(config.cluster_repeats), check_indices, check_offsets, q_sigs,
        int(model.num_checks), int(config.joint_block_size), int(joint_blocks),
        joint_indices, joint_offsets, joint_sigs,
    )


def run_hardcoset_trajectory(model, frame, syndrome, config, seed_identity,
                             initial_state, *, engine="numba", catalog=None,
                             joint=None):
    """Run one fixed-clock hard-coset trajectory and retain every measurement."""
    syndrome = np.ascontiguousarray(syndrome, dtype=np.uint8)
    state = np.ascontiguousarray(initial_state, dtype=np.uint8).copy()
    if state.shape != (model.num_qubits,) or syndrome.shape != (model.num_checks,):
        raise ValueError("hard-coset state or syndrome shape mismatch")
    residual = (
        model.H_check.astype(np.int64) @ state.astype(np.int64) % 2
    ).astype(np.uint8) ^ syndrome
    if residual.any():
        raise GlobalConflictError("initial state is outside the requested hard coset")
    if config.method_id != seed_identity.method_id:
        raise GlobalConflictError("hard-coset config/seed method mismatch")
    catalog = catalog or build_logical_proposal_catalog(model, frame)
    if config.joint_block_size:
        joint = joint or build_joint_blocks(model, frame, catalog, config.joint_block_size)
    elif joint is not None:
        raise ValueError("cluster method must not receive a joint-block catalog")
    # The private attribute is temporary call-local wiring for the low-level
    # kernels; it is removed before returning and never serialized.
    model._global_frame_W = np.ascontiguousarray(frame.W_basis, dtype=np.uint8)
    try:
        runner = _run_hard_stage_reference if engine == "reference" else _run_hard_stage_numba
        if engine not in ("reference", "numba"):
            raise ValueError("hard-coset engine must be reference or numba")
        initial = state.copy()
        label = state_label(frame, state)
        burn = runner(
            state, label, config, model, catalog, joint,
            seed_identity.seed("burn"), config.burn_sweeps, True,
        )
        state, label = burn[0], burn[1]
        burn_endpoint = state.copy()
        measured = runner(
            state, label, config, model, catalog, joint,
            seed_identity.seed("measurement"), config.measurement_sweeps, True,
        )
    finally:
        delattr(model, "_global_frame_W")
    state, label, weight, packed, labels, weights, measurement_counters, _ = measured
    if label != state_label(frame, state) or weight != int(state.sum()):
        raise GlobalConflictError("hard-coset cached label or weight drifted")
    states = unpack_states(packed, model.num_qubits)
    final_residuals = (
        model.H_check.astype(np.int64) @ states.T.astype(np.int64) % 2
    ).T.astype(np.uint8) ^ syndrome[None, :]
    residual_weights = final_residuals.sum(axis=1).astype(np.int32)
    if residual_weights.any():
        raise GlobalConflictError("hard-coset trajectory left the affine coset")
    return {
        "initial_state_packed": pack_state(initial),
        "burn_state_packed": pack_state(burn_endpoint),
        "final_state_packed": pack_state(state),
        "measurement_states_packed": packed,
        "measurement_labels": labels,
        "measurement_weights": weights,
        "measurement_residual_weights": residual_weights,
        "measurement_block": np.repeat(
            np.arange(8, dtype=np.int8), config.measurement_sweeps // 8,
        ),
        "burn_counters": burn[6],
        "measurement_counters": measurement_counters,
        "burn_basis_seen": burn[7],
        "burn_labels": burn[4],
        "initial_label": state_label(frame, initial),
        "burn_label": state_label(frame, burn_endpoint),
        "final_label": np.uint64(label),
        "catalog_sha256": catalog.catalog_sha256,
        "joint_sha256": "none" if joint is None else joint.joint_sha256,
        "engine": engine,
    }


def _defect_flip_probability(log_ratio):
    if log_ratio >= 0.0:
        value = math.exp(-log_ratio)
        return 1.0 / (1.0 + value)
    value = math.exp(log_ratio)
    return value / (1.0 + value)


def _reference_worm_sweep(state, label, weight, residual, defect_count,
                          in_excursion, rng, K, bias, dmax, check_indices,
                          check_offsets, qubit_sigs):
    counters = np.zeros(len(DEFECT_COUNTER_NAMES), dtype=np.int64)
    for qubit_value in rng.permutation(state.size):
        qubit = int(qubit_value)
        ones = 0
        degree = int(check_offsets[qubit + 1] - check_offsets[qubit])
        for position in range(check_offsets[qubit], check_offsets[qubit + 1]):
            ones += int(residual[check_indices[position]])
        new_defect = defect_count + degree - 2 * ones
        delta_weight = 1 - 2 * int(state[qubit])
        draw = rng.random()
        probability = 0.0
        if 0 <= new_defect <= dmax:
            log_ratio = -K * delta_weight + float(bias[new_defect] - bias[defect_count])
            probability = _defect_flip_probability(log_ratio)
        counters[0] += 1
        if draw < probability:
            old_defect = defect_count
            state[qubit] ^= 1
            label ^= qubit_sigs[qubit]
            weight += delta_weight
            for position in range(check_offsets[qubit], check_offsets[qubit + 1]):
                residual[check_indices[position]] ^= 1
            defect_count = new_defect
            counters[1] += 1
            if old_defect == 0 and defect_count > 0:
                in_excursion = True
                counters[2] += 1
            elif old_defect > 0 and defect_count == 0 and in_excursion:
                in_excursion = False
                counters[3] += 1
    return label, weight, defect_count, in_excursion, counters


def _run_defect_stage_reference(state, label, syndrome, config, bias, seed,
                                sweeps, record, frame):
    load_exp101()
    from exp101_certified_src.prng import PortablePrng

    rng = PortablePrng(int(seed))
    check_indices, check_offsets = _column_check_csr(config._model)
    q_sigs = qubit_signatures(frame)
    residual = (
        config._model.H_check.astype(np.int64) @ state.astype(np.int64) % 2
    ).astype(np.uint8) ^ syndrome
    defect = int(residual.sum())
    if defect > config.dmax:
        raise GlobalConflictError("initial worm state exceeds Dmax")
    weight = int(state.sum())
    in_excursion = defect > 0
    counters = np.zeros(len(DEFECT_COUNTER_NAMES), dtype=np.int64)
    seen = np.zeros((frame.k, 2), dtype=np.uint8)
    _update_seen(seen, label)
    count = int(sweeps) if record else 0
    packed = np.empty((count, (state.size + 7) // 8), dtype=np.uint8)
    labels = np.empty(count, dtype=np.uint64)
    weights = np.empty(count, dtype=np.int32)
    defects = np.empty(count, dtype=np.int32)
    K = coupling(config.p)
    for sweep in range(int(sweeps)):
        label, weight, defect, in_excursion, increment = _reference_worm_sweep(
            state, label, weight, residual, defect, in_excursion, rng, K, bias,
            config.dmax, check_indices, check_offsets, q_sigs,
        )
        counters += increment
        _update_seen(seen, label)
        if record:
            packed[sweep] = pack_state(state)
            labels[sweep] = label
            weights[sweep] = weight
            defects[sweep] = defect
    return (
        state, np.uint64(label), weight, residual, defect, in_excursion,
        packed, labels, weights, defects, counters, seen,
    )


if njit is not None:
    @njit(cache=True)
    def _global_nb_worm_sweep(state, label, weight, residual, defect,
                              in_excursion, rng_state, K, bias, dmax,
                              check_indices, check_offsets, qubit_sigs,
                              order):
        counters = np.zeros(4, dtype=np.int64)
        _global_nb_permutation(rng_state, order)
        for slot in range(order.size):
            qubit = order[slot]
            ones = 0
            degree = check_offsets[qubit + 1] - check_offsets[qubit]
            for position in range(check_offsets[qubit], check_offsets[qubit + 1]):
                ones += int(residual[check_indices[position]])
            new_defect = defect + degree - 2 * ones
            delta_weight = 1 - 2 * int(state[qubit])
            draw = _global_nb_random(rng_state)
            probability = 0.0
            if 0 <= new_defect <= dmax:
                log_ratio = -K * delta_weight + bias[new_defect] - bias[defect]
                if log_ratio >= 0.0:
                    value = math.exp(-log_ratio)
                    probability = 1.0 / (1.0 + value)
                else:
                    value = math.exp(log_ratio)
                    probability = value / (1.0 + value)
            counters[0] += 1
            if draw < probability:
                old_defect = defect
                state[qubit] ^= np.uint8(1)
                label ^= qubit_sigs[qubit]
                weight += delta_weight
                for position in range(check_offsets[qubit], check_offsets[qubit + 1]):
                    residual[check_indices[position]] ^= np.uint8(1)
                defect = new_defect
                counters[1] += 1
                if old_defect == 0 and defect > 0:
                    in_excursion = True
                    counters[2] += 1
                elif old_defect > 0 and defect == 0 and in_excursion:
                    in_excursion = False
                    counters[3] += 1
        return label, weight, defect, in_excursion, counters


    @njit(cache=True)
    def _run_defect_stage_numba_core(state, label, weight, residual, defect,
                                     rng_state, sweeps, record, k, K, bias,
                                     dmax, check_indices, check_offsets,
                                     qubit_sigs):
        count = sweeps if record else 0
        packed = np.empty((count, (state.size + 7) // 8), dtype=np.uint8)
        labels = np.empty(count, dtype=np.uint64)
        weights = np.empty(count, dtype=np.int32)
        defects = np.empty(count, dtype=np.int32)
        counters = np.zeros(4, dtype=np.int64)
        seen = np.zeros((k, 2), dtype=np.uint8)
        _global_nb_seen(seen, label)
        order = np.empty(state.size, dtype=np.int32)
        in_excursion = defect > 0
        for sweep in range(sweeps):
            label, weight, defect, in_excursion, increment = _global_nb_worm_sweep(
                state, label, weight, residual, defect, in_excursion, rng_state,
                K, bias, dmax, check_indices, check_offsets, qubit_sigs, order,
            )
            counters += increment
            _global_nb_seen(seen, label)
            if record:
                _global_nb_pack_row(state, packed[sweep])
                labels[sweep] = label
                weights[sweep] = weight
                defects[sweep] = defect
        return (
            state, label, weight, residual, defect, in_excursion, packed,
            labels, weights, defects, counters, seen,
        )


    @njit(cache=True)
    def _tune_defect_bias_numba_core(states, residuals, defects, rng_states,
                                     K, dmax, check_indices, check_offsets,
                                     gammas):
        chains, n = states.shape
        bias = np.zeros(dmax + 1, dtype=np.float64)
        bias_trace = np.empty((gammas.size + 1, dmax + 1), dtype=np.float64)
        bias_trace[0] = bias
        histogram = np.zeros((gammas.size, dmax + 1), dtype=np.uint8)
        weights = np.empty(chains, dtype=np.int32)
        labels = np.zeros(chains, dtype=np.uint64)
        in_excursion = np.zeros(chains, dtype=np.uint8)
        order = np.empty(n, dtype=np.int32)
        zero_sigs = np.zeros(n, dtype=np.uint64)
        for chain in range(chains):
            total = 0
            for qubit in range(n):
                total += int(states[chain, qubit])
            weights[chain] = total
        target_tail = 0.75 / dmax
        for sweep in range(gammas.size):
            for chain in range(chains):
                label, weight, defect, active, unused = _global_nb_worm_sweep(
                    states[chain], labels[chain], weights[chain], residuals[chain],
                    defects[chain], bool(in_excursion[chain]), rng_states[chain],
                    K, bias, dmax, check_indices, check_offsets, zero_sigs, order,
                )
                labels[chain] = label
                weights[chain] = weight
                defects[chain] = defect
                in_excursion[chain] = int(active)
                histogram[sweep, defect] += np.uint8(1)
            for value in range(dmax + 1):
                target = 0.25 if value == 0 else target_tail
                observed = float(histogram[sweep, value]) / chains
                bias[value] += gammas[sweep] * (target - observed)
                bias_trace[sweep + 1, value] = bias[value]
        return bias, bias_trace, histogram, states, residuals, defects
else:  # pragma: no cover
    _run_defect_stage_numba_core = None
    _tune_defect_bias_numba_core = None


def _run_defect_stage_numba(state, label, syndrome, config, bias, seed,
                            sweeps, record, frame):
    if _run_defect_stage_numba_core is None:
        raise RuntimeError("Numba is required for accelerated defect trace")
    load_exp101()
    from exp101_certified_src.prng import PortablePrng

    check_indices, check_offsets = _column_check_csr(config._model)
    residual = (
        config._model.H_check.astype(np.int64) @ state.astype(np.int64) % 2
    ).astype(np.uint8) ^ syndrome
    return _run_defect_stage_numba_core(
        np.ascontiguousarray(state), np.uint64(label), int(state.sum()),
        np.ascontiguousarray(residual), int(residual.sum()),
        PortablePrng(int(seed)).state_array(), int(sweeps), bool(record),
        int(frame.k), coupling(config.p), np.ascontiguousarray(bias, dtype=np.float64),
        int(config.dmax), check_indices, check_offsets, qubit_signatures(frame),
    )


def _tuning_gammas(num_sweeps):
    return np.asarray([
        min(0.1, 0.5 / ((sweep + 10) ** 0.6))
        for sweep in range(int(num_sweeps))
    ], dtype=np.float64)


def tune_defect_bias(model, syndrome, config, seed_identities, *, engine="numba"):
    """Run the frozen 8-chain stochastic-approximation bias tuning stage."""
    if len(seed_identities) != config.tuning_chains:
        raise ValueError("defect tuning requires exactly eight seed identities")
    if any(identity.init_family != "TUNE" for identity in seed_identities):
        raise GlobalConflictError("defect tuning seeds require TUNE initialization")
    syndrome = np.ascontiguousarray(syndrome, dtype=np.uint8)
    states = np.asarray([
        uniform_hard_coset_state(model, syndrome, identity.seed("tuning", "initialize"))
        for identity in seed_identities
    ], dtype=np.uint8)
    residuals = np.asarray([
        (model.H_check.astype(np.int64) @ state.astype(np.int64) % 2).astype(np.uint8)
        ^ syndrome for state in states
    ], dtype=np.uint8)
    defects = residuals.sum(axis=1).astype(np.int32)
    gammas = _tuning_gammas(config.tuning_sweeps)
    load_exp101()
    from exp101_certified_src.prng import PortablePrng

    rng_states = np.asarray([
        PortablePrng(identity.seed("tuning", "stream")).state_array()
        for identity in seed_identities
    ], dtype=np.uint64)
    check_indices, check_offsets = _column_check_csr(model)
    K = coupling(config.p)
    if engine == "numba":
        if _tune_defect_bias_numba_core is None:
            raise RuntimeError("Numba is unavailable for defect bias tuning")
        result = _tune_defect_bias_numba_core(
            states, residuals, defects, rng_states, K, config.dmax,
            check_indices, check_offsets, gammas,
        )
    elif engine == "reference":
        bias = np.zeros(config.dmax + 1, dtype=np.float64)
        trace = np.empty((config.tuning_sweeps + 1, config.dmax + 1), dtype=np.float64)
        trace[0] = bias
        histogram = np.zeros((config.tuning_sweeps, config.dmax + 1), dtype=np.uint8)
        rngs = [PortablePrng(identity.seed("tuning", "stream")) for identity in seed_identities]
        weights = states.sum(axis=1).astype(np.int32)
        labels = np.zeros(config.tuning_chains, dtype=np.uint64)
        active = np.zeros(config.tuning_chains, dtype=bool)
        for sweep in range(config.tuning_sweeps):
            for chain in range(config.tuning_chains):
                labels[chain], weights[chain], defects[chain], active[chain], _ = (
                    _reference_worm_sweep(
                        states[chain], labels[chain], int(weights[chain]), residuals[chain],
                        int(defects[chain]), bool(active[chain]), rngs[chain], K, bias,
                        config.dmax, check_indices, check_offsets,
                        np.zeros(model.num_qubits, dtype=np.uint64),
                    )
                )
                histogram[sweep, defects[chain]] += 1
            observed = histogram[sweep].astype(np.float64) / config.tuning_chains
            target = np.full(config.dmax + 1, 0.75 / config.dmax)
            target[0] = 0.25
            bias += gammas[sweep] * (target - observed)
            trace[sweep + 1] = bias
        result = bias, trace, histogram, states, residuals, defects
    else:
        raise ValueError("defect tuning engine must be reference or numba")
    bias, trace, histogram, final_states, final_residuals, final_defects = result
    digest = _sha256_arrays(
        "exp102.q0_defect_bias.v1",
        (np.asarray(bias, dtype=">f8"),),
        (config.method_id, float(config.p), config.tuning_chains,
         config.tuning_sweeps, *(canonical_json(identity.as_dict()) for identity in seed_identities)),
    )
    return {
        "bias": np.asarray(bias, dtype=np.float64),
        "bias_trace": np.asarray(trace, dtype=np.float64),
        "tuning_histogram": np.asarray(histogram, dtype=np.uint8),
        "tuning_final_states_packed": np.packbits(final_states, axis=1, bitorder="little"),
        "tuning_final_residuals": np.asarray(final_residuals, dtype=np.uint8),
        "tuning_final_defects": np.asarray(final_defects, dtype=np.int32),
        "gammas": gammas,
        "bias_sha256": digest,
        "engine": engine,
    }


def run_defect_trace_trajectory(model, frame, syndrome, config, seed_identity,
                                initial_state, bias, bias_sha256, *, engine="numba"):
    """Run a frozen-bias defect trace with fixed-clock observations."""
    if config.method_id != seed_identity.method_id:
        raise GlobalConflictError("defect config/seed method mismatch")
    bias = np.ascontiguousarray(bias, dtype=np.float64)
    if bias.shape != (config.dmax + 1,) or not np.all(np.isfinite(bias)):
        raise ValueError("defect bias has the wrong shape or non-finite entries")
    if len(str(bias_sha256)) != 64:
        raise ValueError("defect bias SHA256 is malformed")
    state = np.ascontiguousarray(initial_state, dtype=np.uint8).copy()
    syndrome = np.ascontiguousarray(syndrome, dtype=np.uint8)
    initial_residual = (
        model.H_check.astype(np.int64) @ state.astype(np.int64) % 2
    ).astype(np.uint8) ^ syndrome
    if initial_residual.any():
        raise GlobalConflictError("defect trace must start in the hard coset")
    # Call-local wiring avoids changing the frozen config serialization.
    object.__setattr__(config, "_model", model)
    try:
        runner = _run_defect_stage_reference if engine == "reference" else _run_defect_stage_numba
        if engine not in ("reference", "numba"):
            raise ValueError("defect engine must be reference or numba")
        initial = state.copy()
        label = state_label(frame, state)
        burn = runner(
            state, label, syndrome, config, bias, seed_identity.seed("burn"),
            config.burn_sweeps, True, frame,
        )
        state, label = burn[0], burn[1]
        burn_endpoint = state.copy()
        measured = runner(
            state, label, syndrome, config, bias,
            seed_identity.seed("measurement"), config.measurement_sweeps, True, frame,
        )
    finally:
        object.__delattr__(config, "_model")
    state, label, weight, residual, defect = measured[:5]
    packed, labels, weights, defects, counters, _ = measured[6:12]
    states = unpack_states(packed, model.num_qubits)
    recomputed_residual = (
        model.H_check.astype(np.int64) @ states.T.astype(np.int64) % 2
    ).T.astype(np.uint8) ^ syndrome[None, :]
    recomputed_defects = recomputed_residual.sum(axis=1).astype(np.int32)
    if not np.array_equal(defects, recomputed_defects):
        raise GlobalConflictError("defect-count cache drifted")
    if label != state_label(frame, state) or weight != int(state.sum()) or defect != int(residual.sum()):
        raise GlobalConflictError("defect trace cached state metadata drifted")
    return {
        "initial_state_packed": pack_state(initial),
        "burn_state_packed": pack_state(burn_endpoint),
        "final_state_packed": pack_state(state),
        "measurement_states_packed": packed,
        "measurement_labels": labels,
        "measurement_weights": weights,
        "measurement_defect_counts": defects,
        "fixed_clock_d0_mask": defects == 0,
        "measurement_block": np.repeat(
            np.arange(8, dtype=np.int8), config.measurement_sweeps // 8,
        ),
        "burn_counters": burn[10],
        "measurement_counters": counters,
        "burn_basis_seen": burn[11],
        "burn_labels": burn[7],
        "initial_label": state_label(frame, initial),
        "burn_label": state_label(frame, burn_endpoint),
        "final_label": np.uint64(label),
        "bias": bias,
        "bias_sha256": str(bias_sha256),
        "boundary_occupancy": float(np.mean(defects == config.dmax)),
        "engine": engine,
    }


def canonical_global_trajectory_digest(result):
    fields = (
        "initial_state_packed", "burn_state_packed", "final_state_packed",
        "measurement_states_packed", "measurement_labels", "measurement_weights",
        "measurement_block", "burn_counters", "measurement_counters", "burn_labels",
        "burn_basis_seen",
    )
    arrays = [np.asarray(result[field]) for field in fields]
    for optional in ("measurement_residual_weights", "measurement_defect_counts", "fixed_clock_d0_mask"):
        if optional in result:
            arrays.append(np.asarray(result[optional]))
    return _sha256_arrays(
        "exp102.q0_global.trajectory_digest.v1", arrays,
        (result["initial_label"], result["burn_label"], result["final_label"]),
    )
