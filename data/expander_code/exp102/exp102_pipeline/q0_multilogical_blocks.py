"""Code-only multi-logical block heatbath for q=0 discovery.

This is deliberately separate from the terminal ``q0_global.discovery.v1``
contract.  Each block contains several independent logical directions and
stabilizers.  Enumerating the entire block is an exact conditional heatbath
on a fixed subgroup of ``ker(H_Z)``; the construction never uses a planted
error, a decoder output, or a sampled state.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .q0_global import (
    GlobalConflictError,
    JointBlockCatalog,
    LogicalProposalCatalog,
    _rows_to_csr,
    _sha256_arrays,
    _signature_from_bits,
    _signature_rank_masks,
    reduce_logical_basis,
    run_hardcoset_trajectory,
    validate_observable_frame,
)


MULTILOGICAL_BLOCK_VERSION = "exp102.q0_multilogical_blocks.v1"
MULTILOGICAL_EMPTY_CATALOG_VERSION = "exp102.q0_multilogical_empty_catalog.v1"
MULTILOGICAL_KERNEL_MODE = "stabilizer_heatbath_plus_multilogical_block.v1"
MULTILOGICAL_METHOD_ID = "MLB8-J16"


@dataclass(frozen=True)
class MultiLogicalBlockConfig:
    """Frozen execution parameters for the multi-logical block candidate."""

    p: float
    burn_sweeps: int
    measurement_sweeps: int
    logicals_per_block: int = 8
    block_size: int = 16
    logical_catalog_mode: str = "none"
    method_id: str = MULTILOGICAL_METHOD_ID

    def __post_init__(self):
        p = float(self.p)
        if not math.isfinite(p) or not 0.0 < p < 0.5:
            raise ValueError("multi-logical block p must lie in (0, 0.5)")
        object.__setattr__(self, "p", p)
        for name in ("burn_sweeps", "measurement_sweeps", "logicals_per_block", "block_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer")
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, int(value))
        if self.measurement_sweeps % 8:
            raise ValueError("measurement_sweeps must divide into eight blocks")
        if not 2 <= self.block_size <= 16:
            raise ValueError("multi-logical block size must lie in [2, 16]")
        if not 1 <= self.logicals_per_block < self.block_size:
            raise ValueError("logical directions must be in [1, block_size)")
        if self.logical_catalog_mode != "none":
            raise ValueError("MLB permits no logical proposal catalog")
        expected_method = f"MLB{self.logicals_per_block}-J{self.block_size:02d}"
        if self.method_id != expected_method:
            raise ValueError("multi-logical block method ID does not match its block shape")

    @property
    def cluster_repeats(self):
        return 0

    @property
    def joint_block_size(self):
        return self.block_size

    def as_dict(self):
        return {
            "method_id": self.method_id,
            "p": self.p,
            "burn_sweeps": self.burn_sweeps,
            "measurement_sweeps": self.measurement_sweeps,
            "logicals_per_block": self.logicals_per_block,
            "block_size": self.block_size,
            "logical_catalog_mode": self.logical_catalog_mode,
            "kernel_mode": MULTILOGICAL_KERNEL_MODE,
        }


def _logical_windows(reduced_basis, logicals_per_block):
    """Cover every reduced logical direction with fixed, overlapping windows."""
    k = int(reduced_basis.shape[0])
    if logicals_per_block > k:
        raise GlobalConflictError("multi-logical block exceeds the code logical rank")
    windows = []
    for start in range(0, k, logicals_per_block):
        indices = [(start + offset) % k for offset in range(logicals_per_block)]
        if len(set(indices)) != len(indices):
            raise GlobalConflictError("multi-logical window repeats a direction")
        windows.append(indices)
    return windows


def build_multilogical_blocks(model, frame, *, logicals_per_block=8, block_size=16):
    """Build deterministic heatbath blocks with several logical generators.

    A fixed block is sampled exactly over its entire GF(2) span.  Every block
    therefore preserves the hard-coset posterior by detailed balance; the
    uniform mixture of the blocks does too.  The overlap sort only affects
    efficiency, never the target or the transition law within a block.
    """
    from .exp101_bridge import load_exp101

    load_exp101()
    from exp101_certified_src.gf2 import gf2_matmul, gf2_rank

    logicals_per_block = int(logicals_per_block)
    block_size = int(block_size)
    if (not 1 <= logicals_per_block < block_size <= 16
            or model.k < logicals_per_block):
        raise ValueError("invalid multi-logical block shape for this code")
    validate_observable_frame(model, frame)
    reduced = reduce_logical_basis(model.logical_move_basis)
    if int(gf2_rank(reduced)) != model.k:
        raise GlobalConflictError("multi-logical reduced basis lost rank")
    if gf2_matmul(model.H_check, reduced.T).any():
        raise GlobalConflictError("multi-logical direction left the kernel")

    blocks = []
    signatures = []
    for window in _logical_windows(reduced, logicals_per_block):
        logical_rows = np.ascontiguousarray(reduced[window], dtype=np.uint8)
        if int(gf2_rank(logical_rows)) != logicals_per_block:
            raise GlobalConflictError("multi-logical window is dependent")
        # Prefer stabilizers touching the combined logical support, but retain
        # only rows that enlarge the actual GF(2) subgroup.
        union = np.any(logical_rows, axis=0)
        overlap = np.count_nonzero(model.stabilizer_rows & union[None, :], axis=1)
        order = sorted(range(model.stabilizer_rows.shape[0]), key=lambda index: (
            -int(overlap[index]), int(index),
        ))
        rows = [row.copy() for row in logical_rows]
        for index in order:
            candidate = np.asarray(model.stabilizer_rows[index], dtype=np.uint8)
            trial = np.ascontiguousarray([*rows, candidate], dtype=np.uint8)
            if int(gf2_rank(trial)) == len(rows) + 1:
                rows.append(candidate.copy())
            if len(rows) == block_size:
                break
        if len(rows) != block_size:
            raise GlobalConflictError("could not build a full-rank multi-logical block")
        block = np.ascontiguousarray(rows, dtype=np.uint8)
        if (int(gf2_rank(block)) != block_size
                or gf2_matmul(model.H_check, block.T).any()):
            raise GlobalConflictError("multi-logical block algebra changed")
        block_signatures = np.asarray([
            _signature_from_bits(gf2_matmul(frame.W_basis, row[:, None])[:, 0])
            for row in block
        ], dtype=np.uint64)
        if (_signature_rank_masks(block_signatures[:logicals_per_block], model.k)
                != logicals_per_block
                or np.any(block_signatures[logicals_per_block:])):
            raise GlobalConflictError("multi-logical/stabilizer signatures changed")
        blocks.append(block)
        signatures.append(block_signatures)

    generators = np.ascontiguousarray(blocks, dtype=np.uint8)
    signature_array = np.ascontiguousarray(signatures, dtype=np.uint64)
    _, indices, offsets = _rows_to_csr(generators.reshape(-1, model.num_qubits))
    digest = _sha256_arrays(
        MULTILOGICAL_BLOCK_VERSION,
        (
            np.packbits(reduced, axis=1, bitorder="little"),
            np.packbits(generators.reshape(-1, model.num_qubits), axis=1, bitorder="little"),
            signature_array.astype(">u8"),
        ),
        (model.fingerprint(), frame.fingerprint(), logicals_per_block, block_size),
    )
    return JointBlockCatalog(
        generators=generators,
        signatures=signature_array,
        support_indices=indices,
        support_offsets=offsets,
        block_size=block_size,
        joint_sha256=digest,
    )


def validate_multilogical_blocks(model, frame, blocks, *, logicals_per_block, block_size):
    """Rebuild every deterministic byte of a frozen multi-logical catalog."""
    expected = build_multilogical_blocks(
        model, frame, logicals_per_block=logicals_per_block, block_size=block_size,
    )
    if not isinstance(blocks, JointBlockCatalog):
        raise TypeError("multi-logical blocks have the wrong type")
    fields = (
        "generators", "signatures", "support_indices", "support_offsets",
        "block_size", "joint_sha256",
    )
    if any(
            getattr(blocks, name) != getattr(expected, name)
            if name in {"block_size", "joint_sha256"}
            else not np.array_equal(getattr(blocks, name), getattr(expected, name))
            for name in fields):
        raise GlobalConflictError("multi-logical block catalog replay changed")
    return True


def build_multilogical_empty_catalog(model, frame):
    """Freeze the deliberately empty logical-catalog component of MLB.

    MLB gets all logical-sector transport from its exact multi-logical block,
    rather than from the older reduced single/pair/triple proposal catalog.
    The regular stabilizer heatbath remains part of the kernel to mix the
    microscopic degrees of freedom inside every hard coset.
    """
    validate_observable_frame(model, frame)
    moves = np.zeros((0, model.num_qubits), dtype=np.uint8)
    signatures = np.zeros(0, dtype=np.uint64)
    weights = np.zeros(0, dtype=np.int32)
    reduced_basis = np.zeros((0, model.num_qubits), dtype=np.uint8)
    _, support_indices, support_offsets = _rows_to_csr(moves)
    digest = _sha256_arrays(
        MULTILOGICAL_EMPTY_CATALOG_VERSION,
        (
            np.asarray([model.num_qubits, model.k], dtype=">u8"),
            np.packbits(moves, axis=1, bitorder="little"),
        ),
        (model.fingerprint(), frame.fingerprint(), MULTILOGICAL_KERNEL_MODE),
    )
    return LogicalProposalCatalog(
        moves=moves,
        signatures=signatures,
        weights=weights,
        reduced_basis=reduced_basis,
        support_indices=support_indices,
        support_offsets=support_offsets,
        catalog_sha256=digest,
    )


def validate_multilogical_empty_catalog(model, frame, catalog):
    """Reject a catalog that could silently add an old logical mechanism."""
    expected = build_multilogical_empty_catalog(model, frame)
    if not isinstance(catalog, LogicalProposalCatalog):
        raise TypeError("multi-logical empty catalog has the wrong type")
    fields = (
        "moves", "signatures", "weights", "reduced_basis",
        "support_indices", "support_offsets", "catalog_sha256",
    )
    if any(
            getattr(catalog, name) != getattr(expected, name)
            if name == "catalog_sha256"
            else not np.array_equal(getattr(catalog, name), getattr(expected, name))
            for name in fields):
        raise GlobalConflictError("multi-logical empty catalog replay changed")
    return True


def run_multilogical_block_trajectory(model, frame, syndrome, config, seed_identity,
                                      initial_state, *, engine="numba", catalog=None,
                                      blocks=None):
    """Run a catalog-free exact hard-coset heatbath with multi-logical blocks."""
    if not isinstance(config, MultiLogicalBlockConfig):
        raise TypeError("multi-logical block config has the wrong type")
    if seed_identity.method_id != config.method_id:
        raise GlobalConflictError("multi-logical config/seed method mismatch")
    catalog = build_multilogical_empty_catalog(model, frame) if catalog is None else catalog
    validate_multilogical_empty_catalog(model, frame, catalog)
    blocks = blocks or build_multilogical_blocks(
        model, frame, logicals_per_block=config.logicals_per_block,
        block_size=config.block_size,
    )
    validate_multilogical_blocks(
        model, frame, blocks, logicals_per_block=config.logicals_per_block,
        block_size=config.block_size,
    )
    return run_hardcoset_trajectory(
        model, frame, syndrome, config, seed_identity, initial_state,
        engine=engine, catalog=catalog, joint=blocks,
    )
