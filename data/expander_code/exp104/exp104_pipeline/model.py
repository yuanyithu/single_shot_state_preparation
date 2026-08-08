import hashlib
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.sparse import csr_matrix

from data.expander_code.exp102.exp102_pipeline.exp101_bridge import load_exp101

from .config import parse_code_id
from .ensemble import rebuild_code


@dataclass(frozen=True)
class DecoderModel:
    code_id: str
    m: int
    n: int
    k: int
    classical_distance: int
    classical_H_sha256: str
    logical_frame_sha256: str
    H_Z: np.ndarray
    H_X: np.ndarray
    H_Z_sparse: object
    logical_Z: np.ndarray


def frame_digest(H_Z, H_X, logical_X, logical_Z):
    digest = hashlib.sha256()
    for array in (H_Z, H_X, logical_X, logical_Z):
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(array, dtype=np.uint8).tobytes())
    return digest.hexdigest()


def _build(row_items):
    """Build one model from an immutable view of its registry row."""
    row = dict(row_items)
    load_exp101()
    from exp101_certified_src.hgp import hgp_from_H
    from exp101_certified_src.logicals import logical_pauli_operators

    m, _ = parse_code_id(row["code_id"])
    if int(row["m"]) != m:
        raise ValueError(f"registry row disagrees with its code id: {row['code_id']}")
    H = rebuild_code(row)
    H_Z, H_X = hgp_from_H(H)
    frame = logical_pauli_operators(H_X, H_Z)
    if H_Z.shape[1] != int(row["n"]) or frame.k != int(row["k"]):
        raise ValueError(f"code parameter mismatch for {row['code_id']}")
    if int(row["n"]) != 25 * m ** 2 or int(row["k"]) != m ** 2:
        raise ValueError(f"code parameters are off the frozen family: {row['code_id']}")
    for array in (H_Z, H_X, frame.logical_Z):
        array.flags.writeable = False
    return DecoderModel(
        code_id=str(row["code_id"]),
        m=m,
        n=int(row["n"]),
        k=int(row["k"]),
        classical_distance=int(row["classical_distance"]),
        classical_H_sha256=str(row["classical_H_sha256"]),
        logical_frame_sha256=frame_digest(
            H_Z, H_X, frame.logical_X, frame.logical_Z,
        ),
        H_Z=H_Z,
        H_X=H_X,
        H_Z_sparse=csr_matrix(H_Z),
        logical_Z=frame.logical_Z,
    )


@lru_cache(maxsize=2)
def _build_cached(row_items):
    return _build(row_items)


_ROW_FIELDS = (
    "code_id", "m", "code_index", "candidate_index", "graph_seed",
    "construction_attempts", "classical_H_sha256", "classical_rank", "n", "k",
    "classical_distance",
)


def load_model(row):
    """Build the decoder model for one registry row.

    The cache is deliberately tiny: a task sweeps the whole p grid for one code
    before moving on, so one live model is enough and holding more only costs
    resident memory.
    """
    if set(row) != set(_ROW_FIELDS):
        raise ValueError("registry row fields do not match the frozen schema")
    return _build_cached(tuple(sorted((key, row[key]) for key in _ROW_FIELDS)))


def clear_model_cache():
    _build_cached.cache_clear()


def parity_product(matrix, vector):
    return np.asarray(matrix @ vector, dtype=np.uint8) & np.uint8(1)
