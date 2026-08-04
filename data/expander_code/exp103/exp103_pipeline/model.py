import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from data.expander_code.exp102.exp102_pipeline.exp101_bridge import load_exp101
from data.expander_code.exp102.exp102_pipeline.registry import load_frozen_code

from .config import ensure_config


@dataclass(frozen=True)
class DecoderModel:
    code_id: str
    m: int
    n: int
    k: int
    classical_distance: int
    H_Z: np.ndarray
    H_X: np.ndarray
    H_Z_sparse: object
    logical_Z: np.ndarray


def _frame_digest(H_Z, H_X, logical_X, logical_Z, section_fingerprint):
    digest = hashlib.sha256()
    for array in (H_Z, H_X, logical_X, logical_Z):
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(array, dtype=np.uint8).tobytes())
    digest.update(section_fingerprint.encode("ascii"))
    return digest.hexdigest()


@lru_cache(maxsize=3)
def _load_model_cached(registry_path, registry_sha256, code_id):
    load_exp101()
    from exp101_certified_src.hgp import hgp_from_H
    from exp101_certified_src.logicals import logical_pauli_operators
    from exp101_certified_src.section import build_linear_section

    registry, row, classical_H = load_frozen_code(registry_path, code_id)
    if registry["registry_sha256"] != registry_sha256:
        raise ValueError("frozen registry identity mismatch")
    H_Z, H_X = hgp_from_H(classical_H)
    frame = logical_pauli_operators(H_X, H_Z)
    section_fingerprint = build_linear_section(H_Z).fingerprint()
    frame_digest = _frame_digest(
        H_Z, H_X, frame.logical_X, frame.logical_Z, section_fingerprint,
    )
    if section_fingerprint != row["section_fingerprint"]:
        raise ValueError(f"section identity mismatch for {code_id}")
    if frame_digest != row["logical_frame_fingerprint"]:
        raise ValueError(f"logical frame identity mismatch for {code_id}")
    if H_Z.shape[1] != row["n"] or frame.k != row["k"]:
        raise ValueError(f"code parameter mismatch for {code_id}")
    for array in (H_Z, H_X, frame.logical_Z):
        array.flags.writeable = False
    return DecoderModel(
        code_id=code_id,
        m=int(row["m"]),
        n=int(row["n"]),
        k=int(row["k"]),
        classical_distance=int(row["classical_distance"]),
        H_Z=H_Z,
        H_X=H_X,
        H_Z_sparse=csr_matrix(H_Z),
        logical_Z=frame.logical_Z,
    )


def load_model(config, code_id):
    config = ensure_config(config)
    if code_id not in {f"m{m:02d}_c{c:02d}" for m in config["m_values"] for c in range(8)}:
        raise ValueError(f"code is outside the frozen primary panel: {code_id!r}")
    root = Path(__file__).resolve().parents[4]
    registry_path = (root / config["registry_path"]).resolve()
    return _load_model_cached(str(registry_path), config["registry_sha256"], code_id)


def clear_model_cache():
    _load_model_cached.cache_clear()


def parity_product(matrix, vector):
    return np.asarray(matrix @ vector, dtype=np.uint8) & np.uint8(1)
