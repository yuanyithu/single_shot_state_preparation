import hashlib
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.sparse import csr_matrix, hstack, identity

from .config import parse_code_id
from .ensemble import ROW_FIELDS, rebuild_code
from .exp101_bridge import load_exp101


@dataclass(frozen=True)
class DecoderModel:
    code_id: str
    m: int
    n: int
    n_checks: int
    k: int
    classical_distance: int
    classical_H_sha256: str
    logical_frame_sha256: str
    H_Z: np.ndarray
    H_X: np.ndarray
    H_Z_sparse: object
    H_augmented_sparse: object
    logical_Z: np.ndarray
    # phi_r as a matrix. exp101 builds W = Z (I xor r_sec H) and verifies that
    # it annihilates im(r_sec) and the stabilizers and pairs with the logical
    # moves, so the label map arrives already checked rather than re-derived.
    label_basis: np.ndarray
    observable_frame_fingerprint: str


def frame_digest(H_Z, H_X, logical_X, logical_Z, label_basis):
    digest = hashlib.sha256()
    for array in (H_Z, H_X, logical_X, logical_Z, label_basis):
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(array, dtype=np.uint8).tobytes())
    return digest.hexdigest()


def _build(row_items):
    """Build one model from an immutable view of its registry row."""
    row = dict(row_items)
    load_exp101()
    from exp101_certified_src.hgp import hgp_from_H
    from exp101_certified_src.logicals import logical_pauli_operators
    from exp101_certified_src.model import assemble_sector_model
    from exp101_certified_src.observables import build_observable_frame

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

    # The exp101 production sector: sector=x_error, H_check=H_Z, stabilizer
    # moves are H_X rows, logical moves are logical_X, observables are the dual
    # logical_Z characters, prepared state |+>_L.
    sector_model = assemble_sector_model(H_X, H_Z, frame, sector="x_error")
    observable_frame = build_observable_frame(sector_model)
    label_basis = np.ascontiguousarray(observable_frame.W_basis, dtype=np.uint8)
    if label_basis.shape != (int(row["k"]), int(row["n"])):
        raise ValueError(f"label basis shape mismatch for {row['code_id']}")

    n_checks = int(H_Z.shape[0])
    augmented = hstack(
        [csr_matrix(H_Z), identity(n_checks, dtype=np.uint8, format="csr")],
        format="csr",
    )
    for array in (H_Z, H_X, frame.logical_Z, label_basis):
        array.flags.writeable = False
    return DecoderModel(
        code_id=str(row["code_id"]),
        m=m,
        n=int(row["n"]),
        n_checks=n_checks,
        k=int(row["k"]),
        classical_distance=int(row["classical_distance"]),
        classical_H_sha256=str(row["classical_H_sha256"]),
        logical_frame_sha256=frame_digest(
            H_Z, H_X, frame.logical_X, frame.logical_Z, label_basis,
        ),
        H_Z=H_Z,
        H_X=H_X,
        H_Z_sparse=csr_matrix(H_Z),
        H_augmented_sparse=augmented,
        logical_Z=frame.logical_Z,
        label_basis=label_basis,
        observable_frame_fingerprint=observable_frame.fingerprint(),
    )


@lru_cache(maxsize=2)
def _build_cached(row_items):
    return _build(row_items)


def load_model(row):
    """Build the decoder model for one registry row.

    The cache is deliberately tiny: a task sweeps the whole p grid for one code
    before moving on, so one live model is enough and holding more only costs
    resident memory.
    """
    if set(row) != set(ROW_FIELDS):
        raise ValueError("registry row fields do not match the frozen schema")
    return _build_cached(tuple(sorted((key, row[key]) for key in ROW_FIELDS)))


def clear_model_cache():
    _build_cached.cache_clear()


def parity_product(matrix, vector):
    return np.asarray(matrix @ vector, dtype=np.uint8) & np.uint8(1)


def logical_label(model, error):
    """phi_r(e), the exp101 absolute logical label of one error pattern.

    This is the scoring map for q > 0, where the residual data error no longer
    has zero syndrome: H_Z (eps_hat xor eps) = mu_hat xor mu.

    For this family the label basis happens to equal logical_Z. exp101 builds
    W = Z (I xor r_sec H), and r_sec places values only on the RREF pivot
    columns of H_Z, while exp101's logical_Z basis is supported entirely off
    those columns -- so Z r_sec = 0 and the section term drops out. That is a
    measured property of the construction (asserted in tests/test_label_map.py),
    not something exp106 assumes: the label is always computed through the
    certified frame, which self-verifies that it annihilates the stabilizers and
    im(r_sec) and pairs with the logical moves.

    What genuinely changes at q > 0 is therefore the *criterion*, not the map.
    exp104 also required the residual to have zero syndrome. exp106 does not,
    because the protocol's final perfect round measures the residual syndrome
    exactly and removes it; only the logical class survives, which is what
    exp101 section 8 defines MAP success to be.
    """
    return parity_product(model.label_basis, np.asarray(error, dtype=np.uint8))
