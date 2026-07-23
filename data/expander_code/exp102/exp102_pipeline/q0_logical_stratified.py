"""Logical-signature-stratified independence MH for the q=0 hard coset.

The proposal first draws an absolute logical label and then draws only the
stabilizer coordinates conditional on that label.  Catalog labels use frozen
low-weight sector representatives; every other label uses the affine origin.
An explicit uniform-label branch and a full-support stabilizer component make
the proposal positive on the complete hard coset.

For affine coordinates ``x=(s,l)`` and physical state ``e(x)``, the proposal is

    q(x) = r(L(x)) sum_c omega_c Bernoulli(s xor center[L(x)]; theta_c),

where ``r`` is a normalized mixture of a frozen label catalog and the uniform
distribution on all ``2**k`` labels.  The complete independence-MH ratio is
therefore exact even when the sector representatives are approximate decoder
outputs.  Representatives affect efficiency only; they do not affect the
stationary target ``(p/(1-p))**|e|`` on ``H_Z e = y``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
import math

import numpy as np

from .exp101_bridge import load_exp101
from .io import atomic_npz, canonical_json, sha256_file, sha256_json
from .q0_global import (
    GlobalConflictError,
    _signature_rank_masks,
    reduce_logical_basis,
    validate_observable_frame,
)
from .q0_map_mixture import (
    AffineCoordinateSystem,
    independence_log_acceptance,
    build_affine_coordinate_system,
)


STRATIFIED_VERSION = "exp102.q0_logical_stratified.discovery.v1"
STRATIFIED_CODEBOOK_VERSION = "exp102.q0_logical_stratified.codebook.v2"
STRATIFIED_TRANSCRIPT_VERSION = "exp102.q0_logical_stratified.candidate_transcript.v2"
STRATIFIED_CATALOG_VERSION = "exp102.q0_logical_stratified.anchors.v2"
STRATIFIED_PROPOSAL_VERSION = "exp102.q0_logical_stratified.proposal.v2"
STRATIFIED_ARTIFACT_VERSION = "exp102.q0_logical_stratified.artifact.v1"
STRATIFIED_RAW_VERSION = "exp102.q0_logical_stratified.raw.v2"
STRATIFIED_METHOD_ID = "LSI-IMH"

DEFAULT_THETA_STABILIZER = (0.001, 0.003, 0.01, 0.04, 0.15, 0.5)
DEFAULT_COMPONENT_WEIGHTS = (0.35, 0.30, 0.20, 0.10, 0.045, 0.005)


class LogicalStratifiedConflictError(ValueError):
    pass


def _as_bits(value, *, ndim, name):
    array = np.asarray(value)
    if array.ndim != int(ndim):
        raise ValueError(f"{name} must have ndim={ndim}")
    if (not np.issubdtype(array.dtype, np.bool_)
            and not np.issubdtype(array.dtype, np.integer)):
        raise ValueError(f"{name} must be binary")
    if np.any(array < 0) or np.any(array > 1):
        raise ValueError(f"{name} must contain only zero and one")
    return np.ascontiguousarray(array, dtype=np.uint8)


def _readonly(value, dtype=None):
    result = np.array(value, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _as_integer_array(value, *, ndim, name, dtype, minimum=0, maximum=None):
    array = np.asarray(value)
    if array.ndim != int(ndim):
        raise ValueError(f"{name} must have ndim={ndim}")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must contain integers")
    info = np.iinfo(np.dtype(dtype))
    lower = max(int(minimum), int(info.min))
    upper = int(info.max) if maximum is None else min(int(maximum), int(info.max))
    if lower > upper or np.any(array < lower) or np.any(array > upper):
        raise ValueError(f"{name} is outside its allowed integer range")
    return np.ascontiguousarray(array, dtype=dtype)


def _strict_positive_int(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    if int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _strict_nonnegative_int(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a nonnegative integer")
    if int(value) < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _strict_float(value, name, *, lower=None, upper=None, lower_open=False,
                  upper_open=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    if ((lower is not None and (result <= lower if lower_open else result < lower))
            or (upper is not None and (result >= upper if upper_open else result > upper))):
        raise ValueError(f"{name} is outside its allowed range")
    return result


def _strict_sha256(value, name):
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _as_float_array(value, *, ndim, name):
    array = np.asarray(value)
    if array.ndim != int(ndim):
        raise ValueError(f"{name} must have ndim={ndim}")
    if (isinstance(array.dtype, np.dtype)
            and (not np.issubdtype(array.dtype, np.number)
                 or np.issubdtype(array.dtype, np.bool_))):
        raise ValueError(f"{name} must be numeric")
    result = np.ascontiguousarray(array, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _strict_git_sha(value, name):
    if (not isinstance(value, str) or len(value) != 40
            or any(character not in "0123456789abcdef" for character in value)):
        raise ValueError(f"{name} must be a full lowercase Git SHA")
    return value


def _sha256_arrays(version, arrays, scalars=()):
    digest = hashlib.sha256(str(version).encode("ascii") + b"\0")
    for scalar in scalars:
        digest.update(str(scalar).encode("ascii") + b"\0")
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(array.dtype.str.encode("ascii") + b"\0")
        digest.update(np.asarray(array.shape, dtype=">u8").tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _bits_to_uint64(bits):
    bits = _as_bits(bits, ndim=1, name="logical_bits")
    if bits.size > 64:
        raise ValueError("logical labels require at most 64 bits")
    result = np.uint64(0)
    for bit in np.flatnonzero(bits):
        result |= np.uint64(1) << np.uint64(bit)
    return result


def _uint64_to_bits(value, size):
    if not 0 <= int(size) <= 64:
        raise ValueError("logical label size must lie in [0,64]")
    value = np.uint64(value)
    return np.asarray(
        [(value >> np.uint64(bit)) & np.uint64(1) for bit in range(int(size))],
        dtype=np.uint8,
    )


def _label_uint64(frame, state):
    return _bits_to_uint64(frame.label_of(np.asarray(state, dtype=np.uint8)))


def _logsumexp(values):
    values = tuple(float(value) for value in values)
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def _parity_residual(H, state, syndrome):
    return (
        H.astype(np.int64) @ state.astype(np.int64) % 2
    ).astype(np.uint8) ^ syndrome


def _matrix_syndrome_sha256(model, syndrome):
    syndrome = _as_bits(syndrome, ndim=1, name="syndrome")
    return _sha256_arrays(
        "exp102.q0_logical_stratified.H_y.v1",
        (np.packbits(model.H_check, axis=1, bitorder="little"),
         np.packbits(syndrome, bitorder="little")),
        (model.H_check.shape, syndrome.shape),
    )


def _require_zero_packed_padding(packed, num_bits, name):
    packed = _as_integer_array(
        packed, ndim=2, name=name, dtype=np.uint8, maximum=255,
    )
    padding = packed.shape[1] * 8 - int(num_bits)
    if padding < 0:
        raise ValueError(f"{name} is narrower than the requested bit count")
    if padding:
        mask = np.uint8(~((1 << (8 - padding)) - 1) & 0xFF)
        if np.any(packed[:, -1] & mask):
            raise LogicalStratifiedConflictError(f"{name} has nonzero padding")
    return packed


@dataclass(frozen=True)
class LogicalSignatureCodebook:
    signatures: np.ndarray
    logical_move_weights: np.ndarray
    generator_kind: np.ndarray
    moves_packed: np.ndarray
    classical_matrix_sha256: str
    model_fingerprint: str
    frame_fingerprint: str
    codebook_sha256: str
    logical_side: int
    combination_order: int
    rank2_seed_count: int

    def __post_init__(self):
        signatures = _as_integer_array(
            self.signatures, ndim=1, name="codebook signatures",
            dtype=np.uint64, maximum=(1 << 64) - 1,
        )
        weights = _as_integer_array(
            self.logical_move_weights, ndim=1, name="codebook weights",
            dtype=np.int32, maximum=np.iinfo(np.int32).max,
        )
        kinds = _as_integer_array(
            self.generator_kind, ndim=1, name="codebook generator kinds",
            dtype=np.uint8, maximum=2,
        )
        moves = _as_integer_array(
            self.moves_packed, ndim=2, name="codebook packed moves",
            dtype=np.uint8, maximum=255,
        )
        object.__setattr__(self, "signatures", _readonly(signatures, np.uint64))
        object.__setattr__(
            self, "logical_move_weights", _readonly(weights, np.int32),
        )
        object.__setattr__(self, "generator_kind", _readonly(kinds, np.uint8))
        object.__setattr__(self, "moves_packed", _readonly(moves, np.uint8))
        if (signatures.size != weights.size or signatures.size != kinds.size
                or moves.shape[0] != signatures.size or moves.shape[1] == 0):
            raise ValueError("codebook array dimensions are inconsistent")
        object.__setattr__(self, "logical_side", _strict_positive_int(
            self.logical_side, "logical_side",
        ))
        object.__setattr__(self, "combination_order", _strict_positive_int(
            self.combination_order, "combination_order",
        ))
        object.__setattr__(self, "rank2_seed_count", _strict_positive_int(
            self.rank2_seed_count, "rank2_seed_count",
        ))
        object.__setattr__(self, "classical_matrix_sha256", _strict_sha256(
            self.classical_matrix_sha256, "classical_matrix_sha256",
        ))
        object.__setattr__(self, "model_fingerprint", _strict_sha256(
            self.model_fingerprint, "model_fingerprint",
        ))
        object.__setattr__(self, "frame_fingerprint", _strict_sha256(
            self.frame_fingerprint, "frame_fingerprint",
        ))
        object.__setattr__(self, "codebook_sha256", _strict_sha256(
            self.codebook_sha256, "codebook_sha256",
        ))

    @property
    def size(self):
        return int(self.signatures.size)


def _validate_codebook_algebra(model, frame, signatures, moves_packed):
    from scipy.sparse import csr_matrix

    H_sparse = csr_matrix(np.asarray(model.H_check, dtype=np.uint8))
    W_sparse = csr_matrix(np.asarray(frame.W_basis, dtype=np.uint8))
    for start in range(0, len(signatures), 1024):
        stop = min(start + 1024, len(signatures))
        moves = np.unpackbits(
            moves_packed[start:stop], axis=1, count=model.num_qubits,
            bitorder="little",
        ).astype(np.uint8, copy=False)
        residual = np.asarray(H_sparse @ moves.T, dtype=np.uint8) & 1
        if residual.any():
            raise LogicalStratifiedConflictError("codebook move is outside ker(H_Z)")
        label_bits = (np.asarray(W_sparse @ moves.T, dtype=np.uint8) & 1).T
        replayed = np.asarray(
            [_bits_to_uint64(row) for row in label_bits], dtype=np.uint64,
        )
        if not np.array_equal(replayed, signatures[start:stop]):
            raise LogicalStratifiedConflictError("codebook signature does not equal W d")


def validate_hgp_signature_codebook(model, frame, H, codebook):
    from .q0_hgp_collapsed import validate_hgp_wiring
    load_exp101()
    from exp101_certified_src.gf2 import gf2_rank

    if not isinstance(codebook, LogicalSignatureCodebook):
        raise TypeError("codebook must be LogicalSignatureCodebook")
    H = _as_bits(H, ndim=2, name="classical_H")
    validate_hgp_wiring(H, model)
    validate_observable_frame(model, frame)
    classical_sha = _sha256_arrays(
        "exp102.q0_logical_stratified.classical_H.v1",
        (np.packbits(H, axis=1, bitorder="little"),), H.shape,
    )
    expected_side = H.shape[1] - int(gf2_rank(H))
    if (codebook.classical_matrix_sha256 != classical_sha
            or codebook.model_fingerprint != model.fingerprint()
            or codebook.frame_fingerprint != frame.fingerprint()
            or codebook.logical_side != expected_side
            or not 1 <= codebook.combination_order <= 3
            or codebook.rank2_seed_count < 2
            or codebook.signatures.shape != codebook.logical_move_weights.shape
            or codebook.signatures.shape != codebook.generator_kind.shape
            or codebook.moves_packed.shape != (
                codebook.size, (model.num_qubits + 7) // 8,
            )):
        raise LogicalStratifiedConflictError("codebook identity/dimensions changed")
    if (not codebook.size or np.any(codebook.signatures == np.uint64(0))
            or np.unique(codebook.signatures).size != codebook.size
            or _signature_rank_masks(codebook.signatures, model.k) != model.k
            or np.any(codebook.logical_move_weights < 0)
            or np.any(codebook.generator_kind > 2)):
        raise LogicalStratifiedConflictError("codebook signatures/weights/kinds are invalid")
    padding = codebook.moves_packed.shape[1] * 8 - model.num_qubits
    if padding:
        valid_mask = (1 << (8 - padding)) - 1
        if np.any(codebook.moves_packed[:, -1] & np.uint8(~valid_mask & 0xFF)):
            raise LogicalStratifiedConflictError("codebook packed move has nonzero padding")
    replayed_weights = np.empty(codebook.size, dtype=np.int32)
    for start in range(0, codebook.size, 4096):
        stop = min(start + 4096, codebook.size)
        unpacked = np.unpackbits(
            codebook.moves_packed[start:stop], axis=1,
            count=model.num_qubits, bitorder="little",
        )
        replayed_weights[start:stop] = unpacked.sum(axis=1).astype(np.int32)
    if not np.array_equal(replayed_weights, codebook.logical_move_weights):
        raise LogicalStratifiedConflictError("codebook move weight replay failed")
    _validate_codebook_algebra(
        model, frame, codebook.signatures, codebook.moves_packed,
    )
    digest = _sha256_arrays(
        STRATIFIED_CODEBOOK_VERSION,
        (np.packbits(H, axis=1, bitorder="little"), codebook.moves_packed,
         codebook.signatures.astype(">u8"),
         codebook.logical_move_weights.astype(">i4"), codebook.generator_kind),
        (model.num_qubits, model.k, codebook.logical_side,
         codebook.combination_order, codebook.rank2_seed_count,
         model.fingerprint(), frame.fingerprint()),
    )
    if digest != codebook.codebook_sha256:
        raise LogicalStratifiedConflictError("codebook SHA replay failed")
    return True


def build_hgp_signature_codebook(
        model, frame, H, *, combination_order=3, rank2_seed_count=128):
    """Build a deterministic HGP codebook without disorder-result feedback.

    The union contains every physical ``ker(H) x (F_2^n / row(H))`` rank-one
    tensor logical, pairs from a fixed low-weight tensor beam, and all
    combinations through ``combination_order`` of the deterministically
    reduced logical basis. Duplicate signatures retain the lowest physical
    representative, with packed support as the tie-break.
    """
    load_exp101()
    from exp101_certified_src.gf2 import (
        gf2_extend_basis, gf2_matmul, gf2_nullspace, gf2_rank,
        gf2_rowspace_basis,
    )
    from .q0_hgp_collapsed import join_hgp_state, validate_hgp_wiring

    validate_observable_frame(model, frame)
    H = _as_bits(H, ndim=2, name="classical_H")
    validate_hgp_wiring(H, model)
    combination_order = _strict_positive_int(combination_order, "combination_order")
    rank2_seed_count = _strict_positive_int(rank2_seed_count, "rank2_seed_count")
    if not 1 <= combination_order <= 3:
        raise ValueError("combination_order must lie in [1,3]")
    if rank2_seed_count < 2:
        raise ValueError("rank2_seed_count must be at least two")
    side = H.shape[1] - int(gf2_rank(H))
    if (side * side != int(model.k) or not 1 <= side <= 8
            or int(gf2_rank(H)) != H.shape[0]):
        raise ValueError("stratified HGP codebook requires full-row-rank H and k=d^2")

    # signature -> (weight, packed support, kind): tensor single, reduced-basis
    # combination, or tensor pair.
    candidates = {}

    def retain(move, signature, kind):
        packed = np.packbits(move, bitorder="little").tobytes()
        value = (int(move.sum()), packed, int(kind))
        key = int(np.uint64(signature))
        previous = candidates.get(key)
        if previous is None or value < previous:
            candidates[key] = value

    logicals = np.ascontiguousarray(model.logical_move_basis, dtype=np.uint8)
    classical_kernel = np.ascontiguousarray(gf2_nullspace(H), dtype=np.uint8)
    if classical_kernel.shape != (side, H.shape[1]):
        raise LogicalStratifiedConflictError("classical kernel dimension changed")
    _, complement_indices = gf2_extend_basis(
        gf2_rowspace_basis(H), np.eye(H.shape[1], dtype=np.uint8),
    )
    classical_quotient = np.eye(H.shape[1], dtype=np.uint8)[complement_indices]
    if classical_quotient.shape != (side, H.shape[1]):
        raise LogicalStratifiedConflictError("classical quotient dimension changed")
    tensor_basis = []
    zero_B = np.zeros((H.shape[0], H.shape[0]), dtype=np.uint8)
    for left in classical_kernel:
        for right in classical_quotient:
            tensor_basis.append(join_hgp_state(np.outer(left, right), zero_B))
    tensor_basis = np.ascontiguousarray(tensor_basis, dtype=np.uint8)
    tensor_signatures = gf2_matmul(frame.W_basis, tensor_basis.T).T
    if (_signature_rank_masks(
            np.asarray([_bits_to_uint64(row) for row in tensor_signatures]),
            model.k) != model.k):
        raise LogicalStratifiedConflictError("tensor logical basis lacks full quotient rank")
    for left in range(1, 1 << side):
        left_bits = _uint64_to_bits(left, side)
        left_word = np.bitwise_xor.reduce(
            classical_kernel[left_bits.astype(bool)], axis=0,
        )
        for right in range(1, 1 << side):
            right_bits = _uint64_to_bits(right, side)
            right_word = np.bitwise_xor.reduce(
                classical_quotient[right_bits.astype(bool)], axis=0,
            )
            coefficient = np.outer(left_bits, right_bits).reshape(-1)
            move = join_hgp_state(np.outer(left_word, right_word), zero_B)
            signature_bits = np.bitwise_xor.reduce(
                tensor_signatures[coefficient.astype(bool)], axis=0,
            )
            retain(move, _bits_to_uint64(signature_bits), 0)

    # A fixed low-weight tensor beam adds structured rank-two stress modes.
    # This catches low-energy sectors that are invisible to both rank-one and
    # low-order combinations in an unrelated canonical logical basis.
    tensor_records = sorted(
        (weight, signature, packed)
        for signature, (weight, packed, kind) in candidates.items()
        if kind == 0
    )[:rank2_seed_count]
    tensor_moves = [
        np.unpackbits(
            np.frombuffer(packed, dtype=np.uint8), bitorder="little",
            count=model.num_qubits,
        ).astype(np.uint8, copy=False)
        for _, _, packed in tensor_records
    ]
    for left in range(len(tensor_records)):
        for right in range(left + 1, len(tensor_records)):
            signature = tensor_records[left][1] ^ tensor_records[right][1]
            if signature:
                retain(tensor_moves[left] ^ tensor_moves[right], signature, 2)

    reduced = reduce_logical_basis(logicals)
    reduced_signatures = gf2_matmul(frame.W_basis, reduced.T).T
    for order in range(1, combination_order + 1):
        for selected in itertools.combinations(range(model.k), order):
            move = np.bitwise_xor.reduce(reduced[list(selected)], axis=0)
            signature_bits = np.bitwise_xor.reduce(
                reduced_signatures[list(selected)], axis=0,
            )
            retain(move, _bits_to_uint64(signature_bits), 1)

    ordered = sorted(
        (weight, signature, packed, kind)
        for signature, (weight, packed, kind) in candidates.items()
    )
    signatures = np.asarray([row[1] for row in ordered], dtype=np.uint64)
    weights = np.asarray([row[0] for row in ordered], dtype=np.int32)
    kinds = np.asarray([row[3] for row in ordered], dtype=np.uint8)
    moves_packed = np.ascontiguousarray(
        [np.frombuffer(row[2], dtype=np.uint8) for row in ordered], dtype=np.uint8,
    )
    if (not signatures.size or np.any(signatures == np.uint64(0))
            or np.unique(signatures).size != signatures.size
            or _signature_rank_masks(signatures, model.k) != model.k):
        raise LogicalStratifiedConflictError("signature codebook lacks full affine rank")
    _validate_codebook_algebra(model, frame, signatures, moves_packed)
    classical_sha = _sha256_arrays(
        "exp102.q0_logical_stratified.classical_H.v1",
        (np.packbits(H, axis=1, bitorder="little"),), H.shape,
    )
    digest = _sha256_arrays(
        STRATIFIED_CODEBOOK_VERSION,
        (np.packbits(H, axis=1, bitorder="little"), moves_packed,
         signatures.astype(">u8"), weights.astype(">i4"), kinds),
        (model.num_qubits, model.k, side, combination_order,
         rank2_seed_count, model.fingerprint(), frame.fingerprint()),
    )
    result = LogicalSignatureCodebook(
        signatures=signatures,
        logical_move_weights=weights,
        generator_kind=kinds,
        moves_packed=moves_packed,
        classical_matrix_sha256=classical_sha,
        model_fingerprint=model.fingerprint(),
        frame_fingerprint=frame.fingerprint(),
        codebook_sha256=digest,
        logical_side=side,
        combination_order=int(combination_order),
        rank2_seed_count=int(rank2_seed_count),
    )
    validate_hgp_signature_codebook(model, frame, H, result)
    return result


_DECODER_H = None
_DECODER_H_SPARSE = None
_DECODER_SYNDROME = None
_DECODER_BASE_LABEL = None
_DECODER_P = None
_DECODER_MAX_ITER = None
_DECODER_K = None


def _initialize_decoder_worker(H_augmented, syndrome, base_label, p, max_iter, k):
    global _DECODER_H, _DECODER_H_SPARSE, _DECODER_SYNDROME
    global _DECODER_BASE_LABEL, _DECODER_P, _DECODER_MAX_ITER, _DECODER_K

    from scipy.sparse import csr_matrix

    _DECODER_H = np.ascontiguousarray(H_augmented, dtype=np.uint8)
    _DECODER_H_SPARSE = csr_matrix(_DECODER_H, dtype=np.uint8)
    _DECODER_SYNDROME = np.ascontiguousarray(syndrome, dtype=np.uint8)
    _DECODER_BASE_LABEL = np.uint64(base_label)
    _DECODER_P = float(p)
    _DECODER_MAX_ITER = int(max_iter)
    _DECODER_K = int(k)


def _new_sector_decoder():
    try:
        from ldpc import BpLsdDecoder
    except ImportError:  # pragma: no cover - compatibility with ldpc 2.3
        from ldpc.bplsd_decoder import BpLsdDecoder

    return BpLsdDecoder(
        _DECODER_H,
        error_rate=float(_DECODER_P),
        bp_method="product_sum",
        max_iter=int(_DECODER_MAX_ITER),
        schedule="serial",
        lsd_method="lsd_cs",
        lsd_order=0,
    )


def _decode_signature_chunk(chunk):
    """Decode one fixed chunk; decoder history never crosses chunk boundaries."""
    start, signatures = chunk
    decoder = _new_sector_decoder()
    packed_width = (_DECODER_H.shape[1] + 7) // 8
    packed = np.zeros((len(signatures), packed_width), dtype=np.uint8)
    weights = np.full(len(signatures), -1, dtype=np.int32)
    for offset, signature in enumerate(signatures):
        label = _DECODER_BASE_LABEL ^ np.uint64(signature)
        target = np.concatenate((
            _DECODER_SYNDROME,
            _uint64_to_bits(label, _DECODER_K),
        ))
        try:
            decoded = np.asarray(decoder.decode(target))
        except Exception:
            continue
        if (decoded.ndim != 1 or decoded.shape != (_DECODER_H.shape[1],)
                or (not np.issubdtype(decoded.dtype, np.integer)
                    and not np.issubdtype(decoded.dtype, np.bool_))
                or np.any((decoded != 0) & (decoded != 1))):
            continue
        state = np.ascontiguousarray(decoded, dtype=np.uint8)
        recovered = np.asarray(_DECODER_H_SPARSE @ state, dtype=np.uint8).reshape(-1) & 1
        if not np.array_equal(recovered, target):
            continue
        packed[offset] = np.packbits(state, bitorder="little")
        weights[offset] = int(state.sum())
    return int(start), weights, packed


def bplsd_sector_decoder_identity(*, p, max_iter, chunk_size):
    p = _strict_float(p, "decoder p", lower=0.0, upper=0.5,
                      lower_open=True, upper_open=True)
    max_iter = _strict_positive_int(max_iter, "decoder max_iter")
    chunk_size = _strict_positive_int(chunk_size, "decoder chunk_size")
    import ldpc

    version = getattr(ldpc, "__version__", "unknown")
    return (
        f"ldpc={version};numpy={np.__version__};BpLsdDecoder;"
        f"error_rate={p:.17g};bp_method=product_sum;"
        f"max_iter={max_iter};schedule=serial;lsd_method=lsd_cs;"
        f"lsd_order=0;chunk_size={chunk_size}"
    )


@dataclass(frozen=True)
class DecodedCandidateTranscript:
    """Complete, replayable decoder output before catalog selection.

    The decoder is a proposal-construction tool, not a source of scientific
    truth.  Retaining every decoded candidate prevents a later catalog from
    silently being hand-picked after observing sampler behaviour.
    """

    candidate_signatures: np.ndarray
    candidate_move_weights: np.ndarray
    valid: np.ndarray
    decoded_weights: np.ndarray
    decoded_packed: np.ndarray
    codebook_sha256: str
    model_fingerprint: str
    frame_fingerprint: str
    matrix_syndrome_sha256: str
    base_label: np.uint64
    decoder_identity: str
    decoder_max_iter: int
    chunk_size: int
    transcript_sha256: str

    def __post_init__(self):
        signatures = _as_integer_array(
            self.candidate_signatures, ndim=1, name="transcript signatures",
            dtype=np.uint64, maximum=(1 << 64) - 1,
        )
        move_weights = _as_integer_array(
            self.candidate_move_weights, ndim=1,
            name="transcript move weights", dtype=np.int32,
            maximum=np.iinfo(np.int32).max,
        )
        valid = _as_integer_array(
            self.valid, ndim=1, name="transcript valid flags", dtype=np.uint8,
            maximum=1,
        )
        decoded_weights = _as_integer_array(
            self.decoded_weights, ndim=1, name="transcript decoded weights",
            dtype=np.int32, minimum=-1, maximum=np.iinfo(np.int32).max,
        )
        decoded_packed = _as_integer_array(
            self.decoded_packed, ndim=2, name="transcript decoded states",
            dtype=np.uint8, maximum=255,
        )
        size = signatures.size
        if (not size or move_weights.size != size or valid.size != size
                or decoded_weights.size != size or decoded_packed.shape[0] != size
                or decoded_packed.shape[1] == 0):
            raise ValueError("candidate transcript array dimensions are inconsistent")
        if np.any((valid == 0) != (decoded_weights == -1)):
            raise ValueError("candidate transcript validity/weight flags disagree")
        if np.any((valid == 0) & np.any(decoded_packed != 0, axis=1)):
            raise ValueError("invalid transcript candidates must have zero state bytes")
        object.__setattr__(self, "candidate_signatures", _readonly(signatures, np.uint64))
        object.__setattr__(self, "candidate_move_weights", _readonly(
            move_weights, np.int32,
        ))
        object.__setattr__(self, "valid", _readonly(valid, np.uint8))
        object.__setattr__(self, "decoded_weights", _readonly(
            decoded_weights, np.int32,
        ))
        object.__setattr__(self, "decoded_packed", _readonly(decoded_packed, np.uint8))
        object.__setattr__(self, "codebook_sha256", _strict_sha256(
            self.codebook_sha256, "transcript codebook_sha256",
        ))
        object.__setattr__(self, "model_fingerprint", _strict_sha256(
            self.model_fingerprint, "transcript model_fingerprint",
        ))
        object.__setattr__(self, "frame_fingerprint", _strict_sha256(
            self.frame_fingerprint, "transcript frame_fingerprint",
        ))
        object.__setattr__(self, "matrix_syndrome_sha256", _strict_sha256(
            self.matrix_syndrome_sha256, "transcript matrix_syndrome_sha256",
        ))
        object.__setattr__(self, "base_label", np.uint64(self.base_label))
        if not isinstance(self.decoder_identity, str) or not self.decoder_identity:
            raise ValueError("transcript decoder identity is empty")
        object.__setattr__(self, "decoder_max_iter", _strict_positive_int(
            self.decoder_max_iter, "transcript decoder_max_iter",
        ))
        object.__setattr__(self, "chunk_size", _strict_positive_int(
            self.chunk_size, "transcript chunk_size",
        ))
        object.__setattr__(self, "transcript_sha256", _strict_sha256(
            self.transcript_sha256, "transcript_sha256",
        ))

    @property
    def size(self):
        return int(self.candidate_signatures.size)


def _candidate_transcript_digest(codebook, valid, decoded_weights, decoded_packed,
                                 *, matrix_syndrome_sha256, base_label,
                                 decoder_identity, decoder_max_iter, chunk_size):
    return _sha256_arrays(
        STRATIFIED_TRANSCRIPT_VERSION,
        (codebook.signatures.astype(">u8"),
         codebook.logical_move_weights.astype(">i4"),
         np.asarray(valid, dtype=np.uint8),
         np.asarray(decoded_weights, dtype=">i4"),
         np.asarray(decoded_packed, dtype=np.uint8)),
        (codebook.codebook_sha256, matrix_syndrome_sha256, int(np.uint64(base_label)),
         decoder_identity, int(decoder_max_iter), int(chunk_size)),
    )


def _gf2_row_products(matrix, states):
    """Compute ``matrix @ states.T`` over GF(2) without dense integer products."""
    matrix = _as_bits(matrix, ndim=2, name="GF(2) matrix")
    states = _as_bits(states, ndim=2, name="GF(2) states")
    if matrix.shape[1] != states.shape[1]:
        raise ValueError("GF(2) matrix/state dimensions are incompatible")
    result = np.zeros((matrix.shape[0], states.shape[0]), dtype=np.uint8)
    for row_index in range(matrix.shape[0]):
        support = np.flatnonzero(matrix[row_index])
        if support.size:
            result[row_index] = np.bitwise_xor.reduce(
                states[:, support], axis=1,
            )
    return result


def validate_decoded_candidate_transcript(model, frame, H, syndrome, codebook,
                                          base_anchor, transcript):
    """Replay every decoder candidate against the augmented hard-coset system."""
    if not isinstance(transcript, DecodedCandidateTranscript):
        raise TypeError("transcript must be DecodedCandidateTranscript")
    validate_hgp_signature_codebook(model, frame, H, codebook)
    y = _as_bits(syndrome, ndim=1, name="syndrome")
    base = _as_bits(base_anchor, ndim=1, name="base_anchor")
    if y.shape != (model.num_checks,) or base.shape != (model.num_qubits,):
        raise ValueError("candidate transcript model dimensions changed")
    if _parity_residual(model.H_check, base, y).any():
        raise LogicalStratifiedConflictError("transcript base anchor is outside hard coset")
    matrix_sha = _matrix_syndrome_sha256(model, y)
    base_label = _label_uint64(frame, base)
    if (transcript.codebook_sha256 != codebook.codebook_sha256
            or transcript.model_fingerprint != model.fingerprint()
            or transcript.frame_fingerprint != frame.fingerprint()
            or transcript.matrix_syndrome_sha256 != matrix_sha
            or transcript.base_label != base_label
            or not np.array_equal(transcript.candidate_signatures, codebook.signatures)
            or not np.array_equal(
                transcript.candidate_move_weights, codebook.logical_move_weights,
            )
            or transcript.size != codebook.size):
        raise LogicalStratifiedConflictError("candidate transcript identity changed")
    packed = _require_zero_packed_padding(
        transcript.decoded_packed, model.num_qubits, "transcript decoded states",
    )
    valid_indices = np.flatnonzero(transcript.valid)
    augmented = np.vstack((model.H_check, frame.W_basis)).astype(np.uint8)
    bit_positions = np.arange(model.k, dtype=np.uint64)
    for start in range(0, valid_indices.size, 1024):
        indices = valid_indices[start:start + 1024]
        states = np.unpackbits(
            packed[indices], axis=1, count=model.num_qubits, bitorder="little",
        ).astype(np.uint8, copy=False)
        label_values = np.asarray(
            codebook.signatures[indices] ^ np.uint64(base_label), dtype=np.uint64,
        )
        label_targets = (
            (label_values[:, None] >> bit_positions[None, :]) & np.uint64(1)
        ).astype(np.uint8)
        targets = np.concatenate((
            np.repeat(y[:, None], indices.size, axis=1), label_targets.T,
        ), axis=0)
        recovered = _gf2_row_products(augmented, states)
        if (not np.array_equal(recovered, targets)
                or not np.array_equal(
                    states.sum(axis=1).astype(np.int32),
                    transcript.decoded_weights[indices],
                )):
            raise LogicalStratifiedConflictError("candidate transcript algebra replay failed")
    expected = _candidate_transcript_digest(
        codebook, transcript.valid, transcript.decoded_weights,
        transcript.decoded_packed, matrix_syndrome_sha256=matrix_sha,
        base_label=base_label, decoder_identity=transcript.decoder_identity,
        decoder_max_iter=transcript.decoder_max_iter,
        chunk_size=transcript.chunk_size,
    )
    if expected != transcript.transcript_sha256:
        raise LogicalStratifiedConflictError("candidate transcript SHA replay failed")
    return True


def _select_transcript_candidates(codebook, transcript, *, k, max_anchors):
    """Apply the fixed rank-first then fill rule to the whole transcript."""
    max_anchors = _strict_positive_int(max_anchors, "max_anchors")
    k = _strict_positive_int(k, "logical dimension")
    if max_anchors < k + 1:
        raise ValueError("max_anchors must leave room for affine rank k")
    valid_indices = np.flatnonzero(transcript.valid).tolist()
    ordered = sorted(
        valid_indices,
        key=lambda index: (
            int(transcript.decoded_weights[index]),
            int(codebook.logical_move_weights[index]),
            int(codebook.signatures[index]),
            transcript.decoded_packed[index].tobytes(),
        ),
    )
    selected = []
    selected_set = set()
    pivots = {}
    for index in ordered:
        residue = int(codebook.signatures[index])
        while residue:
            pivot = residue.bit_length() - 1
            if pivot not in pivots:
                pivots[pivot] = residue
                selected.append(index)
                selected_set.add(index)
                break
            residue ^= pivots[pivot]
        if len(pivots) == k:
            break
    if len(pivots) != k:
        raise LogicalStratifiedConflictError(
            "valid decoded candidates do not span every logical label",
        )
    for index in ordered:
        if len(selected) >= max_anchors - 1:
            break
        if index not in selected_set:
            selected.append(index)
            selected_set.add(index)
    return np.asarray(selected, dtype=np.int32), ordered


def generate_bplsd_stratified_catalog(
        model, frame, H, syndrome, p, codebook, *, max_anchors=128,
        decoder_max_iter=64, chunk_size=128, num_workers=1):
    """Generate a truth-free sector catalog and replay every retained anchor.

    BpLSD is only a proposal constructor.  Its output need not be optimal, but
    every accepted candidate is checked exactly against the augmented
    ``[H_Z; W]`` system.  The complete candidate transcript is hashed before a
    rank-complete, low-weight subset is retained.
    """
    from .q0_map_mixture import build_milp_map_anchors

    validate_observable_frame(model, frame)
    validate_hgp_signature_codebook(model, frame, H, codebook)
    y = _as_bits(syndrome, ndim=1, name="syndrome")
    if y.shape != (model.num_checks,):
        raise ValueError("syndrome length mismatch")
    if _signature_rank_masks(codebook.signatures, model.k) != model.k:
        raise LogicalStratifiedConflictError("input codebook rank changed")
    p = _strict_float(p, "p", lower=0.0, upper=0.5,
                      lower_open=True, upper_open=True)
    max_anchors = _strict_positive_int(max_anchors, "max_anchors")
    decoder_max_iter = _strict_positive_int(decoder_max_iter, "decoder_max_iter")
    chunk_size = _strict_positive_int(chunk_size, "chunk_size")
    num_workers = _strict_positive_int(num_workers, "num_workers")
    if max_anchors > 128:
        raise ValueError("stratified catalog max_anchors must not exceed 128")
    if max_anchors < model.k + 1:
        raise ValueError("max_anchors must be at least k+1")

    base_catalog = build_milp_map_anchors(
        model.H_check, y, p, max_anchors=1,
    )
    base = np.ascontiguousarray(base_catalog.anchors[0], dtype=np.uint8)
    base_label = _label_uint64(frame, base)
    H_augmented = np.ascontiguousarray(
        np.vstack((model.H_check, frame.W_basis)), dtype=np.uint8,
    )
    chunks = []
    for start in range(0, codebook.size, chunk_size):
        stop = min(start + chunk_size, codebook.size)
        chunks.append((start, codebook.signatures[start:stop].copy()))

    initializer_args = (
        H_augmented, y, base_label, p, decoder_max_iter, int(model.k),
    )
    if num_workers == 1:
        _initialize_decoder_worker(*initializer_args)
        results = [_decode_signature_chunk(chunk) for chunk in chunks]
    else:
        import multiprocessing as mp

        context = mp.get_context("fork")
        with context.Pool(
                processes=num_workers, initializer=_initialize_decoder_worker,
                initargs=initializer_args) as pool:
            results = pool.map(_decode_signature_chunk, chunks, chunksize=1)

    packed_width = (model.num_qubits + 7) // 8
    decoded_weights = np.full(codebook.size, -1, dtype=np.int32)
    decoded_packed = np.zeros((codebook.size, packed_width), dtype=np.uint8)
    for start, weights, packed in results:
        stop = start + len(weights)
        decoded_weights[start:stop] = weights
        decoded_packed[start:stop] = packed
    valid = (decoded_weights >= 0).astype(np.uint8)
    decoder_identity = (
        bplsd_sector_decoder_identity(
            p=p, max_iter=decoder_max_iter, chunk_size=chunk_size,
        )
        + f";valid={int(valid.sum())}/{codebook.size};"
        + f"base_solver={base_catalog.solver_identity};"
        + f"base_anchor_sha256={base_catalog.anchor_sha256}"
    )
    transcript = DecodedCandidateTranscript(
        candidate_signatures=codebook.signatures,
        candidate_move_weights=codebook.logical_move_weights,
        valid=valid,
        decoded_weights=decoded_weights,
        decoded_packed=decoded_packed,
        codebook_sha256=codebook.codebook_sha256,
        model_fingerprint=model.fingerprint(),
        frame_fingerprint=frame.fingerprint(),
        matrix_syndrome_sha256=_matrix_syndrome_sha256(model, y),
        base_label=base_label,
        decoder_identity=decoder_identity,
        decoder_max_iter=decoder_max_iter,
        chunk_size=chunk_size,
        transcript_sha256=_candidate_transcript_digest(
            codebook, valid, decoded_weights, decoded_packed,
            matrix_syndrome_sha256=_matrix_syndrome_sha256(model, y),
            base_label=base_label, decoder_identity=decoder_identity,
            decoder_max_iter=decoder_max_iter, chunk_size=chunk_size,
        ),
    )
    validate_decoded_candidate_transcript(
        model, frame, H, y, codebook, base, transcript,
    )
    selected, _ = _select_transcript_candidates(
        codebook, transcript, k=model.k, max_anchors=max_anchors,
    )
    selected_packed = decoded_packed[selected]
    selected_anchors = np.unpackbits(
        selected_packed, axis=1, count=model.num_qubits, bitorder="little",
    ).astype(np.uint8, copy=False)
    catalog = build_stratified_anchor_catalog(
        model, frame, y, base, selected_anchors,
        codebook.signatures[selected], codebook.logical_move_weights[selected],
        max_anchors=max_anchors, candidate_count=codebook.size,
        decoder_identity=decoder_identity,
        codebook_sha256=codebook.codebook_sha256,
        candidate_transcript_sha256=transcript.transcript_sha256,
        candidate_indices=selected,
        candidate_transcript=transcript,
    )
    validate_bplsd_stratified_catalog(
        model, frame, H, y, codebook, base, transcript, catalog,
        max_anchors=max_anchors,
    )
    return catalog


@dataclass(frozen=True)
class StratifiedAnchorCatalog:
    anchors: np.ndarray
    labels: np.ndarray
    label_deltas: np.ndarray
    anchor_weights: np.ndarray
    logical_move_weights: np.ndarray
    selection_role: np.ndarray
    candidate_count: int
    requested_max_anchors: int
    candidate_indices: np.ndarray
    decoder_identity: str
    codebook_sha256: str
    candidate_transcript_sha256: str
    catalog_sha256: str
    candidate_transcript: DecodedCandidateTranscript | None = None

    def __post_init__(self):
        for name, dtype in (
            ("anchors", np.uint8), ("labels", np.uint64),
            ("label_deltas", np.uint64), ("anchor_weights", np.int32),
            ("logical_move_weights", np.int32), ("selection_role", np.uint8),
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name), dtype))
        candidate_indices = _as_integer_array(
            self.candidate_indices, ndim=1, name="catalog candidate indices",
            dtype=np.int32, minimum=-1, maximum=np.iinfo(np.int32).max,
        )
        size = self.anchors.shape[0]
        if (self.anchors.ndim != 2 or self.anchors.shape[1] == 0
                or any(getattr(self, name).shape != (size,)
                       for name in (
                           "labels", "label_deltas", "anchor_weights",
                           "logical_move_weights", "selection_role",
                       ))
                or candidate_indices.shape != (size,)):
            raise ValueError("anchor catalog array dimensions are inconsistent")
        candidate_count = _strict_positive_int(
            self.candidate_count, "catalog candidate_count",
        )
        requested_max_anchors = _strict_positive_int(
            self.requested_max_anchors, "catalog requested_max_anchors",
        )
        if requested_max_anchors > 128 or size > requested_max_anchors:
            raise ValueError("anchor catalog requested max is invalid")
        if (candidate_indices[0] != -1 or np.any(candidate_indices[1:] < 0)
                or np.any(candidate_indices[1:] >= candidate_count)
                or np.unique(candidate_indices[1:]).size != size - 1):
            raise ValueError("anchor catalog candidate indices are invalid")
        if not isinstance(self.decoder_identity, str) or not self.decoder_identity:
            raise ValueError("anchor catalog decoder identity is empty")
        if self.candidate_transcript is not None:
            if not isinstance(self.candidate_transcript, DecodedCandidateTranscript):
                raise TypeError("catalog transcript has the wrong type")
            if candidate_count != self.candidate_transcript.size:
                raise ValueError("catalog transcript candidate count changed")
            _strict_sha256(self.codebook_sha256, "catalog codebook_sha256")
            _strict_sha256(
                self.candidate_transcript_sha256,
                "catalog candidate_transcript_sha256",
            )
            if (self.candidate_transcript_sha256
                    != self.candidate_transcript.transcript_sha256):
                raise ValueError("catalog transcript SHA changed")
        object.__setattr__(self, "candidate_count", candidate_count)
        object.__setattr__(self, "requested_max_anchors", requested_max_anchors)
        object.__setattr__(self, "candidate_indices", _readonly(
            candidate_indices, np.int32,
        ))
        object.__setattr__(self, "catalog_sha256", _strict_sha256(
            self.catalog_sha256, "catalog_sha256",
        ))

    @property
    def size(self):
        return int(self.anchors.shape[0])


def build_stratified_anchor_catalog(
        model, frame, syndrome, base_anchor, candidate_anchors,
        candidate_deltas, candidate_move_weights, *, max_anchors,
        candidate_count=None, decoder_identity="external_exact_replay",
        codebook_sha256="external", candidate_transcript_sha256="external",
        candidate_indices=None, candidate_transcript=None):
    """Select a rank-complete frozen catalog from exactly replayed candidates."""
    validate_observable_frame(model, frame)
    y = _as_bits(syndrome, ndim=1, name="syndrome")
    base = _as_bits(base_anchor, ndim=1, name="base_anchor")
    anchors = _as_bits(candidate_anchors, ndim=2, name="candidate_anchors")
    deltas = _as_integer_array(
        candidate_deltas, ndim=1, name="candidate_deltas", dtype=np.uint64,
        maximum=(1 << 64) - 1,
    )
    move_weights = _as_integer_array(
        candidate_move_weights, ndim=1, name="candidate_move_weights",
        dtype=np.int32, maximum=np.iinfo(np.int32).max,
    )
    if (y.shape != (model.num_checks,) or base.shape != (model.num_qubits,)
            or anchors.shape[1:] != (model.num_qubits,)
            or anchors.shape[0] != deltas.size
            or anchors.shape[0] != move_weights.size):
        raise ValueError("stratified anchor inputs have incompatible shapes")
    max_anchors = _strict_positive_int(max_anchors, "max_anchors")
    if max_anchors > 128:
        raise ValueError("stratified catalog max_anchors must not exceed 128")
    if max_anchors < model.k + 1:
        raise ValueError("max_anchors must leave room for affine rank k")
    actual_candidate_count = (
        anchors.shape[0] if candidate_count is None
        else _strict_positive_int(candidate_count, "candidate_count")
    )
    if candidate_indices is None:
        candidate_indices = np.arange(anchors.shape[0], dtype=np.int32)
    candidate_indices = _as_integer_array(
        candidate_indices, ndim=1, name="candidate_indices", dtype=np.int32,
        minimum=0, maximum=np.iinfo(np.int32).max,
    )
    if (candidate_indices.size != anchors.shape[0]
            or np.unique(candidate_indices).size != candidate_indices.size
            or np.any(candidate_indices >= actual_candidate_count)):
        raise ValueError("candidate indices are incompatible with catalog inputs")
    if candidate_transcript is not None:
        if not isinstance(candidate_transcript, DecodedCandidateTranscript):
            raise TypeError("candidate_transcript must be DecodedCandidateTranscript")
        if (candidate_transcript.size != actual_candidate_count
                or candidate_transcript.transcript_sha256
                != candidate_transcript_sha256):
            raise ValueError("candidate transcript identity changed")
    if _parity_residual(model.H_check, base, y).any():
        raise LogicalStratifiedConflictError("base anchor is outside the hard coset")
    residuals = (
        model.H_check.astype(np.int64) @ anchors.T.astype(np.int64) % 2
    ).T.astype(np.uint8) ^ y[None, :]
    if residuals.any():
        raise LogicalStratifiedConflictError("candidate anchor is outside the hard coset")

    base_label = _label_uint64(frame, base)
    labels = np.asarray([_label_uint64(frame, row) for row in anchors], dtype=np.uint64)
    replayed_deltas = labels ^ base_label
    if (np.any(replayed_deltas != deltas) or np.any(deltas == np.uint64(0))
            or np.unique(deltas).size != deltas.size):
        raise LogicalStratifiedConflictError("candidate label/delta replay failed")

    packed = np.packbits(anchors, axis=1, bitorder="little")
    records = sorted(
        range(anchors.shape[0]),
        key=lambda index: (
            int(anchors[index].sum()), int(move_weights[index]),
            int(deltas[index]), packed[index].tobytes(),
        ),
    )
    selected = []
    selected_set = set()
    pivots = {}
    for index in records:
        residue = int(deltas[index])
        while residue:
            pivot = residue.bit_length() - 1
            if pivot not in pivots:
                pivots[pivot] = residue
                selected.append(index)
                selected_set.add(index)
                break
            residue ^= pivots[pivot]
        if len(pivots) == model.k:
            break
    if len(pivots) != model.k:
        raise LogicalStratifiedConflictError("candidate anchors do not span logical labels")
    rank_selected = set(selected)
    for index in records:
        if len(selected) >= max_anchors - 1:
            break
        if index not in selected_set:
            selected.append(index)
            selected_set.add(index)

    final_anchors = np.ascontiguousarray(
        np.vstack((base[None, :], anchors[selected])), dtype=np.uint8,
    )
    final_labels = np.concatenate((np.asarray([base_label], dtype=np.uint64), labels[selected]))
    final_deltas = final_labels ^ base_label
    final_weights = final_anchors.sum(axis=1).astype(np.int32)
    final_move_weights = np.concatenate((
        np.zeros(1, dtype=np.int32), move_weights[selected],
    ))
    final_candidate_indices = np.concatenate((
        np.asarray([-1], dtype=np.int32), candidate_indices[selected],
    ))
    roles = np.concatenate((
        np.asarray([0], dtype=np.uint8),
        np.asarray([1 if index in rank_selected else 2 for index in selected], dtype=np.uint8),
    ))
    if (np.unique(final_labels).size != final_labels.size
            or _signature_rank_masks(final_deltas[1:], model.k) != model.k):
        raise LogicalStratifiedConflictError("selected anchor catalog lost affine rank")
    digest = _sha256_arrays(
        STRATIFIED_CATALOG_VERSION,
        (np.packbits(final_anchors, axis=1, bitorder="little"),
         final_labels.astype(">u8"), final_deltas.astype(">u8"),
         final_weights.astype(">i4"), final_move_weights.astype(">i4"), roles,
         final_candidate_indices.astype(">i4")),
        (actual_candidate_count,
         decoder_identity, codebook_sha256, candidate_transcript_sha256,
         max_anchors),
    )
    return StratifiedAnchorCatalog(
        anchors=final_anchors,
        labels=final_labels,
        label_deltas=final_deltas,
        anchor_weights=final_weights,
        logical_move_weights=final_move_weights,
        selection_role=roles,
        candidate_count=actual_candidate_count,
        requested_max_anchors=max_anchors,
        candidate_indices=final_candidate_indices,
        decoder_identity=str(decoder_identity),
        codebook_sha256=str(codebook_sha256),
        candidate_transcript_sha256=str(candidate_transcript_sha256),
        catalog_sha256=digest,
        candidate_transcript=candidate_transcript,
    )


def validate_stratified_anchor_catalog(model, frame, syndrome, catalog):
    if not isinstance(catalog, StratifiedAnchorCatalog):
        raise TypeError("catalog must be StratifiedAnchorCatalog")
    rebuilt = build_stratified_anchor_catalog(
        model, frame, syndrome, catalog.anchors[0], catalog.anchors[1:],
        catalog.label_deltas[1:], catalog.logical_move_weights[1:],
        max_anchors=catalog.requested_max_anchors,
        candidate_count=catalog.candidate_count,
        decoder_identity=catalog.decoder_identity,
        codebook_sha256=catalog.codebook_sha256,
        candidate_transcript_sha256=catalog.candidate_transcript_sha256,
        candidate_indices=catalog.candidate_indices[1:],
        candidate_transcript=catalog.candidate_transcript,
    )
    fields = (
        "anchors", "labels", "label_deltas", "anchor_weights",
        "logical_move_weights", "selection_role", "candidate_indices",
    )
    if (rebuilt.catalog_sha256 != catalog.catalog_sha256
            or any(not np.array_equal(getattr(rebuilt, name), getattr(catalog, name))
                   for name in fields)):
        raise LogicalStratifiedConflictError("anchor catalog content/SHA replay failed")
    return True


def validate_bplsd_stratified_catalog(
        model, frame, H, syndrome, codebook, base_anchor, transcript, catalog,
        *, max_anchors):
    """Verify that a frozen catalog is exactly the full-transcript selection."""
    validate_decoded_candidate_transcript(
        model, frame, H, syndrome, codebook, base_anchor, transcript,
    )
    validate_stratified_anchor_catalog(model, frame, syndrome, catalog)
    max_anchors = _strict_positive_int(max_anchors, "max_anchors")
    selected, _ = _select_transcript_candidates(
        codebook, transcript, k=model.k, max_anchors=max_anchors,
    )
    if (catalog.candidate_count != transcript.size
            or catalog.requested_max_anchors != max_anchors
            or catalog.codebook_sha256 != codebook.codebook_sha256
            or catalog.candidate_transcript_sha256 != transcript.transcript_sha256
            or (catalog.candidate_transcript is not None
                and catalog.candidate_transcript.transcript_sha256
                != transcript.transcript_sha256)
            or not np.array_equal(catalog.candidate_indices[1:], selected)):
        raise LogicalStratifiedConflictError("catalog/transcript binding changed")
    states = np.unpackbits(
        transcript.decoded_packed[selected], axis=1, count=model.num_qubits,
        bitorder="little",
    ).astype(np.uint8, copy=False)
    expected = build_stratified_anchor_catalog(
        model, frame, syndrome, base_anchor, states,
        codebook.signatures[selected], codebook.logical_move_weights[selected],
        max_anchors=max_anchors, candidate_count=transcript.size,
        decoder_identity=transcript.decoder_identity,
        codebook_sha256=codebook.codebook_sha256,
        candidate_transcript_sha256=transcript.transcript_sha256,
        candidate_indices=selected, candidate_transcript=transcript,
    )
    fields = (
        "anchors", "labels", "label_deltas", "anchor_weights",
        "logical_move_weights", "selection_role", "candidate_indices",
    )
    if (expected.catalog_sha256 != catalog.catalog_sha256
            or any(not np.array_equal(getattr(expected, name), getattr(catalog, name))
                   for name in fields)):
        raise LogicalStratifiedConflictError("catalog selection replay failed")
    return True


@dataclass(frozen=True)
class LogicalStratifiedProposal:
    coordinates: AffineCoordinateSystem
    catalog: StratifiedAnchorCatalog
    reference_label: np.uint64
    logical_signature_matrix: np.ndarray
    logical_signature_inverse: np.ndarray
    center_stabilizer_coordinates: np.ndarray
    catalog_probabilities: np.ndarray
    theta_stabilizer: np.ndarray
    component_weights: np.ndarray
    p: float
    alpha_temperature: float
    uniform_label_probability: float
    catalog_uniform_floor: float
    proposal_sha256: str

    def __post_init__(self):
        if not isinstance(self.coordinates, AffineCoordinateSystem):
            raise TypeError("proposal coordinates have the wrong type")
        if not isinstance(self.catalog, StratifiedAnchorCatalog):
            raise TypeError("proposal catalog has the wrong type")
        for name, ndim in (
                ("logical_signature_matrix", 2),
                ("logical_signature_inverse", 2),
                ("center_stabilizer_coordinates", 2)):
            object.__setattr__(self, name, _readonly(
                _as_bits(getattr(self, name), ndim=ndim, name=name), np.uint8,
            ))
        for name, ndim in (
                ("catalog_probabilities", 1), ("theta_stabilizer", 1),
                ("component_weights", 1)):
            object.__setattr__(self, name, _readonly(
                _as_float_array(getattr(self, name), ndim=ndim, name=name),
                np.float64,
            ))
        k = int(self.coordinates.logical_dimension)
        stabilizer_dimension = int(self.coordinates.stabilizer_dimension)
        if (self.logical_signature_matrix.shape != (k, k)
                or self.logical_signature_inverse.shape != (k, k)
                or self.center_stabilizer_coordinates.shape
                != (self.catalog.size, stabilizer_dimension)
                or self.catalog_probabilities.shape != (self.catalog.size,)
                or self.theta_stabilizer.ndim != 1
                or self.component_weights.shape != self.theta_stabilizer.shape
                or not np.all(np.isfinite(self.catalog_probabilities))
                or np.any(self.catalog_probabilities <= 0.0)
                or abs(float(self.catalog_probabilities.sum()) - 1.0) > 1e-13
                or not np.all(np.isfinite(self.theta_stabilizer))
                or np.any(self.theta_stabilizer <= 0.0)
                or np.any(self.theta_stabilizer >= 1.0)
                or not np.all(np.isfinite(self.component_weights))
                or np.any(self.component_weights <= 0.0)
                or abs(float(self.component_weights.sum()) - 1.0) > 1e-13):
            raise ValueError("proposal arrays are invalid")
        if np.unique(self.catalog.labels).size != self.catalog.size:
            raise ValueError("proposal catalog labels are not unique")
        object.__setattr__(self, "p", _strict_float(
            self.p, "proposal p", lower=0.0, upper=0.5,
            lower_open=True, upper_open=True,
        ))
        alpha_temperature = _strict_float(
            self.alpha_temperature, "proposal alpha_temperature",
            lower=0.0, lower_open=True,
        )
        if alpha_temperature not in (0.5, 1.0):
            raise ValueError("proposal alpha_temperature must be one of 0.5 or 1.0")
        object.__setattr__(self, "alpha_temperature", alpha_temperature)
        object.__setattr__(self, "uniform_label_probability", _strict_float(
            self.uniform_label_probability, "proposal uniform_label_probability",
            lower=0.0, upper=1.0, lower_open=True, upper_open=True,
        ))
        object.__setattr__(self, "catalog_uniform_floor", _strict_float(
            self.catalog_uniform_floor, "proposal catalog_uniform_floor",
            lower=0.0, upper=1.0, upper_open=True,
        ))
        object.__setattr__(self, "proposal_sha256", _strict_sha256(
            self.proposal_sha256, "proposal_sha256",
        ))
        object.__setattr__(self, "_label_to_index", {
            int(label): index for index, label in enumerate(self.catalog.labels)
        })

    @property
    def k(self):
        return int(self.coordinates.logical_dimension)

    @property
    def stabilizer_dimension(self):
        return int(self.coordinates.stabilizer_dimension)

    def logical_bits_from_coordinates(self, coordinate):
        load_exp101()
        from exp101_certified_src.gf2 import gf2_matmul

        logical = coordinate[self.stabilizer_dimension:]
        delta = gf2_matmul(self.logical_signature_matrix, logical[:, None])[:, 0]
        return delta ^ _uint64_to_bits(self.reference_label, self.k)

    def label_from_coordinates(self, coordinate):
        coordinate = _as_bits(coordinate, ndim=1, name="coordinate")
        if coordinate.shape != (self.coordinates.dimension,):
            raise ValueError("coordinate length mismatch")
        return _bits_to_uint64(self.logical_bits_from_coordinates(coordinate))

    def logical_coordinates_for_label(self, label):
        load_exp101()
        from exp101_certified_src.gf2 import gf2_matmul

        delta = (
            _uint64_to_bits(label, self.k)
            ^ _uint64_to_bits(self.reference_label, self.k)
        )
        return gf2_matmul(self.logical_signature_inverse, delta[:, None])[:, 0]

    def _catalog_index(self, label):
        return self._label_to_index.get(int(np.uint64(label)))

    def _log_label_probability(self, label):
        log_uniform = math.log(float(self.uniform_label_probability)) - self.k * math.log(2.0)
        index = self._catalog_index(label)
        if index is None:
            return log_uniform
        log_catalog = (
            math.log1p(-float(self.uniform_label_probability))
            + math.log(float(self.catalog_probabilities[index]))
        )
        return _logsumexp((log_uniform, log_catalog))

    def log_probability_coordinates(self, coordinate):
        coordinate = _as_bits(coordinate, ndim=1, name="coordinate")
        if coordinate.shape != (self.coordinates.dimension,):
            raise ValueError("coordinate length mismatch")
        label = self.label_from_coordinates(coordinate)
        index = self._catalog_index(label)
        stabilizer = coordinate[:self.stabilizer_dimension]
        if index is None:
            distance = int(np.count_nonzero(stabilizer))
        else:
            distance = int(np.count_nonzero(
                stabilizer ^ self.center_stabilizer_coordinates[index],
            ))
        terms = []
        for theta, weight in zip(self.theta_stabilizer, self.component_weights):
            theta = float(theta)
            terms.append(
                math.log(float(weight))
                + distance * math.log(theta)
                + (self.stabilizer_dimension - distance) * math.log1p(-theta)
            )
        result = self._log_label_probability(label) + _logsumexp(terms)
        if not math.isfinite(result):
            raise LogicalStratifiedConflictError("proposal log density is non-finite")
        return result

    @staticmethod
    def _categorical(rng, probabilities):
        threshold = rng.random()
        cumulative = 0.0
        for index, probability in enumerate(probabilities):
            cumulative += float(probability)
            if threshold < cumulative:
                return index
        return len(probabilities) - 1

    def sample(self, rng):
        catalog_branch = rng.random() >= float(self.uniform_label_probability)
        if catalog_branch:
            anchor_index = self._categorical(rng, self.catalog_probabilities)
            label = self.catalog.labels[anchor_index]
            center = self.center_stabilizer_coordinates[anchor_index]
        else:
            anchor_index = -1
            raw = np.uint64(rng.next_uint64())
            if self.k < 64:
                raw &= (np.uint64(1) << np.uint64(self.k)) - np.uint64(1)
            label = raw
            catalog_index = self._catalog_index(label)
            center = (
                np.zeros(self.stabilizer_dimension, dtype=np.uint8)
                if catalog_index is None
                else self.center_stabilizer_coordinates[catalog_index]
            )
        component = self._categorical(rng, self.component_weights)
        theta = float(self.theta_stabilizer[component])
        stabilizer = np.asarray(center, dtype=np.uint8).copy()
        for bit in range(stabilizer.size):
            if rng.random() < theta:
                stabilizer[bit] ^= np.uint8(1)
        logical = self.logical_coordinates_for_label(label)
        coordinate = np.concatenate((stabilizer, logical)).astype(np.uint8, copy=False)
        state = self.coordinates.state_from_coordinates(coordinate)
        return {
            "state": state,
            "coordinate": coordinate,
            "label": np.uint64(label),
            "log_q": self.log_probability_coordinates(coordinate),
            "anchor_index": int(anchor_index),
            "component_index": int(component),
        }


def build_logical_stratified_proposal(
        model, frame, catalog, *, p, uniform_label_probability=0.01,
        catalog_uniform_floor=0.02,
        theta_stabilizer=DEFAULT_THETA_STABILIZER,
        component_weights=DEFAULT_COMPONENT_WEIGHTS, alpha_temperature=1.0):
    load_exp101()
    from exp101_certified_src.gf2 import gf2_inverse, gf2_matmul

    validate_stratified_anchor_catalog(
        model, frame,
        (model.H_check.astype(np.int64) @ catalog.anchors[0].astype(np.int64) % 2).astype(np.uint8),
        catalog,
    )
    p = _strict_float(p, "p", lower=0.0, upper=0.5,
                      lower_open=True, upper_open=True)
    uniform_label_probability = _strict_float(
        uniform_label_probability, "uniform_label_probability", lower=0.0,
        upper=1.0, lower_open=True, upper_open=True,
    )
    catalog_uniform_floor = _strict_float(
        catalog_uniform_floor, "catalog_uniform_floor", lower=0.0,
        upper=1.0, upper_open=True,
    )
    alpha_temperature = _strict_float(
        alpha_temperature, "alpha_temperature", lower=0.0, lower_open=True,
    )
    if alpha_temperature not in (0.5, 1.0):
        raise ValueError("alpha_temperature must be pre-registered as 0.5 or 1.0")

    coordinates = build_affine_coordinate_system(model, catalog.anchors[0])
    split = coordinates.stabilizer_dimension
    centers = np.asarray(
        [coordinates.coordinates_of_state(anchor) for anchor in catalog.anchors],
        dtype=np.uint8,
    )
    if np.unique(centers[:, split:], axis=0).shape[0] != catalog.size:
        raise LogicalStratifiedConflictError("catalog anchors do not occupy unique labels")
    logical_basis = coordinates.basis[split:]
    signature_matrix = gf2_matmul(frame.W_basis, logical_basis.T)
    signature_inverse = gf2_inverse(signature_matrix)
    reference_label = _label_uint64(frame, catalog.anchors[0])
    replayed_labels = []
    for center in centers:
        delta = gf2_matmul(signature_matrix, center[split:, None])[:, 0]
        replayed_labels.append(
            reference_label ^ _bits_to_uint64(delta)
        )
    if not np.array_equal(np.asarray(replayed_labels, dtype=np.uint64), catalog.labels):
        raise LogicalStratifiedConflictError("affine coordinate/label replay failed")

    theta = _as_float_array(
        theta_stabilizer, ndim=1, name="theta_stabilizer",
    )
    omega = _as_float_array(
        component_weights, ndim=1, name="component_weights",
    )
    if (theta.ndim != 1 or omega.shape != theta.shape or theta.size == 0
            or not np.all(np.isfinite(theta)) or np.any(theta <= 0.0)
            or np.any(theta >= 1.0) or not np.all(np.isfinite(omega))
            or np.any(omega <= 0.0) or abs(float(omega.sum()) - 1.0) > 1e-14):
        raise ValueError("invalid stabilizer component mixture")
    omega = omega / float(omega.sum())
    omega[-1] = 1.0 - float(omega[:-1].sum())

    log_b = math.log(p / (1.0 - p))
    relative = catalog.anchor_weights.astype(np.float64) - float(catalog.anchor_weights.min())
    mode = np.exp(relative * log_b * alpha_temperature)
    mode /= float(mode.sum())
    alpha = (
        (1.0 - float(catalog_uniform_floor)) * mode
        + float(catalog_uniform_floor) / catalog.size
    )
    alpha /= float(alpha.sum())
    alpha[-1] = 1.0 - float(alpha[:-1].sum())
    if np.any(alpha <= 0.0) or not np.all(np.isfinite(alpha)):
        raise LogicalStratifiedConflictError("catalog label probabilities are invalid")

    digest = _sha256_arrays(
        STRATIFIED_PROPOSAL_VERSION,
        (signature_matrix, signature_inverse,
         np.packbits(centers[:, :split], axis=1, bitorder="little"),
         alpha.astype(">f8"), theta.astype(">f8"), omega.astype(">f8")),
        (catalog.catalog_sha256, coordinates.coordinate_sha256, p,
         alpha_temperature, uniform_label_probability, catalog_uniform_floor,
         int(reference_label)),
    )
    return LogicalStratifiedProposal(
        coordinates=coordinates,
        catalog=catalog,
        reference_label=reference_label,
        logical_signature_matrix=np.ascontiguousarray(signature_matrix),
        logical_signature_inverse=np.ascontiguousarray(signature_inverse),
        center_stabilizer_coordinates=np.ascontiguousarray(centers[:, :split]),
        catalog_probabilities=np.ascontiguousarray(alpha),
        theta_stabilizer=np.ascontiguousarray(theta),
        component_weights=np.ascontiguousarray(omega),
        p=p,
        alpha_temperature=alpha_temperature,
        uniform_label_probability=uniform_label_probability,
        catalog_uniform_floor=catalog_uniform_floor,
        proposal_sha256=digest,
    )


def validate_logical_stratified_proposal(model, frame, syndrome, catalog, proposal):
    """Rebuild a proposal from frozen scalar choices and compare every byte."""
    if not isinstance(proposal, LogicalStratifiedProposal):
        raise TypeError("proposal must be LogicalStratifiedProposal")
    validate_stratified_anchor_catalog(model, frame, syndrome, catalog)
    if proposal.catalog.catalog_sha256 != catalog.catalog_sha256:
        raise LogicalStratifiedConflictError("proposal catalog binding changed")
    rebuilt = build_logical_stratified_proposal(
        model, frame, catalog, p=proposal.p,
        uniform_label_probability=proposal.uniform_label_probability,
        catalog_uniform_floor=proposal.catalog_uniform_floor,
        theta_stabilizer=proposal.theta_stabilizer,
        component_weights=proposal.component_weights,
        alpha_temperature=proposal.alpha_temperature,
    )
    coordinate_fields = (
        "H_check", "reference_anchor", "basis", "pivot_columns",
        "pivot_inverse", "packed_reference", "packed_basis",
    )
    proposal_fields = (
        "logical_signature_matrix", "logical_signature_inverse",
        "center_stabilizer_coordinates", "catalog_probabilities",
        "theta_stabilizer", "component_weights",
    )
    if (rebuilt.proposal_sha256 != proposal.proposal_sha256
            or rebuilt.coordinates.coordinate_sha256
            != proposal.coordinates.coordinate_sha256
            or any(not np.array_equal(
                getattr(rebuilt.coordinates, name),
                getattr(proposal.coordinates, name),
            ) for name in coordinate_fields)
            or any(not np.array_equal(getattr(rebuilt, name), getattr(proposal, name))
                   for name in proposal_fields)):
        raise LogicalStratifiedConflictError("proposal SHA/content replay failed")
    return True


def catalog_character_probability_mass(proposal, character_masks):
    """Return plus/minus label mass of frozen catalog mixture diagnostics.

    The uniform-label branch contributes exactly one half to every nonzero
    character.  This calculation therefore remains O(number of anchors) even
    for the k=64 code, and exposes a catalog that is rank-complete but nearly
    one-sided in a logical direction.
    """
    if not isinstance(proposal, LogicalStratifiedProposal):
        raise TypeError("proposal must be LogicalStratifiedProposal")
    masks = _as_integer_array(
        character_masks, ndim=1, name="character_masks", dtype=np.uint64,
        maximum=(1 << 64) - 1,
    )
    k_mask = np.uint64((1 << proposal.k) - 1) if proposal.k < 64 else np.uint64(
        0xFFFFFFFFFFFFFFFF,
    )
    if (not masks.size or np.any(masks == np.uint64(0))
            or np.any(masks & ~k_mask)
            or np.unique(masks).size != masks.size):
        raise ValueError("character masks are invalid for this logical dimension")
    labels = proposal.catalog.labels
    plus = np.empty(masks.size, dtype=np.float64)
    minus = np.empty(masks.size, dtype=np.float64)
    for index, mask in enumerate(masks):
        parity = np.asarray([
            (int(label & mask).bit_count() & 1) for label in labels
        ], dtype=np.uint8)
        catalog_plus = float(proposal.catalog_probabilities[parity == 0].sum())
        plus[index] = (
            float(proposal.uniform_label_probability) * 0.5
            + (1.0 - float(proposal.uniform_label_probability)) * catalog_plus
        )
        minus[index] = 1.0 - plus[index]
    return plus, minus


@dataclass(frozen=True)
class LogicalStratifiedFrozenArtifact:
    """Fully bound input to the public LSI-IMH trajectory runner."""

    descriptor: dict
    classical_H: np.ndarray
    syndrome: np.ndarray
    codebook: LogicalSignatureCodebook
    transcript: DecodedCandidateTranscript
    catalog: StratifiedAnchorCatalog
    proposal: LogicalStratifiedProposal

    def __post_init__(self):
        if not isinstance(self.descriptor, dict):
            raise TypeError("stratified artifact descriptor must be a dict")
        object.__setattr__(self, "classical_H", _readonly(_as_bits(
            self.classical_H, ndim=2, name="artifact classical_H",
        ), np.uint8))
        object.__setattr__(self, "syndrome", _readonly(_as_bits(
            self.syndrome, ndim=1, name="artifact syndrome",
        ), np.uint8))
        if (not isinstance(self.codebook, LogicalSignatureCodebook)
                or not isinstance(self.transcript, DecodedCandidateTranscript)
                or not isinstance(self.catalog, StratifiedAnchorCatalog)
                or not isinstance(self.proposal, LogicalStratifiedProposal)):
            raise TypeError("stratified artifact components have the wrong type")


def _logical_stratified_artifact_descriptor(model, frame, H, syndrome, codebook,
                                            transcript, catalog, proposal,
                                            identity):
    if not isinstance(identity, dict):
        raise TypeError("stratified artifact identity must be a dict")
    identity_json = canonical_json(identity)
    return {
        "artifact_version": STRATIFIED_ARTIFACT_VERSION,
        "identity": json.loads(identity_json),
        "identity_sha256": sha256_json(identity),
        "classical_matrix_sha256": codebook.classical_matrix_sha256,
        "model_fingerprint": model.fingerprint(),
        "frame_fingerprint": frame.fingerprint(),
        "matrix_syndrome_sha256": _matrix_syndrome_sha256(model, syndrome),
        "codebook_sha256": codebook.codebook_sha256,
        "transcript_sha256": transcript.transcript_sha256,
        "catalog_sha256": catalog.catalog_sha256,
        "coordinate_sha256": proposal.coordinates.coordinate_sha256,
        "proposal_sha256": proposal.proposal_sha256,
        "p": float(proposal.p),
        "alpha_temperature": float(proposal.alpha_temperature),
        "catalog_anchor_count": int(catalog.size),
        "requested_max_anchors": int(catalog.requested_max_anchors),
    }


def build_logical_stratified_frozen_artifact(
        model, frame, H, syndrome, codebook, catalog, proposal, *, identity):
    """Bind a complete transcript-derived catalog to its immutable proposal."""
    if catalog.candidate_transcript is None:
        raise LogicalStratifiedConflictError(
            "frozen LSI artifact requires a complete decoded candidate transcript",
        )
    transcript = catalog.candidate_transcript
    validate_bplsd_stratified_catalog(
        model, frame, H, syndrome, codebook, catalog.anchors[0], transcript,
        catalog, max_anchors=catalog.requested_max_anchors,
    )
    validate_logical_stratified_proposal(model, frame, syndrome, catalog, proposal)
    descriptor = _logical_stratified_artifact_descriptor(
        model, frame, H, syndrome, codebook, transcript, catalog, proposal, identity,
    )
    artifact = LogicalStratifiedFrozenArtifact(
        descriptor=descriptor, classical_H=H, syndrome=syndrome,
        codebook=codebook, transcript=transcript, catalog=catalog,
        proposal=proposal,
    )
    validate_logical_stratified_frozen_artifact(model, frame, artifact)
    return artifact


def validate_logical_stratified_frozen_artifact(model, frame, artifact):
    if not isinstance(artifact, LogicalStratifiedFrozenArtifact):
        raise TypeError("artifact must be LogicalStratifiedFrozenArtifact")
    descriptor = artifact.descriptor
    if (set(descriptor) != {
            "artifact_version", "identity", "identity_sha256",
            "classical_matrix_sha256", "model_fingerprint", "frame_fingerprint",
            "matrix_syndrome_sha256", "codebook_sha256", "transcript_sha256",
            "catalog_sha256", "coordinate_sha256", "proposal_sha256", "p",
            "alpha_temperature", "catalog_anchor_count",
            "requested_max_anchors",
        }
            or descriptor["artifact_version"] != STRATIFIED_ARTIFACT_VERSION
            or not isinstance(descriptor["identity"], dict)
            or descriptor["identity_sha256"] != sha256_json(descriptor["identity"])):
        raise LogicalStratifiedConflictError("stratified artifact descriptor changed")
    validate_hgp_signature_codebook(model, frame, artifact.classical_H, artifact.codebook)
    validate_decoded_candidate_transcript(
        model, frame, artifact.classical_H, artifact.syndrome, artifact.codebook,
        artifact.catalog.anchors[0], artifact.transcript,
    )
    validate_bplsd_stratified_catalog(
        model, frame, artifact.classical_H, artifact.syndrome, artifact.codebook,
        artifact.catalog.anchors[0], artifact.transcript, artifact.catalog,
        max_anchors=artifact.catalog.requested_max_anchors,
    )
    validate_logical_stratified_proposal(
        model, frame, artifact.syndrome, artifact.catalog, artifact.proposal,
    )
    expected = _logical_stratified_artifact_descriptor(
        model, frame, artifact.classical_H, artifact.syndrome, artifact.codebook,
        artifact.transcript, artifact.catalog, artifact.proposal,
        descriptor["identity"],
    )
    if descriptor != expected:
        raise LogicalStratifiedConflictError("stratified artifact component binding changed")
    return True


def _artifact_content_sha256(metadata, arrays):
    digest = hashlib.sha256(STRATIFIED_ARTIFACT_VERSION.encode("ascii") + b"\0")
    digest.update(canonical_json(metadata).encode("ascii") + b"\0")
    for name in sorted(arrays):
        value = np.ascontiguousarray(np.asarray(arrays[name]))
        if value.dtype.hasobject:
            raise LogicalStratifiedConflictError("artifact cannot contain object arrays")
        digest.update(name.encode("ascii") + b"\0")
        digest.update(value.dtype.str.encode("ascii") + b"\0")
        digest.update(np.asarray(value.shape, dtype=">u8").tobytes())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _artifact_arrays(artifact):
    codebook = artifact.codebook
    transcript = artifact.transcript
    catalog = artifact.catalog
    proposal = artifact.proposal
    coordinates = proposal.coordinates
    return {
        "classical_H": np.asarray(artifact.classical_H, dtype=np.uint8),
        "syndrome": np.asarray(artifact.syndrome, dtype=np.uint8),
        "codebook_signatures": np.asarray(codebook.signatures, dtype=np.uint64),
        "codebook_move_weights": np.asarray(
            codebook.logical_move_weights, dtype=np.int32,
        ),
        "codebook_generator_kind": np.asarray(codebook.generator_kind, dtype=np.uint8),
        "codebook_moves_packed": np.asarray(codebook.moves_packed, dtype=np.uint8),
        "transcript_signatures": np.asarray(
            transcript.candidate_signatures, dtype=np.uint64,
        ),
        "transcript_move_weights": np.asarray(
            transcript.candidate_move_weights, dtype=np.int32,
        ),
        "transcript_valid": np.asarray(transcript.valid, dtype=np.uint8),
        "transcript_decoded_weights": np.asarray(
            transcript.decoded_weights, dtype=np.int32,
        ),
        "transcript_decoded_packed": np.asarray(
            transcript.decoded_packed, dtype=np.uint8,
        ),
        "catalog_anchors": np.asarray(catalog.anchors, dtype=np.uint8),
        "catalog_labels": np.asarray(catalog.labels, dtype=np.uint64),
        "catalog_label_deltas": np.asarray(catalog.label_deltas, dtype=np.uint64),
        "catalog_anchor_weights": np.asarray(catalog.anchor_weights, dtype=np.int32),
        "catalog_logical_move_weights": np.asarray(
            catalog.logical_move_weights, dtype=np.int32,
        ),
        "catalog_selection_role": np.asarray(catalog.selection_role, dtype=np.uint8),
        "catalog_candidate_indices": np.asarray(
            catalog.candidate_indices, dtype=np.int32,
        ),
        "coordinate_H_check": np.asarray(coordinates.H_check, dtype=np.uint8),
        "coordinate_reference_anchor": np.asarray(
            coordinates.reference_anchor, dtype=np.uint8,
        ),
        "coordinate_basis": np.asarray(coordinates.basis, dtype=np.uint8),
        "coordinate_pivot_columns": np.asarray(
            coordinates.pivot_columns, dtype=np.int32,
        ),
        "coordinate_pivot_inverse": np.asarray(
            coordinates.pivot_inverse, dtype=np.uint8,
        ),
        "coordinate_packed_reference": np.asarray(
            coordinates.packed_reference, dtype=np.uint8,
        ),
        "coordinate_packed_basis": np.asarray(coordinates.packed_basis, dtype=np.uint8),
        "proposal_signature_matrix": np.asarray(
            proposal.logical_signature_matrix, dtype=np.uint8,
        ),
        "proposal_signature_inverse": np.asarray(
            proposal.logical_signature_inverse, dtype=np.uint8,
        ),
        "proposal_centers": np.asarray(
            proposal.center_stabilizer_coordinates, dtype=np.uint8,
        ),
        "proposal_catalog_probabilities": np.asarray(
            proposal.catalog_probabilities, dtype=np.float64,
        ),
        "proposal_theta_stabilizer": np.asarray(
            proposal.theta_stabilizer, dtype=np.float64,
        ),
        "proposal_component_weights": np.asarray(
            proposal.component_weights, dtype=np.float64,
        ),
    }


def _artifact_metadata(artifact):
    codebook = artifact.codebook
    transcript = artifact.transcript
    catalog = artifact.catalog
    proposal = artifact.proposal
    coordinates = proposal.coordinates
    return {
        "artifact_version": STRATIFIED_ARTIFACT_VERSION,
        "descriptor": artifact.descriptor,
        "codebook_logical_side": int(codebook.logical_side),
        "codebook_combination_order": int(codebook.combination_order),
        "codebook_rank2_seed_count": int(codebook.rank2_seed_count),
        "transcript_base_label": int(transcript.base_label),
        "transcript_decoder_identity": transcript.decoder_identity,
        "transcript_decoder_max_iter": int(transcript.decoder_max_iter),
        "transcript_chunk_size": int(transcript.chunk_size),
        "catalog_candidate_count": int(catalog.candidate_count),
        "catalog_requested_max_anchors": int(catalog.requested_max_anchors),
        "catalog_decoder_identity": catalog.decoder_identity,
        "coordinate_stabilizer_dimension": int(coordinates.stabilizer_dimension),
        "coordinate_logical_dimension": int(coordinates.logical_dimension),
        "proposal_reference_label": int(proposal.reference_label),
        "proposal_uniform_label_probability": float(
            proposal.uniform_label_probability,
        ),
        "proposal_catalog_uniform_floor": float(proposal.catalog_uniform_floor),
    }


def write_logical_stratified_frozen_artifact(path, model, frame, artifact):
    """Write a pickle-free immutable artifact; existing paths are rejected."""
    from pathlib import Path

    validate_logical_stratified_frozen_artifact(model, frame, artifact)
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"stratified artifact already exists: {path}")
    metadata = _artifact_metadata(artifact)
    arrays = _artifact_arrays(artifact)
    content_sha256 = _artifact_content_sha256(metadata, arrays)
    atomic_npz(
        path, metadata_json=np.array(canonical_json(metadata)),
        artifact_content_sha256=np.array(content_sha256), **arrays,
    )
    return {
        "artifact_file_sha256": sha256_file(path),
        "artifact_content_sha256": content_sha256,
        "descriptor": artifact.descriptor,
    }


def load_logical_stratified_frozen_artifact(path, model, frame):
    """Load, reconstruct, and fully validate a pickle-free frozen artifact."""
    expected_arrays = {
        "classical_H", "syndrome", "codebook_signatures", "codebook_move_weights",
        "codebook_generator_kind", "codebook_moves_packed", "transcript_signatures",
        "transcript_move_weights", "transcript_valid", "transcript_decoded_weights",
        "transcript_decoded_packed", "catalog_anchors", "catalog_labels",
        "catalog_label_deltas", "catalog_anchor_weights",
        "catalog_logical_move_weights", "catalog_selection_role",
        "catalog_candidate_indices", "coordinate_H_check",
        "coordinate_reference_anchor", "coordinate_basis",
        "coordinate_pivot_columns", "coordinate_pivot_inverse",
        "coordinate_packed_reference", "coordinate_packed_basis",
        "proposal_signature_matrix", "proposal_signature_inverse", "proposal_centers",
        "proposal_catalog_probabilities", "proposal_theta_stabilizer",
        "proposal_component_weights",
    }
    try:
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != {
                    "metadata_json", "artifact_content_sha256", *expected_arrays}:
                raise LogicalStratifiedConflictError("artifact array schema changed")
            metadata_json = str(data["metadata_json"].item())
            metadata = json.loads(metadata_json)
            if canonical_json(metadata) != metadata_json:
                raise LogicalStratifiedConflictError("artifact metadata is noncanonical")
            arrays = {name: data[name].copy() for name in expected_arrays}
            if any(value.dtype.hasobject for value in arrays.values()):
                raise LogicalStratifiedConflictError("artifact contains object arrays")
            content_sha256 = str(data["artifact_content_sha256"].item())
    except LogicalStratifiedConflictError:
        raise
    except Exception as exc:
        raise LogicalStratifiedConflictError(
            f"stratified artifact cannot be loaded: {exc}",
        ) from exc
    if _artifact_content_sha256(metadata, arrays) != content_sha256:
        raise LogicalStratifiedConflictError("artifact content SHA changed")
    required_metadata = {
        "artifact_version", "descriptor", "codebook_logical_side",
        "codebook_combination_order", "codebook_rank2_seed_count",
        "transcript_base_label", "transcript_decoder_identity",
        "transcript_decoder_max_iter", "transcript_chunk_size",
        "catalog_candidate_count", "catalog_requested_max_anchors",
        "catalog_decoder_identity", "coordinate_stabilizer_dimension",
        "coordinate_logical_dimension", "proposal_reference_label",
        "proposal_uniform_label_probability", "proposal_catalog_uniform_floor",
    }
    if set(metadata) != required_metadata or metadata["artifact_version"] != STRATIFIED_ARTIFACT_VERSION:
        raise LogicalStratifiedConflictError("artifact metadata schema changed")
    codebook = LogicalSignatureCodebook(
        signatures=arrays["codebook_signatures"],
        logical_move_weights=arrays["codebook_move_weights"],
        generator_kind=arrays["codebook_generator_kind"],
        moves_packed=arrays["codebook_moves_packed"],
        classical_matrix_sha256=metadata["descriptor"]["classical_matrix_sha256"],
        model_fingerprint=metadata["descriptor"]["model_fingerprint"],
        frame_fingerprint=metadata["descriptor"]["frame_fingerprint"],
        codebook_sha256=metadata["descriptor"]["codebook_sha256"],
        logical_side=metadata["codebook_logical_side"],
        combination_order=metadata["codebook_combination_order"],
        rank2_seed_count=metadata["codebook_rank2_seed_count"],
    )
    transcript = DecodedCandidateTranscript(
        candidate_signatures=arrays["transcript_signatures"],
        candidate_move_weights=arrays["transcript_move_weights"],
        valid=arrays["transcript_valid"],
        decoded_weights=arrays["transcript_decoded_weights"],
        decoded_packed=arrays["transcript_decoded_packed"],
        codebook_sha256=codebook.codebook_sha256,
        model_fingerprint=codebook.model_fingerprint,
        frame_fingerprint=codebook.frame_fingerprint,
        matrix_syndrome_sha256=metadata["descriptor"]["matrix_syndrome_sha256"],
        base_label=np.uint64(metadata["transcript_base_label"]),
        decoder_identity=metadata["transcript_decoder_identity"],
        decoder_max_iter=metadata["transcript_decoder_max_iter"],
        chunk_size=metadata["transcript_chunk_size"],
        transcript_sha256=metadata["descriptor"]["transcript_sha256"],
    )
    catalog = StratifiedAnchorCatalog(
        anchors=arrays["catalog_anchors"], labels=arrays["catalog_labels"],
        label_deltas=arrays["catalog_label_deltas"],
        anchor_weights=arrays["catalog_anchor_weights"],
        logical_move_weights=arrays["catalog_logical_move_weights"],
        selection_role=arrays["catalog_selection_role"],
        candidate_count=metadata["catalog_candidate_count"],
        requested_max_anchors=metadata["catalog_requested_max_anchors"],
        candidate_indices=arrays["catalog_candidate_indices"],
        decoder_identity=metadata["catalog_decoder_identity"],
        codebook_sha256=codebook.codebook_sha256,
        candidate_transcript_sha256=transcript.transcript_sha256,
        catalog_sha256=metadata["descriptor"]["catalog_sha256"],
        candidate_transcript=transcript,
    )
    coordinates = AffineCoordinateSystem(
        H_check=arrays["coordinate_H_check"],
        reference_anchor=arrays["coordinate_reference_anchor"],
        basis=arrays["coordinate_basis"],
        stabilizer_dimension=metadata["coordinate_stabilizer_dimension"],
        logical_dimension=metadata["coordinate_logical_dimension"],
        pivot_columns=arrays["coordinate_pivot_columns"],
        pivot_inverse=arrays["coordinate_pivot_inverse"],
        packed_reference=arrays["coordinate_packed_reference"],
        packed_basis=arrays["coordinate_packed_basis"],
        coordinate_sha256=metadata["descriptor"]["coordinate_sha256"],
    )
    proposal = LogicalStratifiedProposal(
        coordinates=coordinates, catalog=catalog,
        reference_label=np.uint64(metadata["proposal_reference_label"]),
        logical_signature_matrix=arrays["proposal_signature_matrix"],
        logical_signature_inverse=arrays["proposal_signature_inverse"],
        center_stabilizer_coordinates=arrays["proposal_centers"],
        catalog_probabilities=arrays["proposal_catalog_probabilities"],
        theta_stabilizer=arrays["proposal_theta_stabilizer"],
        component_weights=arrays["proposal_component_weights"],
        p=metadata["descriptor"]["p"],
        alpha_temperature=metadata["descriptor"]["alpha_temperature"],
        uniform_label_probability=metadata["proposal_uniform_label_probability"],
        catalog_uniform_floor=metadata["proposal_catalog_uniform_floor"],
        proposal_sha256=metadata["descriptor"]["proposal_sha256"],
    )
    artifact = LogicalStratifiedFrozenArtifact(
        descriptor=metadata["descriptor"], classical_H=arrays["classical_H"],
        syndrome=arrays["syndrome"], codebook=codebook, transcript=transcript,
        catalog=catalog, proposal=proposal,
    )
    validate_logical_stratified_frozen_artifact(model, frame, artifact)
    if _artifact_metadata(artifact) != metadata or any(
            not np.array_equal(arrays[name], value)
            for name, value in _artifact_arrays(artifact).items()):
        raise LogicalStratifiedConflictError("artifact reconstruction changed")
    return artifact


@dataclass(frozen=True)
class LogicalStratifiedConfig:
    p: float
    burn_steps: int
    measurement_steps: int
    alpha_temperature: float = 1.0
    method_id: str = STRATIFIED_METHOD_ID

    def __post_init__(self):
        if self.method_id != STRATIFIED_METHOD_ID:
            raise ValueError("unknown logical-stratified method")
        object.__setattr__(self, "p", _strict_float(
            self.p, "p", lower=0.0, upper=0.5,
            lower_open=True, upper_open=True,
        ))
        object.__setattr__(self, "burn_steps", _strict_positive_int(
            self.burn_steps, "burn_steps",
        ))
        object.__setattr__(self, "measurement_steps", _strict_positive_int(
            self.measurement_steps, "measurement_steps",
        ))
        alpha_temperature = _strict_float(
            self.alpha_temperature, "alpha_temperature", lower=0.0,
            lower_open=True,
        )
        if alpha_temperature not in (0.5, 1.0):
            raise ValueError("alpha_temperature must be one of 0.5 or 1.0")
        object.__setattr__(self, "alpha_temperature", alpha_temperature)
        if self.measurement_steps % 8:
            raise ValueError("measurement_steps must divide into eight blocks")

    def as_dict(self):
        return {
            "method_id": self.method_id,
            "p": float(self.p),
            "burn_steps": int(self.burn_steps),
            "measurement_steps": int(self.measurement_steps),
            "alpha_temperature": float(self.alpha_temperature),
        }


@dataclass(frozen=True)
class LogicalStratifiedSeedIdentity:
    source_commit: str
    config_sha256: str
    registry_sha256: str
    cell_fingerprint: str
    init_family: str
    trajectory_index: int
    resource_tier: str
    trajectory_namespace: str

    def __post_init__(self):
        object.__setattr__(self, "source_commit", _strict_git_sha(
            self.source_commit, "source_commit",
        ))
        for name in ("config_sha256", "registry_sha256", "cell_fingerprint"):
            object.__setattr__(self, name, _strict_sha256(getattr(self, name), name))
        if self.init_family not in ("P", "U", "L"):
            raise ValueError("logical-stratified initialization family must be P, U, or L")
        object.__setattr__(self, "trajectory_index", _strict_nonnegative_int(
            self.trajectory_index, "trajectory_index",
        ))
        if not isinstance(self.resource_tier, str) or not self.resource_tier:
            raise ValueError("logical-stratified resource tier is empty")
        if (not isinstance(self.trajectory_namespace, str)
                or not self.trajectory_namespace):
            raise ValueError("logical-stratified trajectory namespace is empty")

    def as_dict(self):
        return {
            "source_commit": self.source_commit,
            "config_sha256": self.config_sha256,
            "registry_sha256": self.registry_sha256,
            "cell_fingerprint": self.cell_fingerprint,
            "method_id": STRATIFIED_METHOD_ID,
            "init_family": self.init_family,
            "trajectory_index": int(self.trajectory_index),
            "resource_tier": self.resource_tier,
            "trajectory_namespace": self.trajectory_namespace,
        }

    def seed(self, stage):
        from .seeds import derive_seed

        return derive_seed(
            STRATIFIED_VERSION, self.trajectory_namespace,
            self.source_commit, self.config_sha256, self.registry_sha256,
            self.cell_fingerprint, STRATIFIED_METHOD_ID, self.resource_tier,
            self.init_family, int(self.trajectory_index), str(stage),
        )


def _run_stage(proposal, frame, p, state, coordinate, rng, steps):
    state = state.copy()
    coordinate = coordinate.copy()
    weight = int(state.sum())
    label = proposal.label_from_coordinates(coordinate)
    log_q = proposal.log_probability_coordinates(coordinate)
    physical_width = (state.size + 7) // 8
    coordinate_width = (coordinate.size + 7) // 8
    transcript = {
        "proposal_coordinates_packed": np.empty((steps, coordinate_width), dtype=np.uint8),
        "proposal_states_packed": np.empty((steps, physical_width), dtype=np.uint8),
        "proposal_labels": np.empty(steps, dtype=np.uint64),
        "proposal_weights": np.empty(steps, dtype=np.int32),
        "proposal_log_q": np.empty(steps, dtype=np.float64),
        "current_log_q_before": np.empty(steps, dtype=np.float64),
        "log_acceptance": np.empty(steps, dtype=np.float64),
        "accept_uniform": np.empty(steps, dtype=np.float64),
        "accepted": np.empty(steps, dtype=np.uint8),
        "state_changed": np.empty(steps, dtype=np.uint8),
        "label_changed": np.empty(steps, dtype=np.uint8),
        "proposal_anchor_index": np.empty(steps, dtype=np.int16),
        "proposal_component_index": np.empty(steps, dtype=np.int8),
        "states_packed": np.empty((steps, physical_width), dtype=np.uint8),
        "labels": np.empty(steps, dtype=np.uint64),
        "weights": np.empty(steps, dtype=np.int32),
    }
    for step in range(int(steps)):
        draw = proposal.sample(rng)
        proposed_state = draw["state"]
        proposed_coordinate = draw["coordinate"]
        proposed_label = np.uint64(draw["label"])
        proposed_weight = int(proposed_state.sum())
        proposed_log_q = float(draw["log_q"])
        log_acceptance = independence_log_acceptance(
            p, weight, proposed_weight, log_q, proposed_log_q,
        )
        uniform = rng.random()
        accepted = uniform == 0.0 or math.log(uniform) < log_acceptance
        state_changed = accepted and not np.array_equal(proposed_state, state)
        label_changed = state_changed and proposed_label != label
        transcript["proposal_coordinates_packed"][step] = np.packbits(
            proposed_coordinate, bitorder="little",
        )
        transcript["proposal_states_packed"][step] = np.packbits(
            proposed_state, bitorder="little",
        )
        transcript["proposal_labels"][step] = proposed_label
        transcript["proposal_weights"][step] = proposed_weight
        transcript["proposal_log_q"][step] = proposed_log_q
        transcript["current_log_q_before"][step] = log_q
        transcript["log_acceptance"][step] = log_acceptance
        transcript["accept_uniform"][step] = uniform
        transcript["accepted"][step] = np.uint8(accepted)
        transcript["state_changed"][step] = np.uint8(state_changed)
        transcript["label_changed"][step] = np.uint8(label_changed)
        transcript["proposal_anchor_index"][step] = int(draw["anchor_index"])
        transcript["proposal_component_index"][step] = int(draw["component_index"])
        if accepted:
            state = proposed_state
            coordinate = proposed_coordinate
            label = proposed_label
            weight = proposed_weight
            log_q = proposed_log_q
        transcript["states_packed"][step] = np.packbits(state, bitorder="little")
        transcript["labels"][step] = label
        transcript["weights"][step] = weight
    return state, coordinate, transcript


def _validate_stage_transcript(proposal, frame, p, syndrome, initial_state,
                               initial_coordinate, transcript):
    """Algebraically replay one stored IMH stage before writing raw evidence."""
    steps = int(transcript["accepted"].size)
    if steps <= 0:
        raise LogicalStratifiedConflictError("empty trajectory stage")
    n = int(initial_state.size)
    dimension = int(initial_coordinate.size)
    proposed_states = np.unpackbits(
        transcript["proposal_states_packed"], axis=1, count=n,
        bitorder="little",
    ).astype(np.uint8, copy=False)
    proposed_coordinates = np.unpackbits(
        transcript["proposal_coordinates_packed"], axis=1, count=dimension,
        bitorder="little",
    ).astype(np.uint8, copy=False)
    stored_states = np.unpackbits(
        transcript["states_packed"], axis=1, count=n, bitorder="little",
    ).astype(np.uint8, copy=False)
    if (proposed_states.shape != (steps, n)
            or proposed_coordinates.shape != (steps, dimension)
            or stored_states.shape != (steps, n)):
        raise LogicalStratifiedConflictError("trajectory stage shape changed")
    current_state = np.asarray(initial_state, dtype=np.uint8).copy()
    current_coordinate = np.asarray(initial_coordinate, dtype=np.uint8).copy()
    current_label = proposal.label_from_coordinates(current_coordinate)
    current_weight = int(current_state.sum())
    current_log_q = proposal.log_probability_coordinates(current_coordinate)
    for step in range(steps):
        proposed_coordinate = proposed_coordinates[step]
        proposed_state = proposed_states[step]
        if (not np.array_equal(
                proposal.coordinates.state_from_coordinates(proposed_coordinate),
                proposed_state,
            )
                or _parity_residual(
                    proposal.coordinates.H_check, proposed_state, syndrome,
                ).any()):
            raise LogicalStratifiedConflictError("proposal left the hard coset")
        proposed_label = proposal.label_from_coordinates(proposed_coordinate)
        proposed_weight = int(proposed_state.sum())
        proposed_log_q = proposal.log_probability_coordinates(proposed_coordinate)
        log_acceptance = independence_log_acceptance(
            p, current_weight, proposed_weight, current_log_q, proposed_log_q,
        )
        uniform = float(transcript["accept_uniform"][step])
        accepted = uniform == 0.0 or math.log(uniform) < log_acceptance
        state_changed = accepted and not np.array_equal(proposed_state, current_state)
        label_changed = state_changed and proposed_label != current_label
        if (np.uint64(transcript["proposal_labels"][step]) != proposed_label
                or int(transcript["proposal_weights"][step]) != proposed_weight
                or float(transcript["proposal_log_q"][step]) != proposed_log_q
                or float(transcript["current_log_q_before"][step]) != current_log_q
                or float(transcript["log_acceptance"][step]) != log_acceptance
                or int(transcript["accepted"][step]) != int(accepted)
                or int(transcript["state_changed"][step]) != int(state_changed)
                or int(transcript["label_changed"][step]) != int(label_changed)):
            raise LogicalStratifiedConflictError("trajectory decision replay failed")
        if accepted:
            current_state = proposed_state.copy()
            current_coordinate = proposed_coordinate.copy()
            current_label = proposed_label
            current_weight = proposed_weight
            current_log_q = proposed_log_q
        if (not np.array_equal(stored_states[step], current_state)
                or np.uint64(transcript["labels"][step]) != current_label
                or int(transcript["weights"][step]) != current_weight):
            raise LogicalStratifiedConflictError("trajectory state replay failed")
    return current_state, current_coordinate


def _run_logical_stratified_trajectory_impl(
        model, frame, syndrome, config, seed_identity, initial_state, *,
        proposal, artifact_descriptor):
    if not isinstance(config, LogicalStratifiedConfig):
        raise TypeError("config must be LogicalStratifiedConfig")
    if not isinstance(seed_identity, LogicalStratifiedSeedIdentity):
        raise TypeError("seed_identity must be LogicalStratifiedSeedIdentity")
    if not isinstance(proposal, LogicalStratifiedProposal):
        raise TypeError("proposal must be LogicalStratifiedProposal")
    if config.method_id != STRATIFIED_METHOD_ID:
        raise LogicalStratifiedConflictError("config method changed")
    if (config.p != proposal.p
            or config.alpha_temperature != proposal.alpha_temperature):
        raise LogicalStratifiedConflictError("config/proposal thermodynamic binding changed")
    validate_observable_frame(model, frame)
    y = _as_bits(syndrome, ndim=1, name="syndrome")
    state = _as_bits(initial_state, ndim=1, name="initial_state")
    if (y.shape != (model.num_checks,) or state.shape != (model.num_qubits,)
            or _parity_residual(model.H_check, state, y).any()):
        raise LogicalStratifiedConflictError("initial state is outside the hard coset")
    coordinate = proposal.coordinates.coordinates_of_state(state)
    if proposal.label_from_coordinates(coordinate) != _label_uint64(frame, state):
        raise LogicalStratifiedConflictError("initial affine label replay failed")
    load_exp101()
    from exp101_certified_src.prng import PortablePrng

    initial = state.copy()
    initial_coordinate = coordinate.copy()
    state, coordinate, burn = _run_stage(
        proposal, frame, config.p, state, coordinate,
        PortablePrng(seed_identity.seed("burn")), config.burn_steps,
    )
    burn_state = state.copy()
    burn_coordinate = coordinate.copy()
    state, coordinate, measurement = _run_stage(
        proposal, frame, config.p, state, coordinate,
        PortablePrng(seed_identity.seed("measurement")), config.measurement_steps,
    )
    checked_burn_state, checked_burn_coordinate = _validate_stage_transcript(
        proposal, frame, config.p, y, initial, initial_coordinate, burn,
    )
    checked_final_state, checked_final_coordinate = _validate_stage_transcript(
        proposal, frame, config.p, y, burn_state, burn_coordinate, measurement,
    )
    if (not np.array_equal(checked_burn_state, burn_state)
            or not np.array_equal(checked_burn_coordinate, burn_coordinate)
            or not np.array_equal(checked_final_state, state)
            or not np.array_equal(checked_final_coordinate, coordinate)):
        raise LogicalStratifiedConflictError("trajectory stage boundary replay failed")
    unpacked = np.unpackbits(
        measurement["states_packed"], axis=1, count=model.num_qubits,
        bitorder="little",
    ).astype(np.uint8, copy=False)
    residuals = (
        model.H_check.astype(np.int64) @ unpacked.T.astype(np.int64) % 2
    ).T.astype(np.uint8) ^ y[None, :]
    replayed_labels = np.asarray(
        [_label_uint64(frame, row) for row in unpacked], dtype=np.uint64,
    )
    if (residuals.any()
            or not np.array_equal(unpacked.sum(axis=1), measurement["weights"])
            or not np.array_equal(replayed_labels, measurement["labels"])):
        raise LogicalStratifiedConflictError("trajectory raw replay failed")
    raw = {
        "raw_version": STRATIFIED_RAW_VERSION,
        "method_id": STRATIFIED_METHOD_ID,
        "sampler_config_json": canonical_json(config.as_dict()),
        "sampler_config_sha256": sha256_json(config.as_dict()),
        "seed_identity_json": canonical_json(seed_identity.as_dict()),
        "artifact_descriptor_json": canonical_json(artifact_descriptor),
        "artifact_identity_sha256": artifact_descriptor["identity_sha256"],
        "model_fingerprint": model.fingerprint(),
        "frame_fingerprint": frame.fingerprint(),
        "matrix_syndrome_sha256": _matrix_syndrome_sha256(model, y),
        "codebook_sha256": artifact_descriptor["codebook_sha256"],
        "candidate_transcript_sha256": artifact_descriptor["transcript_sha256"],
        "proposal_sha256": proposal.proposal_sha256,
        "catalog_sha256": proposal.catalog.catalog_sha256,
        "coordinate_sha256": proposal.coordinates.coordinate_sha256,
        "initial_state_packed": np.packbits(initial, bitorder="little"),
        "initial_coordinate_packed": np.packbits(initial_coordinate, bitorder="little"),
        "burn_state_packed": np.packbits(burn_state, bitorder="little"),
        "burn_coordinate_packed": np.packbits(burn_coordinate, bitorder="little"),
        "final_state_packed": np.packbits(state, bitorder="little"),
        "final_coordinate_packed": np.packbits(coordinate, bitorder="little"),
        "burn_seed": np.uint64(seed_identity.seed("burn")),
        "measurement_seed": np.uint64(seed_identity.seed("measurement")),
        **{f"burn_{name}": value for name, value in burn.items()},
        **{f"measurement_{name}": value for name, value in measurement.items()},
        "measurement_residual_weights": residuals.sum(axis=1).astype(np.int32),
        "measurement_block": np.repeat(
            np.arange(8, dtype=np.int8), config.measurement_steps // 8,
        ),
        "burn_cross_label_changes": np.int64(burn["label_changed"].sum()),
        "measurement_cross_label_changes": np.int64(
            measurement["label_changed"].sum()
        ),
        "initial_label": _label_uint64(frame, initial),
        "burn_label": _label_uint64(frame, burn_state),
        "final_label": _label_uint64(frame, state),
        "engine": "reference",
    }
    raw["burn_attempts"] = np.int64(config.burn_steps)
    raw["burn_accepts"] = np.int64(burn["accepted"].sum())
    raw["burn_state_changes"] = np.int64(burn["state_changed"].sum())
    raw["measurement_attempts"] = np.int64(config.measurement_steps)
    raw["measurement_accepts"] = np.int64(measurement["accepted"].sum())
    raw["measurement_state_changes"] = np.int64(measurement["state_changed"].sum())
    return raw


def _run_logical_stratified_trajectory_from_verified_artifact(
        model, frame, syndrome, config, seed_identity, initial_state, *, artifact):
    """Private replay boundary: a bare catalog/proposal is never executable."""
    validate_logical_stratified_frozen_artifact(model, frame, artifact)
    if not np.array_equal(_as_bits(syndrome, ndim=1, name="syndrome"), artifact.syndrome):
        raise LogicalStratifiedConflictError("artifact syndrome binding changed")
    descriptor = artifact.descriptor
    if (config.p != descriptor["p"]
            or config.alpha_temperature != descriptor["alpha_temperature"]
            or artifact.proposal.proposal_sha256 != descriptor["proposal_sha256"]
            or artifact.catalog.catalog_sha256 != descriptor["catalog_sha256"]
            or artifact.transcript.transcript_sha256 != descriptor["transcript_sha256"]):
        raise LogicalStratifiedConflictError("artifact/config binding changed")
    for name in ("source_commit", "config_sha256", "registry_sha256", "cell_fingerprint"):
        if (name in descriptor["identity"]
                and descriptor["identity"][name] != getattr(seed_identity, name)):
            raise LogicalStratifiedConflictError("artifact/seed identity changed")
    return _run_logical_stratified_trajectory_impl(
        model, frame, syndrome, config, seed_identity, initial_state,
        proposal=artifact.proposal, artifact_descriptor=descriptor,
    )


def run_logical_stratified_trajectory(
        model, frame, syndrome, config, seed_identity, initial_state, *, artifact):
    """Run one trajectory only from an authenticated frozen artifact."""
    return _run_logical_stratified_trajectory_from_verified_artifact(
        model, frame, syndrome, config, seed_identity, initial_state,
        artifact=artifact,
    )


def replay_logical_stratified_trajectory(
        model, frame, syndrome, config, seed_identity, initial_state, raw, *,
        artifact):
    """Deterministically replay burn and measurement, including every decision."""
    if not isinstance(raw, dict):
        raise TypeError("raw must be a trajectory dictionary")
    replay = _run_logical_stratified_trajectory_from_verified_artifact(
        model, frame, syndrome, config, seed_identity, initial_state,
        artifact=artifact,
    )
    if set(raw) != set(replay):
        raise LogicalStratifiedConflictError("trajectory raw schema changed")
    for name, expected in replay.items():
        observed = raw[name]
        if isinstance(expected, np.ndarray):
            if not np.array_equal(np.asarray(observed), expected):
                raise LogicalStratifiedConflictError(
                    f"trajectory raw replay mismatch: {name}",
                )
        elif observed != expected:
            raise LogicalStratifiedConflictError(
                f"trajectory raw replay mismatch: {name}",
            )
    return True
