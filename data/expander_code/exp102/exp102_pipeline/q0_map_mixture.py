"""Multi-anchor independence-MH for the exp102 q=0 hard coset.

This module is an isolated diagnostic prototype.  It has no production or
formal-readiness entry point.  For a syndrome ``y`` it samples the exact target

    pi(e | y) proportional to lambda**|e| 1[H e = y],
    lambda = p / (1 - p).

Let ``B`` be a fixed row basis of ``ker(H)`` and let ``a0`` be one valid
anchor.  Every hard-coset state has a unique coordinate ``x`` through

    e(x) = a0 xor x B.

For frozen valid anchors ``ah = e(ch)``, component ``c`` independently toggles
stabilizer and logical coordinates with probabilities ``thetaS[c]`` and
``thetaL[c]``.  The normalized proposal is therefore

    q(e(x) | y) = sum_h alpha[h] sum_c omega[c]
        thetaS[c]**s_h (1-thetaS[c])**(r-s_h)
        thetaL[c]**l_h (1-thetaL[c])**(k-l_h),

where ``s_h`` and ``l_h`` are the two coordinate Hamming distances from
``ch``.  Every summand is a normalized product-Bernoulli law on the bijective
coordinate space.  Positive mixture weights and parameters strictly inside
``(0, 1)`` give full support, including the frozen defensive ``theta=.5``
component.  Proposal density is evaluated by this complete finite log-sum-exp;
anchor discovery frequency is never interpreted as physical mode weight.

The independent Metropolis-Hastings acceptance probability is

    min(1, lambda**(|e'|-|e|) q(e|y) / q(e'|y)).

Consequently ``pi(e) q(e') alpha(e,e')`` equals the same expression with
``e,e'`` exchanged.  Incomplete or imperfect anchor catalogs can reduce
efficiency but cannot change the stationary distribution.

Anchor construction accepts only ``(H, y, p)`` plus frozen tie-break seeds.
It cannot access a planted error.  The MILP result is always checked exactly
over GF(2); its floating-point optimality claim is diagnostic evidence, not a
proof certificate.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import warnings

import numpy as np

from .exp101_bridge import load_exp101
from .io import canonical_json, sha256_json
from .q0_global import (
    GlobalConflictError,
    reduce_logical_basis,
    validate_observable_frame,
)
from .seeds import derive_seed


MAP_MIXTURE_VERSION = "exp102.q0_map_mixture.discovery.v1"
MAP_ANCHOR_VERSION = "exp102.q0_map_mixture.anchors.v2"
MAP_PROPOSAL_VERSION = "exp102.q0_map_mixture.proposal.v1"
MAP_RAW_VERSION = "exp102.q0_map_mixture.raw.v2"
MAP_METHOD_ID = "MAM-IMH8"
MAP_ANCHOR_SEED_NAMESPACE = "q0_hgp_global_screen_map_anchor_v1"
MAP_PRIMARY_ANCHOR_SEED_SENTINEL = 0

# The small defensive component makes the full-support lower bound explicit.
DEFAULT_THETA_STABILIZER = (0.001, 0.003, 0.01, 0.04, 0.15, 0.5)
DEFAULT_THETA_LOGICAL = (0.001, 0.003, 0.02, 0.08, 0.25, 0.5)
DEFAULT_COMPONENT_WEIGHTS = (0.35, 0.30, 0.20, 0.10, 0.045, 0.005)
MILP_OPTIONS = (
    ("mip_rel_gap", 0.0),
    ("parallel", False),
    ("presolve", True),
    ("random_seed", 0),
    ("threads", 1),
)


class MapMixtureConflictError(ValueError):
    pass


def _readonly_copy(value, dtype=None):
    result = np.array(value, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _as_bits(value, *, ndim, name):
    array = np.asarray(value)
    if array.ndim != int(ndim):
        raise ValueError(f"{name} must have ndim={ndim}")
    if not np.issubdtype(array.dtype, np.bool_) and not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must be binary")
    if np.any(array < 0) or np.any(array > 1):
        raise ValueError(f"{name} must contain only zero and one")
    result = np.ascontiguousarray(array, dtype=np.uint8)
    return result


def _state_sha256(state):
    state = _as_bits(state, ndim=1, name="hashed_state")
    return hashlib.sha256(
        np.asarray(state.shape, dtype=">u8").tobytes()
        + np.packbits(state, bitorder="little").tobytes()
    ).hexdigest()


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


def _validate_probability(p):
    p = float(p)
    if not 0.0 < p < 0.5 or not math.isfinite(p):
        raise ValueError("map-mixture p must lie in (0, 0.5)")
    return p


def _parity_residual(H, state, syndrome):
    return (
        H.astype(np.int64) @ state.astype(np.int64) % 2
    ).astype(np.uint8) ^ syndrome


@dataclass(frozen=True)
class MapAnchorCatalog:
    anchors: np.ndarray
    optimum_weight: int
    requested_max_anchors: int
    tie_break_seeds: tuple[int, ...]
    p: float
    anchor_sha256: str
    anchor_state_sha256: tuple[str, ...]
    objective_sha256: tuple[str, ...]
    solver_identity: str
    solver_options: tuple[tuple[str, object], ...] = MILP_OPTIONS
    solver: str = "scipy.optimize.milp/highs"
    optimality_evidence: str = "floating_milp_optimum_exact_gf2_primal_check"
    seed_namespace: str = MAP_ANCHOR_SEED_NAMESPACE

    def __post_init__(self):
        object.__setattr__(self, "anchors", _readonly_copy(self.anchors, np.uint8))

    @property
    def size(self):
        return int(self.anchors.shape[0])


def _milp_constraints(H, syndrome, optimum_weight=None, excluded=()):
    from scipy.optimize import LinearConstraint
    from scipy.sparse import csr_matrix, eye, hstack, vstack

    checks, qubits = H.shape
    parity = hstack(
        (csr_matrix(H, dtype=np.float64), -2.0 * eye(checks, format="csr")),
        format="csr",
    )
    rows = [parity]
    lower = [syndrome.astype(np.float64)]
    upper = [syndrome.astype(np.float64)]
    if optimum_weight is not None:
        weight_row = csr_matrix(
            np.concatenate((np.ones(qubits), np.zeros(checks)))[None, :],
        )
        rows.append(weight_row)
        lower.append(np.asarray([float(optimum_weight)]))
        upper.append(np.asarray([float(optimum_weight)]))
    for anchor in excluded:
        # Hamming(e, anchor) >= 1, written as one linear inequality.
        coefficients = np.concatenate(
            ((1.0 - 2.0 * anchor.astype(np.float64)), np.zeros(checks)),
        )
        rows.append(csr_matrix(coefficients[None, :]))
        lower.append(np.asarray([1.0 - float(anchor.sum())]))
        upper.append(np.asarray([np.inf]))
    matrix = vstack(rows, format="csr")
    return LinearConstraint(matrix, np.concatenate(lower), np.concatenate(upper))


def _run_anchor_milp(H, syndrome, objective, *, optimum_weight=None, excluded=()):
    from scipy.optimize import Bounds, milp

    checks, qubits = H.shape
    variables = qubits + checks
    objective = np.asarray(objective, dtype=np.float64)
    if objective.shape != (qubits,):
        raise ValueError("anchor objective length mismatch")
    c = np.concatenate((objective, np.zeros(checks, dtype=np.float64)))
    row_degrees = H.sum(axis=1).astype(np.float64)
    bounds = Bounds(
        np.zeros(variables, dtype=np.float64),
        np.concatenate((np.ones(qubits), np.floor(row_degrees / 2.0))),
    )
    # SciPy forwards these three HiGHS-native determinism controls while
    # warning that they are not part of its narrow public option schema.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Unrecognized options detected:.*",
            category=RuntimeWarning,
        )
        result = milp(
            c=c,
            integrality=np.ones(variables, dtype=np.int8),
            bounds=bounds,
            constraints=_milp_constraints(
                H, syndrome, optimum_weight=optimum_weight, excluded=excluded,
            ),
            options=dict(MILP_OPTIONS),
        )
    return result


def _solver_identity():
    import scipy

    try:
        from scipy.optimize._highspy import _core as highs_core
        highs_version = (
            f"{highs_core.HIGHS_VERSION_MAJOR}."
            f"{highs_core.HIGHS_VERSION_MINOR}."
            f"{highs_core.HIGHS_VERSION_PATCH}"
        )
    except Exception:  # pragma: no cover - an unknown identity cannot be frozen
        highs_version = "unknown"
    return (
        f"numpy={np.__version__};scipy={scipy.__version__};"
        f"highs={highs_version}"
    )


def map_solver_identity():
    """Return the exact MILP solver identity recorded in MAP artifacts."""
    return _solver_identity()


def _validated_milp_primal(H, syndrome, objective, result, *, expected_weight=None):
    """Replay every integer variable and equality after a HiGHS optimum."""
    checks, qubits = H.shape
    if not result.success or result.x is None or int(result.status) != 0:
        raise MapMixtureConflictError(
            f"anchor MILP did not report an optimum: {result.message}"
        )
    if (not math.isfinite(float(result.fun))
            or not math.isfinite(float(result.mip_dual_bound))
            or abs(float(result.mip_gap)) > 1e-12
            or abs(float(result.fun) - float(result.mip_dual_bound)) > 1e-7):
        raise MapMixtureConflictError("anchor MILP optimum/dual/gap certificate drifted")
    rounded = np.rint(result.x).astype(np.int64)
    if (rounded.shape != (qubits + checks,)
            or np.max(np.abs(result.x - rounded)) > 1e-7):
        raise MapMixtureConflictError("anchor MILP returned non-integral variables")
    error = rounded[:qubits]
    parity_integer = rounded[qubits:]
    if (np.any(error < 0) or np.any(error > 1) or np.any(parity_integer < 0)
            or np.any(parity_integer > (H.sum(axis=1) // 2))):
        raise MapMixtureConflictError("anchor MILP integer bounds failed replay")
    equality = H.astype(np.int64) @ error - 2 * parity_integer
    if not np.array_equal(equality, syndrome.astype(np.int64)):
        raise MapMixtureConflictError("anchor MILP integer parity equations failed replay")
    replay_objective = float(np.dot(np.asarray(objective, dtype=np.float64), error))
    if abs(replay_objective - float(result.fun)) > 1e-7:
        raise MapMixtureConflictError("anchor MILP objective failed exact-primal replay")
    if expected_weight is not None and int(error.sum()) != int(expected_weight):
        raise MapMixtureConflictError("tie-break anchor changed the minimum weight")
    return error.astype(np.uint8)


def _tie_break_seed(H, syndrome, p, slot):
    matrix_sha = hashlib.sha256(
        np.asarray(H.shape, dtype=">u8").tobytes()
        + np.packbits(H, axis=1, bitorder="little").tobytes()
    ).hexdigest()
    syndrome_sha = hashlib.sha256(
        np.packbits(syndrome, bitorder="little").tobytes()
    ).hexdigest()
    return derive_seed(
        MAP_ANCHOR_SEED_NAMESPACE, matrix_sha, syndrome_sha,
        format(float(p), ".17g"), int(slot),
    )


def build_milp_map_anchors(H_check, syndrome, p, *, max_anchors=8,
                           tie_break_seeds=None):
    """Build distinct minimum-weight anchors without a planted-error input.

    The first MILP obtains the common minimum Hamming weight.  Subsequent
    solves fix that weight, use seed-derived deterministic linear tie-break
    objectives, and add exact no-good inequalities for prior anchors.
    """
    H = _as_bits(H_check, ndim=2, name="H_check")
    y = _as_bits(syndrome, ndim=1, name="syndrome")
    p = _validate_probability(p)
    if y.shape != (H.shape[0],):
        raise ValueError("syndrome length mismatch")
    if isinstance(max_anchors, bool) or int(max_anchors) <= 0:
        raise ValueError("max_anchors must be positive")
    max_anchors = int(max_anchors)

    primary_objective = np.ones(H.shape[1], dtype=np.float64)
    primary = _run_anchor_milp(H, y, primary_objective)
    primary_anchor = _validated_milp_primal(
        H, y, primary_objective, primary,
    )
    optimum_weight = int(primary_anchor.sum())

    # The unit-objective optimum is already a canonical, exactly checked MAP
    # anchor.  Keeping it as slot zero avoids a redundant and potentially very
    # expensive tie-break solve when a cell freezes a one-anchor catalog.
    if tie_break_seeds is None:
        seeds = tuple(
            _tie_break_seed(H, y, p, slot)
            for slot in range(max_anchors - 1)
        )
    else:
        seeds = tuple(int(seed) for seed in tie_break_seeds)
        if len(seeds) < max_anchors - 1:
            raise ValueError("not enough frozen tie-break seeds")
        seeds = seeds[:max_anchors - 1]

    load_exp101()
    from exp101_certified_src.prng import PortablePrng

    anchors = [primary_anchor]
    used_seeds = [MAP_PRIMARY_ANCHOR_SEED_SENTINEL]
    objective_hashes = [
        hashlib.sha256(primary_objective.astype(">f8").tobytes()).hexdigest()
    ]
    for seed in seeds:
        permutation = PortablePrng(seed).permutation(H.shape[1])
        ranks = np.empty(H.shape[1], dtype=np.float64)
        ranks[permutation] = np.arange(1, H.shape[1] + 1, dtype=np.float64)
        result = _run_anchor_milp(
            H, y, ranks, optimum_weight=optimum_weight, excluded=anchors,
        )
        if int(result.status) == 2:  # all minimum-weight states were excluded
            break
        anchor = _validated_milp_primal(
            H, y, ranks, result, expected_weight=optimum_weight,
        )
        if _parity_residual(H, anchor, y).any():
            raise MapMixtureConflictError("anchor MILP GF(2) replay failed")
        packed = np.packbits(anchor, bitorder="little").tobytes()
        if any(np.packbits(old, bitorder="little").tobytes() == packed for old in anchors):
            raise MapMixtureConflictError("anchor no-good constraint returned a duplicate")
        anchors.append(anchor)
        used_seeds.append(seed)
        objective_hashes.append(hashlib.sha256(ranks.astype(">f8").tobytes()).hexdigest())
    if not anchors:
        raise MapMixtureConflictError("anchor catalog is empty")
    anchor_array = np.ascontiguousarray(anchors, dtype=np.uint8)
    state_hashes = tuple(
        _state_sha256(anchor)
        for anchor in anchor_array
    )
    solver_identity = _solver_identity()
    if "highs=unknown" in solver_identity:
        raise MapMixtureConflictError("anchor MILP solver identity is unknown")
    # Solver identity is audited separately; the scientific catalog digest is
    # portable when two environments return identical ordered anchors.
    digest = _sha256_arrays(
        MAP_ANCHOR_VERSION,
        (np.packbits(H, axis=1, bitorder="little"),
         np.packbits(y, bitorder="little"),
         np.packbits(anchor_array, axis=1, bitorder="little"),
         np.asarray(used_seeds, dtype=">u8")),
        (MAP_ANCHOR_SEED_NAMESPACE, H.shape, y.shape, anchor_array.shape,
         format(p, ".17g"),
         optimum_weight, max_anchors, repr(MILP_OPTIONS), *state_hashes,
         *objective_hashes),
    )
    return MapAnchorCatalog(
        anchors=anchor_array,
        optimum_weight=optimum_weight,
        requested_max_anchors=max_anchors,
        tie_break_seeds=tuple(used_seeds),
        p=p,
        anchor_sha256=digest,
        anchor_state_sha256=state_hashes,
        objective_sha256=tuple(objective_hashes),
        solver_identity=solver_identity,
    )


def validate_map_anchor_catalog(H_check, syndrome, p, catalog, *,
                                requested_max_anchors=8,
                                require_current_solver_identity=True):
    """Recompute every catalog identity from its current, immutable contents."""
    if not isinstance(require_current_solver_identity, (bool, np.bool_)):
        raise ValueError("solver-identity replay mode must be boolean")
    H = _as_bits(H_check, ndim=2, name="H_check")
    y = _as_bits(syndrome, ndim=1, name="syndrome")
    p = _validate_probability(p)
    if y.shape != (H.shape[0],):
        raise ValueError("syndrome length mismatch")
    anchors = _as_bits(catalog.anchors, ndim=2, name="anchors")
    if (anchors.shape[1:] != (H.shape[1],) or anchors.shape[0] < 1
            or int(catalog.requested_max_anchors) != int(requested_max_anchors)
            or anchors.shape[0] > int(requested_max_anchors)
            or float(catalog.p) != p):
        raise MapMixtureConflictError("anchor catalog shape/config binding changed")
    solver_identity = str(catalog.solver_identity)
    if (len(catalog.tie_break_seeds) != anchors.shape[0]
            or len(catalog.anchor_state_sha256) != anchors.shape[0]
            or len(catalog.objective_sha256) != anchors.shape[0]
            or catalog.solver_options != MILP_OPTIONS
            or catalog.seed_namespace != MAP_ANCHOR_SEED_NAMESPACE
            or not solver_identity
            or "highs=unknown" in solver_identity
            or (bool(require_current_solver_identity)
                and solver_identity != _solver_identity())):
        raise MapMixtureConflictError("anchor catalog solver identity changed")
    residuals = (
        H.astype(np.int64) @ anchors.T.astype(np.int64) % 2
    ).T.astype(np.uint8) ^ y[None, :]
    if residuals.any() or np.any(anchors.sum(axis=1) != int(catalog.optimum_weight)):
        raise MapMixtureConflictError("anchor catalog primal replay failed")
    expected_seeds = (
        MAP_PRIMARY_ANCHOR_SEED_SENTINEL,
        *(
            _tie_break_seed(H, y, p, slot)
            for slot in range(max(0, anchors.shape[0] - 1))
        ),
    )
    if tuple(catalog.tie_break_seeds) != expected_seeds:
        raise MapMixtureConflictError("anchor catalog tie seeds changed")
    load_exp101()
    from exp101_certified_src.prng import PortablePrng

    objective_hashes = [
        hashlib.sha256(
            np.ones(H.shape[1], dtype=">f8").tobytes()
        ).hexdigest()
    ]
    for seed in expected_seeds[1:]:
        permutation = PortablePrng(seed).permutation(H.shape[1])
        ranks = np.empty(H.shape[1], dtype=np.float64)
        ranks[permutation] = np.arange(1, H.shape[1] + 1, dtype=np.float64)
        objective_hashes.append(
            hashlib.sha256(ranks.astype(">f8").tobytes()).hexdigest()
        )
    state_hashes = tuple(_state_sha256(anchor) for anchor in anchors)
    if (tuple(catalog.anchor_state_sha256) != state_hashes
            or tuple(catalog.objective_sha256) != tuple(objective_hashes)):
        raise MapMixtureConflictError("anchor catalog component hashes changed")
    digest = _sha256_arrays(
        MAP_ANCHOR_VERSION,
        (np.packbits(H, axis=1, bitorder="little"),
         np.packbits(y, bitorder="little"),
         np.packbits(anchors, axis=1, bitorder="little"),
         np.asarray(expected_seeds, dtype=">u8")),
        (MAP_ANCHOR_SEED_NAMESPACE, H.shape, y.shape, anchors.shape,
         format(p, ".17g"),
         int(catalog.optimum_weight), int(requested_max_anchors),
         repr(MILP_OPTIONS), *state_hashes, *objective_hashes),
    )
    if digest != catalog.anchor_sha256:
        raise MapMixtureConflictError("anchor catalog SHA failed replay")
    return True


@dataclass(frozen=True)
class AffineCoordinateSystem:
    H_check: np.ndarray
    reference_anchor: np.ndarray
    basis: np.ndarray
    stabilizer_dimension: int
    logical_dimension: int
    pivot_columns: np.ndarray
    pivot_inverse: np.ndarray
    packed_reference: np.ndarray
    packed_basis: np.ndarray
    coordinate_sha256: str

    def __post_init__(self):
        for name, dtype in (
            ("H_check", np.uint8), ("reference_anchor", np.uint8),
            ("basis", np.uint8), ("pivot_columns", np.int32),
            ("pivot_inverse", np.uint8), ("packed_reference", np.uint8),
            ("packed_basis", np.uint8),
        ):
            object.__setattr__(self, name, _readonly_copy(getattr(self, name), dtype))

    @property
    def dimension(self):
        return int(self.basis.shape[0])

    @property
    def num_qubits(self):
        return int(self.basis.shape[1])

    def coordinates_of_state(self, state):
        load_exp101()
        from exp101_certified_src.gf2 import gf2_matmul

        state = _as_bits(state, ndim=1, name="state")
        if state.shape != (self.num_qubits,):
            raise ValueError("state length mismatch")
        delta_pivots = (state ^ self.reference_anchor)[self.pivot_columns]
        coordinates = gf2_matmul(delta_pivots[None, :], self.pivot_inverse)[0]
        if not np.array_equal(self.state_from_coordinates(coordinates), state):
            raise MapMixtureConflictError("state is outside the affine coordinate system")
        return np.ascontiguousarray(coordinates)

    def state_packed_from_coordinates(self, coordinates):
        coordinates = _as_bits(coordinates, ndim=1, name="coordinates")
        if coordinates.shape != (self.dimension,):
            raise ValueError("coordinate length mismatch")
        selected = np.flatnonzero(coordinates)
        packed = self.packed_reference.copy()
        if selected.size:
            packed ^= np.bitwise_xor.reduce(self.packed_basis[selected], axis=0)
        return packed

    def state_from_coordinates(self, coordinates):
        return np.unpackbits(
            self.state_packed_from_coordinates(coordinates), bitorder="little",
            count=self.num_qubits,
        ).astype(np.uint8, copy=False)


def build_affine_coordinate_system(model, reference_anchor):
    """Freeze ``[independent stabilizers; reduced logicals]`` as ``ker(H)``."""
    load_exp101()
    from exp101_certified_src.gf2 import (
        gf2_inverse, gf2_matmul, gf2_rank, gf2_row_echelon,
    )

    H = _as_bits(model.H_check, ndim=2, name="H_check")
    anchor = _as_bits(reference_anchor, ndim=1, name="reference_anchor")
    if anchor.shape != (model.num_qubits,):
        raise ValueError("reference anchor length mismatch")
    if _parity_residual(H, anchor, gf2_matmul(H, anchor[:, None])[:, 0]).any():
        raise AssertionError("internal affine-anchor check failed")
    stabilizers = _as_bits(model.stabilizer_rows, ndim=2, name="stabilizers")
    logicals = reduce_logical_basis(model.logical_move_basis)
    basis = np.ascontiguousarray(np.vstack((stabilizers, logicals)), dtype=np.uint8)
    rank_H = int(gf2_rank(H))
    expected = model.num_qubits - rank_H
    if (int(gf2_rank(stabilizers)) != stabilizers.shape[0]
            or int(gf2_rank(logicals)) != logicals.shape[0]
            or int(gf2_rank(basis)) != expected
            or basis.shape[0] != expected
            or gf2_matmul(H, basis.T).any()):
        raise MapMixtureConflictError("stabilizer/logical rows are not a kernel basis")
    _, pivots = gf2_row_echelon(basis)
    if len(pivots) != expected:
        raise MapMixtureConflictError("kernel coordinate pivots are incomplete")
    pivots = np.asarray(pivots, dtype=np.int32)
    inverse = gf2_inverse(basis[:, pivots])
    digest = _sha256_arrays(
        "exp102.q0_map_mixture.coordinates.v1",
        (np.packbits(H, axis=1, bitorder="little"),
         np.packbits(anchor, bitorder="little"),
         np.packbits(basis, axis=1, bitorder="little"),
         pivots.astype(">i4"), np.packbits(inverse, axis=1, bitorder="little")),
        (stabilizers.shape[0], logicals.shape[0]),
    )
    return AffineCoordinateSystem(
        H_check=H,
        reference_anchor=anchor.copy(),
        basis=basis,
        stabilizer_dimension=int(stabilizers.shape[0]),
        logical_dimension=int(logicals.shape[0]),
        pivot_columns=pivots,
        pivot_inverse=np.ascontiguousarray(inverse),
        packed_reference=np.packbits(anchor, bitorder="little"),
        packed_basis=np.packbits(basis, axis=1, bitorder="little"),
        coordinate_sha256=digest,
    )


@dataclass(frozen=True)
class MapMixtureProposal:
    coordinates: AffineCoordinateSystem
    anchor_catalog: MapAnchorCatalog
    anchor_centers: np.ndarray
    anchor_weights: np.ndarray
    theta_stabilizer: np.ndarray
    theta_logical: np.ndarray
    component_weights: np.ndarray
    proposal_sha256: str

    def __post_init__(self):
        for name in (
            "anchor_centers", "anchor_weights", "theta_stabilizer",
            "theta_logical", "component_weights",
        ):
            object.__setattr__(self, name, _readonly_copy(getattr(self, name)))

    @property
    def num_anchors(self):
        return int(self.anchor_centers.shape[0])

    @property
    def num_components(self):
        return int(self.component_weights.size)

    def log_probability_coordinates(self, coordinate):
        coordinate = _as_bits(coordinate, ndim=1, name="coordinate")
        if coordinate.shape != (self.coordinates.dimension,):
            raise ValueError("coordinate length mismatch")
        split = self.coordinates.stabilizer_dimension
        distances = np.count_nonzero(self.anchor_centers ^ coordinate[None, :], axis=1)
        distances_s = np.count_nonzero(
            self.anchor_centers[:, :split] ^ coordinate[None, :split], axis=1,
        )
        distances_l = distances - distances_s
        r = split
        k = self.coordinates.logical_dimension
        terms = []
        for anchor in range(self.num_anchors):
            for component in range(self.num_components):
                ts = float(self.theta_stabilizer[component])
                tl = float(self.theta_logical[component])
                terms.append(
                    math.log(float(self.anchor_weights[anchor]))
                    + math.log(float(self.component_weights[component]))
                    + int(distances_s[anchor]) * math.log(ts)
                    + (r - int(distances_s[anchor])) * math.log1p(-ts)
                    + int(distances_l[anchor]) * math.log(tl)
                    + (k - int(distances_l[anchor])) * math.log1p(-tl)
                )
        maximum = max(terms)
        result = maximum + math.log(sum(math.exp(value - maximum) for value in terms))
        if not math.isfinite(result):
            raise MapMixtureConflictError("map-mixture log proposal is non-finite")
        return result

    def log_probability_state(self, state):
        return self.log_probability_coordinates(
            self.coordinates.coordinates_of_state(state),
        )

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
        anchor = self._categorical(rng, self.anchor_weights)
        component = self._categorical(rng, self.component_weights)
        coordinate = self.anchor_centers[anchor].copy()
        split = self.coordinates.stabilizer_dimension
        theta_s = float(self.theta_stabilizer[component])
        theta_l = float(self.theta_logical[component])
        for bit in range(split):
            if rng.random() < theta_s:
                coordinate[bit] ^= np.uint8(1)
        for bit in range(split, coordinate.size):
            if rng.random() < theta_l:
                coordinate[bit] ^= np.uint8(1)
        state = self.coordinates.state_from_coordinates(coordinate)
        return {
            "state": state,
            "coordinate": coordinate,
            "log_q": self.log_probability_coordinates(coordinate),
            "anchor_index": int(anchor),
            "component_index": int(component),
        }


def build_map_mixture_proposal(model, anchor_catalog, *,
                               theta_stabilizer=DEFAULT_THETA_STABILIZER,
                               theta_logical=DEFAULT_THETA_LOGICAL,
                               component_weights=DEFAULT_COMPONENT_WEIGHTS,
                               anchor_weights=None):
    anchors = _as_bits(anchor_catalog.anchors, ndim=2, name="anchors")
    if anchors.shape[1] != model.num_qubits:
        raise ValueError("anchor length/model mismatch")
    syndrome = (
        model.H_check.astype(np.int64) @ anchors[0].astype(np.int64) % 2
    ).astype(np.uint8)
    residuals = (
        model.H_check.astype(np.int64) @ anchors.T.astype(np.int64) % 2
    ).T.astype(np.uint8) ^ syndrome[None, :]
    if residuals.any():
        raise MapMixtureConflictError("anchors do not share one hard syndrome")
    coordinates = build_affine_coordinate_system(model, anchors[0])
    centers = np.asarray(
        [coordinates.coordinates_of_state(anchor) for anchor in anchors],
        dtype=np.uint8,
    )
    theta_s = np.asarray(theta_stabilizer, dtype=np.float64)
    theta_l = np.asarray(theta_logical, dtype=np.float64)
    omega = np.asarray(component_weights, dtype=np.float64)
    if theta_s.ndim != 1 or theta_l.shape != theta_s.shape or omega.shape != theta_s.shape:
        raise ValueError("map-mixture component arrays must have equal one-dimensional shape")
    if (theta_s.size == 0 or not np.all(np.isfinite(theta_s))
            or np.any(theta_s <= 0.0) or np.any(theta_s >= 1.0)):
        raise ValueError("all stabilizer theta values must lie strictly inside (0,1)")
    if (not np.all(np.isfinite(theta_l)) or np.any(theta_l <= 0.0)
            or np.any(theta_l >= 1.0)):
        raise ValueError("all logical theta values must lie strictly inside (0,1)")
    if (not np.all(np.isfinite(omega)) or np.any(omega <= 0.0)
            or abs(float(omega.sum()) - 1.0) > 1e-14):
        raise ValueError("component weights must be positive and normalized")
    omega = omega / float(omega.sum())
    omega[-1] = 1.0 - float(omega[:-1].sum())
    if anchor_weights is None:
        alpha = np.full(anchors.shape[0], 1.0 / anchors.shape[0], dtype=np.float64)
    else:
        alpha = np.asarray(anchor_weights, dtype=np.float64)
    if (alpha.shape != (anchors.shape[0],) or not np.all(np.isfinite(alpha))
            or np.any(alpha <= 0.0)
            or abs(float(alpha.sum()) - 1.0) > 1e-14):
        raise ValueError("anchor weights must be positive and normalized")
    alpha = alpha / float(alpha.sum())
    alpha[-1] = 1.0 - float(alpha[:-1].sum())
    digest = _sha256_arrays(
        MAP_PROPOSAL_VERSION,
        (np.packbits(centers, axis=1, bitorder="little"), alpha.astype(">f8"),
         theta_s.astype(">f8"), theta_l.astype(">f8"), omega.astype(">f8")),
        (coordinates.coordinate_sha256, anchor_catalog.anchor_sha256),
    )
    return MapMixtureProposal(
        coordinates=coordinates,
        anchor_catalog=anchor_catalog,
        anchor_centers=np.ascontiguousarray(centers),
        anchor_weights=np.ascontiguousarray(alpha),
        theta_stabilizer=np.ascontiguousarray(theta_s),
        theta_logical=np.ascontiguousarray(theta_l),
        component_weights=np.ascontiguousarray(omega),
        proposal_sha256=digest,
    )


def validate_map_mixture_proposal(model, syndrome, p, catalog, proposal, *,
                                  requested_max_anchors=8,
                                  require_current_solver_identity=True):
    """Bind MAM-IMH8 to the canonical model, catalog and frozen mixture."""
    validate_map_anchor_catalog(
        model.H_check, syndrome, p, catalog,
        requested_max_anchors=requested_max_anchors,
        require_current_solver_identity=require_current_solver_identity,
    )
    if proposal.anchor_catalog.anchor_sha256 != catalog.anchor_sha256:
        raise MapMixtureConflictError("proposal/anchor digest mismatch")
    expected = build_map_mixture_proposal(model, catalog)
    coordinate_fields = (
        "H_check", "reference_anchor", "basis", "pivot_columns",
        "pivot_inverse", "packed_reference", "packed_basis",
    )
    proposal_fields = (
        "anchor_centers", "anchor_weights", "theta_stabilizer",
        "theta_logical", "component_weights",
    )
    if (proposal.coordinates.coordinate_sha256
            != expected.coordinates.coordinate_sha256
            or proposal.proposal_sha256 != expected.proposal_sha256
            or any(not np.array_equal(
                getattr(proposal.coordinates, name),
                getattr(expected.coordinates, name),
            ) for name in coordinate_fields)
            or any(not np.array_equal(
                getattr(proposal, name), getattr(expected, name),
            ) for name in proposal_fields)):
        raise MapMixtureConflictError("map-mixture proposal SHA/content replay failed")
    return True


def independence_log_acceptance(p, current_weight, proposed_weight,
                                current_log_q, proposed_log_q):
    p = _validate_probability(p)
    if (not math.isfinite(float(current_log_q))
            or not math.isfinite(float(proposed_log_q))):
        raise MapMixtureConflictError("map-mixture acceptance received non-finite log q")
    log_lambda = math.log(p / (1.0 - p))
    value = (
        (int(proposed_weight) - int(current_weight)) * log_lambda
        + float(current_log_q) - float(proposed_log_q)
    )
    if not math.isfinite(value):
        raise MapMixtureConflictError("map-mixture acceptance ratio is non-finite")
    return min(0.0, value)


@dataclass(frozen=True)
class MapMixtureConfig:
    p: float
    burn_steps: int
    measurement_steps: int
    method_id: str = MAP_METHOD_ID
    max_anchors: int = 8

    def __post_init__(self):
        _validate_probability(self.p)
        if self.method_id != MAP_METHOD_ID:
            raise ValueError("unknown map-mixture method")
        if (isinstance(self.burn_steps, bool)
                or not isinstance(self.burn_steps, (int, np.integer))
                or isinstance(self.measurement_steps, bool)
                or not isinstance(self.measurement_steps, (int, np.integer))
                or int(self.burn_steps) <= 0 or int(self.measurement_steps) <= 0):
            raise ValueError("map-mixture step counts must be positive")
        if int(self.measurement_steps) % 8:
            raise ValueError("measurement_steps must divide into eight blocks")
        if (isinstance(self.max_anchors, bool)
                or not isinstance(self.max_anchors, (int, np.integer))
                or int(self.max_anchors) not in (1, 8)):
            raise ValueError("MAM-IMH8 freezes one or eight requested anchors")

    def as_dict(self):
        return {
            "method_id": self.method_id,
            "p": float(self.p),
            "burn_steps": int(self.burn_steps),
            "measurement_steps": int(self.measurement_steps),
            "max_anchors": int(self.max_anchors),
            "theta_stabilizer": list(DEFAULT_THETA_STABILIZER),
            "theta_logical": list(DEFAULT_THETA_LOGICAL),
            "component_weights": list(DEFAULT_COMPONENT_WEIGHTS),
        }


@dataclass(frozen=True)
class MapMixtureSeedIdentity:
    source_commit: str
    config_sha256: str
    registry_sha256: str
    cell_fingerprint: str
    init_family: str
    trajectory_index: int
    trajectory_namespace: str
    resource_tier: str = "test"
    method_id: str = MAP_METHOD_ID

    def __post_init__(self):
        if len(self.source_commit) != 40 or any(c not in "0123456789abcdef" for c in self.source_commit):
            raise ValueError("map-mixture source commit must be a full lowercase Git SHA")
        for name in ("config_sha256", "registry_sha256", "cell_fingerprint"):
            value = getattr(self, name)
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"map-mixture {name} must be a lowercase SHA256")
        if self.init_family not in ("P", "U"):
            raise ValueError("map-mixture initialization family must be P or U")
        if (isinstance(self.trajectory_index, bool)
                or not isinstance(self.trajectory_index, (int, np.integer))
                or int(self.trajectory_index) < 0):
            raise ValueError("map-mixture trajectory index is invalid")
        if self.method_id != MAP_METHOD_ID:
            raise ValueError("map-mixture method identity mismatch")
        if not self.resource_tier or not self.trajectory_namespace:
            raise ValueError("map-mixture seed namespace/resource tier is empty")

    def seed(self, stage):
        return derive_seed(
            "q0_map_mixture_diagnostic_v1", self.source_commit,
            self.config_sha256, self.registry_sha256, self.cell_fingerprint,
            self.method_id, self.resource_tier, self.init_family,
            int(self.trajectory_index),
            self.trajectory_namespace, str(stage),
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


def _label_uint64(frame, state):
    if frame.k > 64:
        raise ValueError("map-mixture labels support at most 64 logical bits")
    bits = frame.label_of(state)
    value = np.uint64(0)
    for bit, entry in enumerate(bits):
        if entry:
            value |= np.uint64(1) << np.uint64(bit)
    return value


def _run_imh_stage(proposal, frame, p, state, coordinate, rng, steps):
    state = state.copy()
    coordinate = coordinate.copy()
    current_weight = int(state.sum())
    current_log_q = proposal.log_probability_coordinates(coordinate)
    packed_width = (coordinate.size + 7) // 8
    physical_width = (state.size + 7) // 8
    transcript = {
        "proposal_coordinates_packed": np.empty((steps, packed_width), dtype=np.uint8),
        "proposal_states_packed": np.empty((steps, physical_width), dtype=np.uint8),
        "proposal_weights": np.empty(steps, dtype=np.int32),
        "proposal_log_q": np.empty(steps, dtype=np.float64),
        "current_log_q_before": np.empty(steps, dtype=np.float64),
        "log_acceptance": np.empty(steps, dtype=np.float64),
        "accept_uniform": np.empty(steps, dtype=np.float64),
        "accepted": np.empty(steps, dtype=np.uint8),
        "state_changed": np.empty(steps, dtype=np.uint8),
        "proposal_anchor_index": np.empty(steps, dtype=np.int16),
        "proposal_component_index": np.empty(steps, dtype=np.int8),
        "states_packed": np.empty((steps, physical_width), dtype=np.uint8),
        "weights": np.empty(steps, dtype=np.int32),
        "labels": np.empty(steps, dtype=np.uint64),
    }
    for step in range(steps):
        draw = proposal.sample(rng)
        proposed_state = draw["state"]
        proposed_coordinate = draw["coordinate"]
        proposed_weight = int(proposed_state.sum())
        proposed_log_q = float(draw["log_q"])
        log_acceptance = independence_log_acceptance(
            p, current_weight, proposed_weight, current_log_q, proposed_log_q,
        )
        uniform = rng.random()
        accepted = uniform == 0.0 or math.log(uniform) < log_acceptance
        state_changed = accepted and not np.array_equal(proposed_state, state)
        transcript["proposal_coordinates_packed"][step] = np.packbits(
            proposed_coordinate, bitorder="little",
        )
        transcript["proposal_states_packed"][step] = np.packbits(
            proposed_state, bitorder="little",
        )
        transcript["proposal_weights"][step] = proposed_weight
        transcript["proposal_log_q"][step] = proposed_log_q
        transcript["current_log_q_before"][step] = current_log_q
        transcript["log_acceptance"][step] = log_acceptance
        transcript["accept_uniform"][step] = uniform
        transcript["accepted"][step] = np.uint8(accepted)
        transcript["state_changed"][step] = np.uint8(state_changed)
        transcript["proposal_anchor_index"][step] = draw["anchor_index"]
        transcript["proposal_component_index"][step] = draw["component_index"]
        if accepted:
            state = proposed_state
            coordinate = proposed_coordinate
            current_weight = proposed_weight
            current_log_q = proposed_log_q
        transcript["states_packed"][step] = np.packbits(state, bitorder="little")
        transcript["weights"][step] = current_weight
        transcript["labels"][step] = _label_uint64(frame, state)
    return state, coordinate, current_log_q, transcript


def run_map_mixture_trajectory(model, frame, syndrome, config, seed_identity,
                               initial_state, *, anchor_catalog=None,
                               proposal=None, frozen_artifact_replay=False):
    """Run a fixed-clock IMH trajectory and retain a replayable proposal log."""
    if not isinstance(frozen_artifact_replay, (bool, np.bool_)):
        raise ValueError("frozen MAP replay mode must be boolean")
    if config.method_id != seed_identity.method_id:
        raise MapMixtureConflictError("map-mixture config/seed method mismatch")
    y = _as_bits(syndrome, ndim=1, name="syndrome")
    state = _as_bits(initial_state, ndim=1, name="initial_state").copy()
    if y.shape != (model.num_checks,) or state.shape != (model.num_qubits,):
        raise ValueError("map-mixture state or syndrome shape mismatch")
    if frame.num_qubits != model.num_qubits or frame.k != model.k:
        raise MapMixtureConflictError("map-mixture model/frame dimensions changed")
    try:
        validate_observable_frame(model, frame)
    except GlobalConflictError as exc:
        raise MapMixtureConflictError("map-mixture observable frame mismatch") from exc
    if _parity_residual(model.H_check, state, y).any():
        raise MapMixtureConflictError("initial state is outside the requested hard coset")
    if frozen_artifact_replay and (anchor_catalog is None or proposal is None):
        raise MapMixtureConflictError(
            "frozen MAP replay requires both bound artifact objects",
        )
    if anchor_catalog is None:
        anchor_catalog = build_milp_map_anchors(
            model.H_check, y, config.p, max_anchors=config.max_anchors,
        )
    if proposal is None:
        proposal = build_map_mixture_proposal(model, anchor_catalog)
    validate_map_mixture_proposal(
        model, y, config.p, anchor_catalog, proposal,
        requested_max_anchors=config.max_anchors,
        require_current_solver_identity=not bool(frozen_artifact_replay),
    )
    coordinate = proposal.coordinates.coordinates_of_state(state)
    load_exp101()
    from exp101_certified_src.prng import PortablePrng

    initial = state.copy()
    initial_coordinate = coordinate.copy()
    state, coordinate, _, burn = _run_imh_stage(
        proposal, frame, config.p, state, coordinate,
        PortablePrng(seed_identity.seed("burn")), int(config.burn_steps),
    )
    burn_state = state.copy()
    burn_coordinate = coordinate.copy()
    state, coordinate, _, measurement = _run_imh_stage(
        proposal, frame, config.p, state, coordinate,
        PortablePrng(seed_identity.seed("measurement")),
        int(config.measurement_steps),
    )
    unpacked = np.unpackbits(
        measurement["states_packed"], axis=1, count=model.num_qubits,
        bitorder="little",
    ).astype(np.uint8, copy=False)
    residuals = (
        model.H_check.astype(np.int64) @ unpacked.T.astype(np.int64) % 2
    ).T.astype(np.uint8) ^ y[None, :]
    burn_unpacked = np.unpackbits(
        burn["states_packed"], axis=1, count=model.num_qubits,
        bitorder="little",
    ).astype(np.uint8, copy=False)
    proposed_unpacked = np.vstack((
        np.unpackbits(
            burn["proposal_states_packed"], axis=1, count=model.num_qubits,
            bitorder="little",
        ),
        np.unpackbits(
            measurement["proposal_states_packed"], axis=1,
            count=model.num_qubits, bitorder="little",
        ),
    )).astype(np.uint8, copy=False)
    all_replayed = np.vstack((burn_unpacked, unpacked, proposed_unpacked))
    all_residuals = (
        model.H_check.astype(np.int64) @ all_replayed.T.astype(np.int64) % 2
    ).T.astype(np.uint8) ^ y[None, :]
    replay_burn_labels = np.asarray(
        [_label_uint64(frame, row) for row in burn_unpacked], dtype=np.uint64,
    )
    replay_measurement_labels = np.asarray(
        [_label_uint64(frame, row) for row in unpacked], dtype=np.uint64,
    )
    if (residuals.any() or all_residuals.any()
            or not np.array_equal(burn["weights"], burn_unpacked.sum(axis=1))
            or not np.array_equal(measurement["weights"], unpacked.sum(axis=1))
            or not np.array_equal(burn["labels"], replay_burn_labels)
            or not np.array_equal(measurement["labels"], replay_measurement_labels)):
        raise MapMixtureConflictError("map-mixture trajectory replay failed")
    sampler_dict = config.as_dict()
    seed_dict = seed_identity.as_dict()
    matrix_sha256 = _sha256_arrays(
        "exp102.q0_map_mixture.H_y.v1",
        (np.packbits(model.H_check, axis=1, bitorder="little"),
         np.packbits(y, bitorder="little")),
        (model.H_check.shape, y.shape),
    )
    return {
        "raw_version": MAP_RAW_VERSION,
        "method_id": config.method_id,
        "sampler_config_json": canonical_json(sampler_dict),
        "sampler_config_sha256": sha256_json(sampler_dict),
        "seed_identity_json": canonical_json(seed_dict),
        "model_fingerprint": model.fingerprint(),
        "frame_fingerprint": frame.fingerprint(),
        "matrix_syndrome_sha256": matrix_sha256,
        "proposal_sha256": proposal.proposal_sha256,
        "anchor_sha256": anchor_catalog.anchor_sha256,
        "anchor_state_sha256": anchor_catalog.anchor_state_sha256,
        "anchor_objective_sha256": anchor_catalog.objective_sha256,
        "anchor_tie_break_seeds": np.asarray(
            anchor_catalog.tie_break_seeds, dtype=np.uint64,
        ),
        "anchor_solver_identity": anchor_catalog.solver_identity,
        "anchor_seed_namespace": anchor_catalog.seed_namespace,
        "anchor_optimum_weight": np.int32(anchor_catalog.optimum_weight),
        "anchor_count": np.int16(anchor_catalog.size),
        "coordinate_sha256": proposal.coordinates.coordinate_sha256,
        "burn_seed": np.uint64(seed_identity.seed("burn")),
        "measurement_seed": np.uint64(seed_identity.seed("measurement")),
        "initial_state_packed": np.packbits(initial, bitorder="little"),
        "initial_coordinate_packed": np.packbits(initial_coordinate, bitorder="little"),
        "burn_state_packed": np.packbits(burn_state, bitorder="little"),
        "burn_coordinate_packed": np.packbits(burn_coordinate, bitorder="little"),
        "final_state_packed": np.packbits(state, bitorder="little"),
        "final_coordinate_packed": np.packbits(coordinate, bitorder="little"),
        "burn_proposal_coordinates_packed": burn["proposal_coordinates_packed"],
        "burn_proposal_states_packed": burn["proposal_states_packed"],
        "burn_proposal_weights": burn["proposal_weights"],
        "burn_proposal_log_q": burn["proposal_log_q"],
        "burn_current_log_q_before": burn["current_log_q_before"],
        "burn_log_acceptance": burn["log_acceptance"],
        "burn_accept_uniform": burn["accept_uniform"],
        "burn_accepted": burn["accepted"],
        "burn_state_changed": burn["state_changed"],
        "burn_proposal_anchor_index": burn["proposal_anchor_index"],
        "burn_proposal_component_index": burn["proposal_component_index"],
        "burn_states_packed": burn["states_packed"],
        "burn_weights": burn["weights"],
        "burn_labels": burn["labels"],
        "measurement_proposal_coordinates_packed": measurement["proposal_coordinates_packed"],
        "measurement_proposal_states_packed": measurement["proposal_states_packed"],
        "measurement_proposal_weights": measurement["proposal_weights"],
        "measurement_proposal_log_q": measurement["proposal_log_q"],
        "measurement_current_log_q_before": measurement["current_log_q_before"],
        "measurement_log_acceptance": measurement["log_acceptance"],
        "measurement_accept_uniform": measurement["accept_uniform"],
        "measurement_accepted": measurement["accepted"],
        "measurement_state_changed": measurement["state_changed"],
        "measurement_proposal_anchor_index": measurement["proposal_anchor_index"],
        "measurement_proposal_component_index": measurement["proposal_component_index"],
        "measurement_states_packed": measurement["states_packed"],
        "measurement_labels": measurement["labels"],
        "measurement_weights": measurement["weights"],
        "measurement_residual_weights": residuals.sum(axis=1).astype(np.int32),
        "measurement_block": np.repeat(
            np.arange(8, dtype=np.int8), config.measurement_steps // 8,
        ),
        "burn_attempts": np.int64(config.burn_steps),
        "burn_accepts": np.int64(burn["accepted"].sum()),
        "burn_state_changes": np.int64(burn["state_changed"].sum()),
        "measurement_attempts": np.int64(config.measurement_steps),
        "measurement_accepts": np.int64(measurement["accepted"].sum()),
        "measurement_state_changes": np.int64(
            measurement["state_changed"].sum()
        ),
        "initial_label": _label_uint64(frame, initial),
        "burn_label": _label_uint64(frame, burn_state),
        "final_label": _label_uint64(frame, state),
        "engine": "reference",
    }


def enumerate_affine_states(coordinates, *, max_dimension=20):
    if coordinates.dimension > int(max_dimension):
        raise ValueError("affine enumeration dimension exceeds oracle cap")
    states = np.empty(
        (1 << coordinates.dimension, coordinates.num_qubits), dtype=np.uint8,
    )
    coordinate_rows = np.empty(
        (1 << coordinates.dimension, coordinates.dimension), dtype=np.uint8,
    )
    for mask in range(1 << coordinates.dimension):
        row = np.asarray(
            [(mask >> bit) & 1 for bit in range(coordinates.dimension)],
            dtype=np.uint8,
        )
        coordinate_rows[mask] = row
        states[mask] = coordinates.state_from_coordinates(row)
    return coordinate_rows, states


def independence_transition_matrix(target_probability, proposal_probability):
    """Exact finite-state IMH transition used only by tiny-code oracles."""
    pi = np.asarray(target_probability, dtype=np.float64)
    q = np.asarray(proposal_probability, dtype=np.float64)
    if (pi.ndim != 1 or q.shape != pi.shape or np.any(pi <= 0.0)
            or np.any(q <= 0.0)):
        raise ValueError("finite IMH probabilities must be positive vectors")
    if abs(float(pi.sum()) - 1.0) > 1e-13 or abs(float(q.sum()) - 1.0) > 1e-13:
        raise ValueError("finite IMH probabilities must be normalized")
    size = pi.size
    transition = np.empty((size, size), dtype=np.float64)
    for current in range(size):
        ratios = (pi * q[current]) / (pi[current] * q)
        transition[current] = q * np.minimum(1.0, ratios)
        transition[current, current] = 0.0
        transition[current, current] = 1.0 - float(transition[current].sum())
    return transition


def estimate_proposal_overlap(proposal, p, num_samples, seed):
    """Self-normalized IS and stationary-IMH diagnostics from iid q draws."""
    p = _validate_probability(p)
    if int(num_samples) <= 1:
        raise ValueError("proposal-overlap estimate needs at least two samples")
    load_exp101()
    from exp101_certified_src.prng import PortablePrng

    rng = PortablePrng(int(seed))
    log_weights = np.empty(int(num_samples), dtype=np.float64)
    physical_weights = np.empty(int(num_samples), dtype=np.int32)
    for sample in range(int(num_samples)):
        draw = proposal.sample(rng)
        weight = int(draw["state"].sum())
        physical_weights[sample] = weight
        log_weights[sample] = (
            weight * math.log(p / (1.0 - p)) - float(draw["log_q"])
        )
    shifted = np.exp(log_weights - float(log_weights.max()))
    normalized = shifted / shifted.sum()
    ess = 1.0 / float(np.dot(normalized, normalized))
    # For x~pi and y~q, w(x) min(1,w(y)/w(x)) = min(w(x),w(y)).
    # The sorted O(N) sum avoids divisions by proposal draws whose relative
    # importance underflows harmlessly to zero.
    ordered = np.sort(shifted)
    multiplicity = 1.0 + 2.0 * np.arange(
        ordered.size - 1, -1, -1, dtype=np.float64,
    )
    stationary_acceptance = float(
        np.dot(ordered, multiplicity)
        / (float(ordered.size) * float(ordered.sum()))
    )
    return {
        "num_samples": int(num_samples),
        "importance_ess": ess,
        "importance_ess_fraction": ess / float(num_samples),
        "max_normalized_weight": float(normalized.max()),
        "top10_normalized_weight": float(np.sort(normalized)[-10:].sum()),
        "weighted_mean_physical_weight": float(np.dot(normalized, physical_weights)),
        "stationary_imh_acceptance": stationary_acceptance,
        "minimum_sampled_physical_weight": int(physical_weights.min()),
    }
