"""Canonical reduced-posterior model for the exp101 expander-code study.

The production ``x_error`` sector samples the absolute candidate data error
``e`` after the preparation syndrome has been eliminated::

    pi(e | y_eff) proportional to
        exp(-K_p |e| - K_q |H_check e xor y_eff|).

``epsilon_data_true`` is ground truth.  It is used to construct ``y_eff`` and
to centre optional Mattis observables, but is never read by the Gibbs energy.
``legacy_delta_only`` is retained solely to reproduce the old repository
model.  Deprecated names remain read-compatible while all stored ensemble
names and model fields are canonical.
"""

import hashlib
import warnings
from dataclasses import dataclass, field

import numpy as np

from .gf2 import as_gf2_matrix, as_gf2_vector, gf2_matmul, gf2_rank
from .section import LinearSection, build_linear_section


PHYSICS_CONTRACT_VERSION = "exp101.physics.v2"
STATE_PREP_PROTOCOL = "plus_Zcheck_X"
SYNDROME_SEMANTICS = "effective_y"
PRODUCTION_ENSEMBLE = "true_posterior"
PRODUCTION_SECTOR = "x_error"
CANONICAL_ENSEMBLES = ("true_posterior", "legacy_delta_only")
ENSEMBLE_ALIASES = {
    "paper_true_posterior": "true_posterior",
    "repo_compat": "legacy_delta_only",
}
ENSEMBLES = CANONICAL_ENSEMBLES
ACCEPTED_ENSEMBLES = CANONICAL_ENSEMBLES + tuple(ENSEMBLE_ALIASES)
SECTORS = ("x_error", "z_error")


def normalize_ensemble(ensemble, *, warn_alias=True):
    """Return the canonical ensemble name before seeding or serialization."""
    name = str(ensemble)
    if name in CANONICAL_ENSEMBLES:
        return name
    canonical = ENSEMBLE_ALIASES.get(name)
    if canonical is None:
        raise ValueError(
            f"ensemble must be one of {CANONICAL_ENSEMBLES}; deprecated "
            f"aliases accepted: {tuple(ENSEMBLE_ALIASES)}"
        )
    if warn_alias:
        warnings.warn(
            f"ensemble {name!r} is deprecated; use {canonical!r}",
            DeprecationWarning,
            stacklevel=2,
        )
    return canonical


def coupling_from_probability(probability):
    """Return ``log((1-p)/p)``; ``p=0`` denotes a hard constraint."""
    probability = float(probability)
    if not 0.0 <= probability <= 0.5:
        raise ValueError("probability must be in [0, 0.5]")
    if probability == 0.0:
        return float("inf")
    return float(np.log((1.0 - probability) / probability))


@dataclass
class SectorModel:
    """Validated CSS data for one Pauli-error sector.

    ``x_error`` means ``H_check=H_Z``, X-stabilizer and logical-X moves,
    logical-Z characters, and the logical ``|+>`` preparation problem.  The
    ``z_error`` sector is its Hadamard-dual logical ``|0>`` problem.
    """

    sector: str
    H_check: np.ndarray
    stabilizer_rows: np.ndarray
    logical_obs_basis: np.ndarray
    logical_move_basis: np.ndarray
    logical_sector_section: LinearSection
    k: int
    num_checks: int
    num_qubits: int
    checks_touching_each_qubit: list = field(default_factory=list)

    @property
    def section(self):
        """Deprecated read alias for ``logical_sector_section``."""
        warned = self.__dict__.setdefault("_warned_legacy_aliases", set())
        if "section" not in warned:
            warnings.warn(
                "SectorModel.section is deprecated; use logical_sector_section",
                DeprecationWarning,
                stacklevel=2,
            )
            warned.add("section")
        return self.logical_sector_section

    def fingerprint(self):
        payload = (
            b"exp101.sector_model.v2\0"
            + self.sector.encode()
            + np.asarray(
                [self.num_checks, self.num_qubits, self.k], dtype=np.int64
            ).tobytes()
            + np.ascontiguousarray(self.H_check).tobytes()
            + np.ascontiguousarray(self.stabilizer_rows).tobytes()
            + np.ascontiguousarray(self.logical_obs_basis).tobytes()
            + np.ascontiguousarray(self.logical_move_basis).tobytes()
            + self.logical_sector_section.fingerprint().encode()
        )
        return hashlib.sha256(payload).hexdigest()


def build_checks_touching_each_qubit(parity_check_matrix):
    parity_check_matrix = as_gf2_matrix(parity_check_matrix)
    return [
        np.flatnonzero(parity_check_matrix[:, j]).astype(np.int32)
        for j in range(parity_check_matrix.shape[1])
    ]


def assemble_sector_model(H_X, H_Z, logicals, sector="x_error"):
    """Assemble a sector without changing the established CSS matrix wiring."""
    if sector not in SECTORS:
        raise ValueError(f"sector must be one of {SECTORS}")
    H_X = as_gf2_matrix(H_X)
    H_Z = as_gf2_matrix(H_Z)
    if sector == "x_error":
        H_check, stabilizer_rows = H_Z, H_X
        obs_basis, move_basis = logicals.logical_Z, logicals.logical_X
    else:
        H_check, stabilizer_rows = H_X, H_Z
        obs_basis, move_basis = logicals.logical_X, logicals.logical_Z

    obs_basis = as_gf2_matrix(obs_basis)
    move_basis = as_gf2_matrix(move_basis)
    k = obs_basis.shape[0]
    if move_basis.shape != (k, H_check.shape[1]):
        raise AssertionError(
            "logical move and observable bases must have matching shape"
        )
    if k and gf2_matmul(H_check, move_basis.T).any():
        raise AssertionError("logical move basis not in ker(H_check)")
    if gf2_matmul(H_check, stabilizer_rows.T).any():
        raise AssertionError("stabilizer rows not in ker(H_check) (CSS violated)")
    if gf2_matmul(stabilizer_rows, obs_basis.T).any():
        raise AssertionError(
            "logical observable basis does not annihilate stabilizers"
        )
    if k:
        pairing = gf2_matmul(move_basis, obs_basis.T)
        if not np.array_equal(pairing, np.eye(k, dtype=np.uint8)):
            raise AssertionError("move/observable bases are not pairing-normalized")

    kernel_dimension = H_check.shape[1] - gf2_rank(H_check)
    kernel_generators = np.vstack((stabilizer_rows, move_basis))
    generated_dimension = gf2_rank(kernel_generators)
    if generated_dimension != kernel_dimension:
        raise AssertionError(
            "stabilizer rows plus logical moves do not span ker(H_check): "
            f"generated rank {generated_dimension}, expected {kernel_dimension}"
        )

    logical_sector_section = build_linear_section(H_check)
    return SectorModel(
        sector=sector,
        H_check=H_check,
        stabilizer_rows=stabilizer_rows,
        logical_obs_basis=obs_basis,
        logical_move_basis=move_basis,
        logical_sector_section=logical_sector_section,
        k=k,
        num_checks=H_check.shape[0],
        num_qubits=H_check.shape[1],
        checks_touching_each_qubit=build_checks_touching_each_qubit(H_check),
    )


@dataclass(init=False)
class DisorderRealization:
    """One physical disorder draw in unambiguous reduced-model variables."""

    epsilon_data_true: np.ndarray
    measurement_error: np.ndarray
    effective_syndrome: np.ndarray
    p: float
    q: float
    epsilon_data_weight: int
    measurement_error_weight: int

    def __init__(
        self,
        epsilon_data_true=None,
        measurement_error=None,
        effective_syndrome=None,
        p=0.0,
        q=0.0,
        epsilon_data_weight=None,
        measurement_error_weight=None,
        **legacy,
    ):
        legacy_map = {
            "eta": "epsilon_data_true",
            "delta": "measurement_error",
            "observed_syndrome": "effective_syndrome",
            "eta_weight": "epsilon_data_weight",
            "delta_weight": "measurement_error_weight",
        }
        supplied = {
            "epsilon_data_true": epsilon_data_true,
            "measurement_error": measurement_error,
            "effective_syndrome": effective_syndrome,
            "epsilon_data_weight": epsilon_data_weight,
            "measurement_error_weight": measurement_error_weight,
        }
        for old_name, value in legacy.items():
            canonical = legacy_map.get(old_name)
            if canonical is None:
                raise TypeError(f"unexpected DisorderRealization argument {old_name!r}")
            if supplied[canonical] is not None:
                raise TypeError(
                    f"cannot pass both {canonical!r} and deprecated {old_name!r}"
                )
            warnings.warn(
                f"{old_name} is deprecated; use {canonical}",
                DeprecationWarning,
                stacklevel=2,
            )
            supplied[canonical] = value

        missing = [
            name for name in (
                "epsilon_data_true", "measurement_error", "effective_syndrome"
            ) if supplied[name] is None
        ]
        if missing:
            raise TypeError(f"missing required disorder fields: {', '.join(missing)}")
        self.epsilon_data_true = as_gf2_vector(supplied["epsilon_data_true"]).copy()
        self.measurement_error = as_gf2_vector(supplied["measurement_error"]).copy()
        self.effective_syndrome = as_gf2_vector(supplied["effective_syndrome"]).copy()
        if self.measurement_error.shape != self.effective_syndrome.shape:
            raise ValueError(
                "measurement_error and effective_syndrome length mismatch"
            )
        self.p = float(p)
        self.q = float(q)
        self.epsilon_data_weight = int(
            self.epsilon_data_true.sum()
            if supplied["epsilon_data_weight"] is None
            else supplied["epsilon_data_weight"]
        )
        self.measurement_error_weight = int(
            self.measurement_error.sum()
            if supplied["measurement_error_weight"] is None
            else supplied["measurement_error_weight"]
        )
        self._warned_legacy_aliases = set()

    def validate_for_model(self, model):
        """Reject disorder vectors that do not belong to ``model``."""
        expected_data = (int(model.num_qubits),)
        expected_syndrome = (int(model.num_checks),)
        if self.epsilon_data_true.shape != expected_data:
            raise ValueError(
                "epsilon_data_true length mismatch: expected "
                f"{expected_data[0]}, got {self.epsilon_data_true.size}"
            )
        if self.measurement_error.shape != expected_syndrome:
            raise ValueError(
                "measurement_error length mismatch: expected "
                f"{expected_syndrome[0]}, got {self.measurement_error.size}"
            )
        if self.effective_syndrome.shape != expected_syndrome:
            raise ValueError(
                "effective_syndrome length mismatch: expected "
                f"{expected_syndrome[0]}, got {self.effective_syndrome.size}"
            )
        return self

    def _warn(self, old, new):
        if old in self._warned_legacy_aliases:
            return
        warnings.warn(
            f"DisorderRealization.{old} is deprecated; use {new}",
            DeprecationWarning,
            stacklevel=3,
        )
        self._warned_legacy_aliases.add(old)

    @property
    def eta(self):
        self._warn("eta", "epsilon_data_true")
        view = self.epsilon_data_true.view()
        view.flags.writeable = False
        return view

    @property
    def delta(self):
        self._warn("delta", "measurement_error")
        view = self.measurement_error.view()
        view.flags.writeable = False
        return view

    @property
    def observed_syndrome(self):
        self._warn("observed_syndrome", "effective_syndrome")
        view = self.effective_syndrome.view()
        view.flags.writeable = False
        return view

    @property
    def eta_weight(self):
        self._warn("eta_weight", "epsilon_data_weight")
        return self.epsilon_data_weight

    @property
    def delta_weight(self):
        self._warn("delta_weight", "measurement_error_weight")
        return self.measurement_error_weight

    def syndrome_argument(self, ensemble):
        """Compatibility helper returning the canonical Gibbs argument."""
        canonical = normalize_ensemble(ensemble)
        if canonical == "true_posterior":
            return self.effective_syndrome
        return self.measurement_error


def draw_disorder(model, p, q, rng):
    """Draw data truth and readout error, then form ``effective_syndrome``."""
    return disorder_from_uniforms(
        model,
        p,
        q,
        data_uniforms=rng.random(model.num_qubits),
        syndrome_uniforms=rng.random(model.num_checks),
    )


def disorder_from_uniforms(model, p, q, data_uniforms, syndrome_uniforms):
    """Construct a disorder draw from reusable common random numbers."""
    data_uniforms = np.asarray(data_uniforms, dtype=np.float64)
    syndrome_uniforms = np.asarray(syndrome_uniforms, dtype=np.float64)
    if data_uniforms.shape != (model.num_qubits,):
        raise ValueError("data_uniforms shape mismatch")
    if syndrome_uniforms.shape != (model.num_checks,):
        raise ValueError("syndrome_uniforms shape mismatch")
    epsilon_data_true = (data_uniforms < float(p)).astype(np.uint8)
    measurement_error = (syndrome_uniforms < float(q)).astype(np.uint8)
    effective_syndrome = (
        gf2_matmul(model.H_check, epsilon_data_true[:, None])[:, 0]
        ^ measurement_error
    ).astype(np.uint8)
    return DisorderRealization(
        epsilon_data_true=epsilon_data_true,
        measurement_error=measurement_error,
        effective_syndrome=effective_syndrome,
        p=p,
        q=q,
    )


@dataclass(init=False)
class EnsembleWiring:
    """A canonical Gibbs problem plus a separate planted logical reference."""

    ensemble: str
    gibbs_syndrome_argument: np.ndarray
    planted_logical_class: np.ndarray
    K_p: float
    K_q: float
    q_zero: bool

    def __init__(
        self,
        ensemble,
        gibbs_syndrome_argument=None,
        planted_logical_class=None,
        K_p=0.0,
        K_q=0.0,
        q_zero=False,
        **legacy,
    ):
        legacy_map = {
            "sigma_arg": "gibbs_syndrome_argument",
            "reference_label": "planted_logical_class",
            "ell_ref": "planted_logical_class",
        }
        supplied = {
            "gibbs_syndrome_argument": gibbs_syndrome_argument,
            "planted_logical_class": planted_logical_class,
        }
        for old_name, value in legacy.items():
            canonical = legacy_map.get(old_name)
            if canonical is None:
                raise TypeError(f"unexpected EnsembleWiring argument {old_name!r}")
            if supplied[canonical] is not None:
                raise TypeError(
                    f"cannot pass both {canonical!r} and deprecated {old_name!r}"
                )
            warnings.warn(
                f"{old_name} is deprecated; use {canonical}",
                DeprecationWarning,
                stacklevel=2,
            )
            supplied[canonical] = value
        if supplied["gibbs_syndrome_argument"] is None:
            raise TypeError("missing gibbs_syndrome_argument")
        if supplied["planted_logical_class"] is None:
            raise TypeError("missing planted_logical_class")
        self.ensemble = normalize_ensemble(ensemble)
        self.gibbs_syndrome_argument = as_gf2_vector(
            supplied["gibbs_syndrome_argument"]
        ).copy()
        self.planted_logical_class = as_gf2_vector(
            supplied["planted_logical_class"]
        ).copy()
        self.K_p = float(K_p)
        self.K_q = float(K_q)
        self.q_zero = bool(q_zero)
        self._warned_legacy_aliases = set()

    def _warn(self, old, new):
        if old in self._warned_legacy_aliases:
            return
        warnings.warn(
            f"EnsembleWiring.{old} is deprecated; use {new}",
            DeprecationWarning,
            stacklevel=3,
        )
        self._warned_legacy_aliases.add(old)

    @property
    def sigma_arg(self):
        self._warn("sigma_arg", "gibbs_syndrome_argument")
        view = self.gibbs_syndrome_argument.view()
        view.flags.writeable = False
        return view

    @property
    def reference_label(self):
        self._warn("reference_label", "planted_logical_class")
        view = self.planted_logical_class.view()
        view.flags.writeable = False
        return view

    @property
    def ell_ref(self):
        self._warn("ell_ref", "planted_logical_class")
        view = self.planted_logical_class.view()
        view.flags.writeable = False
        return view

    def total_energy(self, model, e):
        """Evaluate the reduced Gibbs energy without consulting ground truth."""
        e = as_gf2_vector(e)
        if e.shape != (model.num_qubits,):
            raise ValueError("candidate error length mismatch")
        syndrome_term = (
            gf2_matmul(model.H_check, e[:, None])[:, 0]
            ^ self.gibbs_syndrome_argument
        )
        weight_p = int(e.sum())
        weight_s = int(syndrome_term.sum())
        data_energy = (
            0.0 if weight_p == 0 else self.K_p * float(weight_p)
        )
        if self.q_zero:
            if weight_s:
                raise ValueError(
                    "q=0 hard constraint violated: H e != "
                    "gibbs_syndrome_argument"
                )
            return data_energy
        syndrome_energy = (
            0.0 if weight_s == 0 else self.K_q * float(weight_s)
        )
        return data_energy + syndrome_energy


def wire_ensemble(model, disorder, ensemble, observable_frame=None):
    """Create the Gibbs argument and the independent Mattis reference.

    For ``true_posterior`` the Gibbs argument is ``effective_syndrome`` and
    the planted class is ``phi(epsilon_data_true)``.  The legacy ensemble uses
    only ``measurement_error`` and has the clean class as its formal origin.
    """
    disorder.validate_for_model(model)
    canonical = normalize_ensemble(ensemble)
    if canonical == "true_posterior":
        gibbs_syndrome_argument = disorder.effective_syndrome.copy()
    else:
        gibbs_syndrome_argument = disorder.measurement_error.copy()

    q_zero = disorder.q == 0.0
    if q_zero and not model.logical_sector_section.in_image(
        gibbs_syndrome_argument
    ):
        raise AssertionError(
            "q=0 gibbs_syndrome_argument must lie in im(H_check)"
        )
    if canonical == "true_posterior":
        if observable_frame is None:
            raise ValueError(
                "true_posterior requires observable_frame to compute the "
                "planted logical class"
            )
        planted_logical_class = observable_frame.label_of(
            disorder.epsilon_data_true
        )
    else:
        planted_logical_class = np.zeros(model.k, dtype=np.uint8)
    return EnsembleWiring(
        ensemble=canonical,
        gibbs_syndrome_argument=gibbs_syndrome_argument,
        planted_logical_class=planted_logical_class,
        K_p=coupling_from_probability(disorder.p),
        K_q=(
            coupling_from_probability(disorder.q)
            if not q_zero
            else float("inf")
        ),
        q_zero=q_zero,
    )
