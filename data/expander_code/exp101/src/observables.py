"""Logical-sector characters and statistically correct aggregate estimators.

The absolute character is

``m_u_absolute = E[(-1) ** (u dot phi_r(e))]``.

The planted-relative (Mattis) character multiplies it by the deterministic
sign ``(-1) ** (u dot phi_r(epsilon_data_true))``.  The sign changes first
moments but not squared moments, posterior purity, ``q_top``, or the largest
sector mass.  Changing the logical-sector section is not generally a gauge
transformation, so the section is included in the frame fingerprint.
"""

import hashlib
from dataclasses import dataclass

import numpy as np

from .gf2 import as_gf2_matrix, as_gf2_vector, gf2_matmul


DEFAULT_FULL_MAX_K = 10
DEFAULT_NUM_RANDOM_U = 64


@dataclass
class ObservableFrame:
    """Linear logical-sector frame ``phi_r(e) = W e``."""

    W_basis: np.ndarray
    k: int
    num_qubits: int
    section_fingerprint: str

    def label_of(self, vector):
        vector = as_gf2_vector(vector)
        if vector.shape != (self.num_qubits,):
            raise ValueError("vector length mismatch")
        if self.k == 0:
            return np.zeros(0, dtype=np.uint8)
        return gf2_matmul(self.W_basis, vector[:, None])[:, 0]

    def fingerprint(self):
        payload = (
            b"exp101.observable_frame.v2\0"
            + np.asarray([self.k, self.num_qubits], dtype=np.int64).tobytes()
            + self.section_fingerprint.encode("ascii")
            + np.ascontiguousarray(self.W_basis, dtype=np.uint8).tobytes()
        )
        return hashlib.sha256(payload).hexdigest()


def build_observable_frame(model):
    """Build ``W = Z (I xor r_sec H)`` and verify its three defining laws."""
    Z = as_gf2_matrix(model.logical_obs_basis)
    k = Z.shape[0]
    section = model.logical_sector_section
    if k == 0:
        return ObservableFrame(
            W_basis=np.zeros((0, model.num_qubits), dtype=np.uint8),
            k=0,
            num_qubits=model.num_qubits,
            section_fingerprint=section.fingerprint(),
        )
    RH = section.section_after_H(model.H_check)
    W = (Z ^ gf2_matmul(Z, RH)).astype(np.uint8)

    if gf2_matmul(W, RH).any():
        raise AssertionError("logical characters do not annihilate im(r_sec)")
    if gf2_matmul(W, as_gf2_matrix(model.stabilizer_rows).T).any():
        raise AssertionError("logical characters do not annihilate stabilizers")
    pairing = gf2_matmul(W, as_gf2_matrix(model.logical_move_basis).T)
    if not np.array_equal(pairing, np.eye(k, dtype=np.uint8)):
        raise AssertionError("logical characters do not pair with logical moves")

    return ObservableFrame(
        W_basis=W,
        k=k,
        num_qubits=model.num_qubits,
        section_fingerprint=section.fingerprint(),
    )


@dataclass
class ObservableSet:
    """Nonzero logical characters selected for exact or sampled estimation."""

    tier: str
    k: int
    u_bitmasks: np.ndarray
    W_rows: np.ndarray
    basis_positions: np.ndarray
    u_rand_seed: object = None
    num_random_u: int = 0

    @property
    def num_u(self):
        return int(self.u_bitmasks.shape[0])

    @property
    def num_nonzero_characters(self):
        return (1 << self.k) - 1


def build_observable_set(
    frame,
    full_max_k=DEFAULT_FULL_MAX_K,
    num_random_u=DEFAULT_NUM_RANDOM_U,
    u_rand_seed=None,
):
    """Return every character for small ``k`` or basis plus sampled nonbasis."""
    k = frame.k
    if k == 0:
        return ObservableSet(
            tier="basis_only",
            k=0,
            u_bitmasks=np.zeros(0, dtype=np.int64),
            W_rows=np.zeros((0, frame.num_qubits), dtype=np.uint8),
            basis_positions=np.zeros(0, dtype=np.int64),
        )

    def rows_for(bitmasks):
        rows = np.zeros((len(bitmasks), frame.num_qubits), dtype=np.uint8)
        for row_index, bitmask in enumerate(bitmasks):
            for bit in range(k):
                if (bitmask >> bit) & 1:
                    rows[row_index] ^= frame.W_basis[bit]
        return rows

    basis_masks = [1 << i for i in range(k)]
    if k <= int(full_max_k):
        u_list = list(range(1, 1 << k))
        tier = "full"
        seed_used = None
        num_rand = 0
    else:
        if u_rand_seed is None:
            raise ValueError("k > full_max_k requires explicit u_rand_seed")
        num_random_u = int(num_random_u)
        if num_random_u < 0:
            raise ValueError("num_random_u must be nonnegative")
        available = (1 << k) - 1 - k
        if num_random_u > available:
            raise ValueError(
                f"num_random_u={num_random_u} exceeds available distinct "
                f"non-basis u count {available} for k={k}"
            )
        rng = np.random.default_rng(u_rand_seed)
        chosen = set()
        basis_set = set(basis_masks)
        max_draws = 1000 * max(num_random_u, 1)
        draws = 0
        while len(chosen) < num_random_u:
            draws += 1
            if draws > max_draws:
                raise RuntimeError(
                    "rejection sampling for random characters exceeded its "
                    "defensive draw limit"
                )
            candidate = int(rng.integers(1, 1 << k))
            if candidate not in basis_set and candidate not in chosen:
                chosen.add(candidate)
        u_list = basis_masks + sorted(chosen)
        tier = "sampled"
        seed_used = int(u_rand_seed)
        num_rand = num_random_u

    u_bitmasks = np.asarray(u_list, dtype=np.int64)
    basis_positions = np.asarray(
        [u_list.index(mask) for mask in basis_masks], dtype=np.int64
    )
    return ObservableSet(
        tier=tier,
        k=k,
        u_bitmasks=u_bitmasks,
        W_rows=rows_for(u_list),
        basis_positions=basis_positions,
        u_rand_seed=seed_used,
        num_random_u=num_rand,
    )


def absolute_observable_values(observable_set, e):
    """Evaluate absolute characters for one candidate error ``e``."""
    e = as_gf2_vector(e)
    parities = gf2_matmul(observable_set.W_rows, e[:, None])[:, 0]
    return (1 - 2 * parities.astype(np.int8)).astype(np.int8)


def character_signs_for_label(observable_set, logical_class):
    """Return ``(-1)^(u dot logical_class)`` for each selected ``u``."""
    logical_class = as_gf2_vector(logical_class)
    if logical_class.shape != (observable_set.k,):
        raise ValueError("logical_class length mismatch")
    label_mask = 0
    for bit, value in enumerate(logical_class):
        if value:
            label_mask |= 1 << bit
    signs = np.empty(observable_set.num_u, dtype=np.int8)
    for index, u in enumerate(observable_set.u_bitmasks):
        parity = (int(u) & label_mask).bit_count() & 1
        signs[index] = 1 - 2 * parity
    return signs


def relative_observable_values(observable_set, wiring, e):
    """Evaluate Mattis-centred characters for one candidate error ``e``."""
    return (
        absolute_observable_values(observable_set, e)
        * character_signs_for_label(
            observable_set, wiring.planted_logical_class
        )
    ).astype(np.int8)


def observable_values(observable_set, wiring, v):
    """Compatibility API returning planted-relative character values."""
    return relative_observable_values(observable_set, wiring, v)


def relative_character_means(
    observable_set, absolute_means, planted_logical_class
):
    """Apply the exact Mattis sign to absolute character means."""
    absolute_means = _validate_character_vector(observable_set, absolute_means)
    return absolute_means * character_signs_for_label(
        observable_set, planted_logical_class
    )


def _validate_character_vector(observable_set, values):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (observable_set.num_u,):
        raise ValueError("character vector shape mismatch")
    return values


def _nonbasis_mask(observable_set):
    mask = np.ones(observable_set.num_u, dtype=bool)
    mask[observable_set.basis_positions] = False
    return mask


def sampled_nonzero_character_mean(observable_set, values):
    """Estimate the mean over all ``2^k-1`` nonzero characters.

    Basis characters are included exactly.  Uniformly sampled nonbasis
    characters represent only the remaining ``N-k`` population; this is the
    weighting that the pre-alignment implementation omitted.

    Returns ``(estimate, finite_population_standard_error)``.  The latter is
    solely random-character sampling error and excludes MCMC chain error.
    """
    values = _validate_character_vector(observable_set, values)
    k = observable_set.k
    if k == 0:
        return None, None
    N = (1 << k) - 1
    if observable_set.tier == "full":
        return float(np.mean(values)), 0.0

    basis_sum = float(np.sum(values[observable_set.basis_positions]))
    nonbasis_population = N - k
    if nonbasis_population == 0:
        return basis_sum / N, 0.0
    sample = values[_nonbasis_mask(observable_set)]
    if sample.size == 0:
        raise ValueError(
            "sampled character aggregation requires at least one nonbasis "
            "character"
        )
    estimate = (basis_sum + nonbasis_population * float(np.mean(sample))) / N
    if sample.size == nonbasis_population:
        standard_error = 0.0
    elif sample.size < 2:
        standard_error = float("nan")
    else:
        fpc = 1.0 - sample.size / nonbasis_population
        mean_se = np.sqrt(fpc * np.var(sample, ddof=1) / sample.size)
        standard_error = (nonbasis_population / N) * float(mean_se)
    return float(estimate), float(standard_error)


def _fwht(values):
    transformed = np.asarray(values, dtype=np.float64).copy()
    n = transformed.size
    width = 1
    while width < n:
        for start in range(0, n, 2 * width):
            left = transformed[start:start + width].copy()
            right = transformed[start + width:start + 2 * width].copy()
            transformed[start:start + width] = left + right
            transformed[start + width:start + 2 * width] = left - right
        width *= 2
    return transformed


def characters_from_sector_weights(weights):
    """Return all nonzero Walsh characters in bitmask order ``1..2^k-1``."""
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1 or weights.size == 0 or weights.size & (weights.size - 1):
        raise ValueError("sector weights length must be a nonzero power of two")
    return _fwht(weights)[1:]


def sector_weights_from_characters(observable_set, character_means):
    """Invert a complete character table into sector weights."""
    character_means = _validate_character_vector(observable_set, character_means)
    if observable_set.tier != "full":
        raise ValueError("sector-weight inversion requires the full character set")
    spectrum = np.empty(1 << observable_set.k, dtype=np.float64)
    spectrum[0] = 1.0
    spectrum[observable_set.u_bitmasks.astype(np.int64)] = character_means
    return _fwht(spectrum) / (1 << observable_set.k)


def posterior_statistics(weights_absolute, planted_class=0):
    """Compute distinct purity, planted mass, and MAP-success statistics."""
    weights = np.asarray(weights_absolute, dtype=np.float64)
    if weights.ndim != 1 or weights.size == 0 or weights.size & (weights.size - 1):
        raise ValueError("weights length must be a nonzero power of two")
    if not np.all(np.isfinite(weights)):
        raise ValueError("weights must be finite")
    planted_class = int(planted_class)
    if not 0 <= planted_class < weights.size:
        raise ValueError("planted_class outside sector range")
    purity = float(np.sum(weights**2))
    map_success = float(np.max(weights))
    if weights.size == 1:
        q_top = None
    else:
        q_top = float((weights.size * purity - 1.0) / (weights.size - 1.0))
    minimum_purity = 1.0 / weights.size
    bounds_valid = minimum_purity - 1e-14 <= purity <= 1.0 + 1e-14
    return {
        "posterior_purity": purity,
        "posterior_mass_on_planted_class": float(weights[planted_class]),
        "map_success_probability": map_success,
        "map_success_lower_bound": purity if bounds_valid else None,
        "map_success_upper_bound": float(np.sqrt(purity)) if bounds_valid else None,
        "posterior_purity_within_physical_bounds": bool(bounds_valid),
        "q_top": q_top,
    }


def aggregate_observables(
    observable_set,
    m_u_values,
    *,
    m2_u_values=None,
    character_frame="relative",
    planted_logical_class=None,
):
    """Aggregate selected characters with correct population weighting.

    ``m2_u_values`` may contain cross-chain U-statistics.  If omitted, pooled
    squares are used and the returned ``q_top`` is explicitly named ``raw``.
    No clipping is applied when a debiased estimate leaves the physical range.
    """
    m_u_values = _validate_character_vector(observable_set, m_u_values)
    if m2_u_values is None:
        m2_u_values = m_u_values**2
        estimator_name = "pooled_square_raw"
    else:
        m2_u_values = _validate_character_vector(observable_set, m2_u_values)
        estimator_name = "independent_chain_u_statistic"
    if character_frame not in ("absolute", "relative"):
        raise ValueError("character_frame must be 'absolute' or 'relative'")

    k = observable_set.k
    if k == 0:
        return {
            "q_top": None,
            "q_top_all": None,
            "q_top_basis": None,
            "q_top_estimator_name": estimator_name,
            "posterior_purity": 1.0,
            "purity": 1.0,
            "posterior_mass_on_planted_class": 1.0,
            "map_success_probability": 1.0,
            "map_success_lower_bound": 1.0,
            "map_success_upper_bound": 1.0,
            "q_top_character_sampling_se": 0.0,
            "posterior_purity_within_physical_bounds": True,
        }

    q_top, q_top_sampling_se = sampled_nonzero_character_mean(
        observable_set, m2_u_values
    )
    basis_m2 = m2_u_values[observable_set.basis_positions]
    N = (1 << k) - 1
    purity = (1.0 + N * q_top) / (1 << k)

    if character_frame == "relative":
        relative_means = m_u_values
    elif planted_logical_class is not None:
        relative_means = relative_character_means(
            observable_set, m_u_values, planted_logical_class
        )
    else:
        relative_means = None

    planted_mass = None
    planted_mass_sampling_se = None
    if relative_means is not None:
        mean_relative, mean_relative_se = sampled_nonzero_character_mean(
            observable_set, relative_means
        )
        planted_mass = (1.0 + N * mean_relative) / (1 << k)
        planted_mass_sampling_se = N * mean_relative_se / (1 << k)

    map_success = None
    weights_in_input_frame = None
    if observable_set.tier == "full":
        weights_in_input_frame = sector_weights_from_characters(
            observable_set, m_u_values
        )
        map_success = float(np.max(weights_in_input_frame))

    minimum_purity = 1.0 / (1 << k)
    bounds_valid = minimum_purity - 1e-14 <= purity <= 1.0 + 1e-14
    return {
        "q_top": float(q_top),
        "q_top_all": float(q_top),
        "q_top_basis": float(np.mean(basis_m2)),
        "q_top_estimator_name": estimator_name,
        "q_top_character_sampling_se": float(q_top_sampling_se),
        "posterior_purity": float(purity),
        "purity": float(purity),
        "posterior_mass_on_planted_class": (
            None if planted_mass is None else float(planted_mass)
        ),
        "posterior_mass_character_sampling_se": (
            None
            if planted_mass_sampling_se is None
            else float(planted_mass_sampling_se)
        ),
        "map_success_probability": map_success,
        "map_success_lower_bound": float(purity) if bounds_valid else None,
        "map_success_upper_bound": (
            float(np.sqrt(purity)) if bounds_valid else None
        ),
        "posterior_purity_within_physical_bounds": bool(bounds_valid),
        "weights_in_character_frame": weights_in_input_frame,
    }


def independent_chain_squared_character_estimates(chain_means):
    """Debias character squares using independent-chain cross products.

    The delete-one-chain jackknife standard error is returned per character.
    Production uses at least four chains; two are mathematically sufficient for
    the U-statistic and three for a nondegenerate delete-one estimate.
    """
    chain_means = np.asarray(chain_means, dtype=np.float64)
    if chain_means.ndim != 2:
        raise ValueError("chain_means must have shape (num_chains, num_u)")
    C = chain_means.shape[0]
    if C < 2:
        raise ValueError("at least two independent chains are required")
    pooled = np.mean(chain_means, axis=0)
    raw = pooled**2
    sums = np.sum(chain_means, axis=0)
    sum_squares = np.sum(chain_means**2, axis=0)
    debiased = (sums**2 - sum_squares) / (C * (C - 1))

    if C < 3:
        jackknife_se = np.full(chain_means.shape[1], np.nan)
        delete_one = np.empty((0, chain_means.shape[1]))
    else:
        delete_one = np.empty_like(chain_means)
        for omitted in range(C):
            reduced = np.delete(chain_means, omitted, axis=0)
            r_sum = np.sum(reduced, axis=0)
            r_sum_squares = np.sum(reduced**2, axis=0)
            count = C - 1
            delete_one[omitted] = (
                (r_sum**2 - r_sum_squares) / (count * (count - 1))
            )
        delete_mean = np.mean(delete_one, axis=0)
        jackknife_se = np.sqrt(
            (C - 1) / C * np.sum((delete_one - delete_mean) ** 2, axis=0)
        )
    return {
        "m_u_pooled": pooled,
        "m2_u_pooled_square_raw": raw,
        "m2_u_debiased": debiased,
        "m2_u_delete_one_chain": delete_one,
        "m2_u_debiased_jackknife_se": jackknife_se,
    }


def aggregate_independent_chain_observables(
    observable_set,
    chain_means,
    *,
    character_frame="relative",
    planted_logical_class=None,
):
    """Combine independent-chain debiasing with character-population weights."""
    estimates = independent_chain_squared_character_estimates(chain_means)
    aggregates = aggregate_observables(
        observable_set,
        estimates["m_u_pooled"],
        m2_u_values=estimates["m2_u_debiased"],
        character_frame=character_frame,
        planted_logical_class=planted_logical_class,
    )
    delete_one = estimates["m2_u_delete_one_chain"]
    if delete_one.shape[0]:
        delete_q = np.array([
            sampled_nonzero_character_mean(observable_set, row)[0]
            for row in delete_one
        ])
        C = delete_q.size
        q_mean = float(np.mean(delete_q))
        q_jackknife_se = float(
            np.sqrt((C - 1) / C * np.sum((delete_q - q_mean) ** 2))
        )
    else:
        q_jackknife_se = float("nan")
    return {
        **estimates,
        **aggregates,
        "q_top_chain_jackknife_se": q_jackknife_se,
    }
