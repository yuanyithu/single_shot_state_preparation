"""Full logical-sector thermodynamic integration for ``k <= 10``.

固定 label 扇区的受限配分函数 Z_ℓ(K_p)，dF/dK_p = ⟨|v|⟩_ℓ ⇒
    ΔF_ℓ(K_p*) = ∫_0^{K_p*} μ_ℓ dK_p − (ℓ_anchor 同式)，trapezoid。
K_p 网格自 0（p=0.5，各扇区 Z 相等 ⇒ ΔF(0)=0 解析成立）升到目标值，
链沿网格退火续跑。天然绕开冻结扇区（每扇区独立采样，无需跨壁垒）。

Production TI always enumerates all ``2**k`` sectors and is rejected before
work starts when ``k > 10``.  The old pairwise approximation is not a purity
estimator.  It survives only as
``basis_sector_free_energy_gap_diagnostics`` and deliberately exposes no
character or ``q_top`` fields.

sector-preserving proposals（q>0；全部保 label）：
  - 零签名单比特（W 列全零的 qubit）：动 T⊕S，改 syndrome（K_q 计费）
  - 同签名 qubit 对（W 列相同且非零）：组内全对
  - stabilizer 行（S-move，syndrome/label 双保）
q=0：仅 stabilizer 行（coset 内固定扇区采样），代表元为
`logical_sector_section(effective_syndrome) xor combo`。

numba fast path：**挂账至 G4.2**（性能门槛处落地；本文件为正确性权威参考版）。
"""

from dataclasses import dataclass

import numpy as np

from .gf2 import gf2_matmul
from .observables import (
    DEFAULT_FULL_MAX_K,
    characters_from_sector_weights,
    posterior_statistics,
)
from .prng import PortablePrng


@dataclass
class SectorTiConfig:
    num_kp_grid_points: int = 33
    num_burn_in_sweeps: int = 200        # 每个网格点
    num_measurements: int = 400          # 每个网格点
    num_sweeps_between_measurements: int = 1
    block_count: int = 8
    num_bootstrap: int = 200
    full_max_k: int = DEFAULT_FULL_MAX_K
    grid_tv_warning: float = 0.02
    grid_q_top_warning: float = 0.01


FULL_SECTOR_TI_MAX_K = 10


def build_sector_preserving_proposals(model, frame):
    """返回 proposals dict：supports 列表（int64 数组）+ 统计。全部保 label。"""
    W = frame.W_basis
    n = model.num_qubits
    supports = []
    kinds = []
    if model.k == 0:
        signatures = [0] * n
    else:
        signatures = []
        for j in range(n):
            mask = 0
            for u in range(model.k):
                if W[u, j]:
                    mask |= 1 << u
            signatures.append(mask)
    # 零签名单比特
    num_single = 0
    for j in range(n):
        if signatures[j] == 0:
            supports.append(np.array([j], dtype=np.int64))
            kinds.append("single")
            num_single += 1
    # 同签名对（组内全对）
    groups = {}
    for j, sig in enumerate(signatures):
        if sig != 0:
            groups.setdefault(sig, []).append(j)
    num_pairs = 0
    for members in groups.values():
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                supports.append(
                    np.array([members[a], members[b]], dtype=np.int64)
                )
                kinds.append("pair")
                num_pairs += 1
    # stabilizer 行
    num_stab = 0
    for row in model.stabilizer_rows:
        supports.append(np.flatnonzero(row).astype(np.int64))
        kinds.append("stab")
        num_stab += 1
    return {
        "supports": supports,
        "kinds": kinds,
        "num_single": num_single,
        "num_pairs": num_pairs,
        "num_stab": num_stab,
        "signature_group_sizes": sorted(
            (len(v) for v in groups.values()), reverse=True
        ),
    }


def sector_representative(model, wiring, label_bits):
    """label（相对绝对 0 类的 bit 向量/int）→ 代表构型。

    q>0：v = combo(label)（零 syndrome 代表）；q=0：v = r(y_eff) ⊕ combo。
    """
    v = np.zeros(model.num_qubits, dtype=np.uint8)
    for bit in range(model.k):
        if (int(label_bits) >> bit) & 1:
            v ^= model.logical_move_basis[bit]
    if wiring.q_zero:
        v = v ^ model.logical_sector_section.apply(
            wiring.gibbs_syndrome_argument, strict=True
        )
    return v


def _run_fixed_sector_chain(model, wiring, proposals, v0, kp_grid, config,
                            rng):
    """单扇区链：沿 kp_grid 退火，逐点测 μ=⟨|v|⟩、μ_s=⟨|Hv⊕σ|⟩、block 均值。"""
    v = v0.copy()
    syndrome_term = (
        gf2_matmul(model.H_check, v[:, None])[:, 0]
        ^ wiring.gibbs_syndrome_argument
    ).astype(np.uint8)
    data_weight = int(v.sum())
    syndrome_weight = int(syndrome_term.sum())
    if wiring.q_zero and syndrome_weight:
        raise AssertionError("q=0 sector representative violates constraint")
    K_q = 0.0 if wiring.q_zero else wiring.K_q
    supports = proposals["supports"]
    if wiring.q_zero:
        usable = [s for s, kind in zip(supports, proposals["kinds"])
                  if kind == "stab"]
    else:
        usable = supports
    if not usable:
        raise ValueError("no sector-preserving proposals available")

    num_grid = len(kp_grid)
    mu = np.zeros(num_grid)
    syndrome_mu = np.zeros(num_grid)
    block_mu = np.zeros((num_grid, config.block_count))
    acceptance = np.zeros(num_grid)
    H = model.H_check

    def attempt(kp_value, support):
        nonlocal data_weight, syndrome_weight
        overlap = int(v[support].sum())
        delta_data = int(support.shape[0]) - 2 * overlap
        if wiring.q_zero:
            delta_syn = 0
        else:
            # syndrome 变化：受 support 触及的 check 集合
            touched = np.unique(np.concatenate(
                [model.checks_touching_each_qubit[j] for j in support]
            )) if support.shape[0] else np.zeros(0, dtype=np.int32)
            # 逐 check 计算翻转次数的奇偶
            delta_syn = 0
            for c in touched:
                flips = 0
                for j in support:
                    if H[c, j]:
                        flips ^= 1
                if flips:
                    delta_syn += 1 - 2 * int(syndrome_term[c])
        log_acc = -kp_value * delta_data - K_q * delta_syn
        u = rng.random()
        accepted = log_acc >= 0.0 or u < np.exp(log_acc)
        if accepted:
            v[support] ^= 1
            data_weight += delta_data
            if not wiring.q_zero:
                for c in touched:
                    flips = 0
                    for j in support:
                        if H[c, j]:
                            flips ^= 1
                    if flips:
                        syndrome_weight += 1 - 2 * int(syndrome_term[c])
                        syndrome_term[c] ^= 1
        return accepted

    num_props = len(usable)
    for grid_index, kp_value in enumerate(kp_grid):
        accepted_count = 0
        attempted_count = 0
        for _ in range(config.num_burn_in_sweeps):
            order = rng.permutation(num_props)
            for idx in order:
                attempted_count += 1
                accepted_count += attempt(float(kp_value), usable[idx])
        samples = np.zeros(config.num_measurements)
        syn_samples = np.zeros(config.num_measurements)
        for m in range(config.num_measurements):
            for _ in range(config.num_sweeps_between_measurements):
                order = rng.permutation(num_props)
                for idx in order:
                    attempted_count += 1
                    accepted_count += attempt(float(kp_value), usable[idx])
            samples[m] = data_weight
            syn_samples[m] = syndrome_weight
        mu[grid_index] = samples.mean()
        syndrome_mu[grid_index] = syn_samples.mean()
        for b, chunk in enumerate(
                np.array_split(samples, config.block_count)):
            block_mu[grid_index, b] = chunk.mean()
        acceptance[grid_index] = accepted_count / max(attempted_count, 1)
    return {"mu": mu, "syndrome_mu": syndrome_mu, "block_mu": block_mu,
            "acceptance": acceptance}


def _integrate(kp_grid, mu_rows):
    return np.trapezoid(mu_rows, x=kp_grid, axis=-1)


def _bootstrap_delta_f(block_arrays, kp_grid, num_bootstrap, seed):
    """Bootstrap independent sector chains without inducing correlations.

    Each logical sector is simulated by an independent chain, so its block
    indices must also be resampled independently.  Reusing one index vector
    across sectors can cancel matched block fluctuations and spuriously drive
    free-energy-gap uncertainty to zero.
    """
    block_arrays = np.asarray(block_arrays, dtype=np.float64)
    if block_arrays.ndim != 3:
        raise ValueError("block_arrays must have shape (labels, grid, blocks)")
    num_labels, _, num_blocks = block_arrays.shape
    rng_boot = np.random.default_rng(int(seed) ^ 0x5EED)
    boot_stats = np.empty((int(num_bootstrap), num_labels), dtype=np.float64)
    for bootstrap_index in range(int(num_bootstrap)):
        integrals = np.empty(num_labels, dtype=np.float64)
        for label_index in range(num_labels):
            indices = rng_boot.integers(0, num_blocks, size=num_blocks)
            mu_boot = block_arrays[label_index][:, indices].mean(axis=1)
            integrals[label_index] = _integrate(kp_grid, mu_boot)
        boot_stats[bootstrap_index] = integrals - integrals[0]
    return boot_stats


def _q_top_from_weights(weights, k):
    weights = np.asarray(weights, dtype=np.float64)
    if k == 0:
        return None
    return float(((1 << k) * np.sum(weights ** 2) - 1.0) / ((1 << k) - 1))


def _validate_ti_inputs(model, wiring):
    if model.k == 0:
        raise ValueError("sector TI requires k >= 1")
    kp_target = wiring.K_p
    if not np.isfinite(kp_target) or kp_target <= 0:
        raise ValueError("K_p target must be finite positive (p<0.5, p>0)")


def _run_label_integrations(model, frame, wiring, config, seed, labels):
    """Run fixed-label chains and return common integration diagnostics."""
    _validate_ti_inputs(model, wiring)
    kp_target = wiring.K_p
    kp_grid = np.linspace(0.0, kp_target, int(config.num_kp_grid_points))
    proposals = build_sector_preserving_proposals(model, frame)
    rng = PortablePrng(seed)

    chains = {}
    for label in labels:
        v0 = sector_representative(model, wiring, label)
        # 断言代表元 label 正确（frame 一致性）
        got = frame.label_of(v0 if not wiring.q_zero else v0)
        got_bits = 0
        for bit, val in enumerate(got):
            if val:
                got_bits |= 1 << bit
        if not wiring.q_zero and got_bits != label:
            raise AssertionError("sector representative label mismatch")
        chains[label] = _run_fixed_sector_chain(
            model, wiring, proposals, v0, kp_grid, config, rng
        )

    integrals = {lab: float(_integrate(kp_grid, chains[lab]["mu"]))
                 for lab in labels}
    anchor = labels[0]
    delta_f = {lab: integrals[lab] - integrals[anchor] for lab in labels}

    # Independent chains require independent block resampling by sector.
    block_arrays = np.stack(
        [chains[lab]["block_mu"] for lab in labels]
    )  # (L, G, B)
    boot_stats = _bootstrap_delta_f(
        block_arrays, kp_grid, config.num_bootstrap, seed
    )

    return {
        "labels": labels,
        "kp_grid": kp_grid,
        "delta_f": np.array([delta_f[lab] for lab in labels]),
        "delta_f_infinite_mask": np.zeros(len(labels), dtype=bool),
        "delta_f_stderr": boot_stats.std(axis=0, ddof=1),
        "acceptance_per_label": {
            lab: chains[lab]["acceptance"] for lab in labels
        },
        "proposal_summary": {k_: v_ for k_, v_ in proposals.items()
                             if k_ != "supports"},
        "_chains": chains,
        "_boot_stats": boot_stats,
    }


def _bitmask(bits):
    value = 0
    for bit, enabled in enumerate(np.asarray(bits, dtype=np.uint8)):
        if enabled:
            value |= 1 << bit
    return value


def _weights_from_delta(delta_values):
    shifted = -np.asarray(delta_values, dtype=np.float64)
    shifted -= shifted.max()
    weights = np.exp(shifted)
    return weights / weights.sum()


def _attach_full_sector_statistics(
    result, model, wiring, weights_absolute, q_top_stderr,
    estimator_name="full_sector_weights",
):
    """Attach paper posterior fields or legacy formal-only diagnostics."""
    labels = result["labels"]
    planted_class = _bitmask(wiring.planted_logical_class)
    weights_absolute = np.asarray(weights_absolute, dtype=np.float64)
    weights_relative = np.asarray([
        weights_absolute[t ^ planted_class] for t in labels
    ])
    characters_absolute = characters_from_sector_weights(weights_absolute)
    characters_relative = characters_from_sector_weights(weights_relative)
    basis_indices = np.asarray([(1 << u) - 1 for u in range(model.k)])
    m_basis_absolute = characters_absolute[basis_indices]
    m_basis_relative = characters_relative[basis_indices]
    q_top = _q_top_from_weights(weights_absolute, model.k)
    formal_purity = float(np.sum(weights_absolute**2))
    largest_mass = float(np.max(weights_absolute))
    result.update({
        "tier": "full",
        "engine_name": "full_sector_ti",
        "planted_logical_class_bitmask": planted_class,
        "formal_sector_purity": formal_purity,
        "largest_sector_mass": largest_mass,
    })
    if wiring.ensemble == "true_posterior":
        result.update({
            "weights_absolute": weights_absolute,
            "weights_relative": weights_relative,
            "characters_absolute": characters_absolute,
            "characters_relative": characters_relative,
            "m_u_absolute": m_basis_absolute,
            "m_u_relative": m_basis_relative,
            "m_u_basis_absolute": m_basis_absolute,
            "m_u_basis_relative": m_basis_relative,
            "m_u_basis": m_basis_relative,
            "q_top": q_top,
            "q_top_absolute": q_top,
            "q_top_relative": q_top,
            "q_top_estimator_name": estimator_name,
            "q_top_stderr": float(q_top_stderr),
        })
        result.update(posterior_statistics(weights_absolute, planted_class))
        return result

    result.update({
        "formal_sector_weights_absolute": weights_absolute,
        "formal_sector_weights_relative": weights_relative,
        "formal_sector_characters_absolute": characters_absolute,
        "formal_sector_characters_relative": characters_relative,
        "formal_m_u_basis_absolute": m_basis_absolute,
        "formal_m_u_basis_relative": m_basis_relative,
        "formal_q_top": q_top,
        "formal_q_top_absolute": q_top,
        "formal_q_top_relative": q_top,
        "formal_q_top_estimator_name": estimator_name,
        "formal_q_top_stderr": float(q_top_stderr),
    })
    for name in (
        "weights_absolute", "weights_relative", "characters_absolute",
        "characters_relative", "m_u_absolute", "m_u_relative",
        "m_u_basis_absolute", "m_u_basis_relative", "m_u_basis", "q_top",
        "q_top_absolute", "q_top_relative", "q_top_estimator_name",
        "q_top_stderr", "posterior_purity",
        "posterior_mass_on_planted_class", "map_success_probability",
        "map_success_lower_bound", "map_success_upper_bound",
        "posterior_purity_within_physical_bounds",
    ):
        result[name] = None
    return result


def _analytic_endpoint_result(model, frame, wiring):
    """Return exact full-sector results for Kp=0 or Kp=+infinity."""
    labels = list(range(1 << model.k))
    num_labels = len(labels)
    if wiring.K_p == 0.0:
        endpoint_mode = "p_half_uniform"
        weights_absolute = np.full(num_labels, 1.0 / num_labels)
        delta_f = np.zeros(num_labels, dtype=np.float64)
    elif np.isposinf(wiring.K_p):
        if wiring.q_zero and np.any(wiring.gibbs_syndrome_argument):
            raise ValueError(
                "p=0 and q=0 with nonzero Gibbs syndrome has zero support"
            )
        endpoint_mode = "p_zero_delta"
        weights_absolute = np.zeros(num_labels, dtype=np.float64)
        weights_absolute[0] = 1.0
        delta_f = np.full(num_labels, np.inf, dtype=np.float64)
        delta_f[0] = 0.0
    else:
        return None

    proposals = build_sector_preserving_proposals(model, frame)
    result = {
        "labels": labels,
        "kp_grid": np.zeros(0, dtype=np.float64),
        "delta_f": delta_f,
        "delta_f_infinite_mask": np.isposinf(delta_f),
        "delta_f_stderr": np.zeros(num_labels, dtype=np.float64),
        "acceptance_per_label": {
            label: np.zeros(0, dtype=np.float64) for label in labels
        },
        "proposal_summary": {
            key: value for key, value in proposals.items()
            if key != "supports"
        },
        "endpoint_mode": endpoint_mode,
        "grid_tv": 0.0,
        "grid_q_top_abs_diff": 0.0,
        "flags": "PASS",
        "valid_for_aggregation": True,
    }
    return _attach_full_sector_statistics(
        result, model, wiring, weights_absolute, q_top_stderr=0.0,
        estimator_name="analytic_full_sector_endpoint",
    )


def run_sector_ti(model, frame, wiring, config, seed):
    """Enumerate every logical sector and integrate its free energy."""
    if model.k > FULL_SECTOR_TI_MAX_K:
        raise ValueError(
            "full-sector thermodynamic integration requires k<=10; "
            "use PT observable sampling for large-k production"
        )
    if model.k == 0:
        raise ValueError("sector TI requires k >= 1")
    endpoint = _analytic_endpoint_result(model, frame, wiring)
    if endpoint is not None:
        return endpoint
    labels = list(range(1 << model.k))
    result = _run_label_integrations(
        model, frame, wiring, config, seed, labels
    )
    chains = result.pop("_chains")
    boot_stats = result.pop("_boot_stats")
    kp_grid = result["kp_grid"]
    num_grid = len(kp_grid)

    weights_absolute = _weights_from_delta(result["delta_f"])
    boot_q_top = np.asarray([
        _q_top_from_weights(_weights_from_delta(row), model.k)
        for row in boot_stats
    ])
    q_top = _q_top_from_weights(weights_absolute, model.k)

    coarse_idx = np.arange(0, num_grid, 2)
    if coarse_idx[-1] != num_grid - 1:
        coarse_idx = np.append(coarse_idx, num_grid - 1)
    ints_coarse = np.array([
        float(_integrate(kp_grid[coarse_idx], chains[label]["mu"][coarse_idx]))
        for label in labels
    ])
    weights_coarse = _weights_from_delta(ints_coarse - ints_coarse[0])
    result["grid_tv"] = float(
        0.5 * np.abs(weights_absolute - weights_coarse).sum()
    )
    result["grid_q_top_abs_diff"] = float(abs(
        q_top - _q_top_from_weights(weights_coarse, model.k)
    ))
    failures = []
    if result["grid_tv"] > config.grid_tv_warning:
        failures.append("ti_grid_tv_exceeded")
    if result["grid_q_top_abs_diff"] > config.grid_q_top_warning:
        failures.append("ti_grid_q_top_exceeded")
    result["flags"] = "PASS" if not failures else ";".join(failures)
    result["valid_for_aggregation"] = not failures
    return _attach_full_sector_statistics(
        result, model, wiring, weights_absolute,
        q_top_stderr=float(np.std(boot_q_top, ddof=1)),
    )


def basis_sector_free_energy_gap_diagnostics(
    model, frame, wiring, config, seed
):
    """Return basis-sector free-energy gaps, never logical purity estimates."""
    _validate_ti_inputs(model, wiring)
    planted_class = _bitmask(wiring.planted_logical_class)
    labels = [planted_class] + [
        planted_class ^ (1 << u) for u in range(model.k)
    ]
    result = _run_label_integrations(
        model, frame, wiring, config, seed, labels
    )
    result.pop("_chains")
    result.pop("_boot_stats")
    return {
        "diagnostic_name": "basis_sector_free_energy_gap_diagnostics",
        "labels": result["labels"],
        "planted_logical_class_bitmask": planted_class,
        "kp_grid": result["kp_grid"],
        "free_energy_difference_by_label": result["delta_f"],
        "free_energy_difference_stderr_by_label": result["delta_f_stderr"],
        "basis_sector_free_energy_gaps": result["delta_f"][1:],
        "basis_sector_free_energy_gap_stderr": result["delta_f_stderr"][1:],
        "acceptance_per_label": result["acceptance_per_label"],
        "proposal_summary": result["proposal_summary"],
        "flags": "DIAGNOSTIC_ONLY",
    }
