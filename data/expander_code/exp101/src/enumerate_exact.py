"""Exact oracle for the canonical reduced posterior and logical sectors.

核心思想：一次枚举把全空间按 (数据重量 w_p, syndrome 重量 w_s, 线性 label ℓ)
分箱计数 ⇒ 任意 (K_p, K_q) 的 Z_ℓ、类权重、characters、q_top、⟨W_p⟩、⟨W_s⟩、
μ_ℓ(K_p)（TI 曲线）全部**精确**求值——比逐 (p,q) 枚举强得多。

实现：v/syndrome/label 全部 uint64 位打包（守卫 n≤28、m_c≤64、k≤13、表≤256MB）；
binary-reflected Gray code 每步翻 1 bit，SWAR popcount 增量维护 w_s；
numba kernel + 纯 python 参考实现（小 n 互证）。
q=0：coset 版（e = r(y_eff) ⊕ ker 组合，表为 (w_p, ℓ) 二维；dim ≤ 26 守卫）。
结构恒等式（tests）：y_eff ∈ im(H) 时 N_full[:, 0, :] ≡ N_coset。

标签一律线性 frame（φ(v)=W·v，增量 XOR 的前提）；frame A/B 的 decoder 标签
对照走小 n 直接枚举（G3.3）。
"""

import warnings
from dataclasses import dataclass

import numpy as np

from .gf2 import gf2_matmul, gf2_nullspace
from .observables import characters_from_sector_weights, posterior_statistics

try:
    from numba import njit, uint64

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover
    njit = None
    NUMBA_AVAILABLE = False

MAX_N = 28
MAX_CHECKS = 64
MAX_K = 13
MAX_TABLE_BYTES = 256 * 1024 * 1024
MAX_COSET_DIM = 26


@dataclass
class ExactTable:
    kind: str                 # "full" | "coset"
    table: np.ndarray         # full: (n+1, m_c+1, 2^k)；coset: (n+1, 2^k)
    n: int
    num_checks: int
    k: int
    gibbs_syndrome_argument: np.ndarray
    total_states: int

    @property
    def sigma_arg(self):
        warnings.warn(
            "ExactTable.sigma_arg is deprecated; use gibbs_syndrome_argument",
            DeprecationWarning,
            stacklevel=2,
        )
        view = self.gibbs_syndrome_argument.view()
        view.flags.writeable = False
        return view


def _bits_to_uint64(bits):
    value = 0
    for index, bit in enumerate(np.asarray(bits).astype(np.uint8)):
        if bit:
            value |= 1 << index
    return np.uint64(value)


def _label_bits_of(frame, vector):
    label = frame.label_of(vector)
    bits = 0
    for b, val in enumerate(label):
        if val:
            bits |= 1 << b
    return bits


def _coupling_energy(coupling, weights):
    """Multiply a coupling by integer weights without producing ``inf*0``."""
    coupling = float(coupling)
    weights = np.asarray(weights)
    if np.isposinf(coupling):
        return np.where(weights == 0, 0.0, np.inf)
    return coupling * weights


def _guards_full(model):
    n, m_c, k = model.num_qubits, model.num_checks, model.k
    if n > MAX_N:
        raise ValueError(f"full enumeration guard: n={n} > {MAX_N}")
    if m_c > MAX_CHECKS:
        raise ValueError(f"full enumeration guard: m_c={m_c} > {MAX_CHECKS}")
    if k > MAX_K:
        raise ValueError(f"full enumeration guard: k={k} > {MAX_K}")
    table_bytes = (n + 1) * (m_c + 1) * (1 << k) * 8
    if table_bytes > MAX_TABLE_BYTES:
        raise ValueError("full enumeration guard: table too large")


def _guards_coset(model):
    n, k = model.num_qubits, model.k
    if n > MAX_N:
        raise ValueError(f"coset guard: n={n} > {MAX_N}")
    if k > MAX_K:
        raise ValueError(f"coset guard: k={k} > {MAX_K}")
    table_bytes = (n + 1) * (1 << k) * np.dtype(np.int64).itemsize
    if table_bytes > MAX_TABLE_BYTES:
        raise ValueError("coset guard: table too large")


def _validated_gibbs_argument(model, gibbs_syndrome_argument):
    argument = np.asarray(gibbs_syndrome_argument, dtype=np.uint8)
    if argument.shape != (model.num_checks,):
        raise ValueError(
            "gibbs_syndrome_argument length mismatch: expected "
            f"{model.num_checks}, got shape {argument.shape}"
        )
    return argument


if NUMBA_AVAILABLE:

    @njit(cache=True)
    def _popcount64(x):
        x = x - ((x >> uint64(1)) & uint64(0x5555555555555555))
        x = (x & uint64(0x3333333333333333)) + (
            (x >> uint64(2)) & uint64(0x3333333333333333)
        )
        x = (x + (x >> uint64(4))) & uint64(0x0F0F0F0F0F0F0F0F)
        return int((x * uint64(0x0101010101010101)) >> uint64(56))

    @njit(cache=True)
    def _fill_full_table_nb(table_flat, n, stride_ws, stride_wp,
                            syndrome_masks, label_masks, sigma_init):
        v = uint64(0)
        syndrome = sigma_init
        w_p = 0
        w_s = _popcount64(syndrome)
        ell = 0
        table_flat[w_p * stride_wp + w_s * stride_ws + ell] += 1
        total = 1 << n
        for step in range(1, total):
            s = step
            j = 0
            while not (s & 1):
                s >>= 1
                j += 1
            bit = uint64(1) << uint64(j)
            v ^= bit
            if v & bit:
                w_p += 1
            else:
                w_p -= 1
            mask = syndrome_masks[j]
            w_s += _popcount64(mask & ~syndrome) - _popcount64(mask & syndrome)
            syndrome ^= mask
            ell ^= label_masks[j]
            table_flat[w_p * stride_wp + w_s * stride_ws + ell] += 1

    @njit(cache=True)
    def _fill_coset_table_nb(table_flat, dim, stride_wp, generator_masks,
                             generator_labels, v0, ell0):
        v = v0
        ell = ell0
        w_p = _popcount64(v)
        table_flat[w_p * stride_wp + ell] += 1
        total = 1 << dim
        for step in range(1, total):
            s = step
            j = 0
            while not (s & 1):
                s >>= 1
                j += 1
            v ^= generator_masks[j]
            ell ^= generator_labels[j]
            w_p = _popcount64(v)
            table_flat[w_p * stride_wp + ell] += 1


def _fill_full_table_py(table, n, syndrome_masks, label_masks, sigma_init):
    v = 0
    syndrome = int(sigma_init)
    w_p = 0
    w_s = bin(syndrome).count("1")
    ell = 0
    table[w_p, w_s, ell] += 1
    for step in range(1, 1 << n):
        j = (step & -step).bit_length() - 1
        bit = 1 << j
        v ^= bit
        w_p += 1 if (v & bit) else -1
        mask = int(syndrome_masks[j])
        w_s += bin(mask & ~syndrome).count("1") - bin(mask & syndrome).count("1")
        syndrome ^= mask
        ell ^= int(label_masks[j])
        table[w_p, w_s, ell] += 1


def build_full_table(
    model,
    frame,
    gibbs_syndrome_argument=None,
    force_python=False,
    **legacy,
):
    """Build ``N[w_p,w_s,label]`` for an arbitrary Gibbs argument."""
    if "sigma_arg" in legacy:
        if gibbs_syndrome_argument is not None:
            raise TypeError("pass only gibbs_syndrome_argument")
        warnings.warn(
            "sigma_arg is deprecated; use gibbs_syndrome_argument",
            DeprecationWarning,
            stacklevel=2,
        )
        gibbs_syndrome_argument = legacy.pop("sigma_arg")
    if legacy:
        raise TypeError(f"unexpected arguments: {tuple(legacy)}")
    if gibbs_syndrome_argument is None:
        raise TypeError("missing gibbs_syndrome_argument")
    _guards_full(model)
    gibbs_syndrome_argument = _validated_gibbs_argument(
        model, gibbs_syndrome_argument
    )
    n, m_c, k = model.num_qubits, model.num_checks, model.k
    syndrome_masks = np.array(
        [_bits_to_uint64(model.H_check[:, j]) for j in range(n)],
        dtype=np.uint64,
    )
    label_masks = np.array(
        [
            sum(
                (1 << u) for u in range(k) if frame.W_basis[u, j]
            ) if k else 0
            for j in range(n)
        ],
        dtype=np.int64,
    )
    sigma_init = _bits_to_uint64(gibbs_syndrome_argument)
    table = np.zeros((n + 1, m_c + 1, 1 << k), dtype=np.int64)
    if NUMBA_AVAILABLE and not force_python:
        stride_ws = 1 << k
        stride_wp = (m_c + 1) * stride_ws
        _fill_full_table_nb(table.reshape(-1), n, stride_ws, stride_wp,
                            syndrome_masks, label_masks, sigma_init)
    else:
        _fill_full_table_py(table, n, syndrome_masks, label_masks, sigma_init)
    assert int(table.sum()) == (1 << n), "full table total mismatch"
    return ExactTable(
        kind="full",
        table=table,
        n=n,
        num_checks=m_c,
        k=k,
        gibbs_syndrome_argument=np.asarray(
            gibbs_syndrome_argument, dtype=np.uint8
        ).copy(),
        total_states=1 << n,
    )


def build_coset_table(
    model,
    frame,
    gibbs_syndrome_argument=None,
    force_python=False,
    **legacy,
):
    """Build the ``q=0`` quenched-coset table for an image syndrome."""
    if "sigma_arg" in legacy:
        if gibbs_syndrome_argument is not None:
            raise TypeError("pass only gibbs_syndrome_argument")
        warnings.warn(
            "sigma_arg is deprecated; use gibbs_syndrome_argument",
            DeprecationWarning,
            stacklevel=2,
        )
        gibbs_syndrome_argument = legacy.pop("sigma_arg")
    if legacy:
        raise TypeError(f"unexpected arguments: {tuple(legacy)}")
    if gibbs_syndrome_argument is None:
        raise TypeError("missing gibbs_syndrome_argument")
    _guards_coset(model)
    gibbs_syndrome_argument = _validated_gibbs_argument(
        model, gibbs_syndrome_argument
    )
    n, k = model.num_qubits, model.k
    kernel = gf2_nullspace(model.H_check)
    dim = kernel.shape[0]
    if dim > MAX_COSET_DIM:
        raise ValueError(f"coset guard: kernel dim={dim} > {MAX_COSET_DIM}")
    v0_bits = model.logical_sector_section.apply(
        np.asarray(gibbs_syndrome_argument), strict=True
    )
    generator_masks = np.array(
        [_bits_to_uint64(kernel[i]) for i in range(dim)], dtype=np.uint64
    )
    generator_labels = np.array(
        [_label_bits_of(frame, kernel[i]) for i in range(dim)], dtype=np.int64
    )
    v0 = _bits_to_uint64(v0_bits)
    ell0 = _label_bits_of(frame, v0_bits)
    table = np.zeros((n + 1, 1 << k), dtype=np.int64)
    if NUMBA_AVAILABLE and not force_python and dim > 0:
        _fill_coset_table_nb(table.reshape(-1), dim, 1 << k,
                             generator_masks, generator_labels, v0, ell0)
    else:
        v = int(v0)
        ell = ell0
        table[bin(v).count("1"), ell] += 1
        for step in range(1, 1 << dim):
            j = (step & -step).bit_length() - 1
            v ^= int(generator_masks[j])
            ell ^= int(generator_labels[j])
            table[bin(v).count("1"), ell] += 1
    assert int(table.sum()) == (1 << dim), "coset table total mismatch"
    return ExactTable(
        kind="coset",
        table=table,
        n=n,
        num_checks=model.num_checks,
        k=k,
        gibbs_syndrome_argument=np.asarray(
            gibbs_syndrome_argument, dtype=np.uint8
        ).copy(),
        total_states=1 << dim,
    )


def evaluate_table(
    exact_table,
    K_p,
    K_q=None,
    planted_class_bits=0,
    **legacy,
):
    """Evaluate an exact table and return the complete v2 sector oracle."""
    if "ell_ref_bits" in legacy:
        if planted_class_bits != 0:
            raise TypeError("pass only planted_class_bits")
        warnings.warn(
            "ell_ref_bits is deprecated; use planted_class_bits",
            DeprecationWarning,
            stacklevel=2,
        )
        planted_class_bits = legacy.pop("ell_ref_bits")
    if legacy:
        raise TypeError(f"unexpected arguments: {tuple(legacy)}")
    k = exact_table.k
    num_labels = 1 << k
    table = exact_table.table
    if exact_table.kind == "full":
        if K_q is None:
            raise ValueError("full table evaluation requires K_q")
        w_p_axis = np.arange(table.shape[0])[:, None, None]
        w_s_axis = np.arange(table.shape[1])[None, :, None]
        log_terms = np.where(
            table > 0,
            np.log(np.maximum(table, 1))
            - _coupling_energy(K_p, w_p_axis)
            - _coupling_energy(K_q, w_s_axis),
            -np.inf,
        )
        wp_grid = np.broadcast_to(w_p_axis, table.shape)
        ws_grid = np.broadcast_to(w_s_axis, table.shape)
    else:
        w_p_axis = np.arange(table.shape[0])[:, None]
        log_terms = np.where(
            table > 0,
            np.log(np.maximum(table, 1))
            - _coupling_energy(K_p, w_p_axis),
            -np.inf,
        )
        wp_grid = np.broadcast_to(w_p_axis, table.shape)
        ws_grid = np.zeros_like(wp_grid)

    flat = log_terms.reshape(-1, num_labels)
    wp_flat = wp_grid.reshape(-1, num_labels)
    ws_flat = ws_grid.reshape(-1, num_labels)
    log_Z_label = np.full(num_labels, -np.inf)
    mean_wp_label = np.zeros(num_labels)
    mean_ws_label = np.zeros(num_labels)
    for ell in range(num_labels):
        col = flat[:, ell]
        finite = np.isfinite(col)
        if not finite.any():
            continue
        m = col[finite].max()
        expcol = np.exp(col[finite] - m)
        z = expcol.sum()
        log_Z_label[ell] = m + np.log(z)
        mean_wp_label[ell] = float(
            (expcol * wp_flat[finite, ell]).sum() / z
        )
        mean_ws_label[ell] = float(
            (expcol * ws_flat[finite, ell]).sum() / z
        )
    m_all = log_Z_label[np.isfinite(log_Z_label)].max()
    weights_abs = np.exp(np.where(np.isfinite(log_Z_label),
                                  log_Z_label - m_all, -np.inf))
    total = weights_abs.sum()
    weights_abs = weights_abs / total
    log_Z = m_all + np.log(total)

    planted_class_bits = int(planted_class_bits)
    if not 0 <= planted_class_bits < num_labels:
        raise ValueError("planted_class_bits outside logical-sector range")
    relative = np.array([
        weights_abs[t ^ planted_class_bits] for t in range(num_labels)
    ])
    u_bitmasks = np.arange(1, num_labels, dtype=np.int64)
    characters_absolute = characters_from_sector_weights(weights_abs)
    characters_relative = characters_from_sector_weights(relative)
    basis_indices = np.array([(1 << bit) - 1 for bit in range(k)], dtype=int)
    m_basis_absolute = characters_absolute[basis_indices]
    m_basis_relative = characters_relative[basis_indices]
    statistics = posterior_statistics(weights_abs, planted_class_bits)
    mean_wp = float(np.dot(weights_abs, mean_wp_label))
    mean_ws = float(np.dot(weights_abs, mean_ws_label))
    result = {
        "log_Z": float(log_Z),
        "weights_absolute": weights_abs,
        "weights_relative": relative,
        "u_bitmasks": u_bitmasks,
        "characters_absolute": characters_absolute,
        "characters_relative": characters_relative,
        "m_u_absolute": characters_absolute,
        "m_u_relative": characters_relative,
        "m_u_basis_absolute": m_basis_absolute,
        "m_u_basis_relative": m_basis_relative,
        # Compatibility key: historical m_u_basis was planted-relative.
        "m_u_basis": m_basis_relative,
        "q_top_absolute": statistics["q_top"],
        "q_top_relative": statistics["q_top"],
        "mean_Wp": mean_wp,
        "mean_Ws": mean_ws,
        "mu_per_label": mean_wp_label,
        "log_Z_per_label": log_Z_label,
    }
    result.update(statistics)
    return result


def exact_reference(model, frame, wiring, force_python=False):
    """Evaluate the exact oracle for an ``EnsembleWiring`` instance."""
    planted_class_bits = 0
    for b, val in enumerate(np.asarray(wiring.planted_logical_class)):
        if val:
            planted_class_bits |= 1 << b
    if wiring.q_zero:
        table = build_coset_table(
            model,
            frame,
            wiring.gibbs_syndrome_argument,
            force_python=force_python,
        )
        result = evaluate_table(
            table, wiring.K_p, planted_class_bits=planted_class_bits
        )
    else:
        table = build_full_table(
            model,
            frame,
            wiring.gibbs_syndrome_argument,
            force_python=force_python,
        )
        result = evaluate_table(
            table,
            wiring.K_p,
            wiring.K_q,
            planted_class_bits=planted_class_bits,
        )
    result["table"] = table
    result["planted_logical_class_bits"] = planted_class_bits
    formal_purity = float(
        np.sum(np.asarray(result["weights_absolute"]) ** 2)
    )
    result["formal_sector_purity"] = formal_purity
    result["largest_sector_mass"] = float(
        np.max(np.asarray(result["weights_absolute"]))
    )
    if wiring.ensemble == "legacy_delta_only":
        # Preserve the exact regression data under explicitly formal names.
        # None-valued paper fields keep the schema stable without presenting
        # legacy delta-only sectors as a decoding posterior.
        result.update({
            "formal_sector_weights_absolute": result["weights_absolute"],
            "formal_sector_weights_relative": result["weights_relative"],
            "formal_sector_characters_absolute": result[
                "characters_absolute"
            ],
            "formal_sector_characters_relative": result[
                "characters_relative"
            ],
            "formal_m_u_basis_absolute": result["m_u_basis_absolute"],
            "formal_m_u_basis_relative": result["m_u_basis_relative"],
            "formal_q_top": result["q_top"],
            "formal_q_top_absolute": result["q_top_absolute"],
            "formal_q_top_relative": result["q_top_relative"],
        })
        for name in (
            "weights_absolute",
            "weights_relative",
            "characters_absolute",
            "characters_relative",
            "m_u_absolute",
            "m_u_relative",
            "m_u_basis_absolute",
            "m_u_basis_relative",
            "m_u_basis",
            "q_top",
            "q_top_absolute",
            "q_top_relative",
            "posterior_purity",
            "posterior_mass_on_planted_class",
            "map_success_probability",
            "map_success_lower_bound",
            "map_success_upper_bound",
            "posterior_purity_within_physical_bounds",
        ):
            result[name] = None
    return result
