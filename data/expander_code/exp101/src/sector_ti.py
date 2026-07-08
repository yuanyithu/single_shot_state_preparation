"""sector-TI 引擎（G2.8）：exp37 生产路径的泛化（notes/00 §2.3、notes/01 §7）。

固定 label 扇区的受限配分函数 Z_ℓ(K_p)，dF/dK_p = ⟨|v|⟩_ℓ ⇒
    ΔF_ℓ(K_p*) = ∫_0^{K_p*} μ_ℓ dK_p − (ℓ_anchor 同式)，trapezoid。
K_p 网格自 0（p=0.5，各扇区 Z 相等 ⇒ ΔF(0)=0 解析成立）升到目标值，
链沿网格退火续跑。天然绕开冻结扇区（每扇区独立采样，无需跨壁垒）。

扇区档位：
  - "full"（k ≤ full_max_k）：全部 2^k 扇区（绝对 label），事后相对 ℓ_ref 重排；
    q_top = (2^k Σw² − 1)/(2^k − 1)，w0 = w_{ℓ_ref}，m_u = Σ_t P̃(t)(−1)^{⟨u,t⟩}。
  - "pairwise"（k > full_max_k）：k+1 条链，label ∈ {ℓ_ref, ℓ_ref⊕e_u}——
    **围绕真类 ℓ_ref 展开**（true 系综的正确锚点；repo_compat 的 ℓ_ref=0
    自动退化为 exp37 的 {0, e_u}）。m_u^{pair} = tanh(ΔF̃_u/2)；
    其估计性质须在小 k 与 full/枚举定量标定（G3.2），未标定不得外推（plan 风险 12）。

sector-preserving proposals（q>0；全部保 label）：
  - 零签名单比特（W 列全零的 qubit）：动 T⊕S，改 syndrome（K_q 计费）
  - 同签名 qubit 对（W 列相同且非零）：组内全对
  - stabilizer 行（S-move，syndrome/label 双保）
q=0：仅 stabilizer 行（coset 内固定扇区采样），代表元 = r(σ_arg)⊕combo。

numba fast path：**挂账至 G4.2**（性能门槛处落地；本文件为正确性权威参考版）。
"""

from dataclasses import dataclass

import numpy as np

from .gf2 import gf2_matmul
from .observables import DEFAULT_FULL_MAX_K
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

    q>0：v = combo(label)（零 syndrome 代表）；q=0：v = r(σ_arg) ⊕ combo。
    """
    v = np.zeros(model.num_qubits, dtype=np.uint8)
    for bit in range(model.k):
        if (int(label_bits) >> bit) & 1:
            v ^= model.logical_move_basis[bit]
    if wiring.q_zero:
        v = v ^ model.section.apply(wiring.sigma_arg, strict=True)
    return v


def _run_fixed_sector_chain(model, wiring, proposals, v0, kp_grid, config,
                            rng):
    """单扇区链：沿 kp_grid 退火，逐点测 μ=⟨|v|⟩、μ_s=⟨|Hv⊕σ|⟩、block 均值。"""
    v = v0.copy()
    syndrome_term = (
        gf2_matmul(model.H_check, v[:, None])[:, 0] ^ wiring.sigma_arg
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


def _q_top_from_weights(weights, k):
    weights = np.asarray(weights, dtype=np.float64)
    if k == 0:
        return None
    return float(((1 << k) * np.sum(weights ** 2) - 1.0) / ((1 << k) - 1))


def run_sector_ti(model, frame, wiring, config, seed):
    """TI 主入口。返回 dict（tier 依 k 与 full_max_k 自动选择）。"""
    if model.k == 0:
        raise ValueError("sector TI requires k >= 1")
    kp_target = wiring.K_p
    if not np.isfinite(kp_target) or kp_target <= 0:
        raise ValueError("K_p target must be finite positive (p<0.5, p>0)")
    kp_grid = np.linspace(0.0, kp_target, int(config.num_kp_grid_points))
    proposals = build_sector_preserving_proposals(model, frame)
    rng = PortablePrng(seed)
    ell_ref_bits = 0
    for bit, val in enumerate(np.asarray(wiring.reference_label)):
        if val:
            ell_ref_bits |= 1 << bit

    tier = "full" if model.k <= config.full_max_k else "pairwise"
    if tier == "full":
        labels = list(range(1 << model.k))
    else:
        labels = [ell_ref_bits] + [
            ell_ref_bits ^ (1 << u) for u in range(model.k)
        ]

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

    # bootstrap（block 重采样 → ΔF/权重/q_top 的不确定度）
    rng_boot = np.random.default_rng(seed ^ 0x5EED)
    block_arrays = np.stack(
        [chains[lab]["block_mu"] for lab in labels]
    )  # (L, G, B)
    num_labels, num_grid, num_blocks = block_arrays.shape

    def weights_from_delta(delta_values):
        shifted = -np.asarray(delta_values)
        shifted -= shifted.max()
        w = np.exp(shifted)
        return w / w.sum()

    boot_stats = []
    for _ in range(int(config.num_bootstrap)):
        idx = rng_boot.integers(0, num_blocks, size=num_blocks)
        mu_boot = block_arrays[:, :, idx].mean(axis=2)
        ints = _integrate(kp_grid, mu_boot)
        boot_stats.append(ints - ints[0])
    boot_stats = np.array(boot_stats)  # (B, L)

    result = {
        "tier": tier,
        "labels": labels,
        "ell_ref": ell_ref_bits,
        "kp_grid": kp_grid,
        "delta_f": np.array([delta_f[lab] for lab in labels]),
        "delta_f_stderr": boot_stats.std(axis=0, ddof=1),
        "acceptance_per_label": {
            lab: chains[lab]["acceptance"] for lab in labels
        },
        "proposal_summary": {k_: v_ for k_, v_ in proposals.items()
                             if k_ != "supports"},
    }

    if tier == "full":
        weights_abs = weights_from_delta(result["delta_f"])
        # 相对真类重排：P̃(t) = w_{t ⊕ ℓ_ref}
        relative = np.array(
            [weights_abs[labels.index(t ^ ell_ref_bits)]
             for t in range(1 << model.k)]
        )
        m_u = np.zeros(model.k)
        for u in range(model.k):
            s = 0.0
            for t in range(1 << model.k):
                sign = -1.0 if (t >> u) & 1 else 1.0
                s += relative[t] * sign
            m_u[u] = s
        boot_qtop = []
        for b in range(boot_stats.shape[0]):
            w_b = weights_from_delta(boot_stats[b])
            boot_qtop.append(_q_top_from_weights(w_b, model.k))
        result.update({
            "weights_absolute": weights_abs,
            "weights_relative": relative,
            "w0": float(relative[0]),
            "q_top": _q_top_from_weights(weights_abs, model.k),
            "q_top_stderr": float(np.std(boot_qtop, ddof=1)),
            "m_u_basis": m_u,
        })
        # 粗网格 flags
        coarse_idx = np.arange(0, num_grid, 2)
        if coarse_idx[-1] != num_grid - 1:
            coarse_idx = np.append(coarse_idx, num_grid - 1)
        ints_coarse = np.array([
            float(_integrate(kp_grid[coarse_idx],
                             chains[lab]["mu"][coarse_idx]))
            for lab in labels
        ])
        delta_coarse = ints_coarse - ints_coarse[0]
        w_coarse = weights_from_delta(delta_coarse)
        grid_tv = float(0.5 * np.abs(weights_abs - w_coarse).sum())
        q_top_coarse = _q_top_from_weights(w_coarse, model.k)
        result["grid_tv"] = grid_tv
        result["grid_q_top_abs_diff"] = float(
            abs(result["q_top"] - q_top_coarse)
        )
        flags = []
        if grid_tv > config.grid_tv_warning:
            flags.append("TI_GRID_TV_WARN")
        if result["grid_q_top_abs_diff"] > config.grid_q_top_warning:
            flags.append("TI_GRID_QTOP_WARN")
        result["flags"] = ";".join(flags) if flags else "PASS"
    else:
        # pairwise：ΔF̃_u = F(ℓ_ref⊕e_u) − F(ℓ_ref)
        delta_u = np.array([delta_f[labels[1 + u]] for u in range(model.k)])
        m_u_pair = np.tanh(delta_u / 2.0)
        boot_mu = np.tanh(boot_stats[:, 1:] / 2.0)
        result.update({
            "delta_f_per_u": delta_u,
            "m_u_pairwise": m_u_pair,
            "m_u_pairwise_stderr": boot_mu.std(axis=0, ddof=1),
            "q_top_pairwise": float(np.mean(m_u_pair ** 2)),
            "flags": "PAIRWISE",
        })
    return result
