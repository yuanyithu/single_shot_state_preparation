"""G0.1/G0.2 判决实验：主项目 Gibbs 模型的盘度语义。

问题：主项目模型 pi(v) ∝ exp[-K_p|v⊕η| - K_q|H_Z v⊕s|]（CLAUDE.md 公式、
exact_enumeration.py 与 mcmc.py 一致实现）。换元 u=v⊕η 后得
exp[-K_p|u| - K_q|H_Z u⊕δ|]，即数据项 η-盘度消失、只剩测量噪声 δ 盘度。
标准 decoding posterior 应为 pi*(c) ∝ exp[-K_p|c| - K_q|H_Z c⊕s|]
（换元后 = exp[-K_p|x⊕η| - K_q|H_Z x⊕δ|]，两种盘度都在）。

本脚本在 2D toric L=2/L=3 上全枚举验证：
  T1  repo-模型的 m_u 在线性 frame 下与 η 无关（只依赖 δ）；
  T2  repo exact_enumeration.compute_exact_logical_observable_means 与本脚本
      repo-模型复算（同一 bplsd section frame）逐位一致 —— 证明代码读得对；
  T3  true-模型满足恒等式 m_u^true(η,δ) = (-1)^<w_u,η> · m_u^true(0, s)，
      其中 s = H_Zη⊕δ（线性 frame）；
  T4  当 H_Zη ≠ 0 时 repo-模型与 true-模型的 m_u/q_top 数值不同；
  T5  q=0 限制（枚举限制在 coset）：repo-模型类分布与 η 无关（clean），
      true-模型 η-依赖（RBIM 型 quenched）。

结论写入 result.json 与 stdout 表格。
"""

import json
import sys
from pathlib import Path

import numpy as np

SRC = "/Users/jarvis/Desktop/sync/project D/src"
sys.path.insert(0, SRC)

from build_toric_code_examples import build_2d_toric_code  # noqa: E402
from preprocessing import build_logical_observable_masks  # noqa: E402
from linear_section import (  # noqa: E402
    apply_linear_section,
    apply_section,
    build_linear_section,
    build_syndrome_representative_section,
)
from exact_enumeration import compute_exact_logical_observable_means  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent


def build_all_v(num_qubits):
    return (
        (np.arange(1 << num_qubits, dtype=np.int64)[:, None]
         >> np.arange(num_qubits)[None, :]) & 1
    ).astype(np.uint8)


def linear_section_matrix(parity_check_matrix, linear_section_data):
    """R ∈ F2^{n×m}: r(σ)=Rσ，逐个单位 syndrome 列构造。"""
    num_checks = parity_check_matrix.shape[0]
    num_qubits = parity_check_matrix.shape[1]
    R = np.zeros((num_qubits, num_checks), dtype=np.uint8)
    for i in range(num_checks):
        e = np.zeros(num_checks, dtype=bool)
        e[i] = True
        R[:, i] = apply_linear_section(e, linear_section_data).astype(np.uint8)
    return R


def enum_m_u(parity_check_matrix, masks, eta, delta, p, q, mode, frame,
             section_obj=None, R=None, q_zero=False):
    """全枚举 m_u。

    mode: "repo"  → 权重 -Kp|v⊕η| - Kq|Hv⊕s|
          "true"  → 权重 -Kp|v|   - Kq|Hv⊕s|
    frame: "linear"（R 矩阵） 或 "bplsd"（section_obj，逐样本 apply）
    label 一律 = v ⊕ η ⊕ r(Hv) ⊕ r(Hη)（与主项目观测量一致）。
    q_zero=True 时把权重限制在 Hv==s 的 coset 上（q=0 硬约束）。
    """
    H8 = parity_check_matrix.astype(np.uint8)
    n = H8.shape[1]
    eta8 = eta.astype(np.uint8)
    s8 = ((H8 @ eta8) % 2) ^ delta.astype(np.uint8)
    Kp = float(np.log((1 - p) / p))
    all_v = build_all_v(n)
    syn = (all_v @ H8.T) % 2
    syn_term = np.count_nonzero(syn ^ s8[None, :], axis=1)
    if mode == "repo":
        data_term = np.count_nonzero(all_v ^ eta8[None, :], axis=1)
    elif mode == "true":
        data_term = np.count_nonzero(all_v, axis=1)
    else:
        raise ValueError(mode)
    if q_zero:
        logw = np.where(syn_term == 0, -Kp * data_term.astype(np.float64), -np.inf)
    else:
        Kq = float(np.log((1 - q) / q))
        logw = -Kp * data_term.astype(np.float64) - Kq * syn_term.astype(np.float64)
    logw -= logw.max()
    w = np.exp(logw)
    w /= w.sum()

    if frame == "linear":
        rep = (syn @ R.T.astype(np.uint8)) % 2
        rep_eta = (R @ ((H8 @ eta8) % 2)) % 2
    elif frame == "bplsd":
        rep = np.empty_like(all_v)
        for row in range(all_v.shape[0]):
            rep[row] = apply_section(syn[row].astype(bool), section_obj).astype(np.uint8)
        rep_eta = apply_section(((H8 @ eta8) % 2).astype(bool), section_obj).astype(np.uint8)
    else:
        raise ValueError(frame)
    label = all_v ^ eta8[None, :] ^ rep ^ rep_eta[None, :]
    parity = (label @ masks.astype(np.uint8).T) % 2
    O = 1.0 - 2.0 * parity.astype(np.float64)
    m_u = w @ O
    return m_u


def main():
    rng = np.random.default_rng(20260707)
    report = {"cases": [], "conclusions": {}}
    tol = 1e-12

    for L in (2, 3):
        H, zbasis = build_2d_toric_code(L)
        n = H.shape[1]
        masks = build_logical_observable_masks(H, zbasis, None)
        lin = build_linear_section(H)
        R = linear_section_matrix(H, lin)
        # w_u（线性 frame 观测泛函）用于 T3 的符号 (-1)^<w_u,η>
        W = np.zeros((masks.shape[0], n), dtype=np.uint8)
        for j in range(n):
            e = np.zeros(n, dtype=bool)
            e[j] = True
            v = e ^ apply_linear_section(
                ((H.astype(np.uint8) @ e.astype(np.uint8)) % 2).astype(bool), lin)
            W[:, j] = (masks.astype(np.uint8) @ v.astype(np.uint8)) % 2

        p, q = 0.15, 0.10
        delta = rng.random(H.shape[0]) < q
        eta0 = np.zeros(n, dtype=bool)
        eta1 = np.zeros(n, dtype=bool)
        eta1[0] = True                      # 单比特，H_Zη ≠ 0
        eta2 = rng.random(n) < 0.25         # 随机较重
        if not ((H.astype(np.uint8) @ eta2.astype(np.uint8)) % 2).any():
            eta2[1] ^= True

        # --- T1: repo 模型 η-无关（线性 frame）---
        m_repo = {}
        for name, eta in (("eta0", eta0), ("eta1", eta1), ("eta2", eta2)):
            m_repo[name] = enum_m_u(H, masks, eta, delta, p, q, "repo", "linear", R=R)
        t1_dev = max(
            float(np.max(np.abs(m_repo["eta1"] - m_repo["eta0"]))),
            float(np.max(np.abs(m_repo["eta2"] - m_repo["eta0"]))),
        )

        # --- T2: 与主项目 exact_enumeration 互证（bplsd frame）---
        section_obj = build_syndrome_representative_section(H)
        eta_t2 = eta2
        s_t2 = (((H.astype(np.uint8) @ eta_t2.astype(np.uint8)) % 2).astype(bool)
                ^ delta)
        repo_func = compute_exact_logical_observable_means(
            parity_check_matrix=H,
            observed_syndrome_bits=s_t2,
            disorder_data_error_bits=eta_t2,
            syndrome_error_probability=q,
            data_error_probability=p,
            logical_observable_masks=masks,
        )
        mine_bplsd = enum_m_u(H, masks, eta_t2, delta, p, q, "repo", "bplsd",
                              section_obj=section_obj)
        t2_dev = float(np.max(np.abs(repo_func["m_u_values"] - mine_bplsd)))

        # --- T3: true 模型 gauge 恒等式（线性 frame）---
        m_true_eta2 = enum_m_u(H, masks, eta2, delta, p, q, "true", "linear", R=R)
        s2 = (((H.astype(np.uint8) @ eta2.astype(np.uint8)) % 2).astype(bool) ^ delta)
        m_true_0_s = enum_m_u(H, masks, eta0, s2, p, q, "true", "linear", R=R)
        sign = 1.0 - 2.0 * ((W @ eta2.astype(np.uint8)) % 2).astype(np.float64)
        t3_dev = float(np.max(np.abs(m_true_eta2 - sign * m_true_0_s)))

        # --- T4: repo vs true 数值差异（H_Zη≠0）---
        q_top_repo = float(np.mean(m_repo["eta2"] ** 2))
        q_top_true = float(np.mean(m_true_eta2 ** 2))
        t4_mu_diff = float(np.max(np.abs(m_repo["eta2"] - m_true_eta2)))
        t4_qtop_diff = abs(q_top_repo - q_top_true)

        # --- T5: q=0 限制下 repo=clean(η-无关) / true=quenched(η-依赖) ---
        m_repo_q0_a = enum_m_u(H, masks, eta0, np.zeros(H.shape[0], bool),
                               p, q, "repo", "linear", R=R, q_zero=True)
        m_repo_q0_b = enum_m_u(H, masks, eta2, np.zeros(H.shape[0], bool),
                               p, q, "repo", "linear", R=R, q_zero=True)
        m_true_q0_a = enum_m_u(H, masks, eta0, np.zeros(H.shape[0], bool),
                               p, q, "true", "linear", R=R, q_zero=True)
        m_true_q0_b = enum_m_u(H, masks, eta2, np.zeros(H.shape[0], bool),
                               p, q, "true", "linear", R=R, q_zero=True)
        t5_repo_dev = float(np.max(np.abs(m_repo_q0_a - m_repo_q0_b)))
        t5_true_dev = float(np.max(np.abs(np.abs(m_true_q0_a) - np.abs(m_true_q0_b))))

        case = {
            "L": L, "n": n, "p": p, "q": q,
            "delta_weight": int(np.count_nonzero(delta)),
            "eta2_weight": int(np.count_nonzero(eta2)),
            "T1_repo_eta_independence_maxdev": t1_dev,
            "T2_repo_func_vs_reimpl_maxdev": t2_dev,
            "T3_true_gauge_identity_maxdev": t3_dev,
            "T4_repo_vs_true_max_mu_diff": t4_mu_diff,
            "T4_qtop_repo": q_top_repo,
            "T4_qtop_true": q_top_true,
            "T5_q0_repo_eta_dependence": t5_repo_dev,
            "T5_q0_true_eta_dependence_absmu": t5_true_dev,
        }
        report["cases"].append(case)
        print(f"\n=== 2D toric L={L} (n={n}) p={p} q={q} ===")
        for k, v in case.items():
            if k in ("L", "n", "p", "q"):
                continue
            print(f"  {k}: {v:.6g}" if isinstance(v, float) else f"  {k}: {v}")

    ok_t1 = all(c["T1_repo_eta_independence_maxdev"] < tol for c in report["cases"])
    ok_t2 = all(c["T2_repo_func_vs_reimpl_maxdev"] < tol for c in report["cases"])
    ok_t3 = all(c["T3_true_gauge_identity_maxdev"] < tol for c in report["cases"])
    sig_t4 = all(c["T4_repo_vs_true_max_mu_diff"] > 1e-3 for c in report["cases"])
    sig_t5 = (all(c["T5_q0_repo_eta_dependence"] < tol for c in report["cases"])
              and all(c["T5_q0_true_eta_dependence_absmu"] > 1e-3
                      for c in report["cases"]))
    report["conclusions"] = {
        "T1_repo_model_is_eta_free_in_linear_frame": bool(ok_t1),
        "T2_code_reading_confirmed_bitwise": bool(ok_t2),
        "T3_true_model_gauge_identity_holds": bool(ok_t3),
        "T4_repo_and_true_models_differ_numerically": bool(sig_t4),
        "T5_q0_repo_clean_true_quenched": bool(sig_t5),
    }
    print("\n=== conclusions ===")
    for k, v in report["conclusions"].items():
        print(f"  {k}: {v}")
    with (OUT_DIR / "result.json").open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"\nwritten: {OUT_DIR / 'result.json'}")


if __name__ == "__main__":
    main()
