"""G3.3 V1c：section-frame A/B（plan §3 G3.3, notes/01 §4）。

frames：linear-A（默认列序）、linear-B（逆列序，不同主元）、decoder（BpLsd）。

关键物理（notes/01 §4，本脚本据此设计判据）：
  换 frame r→r' ⇒ 标签整体平移 φ'(c)=φ(c)⊕λ(Hc)。q=0 时 Hc=s 固定 ⇒ λ 为常数
  c0=⟨z,r_A(s)⊕r_B(s)⟩；ℓ_ref=φ(η) 随之平移同一 c0（δ=0⇒Hη=s）⇒ **相对真类的
  分布 P̃(t)=w(t⊕ℓ_ref)、m_u、q_top 精确 frame 无关**（绝对 label 权重则平移，不比）。
  q>0 时不同 syndrome 扇区被 λ 重排 ⇒ 相对分布也 frame 依赖（gauge=修正协议）。

gates：
  G1（主）每 frame 内 枚举=MCMC：单条 frame 无关 state 轨迹在 A/B 两 frame 下各自
       读出 m_u，对该 frame 枚举 z ≤ 5（观测机制 frame 自洽）。
  G2 q=0 frame 无关（精确）：linear-A/B/decoder 三 frame 的**相对**类权重两两 TVD
       ≤ 1e-9，且 m_u_basis 逐分量 ≤ 1e-9。
  G3 q>0 frame 依赖被观测：至少一实例 A/B 相对权重 TVD > 1e-3（记录，非失败）。
  G4 三 frame 指纹互异。
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

EXP101_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXP101_ROOT))

from src.enumerate_exact import build_full_table, evaluate_table  # noqa: E402
from src.gf2 import gf2_matmul  # noqa: E402
from src.graphs import cycle_parity_check_matrix, repetition_parity_check_matrix  # noqa: E402
from src.hgp import hgp_from_H  # noqa: E402
from src.logicals import logical_pauli_operators  # noqa: E402
from src.model import assemble_sector_model, draw_disorder, wire_ensemble  # noqa: E402
from src.observables import ObservableFrame, build_observable_set  # noqa: E402
from src.reference_mcmc import ReferenceMcmcConfig, run_reference_mcmc  # noqa: E402
from src.section import (  # noqa: E402
    DecoderObservableFrame,
    DecoderSection,
    build_linear_section,
)

OUT_DIR = Path(__file__).resolve().parent
Z_FLOOR = 5e-3

INSTANCES = {
    "toric_m2": lambda: cycle_parity_check_matrix(2),
    "surface_m3": lambda: repetition_parity_check_matrix(3),
    "toric_m3": lambda: cycle_parity_check_matrix(3),
}


def frame_from_section(model, section):
    Z = model.logical_obs_basis
    k = Z.shape[0]
    RH = section.section_after_H(model.H_check)
    W = (Z ^ gf2_matmul(Z, RH)).astype(np.uint8)
    assert not gf2_matmul(W, RH).any()
    assert not gf2_matmul(W, model.stabilizer_rows.T).any()
    assert np.array_equal(gf2_matmul(W, model.logical_move_basis.T),
                          np.eye(k, dtype=np.uint8))
    return ObservableFrame(W_basis=W, k=k, num_qubits=model.num_qubits,
                           section_fingerprint=section.fingerprint())


def label_bits(W, v):
    par = gf2_matmul(W, np.asarray(v)[:, None])[:, 0]
    return sum(1 << b for b, x in enumerate(par) if x)


def m_u_from_states(states, W, ell_ref_bits, k):
    par = (states.astype(np.uint8) @ W.T) % 2
    signs = 1 - 2 * par.astype(np.int64)
    ref = np.array([1 - 2 * ((ell_ref_bits >> u) & 1) for u in range(k)])
    return signs.mean(axis=0) * ref


def sign_traj(states, W):
    par = (states.astype(np.uint8) @ W.T) % 2
    return (1 - 2 * par.astype(np.int64))


def stderr_of(sign_series):
    blocks = np.array_split(sign_series.astype(np.float64), 20, axis=0)
    means = np.array([b.mean(axis=0) for b in blocks])
    return np.maximum(means.std(axis=0, ddof=1) / np.sqrt(20), Z_FLOOR)


def enum_linear(model, frame, wiring, ell_ref_bits):
    table = build_full_table(model, frame, wiring.sigma_arg)
    return evaluate_table(table, wiring.K_p,
                          None if wiring.q_zero else wiring.K_q,
                          ell_ref_bits=ell_ref_bits)


def enum_decoder_relative(model, dec_frame, wiring, eta):
    """decoder（非线性）frame 精确相对类权重（相对 ℓ_ref=φ_dec(η)）。"""
    n, k = model.num_qubits, model.k
    weights = np.zeros(1 << k)
    for value in range(1 << n):
        v = np.array([(value >> j) & 1 for j in range(n)], dtype=np.uint8)
        syn = gf2_matmul(model.H_check, v[:, None])[:, 0] ^ wiring.sigma_arg
        ws = int(syn.sum())
        if wiring.q_zero:
            if ws:
                continue
            lw = -wiring.K_p * float(v.sum())
        else:
            lw = -wiring.K_p * float(v.sum()) - wiring.K_q * float(ws)
        lab = dec_frame.label_of(v)
        bits = sum(1 << b for b, x in enumerate(lab) if x)
        weights[bits] += np.exp(lw)
    weights /= weights.sum()
    ell_ref = sum(1 << b for b, x in enumerate(dec_frame.label_of(eta)) if x)
    return np.array([weights[t ^ ell_ref] for t in range(1 << k)])


def main():
    started = time.perf_counter()
    records = []
    for name, builder in INSTANCES.items():
        classical = builder()
        H_Z, H_X = hgp_from_H(classical)
        logicals = logical_pauli_operators(H_X, H_Z)
        model = assemble_sector_model(H_X, H_Z, logicals, sector="x_error")
        n = model.num_qubits
        section_A = build_linear_section(model.H_check)
        section_B = build_linear_section(
            model.H_check, column_priority=list(reversed(range(n))))
        frame_A = frame_from_section(model, section_A)
        frame_B = frame_from_section(model, section_B)
        dec_section = DecoderSection(model.H_check)
        dec_frame = DecoderObservableFrame(
            model.H_check, model.logical_obs_basis, dec_section)
        fps = [frame_A.section_fingerprint, frame_B.section_fingerprint,
               dec_section.fingerprint()]

        for p, q in [(0.15, 0.0), (0.12, 0.08), (0.2, 0.15)]:
            for d in range(6):
                rng = np.random.default_rng(abs(hash((name, p, q, d))) % 2**31)
                disorder = draw_disorder(model, p, q, rng)
                wiring = wire_ensemble(model, disorder, "true_posterior",
                                       frame_A)
                ellA = label_bits(frame_A.W_basis, disorder.eta)
                ellB = label_bits(frame_B.W_basis, disorder.eta)
                enumA = enum_linear(model, frame_A, wiring, ellA)
                enumB = enum_linear(model, frame_B, wiring, ellB)

                config = ReferenceMcmcConfig(
                    num_burn_in_sweeps=600, num_measurements=6000,
                    record_state_trajectory=True)
                obs_set = build_observable_set(frame_A)
                res = run_reference_mcmc(model, frame_A, obs_set, wiring, config,
                                         seed=int(rng.integers(0, 2**60)))
                states = res["state_trajectory"]

                mA = m_u_from_states(states, frame_A.W_basis, ellA, model.k)
                mB = m_u_from_states(states, frame_B.W_basis, ellB, model.k)
                refA = np.array([1 - 2 * ((ellA >> u) & 1)
                                 for u in range(model.k)])
                refB = np.array([1 - 2 * ((ellB >> u) & 1)
                                 for u in range(model.k)])
                stderrA = stderr_of(sign_traj(states, frame_A.W_basis) * refA)
                stderrB = stderr_of(sign_traj(states, frame_B.W_basis) * refB)
                zA = float(np.max(np.abs((mA - enumA["m_u_basis"])
                                         / np.maximum(stderrA, Z_FLOOR))))
                zB = float(np.max(np.abs((mB - enumB["m_u_basis"])
                                         / np.maximum(stderrB, Z_FLOOR))))
                rel_tvd = float(0.5 * np.abs(
                    enumA["weights_relative"] - enumB["weights_relative"]).sum())
                mu_basis_diff = float(np.max(np.abs(
                    enumA["m_u_basis"] - enumB["m_u_basis"])))
                records.append({
                    "instance": name, "p": p, "q": q, "disorder": d,
                    "q_zero": q == 0.0, "zA": zA, "zB": zB,
                    "ab_rel_tvd": rel_tvd, "mu_basis_diff_AB": mu_basis_diff,
                })

        # q=0 三 frame 精确相对无关（含 decoder），单独精确对比
        rng = np.random.default_rng(999)
        disorder = draw_disorder(model, 0.15, 0.0, rng)
        wiring = wire_ensemble(model, disorder, "true_posterior", frame_A)
        ellA = label_bits(frame_A.W_basis, disorder.eta)
        ellB = label_bits(frame_B.W_basis, disorder.eta)
        wA = enum_linear(model, frame_A, wiring, ellA)
        wB = enum_linear(model, frame_B, wiring, ellB)
        rel_dec = enum_decoder_relative(model, dec_frame, wiring, disorder.eta)
        records.append({
            "instance": name, "check": "q0_frame_independence",
            "rel_tvd_AB": float(0.5 * np.abs(
                wA["weights_relative"] - wB["weights_relative"]).sum()),
            "rel_tvd_A_dec": float(0.5 * np.abs(
                wA["weights_relative"] - rel_dec).sum()),
            "fingerprints_distinct": bool(len(set(fps)) == 3),
        })
        print(f"[{time.perf_counter()-started:7.1f}s] done {name}", flush=True)

    within = [r for r in records if "zA" in r]
    q0c = [r for r in records if r.get("check") == "q0_frame_independence"]
    G1 = all(r["zA"] < 5 and r["zB"] < 5 for r in within)
    G2 = (all(r["ab_rel_tvd"] < 1e-9 and r["mu_basis_diff_AB"] < 1e-9
              for r in within if r["q_zero"])
          and all(r["rel_tvd_AB"] < 1e-9 and r["rel_tvd_A_dec"] < 1e-9
                  for r in q0c))
    G3_tvd = max((r["ab_rel_tvd"] for r in within if r["q"] > 0), default=0.0)
    G3 = bool(G3_tvd > 1e-3)
    G4 = all(r["fingerprints_distinct"] for r in q0c)
    gates = {
        "G1_within_frame_pass": bool(G1),
        "G1_max_z": float(max(max(r["zA"], r["zB"]) for r in within)),
        "G2_q0_frame_independence_pass": bool(G2),
        "G2_q0_max_rel_tvd": float(max(
            [r["ab_rel_tvd"] for r in within if r["q_zero"]]
            + [r["rel_tvd_AB"] for r in q0c] + [r["rel_tvd_A_dec"] for r in q0c])),
        "G3_qpos_frame_dependence_max_tvd": float(G3_tvd),
        "G3_observed": G3,
        "G4_fingerprints_distinct": bool(G4),
        "wall_time_seconds": time.perf_counter() - started,
    }
    gates["ALL_PASS"] = bool(G1 and G2 and G3 and G4)
    with (OUT_DIR / "results.json").open("w", encoding="utf-8") as fh:
        json.dump({"records": records, "gates": gates}, fh, indent=1,
                  ensure_ascii=False)
    g = gates
    lines = [
        "# G3.3 V1c section-frame A/B 结果", "",
        f"墙钟 {g['wall_time_seconds']:.0f}s", "",
        "| gate | 内容 | 值 | 结果 |", "|---|---|---|---|",
        f"| G1 | 每 frame 内 枚举=MCMC (max z) | {g['G1_max_z']:.2f} (≤5) | "
        f"{'✅' if g['G1_within_frame_pass'] else '❌'} |",
        f"| G2 | q=0 相对分布 frame 无关（含 decoder，max rel-TVD） | "
        f"{g['G2_q0_max_rel_tvd']:.2e} (≤1e-9) | "
        f"{'✅' if g['G2_q0_frame_independence_pass'] else '❌'} |",
        f"| G3 | q>0 frame 依赖被观测（max A/B rel-TVD） | "
        f"{g['G3_qpos_frame_dependence_max_tvd']:.4f} (>1e-3) | "
        f"{'✅' if g['G3_observed'] else '❌'} |",
        f"| G4 | 三 frame 指纹互异 | — | "
        f"{'✅' if g['G4_fingerprints_distinct'] else '❌'} |",
        "", f"**总判定：{'ALL PASS ✅' if g['ALL_PASS'] else 'FAIL ❌'}**",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n",
                                        encoding="utf-8")
    print("\n".join(lines))
    return 0 if g["ALL_PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
