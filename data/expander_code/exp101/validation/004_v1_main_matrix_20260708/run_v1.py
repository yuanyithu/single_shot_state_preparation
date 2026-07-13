# PRE_ALIGNMENT: historical v1 runner; it does not certify exp101.physics.v2.
"""G3.2 V1 主矩阵（regime-aware，plan §3 G3.2 + 2026-07-09 修订）。

每 regime 由其对应工具验证，各工具在其有效域内被检验。正确 instrument 见 plan
changelog 2026-07-09 条目。

引擎 × regime：
  - DIRECT（单链，q>0）：跑 q>0 网格；**逐任务自证遍历**（worst-u 冷端 logical
    接受率 ≥ ERGODIC_ACC）；仅遍历任务进入统计，冻结任务披露计数。
    判据（z-OR-绝对，处理 m_u≈±1 饱和）：
      · 偏差：非饱和比较（|m_exact|≤0.98）的 mean(Δm/σ)，|mean| ≤ 0.10
      · 尾部：discrepant 比例 ≤ 0.005，discrepant := |Δm|>max(4σ,0.02)
      · 能量：|ΔE| ≤ max(4σ_E, 0.01|E|) 的 relative-or-z
      · 类分布 TVD（k≤10 全档逆 Fourier）：max ≤ 0.05
  - PT（q>0 冷点）：验证 PT 解冻 sector。同 z-OR-绝对 + weight-TVD 判据。
  - TI（含 q=0 全谱、冷点足配置）：物理量判据
      · |q_top_ti − q_top_exact| ≤ max(5σ_qtop, 0.02)
      · weight-TVD ≤ 0.05
      · pairwise 档（K43）：m_u^pair 对精确 tanh(gap/2) 绝对差 ≤ 0.05
    raw ΔF-z 仅作诊断随 grid_tv flag 记录，不 hard-gate。
  - 对偶 sector：toric_m2 另跑 z_error 全流程。
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

EXP101_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXP101_ROOT))

from src.enumerate_exact import exact_reference  # noqa: E402
from src.fast_mcmc import build_fast_chain_data, run_fast_mcmc  # noqa: E402
from src.gf2 import gf2_matmul  # noqa: E402
from src.graphs import (  # noqa: E402
    cycle_parity_check_matrix,
    repetition_parity_check_matrix,
)
from src.hgp import hgp_from_H  # noqa: E402
from src.logicals import logical_pauli_operators  # noqa: E402
from src.model import DisorderRealization, assemble_sector_model, wire_ensemble  # noqa: E402
from src.observables import build_observable_frame, build_observable_set  # noqa: E402
from src.pt import PtConfig, run_parallel_tempering  # noqa: E402
from src.reference_mcmc import ReferenceMcmcConfig  # noqa: E402
from src.sector_ti import SectorTiConfig, run_sector_ti  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent

INSTANCES = {
    "toric_m2": lambda: cycle_parity_check_matrix(2),          # [[8,2]]
    "surface_m3": lambda: repetition_parity_check_matrix(3),   # [[13,1]]
    "toric_m3": lambda: cycle_parity_check_matrix(3),          # [[18,2]]
    "irregular_2x4": lambda: np.array(                         # [[20,4]]
        [[1, 1, 0, 1], [0, 1, 1, 0]], dtype=np.uint8),
    "K43": lambda: np.ones((3, 4), dtype=np.uint8),            # [[25,13]]
}

DIRECT_PQ = [(0.08, 0.05), (0.08, 0.30), (0.15, 0.05), (0.15, 0.15),
             (0.15, 0.08), (0.30, 0.05), (0.30, 0.30)]
PT_PQ = [(0.02, 0.01), (0.02, 0.05), (0.05, 0.03), (0.08, 0.02)]
TI_PQ = [(0.15, 0.0), (0.30, 0.0), (0.08, 0.05), (0.15, 0.15),
         (0.05, 0.0), (0.02, 0.01)]
TI_COLD = {(0.05, 0.0), (0.02, 0.01)}
NUM_DIS_DIRECT = 10
NUM_DIS_PT = 6
NUM_DIS_TI = 4

ERGODIC_ACC = 0.02
Z_FLOOR = 5e-3
SATURATION = 0.98

DIRECT_CFG = ReferenceMcmcConfig(num_burn_in_sweeps=800, num_measurements=6000,
                                 record_observable_trajectory=True)
PT_CFG = PtConfig(num_temperatures=6, q_hot=0.45, num_burn_in_rounds=400,
                  num_measurement_rounds=8000,
                  record_observable_trajectory=True)
TI_CFG = SectorTiConfig(num_kp_grid_points=25, num_burn_in_sweeps=150,
                        num_measurements=400, block_count=8, num_bootstrap=150)
TI_CFG_COLD = SectorTiConfig(num_kp_grid_points=49, num_burn_in_sweeps=400,
                             num_measurements=1200, block_count=10,
                             num_bootstrap=200)


def build_all(name):
    classical = INSTANCES[name]()
    H_Z, H_X = hgp_from_H(classical)
    logicals = logical_pauli_operators(H_X, H_Z)
    return H_Z, H_X, logicals


def planted(model, p, q, index, rng):
    n, m_c = model.num_qubits, model.num_checks
    eta = np.zeros(n, dtype=np.uint8)
    delta = np.zeros(m_c, dtype=np.uint8)
    if index == 0:
        pass
    elif index == 1:
        eta[int(rng.integers(0, n))] = 1
    elif index == 2 and q > 0:
        delta[int(rng.integers(0, m_c))] = 1
    else:
        eta = (rng.random(n) < p).astype(np.uint8)
        if q > 0:
            delta = (rng.random(m_c) < q).astype(np.uint8)
    observed = (gf2_matmul(model.H_check, eta[:, None])[:, 0] ^ delta).astype(
        np.uint8)
    return DisorderRealization(eta=eta, delta=delta, observed_syndrome=observed,
                              p=p, q=q, eta_weight=int(eta.sum()),
                              delta_weight=int(delta.sum()))


def exact_m_vector(exact, u_bitmasks, k):
    rel = exact["weights_relative"]
    out = np.empty(len(u_bitmasks))
    for i, u in enumerate(u_bitmasks):
        out[i] = sum(rel[t] * (1 - 2 * (int(u & t).bit_count() & 1))
                     for t in range(1 << k))
    return out


def block_stderr(traj):
    blocks = np.array_split(traj.astype(np.float64), 20, axis=0)
    means = np.array([b.mean(axis=0) for b in blocks])
    return means.std(axis=0, ddof=1) / np.sqrt(len(blocks))


def compare_m(m_mcmc, exact_m, stderr):
    """z-OR-绝对 逐比较；返回 (delta, z(非饱和), discrepant 布尔)。"""
    delta = m_mcmc - exact_m
    z = delta / np.maximum(stderr, Z_FLOOR)
    saturated = np.abs(exact_m) > SATURATION
    discrepant = (np.abs(delta) > np.maximum(4 * stderr, 0.02))
    return delta, z, saturated, discrepant


def tvd_full(m_mcmc, u_bitmasks, exact_rel, k):
    p_hat = np.zeros(1 << k)
    for t in range(1 << k):
        acc = 1.0
        for i, u in enumerate(u_bitmasks):
            acc += m_mcmc[i] * (1 - 2 * (int(u & t).bit_count() & 1))
        p_hat[t] = acc / (1 << k)
    p_hat = np.clip(p_hat, 0.0, None)
    p_hat /= p_hat.sum()
    return float(0.5 * np.abs(p_hat - exact_rel).sum())


def energy_check(traj_energy, exact, wiring):
    e_blocks = np.array([b.mean() for b in np.array_split(traj_energy, 20)])
    e_stderr = max(float(e_blocks.std(ddof=1) / np.sqrt(20)), Z_FLOOR)
    exact_e = (wiring.K_p * exact["mean_Wp"]
               + (0.0 if wiring.q_zero else wiring.K_q * exact["mean_Ws"]))
    delta = float(traj_energy.mean() - exact_e)
    tol = max(4 * e_stderr, 0.01 * abs(exact_e))
    return {"delta": delta, "z": delta / e_stderr, "ok": abs(delta) <= tol}


def run():
    started = time.perf_counter()
    records = []
    for name in INSTANCES:
        H_Z, H_X, logicals = build_all(name)
        sectors = ["x_error"] + (["z_error"] if name == "toric_m2" else [])
        for sector in sectors:
            model = assemble_sector_model(H_X, H_Z, logicals, sector=sector)
            frame = build_observable_frame(model)
            obs_kwargs = dict(num_random_u=64, u_rand_seed=777) \
                if model.k > 10 else {}
            obs_set = build_observable_set(frame, **obs_kwargs)
            chain_data = build_fast_chain_data(model, obs_set)

            # ---- DIRECT (q>0 ergodic-classified) ----
            for p, q in DIRECT_PQ:
                for d in range(NUM_DIS_DIRECT):
                    rng = np.random.default_rng(
                        abs(hash(("dir", name, sector, p, q, d))) % (2**31))
                    dis = planted(model, p, q, d, rng)
                    wiring = wire_ensemble(model, dis, "true_posterior", frame)
                    exact = exact_reference(model, frame, wiring)
                    res = run_fast_mcmc(model, frame, obs_set, wiring,
                                        DIRECT_CFG, seed=int(rng.integers(0, 2**60)),
                                        chain_data=chain_data)
                    per_u = np.asarray(res["acceptance"]["logical_per_u"],
                                       dtype=np.float64)
                    per_u = np.where(np.isfinite(per_u), per_u, 0.0)
                    ergodic = bool(per_u.min() >= ERGODIC_ACC) if per_u.size \
                        else True
                    exact_m = exact_m_vector(exact, obs_set.u_bitmasks, model.k)
                    stderr = block_stderr(res["observable_trajectory"])
                    delta, z, sat, disc = compare_m(res["m_u"], exact_m, stderr)
                    en = energy_check(res["energy_trace"], exact, wiring)
                    tvd = tvd_full(res["m_u"], obs_set.u_bitmasks,
                                   exact["weights_relative"], model.k) \
                        if obs_set.tier == "full" else None
                    records.append({
                        "kind": "direct", "instance": name, "sector": sector,
                        "p": p, "q": q, "disorder": d, "ergodic": ergodic,
                        "worst_u_acc": float(per_u.min()) if per_u.size else None,
                        "z_nonsat": z[~sat].tolist(),
                        "delta_nonsat": delta[~sat].tolist(),
                        "n_discrepant": int(disc.sum()),
                        "n_compare": int(z.size),
                        "energy_ok": en["ok"], "energy_z": en["z"],
                        "tvd": tvd,
                    })

            # ---- PT (q>0 cold) ----
            if sector == "x_error":
                for p, q in PT_PQ:
                    for d in range(NUM_DIS_PT):
                        rng = np.random.default_rng(
                            abs(hash(("pt", name, p, q, d))) % (2**31))
                        dis = planted(model, p, q, d, rng)
                        wiring = wire_ensemble(model, dis, "true_posterior",
                                               frame)
                        exact = exact_reference(model, frame, wiring)
                        pt = run_parallel_tempering(
                            model, frame, obs_set, wiring, PT_CFG,
                            seed=int(rng.integers(0, 2**60)))
                        exact_m = exact_m_vector(exact, obs_set.u_bitmasks,
                                                 model.k)
                        stderr = block_stderr(pt.observable_trajectory_cold)
                        delta, z, sat, disc = compare_m(pt.m_u_cold, exact_m,
                                                        stderr)
                        tvd = tvd_full(pt.m_u_cold, obs_set.u_bitmasks,
                                       exact["weights_relative"], model.k) \
                            if obs_set.tier == "full" else None
                        records.append({
                            "kind": "pt", "instance": name, "p": p, "q": q,
                            "disorder": d, "round_trips": pt.round_trips,
                            "min_swap_rate": float(pt.swap_rates().min()),
                            "z_nonsat": z[~sat].tolist(),
                            "delta_nonsat": delta[~sat].tolist(),
                            "n_discrepant": int(disc.sum()),
                            "n_compare": int(z.size), "tvd": tvd,
                        })

            # ---- TI (q=0 + spread, cold gets rich config) ----
            if sector == "x_error":
                for p, q in TI_PQ:
                    cfg = TI_CFG_COLD if (p, q) in TI_COLD else TI_CFG
                    for d in range(NUM_DIS_TI):
                        rng = np.random.default_rng(
                            abs(hash(("ti", name, p, q, d))) % (2**31))
                        dis = planted(model, p, q, d, rng)
                        wiring = wire_ensemble(model, dis, "true_posterior",
                                               frame)
                        exact = exact_reference(model, frame, wiring)
                        ti = run_sector_ti(model, frame, wiring, cfg,
                                           seed=int(rng.integers(0, 2**60)))
                        rec = {"kind": "ti", "instance": name, "p": p, "q": q,
                               "disorder": d, "tier": ti["tier"],
                               "flags": ti["flags"],
                               "delta_f_max_abs_z": float(np.max(np.abs(
                                   ti["delta_f"] / np.maximum(
                                       ti["delta_f_stderr"], 1e-3))))
                               if ti["tier"] == "full" else None}
                        if ti["tier"] == "full":
                            wt = float(0.5 * np.abs(
                                ti["weights_relative"]
                                - exact["weights_relative"]).sum())
                            qd = float(abs(ti["q_top"] - exact["q_top"]))
                            rec.update({
                                "weights_tvd": wt, "q_top_diff": qd,
                                "q_top_tol": float(max(5 * ti["q_top_stderr"],
                                                       0.02)),
                                "q_top_ok": bool(qd <= max(5 * ti["q_top_stderr"],
                                                           0.02)),
                                "tvd_ok": bool(wt <= 0.05),
                            })
                        else:
                            ell_ref = ti["ell_ref"]
                            wa = exact["weights_absolute"]
                            gaps = np.array([
                                -np.log(max(wa[ell_ref ^ (1 << u)], 1e-300)
                                        / max(wa[ell_ref], 1e-300))
                                for u in range(model.k)])
                            exact_pair = np.tanh(gaps / 2.0)
                            dev = float(np.max(np.abs(
                                ti["m_u_pairwise"] - exact_pair)))
                            rec.update({
                                "pairwise_m_max_dev": dev,
                                "pairwise_ok": bool(dev <= 0.05),
                            })
                        records.append(rec)
        print(f"[{time.perf_counter()-started:8.1f}s] done {name}", flush=True)

    return records, time.perf_counter() - started


def aggregate(records):
    direct = [r for r in records if r["kind"] == "direct"]
    direct_erg = [r for r in direct if r["ergodic"]]
    direct_frozen = [r for r in direct if not r["ergodic"]]
    pt = [r for r in records if r["kind"] == "pt"]
    ti = [r for r in records if r["kind"] == "ti"]

    def pool_z(rs):
        z = np.concatenate([np.asarray(r["z_nonsat"]) for r in rs
                            if r["z_nonsat"]]) if rs else np.zeros(0)
        nd = sum(r["n_discrepant"] for r in rs)
        nc = sum(r["n_compare"] for r in rs)
        return z, nd, nc

    dz, dnd, dnc = pool_z(direct_erg)
    ptz, ptnd, ptnc = pool_z(pt)
    d_tvd = [r["tvd"] for r in direct_erg if r["tvd"] is not None]
    pt_tvd = [r["tvd"] for r in pt if r["tvd"] is not None]
    d_energy_fail = sum(1 for r in direct_erg if not r["energy_ok"])
    ti_full = [r for r in ti if r["tier"] == "full"]
    ti_pair = [r for r in ti if r["tier"] == "pairwise"]

    gates = {
        "A_direct_mean_z": float(dz.mean()) if dz.size else 0.0,
        "A_direct_frac_discrepant": dnd / max(dnc, 1),
        "A_direct_tvd_max": float(np.max(d_tvd)) if d_tvd else 0.0,
        "A_pass": bool(abs(dz.mean() if dz.size else 0) <= 0.10
                       and dnd / max(dnc, 1) <= 0.005
                       and (not d_tvd or np.max(d_tvd) <= 0.05)),
        "B_direct_energy_fail": d_energy_fail,
        "B_pass": bool(d_energy_fail == 0),
        "PT_mean_z": float(ptz.mean()) if ptz.size else 0.0,
        "PT_frac_discrepant": ptnd / max(ptnc, 1),
        "PT_tvd_max": float(np.max(pt_tvd)) if pt_tvd else 0.0,
        "PT_all_round_trips_positive": bool(all(r["round_trips"] > 0
                                                for r in pt)),
        "PT_pass": bool(abs(ptz.mean() if ptz.size else 0) <= 0.10
                        and ptnd / max(ptnc, 1) <= 0.005
                        and (not pt_tvd or np.max(pt_tvd) <= 0.05)),
        "TI_full_qtop_fail": sum(1 for r in ti_full if not r["q_top_ok"]),
        "TI_full_tvd_fail": sum(1 for r in ti_full if not r["tvd_ok"]),
        "TI_full_tvd_max": float(np.max([r["weights_tvd"] for r in ti_full]))
        if ti_full else 0.0,
        "TI_pair_fail": sum(1 for r in ti_pair if not r["pairwise_ok"]),
        "TI_pair_max_dev": float(np.max([r["pairwise_m_max_dev"]
                                         for r in ti_pair])) if ti_pair else 0.0,
        "TI_pass": bool(all(r["q_top_ok"] and r["tvd_ok"] for r in ti_full)
                        and all(r["pairwise_ok"] for r in ti_pair)),
        "direct_ergodic": len(direct_erg),
        "direct_frozen_disclosed": len(direct_frozen),
        "num_direct_z": int(dnc),
        "num_pt_z": int(ptnc),
    }
    gates["ALL_PASS"] = bool(gates["A_pass"] and gates["B_pass"]
                             and gates["PT_pass"] and gates["TI_pass"])
    return gates


def main():
    records, wall = run()
    gates = aggregate(records)
    gates["wall_time_seconds"] = wall
    with (OUT_DIR / "results.json").open("w", encoding="utf-8") as fh:
        json.dump({"records": records, "gates": gates}, fh, indent=1,
                  ensure_ascii=False)
    g = gates
    lines = [
        "# G3.2 V1 主矩阵结果（regime-aware）", "",
        f"墙钟 {wall:.0f}s。direct 遍历 {g['direct_ergodic']} / 冻结披露 "
        f"{g['direct_frozen_disclosed']}（冻结点由 TI 覆盖）。", "",
        "| regime | 指标 | 值 | 阈值 | 结果 |",
        "|---|---|---|---|---|",
        f"| A direct | mean z / discrepant / tvd | {g['A_direct_mean_z']:.3f} / "
        f"{g['A_direct_frac_discrepant']:.4f} / {g['A_direct_tvd_max']:.3f} | "
        f"≤0.10 / ≤0.005 / ≤0.05 | {'✅' if g['A_pass'] else '❌'} |",
        f"| B direct 能量 | 失败任务数 | {g['B_direct_energy_fail']} | 0 | "
        f"{'✅' if g['B_pass'] else '❌'} |",
        f"| PT 冷点 | mean z / discrepant / tvd | {g['PT_mean_z']:.3f} / "
        f"{g['PT_frac_discrepant']:.4f} / {g['PT_tvd_max']:.3f} | "
        f"≤0.10 / ≤0.005 / ≤0.05 | {'✅' if g['PT_pass'] else '❌'} |",
        f"| PT 传输 | 全部 round_trips>0 | {g['PT_all_round_trips_positive']} "
        f"| True | {'✅' if g['PT_all_round_trips_positive'] else '❌'} |",
        f"| TI full | q_top 失败 / tvd 失败 / tvd_max | "
        f"{g['TI_full_qtop_fail']} / {g['TI_full_tvd_fail']} / "
        f"{g['TI_full_tvd_max']:.3f} | 0 / 0 / ≤0.05 | "
        f"{'✅' if g['TI_full_qtop_fail']==0 and g['TI_full_tvd_fail']==0 else '❌'} |",
        f"| TI pairwise(K43) | 失败数 / max dev | {g['TI_pair_fail']} / "
        f"{g['TI_pair_max_dev']:.3f} | 0 / ≤0.05 | "
        f"{'✅' if g['TI_pair_fail']==0 else '❌'} |",
        "",
        f"**总判定：{'ALL PASS ✅' if g['ALL_PASS'] else 'FAIL ❌'}**",
    ]
    lines[1:1] = [
        "",
        "> **PRE_ALIGNMENT（自动生成保护）：** 本页及 raw 数据只记录 v1 历史内部一致性；",
        "> 重新运行本 runner 不认证 `exp101.physics.v2`，也不得覆盖 014 结论。",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n",
                                        encoding="utf-8")
    print("\n".join(lines))
    return 0 if g["ALL_PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
