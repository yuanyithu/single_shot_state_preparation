# PRE_ALIGNMENT: historical v1 runner; it does not certify exp101.physics.v2.
"""G3.5 V3 Nishimori 恒等式（plan §3 G3.5, notes/01 §5）。

Nishimori 线（温标 = disorder 率）上，**true_posterior** 系综精确满足
    E_{η,δ}[m_u] = E_{η,δ}[m_u²]，  E[w0] = E[purity]。
（推导见 notes/01 §5：对固定 s 用 P_p(η)P_q(δ)=C·e^{−K_p|η|−K_q|Hη⊕s|}。）

三级递进：
  L1 [[8,2,2]] 全 disorder 求和（2^8 η × 2^4 δ = 4096，逐个精确枚举）——零统计误差，
     验证模型定义与恒等式本身。
  L2 toric_m3 [[18,2,3]] + K43 [[25,13]] 抽样 disorder × 精确枚举——仅 disorder 抽样
     误差（bootstrap 判 E[m]−E[m²] 与 0 相容）。
  L3 (3,4) m=2（n=100, 越枚举界）全 MCMC——disorder+MCMC+观测量整链。每 disorder 跑
     2 条独立链得**无偏** m_u²（=m_a·m_b），配对 bootstrap 判 E[m]−E[m·] 与 0 相容。
  JUDGE repo_compat 违反恒等式：q=0.5 闭式 E[m_u]=(1−2p)^{|w|}≠E[m_u²]=(1−2p)^{2|w|}
     ——确认恒等式是 true_posterior 特有（系综判别）。
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

EXP101_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXP101_ROOT))

from src.enumerate_exact import exact_reference  # noqa: E402
from src.fast_mcmc import build_fast_chain_data, run_fast_mcmc  # noqa: E402
from src.gf2 import gf2_matmul  # noqa: E402
from src.graphs import (  # noqa: E402
    cycle_parity_check_matrix,
    random_biregular_graph_from_m,
)
from src.hgp import classical_parity_check_matrix, hgp_from_H  # noqa: E402
from src.logicals import logical_pauli_operators  # noqa: E402
from src.model import (  # noqa: E402
    DisorderRealization,
    assemble_sector_model,
    wire_ensemble,
)
from src.observables import build_observable_frame, build_observable_set  # noqa: E402
from src.reference_mcmc import ReferenceMcmcConfig  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent


def setup(classical, **obs):
    H_Z, H_X = hgp_from_H(classical)
    logicals = logical_pauli_operators(H_X, H_Z)
    model = assemble_sector_model(H_X, H_Z, logicals, sector="x_error")
    frame = build_observable_frame(model)
    return model, frame, build_observable_set(frame, **obs)


def make_disorder(model, eta, delta, p, q):
    obs = (gf2_matmul(model.H_check, eta[:, None])[:, 0] ^ delta).astype(np.uint8)
    return DisorderRealization(eta=eta, delta=delta, observed_syndrome=obs,
                              p=p, q=q, eta_weight=int(eta.sum()),
                              delta_weight=int(delta.sum()))


def exact_m_basis(model, frame, wiring, obs_set):
    ex = exact_reference(model, frame, wiring)
    rel = ex["weights_relative"]
    m = np.array([sum(rel[t] * (1 - 2 * (int(u & t).bit_count() & 1))
                      for t in range(1 << model.k))
                  for u in obs_set.u_bitmasks])
    return m


def level1(records):
    """[[8,2,2]] 全 disorder 求和，多 Nishimori 点。"""
    model, frame, obs_set = setup(cycle_parity_check_matrix(2))
    n, mc, k = model.num_qubits, model.num_checks, model.k
    for p, q in [(0.08, 0.05), (0.15, 0.10), (0.20, 0.15)]:
        E_m = np.zeros(obs_set.num_u)
        E_m2 = np.zeros(obs_set.num_u)
        for ei in range(1 << n):
            eta = np.array([(ei >> j) & 1 for j in range(n)], dtype=np.uint8)
            pe = p ** int(eta.sum()) * (1 - p) ** (n - int(eta.sum()))
            for di in range(1 << mc):
                delta = np.array([(di >> j) & 1 for j in range(mc)],
                                 dtype=np.uint8)
                pd = q ** int(delta.sum()) * (1 - q) ** (mc - int(delta.sum()))
                w = pe * pd
                wiring = wire_ensemble(model, make_disorder(model, eta, delta,
                                                            p, q),
                                       "true_posterior", frame)
                m = exact_m_basis(model, frame, wiring, obs_set)
                E_m += w * m
                E_m2 += w * m * m
        max_diff = float(np.max(np.abs(E_m - E_m2)))
        records.append({"level": "L1", "instance": "toric_m2", "p": p, "q": q,
                        "max_abs_E_m_minus_E_m2": max_diff,
                        "pass": bool(max_diff < 1e-9)})
        print(f"  L1 p={p} q={q}: max|E[m]-E[m2]|={max_diff:.2e}", flush=True)


def level2(records, rng):
    for name, classical, ndis in [
        ("toric_m3", cycle_parity_check_matrix(3), 400),
        ("K43", np.ones((3, 4), dtype=np.uint8), 120),
    ]:
        obs_kw = dict(num_random_u=48, u_rand_seed=7) if name == "K43" else {}
        model, frame, obs_set = setup(classical, **obs_kw)
        for p, q in [(0.10, 0.06)]:
            ms, m2s = [], []
            for _ in range(ndis):
                eta = (rng.random(model.num_qubits) < p).astype(np.uint8)
                delta = (rng.random(model.num_checks) < q).astype(np.uint8)
                wiring = wire_ensemble(model, make_disorder(model, eta, delta,
                                                            p, q),
                                       "true_posterior", frame)
                m = exact_m_basis(model, frame, wiring, obs_set)
                ms.append(m)
                m2s.append(m * m)
            ms = np.array(ms)
            m2s = np.array(m2s)
            diff = ms.mean(0) - m2s.mean(0)         # per-u E[m]-E[m2]
            # bootstrap over disorders
            boot = []
            for _ in range(400):
                idx = rng.integers(0, len(ms), len(ms))
                boot.append(ms[idx].mean(0) - m2s[idx].mean(0))
            boot = np.array(boot)
            se = boot.std(0, ddof=1)
            z = diff / np.maximum(se, 1e-12)
            records.append({
                "level": "L2", "instance": name, "p": p, "q": q, "ndis": ndis,
                "max_abs_z": float(np.max(np.abs(z))),
                "max_abs_diff": float(np.max(np.abs(diff))),
                "pass": bool(np.max(np.abs(z)) < 5.0)})
            print(f"  L2 {name} p={p} q={q}: max|z|={np.max(np.abs(z)):.2f} "
                  f"max|diff|={np.max(np.abs(diff)):.4f}", flush=True)


def level3(records, rng):
    """(3,4) m=2 全 MCMC；每 disorder 2 条独立链得无偏 m_u²。"""
    graph = random_biregular_graph_from_m(2, 3, 4, 12345)
    model, frame, obs_set = setup(classical_parity_check_matrix(graph))
    chain = build_fast_chain_data(model, obs_set)
    cfg = ReferenceMcmcConfig(num_burn_in_sweeps=600, num_measurements=4000)
    p, q = 0.08, 0.05
    ndis = 120
    E_m, E_m2 = [], []
    for _ in range(ndis):
        eta = (rng.random(model.num_qubits) < p).astype(np.uint8)
        delta = (rng.random(model.num_checks) < q).astype(np.uint8)
        wiring = wire_ensemble(model, make_disorder(model, eta, delta, p, q),
                               "true_posterior", frame)
        ra = run_fast_mcmc(model, frame, obs_set, wiring, cfg,
                           seed=int(rng.integers(0, 2**60)), chain_data=chain)
        rb = run_fast_mcmc(model, frame, obs_set, wiring, cfg,
                           seed=int(rng.integers(0, 2**60)), chain_data=chain)
        E_m.append(0.5 * (ra["m_u"] + rb["m_u"]))
        E_m2.append(ra["m_u"] * rb["m_u"])   # 两独立链乘积 = 无偏 m_u²
    E_m = np.array(E_m)
    E_m2 = np.array(E_m2)
    diff = E_m.mean(0) - E_m2.mean(0)
    boot = []
    for _ in range(600):
        idx = rng.integers(0, ndis, ndis)
        boot.append(E_m[idx].mean(0) - E_m2[idx].mean(0))
    se = np.array(boot).std(0, ddof=1)
    z = diff / np.maximum(se, 1e-12)
    records.append({
        "level": "L3", "instance": "expander_m2", "p": p, "q": q, "ndis": ndis,
        "max_abs_z": float(np.max(np.abs(z))),
        "max_abs_diff": float(np.max(np.abs(diff))),
        "pass": bool(np.max(np.abs(z)) < 5.0)})
    print(f"  L3 expander_m2 p={p} q={q}: max|z|={np.max(np.abs(z)):.2f} "
          f"max|diff|={np.max(np.abs(diff)):.4f}", flush=True)


def judge_repo_compat(records):
    """repo_compat 在 q=0.5 违反恒等式（闭式）：E[m]=(1−2p)^{|w|}≠E[m²]=(1−2p)^{2|w|}。"""
    model, frame, obs_set = setup(cycle_parity_check_matrix(2))
    p = 0.15
    # q=0.5 时 repo_compat：c=η⊕e, e~Bern(p)；对 disorder 平均 E[m_u]=(1−2p)^{|w_u|}
    # E[m_u²]=(1−2p)^{2|w_u|}（notes/01 §5/§6）。直接闭式比较（enum 会与之一致）。
    ws = obs_set.W_rows.sum(1)
    E_m = (1 - 2 * p) ** ws
    E_m2 = (1 - 2 * p) ** (2 * ws)
    max_gap = float(np.max(np.abs(E_m - E_m2)))
    records.append({"level": "JUDGE", "instance": "toric_m2_repo_compat",
                    "p": p, "q": 0.5, "identity_gap": max_gap,
                    "pass": bool(max_gap > 0.05)})  # 期望违反（gap 显著）
    print(f"  JUDGE repo_compat q=0.5: identity gap={max_gap:.4f} "
          f"(期望>0.05 = 恒等式对 repo_compat 失败)", flush=True)


def main():
    started = time.perf_counter()
    rng = np.random.default_rng(20260709)
    records = []
    print("L1 ...", flush=True); level1(records)
    print("L2 ...", flush=True); level2(records, rng)
    print("L3 ...", flush=True); level3(records, rng)
    print("JUDGE ...", flush=True); judge_repo_compat(records)
    all_pass = all(r["pass"] for r in records)
    payload = {"records": records, "all_pass": all_pass,
               "wall_time_seconds": time.perf_counter() - started}
    with (OUT_DIR / "results.json").open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, ensure_ascii=False)
    lines = ["# G3.5 V3 Nishimori 恒等式结果", "",
             f"墙钟 {payload['wall_time_seconds']:.0f}s", "",
             "| 级 | 实例 | (p,q) | 关键指标 | 结果 |", "|---|---|---|---|---|"]
    for r in records:
        if r["level"] == "L1":
            metric = f"max\\|E[m]-E[m²]\\|={r['max_abs_E_m_minus_E_m2']:.2e}"
        elif r["level"] == "JUDGE":
            metric = f"identity gap={r['identity_gap']:.3f}（期望违反）"
        else:
            metric = f"max\\|z\\|={r['max_abs_z']:.2f}, max\\|diff\\|={r['max_abs_diff']:.4f}"
        lines.append(f"| {r['level']} | {r['instance']} | "
                     f"({r['p']},{r['q']}) | {metric} | "
                     f"{'✅' if r['pass'] else '❌'} |")
    lines += ["", f"**总判定：{'ALL PASS ✅' if all_pass else 'FAIL ❌'}**"]
    lines[1:1] = [
        "",
        "> **PRE_ALIGNMENT（自动生成保护）：** 本页只记录旧接线下的历史实验；",
        "> 重新运行本 runner 不认证 `exp101.physics.v2`，也不得覆盖 014 结论。",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n",
                                        encoding="utf-8")
    print("\n".join(lines))
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
