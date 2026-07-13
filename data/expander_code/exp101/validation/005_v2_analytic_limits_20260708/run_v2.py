# PRE_ALIGNMENT: historical v1 runner; it does not certify exp101.physics.v2.
"""G3.4 V2 解析极限（notes/01 §6 的全部落地；q=0.5 闭式覆盖生产规模 m=2,4,6）。

检查项（全部对 fast 引擎；m_u 的 z 用分块 stderr，floor 5e-3）：
  V2a p=0.5 零化：任意 (q, disorder) 全部 m_u = 0；L-move 接受率 ≡ 1。
      实例：toric_m3、expander m=2。
  V2b q=0.5 闭式：m_u = (−1)^{⟨u,ℓ_ref⟩}(1−2p)^{|w_u|}；⟨W_p⟩ = np；
      ⟨W_s⟩ = Σ_c [σ_c ? (1+(1−2p)^{w_c})/2 : (1−(1−2p)^{w_c})/2]。
      实例：expander m=2 (p=0.05)、m=4 (p=0.01)、m=6 (p=0.002)——
      **无枚举依赖，直接在 n=900 生产规模验证观测量+采样器**。
  V2c q→0⁺ vs q=0：toric_m3 同 disorder（δ=0），q=1e−3 引擎与 q=0 引擎
      各自对枚举 z ≤ 5 且互差一致。
  V2d p→0⁺：toric_m3，p=1e−3、δ=0：枚举 q_top > 0.99 且引擎与枚举一致。
  V2e disorder=0：η=0,δ=0 时两系综 wiring 完全重合（σ_arg/ℓ_ref 相同），
      引擎结果对枚举一致（快测路径）。
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
    draw_disorder,
    wire_ensemble,
)
from src.observables import build_observable_frame, build_observable_set  # noqa: E402
from src.reference_mcmc import ReferenceMcmcConfig  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
Z_FLOOR = 5e-3


def build_setup(classical, num_random_u=48, seed_u=99):
    H_Z, H_X = hgp_from_H(classical)
    logicals = logical_pauli_operators(H_X, H_Z)
    model = assemble_sector_model(H_X, H_Z, logicals, sector="x_error")
    frame = build_observable_frame(model)
    kwargs = {}
    if model.k > 10:
        kwargs = dict(num_random_u=num_random_u, u_rand_seed=seed_u)
    obs_set = build_observable_set(frame, **kwargs)
    return model, frame, obs_set


def expander_setup(m, seed=12345, **kw):
    graph = random_biregular_graph_from_m(m, 3, 4, seed)
    return build_setup(classical_parity_check_matrix(graph), **kw)


def stderr_of(traj):
    blocks = np.array_split(traj.astype(np.float64), 20, axis=0)
    means = np.array([b.mean(axis=0) for b in blocks])
    return np.maximum(means.std(axis=0, ddof=1) / np.sqrt(20), Z_FLOOR)


def run_engine(model, frame, obs_set, wiring, seed, meas=6000, burn=600):
    config = ReferenceMcmcConfig(
        num_burn_in_sweeps=burn, num_measurements=meas,
        record_observable_trajectory=True,
    )
    chain_data = build_fast_chain_data(model, obs_set)
    return run_fast_mcmc(model, frame, obs_set, wiring, config, seed=seed,
                         chain_data=chain_data)


def check_v2a(records):
    for tag, setup, p_q_seed in [
        ("toric_m3", build_setup(cycle_parity_check_matrix(3)), (0.5, 0.1, 1)),
        ("expander_m2", expander_setup(2), (0.5, 0.2, 2)),
    ]:
        model, frame, obs_set = setup
        p, q, seed = p_q_seed
        wiring = wire_ensemble(
            model, draw_disorder(model, p, q, np.random.default_rng(seed)),
            "true_posterior", frame)
        assert wiring.K_p == 0.0
        result = run_engine(model, frame, obs_set, wiring, seed=seed + 100)
        z = result["m_u"] / stderr_of(result["observable_trajectory"])
        rates = result["acceptance"]["logical_per_u"]
        records.append({
            "check": "V2a", "instance": tag,
            "max_abs_z": float(np.max(np.abs(z))),
            "logical_acceptance_all_one": bool(np.all(rates == 1.0)),
            "pass": bool(np.max(np.abs(z)) < 5.0 and np.all(rates == 1.0)),
        })


def check_v2b(records):
    for m, p, seed in [(2, 0.05, 3), (4, 0.01, 4), (6, 0.002, 5)]:
        model, frame, obs_set = expander_setup(m)
        wiring = wire_ensemble(
            model, draw_disorder(model, p, 0.5, np.random.default_rng(seed)),
            "true_posterior", frame)
        assert wiring.K_q == 0.0
        result = run_engine(model, frame, obs_set, wiring, seed=seed + 200,
                            meas=6000, burn=400)
        # 闭式：m_u = (−1)^{⟨u,ℓ_ref⟩}(1−2p)^{|w_u|}
        ell_ref_bits = sum(1 << b for b, v in
                           enumerate(wiring.reference_label) if v)
        expected = np.zeros(obs_set.num_u)
        for i, u in enumerate(obs_set.u_bitmasks):
            w_row = obs_set.W_rows[i]
            sign = 1.0 - 2.0 * (int(u & ell_ref_bits).bit_count() & 1)
            expected[i] = sign * (1 - 2 * p) ** int(w_row.sum())
        z = (result["m_u"] - expected) / stderr_of(
            result["observable_trajectory"])
        # 能量闭式
        energy = result["energy_trace"]
        e_blocks = np.array([b.mean() for b in np.array_split(energy, 20)])
        e_stderr = max(float(e_blocks.std(ddof=1) / np.sqrt(20)), Z_FLOOR)
        check_weights = model.H_check.sum(axis=1)
        p_odd = (1 - (1 - 2 * p) ** check_weights.astype(float)) / 2
        sigma = wiring.sigma_arg.astype(bool)
        mean_ws = float(np.where(sigma, 1 - p_odd, p_odd).sum())
        exact_energy = wiring.K_p * model.num_qubits * p + 0.0 * mean_ws
        energy_z = float((energy.mean() - exact_energy) / e_stderr)
        records.append({
            "check": "V2b", "instance": f"expander_m{m}", "p": p,
            "n": model.num_qubits, "k": model.k,
            "max_abs_z": float(np.max(np.abs(z))),
            "energy_z": energy_z,
            "pass": bool(np.max(np.abs(z)) < 5.0 and abs(energy_z) < 5.0),
        })


def check_v2c(records):
    model, frame, obs_set = build_setup(cycle_parity_check_matrix(3))
    rng = np.random.default_rng(6)
    eta = (rng.random(model.num_qubits) < 0.12).astype(np.uint8)
    observed = gf2_matmul(model.H_check, eta[:, None])[:, 0].astype(np.uint8)
    results = {}
    for tag, q in [("q_zero", 0.0), ("q_tiny", 1e-3)]:
        disorder = DisorderRealization(
            eta=eta, delta=np.zeros(model.num_checks, dtype=np.uint8),
            observed_syndrome=observed, p=0.12, q=q,
            eta_weight=int(eta.sum()), delta_weight=0,
        )
        wiring = wire_ensemble(model, disorder, "true_posterior", frame)
        exact = exact_reference(model, frame, wiring)
        result = run_engine(model, frame, obs_set, wiring, seed=61)
        exact_m = np.array([
            sum(exact["weights_relative"][t] * (1 - 2 * (int(u & t).bit_count() & 1))
                for t in range(1 << model.k))
            for u in obs_set.u_bitmasks
        ])
        z = (result["m_u"] - exact_m) / stderr_of(
            result["observable_trajectory"])
        results[tag] = {"m_u": result["m_u"], "exact": exact_m,
                        "max_abs_z": float(np.max(np.abs(z)))}
    diff = float(np.max(np.abs(
        results["q_zero"]["exact"] - results["q_tiny"]["exact"])))
    records.append({
        "check": "V2c",
        "engine_z_qzero": results["q_zero"]["max_abs_z"],
        "engine_z_qtiny": results["q_tiny"]["max_abs_z"],
        "exact_continuity_diff": diff,
        "pass": bool(results["q_zero"]["max_abs_z"] < 5.0
                     and results["q_tiny"]["max_abs_z"] < 5.0
                     and diff < 0.02),
    })


def check_v2d(records):
    # p→0 AND q→0（低错误率 Bayes-optimal 恢复极限）：后验集中于真错误 η 所在
    # 逻辑类 ⇒ q_top→1。注意：固定中等 q 时该极限不成立（留 c=0 不解释
    # 比放置 η 更省能量），必须同时 q→0——见 notes/01 §6 与 V2d 修订说明。
    model, frame, obs_set = build_setup(cycle_parity_check_matrix(3))
    rng = np.random.default_rng(7)
    eta = np.zeros(model.num_qubits, dtype=np.uint8)
    eta[int(rng.integers(0, model.num_qubits))] = 1
    observed = gf2_matmul(model.H_check, eta[:, None])[:, 0].astype(np.uint8)
    disorder = DisorderRealization(
        eta=eta, delta=np.zeros(model.num_checks, dtype=np.uint8),
        observed_syndrome=observed, p=1e-3, q=1e-3,
        eta_weight=1, delta_weight=0,
    )
    wiring = wire_ensemble(model, disorder, "true_posterior", frame)
    exact = exact_reference(model, frame, wiring)
    result = run_engine(model, frame, obs_set, wiring, seed=71)
    exact_m = np.array([
        sum(exact["weights_relative"][t] * (1 - 2 * (int(u & t).bit_count() & 1))
            for t in range(1 << model.k))
        for u in obs_set.u_bitmasks
    ])
    z = (result["m_u"] - exact_m) / stderr_of(result["observable_trajectory"])
    records.append({
        "check": "V2d", "exact_q_top": exact["q_top"],
        "max_abs_z": float(np.max(np.abs(z))),
        "pass": bool(exact["q_top"] > 0.99 and np.max(np.abs(z)) < 5.0),
    })


def check_v2e(records):
    model, frame, obs_set = build_setup(cycle_parity_check_matrix(3))
    zero = DisorderRealization(
        eta=np.zeros(model.num_qubits, dtype=np.uint8),
        delta=np.zeros(model.num_checks, dtype=np.uint8),
        observed_syndrome=np.zeros(model.num_checks, dtype=np.uint8),
        p=0.15, q=0.10, eta_weight=0, delta_weight=0,
    )
    w_true = wire_ensemble(model, zero, "true_posterior", frame)
    w_repo = wire_ensemble(model, zero, "repo_compat", frame)
    same_wiring = bool(
        np.array_equal(w_true.sigma_arg, w_repo.sigma_arg)
        and np.array_equal(w_true.reference_label, w_repo.reference_label)
    )
    exact = exact_reference(model, frame, w_true)
    result = run_engine(model, frame, obs_set, w_true, seed=81)
    exact_m = np.array([
        sum(exact["weights_relative"][t] * (1 - 2 * (int(u & t).bit_count() & 1))
            for t in range(1 << model.k))
        for u in obs_set.u_bitmasks
    ])
    z = (result["m_u"] - exact_m) / stderr_of(result["observable_trajectory"])
    records.append({
        "check": "V2e", "ensembles_identical_at_zero_disorder": same_wiring,
        "max_abs_z": float(np.max(np.abs(z))),
        "pass": bool(same_wiring and np.max(np.abs(z)) < 5.0),
    })


def main():
    started = time.perf_counter()
    records = []
    for fn in (check_v2a, check_v2b, check_v2c, check_v2d, check_v2e):
        fn(records)
        print(f"[{time.perf_counter()-started:7.1f}s] {fn.__name__} done",
              flush=True)
    all_pass = all(r["pass"] for r in records)
    payload = {"records": records, "all_pass": all_pass,
               "wall_time_seconds": time.perf_counter() - started}
    with (OUT_DIR / "results.json").open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, ensure_ascii=False)
    lines = ["# G3.4 V2 解析极限结果", "",
             "| check | 实例/参数 | 关键指标 | 结果 |", "|---|---|---|---|"]
    for r in records:
        key_metrics = ", ".join(
            f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
            for k, v in r.items() if k not in ("check", "pass", "instance")
        )
        lines.append(
            f"| {r['check']} | {r.get('instance','—')} | {key_metrics[:150]} "
            f"| {'✅' if r['pass'] else '❌'} |"
        )
    lines += ["", f"**总判定：{'ALL PASS ✅' if all_pass else 'FAIL ❌'}**",
              f"墙钟 {payload['wall_time_seconds']:.0f}s"]
    lines[1:1] = [
        "",
        "> **PRE_ALIGNMENT（自动生成保护）：** 标题中的 V2 早于 `exp101.physics.v2`；",
        "> 重新运行本 runner 只生成历史证据，不得覆盖当前 014 认证结论。",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n",
                                        encoding="utf-8")
    print("\n".join(lines))
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
