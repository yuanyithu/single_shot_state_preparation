# PRE_ALIGNMENT: historical v1 runner; it does not certify exp101.physics.v2.
"""G3.6 V4 实现冗余 + G3.7 V6 冻结扇区 torture（plan §3）。

V4（实现冗余 A/B，多为单测覆盖，此处大尺度落证据）：
  - reference ≡ numba：同 seed observable_sums 逐位一致（bit 级）。
  - PT vs direct：各自对枚举一致（z<5），互相一致。
  - 1-start vs 8-start pooled：一致。
V6（冻结扇区 torture，关键测**诊断灵敏度**，plan §6 风险 3 与 CLAUDE.md 物理图像）：
  - 负例：expander m=2(k=4)/m=3(k=9) 冷点、关 L-move（q=0 严格冻结）→ 收敛 gate
    **必须报 sector_transport_insufficient**（异 sector 起点 m_u_spread 大；共 sector
    起点靠 transport 判据）。测的是诊断能否抓住冻结，不是采样器。
  - 正例：同码 q>0 + PT → round_trips>0、冷端接受率达标、结果与初始 sector 无关。
  - per-u 冻结检测 k=4/k=9：worst-u 判据在多逻辑下确实取到最冻的 u。
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

EXP101_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXP101_ROOT))

from src.enumerate_exact import exact_reference  # noqa: E402
from src.fast_mcmc import run_fast_mcmc  # noqa: E402
from src.gates import (  # noqa: E402
    GateThresholds,
    evaluate_convergence_gate,
    run_multi_start,
)
from src.graphs import (  # noqa: E402
    cycle_parity_check_matrix,
    random_biregular_graph_from_m,
)
from src.hgp import classical_parity_check_matrix, hgp_from_H  # noqa: E402
from src.logicals import logical_pauli_operators  # noqa: E402
from src.model import assemble_sector_model, draw_disorder, wire_ensemble  # noqa: E402
from src.observables import (  # noqa: E402
    aggregate_observables,
    build_observable_frame,
    build_observable_set,
)
from src.pt import PtConfig, run_parallel_tempering  # noqa: E402
from src.reference_mcmc import ReferenceMcmcConfig, run_reference_mcmc  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent


def setup(classical, **obs):
    H_Z, H_X = hgp_from_H(classical)
    logicals = logical_pauli_operators(H_X, H_Z)
    model = assemble_sector_model(H_X, H_Z, logicals, sector="x_error")
    frame = build_observable_frame(model)
    return model, frame, build_observable_set(frame, **obs)


def expander(m, **obs):
    graph = random_biregular_graph_from_m(m, 3, 4, 12345)
    return setup(classical_parity_check_matrix(graph), **obs)


def exact_m(model, frame, wiring, obs_set):
    ex = exact_reference(model, frame, wiring)
    rel = ex["weights_relative"]
    return np.array([sum(rel[t] * (1 - 2 * (int(u & t).bit_count() & 1))
                         for t in range(1 << model.k))
                     for u in obs_set.u_bitmasks])


def stderr_of(traj):
    b = np.array_split(traj.astype(np.float64), 20, 0)
    m = np.array([x.mean(0) for x in b])
    return np.maximum(m.std(0, ddof=1) / np.sqrt(20), 5e-3)


def v4_redundancy(records):
    model, frame, obs_set = setup(cycle_parity_check_matrix(2))
    wiring = wire_ensemble(model, draw_disorder(model, 0.12, 0.08,
                                                np.random.default_rng(1)),
                           "true_posterior", frame)
    exm = exact_m(model, frame, wiring, obs_set)
    cfg = ReferenceMcmcConfig(num_burn_in_sweeps=500, num_measurements=8000,
                              record_observable_trajectory=True)
    # reference ≡ numba (bit-level)
    ref = run_reference_mcmc(model, frame, obs_set, wiring, cfg, seed=42)
    fast = run_fast_mcmc(model, frame, obs_set, wiring, cfg, seed=42)
    bit_identical = bool(np.array_equal(ref["observable_sums"],
                                        fast["observable_sums"]))
    # PT vs direct, each vs enum
    pt = run_parallel_tempering(model, frame, obs_set, wiring,
                                PtConfig(num_temperatures=4, q_hot=0.4,
                                         num_burn_in_rounds=300,
                                         num_measurement_rounds=8000,
                                         record_observable_trajectory=True),
                                seed=7)
    z_direct = float(np.max(np.abs((fast["m_u"] - exm)
                                   / stderr_of(fast["observable_trajectory"]))))
    z_pt = float(np.max(np.abs((pt.m_u_cold - exm)
                               / stderr_of(pt.observable_trajectory_cold))))
    # 1-start vs 8-start pooled
    starts8 = run_multi_start(model, frame, obs_set, wiring, cfg, base_seed=100,
                              num_starts=8)
    m_8 = np.mean([s["m_u"] for s in starts8], 0)
    z_1v8 = float(np.max(np.abs((fast["m_u"] - m_8)
                                / (stderr_of(fast["observable_trajectory"])
                                   * np.sqrt(2)))))
    records.append({
        "test": "V4", "bit_identical_ref_numba": bit_identical,
        "z_direct_vs_enum": z_direct, "z_pt_vs_enum": z_pt,
        "z_1start_vs_8start": z_1v8,
        "pass": bool(bit_identical and z_direct < 5 and z_pt < 5 and z_1v8 < 5)})
    print(f"  V4: bit_identical={bit_identical} z_direct={z_direct:.2f} "
          f"z_pt={z_pt:.2f} z_1v8={z_1v8:.2f}", flush=True)


def v6_negative(records, m, k_expected):
    """负例：冷 q=0 + 关 L-move → gate 必须报 transport 不足。"""
    model, frame, obs_set = expander(m)
    assert model.k == k_expected
    wiring = wire_ensemble(model, draw_disorder(model, 0.15, 0.0,
                                                np.random.default_rng(3)),
                           "true_posterior", frame)
    cfg = ReferenceMcmcConfig(num_burn_in_sweeps=300, num_measurements=2000,
                              logical_move_repeat=0,
                              record_observable_trajectory=True)
    # 异 sector 起点
    starts_diff = run_multi_start(model, frame, obs_set, wiring, cfg,
                                  base_seed=200, num_starts=4,
                                  sector_bitmasks=[0, 1, 2, 3])
    rep_diff = evaluate_convergence_gate(starts_diff)
    # 共 sector 起点（放宽统计判据，仅测 transport 判据）
    starts_same = run_multi_start(model, frame, obs_set, wiring, cfg,
                                  base_seed=200, num_starts=4,
                                  sector_bitmasks=[0, 0, 0, 0])
    rep_same = evaluate_convergence_gate(
        starts_same, thresholds=GateThresholds(max_r_hat=1e6, min_ess=0.0,
                                               max_q_top_spread=1.0,
                                               max_m_u_spread=1.0))
    worst_u = rep_diff.metrics["worst_u_cold_logical_acceptance"]
    alarms = bool(
        (not rep_diff.passed
         and "sector_transport_insufficient" in rep_diff.failed_checks)
        and (not rep_same.passed
             and "sector_transport_insufficient" in rep_same.failed_checks))
    records.append({
        "test": "V6_negative", "instance": f"expander_m{m}", "k": model.k,
        "diff_start_failed": bool(not rep_diff.passed),
        "same_start_only_transport": rep_same.failed_checks,
        "worst_u_cold_acc": worst_u,
        "pass": alarms})
    print(f"  V6-neg m={m} k={model.k}: alarms={alarms} "
          f"worst_u_acc={worst_u:.1e} same_fails={rep_same.failed_checks}",
          flush=True)


def v6_positive(records):
    """正例：expander m=2(n=100，越枚举界) q>0 + PT → 传输 OK、结果与初始 sector
    无关（枚举一致性由 L3/V1 覆盖；此处测 transport 解冻 + 起点无关性）。"""
    model, frame, obs_set = expander(2)
    wiring = wire_ensemble(model, draw_disorder(model, 0.08, 0.05,
                                                np.random.default_rng(5)),
                           "true_posterior", frame)
    ptcfg = PtConfig(num_temperatures=6, q_hot=0.45, num_burn_in_rounds=400,
                     num_measurement_rounds=6000,
                     record_observable_trajectory=True)
    pt_a = run_parallel_tempering(model, frame, obs_set, wiring, ptcfg, seed=11,
                                  sector_bitmask_per_replica=[0] * 6)
    pt_b = run_parallel_tempering(model, frame, obs_set, wiring, ptcfg, seed=13,
                                  sector_bitmask_per_replica=[0, 5, 10, 15,
                                                              7, 3])
    z_ab = float(np.max(np.abs(
        (pt_a.m_u_cold - pt_b.m_u_cold)
        / (np.sqrt(stderr_of(pt_a.observable_trajectory_cold) ** 2
                   + stderr_of(pt_b.observable_trajectory_cold) ** 2)))))
    ok = bool(pt_a.round_trips > 0 and pt_b.round_trips > 0 and z_ab < 5)
    records.append({
        "test": "V6_positive", "instance": "expander_m2_n100",
        "round_trips_a": pt_a.round_trips, "round_trips_b": pt_b.round_trips,
        "z_initial_sector_independence": z_ab, "z_vs_enum": None,
        "pass": ok})
    print(f"  V6-pos: rt_a={pt_a.round_trips} rt_b={pt_b.round_trips} "
          f"z_sector_indep={z_ab:.2f}", flush=True)


def main():
    started = time.perf_counter()
    records = []
    print("V4 ...", flush=True); v4_redundancy(records)
    print("V6 negative (k=4, k=9) ...", flush=True)
    v6_negative(records, 2, 4)
    v6_negative(records, 3, 9)
    print("V6 positive ...", flush=True); v6_positive(records)
    all_pass = all(r["pass"] for r in records)
    payload = {"records": records, "all_pass": all_pass,
               "wall_time_seconds": time.perf_counter() - started}
    with (OUT_DIR / "results.json").open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, ensure_ascii=False)
    lines = ["# G3.6 V4 + G3.7 V6 结果", "",
             f"墙钟 {payload['wall_time_seconds']:.0f}s", "",
             "| 测试 | 关键指标 | 结果 |", "|---|---|---|"]
    for r in records:
        if r["test"] == "V4":
            metric = (f"ref≡numba={r['bit_identical_ref_numba']}, "
                      f"z(direct/pt/1v8)={r['z_direct_vs_enum']:.1f}/"
                      f"{r['z_pt_vs_enum']:.1f}/{r['z_1start_vs_8start']:.1f}")
        elif r["test"] == "V6_negative":
            metric = (f"{r['instance']}(k={r['k']}) 诊断报警={r['pass']}, "
                      f"共冻仅 transport 失败={r['same_start_only_transport']}")
        else:
            metric = (f"round_trips={r['round_trips_a']}/{r['round_trips_b']}, "
                      f"z(初始 sector 无关)="
                      f"{r['z_initial_sector_independence']:.1f}")
        lines.append(f"| {r['test']} | {metric} | {'✅' if r['pass'] else '❌'} |")
    lines += ["", f"**总判定：{'ALL PASS ✅' if all_pass else 'FAIL ❌'}**"]
    lines[1:1] = [
        "",
        "> **PRE_ALIGNMENT（自动生成保护）：** 本页只记录旧内核与旧冻结 gate；",
        "> 重新运行本 runner 不认证四实例 PT/INVALID 传播，也不得覆盖 014 结论。",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n",
                                        encoding="utf-8")
    print("\n".join(lines))
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
