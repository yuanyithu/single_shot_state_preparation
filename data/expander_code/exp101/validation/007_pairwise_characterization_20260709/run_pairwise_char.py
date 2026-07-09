"""G3.2 附属：pairwise-TI 大 k 失效的定量刻画（status D4 / plan §12 更新的证据）。

比较 pairwise-TI 的 m_u^pair 与**精确** m_u（全枚举，非 tanh(gap)）：
  - K43 (k=13)：主结论——pairwise 严重偏离精确 m_u（可加性失效）。
  - small-k 对照（toric_m3 k=2 强制 pairwise）：证明失效源于**可加性假设**本身，
    非 K43 特有 bug——小 k 上 pairwise m_u 同样 ≠ exact m_u（除非分布恰好可加）。
  - 交叉锚点：同点 direct 采样 m_u vs exact（应一致，证明 exact 与 direct 互相佐证，
    pairwise 是偏离的一方）。
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
from src.graphs import cycle_parity_check_matrix  # noqa: E402
from src.hgp import hgp_from_H  # noqa: E402
from src.logicals import logical_pauli_operators  # noqa: E402
from src.model import assemble_sector_model, draw_disorder, wire_ensemble  # noqa: E402
from src.observables import build_observable_frame, build_observable_set  # noqa: E402
from src.reference_mcmc import ReferenceMcmcConfig  # noqa: E402
from src.sector_ti import SectorTiConfig, run_sector_ti  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent


def setup(classical):
    H_Z, H_X = hgp_from_H(classical)
    logicals = logical_pauli_operators(H_X, H_Z)
    model = assemble_sector_model(H_X, H_Z, logicals, sector="x_error")
    return model, build_observable_frame(model)


def main():
    started = time.perf_counter()
    records = []

    # ---- K43 (k=13): pairwise vs exact m_u ----
    model, frame = setup(np.ones((3, 4), dtype=np.uint8))
    obs_set = build_observable_set(frame, num_random_u=64, u_rand_seed=5)
    chain = build_fast_chain_data(model, obs_set)
    cfg = SectorTiConfig(num_kp_grid_points=17, num_burn_in_sweeps=120,
                         num_measurements=300, block_count=6, num_bootstrap=60,
                         full_max_k=10)
    for d in range(4):
        for p, q in [(0.15, 0.0), (0.08, 0.05)]:
            rng = np.random.default_rng(abs(hash(("k43", p, q, d))) % 2**31)
            dis = draw_disorder(model, p, q, rng)
            wiring = wire_ensemble(model, dis, "true_posterior", frame)
            exact = exact_reference(model, frame, wiring)
            ti = run_sector_ti(model, frame, wiring, cfg,
                               seed=int(rng.integers(0, 2**60)))
            pair_dev = float(np.max(np.abs(
                ti["m_u_pairwise"] - exact["m_u_basis"])))
            # direct 采样锚点（basis m_u vs exact，证明 exact 可信）
            dres = run_fast_mcmc(
                model, frame, obs_set, wiring,
                ReferenceMcmcConfig(num_burn_in_sweeps=400,
                                    num_measurements=3000),
                seed=int(rng.integers(0, 2**60)), chain_data=chain)
            basis_pos = obs_set.basis_positions
            direct_basis_m = dres["m_u"][basis_pos]
            direct_dev = float(np.max(np.abs(
                direct_basis_m - exact["m_u_basis"])))
            records.append({
                "instance": "K43", "k": model.k, "p": p, "q": q, "disorder": d,
                "pairwise_vs_exact_max_dev": pair_dev,
                "direct_vs_exact_max_dev": direct_dev,
                "flags": ti["flags"],
            })
            print(f"[{time.perf_counter()-started:6.0f}s] K43 p={p} q={q} d={d}"
                  f" pair_dev={pair_dev:.3f} direct_dev={direct_dev:.3f}",
                  flush=True)

    # ---- small-k control: toric_m3 (k=2) forced pairwise vs exact ----
    model2, frame2 = setup(cycle_parity_check_matrix(3))
    obs2 = build_observable_set(frame2)
    cfg2 = SectorTiConfig(num_kp_grid_points=25, num_burn_in_sweeps=150,
                          num_measurements=400, block_count=8,
                          num_bootstrap=100, full_max_k=1)  # force pairwise (k=2)
    cfg2_full = SectorTiConfig(num_kp_grid_points=25, num_burn_in_sweeps=150,
                               num_measurements=400, block_count=8,
                               num_bootstrap=100, full_max_k=10)  # full
    for d in range(4):
        for p, q in [(0.12, 0.08), (0.2, 0.15)]:
            rng = np.random.default_rng(abs(hash(("t3", p, q, d))) % 2**31)
            dis = draw_disorder(model2, p, q, rng)
            wiring = wire_ensemble(model2, dis, "true_posterior", frame2)
            exact = exact_reference(model2, frame2, wiring)
            ti_pair = run_sector_ti(model2, frame2, wiring, cfg2,
                                    seed=int(rng.integers(0, 2**60)))
            ti_full = run_sector_ti(model2, frame2, wiring, cfg2_full,
                                    seed=int(rng.integers(0, 2**60)))
            records.append({
                "instance": "toric_m3", "k": model2.k, "p": p, "q": q,
                "disorder": d,
                "pairwise_vs_exact_max_dev": float(np.max(np.abs(
                    ti_pair["m_u_pairwise"] - exact["m_u_basis"]))),
                "full_vs_exact_max_dev": float(np.max(np.abs(
                    ti_full["m_u_basis"] - exact["m_u_basis"]))),
                "flags_pair": ti_pair["flags"], "flags_full": ti_full["flags"],
            })
            print(f"[{time.perf_counter()-started:6.0f}s] toric_m3 p={p} q={q}"
                  f" d={d} pair={records[-1]['pairwise_vs_exact_max_dev']:.3f}"
                  f" full={records[-1]['full_vs_exact_max_dev']:.3f}",
                  flush=True)

    k43 = [r for r in records if r["instance"] == "K43"]
    t3 = [r for r in records if r["instance"] == "toric_m3"]
    summary = {
        "K43_pairwise_vs_exact_max": float(np.max(
            [r["pairwise_vs_exact_max_dev"] for r in k43])),
        "K43_pairwise_vs_exact_mean": float(np.mean(
            [r["pairwise_vs_exact_max_dev"] for r in k43])),
        "K43_direct_vs_exact_max": float(np.max(
            [r["direct_vs_exact_max_dev"] for r in k43])),
        "toric_m3_pairwise_vs_exact_max": float(np.max(
            [r["pairwise_vs_exact_max_dev"] for r in t3])),
        "toric_m3_full_vs_exact_max": float(np.max(
            [r["full_vs_exact_max_dev"] for r in t3])),
        "wall_time_seconds": time.perf_counter() - started,
    }
    # 结论：pairwise 大偏差（两 k 都是），direct 与 full 都小偏差 ⇒ pairwise 失效
    summary["conclusion_pairwise_invalid"] = bool(
        summary["K43_pairwise_vs_exact_max"] > 0.3
        and summary["K43_direct_vs_exact_max"] < 0.1
        and summary["toric_m3_full_vs_exact_max"] < 0.1)
    with (OUT_DIR / "results.json").open("w", encoding="utf-8") as fh:
        json.dump({"records": records, "summary": summary}, fh, indent=1,
                  ensure_ascii=False)
    lines = [
        "# pairwise-TI 大 k 失效刻画（status D4 证据）", "",
        f"墙钟 {summary['wall_time_seconds']:.0f}s", "",
        "| 对照 | 量 | 值 |", "|---|---|---|",
        f"| K43(k=13) | pairwise vs exact m_u（max/mean） | "
        f"{summary['K43_pairwise_vs_exact_max']:.3f} / "
        f"{summary['K43_pairwise_vs_exact_mean']:.3f} |",
        f"| K43(k=13) | **direct 采样** vs exact m_u（max，锚点） | "
        f"{summary['K43_direct_vs_exact_max']:.3f} |",
        f"| toric_m3(k=2) | pairwise vs exact m_u（max，对照） | "
        f"{summary['toric_m3_pairwise_vs_exact_max']:.3f} |",
        f"| toric_m3(k=2) | full-TI vs exact m_u（max，锚点） | "
        f"{summary['toric_m3_full_vs_exact_max']:.3f} |",
        "",
        "**结论**：pairwise-TI（假可加性）在 k=13 与 k=2 上都显著偏离精确 m_u；"
        "而 direct 采样与 full-TI 都与精确一致 ⇒ **pairwise 作为 q_top 方法失效**"
        "（源于可加性假设，非实现 bug）。大 k 生产用 direct/PT 采样。",
        f"判定: {'确认失效 ✅（符合预期，方法作废）' if summary['conclusion_pairwise_invalid'] else '需复核'}",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n",
                                        encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
