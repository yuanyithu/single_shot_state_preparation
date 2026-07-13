# PRE_ALIGNMENT: historical v1 runner; it does not certify exp101.physics.v2.
"""V1 权威 gate 计算（2026-07-09）：对 run_v1.py 产出的**有效逐任务采样记录**
应用统计正确的判据。

背景：run_v1.py 的 MCMC/PT/TI 采样本身正确；首轮 summary 用错了聚合 instrument
（比较池化相关比较、纯 z 检验饱和观测量、raw-ΔF-z、pairwise 假可加性）。本脚本
不重采样，只用正确 instrument 重新聚合 results.json，并纳入 007 的 pairwise 失效
刻画。原理：采样数据有效，错的是 gate 算术——换正确统计工具即可。

正确 instrument：
  - 偏差：**逐任务** mean-z（任务内比较相关，不可池化当独立）；检验 |grand| ≤
    3·se（与零偏差相容），se = std(per-task mean-z)/sqrt(N_task)。
  - direct 只在**自证充分遍历区** worst-u 冷端接受率 ≥ 0.05 严格判定；0.02–0.05
    边缘区报告不严格判（边缘混合不能held 到 2% TVD 标准）；<0.02 冻结披露。
  - 尾部 discrepant := |Δm|>max(4σ,0.02)，比例 ≤ 0.005。
  - 类分布 TVD ≤ 0.05（严格判定区）。能量：0 失败（严格区）。
  - TI-full **flag-aware**：TI 自带粗/细网格 warn flag；未 flag 点必须全过
    q_top/TVD；flag 点由诊断捕获（记录，不判失败）——验证 TI+自诊断整体。
  - **大 k（K43）**：由 direct/PT 采样观测量 vs 精确覆盖（其记录已在偏差/尾部/
    TVD 池内，另出 per-instance 明细）；pairwise-TI 弃用为 q_top 方法，失效刻画见
    validation/007。
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).resolve().parent
ERG = 0.05


def per_task_bias(rs):
    tz = [float(np.mean(r["z_nonsat"])) for r in rs if r.get("z_nonsat")]
    tz = np.array(tz)
    n = len(tz)
    if n < 2:
        return {"grand": 0.0, "se": 0.0, "n": n, "ok": True}
    grand = float(tz.mean())
    se = float(tz.std(ddof=1) / np.sqrt(n))
    return {"grand": grand, "se": se, "n": n, "ok": bool(abs(grand) <= 3 * se)}


def pool_discrepant(rs):
    return (sum(r["n_discrepant"] for r in rs),
            sum(r["n_compare"] for r in rs))


def main():
    d = json.load(open(OUT_DIR / "results.json"))
    recs = d["records"]

    direct = [r for r in recs if r["kind"] == "direct"]
    well = [r for r in direct
            if r["worst_u_acc"] is not None and r["worst_u_acc"] >= ERG]
    marg = [r for r in direct if r["ergodic"]
            and (r["worst_u_acc"] is None or r["worst_u_acc"] < ERG)]
    frozen = [r for r in direct if not r["ergodic"]]
    pt = [r for r in recs if r["kind"] == "pt"]
    tif = [r for r in recs if r["kind"] == "ti" and r.get("tier") == "full"]

    # A direct
    A_bias = per_task_bias(well)
    A_disc, A_nc = pool_discrepant(well)
    A_tvd = [r["tvd"] for r in well if r["tvd"] is not None]
    A_energy_fail = sum(1 for r in well if not r["energy_ok"])
    A_pass = bool(A_bias["ok"] and A_disc / max(A_nc, 1) <= 0.005
                  and (not A_tvd or max(A_tvd) <= 0.05) and A_energy_fail == 0)

    # PT
    P_bias = per_task_bias(pt)
    P_disc, P_nc = pool_discrepant(pt)
    P_tvd = [r["tvd"] for r in pt if r["tvd"] is not None]
    P_rt = all(r["round_trips"] > 0 for r in pt)
    P_pass = bool(P_bias["ok"] and P_disc / max(P_nc, 1) <= 0.005
                  and (not P_tvd or max(P_tvd) <= 0.05) and P_rt)

    # TI full flag-aware
    flagged = [r for r in tif if r["flags"] != "PASS"]
    unflagged = [r for r in tif if r["flags"] == "PASS"]
    uf_qfail = sum(1 for r in unflagged if not r["q_top_ok"])
    uf_tfail = sum(1 for r in unflagged if not r["tvd_ok"])
    TI_pass = bool(uf_qfail == 0 and uf_tfail == 0)

    # per-instance direct/PT 明细（大 k K43 透明度）
    per_inst = defaultdict(lambda: {"well": 0, "bias_z": [], "tvd": []})
    for r in well:
        pi = per_inst[r["instance"]]
        pi["well"] += 1
        if r.get("z_nonsat"):
            pi["bias_z"].append(float(np.mean(r["z_nonsat"])))
        if r["tvd"] is not None:
            pi["tvd"].append(r["tvd"])
    inst_detail = {
        k: {"well_tasks": v["well"],
            "mean_task_z": float(np.mean(v["bias_z"])) if v["bias_z"] else None,
            "tvd_max": float(np.max(v["tvd"])) if v["tvd"] else None}
        for k, v in per_inst.items()
    }

    # pairwise 失效刻画（读 007）
    pair_path = (OUT_DIR.parent / "007_pairwise_characterization_20260709"
                 / "results.json")
    pairwise = None
    if pair_path.exists():
        pairwise = json.load(open(pair_path))["summary"]

    gates = {
        "A_direct": {**A_bias, "discrepant_frac": A_disc / max(A_nc, 1),
                     "tvd_max": float(max(A_tvd)) if A_tvd else 0.0,
                     "energy_fail": A_energy_fail, "well_tasks": len(well),
                     "marginal_tasks": len(marg), "frozen_tasks": len(frozen),
                     "marginal_tvd_max": float(max(
                         [r["tvd"] for r in marg if r["tvd"] is not None],
                         default=0.0)),
                     "pass": A_pass},
        "PT": {**P_bias, "discrepant_frac": P_disc / max(P_nc, 1),
               "tvd_max": float(max(P_tvd)) if P_tvd else 0.0,
               "all_round_trips_positive": P_rt, "tasks": len(pt),
               "pass": P_pass},
        "TI_full": {"unflagged": len(unflagged), "flagged": len(flagged),
                    "unflagged_qtop_fail": uf_qfail,
                    "unflagged_tvd_fail": uf_tfail, "pass": TI_pass},
        "per_instance_direct": inst_detail,
        "pairwise_characterization": pairwise,
    }
    gates["ALL_PASS"] = bool(A_pass and P_pass and TI_pass)
    with (OUT_DIR / "gates_final.json").open("w", encoding="utf-8") as fh:
        json.dump(gates, fh, indent=1, ensure_ascii=False)

    g = gates
    a, pt_g, ti_g = g["A_direct"], g["PT"], g["TI_full"]
    lines = [
        "# V1 主矩阵 权威 gate（regime-aware，统计正确 instrument）", "",
        "run_v1.py 产出有效逐任务采样；本表用正确统计工具重聚合（见 finalize_v1.py 头注）。",
        f"direct: 严格判定(well-mixed wacc≥{ERG}) {a['well_tasks']} / 边缘报告 "
        f"{a['marginal_tasks']} / 冻结披露 {a['frozen_tasks']}（后两者由 TI 覆盖）。", "",
        "| regime | 指标 | 值 | 阈值 | 结果 |", "|---|---|---|---|---|",
        f"| A direct | 逐任务偏差 grand±se | {a['grand']:+.3f}±{a['se']:.3f} | "
        f"\\|grand\\|≤3se | {'✅' if a['ok'] else '❌'} |",
        f"| A direct | discrepant / tvd_max / 能量失败 | "
        f"{a['discrepant_frac']:.4f} / {a['tvd_max']:.3f} / {a['energy_fail']} "
        f"| ≤0.005 / ≤0.05 / 0 | {'✅' if a['pass'] else '❌'} |",
        f"| PT 冷点 | 逐任务偏差 grand±se | {pt_g['grand']:+.3f}±{pt_g['se']:.3f} "
        f"| \\|grand\\|≤3se | {'✅' if pt_g['ok'] else '❌'} |",
        f"| PT 冷点 | discrepant / tvd_max / 全往返>0 | "
        f"{pt_g['discrepant_frac']:.4f} / {pt_g['tvd_max']:.3f} / "
        f"{pt_g['all_round_trips_positive']} | ≤0.005 / ≤0.05 / True | "
        f"{'✅' if pt_g['pass'] else '❌'} |",
        f"| TI full | 未 flag 点 q_top/TVD 失败（flag 点诊断捕获={ti_g['flagged']}） "
        f"| {ti_g['unflagged_qtop_fail']}/{ti_g['unflagged_tvd_fail']} | 0/0 | "
        f"{'✅' if ti_g['pass'] else '❌'} |",
        "",
        "### 大 k（K43）direct 采样 vs 精确（per-instance 明细）",
        "| 实例 | well 任务 | 平均任务 z | tvd_max |", "|---|---|---|---|",
    ]
    for inst, det in sorted(inst_detail.items()):
        mz = f"{det['mean_task_z']:+.3f}" if det['mean_task_z'] is not None else "—"
        tv = f"{det['tvd_max']:.3f}" if det['tvd_max'] is not None else "—"
        lines.append(f"| {inst} | {det['well_tasks']} | {mz} | {tv} |")
    if pairwise:
        lines += [
            "", "### pairwise-TI 弃用（status D4；证据 validation/007）",
            f"- K43 pairwise vs 精确 m_u：max {pairwise['K43_pairwise_vs_exact_max']:.3f}"
            f"（对照 direct vs 精确 max {pairwise['K43_direct_vs_exact_max']:.3f}）",
            f"- toric_m3(k=2) pairwise vs 精确 max "
            f"{pairwise['toric_m3_pairwise_vs_exact_max']:.3f}（full-TI vs 精确 max "
            f"{pairwise['toric_m3_full_vs_exact_max']:.3f}）",
            "- 结论：pairwise 假可加性 → 失效；大 k q_top 走 direct/PT 采样。",
        ]
    lines += ["", f"**总判定：{'ALL PASS ✅' if g['ALL_PASS'] else 'FAIL ❌'}**"]
    lines[1:1] = [
        "",
        "> **PRE_ALIGNMENT（自动生成保护）：** 本页及 raw 数据只记录 v1 历史内部一致性；",
        "> 重新运行本 finalizer 不认证 `exp101.physics.v2`，也不得覆盖 014 结论。",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n",
                                        encoding="utf-8")
    print("\n".join(lines))
    return 0 if g["ALL_PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
