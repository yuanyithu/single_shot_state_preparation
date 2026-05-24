#!/usr/bin/env python3
"""Summarize small 3D toric production optimization runs in Chinese."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


CONFIG_LABELS = {
    "pt7_single005_coldobs": "PT 7温度；单比特抽样5%；只测冷端；cluster关",
    "pt7_single010_coldobs": "PT 7温度；单比特抽样10%；只测冷端；cluster关",
    "pt7_single100_coldobs": "PT 7温度；单比特抽样100%；只测冷端；cluster关",
    "nopt_single010": "无PT；单比特抽样10%；cluster关",
}


def fmt(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(number):
        return "-"
    if abs(number) >= 100:
        return f"{number:.1f}"
    if abs(number) >= 10:
        return f"{number:.2f}"
    return f"{number:.{digits}f}"


def fmt_prob(value: Any) -> str:
    return fmt(value, 4)


def fmt_int(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(number):
        return "-"
    return str(int(round(number)))


def scalar(data: np.lib.npyio.NpzFile, key: str, default: Any = None) -> Any:
    if key not in data.files:
        return default
    array = np.asarray(data[key])
    if array.shape == ():
        return array.item()
    return array.tolist()


def as_float_array(data: np.lib.npyio.NpzFile, key: str) -> np.ndarray | None:
    if key not in data.files:
        return None
    return np.asarray(data[key], dtype=float)


def load_manifest(npz_path: Path) -> dict[str, Any]:
    manifest_path = npz_path.parent / "manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def nan_sum(data: np.lib.npyio.NpzFile, key: str) -> float:
    array = as_float_array(data, key)
    if array is None:
        return 0.0
    return float(np.nansum(array))


def value_at(matrix: np.ndarray | None, l_index: int, p_index: int = 0) -> float | None:
    if matrix is None:
        return None
    try:
        return float(matrix[l_index, p_index])
    except IndexError:
        return None


def timing_rows(data: np.lib.npyio.NpzFile) -> list[dict[str, Any]]:
    measurement = as_float_array(
        data, "chain_measurement_wall_time_per_disorder_per_start_replica_tensor"
    )
    ordinary = as_float_array(
        data, "chain_ordinary_update_wall_time_per_disorder_per_start_replica_tensor"
    )
    pt_swap = as_float_array(
        data, "chain_pt_swap_wall_time_per_disorder_per_start_replica_tensor"
    )
    observable = as_float_array(
        data, "chain_observable_wall_time_per_disorder_per_start_replica_tensor"
    )
    if measurement is None:
        return []

    rows = []
    for l_index, lattice_size in enumerate(np.asarray(data["lattice_size_list"], dtype=int)):
        total = float(np.nansum(measurement[l_index]))
        ordinary_seconds = float(np.nansum(ordinary[l_index])) if ordinary is not None else 0.0
        swap_seconds = float(np.nansum(pt_swap[l_index])) if pt_swap is not None else 0.0
        observable_seconds = (
            float(np.nansum(observable[l_index])) if observable is not None else 0.0
        )
        rows.append(
            {
                "L": int(lattice_size),
                "measurement": total,
                "ordinary": ordinary_seconds,
                "pt_swap": swap_seconds,
                "observable": observable_seconds,
                "other": total - ordinary_seconds - swap_seconds - observable_seconds,
            }
        )
    return rows


def config_summary(npz_path: Path) -> dict[str, Any]:
    data = np.load(npz_path, allow_pickle=True)
    manifest = load_manifest(npz_path)
    cfg = npz_path.parent.name
    lattice_sizes = np.asarray(data["lattice_size_list"], dtype=int)
    p_values = np.asarray(data["data_error_probability_list"], dtype=float)
    q_top = as_float_array(data, "q_top_curve_matrix")
    q_std = as_float_array(data, "q_top_std_curve_matrix")
    q_spread = as_float_array(data, "mean_q_top_spread_curve_matrix")
    rhat = as_float_array(data, "max_r_hat_curve_matrix")
    ess = as_float_array(data, "min_effective_sample_size_curve_matrix")
    accept = as_float_array(data, "average_acceptance_rate_curve_matrix")
    winding_accept = as_float_array(data, "mean_cold_winding_acceptance_rate_curve_matrix")
    pt_min = as_float_array(data, "mean_pt_min_swap_acceptance_rate_curve_matrix")
    pt_mean = as_float_array(data, "mean_pt_mean_swap_acceptance_rate_curve_matrix")
    timing = timing_rows(data)
    total_measurement = sum(row["measurement"] for row in timing)
    total_ordinary = sum(row["ordinary"] for row in timing)
    total_pt_swap = sum(row["pt_swap"] for row in timing)
    total_observable = sum(row["observable"] for row in timing)
    total_other = total_measurement - total_ordinary - total_pt_swap - total_observable

    l_rows = []
    for l_index, lattice_size in enumerate(lattice_sizes):
        l_rows.append(
            {
                "L": int(lattice_size),
                "q_top": value_at(q_top, l_index),
                "q_std": value_at(q_std, l_index),
                "q_spread": value_at(q_spread, l_index),
                "rhat": value_at(rhat, l_index),
                "ess": value_at(ess, l_index),
                "accept": value_at(accept, l_index),
                "winding_accept": value_at(winding_accept, l_index),
                "pt_min": value_at(pt_min, l_index),
                "pt_mean": value_at(pt_mean, l_index),
            }
        )

    config = manifest.get("config", {})
    final_outputs = manifest.get("final_outputs", {})
    return {
        "config": cfg,
        "label": CONFIG_LABELS.get(cfg, cfg),
        "npz_path": str(npz_path),
        "manifest_path": str(npz_path.parent / "manifest.json"),
        "remote_run_root": manifest.get("run_root", "-"),
        "status": final_outputs.get("status", "-"),
        "completed_chunks": manifest.get("summary", {}).get("completed_chunks", "-"),
        "failed_chunks": manifest.get("summary", {}).get("failed_chunks", "-"),
        "workers": config.get("workers", scalar(data, "workers", "-")),
        "lattice_sizes": lattice_sizes.tolist(),
        "p_values": p_values.tolist(),
        "q": float(np.asarray(data["syndrome_error_probability"])),
        "num_disorder": int(np.asarray(data["num_disorder_samples"])),
        "burn_in_requested": int(np.asarray(data["num_burn_in_sweeps"])),
        "effective_burn_in": np.asarray(data["effective_num_burn_in_sweeps_list"], dtype=int).tolist(),
        "measurements": int(np.asarray(data["num_measurements_per_disorder"])),
        "sweeps_between": int(np.asarray(data["num_sweeps_between_measurements"])),
        "starts": int(np.asarray(data["num_start_chains"])),
        "reps": int(np.asarray(data["num_replicas_per_start"])),
        "single_fraction": float(np.asarray(data["single_bit_proposal_fraction"])),
        "observable_mode": str(np.asarray(data["observable_temperature_mode"]).item()),
        "cluster_enabled": bool(np.asarray(data["cluster_update_config_enabled"]).item()),
        "cluster_attempts": int(np.nansum(np.asarray(data["cluster_num_attempts"]))),
        "cluster_nonzero": int(np.nansum(np.asarray(data["cluster_num_nonzero_moves"]))),
        "cluster_wall": float(np.nansum(np.asarray(data["cluster_total_wall_time"], dtype=float))),
        "pt_enabled": bool(np.asarray(data["pt_enabled"]).item()),
        "pt_temperatures": scalar(data, "pt_num_temperatures", None),
        "pt_p_hot": scalar(data, "pt_p_hot", None),
        "pt_swap_every": config.get("pt_swap_attempt_every_num_sweeps", "-"),
        "total_measurement": total_measurement,
        "total_ordinary": total_ordinary,
        "total_pt_swap": total_pt_swap,
        "total_observable": total_observable,
        "total_other": total_other,
        "timing": timing,
        "l_rows": l_rows,
    }


def timing_percent(seconds: float, total: float) -> str:
    if total <= 0:
        return "-"
    return fmt(100.0 * seconds / total, 2)


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def build_report(root: Path, summaries: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("# 3D toric q=0.05, p=0.2 生产路径优化小实验汇总")
    lines.append("")
    lines.append(f"- 本地目录：`{root}`")
    if summaries:
        lines.append(f"- 远端总目录：`{Path(summaries[0]['remote_run_root']).parent}`")
    lines.append("- 目标：比较 PT、单比特抽样比例、cold-only observable 对真实运行时间和收敛诊断的影响。")
    lines.append("- 重要口径：`q_top跨链极差` 是同一个 disorder 内不同 start chain 的 `q_top` 最大值减最小值，再对 disorder 平均；它不是某个更新环节内部的物理量。")
    lines.append("- `Rhat最大值` 是多链收敛诊断，越接近 1 越好；`最小ESS` 是跨逻辑观测量/链诊断里最差的有效样本量估计，越大越好。")
    lines.append("- 备注：本轮 no-PT 的真实时间字段来自旧口径，少计了 burn-in 计时；其采样值、接受概率、Rhat、ESS 仍可用，真实时间只能当偏乐观下界。")
    lines.append("")

    lines.append("## 每次实验内部明细")
    lines.append("")
    for summary in summaries:
        lines.append(f"### {summary['config']}：{summary['label']}")
        lines.append("")
        param_rows = [
            ["L", ",".join(str(value) for value in summary["lattice_sizes"])],
            ["p", ",".join(fmt_prob(value) for value in summary["p_values"])],
            ["q", fmt_prob(summary["q"])],
            ["disorder数/每个(L,p)", str(summary["num_disorder"])],
            ["burn-in sweep", f"{summary['burn_in_requested']} 请求；实际 {summary['effective_burn_in']}"],
            ["测量数", str(summary["measurements"])],
            ["两次测量间local sweep", str(summary["sweeps_between"])],
            ["start chain数", str(summary["starts"])],
            ["每个start的replica数", str(summary["reps"])],
            ["PT温度数", str(summary["pt_temperatures"] or 1)],
            ["PT热端p", "-" if summary["pt_p_hot"] is None else fmt_prob(summary["pt_p_hot"])],
            ["PT交换频率", f"每 {summary['pt_swap_every']} 个sweep尝试一次" if summary["pt_enabled"] else "-"],
            ["单比特抽样比例", fmt_prob(summary["single_fraction"])],
            ["observable温度", summary["observable_mode"]],
            ["cluster", "开" if summary["cluster_enabled"] else "关"],
            ["完成chunk", f"{summary['completed_chunks']} completed / {summary['failed_chunks']} failed"],
        ]
        lines.extend(markdown_table(["项目", "值"], param_rows))
        lines.append("")

        timing_rows_md = []
        for row in summary["timing"]:
            total = row["measurement"]
            timing_rows_md.append(
                [
                    str(row["L"]),
                    fmt(row["measurement"], 3),
                    fmt(row["ordinary"], 3),
                    timing_percent(row["ordinary"], total),
                    fmt(row["pt_swap"], 3),
                    timing_percent(row["pt_swap"], total),
                    fmt(row["observable"], 3),
                    timing_percent(row["observable"], total),
                    fmt(row["other"], 3),
                    timing_percent(row["other"], total),
                ]
            )
        total = summary["total_measurement"]
        timing_rows_md.append(
            [
                "合计",
                fmt(total, 3),
                fmt(summary["total_ordinary"], 3),
                timing_percent(summary["total_ordinary"], total),
                fmt(summary["total_pt_swap"], 3),
                timing_percent(summary["total_pt_swap"], total),
                fmt(summary["total_observable"], 3),
                timing_percent(summary["total_observable"], total),
                fmt(summary["total_other"], 3),
                timing_percent(summary["total_other"], total),
            ]
        )
        lines.extend(
            markdown_table(
                [
                    "L",
                    "真实时间(s)",
                    "普通更新(s)",
                    "普通更新占比(%)",
                    "PT交换(s)",
                    "PT交换占比(%)",
                    "可观测量(s)",
                    "可观测量占比(%)",
                    "其他(s)",
                    "其他占比(%)",
                ],
                timing_rows_md,
            )
        )
        lines.append("")

        metric_rows = []
        for row in summary["l_rows"]:
            metric_rows.append(
                [
                    str(row["L"]),
                    fmt(row["q_top"], 4),
                    fmt(row["q_std"], 4),
                    fmt(row["q_spread"], 4),
                    fmt(row["rhat"], 4),
                    fmt(row["ess"], 1),
                    fmt(row["accept"], 4),
                    fmt(row["winding_accept"], 4),
                    "-" if row["pt_min"] is None else fmt(row["pt_min"], 4),
                    "-" if row["pt_mean"] is None else fmt(row["pt_mean"], 4),
                ]
            )
        lines.extend(
            markdown_table(
                [
                    "L",
                    "q_top均值",
                    "q_top disorder标准差",
                    "q_top跨链极差",
                    "Rhat最大值",
                    "最小ESS",
                    "普通更新接受概率",
                    "winding接受概率",
                    "PT最小交换接受概率",
                    "PT平均交换接受概率",
                ],
                metric_rows,
            )
        )
        lines.append("")

    lines.append("## 跨实验对比")
    lines.append("")
    compare_rows = []
    for summary in summaries:
        for row in summary["l_rows"]:
            compare_rows.append(
                [
                    summary["config"],
                    str(row["L"]),
                    "是" if summary["pt_enabled"] else "否",
                    str(summary["pt_temperatures"] or 1),
                    fmt_prob(summary["single_fraction"]),
                    fmt(summary["total_measurement"], 3),
                    fmt(
                        next(t["measurement"] for t in summary["timing"] if t["L"] == row["L"]),
                        3,
                    ),
                    fmt(row["q_top"], 4),
                    fmt(row["q_spread"], 4),
                    fmt(row["rhat"], 4),
                    fmt(row["ess"], 1),
                    fmt(row["accept"], 4),
                    "-" if row["pt_min"] is None else fmt(row["pt_min"], 4),
                ]
            )
    lines.extend(
        markdown_table(
            [
                "实验",
                "L",
                "PT",
                "温度数",
                "单比特抽样比例",
                "总真实时间(s)",
                "该L真实时间(s)",
                "q_top均值",
                "q_top跨链极差",
                "Rhat最大值",
                "最小ESS",
                "普通更新接受概率",
                "PT最小交换接受概率",
            ],
            compare_rows,
        )
    )
    lines.append("")

    lines.append("## 当前判断")
    lines.append("")
    lines.append("- 这批实验已经完成并回收到本地；4 个配置全部 `completed=4, failed=0`。")
    lines.append("- cluster 在这批生产优化实验里全部关闭，所以本轮不能用来判断 cluster 的物理收益，只能确认关闭 cluster 后路径正常。")
    lines.append("- PT7 的交换接受率不低，说明相邻温度之间能交换；但 L=5 的 `q_top跨链极差` 和 `Rhat` 仍需要看具体配置，不能只看交换接受率判断收敛。")
    lines.append("- 单比特抽样比例从 100% 降到 5% 明显降低普通更新时间；是否足够收敛要以 `q_top跨链极差/Rhat/ESS` 为准。")
    lines.append("- no-PT 10% 路径便宜，但本轮时间口径少计 burn-in，且它没有 PT 的跨温度输运，不能只凭短时间胜出。")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    npz_paths = sorted(args.run_dir.glob("*/scan_*.npz"))
    if not npz_paths:
        raise SystemExit(f"No scan_*.npz files found under {args.run_dir}")
    summaries = [config_summary(path) for path in npz_paths]
    order = {
        "pt7_single005_coldobs": 0,
        "pt7_single010_coldobs": 1,
        "pt7_single100_coldobs": 2,
        "nopt_single010": 3,
    }
    summaries.sort(key=lambda item: order.get(item["config"], 99))
    report = build_report(args.run_dir, summaries)
    if args.output:
        args.output.write_text(report + "\n", encoding="utf-8")
    else:
        print(report)


if __name__ == "__main__":
    main()
