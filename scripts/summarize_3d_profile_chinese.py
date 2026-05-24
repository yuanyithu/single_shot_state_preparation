#!/usr/bin/env python3
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


STAGE_LABELS = {
    "single_bit": "单比特翻转",
    "contractible": "局部零syndrome翻转",
    "winding": "绕圈零syndrome翻转",
    "cluster": "cluster翻转",
    "pt_swap": "PT副本交换",
    "observable": "可观测量计算",
}


CONFIG_LABELS = {
    "opt_no_cluster_PT7_full_single_coldobs": "PT7；cluster关；全量单比特；只算冷端可观测量",
    "opt_no_cluster_PT7_single_0p25_coldobs": "PT7；cluster关；25%单比特；只算冷端可观测量",
    "opt_no_cluster_PT7_single_0p10_coldobs": "PT7；cluster关；10%单比特；只算冷端可观测量",
    "opt_no_cluster_PT7_single_0p05_coldobs": "PT7；cluster关；5%单比特；只算冷端可观测量",
    "opt_noPT_no_cluster_full_single": "无PT；cluster关；全量单比特",
    "opt_noPT_no_cluster_single_0p10": "无PT；cluster关；10%单比特",
}


def fmt(value, digits=3):
    if value is None:
        return "-"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(value):
        return "-"
    if abs(value) >= 100:
        return f"{value:.1f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    return f"{value:.{digits}f}"


def pct(value):
    return fmt(100.0 * float(value), 2)


def integer(value):
    return str(int(round(float(value))))


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def aggregate_stage(rows):
    totals = {
        name: {
            "wall_time": 0.0,
            "attempted": 0,
            "accepted": 0,
            "sector_changes": 0,
        }
        for name in STAGE_LABELS
    }
    for row in rows:
        for name, source in row["stage_totals"].items():
            target = totals[name]
            target["wall_time"] += float(source.get("wall_time", 0.0))
            target["attempted"] += int(source.get("attempted", 0))
            target["accepted"] += int(source.get("accepted", 0))
            target["sector_changes"] += int(source.get("sector_changes", 0))
    total_stage_wall = sum(item["wall_time"] for item in totals.values())
    for item in totals.values():
        item["wall_fraction"] = (
            item["wall_time"] / total_stage_wall if total_stage_wall else 0.0
        )
        item["acceptance_rate"] = (
            item["accepted"] / item["attempted"] if item["attempted"] else None
        )
    return totals, total_stage_wall


def task_param(rows):
    task = rows[0]["_task"]
    ladder = task.get("data_error_probability_ladder") or []
    if ladder:
        ladder_text = ", ".join(fmt(value, 4) for value in ladder)
    else:
        ladder_text = "-"
    return {
        "pt_num_temperatures": int(task.get("pt_num_temperatures", 1)),
        "pt_p_hot": task.get("pt_p_hot"),
        "swap_every": int(task.get("pt_swap_attempt_every_num_sweeps", 0)),
        "cluster_enabled": bool(task.get("cluster_update_enabled", False)),
        "single_fraction": float(task.get("single_bit_proposal_fraction", 1.0)),
        "observable_mode": str(task.get("observable_temperature_mode", "all")),
        "zero_sweeps": int(task.get("num_zero_syndrome_sweeps_per_cycle", 1)),
        "winding_repeat": int(task.get("winding_repeat_factor", 1)),
        "starts": int(task.get("num_start_chains", 1)),
        "reps": int(task.get("num_replicas_per_start", 1)),
        "ladder": ladder_text,
        "stage_signature_mode": str(task.get("stage_signature_mode", "stage")),
    }


def summarize_group(rows):
    stage_totals, stage_total_wall = aggregate_stage(rows)
    q_top_values = [float(row["mean_q_top"]) for row in rows]
    ess_values = [float(row["mean_ess_per_total_second"]) for row in rows]
    total_wall = sum(float(row["total_wall_time"]) for row in rows)
    signature_wall = sum(float(row.get("signature_probe_wall_time", 0.0)) for row in rows)
    instrumentation_wall = sum(
        float(row.get("instrumentation_wall_time", 0.0)) for row in rows
    )
    swap_attempts = None
    swap_accepts = None
    for row in rows:
        attempts = [int(value) for value in row.get("swap_attempt_counts", [])]
        accepts = [int(value) for value in row.get("swap_accept_counts", [])]
        if swap_attempts is None:
            swap_attempts = [0] * len(attempts)
            swap_accepts = [0] * len(accepts)
        for i, value in enumerate(attempts):
            swap_attempts[i] += value
        for i, value in enumerate(accepts):
            swap_accepts[i] += value
    swap_rates = []
    if swap_attempts:
        for accepts, attempts in zip(swap_accepts, swap_attempts):
            if attempts:
                swap_rates.append(accepts / attempts)
    dominant_stage = max(
        stage_totals,
        key=lambda name: stage_totals[name]["wall_time"],
    )
    q_top_spreads = [float(row["q_top_spread"]) for row in rows]
    r_hats = [float(row["max_r_hat"]) for row in rows]
    return {
        "rows": rows,
        "stage_totals": stage_totals,
        "stage_total_wall": stage_total_wall,
        "total_wall": total_wall,
        "signature_wall": signature_wall,
        "instrumentation_wall": instrumentation_wall,
        "mean_q_top": sum(q_top_values) / len(q_top_values),
        "mean_ess_per_total_second": sum(ess_values) / len(ess_values),
        "mean_q_top_spread": sum(q_top_spreads) / len(q_top_spreads),
        "max_r_hat": max(r_hats),
        "dominant_stage": dominant_stage,
        "dominant_fraction": stage_totals[dominant_stage]["wall_fraction"],
        "swap_mean": sum(swap_rates) / len(swap_rates) if swap_rates else None,
        "swap_min": min(swap_rates) if swap_rates else None,
        "cold_flips": sum(int(row.get("cold_sector_flip_count", 0)) for row in rows),
        "hot_flips": sum(int(row.get("hot_sector_flip_count", 0)) for row in rows),
        "hot_to_cold": sum(
            int(row.get("hot_to_cold_sector_delivery_count", 0))
            for row in rows
        ),
        "never_flipped": sum(
            int(row.get("num_chains_that_never_flipped_sector", 0))
            for row in rows
        ),
        "cluster_nonzero": sum(
            int(row.get("cluster_nonzero_count", 0)) for row in rows
        ),
        "cluster_attempts": sum(
            int(row.get("cluster_attempt_count", 0)) for row in rows
        ),
    }


def load_rows(run_dir):
    rows = []
    for path in sorted((run_dir / "raw_json").glob("*.json")):
        payload = read_json(path)
        row = dict(payload["task_summary"])
        row["_task"] = payload["task"]
        row["_path"] = str(path)
        rows.append(row)
    return rows


def build_report(run_dir):
    summary_path = run_dir / "profile_summary.json"
    summary = read_json(summary_path) if summary_path.exists() else {}
    manifest_path = run_dir / "profile_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    rows = load_rows(run_dir)
    all_tasks = manifest.get("tasks") or summary.get("tasks") or [
        row["_task"] for row in rows
    ]
    lattice_sizes = sorted({int(task["lattice_size"]) for task in all_tasks})
    p_values = sorted({float(task["p_value"]) for task in all_tasks})
    num_disorders_per_group = sorted(
        {
            len([
                row for row in rows
                if (
                    int(row["lattice_size"]) == int(task["lattice_size"])
                    and float(row["p_value"]) == float(task["p_value"])
                    and str(row["config_label"]) == str(task["config_label"])
                )
            ])
            for task in all_tasks
        }
    )
    first_task = all_tasks[0] if all_tasks else {}
    stage_signature_modes = sorted({
        str(task.get("stage_signature_mode", "stage")) for task in all_tasks
    })
    stage_signature_disabled = stage_signature_modes == ["none"]
    grouped = defaultdict(list)
    for row in rows:
        key = (
            int(row["lattice_size"]),
            float(row["p_value"]),
            str(row["config_label"]),
        )
        grouped[key].append(row)

    groups = []
    for key in sorted(grouped):
        group = summarize_group(grouped[key])
        group["key"] = key
        group["params"] = task_param(grouped[key])
        groups.append(group)

    lines = []
    lines.append("# 3D q>0 优化 profiling 中文汇总")
    lines.append("")
    lines.append("## 运行信息")
    lines.append("")
    lines.append(f"- 本地目录：`{run_dir}`")
    lines.append(f"- 远端目录：`{manifest.get('run_root', '-')}`")
    if summary:
        lines.append(f"- suite：`{summary.get('suite')}`")
        lines.append(f"- q：`{summary.get('q_value')}`")
        lines.append(
            f"- 完成任务：`{summary.get('num_completed_tasks')}` / `{summary.get('num_tasks')}`，跳过 `{summary.get('num_skipped_tasks')}`"
        )
    if manifest:
        lines.append(
            f"- workers：`{manifest.get('workers')}`；max wall seconds：`{manifest.get('max_wall_seconds')}`"
        )
    disorder_text = (
        str(num_disorders_per_group[0])
        if len(num_disorders_per_group) == 1
        else ",".join(str(value) for value in num_disorders_per_group)
    )
    lines.append(
        "- 共同参数："
        f"`L={','.join(str(value) for value in lattice_sizes)}`，"
        f"`p={','.join(fmt(value, 4) for value in p_values)}`，"
        f"每组 disorder 数 `{disorder_text}`；"
        f"burn-in `{first_task.get('num_burn_in_sweeps', '-')}` sweeps，"
        f"measurement `{first_task.get('num_measurements', '-')}` 次，"
        f"每次间隔 `{first_task.get('num_sweeps_between_measurements', '-')}` sweeps。"
    )
    lines.append(
        f"- stage signature mode：`{','.join(stage_signature_modes)}`。"
    )
    lines.append("- 本轮所有优化配置都关闭了 cluster；PT 配置使用 7 个副本温度，noPT 配置只有冷端一个副本。")
    if stage_signature_disabled:
        lines.append(
            "- 注意：本轮关闭了逐环节逻辑签名诊断，因此表中的 `逻辑sector变化次数` 和 `hot→cold交付` 显示为未统计；这不表示 sector 没有变化。冷端/热端 sector 翻转仍由 measurement trace 估计。"
        )
    lines.append("")
    lines.append("## 指标怎么读")
    lines.append("")
    lines.append("- `真实时间(s)`：该环节在所有链、所有 disorder 上累计花掉的 wall time。注意本 profiler 为了判断每个环节是否改变逻辑 sector，会额外反复计算逻辑签名；这部分诊断开销单独列在每个实验摘要里。")
    lines.append("- `占比(%)`：该环节占六个被计时更新/观测环节总时间的比例；不是占总任务 wall time 的比例，也不是物理概率。")
    lines.append("- `尝试次数`：该环节 proposal 或计算发生的次数。单比特翻转的尝试次数会随稀疏比例下降。")
    lines.append("- `接受/有效次数`：Metropolis move 被接受的次数；对可观测量计算表示计算次数，不代表接受。")
    lines.append("- `接受概率`：`接受/有效次数 / 尝试次数`。可观测量没有 Metropolis 接受率，因此记为 `-`。")
    lines.append("- `逻辑sector变化次数`：该环节执行前后，primitive logical signature 是否变化的计数，用来判断这个环节是否在帮助跨逻辑扇区移动；若 `stage_signature_mode=none`，这一列未统计。")
    lines.append("- `q_top跨链极差`：同一个 `(L,p,config,disorder)` 内多条起点链得到的 `q_top` 最大值减最小值，再对 disorder 求平均。它是整条采样链的收敛/混合诊断，不属于某一个更新环节。")
    lines.append("- `Rhat最大值`：Gelman-Rubin 多链诊断，越接近 1 越好；这里取所有逻辑 observable 中最大的那个。")
    lines.append("- `ESS/秒`：按 logical observable 自相关估计的有效样本数除以总真实时间，越高表示单位机器时间得到的有效统计量越多。")
    lines.append("")
    lines.append("## 每个实验内部明细")
    lines.append("")
    for group in groups:
        lattice_size, p_value, config_label = group["key"]
        params = group["params"]
        readable = CONFIG_LABELS.get(config_label, config_label)
        lines.append(
            f"### L={lattice_size}, p={fmt(p_value, 4)}, 配置：{readable}"
        )
        lines.append("")
        lines.append(
            f"- disorder数：`{len(group['rows'])}`；总真实时间：`{fmt(group['total_wall'], 2)}s`；其中六个更新/观测环节合计 `{fmt(group['stage_total_wall'], 2)}s`，逻辑签名诊断开销约 `{fmt(group['signature_wall'], 2)}s`。ESS/秒：`{fmt(group['mean_ess_per_total_second'], 3)}`；q_top跨链极差：`{fmt(group['mean_q_top_spread'], 4)}`；Rhat最大值：`{fmt(group['max_r_hat'], 4)}`。"
        )
        lines.append(
            f"- PT副本数：`{params['pt_num_temperatures']}`；温度/等效p ladder：`{params['ladder']}`；swap间隔：每 `{params['swap_every']}` sweep 尝试一次相邻副本交换。"
        )
        lines.append(
            f"- cluster：`{'开' if params['cluster_enabled'] else '关'}`；单比特尝试比例：`{fmt(params['single_fraction'], 2)}`；可观测量模式：`{params['observable_mode']}`；局部零syndrome sweep：`{params['zero_sweeps']}`；winding重复：`{params['winding_repeat']}`；起点×replica：`{params['starts']}×{params['reps']}`。"
        )
        lines.append("")
        lines.append("| 环节 | 真实时间(s) | 占比(%) | 尝试次数 | 接受/有效次数 | 接受概率 | 逻辑sector变化次数 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for stage_name in STAGE_LABELS:
            item = group["stage_totals"][stage_name]
            acceptance = "-"
            if item["acceptance_rate"] is not None and stage_name != "observable":
                acceptance = fmt(item["acceptance_rate"], 4)
            accepted_label = integer(item["accepted"])
            if stage_name == "observable":
                accepted_label = "-"
            sector_change_label = (
                "未统计"
                if params["stage_signature_mode"] == "none"
                else integer(item["sector_changes"])
            )
            lines.append(
                "| "
                + " | ".join([
                    STAGE_LABELS[stage_name],
                    fmt(item["wall_time"], 3),
                    pct(item["wall_fraction"]),
                    integer(item["attempted"]),
                    accepted_label,
                    acceptance,
                    sector_change_label,
                ])
                + " |"
            )
        lines.append("")
        if params["pt_num_temperatures"] > 1:
            hot_to_cold_text = (
                "未统计"
                if params["stage_signature_mode"] == "none"
                else str(group["hot_to_cold"])
            )
            lines.append(
                f"- PT交换接受率：平均 `{fmt(group['swap_mean'], 4)}`，最小 `{fmt(group['swap_min'], 4)}`；hot→cold sector delivery `{hot_to_cold_text}`；冷端sector翻转 `{group['cold_flips']}` 次，热端sector翻转 `{group['hot_flips']}` 次。"
            )
        else:
            lines.append(
                f"- 无PT；冷端sector翻转 `{group['cold_flips']}` 次；从未翻过sector的链数 `{group['never_flipped']}`。"
            )
        lines.append("")

    lines.append("## 跨实验对比")
    lines.append("")
    lines.append("| L | 配置 | disorder数 | 总真实时间(s) | 更新环节时间(s) | 诊断签名时间(s) | ESS/秒 | q_top跨链极差 | Rhat最大值 | 主耗时环节 | 主耗时占比(%) | 单比特比例 | 单比特接受概率 | PT交换平均/最小 | hot→cold交付 |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|")
    for group in groups:
        lattice_size, _p_value, config_label = group["key"]
        params = group["params"]
        single = group["stage_totals"]["single_bit"]
        single_acc = (
            single["accepted"] / single["attempted"]
            if single["attempted"]
            else None
        )
        swap_text = "-"
        if group["swap_mean"] is not None:
            swap_text = f"{fmt(group['swap_mean'], 3)} / {fmt(group['swap_min'], 3)}"
        hot_to_cold_text = (
            "未统计"
            if params["stage_signature_mode"] == "none"
            else str(group["hot_to_cold"])
        )
        lines.append(
            "| "
            + " | ".join([
                str(lattice_size),
                CONFIG_LABELS.get(config_label, config_label),
                str(len(group["rows"])),
                fmt(group["total_wall"], 2),
                fmt(group["stage_total_wall"], 2),
                fmt(group["signature_wall"], 2),
                fmt(group["mean_ess_per_total_second"], 3),
                fmt(group["mean_q_top_spread"], 4),
                fmt(group["max_r_hat"], 4),
                STAGE_LABELS[group["dominant_stage"]],
                pct(group["dominant_fraction"]),
                fmt(params["single_fraction"], 2),
                fmt(single_acc, 4),
                swap_text,
                hot_to_cold_text,
            ])
            + " |"
        )
    lines.append("")
    lines.append("## 本轮判断")
    lines.append("")
    lines.append("- cluster 已全部关闭，因此本轮验证的是无 cluster 的成本结构；这避免了旧 profiling 中 cluster 的额外 wall time 和低有效 move 问题。")
    if stage_signature_disabled:
        lines.append("- 本轮使用低诊断开销模式，签名诊断时间为 `0`；总真实时间基本就是更新/观测环节时间。上一轮最慢 task 接近 952s，主要是逐环节 signature probe，而不是真实 MCMC 更新。")
    l5_groups = [group for group in groups if group["key"][0] == 5]
    l5_pt_groups = [
        group for group in l5_groups
        if group["params"]["pt_num_temperatures"] > 1
    ]
    l5_nopt_groups = [
        group for group in l5_groups
        if group["params"]["pt_num_temperatures"] == 1
    ]
    if l5_pt_groups:
        best_l5_pt = min(
            l5_pt_groups,
            key=lambda group: (
                group["mean_q_top_spread"],
                group["max_r_hat"],
                group["total_wall"],
            ),
        )
        params = best_l5_pt["params"]
        lines.append(
            "- L=5 的 PT7 候选里，综合 q_top 跨链极差、Rhat 和机器时间，当前最好的是 "
            f"`{CONFIG_LABELS.get(best_l5_pt['key'][2], best_l5_pt['key'][2])}`："
            f"总真实时间 `{fmt(best_l5_pt['total_wall'], 2)}s`，"
            f"ESS/秒 `{fmt(best_l5_pt['mean_ess_per_total_second'], 3)}`，"
            f"q_top跨链极差 `{fmt(best_l5_pt['mean_q_top_spread'], 4)}`，"
            f"Rhat最大值 `{fmt(best_l5_pt['max_r_hat'], 4)}`，"
            f"单比特比例 `{fmt(params['single_fraction'], 2)}`。"
        )
    if l5_nopt_groups:
        nopt = min(l5_nopt_groups, key=lambda group: group["total_wall"])
        lines.append(
            "- L=5 无PT虽然便宜，但本轮仍不可信："
            f"最快 noPT 组总真实时间 `{fmt(nopt['total_wall'], 2)}s`，"
            f"但 q_top跨链极差 `{fmt(nopt['mean_q_top_spread'], 4)}`，"
            f"Rhat最大值 `{fmt(nopt['max_r_hat'], 4)}`。"
            "它适合快速摸底，不适合作为生产估计。"
        )
    lines.append("- PT swap 本身不是主要耗时项；L=5 PT7 中 PT交换只占几个百分点。真正主耗时在局部零syndrome、单比特和可观测量计算之间切换。")
    lines.append("- 可观测量计算仍然偏贵，尤其 L=5 的 noPT/25% 单比特配置中占比很高。后续加速重点应该是减少可观测量计算频率或缓存/向量化冷端 observable。")
    lines.append("")
    lines.append("## 下一轮计划")
    lines.append("")
    lines.append("1. 以 `PT7 + cluster关 + 5%单比特 + cold-only observable + stage_signature_mode=none` 作为 L=5 的当前默认候选，再做稍长一点的复测确认 Rhat 和 q_top 跨链极差。")
    lines.append("2. 保留 noPT + 10% 单比特作为快速摸底配置，但不要用于正式统计结论。")
    lines.append("3. 下一步优先优化可观测量计算成本；PT swap 和 cluster 暂时不是主要优化对象。")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = build_report(args.run_dir)
    output = args.output or (args.run_dir / "中文_优化profiling_明细表.md")
    output.write_text(report, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
