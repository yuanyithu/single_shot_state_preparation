import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _as_scalar(value):
    array = np.asarray(value)
    if array.shape == ():
        return array.item()
    return value


def _float_or_none(value):
    value = float(value)
    if math.isfinite(value):
        return value
    return None


def _format_probability(value):
    return f"{float(value):0.4f}"


def _format_probability_tag(value):
    return _format_probability(value).replace(".", "p")


def _load_npz(path):
    with np.load(path, allow_pickle=True) as loaded:
        return {key: loaded[key] for key in loaded.files}


def _is_candidate_npz(path):
    parts = set(path.parts)
    if "chunks" in parts or "preflight" in parts:
        return False
    return path.suffix == ".npz"


def _adaptive_arrays(result):
    if "adaptive_pt_round_f_mono_tensor" in result:
        return {
            "f_raw": np.asarray(result["adaptive_pt_round_f_raw_tensor"], dtype=float),
            "f_mono": np.asarray(result["adaptive_pt_round_f_mono_tensor"], dtype=float),
            "f_target": np.asarray(
                result["adaptive_pt_round_f_target_tensor"],
                dtype=float,
            ),
            "completed": np.asarray(
                result["adaptive_pt_num_rounds_completed_per_disorder_tensor"],
                dtype=np.int64,
            ),
        }
    if "adaptive_pt_round_f_mono_per_disorder" in result:
        return {
            "f_raw": np.asarray(
                result["adaptive_pt_round_f_raw_per_disorder"],
                dtype=float,
            )[None, None, ...],
            "f_mono": np.asarray(
                result["adaptive_pt_round_f_mono_per_disorder"],
                dtype=float,
            )[None, None, ...],
            "f_target": np.asarray(
                result["adaptive_pt_round_f_target_per_disorder"],
                dtype=float,
            )[None, None, ...],
            "completed": np.asarray(
                result["adaptive_pt_num_rounds_completed_per_disorder"],
                dtype=np.int64,
            )[None, None, ...],
        }
    return None


def _first_value(result, key, default=np.nan):
    if key not in result:
        return default
    array = np.asarray(result[key], dtype=float)
    if array.size == 0:
        return default
    return float(array.reshape(-1)[0])


def _entry_from_npz(path):
    result = _load_npz(path)
    adaptive = _adaptive_arrays(result)
    if adaptive is None:
        return None
    lattice_size = int(np.asarray(result["lattice_size_list"]).reshape(-1)[0])
    p_value = float(np.asarray(result["data_error_probability_list"]).reshape(-1)[0])
    q_value = float(_as_scalar(result["syndrome_error_probability"]))
    adaptive_rounds = int(_as_scalar(result.get("adaptive_pt_rounds", 0)))
    f_mono = adaptive["f_mono"]
    f_target = adaptive["f_target"]
    final_round_index = max(0, adaptive_rounds - 1)
    if f_mono.shape[-2] <= final_round_index:
        final_round_index = f_mono.shape[-2] - 1
    final_delta = np.abs(f_mono[..., final_round_index, :] - f_target[..., final_round_index, :])
    final_delta = final_delta[np.isfinite(final_delta)]
    completed = adaptive["completed"]

    ordinary_time = np.asarray(
        result.get(
            "chain_ordinary_update_wall_time_per_disorder_per_start_replica_tensor",
            np.asarray([], dtype=float),
        ),
        dtype=float,
    )
    pt_swap_time = np.asarray(
        result.get(
            "chain_pt_swap_wall_time_per_disorder_per_start_replica_tensor",
            np.asarray([], dtype=float),
        ),
        dtype=float,
    )
    observable_time = np.asarray(
        result.get(
            "chain_observable_wall_time_per_disorder_per_start_replica_tensor",
            np.asarray([], dtype=float),
        ),
        dtype=float,
    )
    if ordinary_time.size and pt_swap_time.size and observable_time.size:
        local_denominator = ordinary_time + pt_swap_time + observable_time
        local_mask = local_denominator > 0
        if np.any(local_mask):
            local_update_wall_fraction = float(
                np.mean(ordinary_time[local_mask] / local_denominator[local_mask])
            )
        else:
            local_update_wall_fraction = math.nan
    else:
        local_update_wall_fraction = math.nan

    acceptance = np.asarray(
        result.get("average_acceptance_rate_per_disorder_tensor", np.asarray([], dtype=float)),
        dtype=float,
    )
    local_acceptance_rate = (
        math.nan if acceptance.size == 0 else float(np.nanmean(acceptance))
    )

    return {
        "path": str(path),
        "lattice_size": lattice_size,
        "p": p_value,
        "q": q_value,
        "adaptive_pt_rounds": adaptive_rounds,
        "num_disorder_samples": int(_as_scalar(result["num_disorder_samples"])),
        "pt_num_temperatures": int(_as_scalar(result.get("pt_num_temperatures", f_mono.shape[-1]))),
        "pt_ladder_mode": str(_as_scalar(result.get("pt_ladder_mode", "unknown"))),
        "pt_q_hot": _float_or_none(_as_scalar(result.get("pt_q_hot", np.nan))),
        "mean_abs_final_flow_error": (
            None if final_delta.size == 0 else float(np.mean(final_delta))
        ),
        "max_abs_final_flow_error": (
            None if final_delta.size == 0 else float(np.max(final_delta))
        ),
        "mean_completed_rounds": float(np.mean(completed)),
        "local_update_wall_fraction": _float_or_none(local_update_wall_fraction),
        "local_acceptance_rate": _float_or_none(local_acceptance_rate),
        "mean_pt_min_swap_acceptance_rate": _float_or_none(
            _first_value(result, "mean_pt_min_swap_acceptance_rate_curve_matrix")
        ),
        "mean_pt_mean_swap_acceptance_rate": _float_or_none(
            _first_value(result, "mean_pt_mean_swap_acceptance_rate_curve_matrix")
        ),
        "f_raw": adaptive["f_raw"],
        "f_mono": f_mono,
        "f_target": f_target,
    }


def _finite_mean(values):
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return None
    return float(np.mean(finite))


def _aggregate_rows(entries):
    rows = []
    for rounds in sorted({entry["adaptive_pt_rounds"] for entry in entries}):
        group = [entry for entry in entries if entry["adaptive_pt_rounds"] == rounds]
        rows.append({
            "adaptive_pt_rounds": rounds,
            "num_npz": len(group),
            "num_disorder_samples": int(sum(entry["num_disorder_samples"] for entry in group)),
            "mean_abs_final_flow_error": _finite_mean(
                entry["mean_abs_final_flow_error"] for entry in group
            ),
            "max_abs_final_flow_error": max(
                (
                    float(entry["max_abs_final_flow_error"])
                    for entry in group
                    if entry["max_abs_final_flow_error"] is not None
                ),
                default=None,
            ),
            "local_update_wall_fraction": _finite_mean(
                entry["local_update_wall_fraction"] for entry in group
            ),
            "local_acceptance_rate": _finite_mean(
                entry["local_acceptance_rate"] for entry in group
            ),
            "mean_pt_min_swap_acceptance_rate": _finite_mean(
                entry["mean_pt_min_swap_acceptance_rate"] for entry in group
            ),
            "mean_pt_mean_swap_acceptance_rate": _finite_mean(
                entry["mean_pt_mean_swap_acceptance_rate"] for entry in group
            ),
        })
    return rows


def _mean_final_f(entry):
    rounds = max(1, int(entry["adaptive_pt_rounds"]))
    round_index = min(rounds - 1, entry["f_mono"].shape[-2] - 1)
    f_mono = entry["f_mono"][..., round_index, :]
    f_target = entry["f_target"][..., round_index, :]
    return (
        np.nanmean(f_mono.reshape(-1, f_mono.shape[-1]), axis=0),
        np.nanmean(f_target.reshape(-1, f_target.shape[-1]), axis=0),
    )


def _write_overall_f_plot(entries, output_dir):
    if not entries:
        return None
    rounds_values = sorted({entry["adaptive_pt_rounds"] for entry in entries})
    fig, axes = plt.subplots(
        1,
        len(rounds_values),
        figsize=(4.8 * len(rounds_values), 3.6),
        squeeze=False,
        sharey=True,
    )
    for axis, rounds in zip(axes[0], rounds_values):
        group = [entry for entry in entries if entry["adaptive_pt_rounds"] == rounds]
        f_values = []
        target_values = []
        for entry in group:
            f_mean, target_mean = _mean_final_f(entry)
            f_values.append(f_mean)
            target_values.append(target_mean)
        f_mean = np.nanmean(np.asarray(f_values, dtype=float), axis=0)
        target_mean = np.nanmean(np.asarray(target_values, dtype=float), axis=0)
        indices = np.arange(f_mean.shape[0])
        axis.bar(indices, f_mean, color="#4c78a8", width=0.75, label="f")
        axis.plot(indices, target_mean, color="#f58518", marker="o", label="target")
        axis.set_title(f"{rounds} rounds")
        axis.set_xlabel("temperature index")
        axis.set_ylim(-0.05, 1.05)
        axis.grid(axis="y", alpha=0.25)
    axes[0][0].set_ylabel("mean flow f")
    axes[0][-1].legend(loc="best")
    fig.suptitle("exp35 adaptive PT final flow comparison")
    fig.tight_layout()
    output_path = output_dir / "adaptive_pt_f_final_overall.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _write_point_plots(entries, output_dir, max_plots):
    paths = []
    grouped = {}
    for entry in entries:
        key = (entry["lattice_size"], round(entry["q"], 10))
        grouped.setdefault(key, []).append(entry)
    for index, ((lattice_size, q_value), group) in enumerate(sorted(grouped.items())):
        if index >= max_plots:
            break
        rounds_values = sorted({entry["adaptive_pt_rounds"] for entry in group})
        fig, axes = plt.subplots(
            1,
            len(rounds_values),
            figsize=(4.8 * len(rounds_values), 3.6),
            squeeze=False,
            sharey=True,
        )
        for axis, rounds in zip(axes[0], rounds_values):
            round_group = [
                entry for entry in group if entry["adaptive_pt_rounds"] == rounds
            ]
            f_values = []
            target_values = []
            for entry in round_group:
                f_mean, target_mean = _mean_final_f(entry)
                f_values.append(f_mean)
                target_values.append(target_mean)
            f_mean = np.nanmean(np.asarray(f_values, dtype=float), axis=0)
            target_mean = np.nanmean(np.asarray(target_values, dtype=float), axis=0)
            indices = np.arange(f_mean.shape[0])
            axis.bar(indices, f_mean, color="#4c78a8", width=0.75, label="f")
            axis.plot(indices, target_mean, color="#f58518", marker="o", label="target")
            axis.set_title(f"{rounds} rounds")
            axis.set_xlabel("temperature index")
            axis.set_ylim(-0.05, 1.05)
            axis.grid(axis="y", alpha=0.25)
        axes[0][0].set_ylabel("mean flow f")
        axes[0][-1].legend(loc="best")
        fig.suptitle(f"L={lattice_size}, q={_format_probability(q_value)}")
        fig.tight_layout()
        output_path = (
            output_dir
            / f"adaptive_pt_f_L{lattice_size}_q{_format_probability_tag(q_value)}.png"
        )
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        paths.append(output_path)
    return paths


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _format_optional(value, precision=4):
    if value is None:
        return "NA"
    return f"{float(value):.{precision}f}"


def _parameter_guidance(rows):
    if not rows:
        return "当前没有足够的 adaptive PT 结果，暂不建议进入生产参数选择。"
    by_round = {int(row["adaptive_pt_rounds"]): row for row in rows}
    if 1 not in by_round:
        return "当前 pilot 缺少 1 轮基线；先补齐 1/3/5 轮对比再定生产轮数。"
    if 3 not in by_round:
        return "当前 pilot 缺少 3 轮结果；建议至少补 3 轮后再定生产轮数。"
    err1 = by_round[1]["mean_abs_final_flow_error"]
    err3 = by_round[3]["mean_abs_final_flow_error"]
    err5 = by_round.get(5, {}).get("mean_abs_final_flow_error")
    if err1 is None or err3 is None:
        return "f 误差存在缺失值；先检查 adaptive flow 观测是否太短或 swap 是否过低。"
    improvement_13 = err1 - err3
    if err5 is None:
        if improvement_13 > 0.02:
            return "3 轮相对 1 轮已有可见改善；生产可先用 3 轮，同时补 5 轮硬点验证。"
        return "1 到 3 轮改善不明显；若 swap 和 local update 比率正常，生产可用 1-3 轮。"
    improvement_35 = err3 - err5
    if improvement_13 > 0.02 and improvement_35 <= max(0.01, 0.25 * abs(improvement_13)):
        return "3 轮已经吸收主要收益，5 轮边际收益小；生产建议默认 3 轮。"
    if improvement_35 > 0.02:
        return "5 轮仍显著改善 f；高 q 或大 L 点建议用 5 轮，其他点至少 3 轮。"
    return "1/3/5 轮差别较小；生产可用 3 轮作为稳健默认。"


def _write_markdown(path, entries, rows, overall_plot, point_plots, allow_partial):
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# exp35 adaptive PT 参数搜索汇总\n\n")
        handle.write(f"- 输入 NPZ 数量: {len(entries)}\n")
        handle.write(f"- allow_partial: {bool(allow_partial)}\n")
        if entries:
            qs = sorted({_format_probability(entry["q"]) for entry in entries})
            ls = sorted({entry["lattice_size"] for entry in entries})
            handle.write(f"- 覆盖尺寸 L: {ls}\n")
            handle.write(f"- 覆盖 q: {', '.join(qs)}\n")
        handle.write("\n## 轮数汇总\n\n")
        handle.write(
            "| adaptive轮数 | NPZ数 | disorder数 | mean | max | local墙时比 | local接受率 | min swap | mean swap |\n"
        )
        handle.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            handle.write(
                f"| {row['adaptive_pt_rounds']} | {row['num_npz']} | "
                f"{row['num_disorder_samples']} | "
                f"{_format_optional(row['mean_abs_final_flow_error'])} | "
                f"{_format_optional(row['max_abs_final_flow_error'])} | "
                f"{_format_optional(row['local_update_wall_fraction'])} | "
                f"{_format_optional(row['local_acceptance_rate'])} | "
                f"{_format_optional(row['mean_pt_min_swap_acceptance_rate'])} | "
                f"{_format_optional(row['mean_pt_mean_swap_acceptance_rate'])} |\n"
            )
        handle.write("\n## 参数指导\n\n")
        handle.write(_parameter_guidance(rows))
        handle.write("\n\n## 图像\n\n")
        if overall_plot is not None:
            handle.write(f"- 总体 f 柱状图: `{overall_plot.name}`\n")
        for plot_path in point_plots:
            handle.write(f"- 典型点 f 柱状图: `{plot_path.name}`\n")
        handle.write("\n## 说明\n\n")
        handle.write(
            "- mean/max 是最终一轮 monotone flow `f_mono` 到线性目标 `f_target` 的绝对误差。\n"
        )
        handle.write(
            "- local墙时比按 ordinary local update / (ordinary + PT swap + observable) 估算。\n"
        )
        handle.write(
            "- local接受率使用冷端链的平均总接受率，包含 single-bit 与 zero-syndrome local 更新。\n"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Summarize exp35 adaptive PT flow diagnostics in Chinese.",
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--max-point-plots", type=int, default=24)
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    errors = []
    for path in sorted(input_dir.rglob("*.npz")):
        if not _is_candidate_npz(path):
            continue
        try:
            entry = _entry_from_npz(path)
        except Exception as exc:
            if not args.allow_partial:
                raise
            errors.append({"path": str(path), "error": repr(exc)})
            continue
        if entry is not None:
            entries.append(entry)

    if not entries and not args.allow_partial:
        raise RuntimeError("No adaptive PT NPZ results found.")

    public_entries = []
    for entry in entries:
        public_entry = dict(entry)
        public_entry.pop("f_raw", None)
        public_entry.pop("f_mono", None)
        public_entry.pop("f_target", None)
        public_entries.append(public_entry)

    rows = _aggregate_rows(entries)
    overall_plot = _write_overall_f_plot(entries, output_dir) if entries else None
    point_plots = (
        _write_point_plots(entries, output_dir, args.max_point_plots)
        if entries
        else []
    )

    json_path = output_dir / "adaptive_pt_summary.json"
    csv_path = output_dir / "adaptive_pt_summary.csv"
    md_path = output_dir / "adaptive_pt_summary.md"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "entries": public_entries,
                "by_round": rows,
                "errors": errors,
                "guidance": _parameter_guidance(rows),
                "overall_plot": None if overall_plot is None else str(overall_plot),
                "point_plots": [str(path) for path in point_plots],
            },
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    if rows:
        _write_csv(csv_path, rows)
    _write_markdown(md_path, entries, rows, overall_plot, point_plots, args.allow_partial)

    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
