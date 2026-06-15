"""Summarize exp36 common-disorder A/B NPZ files."""

import argparse
import json
from pathlib import Path

import numpy as np


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serializable"
    )


def _scalar(data, key, default=None, cast=None):
    if key not in data:
        return default
    value = np.asarray(data[key])
    if value.shape == ():
        value = value.item()
    if cast is not None and value is not None:
        return cast(value)
    return value


def _curve_scalar(data, key, default=None):
    if key not in data:
        return default
    arr = np.asarray(data[key], dtype=np.float64)
    if arr.size == 0:
        return default
    return float(arr.reshape(-1)[0])


def _point_disorder_vector(data, key, dtype=np.float64):
    if key not in data:
        return np.asarray([], dtype=dtype)
    arr = np.asarray(data[key], dtype=dtype)
    if arr.size == 0:
        return np.asarray([], dtype=dtype)
    return arr[0, 0, :].copy()


def _point_disorder_blocks(data, key):
    if key not in data:
        return np.asarray([], dtype=np.float64)
    arr = np.asarray(data[key], dtype=np.float64)
    if arr.size == 0:
        return np.asarray([], dtype=np.float64)
    return arr[0, 0, :, :].copy()


def _sum_tensor(data, key):
    if key not in data:
        return 0.0
    arr = np.asarray(data[key], dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.sum(arr))


def _list_floats(values, precision=6):
    return [round(float(value), precision) for value in values]


def _format_float(value, precision=6):
    if value is None:
        return ""
    return f"{float(value):.{precision}g}"


def _format_list(values, precision=6):
    return "[" + ", ".join(_format_float(v, precision) for v in values) + "]"


def summarize_npz(path):
    path = Path(path)
    with np.load(path, allow_pickle=True) as data:
        q_top_by_disorder = _point_disorder_vector(
            data,
            "disorder_q_top_values_tensor",
        )
        block_values = _point_disorder_blocks(
            data,
            "q_top_block_values_per_disorder_tensor",
        )
        block_drift = _point_disorder_vector(
            data,
            "q_top_block_drift_per_disorder_tensor",
        )
        block_range = _point_disorder_vector(
            data,
            "q_top_block_range_per_disorder_tensor",
        )
        last_half = _point_disorder_vector(
            data,
            "q_top_last_half_minus_full_per_disorder_tensor",
        )
        q_top_spread = _point_disorder_vector(
            data,
            "q_top_spread_per_disorder_tensor",
        )
        rhat = _point_disorder_vector(
            data,
            "max_r_hat_per_disorder_tensor",
        )
        ess = _point_disorder_vector(
            data,
            "min_effective_sample_size_per_disorder_tensor",
        )
        wall_time_parts = {
            "ordinary": _sum_tensor(
                data,
                "chain_ordinary_update_wall_time_per_disorder_per_start_replica_tensor",
            ),
            "swap": _sum_tensor(
                data,
                "chain_pt_swap_wall_time_per_disorder_per_start_replica_tensor",
            ),
            "observable": _sum_tensor(
                data,
                "chain_observable_wall_time_per_disorder_per_start_replica_tensor",
            ),
            "measurement": _sum_tensor(
                data,
                "chain_measurement_wall_time_per_disorder_per_start_replica_tensor",
            ),
            "cluster": float(_scalar(data, "cluster_total_wall_time", 0.0)),
        }
        wall_time_total = wall_time_parts["measurement"]
        if wall_time_total <= 0.0:
            wall_time_total = float(
                wall_time_parts["ordinary"]
                + wall_time_parts["swap"]
                + wall_time_parts["observable"]
                + wall_time_parts["cluster"]
            )
        row = {
            "run": path.stem,
            "path": str(path),
            "lattice_size": int(_scalar(data, "lattice_size_list")[0]),
            "p": float(_scalar(data, "data_error_probability_list")[0]),
            "q": float(_scalar(data, "syndrome_error_probability", 0.0)),
            "q_hot": float(_scalar(data, "pt_q_hot", np.nan)),
            "num_temperatures": int(_scalar(data, "pt_num_temperatures", 0)),
            "cluster_rho": float(
                _scalar(data, "cluster_budget_fraction_rho", np.nan)
            ),
            "cluster_enabled": bool(
                _scalar(data, "cluster_update_enabled", False, bool)
            ),
            "cold_edge_stride": int(
                _scalar(data, "pt_cold_edge_swap_stride", 1)
            ),
            "measurements": int(
                _scalar(data, "num_measurements_per_disorder", 0)
            ),
            "num_disorder": int(q_top_by_disorder.size),
            "q_top_mean": float(np.mean(q_top_by_disorder)),
            "q_top_std": float(np.std(q_top_by_disorder, ddof=0)),
            "q_top_by_disorder": _list_floats(q_top_by_disorder),
            "block_values_by_disorder": [
                _list_floats(block_values[index])
                for index in range(block_values.shape[0])
            ],
            "block_drift_by_disorder": _list_floats(block_drift),
            "block_range_by_disorder": _list_floats(block_range),
            "last_half_minus_full_by_disorder": _list_floats(last_half),
            "block_drift_mean": (
                float(np.mean(block_drift)) if block_drift.size else None
            ),
            "block_drift_max_abs": (
                float(np.max(np.abs(block_drift)))
                if block_drift.size
                else None
            ),
            "block_range_max": (
                float(np.max(block_range)) if block_range.size else None
            ),
            "last_half_max_abs": (
                float(np.max(np.abs(last_half))) if last_half.size else None
            ),
            "spread_max": (
                float(np.max(q_top_spread)) if q_top_spread.size else None
            ),
            "rhat_max": float(np.max(rhat)) if rhat.size else None,
            "ess_min": float(np.min(ess)) if ess.size else None,
            "q_top_curve": _curve_scalar(data, "q_top_curve_matrix"),
            "q_top_curve_std": _curve_scalar(data, "q_top_std_curve_matrix"),
            "wall_time_total": wall_time_total,
            "wall_time_parts": wall_time_parts,
        }
    return row


def build_comparisons(rows, reference_run=None):
    if not rows:
        return []
    if reference_run is None:
        reference = rows[-1]
    else:
        matches = [row for row in rows if row["run"] == reference_run]
        if not matches:
            raise ValueError(f"reference run not found: {reference_run}")
        reference = matches[0]
    reference_values = np.asarray(reference["q_top_by_disorder"], dtype=float)
    comparisons = []
    for row in rows:
        values = np.asarray(row["q_top_by_disorder"], dtype=float)
        if values.shape != reference_values.shape:
            continue
        delta = values - reference_values
        comparisons.append(
            {
                "run": row["run"],
                "reference": reference["run"],
                "delta_q_top_by_disorder": _list_floats(delta),
                "mean_delta_q_top": float(np.mean(delta)),
                "mean_abs_delta_q_top": float(np.mean(np.abs(delta))),
                "max_abs_delta_q_top": float(np.max(np.abs(delta))),
                "wall_time_ratio_to_reference": (
                    row["wall_time_total"] / reference["wall_time_total"]
                    if reference["wall_time_total"] > 0
                    else None
                ),
            }
        )
    return comparisons


def write_markdown(summary, output_path):
    lines = [
        f"# {summary['experiment']} summary",
    ]
    if summary.get("status"):
        lines.extend(["", summary["status"]])
    lines.extend(["", "## Runs", ""])
    lines.append(
        "| run | config | q_top mean | q_top by disorder | drift max | "
        "block range max | half max | spread max | Rhat max | ESS min | "
        "wall s | ordinary | swap | observable | cluster |"
    )
    lines.append(
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for row in summary["rows"]:
        config = (
            f"qhot={row['q_hot']:.2f},"
            f"rho={row['cluster_rho']:.2f},"
            f"edge={row['cold_edge_stride']},"
            f"m={row['measurements']},"
            f"cluster={'on' if row['cluster_enabled'] else 'off'}"
        )
        wall = row["wall_time_parts"]
        lines.append(
            "| {run} | {config} | {qmean} | {qvals} | {drift} | "
            "{brange} | {half} | {spread} | {rhat} | {ess} | {wall} | "
            "{ordinary} | {swap} | {observable} | {cluster} |".format(
                run=row["run"],
                config=config,
                qmean=_format_float(row["q_top_mean"]),
                qvals=_format_list(row["q_top_by_disorder"]),
                drift=_format_float(row["block_drift_max_abs"]),
                brange=_format_float(row["block_range_max"]),
                half=_format_float(row["last_half_max_abs"]),
                spread=_format_float(row["spread_max"]),
                rhat=_format_float(row["rhat_max"]),
                ess=_format_float(row["ess_min"], precision=4),
                wall=_format_float(row["wall_time_total"]),
                ordinary=_format_float(wall["ordinary"]),
                swap=_format_float(wall["swap"]),
                observable=_format_float(wall["observable"]),
                cluster=_format_float(wall["cluster"]),
            )
        )
    lines.extend(["", "## Reference Deltas", ""])
    lines.append(
        "| run | reference | delta q_top by disorder | mean abs delta | "
        "max abs delta | wall/reference |"
    )
    lines.append("|---|---|---|---:|---:|---:|")
    for row in summary["comparisons"]:
        lines.append(
            "| {run} | {ref} | {delta} | {mean_abs} | {max_abs} | "
            "{wall_ratio} |".format(
                run=row["run"],
                ref=row["reference"],
                delta=_format_list(row["delta_q_top_by_disorder"]),
                mean_abs=_format_float(row["mean_abs_delta_q_top"]),
                max_abs=_format_float(row["max_abs_delta_q_top"]),
                wall_ratio=_format_float(
                    row["wall_time_ratio_to_reference"]
                ),
            )
        )
    if summary.get("conclusion"):
        lines.extend(["", "## Conclusion", ""])
        for item in summary["conclusion"]:
            lines.append(f"- {item}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--status", default=None)
    parser.add_argument("--reference-run", default=None)
    parser.add_argument("--conclusion", action="append", default=[])
    parser.add_argument("npz", nargs="+")
    args = parser.parse_args(argv)

    rows = [summarize_npz(path) for path in args.npz]
    summary = {
        "experiment": args.experiment,
        "status": args.status,
        "rows": rows,
        "comparisons": build_comparisons(
            rows,
            reference_run=args.reference_run,
        ),
        "conclusion": args.conclusion,
    }
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    write_markdown(summary, output_md)


if __name__ == "__main__":
    main()
