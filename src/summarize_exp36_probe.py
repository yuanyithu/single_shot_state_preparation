"""Summarize exp36 PT/logical-sector probe NPZ files."""

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
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _scalar(data, key, default=None, cast=None):
    if key not in data:
        return default
    value = data[key]
    if np.asarray(value).shape == ():
        value = value.item()
    if cast is not None and value is not None:
        return cast(value)
    return value


def _array(data, key, default_shape=None, dtype=np.int64):
    if key in data:
        return np.asarray(data[key])
    if default_shape is None:
        return np.asarray([], dtype=dtype)
    return np.zeros(default_shape, dtype=dtype)


def _chain_temperature_tensor(data, key):
    arr = _array(data, key)
    if arr.size == 0:
        return arr
    return arr[0, 0, 0, :, 0, :]


def _chain_scalar_tensor(data, key):
    arr = _array(data, key)
    if arr.size == 0:
        return arr
    return arr[0, 0, 0, :, 0]


def _first_point_scalar(data, key, default=None, cast=float):
    if key not in data:
        return default
    arr = np.asarray(data[key])
    if arr.size == 0:
        return default
    value = arr.reshape(-1)[0]
    if cast is None:
        return value
    return cast(value)


def _first_disorder_scalar(data, key, default=None, cast=float):
    if key not in data:
        return default
    arr = np.asarray(data[key])
    if arr.size == 0:
        return default
    value = arr[0, 0, 0]
    if cast is None:
        return value
    return cast(value)


def _chain_block_tensor(data, key):
    arr = _array(data, key, dtype=np.float64)
    if arr.size == 0:
        return arr
    return arr[0, 0, 0, :, 0, :]


def _disorder_block_vector(data, key):
    arr = _array(data, key, dtype=np.float64)
    if arr.size == 0:
        return arr
    return arr[0, 0, 0, :]


def _sum_key(data, key):
    return int(np.sum(_chain_temperature_tensor(data, key)))


def _sum_vector_key(data, key):
    arr = _chain_temperature_tensor(data, key)
    if arr.size == 0:
        return []
    return np.sum(arr, axis=0).astype(int).tolist()


def summarize_npz(path, run_name=None):
    path = Path(path)
    with np.load(path, allow_pickle=True) as data:
        swap_rates = _chain_temperature_tensor(
            data,
            "chain_pt_swap_acceptance_rate_per_pair_per_disorder_per_start_replica_tensor",
        )
        if swap_rates.size == 0:
            min_swap = None
            bottleneck_pair = None
        else:
            mean_swap = np.mean(swap_rates, axis=0)
            bottleneck_pair = int(np.argmin(mean_swap))
            min_swap = float(mean_swap[bottleneck_pair])

        sector_flips = _chain_temperature_tensor(
            data,
            "chain_pt_sector_flip_count_per_temperature_per_disorder_per_start_replica_tensor",
        )
        if sector_flips.size == 0:
            cold_flips = []
            hot_flips_mean = None
        else:
            cold_flips = sector_flips[:, 0].astype(int).tolist()
            hot_flips_mean = float(np.mean(sector_flips[:, -1]))

        roundtrip = _chain_temperature_tensor(
            data,
            "chain_pt_replica_endpoint_round_trip_count_per_disorder_per_start_replica_tensor",
        )
        if roundtrip.size == 0:
            roundtrip_sum = 0
            roundtrip_per_chain = []
        else:
            roundtrip_sum = int(np.sum(roundtrip))
            roundtrip_per_chain = np.sum(roundtrip, axis=1).astype(int).tolist()

        cluster_nonzero = _scalar(data, "cluster_num_nonzero_moves", 0, int)
        if cluster_nonzero == 0 and "cluster_by_temperature_nonzero_moves" in data:
            cluster_nonzero = int(np.sum(data["cluster_by_temperature_nonzero_moves"]))

        chain_q_top_values = _chain_scalar_tensor(
            data,
            "chain_q_top_values_per_disorder_per_start_replica_tensor",
        )
        chain_ordinary_wall = _chain_scalar_tensor(
            data,
            "chain_ordinary_update_wall_time_per_disorder_per_start_replica_tensor",
        )
        chain_pt_swap_wall = _chain_scalar_tensor(
            data,
            "chain_pt_swap_wall_time_per_disorder_per_start_replica_tensor",
        )
        chain_observable_wall = _chain_scalar_tensor(
            data,
            "chain_observable_wall_time_per_disorder_per_start_replica_tensor",
        )
        chain_measurement_wall = _chain_scalar_tensor(
            data,
            "chain_measurement_wall_time_per_disorder_per_start_replica_tensor",
        )
        q_top_block_values = _disorder_block_vector(
            data,
            "q_top_block_values_per_disorder_tensor",
        )
        chain_q_top_block_values = _chain_block_tensor(
            data,
            "chain_q_top_block_values_per_disorder_per_start_replica_tensor",
        )
        chain_q_top_block_drift = _chain_scalar_tensor(
            data,
            "chain_q_top_block_drift_per_disorder_per_start_replica_tensor",
        )
        chain_q_top_block_range = _chain_scalar_tensor(
            data,
            "chain_q_top_block_range_per_disorder_per_start_replica_tensor",
        )
        chain_q_top_last_half_minus_full = _chain_scalar_tensor(
            data,
            "chain_q_top_last_half_minus_full_per_disorder_per_start_replica_tensor",
        )
        q_top_spread = _first_disorder_scalar(
            data,
            "q_top_spread_per_disorder_tensor",
        )
        max_r_hat = _first_point_scalar(data, "max_r_hat_curve_matrix")
        min_ess = _first_point_scalar(
            data,
            "min_effective_sample_size_curve_matrix",
        )
        target_gate_pass = None
        if q_top_spread is not None and max_r_hat is not None and min_ess is not None:
            target_gate_pass = bool(
                np.isfinite(max_r_hat)
                and max_r_hat <= 1.05
                and np.isfinite(min_ess)
                and min_ess >= 100.0
                and q_top_spread <= 0.02
            )

        row = {
            "run": run_name or path.parent.name,
            "path": str(path),
            "lattice_size": int(np.asarray(data.get("lattice_size_list", [0]))[0]),
            "p": float(np.asarray(data.get("data_error_probability_list", [np.nan]))[0]),
            "q": _scalar(data, "syndrome_error_probability", None, float),
            "q_hot": _scalar(data, "pt_q_hot", None, float),
            "num_temperatures": _scalar(data, "pt_num_temperatures", None, int),
            "num_start_chains": _scalar(data, "num_start_chains", None, int),
            "measurements": _scalar(data, "num_measurements_per_disorder", None, int),
            "sweeps_between_measurements": _scalar(
                data,
                "num_sweeps_between_measurements",
                None,
                int,
            ),
            "pt_sector_diagnostic_stride": _scalar(
                data,
                "pt_sector_diagnostic_stride",
                None,
                int,
            ),
            "pt_swap_attempt_every_num_sweeps": _scalar(
                data,
                "pt_swap_attempt_every_num_sweeps",
                None,
                int,
            ),
            "pt_swap_sweeps_per_attempt": _scalar(
                data,
                "pt_swap_sweeps_per_attempt",
                None,
                int,
            ),
            "pt_cold_edge_swap_stride": _scalar(
                data,
                "pt_cold_edge_swap_stride",
                1,
                int,
            ),
            "cluster_budget_fraction_rho": _scalar(
                data,
                "cluster_budget_fraction_rho",
                None,
                float,
            ),
            "q_top": _first_point_scalar(data, "q_top_curve_matrix"),
            "q_top_std": _first_point_scalar(data, "q_top_std_curve_matrix"),
            "chain_q_top_values": (
                chain_q_top_values.astype(float).tolist()
                if chain_q_top_values.size
                else []
            ),
            "q_top_spread": q_top_spread,
            "mean_q_top_spread": _first_point_scalar(
                data,
                "mean_q_top_spread_curve_matrix",
            ),
            "max_r_hat": max_r_hat,
            "min_effective_sample_size": min_ess,
            "target_gate_pass": target_gate_pass,
            "ordinary_wall_time_sum": (
                float(np.sum(chain_ordinary_wall))
                if chain_ordinary_wall.size
                else None
            ),
            "pt_swap_wall_time_sum": (
                float(np.sum(chain_pt_swap_wall))
                if chain_pt_swap_wall.size
                else None
            ),
            "observable_wall_time_sum": (
                float(np.sum(chain_observable_wall))
                if chain_observable_wall.size
                else None
            ),
            "measurement_wall_time_sum": (
                float(np.sum(chain_measurement_wall))
                if chain_measurement_wall.size
                else None
            ),
            "cluster_wall_time_sum": _scalar(
                data,
                "cluster_total_wall_time",
                0.0,
                float,
            ),
            "q_top_block_count": _scalar(data, "q_top_block_count", 0, int),
            "q_top_block_values": (
                q_top_block_values.astype(float).tolist()
                if q_top_block_values.size
                else []
            ),
            "q_top_block_drift": _first_disorder_scalar(
                data,
                "q_top_block_drift_per_disorder_tensor",
            ),
            "q_top_block_range": _first_disorder_scalar(
                data,
                "q_top_block_range_per_disorder_tensor",
            ),
            "q_top_last_half_minus_full": _first_disorder_scalar(
                data,
                "q_top_last_half_minus_full_per_disorder_tensor",
            ),
            "chain_q_top_block_values": (
                chain_q_top_block_values.astype(float).tolist()
                if chain_q_top_block_values.size
                else []
            ),
            "chain_q_top_block_drift": (
                chain_q_top_block_drift.astype(float).tolist()
                if chain_q_top_block_drift.size
                else []
            ),
            "chain_q_top_block_range": (
                chain_q_top_block_range.astype(float).tolist()
                if chain_q_top_block_range.size
                else []
            ),
            "chain_q_top_last_half_minus_full": (
                chain_q_top_last_half_minus_full.astype(float).tolist()
                if chain_q_top_last_half_minus_full.size
                else []
            ),
            "min_swap": min_swap,
            "bottleneck_pair": bottleneck_pair,
            "cold_flips": cold_flips,
            "hot_flips_mean": hot_flips_mean,
            "roundtrip_sum": roundtrip_sum,
            "roundtrip_per_chain": roundtrip_per_chain,
            "transport_samples_per_chain": _chain_scalar_tensor(
                data,
                "chain_pt_transport_position_sample_count_per_disorder_per_start_replica_tensor",
            ).astype(int).tolist(),
            "cluster_attempts": _scalar(data, "cluster_num_attempts", 0, int),
            "cluster_nonzero": int(cluster_nonzero),
            "changed": _sum_key(
                data,
                "chain_pt_cluster_sector_changed_count_per_temperature_per_disorder_per_start_replica_tensor",
            ),
            "arrival": _sum_key(
                data,
                "chain_pt_cluster_sector_cold_arrival_count_per_origin_temperature_per_disorder_per_start_replica_tensor",
            ),
            "arrival_survived": _sum_key(
                data,
                "chain_pt_cluster_sector_cold_survived_count_per_origin_temperature_per_disorder_per_start_replica_tensor",
            ),
            "arrival_reverted": _sum_key(
                data,
                "chain_pt_cluster_sector_cold_reverted_count_per_origin_temperature_per_disorder_per_start_replica_tensor",
            ),
            "arrival_other": _sum_key(
                data,
                "chain_pt_cluster_sector_cold_other_count_per_origin_temperature_per_disorder_per_start_replica_tensor",
            ),
            "diagnostic_survived": _sum_key(
                data,
                "chain_pt_cluster_sector_cold_diagnostic_survived_count_per_origin_temperature_per_disorder_per_start_replica_tensor",
            ),
            "diagnostic_reverted": _sum_key(
                data,
                "chain_pt_cluster_sector_cold_diagnostic_reverted_count_per_origin_temperature_per_disorder_per_start_replica_tensor",
            ),
            "diagnostic_other": _sum_key(
                data,
                "chain_pt_cluster_sector_cold_diagnostic_other_count_per_origin_temperature_per_disorder_per_start_replica_tensor",
            ),
            "diagnostic_missed": _sum_key(
                data,
                "chain_pt_cluster_sector_cold_diagnostic_missed_count_per_origin_temperature_per_disorder_per_start_replica_tensor",
            ),
            "departure_survived": _sum_key(
                data,
                "chain_pt_cluster_sector_cold_departure_survived_count_per_origin_temperature_per_disorder_per_start_replica_tensor",
            ),
            "departure_reverted": _sum_key(
                data,
                "chain_pt_cluster_sector_cold_departure_reverted_count_per_origin_temperature_per_disorder_per_start_replica_tensor",
            ),
            "departure_other": _sum_key(
                data,
                "chain_pt_cluster_sector_cold_departure_other_count_per_origin_temperature_per_disorder_per_start_replica_tensor",
            ),
            "dwell_sum": _sum_key(
                data,
                "chain_pt_cluster_sector_cold_dwell_sample_sum_per_origin_temperature_per_disorder_per_start_replica_tensor",
            ),
            "dwell_max": int(np.max(_chain_temperature_tensor(
                data,
                "chain_pt_cluster_sector_cold_dwell_sample_max_per_origin_temperature_per_disorder_per_start_replica_tensor",
            ))) if _chain_temperature_tensor(
                data,
                "chain_pt_cluster_sector_cold_dwell_sample_max_per_origin_temperature_per_disorder_per_start_replica_tensor",
            ).size else 0,
            "changed_by_temperature": _sum_vector_key(
                data,
                "chain_pt_cluster_sector_changed_count_per_temperature_per_disorder_per_start_replica_tensor",
            ),
            "arrival_by_origin": _sum_vector_key(
                data,
                "chain_pt_cluster_sector_cold_arrival_count_per_origin_temperature_per_disorder_per_start_replica_tensor",
            ),
            "diagnostic_missed_by_origin": _sum_vector_key(
                data,
                "chain_pt_cluster_sector_cold_diagnostic_missed_count_per_origin_temperature_per_disorder_per_start_replica_tensor",
            ),
            "departure_survived_by_origin": _sum_vector_key(
                data,
                "chain_pt_cluster_sector_cold_departure_survived_count_per_origin_temperature_per_disorder_per_start_replica_tensor",
            ),
            "departure_reverted_by_origin": _sum_vector_key(
                data,
                "chain_pt_cluster_sector_cold_departure_reverted_count_per_origin_temperature_per_disorder_per_start_replica_tensor",
            ),
            "departure_other_by_origin": _sum_vector_key(
                data,
                "chain_pt_cluster_sector_cold_departure_other_count_per_origin_temperature_per_disorder_per_start_replica_tensor",
            ),
        }
    return row


def _format_float(value):
    if value is None:
        return ""
    try:
        if not np.isfinite(float(value)):
            return ""
    except (TypeError, ValueError):
        return ""
    return f"{float(value):.6f}"


def _format_compact_float(value):
    if value is None:
        return ""
    try:
        if not np.isfinite(float(value)):
            return ""
    except (TypeError, ValueError):
        return ""
    return f"{float(value):.4g}"


def _format_float_list(values, max_items=8):
    if not values:
        return "[]"
    formatted_values = [
        _format_compact_float(value)
        for value in list(values)[:max_items]
    ]
    if len(values) > max_items:
        formatted_values.append("...")
    return "[" + ", ".join(formatted_values) + "]"


def _format_gate(value):
    if value is None:
        return ""
    return "pass" if value else "fail"


def build_summary(experiment, rows, status=None, conclusion=None):
    common = {}
    if rows:
        first = rows[0]
        for key in (
                "lattice_size", "p", "q", "q_hot", "num_temperatures",
                "num_start_chains", "cluster_budget_fraction_rho"):
            common[key] = first.get(key)
    summary = {
        "experiment": experiment,
        "status": status,
        "common_parameters": common,
        "rows": rows,
    }
    if conclusion:
        summary["conclusion"] = conclusion
    return summary


def write_markdown(summary, output_path):
    lines = [f"# {summary['experiment']} summary", ""]
    if summary.get("status"):
        lines.append(summary["status"])
        lines.append("")
    common = summary.get("common_parameters") or {}
    if common:
        parts = [f"{key}={value}" for key, value in common.items()]
        lines.append("共同参数：`" + ",".join(parts) + "`。")
        lines.append("")
    lines.append("## 目标指标")
    lines.append("")
    target_header = (
        "| run | q_top | chain q_top | spread | Rhat | ESS | gate | "
        "block q_top | block range | last-half-full | wall s | ordinary | "
        "swap | observable | cluster |"
    )
    lines.append(target_header)
    lines.append("|---|---:|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary["rows"]:
        lines.append(
            "| {run} | {q_top} | {chains} | {spread} | {rhat} | {ess} | {gate} | {blocks} | {block_range} | {half} | {wall} | {ordinary} | {swap} | {observable} | {cluster} |".format(
                run=row["run"],
                q_top=_format_float(row.get("q_top")),
                chains=_format_float_list(row.get("chain_q_top_values", [])),
                spread=_format_float(row.get("q_top_spread")),
                rhat=_format_float(row.get("max_r_hat")),
                ess=_format_compact_float(
                    row.get("min_effective_sample_size")
                ),
                gate=_format_gate(row.get("target_gate_pass")),
                blocks=_format_float_list(row.get("q_top_block_values", [])),
                block_range=_format_float(row.get("q_top_block_range")),
                half=_format_float(
                    row.get("q_top_last_half_minus_full")
                ),
                wall=_format_compact_float(
                    row.get("measurement_wall_time_sum")
                ),
                ordinary=_format_compact_float(
                    row.get("ordinary_wall_time_sum")
                ),
                swap=_format_compact_float(row.get("pt_swap_wall_time_sum")),
                observable=_format_compact_float(
                    row.get("observable_wall_time_sum")
                ),
                cluster=_format_compact_float(
                    row.get("cluster_wall_time_sum")
                ),
            )
        )
    lines.append("")
    lines.append("## 解释指标")
    lines.append("")
    header = (
        "| run | cold edge | swap every | sweeps/meas | m | stride | min swap | "
        "cold flips | roundtrip | changed | arrival | arr survived | arr reverted | "
        "diag survived | diag missed | dep survived | dep reverted | dep other | dwell sum/max |"
    )
    lines.append(header)
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in summary["rows"]:
        lines.append(
            "| {run} | {cold_edge} | {swap_every} | {sweeps} | {m} | {stride} | {min_swap} | {cold} | {rt} | {changed} | {arrival} | {arr_s} | {arr_r} | {diag_s} | {diag_m} | {dep_s} | {dep_r} | {dep_o} | {dwell}/{dwell_max} |".format(
                run=row["run"],
                cold_edge=row.get("pt_cold_edge_swap_stride", 1),
                swap_every=row.get("pt_swap_attempt_every_num_sweeps"),
                sweeps=row.get("sweeps_between_measurements"),
                m=row.get("measurements"),
                stride=row.get("pt_sector_diagnostic_stride"),
                min_swap=_format_float(row.get("min_swap")),
                cold=row.get("cold_flips"),
                rt=row.get("roundtrip_sum"),
                changed=row.get("changed"),
                arrival=row.get("arrival"),
                arr_s=row.get("arrival_survived"),
                arr_r=row.get("arrival_reverted"),
                diag_s=row.get("diagnostic_survived"),
                diag_m=row.get("diagnostic_missed"),
                dep_s=row.get("departure_survived"),
                dep_r=row.get("departure_reverted"),
                dep_o=row.get("departure_other"),
                dwell=row.get("dwell_sum"),
                dwell_max=row.get("dwell_max"),
            )
        )
    if summary.get("conclusion"):
        lines.append("")
        lines.append("结论：")
        lines.append("")
        for item in summary["conclusion"]:
            lines.append(f"- {item}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--status", default=None)
    parser.add_argument(
        "--conclusion",
        action="append",
        default=[],
        help="Conclusion bullet; may be provided multiple times.",
    )
    parser.add_argument("npz", nargs="+")
    args = parser.parse_args(argv)

    rows = [summarize_npz(path) for path in args.npz]
    summary = build_summary(
        experiment=args.experiment,
        rows=rows,
        status=args.status,
        conclusion=args.conclusion,
    )
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )
    write_markdown(summary, output_md)


if __name__ == "__main__":
    main()
