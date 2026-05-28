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
    return f"{float(value):.6f}"


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
