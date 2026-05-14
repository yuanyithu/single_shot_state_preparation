import argparse
import json
import math
from pathlib import Path

import numpy as np

from plot_fixed_p_q_scan_lattice_union import plot_fixed_p_q_scan_lattice_union
from pool_independent_threshold_runs import pool_independent_runs
from production_chunked_scan import _format_probability_tag


def _parse_int_csv(csv_value):
    return [int(value.strip()) for value in csv_value.split(",") if value.strip()]


def _parse_float_csv(csv_value):
    return [
        float(value.strip())
        for value in csv_value.split(",")
        if value.strip()
    ]


def _as_python_scalar(value):
    array = np.asarray(value)
    if array.shape == ():
        return array.item()
    return value


def _load_npz(path):
    with np.load(path, allow_pickle=True) as loaded:
        return {key: loaded[key] for key in loaded.files}


def _load_json_or_none(path):
    path = Path(path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _find_final_npz(run_root, lattice_size, q_tag):
    run_root = Path(run_root)
    if not run_root.exists():
        return None
    pattern = f"scan_result_L{int(lattice_size)}_*_q{q_tag}_*.npz"
    matches = sorted(path for path in run_root.glob(pattern) if path.is_file())
    if len(matches) == 0:
        return None
    if len(matches) > 1:
        raise ValueError(f"multiple final NPZ matches in {run_root}: {matches}")
    return matches[0]


def _manifest_summary(path):
    manifest = _load_json_or_none(path)
    if manifest is None:
        return {
            "manifest_path": str(path),
            "exists": False,
        }
    summary = manifest.get("summary", {})
    config = manifest.get("config", {})
    final_outputs = manifest.get("final_outputs", {})
    return {
        "manifest_path": str(path),
        "exists": True,
        "completed_chunks": int(summary.get("completed_chunks", 0)),
        "failed_chunks": int(summary.get("failed_chunks", 0)),
        "pending_chunks": int(summary.get("pending_chunks", 0)),
        "total_chunks": int(summary.get("total_chunks", 0)),
        "final_status": final_outputs.get("status"),
        "final_npz_path": final_outputs.get("npz_path"),
        "num_disorder_samples_total": config.get("num_disorder_samples_total"),
        "chunk_size": config.get("chunk_size"),
        "workers": config.get("workers"),
        "num_start_chains": config.get("num_start_chains"),
        "q0_num_start_chains": config.get("q0_num_start_chains"),
        "num_replicas_per_start": config.get("num_replicas_per_start"),
        "pt_p_hot": config.get("pt_p_hot"),
        "pt_num_temperatures": config.get("pt_num_temperatures"),
        "max_effective_num_burn_in_sweeps": config.get(
            "max_effective_num_burn_in_sweeps"
        ),
        "effective_num_burn_in_sweeps_list": config.get(
            "effective_num_burn_in_sweeps_list"
        ),
        "common_random_disorder_across_p": config.get(
            "common_random_disorder_across_p"
        ),
        "git_commit_sha": manifest.get("git_commit_sha"),
        "hostname": manifest.get("hostname"),
    }


def _finite_or_none(value):
    value = float(value)
    if not math.isfinite(value):
        return None
    return value


def _metric_from_matrix(result, key):
    if key not in result:
        return None
    matrix = np.asarray(result[key], dtype=np.float64)
    if matrix.size == 0:
        return None
    return _finite_or_none(matrix.reshape(-1)[0])


def _pooled_diagnostic_row(path, lattice_size, q_value):
    result = _load_npz(path)
    row = {
        "lattice_size": int(lattice_size),
        "q": float(q_value),
        "pooled_npz_path": str(path),
        "num_disorder_samples": int(_as_python_scalar(result["num_disorder_samples"])),
        "pt_enabled": bool(_as_python_scalar(result.get("pt_enabled", False))),
        "converged": bool(np.asarray(result["converged_mask_matrix"]).reshape(-1)[0]),
        "q_top": _metric_from_matrix(result, "q_top_curve_matrix"),
        "q_top_std": _metric_from_matrix(result, "q_top_std_curve_matrix"),
    }
    if float(q_value) == 0.0:
        row["q0_mean_q_top_spread"] = _metric_from_matrix(
            result,
            "q0_mean_q_top_spread_curve_matrix",
        )
        row["q0_mean_m_u_spread_linf"] = _metric_from_matrix(
            result,
            "q0_mean_m_u_spread_linf_curve_matrix",
        )
    else:
        row["mean_q_top_spread"] = _metric_from_matrix(
            result,
            "mean_q_top_spread_curve_matrix",
        )
        row["mean_m_u_spread_linf"] = _metric_from_matrix(
            result,
            "mean_m_u_spread_linf_curve_matrix",
        )
        row["max_r_hat"] = _metric_from_matrix(
            result,
            "max_r_hat_curve_matrix",
        )
        row["min_effective_sample_size"] = _metric_from_matrix(
            result,
            "min_effective_sample_size_curve_matrix",
        )
        row["mean_pt_min_swap_acceptance_rate"] = _metric_from_matrix(
            result,
            "mean_pt_min_swap_acceptance_rate_curve_matrix",
        )
        row["mean_pt_mean_swap_acceptance_rate"] = _metric_from_matrix(
            result,
            "mean_pt_mean_swap_acceptance_rate_curve_matrix",
        )
    return row


def _summarize_diagnostics(rows):
    q_positive_rows = [row for row in rows if row["q"] > 0.0]
    q0_rows = [row for row in rows if row["q"] == 0.0]
    q_positive_passed = sum(1 for row in q_positive_rows if row["converged"])
    q0_passed = sum(1 for row in q0_rows if row["converged"])
    summary = {
        "num_rows": len(rows),
        "q0": {
            "num_points": len(q0_rows),
            "num_passed": q0_passed,
            "max_mean_q_top_spread": None,
        },
        "q_positive": {
            "num_points": len(q_positive_rows),
            "num_passed": q_positive_passed,
            "max_mean_q_top_spread": None,
            "max_r_hat": None,
            "min_effective_sample_size": None,
            "min_pt_min_swap_acceptance_rate": None,
        },
        "rows": rows,
    }
    if q0_rows:
        summary["q0"]["max_mean_q_top_spread"] = max(
            row.get("q0_mean_q_top_spread")
            for row in q0_rows
            if row.get("q0_mean_q_top_spread") is not None
        )
    finite_q_spreads = [
        row.get("mean_q_top_spread")
        for row in q_positive_rows
        if row.get("mean_q_top_spread") is not None
    ]
    finite_r_hat = [
        row.get("max_r_hat")
        for row in q_positive_rows
        if row.get("max_r_hat") is not None
    ]
    finite_ess = [
        row.get("min_effective_sample_size")
        for row in q_positive_rows
        if row.get("min_effective_sample_size") is not None
    ]
    finite_swap = [
        row.get("mean_pt_min_swap_acceptance_rate")
        for row in q_positive_rows
        if row.get("mean_pt_min_swap_acceptance_rate") is not None
    ]
    if finite_q_spreads:
        summary["q_positive"]["max_mean_q_top_spread"] = max(finite_q_spreads)
    if finite_r_hat:
        summary["q_positive"]["max_r_hat"] = max(finite_r_hat)
    if finite_ess:
        summary["q_positive"]["min_effective_sample_size"] = min(finite_ess)
    if finite_swap:
        summary["q_positive"]["min_pt_min_swap_acceptance_rate"] = min(
            finite_swap
        )
    return summary


def _write_readme(output_dir, analysis_result):
    lines = [
        "# exp32 fixed p=0.0500 q scan, nd-2/nd-3",
        "",
        f"- Status: `{analysis_result['status']}`.",
        "- Grid: fixed `p=0.0500`, `q=0.0000,0.0050,...,0.0750`, `L=3,4,5,6,7`.",
        "- Pooling: independent nd-2 and nd-3 source runs, expected `2048` disorder per `(L,q)` after pooling.",
        f"- Manifest summary: [`manifest_summary.json`](manifest_summary.json).",
        f"- Diagnostics summary: [`diagnostics_summary.json`](diagnostics_summary.json).",
    ]
    if analysis_result.get("q_top_plot_path") is not None:
        q_top_name = Path(analysis_result["q_top_plot_path"]).name
        gap_name = Path(analysis_result["gap_plot_path"]).name
        summary_name = Path(analysis_result["fixed_p_summary_path"]).name
        lines.extend([
            f"- q_top plot: [`analysis/{q_top_name}`](analysis/{q_top_name}).",
            f"- gap plot: [`analysis/{gap_name}`](analysis/{gap_name}).",
            f"- fixed-p summary: [`analysis/{summary_name}`](analysis/{summary_name}).",
        ])
    lines.append("")
    Path(output_dir, "README.md").write_text("\n".join(lines), encoding="utf-8")


def analyze_exp32(
        output_dir,
        host_tags,
        lattice_sizes,
        q_values,
        fixed_p,
        output_stem,
        allow_partial):
    output_dir = Path(output_dir).resolve()
    remote_runs_dir = output_dir / "remote_runs"
    pooled_dir = output_dir / "pooled"
    analysis_dir = output_dir / "analysis"
    pooled_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    missing = []
    pooled_paths = {}
    diagnostic_rows = []
    fixed_p_tag = _format_probability_tag(fixed_p)

    for lattice_size in lattice_sizes:
        for q_value in q_values:
            q_tag = _format_probability_tag(q_value)
            source_npz_paths = []
            for host_tag in host_tags:
                child_run_root = (
                    remote_runs_dir
                    / host_tag
                    / f"L{int(lattice_size)}"
                    / f"q_{q_tag}"
                )
                manifest_path = child_run_root / "manifest.json"
                manifest_row = _manifest_summary(manifest_path)
                manifest_row.update({
                    "host_tag": host_tag,
                    "lattice_size": int(lattice_size),
                    "q": float(q_value),
                    "local_run_root": str(child_run_root),
                })
                manifest_rows.append(manifest_row)
                npz_path = _find_final_npz(child_run_root, lattice_size, q_tag)
                if npz_path is not None:
                    source_npz_paths.append(npz_path)
                else:
                    missing.append({
                        "host_tag": host_tag,
                        "lattice_size": int(lattice_size),
                        "q": float(q_value),
                        "run_root": str(child_run_root),
                    })
            if len(source_npz_paths) != len(host_tags):
                continue
            point_output_dir = pooled_dir / f"L{int(lattice_size)}" / f"q_{q_tag}"
            point_output_stem = (
                f"pooled_L{int(lattice_size)}_p{fixed_p_tag}_q{q_tag}_"
                "exp32_nd23"
            )
            pool_summary = pool_independent_runs(
                input_paths=source_npz_paths,
                output_dir=point_output_dir,
                output_stem=point_output_stem,
                skip_threshold_analysis=True,
            )
            pooled_path = Path(pool_summary["output_path"])
            pooled_paths[(int(lattice_size), float(q_value))] = pooled_path
            diagnostic_rows.append(
                _pooled_diagnostic_row(
                    path=pooled_path,
                    lattice_size=lattice_size,
                    q_value=q_value,
                )
            )

    manifest_summary = {
        "host_tags": host_tags,
        "lattice_sizes": lattice_sizes,
        "q_values": q_values,
        "fixed_p": fixed_p,
        "num_manifest_rows": len(manifest_rows),
        "num_missing_final_npz": len(missing),
        "missing_final_npz": missing,
        "rows": manifest_rows,
    }
    _write_json(output_dir / "manifest_summary.json", manifest_summary)

    diagnostics_summary = _summarize_diagnostics(diagnostic_rows)
    _write_json(output_dir / "diagnostics_summary.json", diagnostics_summary)

    incomplete = len(missing) > 0
    if incomplete and not allow_partial:
        raise RuntimeError(
            "Missing final NPZ files; rerun with --allow-partial to summarize "
            "completed points. See manifest_summary.json."
        )

    complete_lattices = []
    for lattice_size in lattice_sizes:
        if all((int(lattice_size), float(q_value)) in pooled_paths for q_value in q_values):
            complete_lattices.append(int(lattice_size))

    plot_result = {
        "q_top_plot_path": None,
        "gap_plot_path": None,
        "fixed_p_summary_path": None,
    }
    if complete_lattices:
        plot_inputs = [
            pooled_paths[(int(lattice_size), float(q_value))]
            for lattice_size in complete_lattices
            for q_value in q_values
        ]
        fixed_p_result = plot_fixed_p_q_scan_lattice_union(
            input_paths=plot_inputs,
            output_dir=analysis_dir,
            output_stem=output_stem,
            fixed_p=fixed_p,
            p_tolerance=1.0e-12,
        )
        plot_result = {
            "q_top_plot_path": fixed_p_result["q_top_plot_path"],
            "gap_plot_path": fixed_p_result["gap_plot_path"],
            "fixed_p_summary_path": fixed_p_result["summary_path"],
        }

    analysis_result = {
        "status": "partial" if incomplete else "complete",
        "output_dir": str(output_dir),
        "num_pooled_points": len(pooled_paths),
        "expected_num_points": len(lattice_sizes) * len(q_values),
        "complete_lattices_for_plot": complete_lattices,
        "manifest_summary_path": str(output_dir / "manifest_summary.json"),
        "diagnostics_summary_path": str(output_dir / "diagnostics_summary.json"),
        **plot_result,
    }
    _write_json(output_dir / "analysis_result.json", analysis_result)
    _write_readme(output_dir, analysis_result)
    return analysis_result


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Pool and plot exp32 fixed-p q scan results after remote collection."
        )
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--host-tags", default="nd2,nd3")
    parser.add_argument("--lattice-sizes", default="3,4,5,6,7")
    parser.add_argument(
        "--q-values",
        default=(
            "0.0000,0.0050,0.0100,0.0150,0.0200,0.0250,0.0300,0.0350,"
            "0.0400,0.0450,0.0500,0.0550,0.0600,0.0650,0.0700,0.0750"
        ),
    )
    parser.add_argument("--fixed-p", type=float, default=0.0500)
    parser.add_argument(
        "--output-stem",
        default="fixed_p050_q000_075_exp32_nd23_pooled",
    )
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    result = analyze_exp32(
        output_dir=args.output_dir,
        host_tags=[
            value.strip() for value in args.host_tags.split(",") if value.strip()
        ],
        lattice_sizes=_parse_int_csv(args.lattice_sizes),
        q_values=_parse_float_csv(args.q_values),
        fixed_p=float(args.fixed_p),
        output_stem=args.output_stem,
        allow_partial=bool(args.allow_partial),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
