#!/usr/bin/env python3
"""Combine retained q=0 evidence with the audited nd-2 formal extension.

The output is an evidence table and crossing/boundary classification only.
It intentionally does not generate a paper figure or use q-positive curves.
"""

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np


CI_Z = 1.959963984540054
REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent
RETAINED_INPUTS = {
    "deep": {
        "path": (
            "data/2d_toric_code/without_measurement_noise/"
            "q0_threshold_deep_nd3_20260420_221142/"
            "scan_result_multi_L_q0_geometric_multistart_threshold_deep.npz"
        ),
        "sha256": "f3821be7f779119603f1464b9f201ece184106d515f9f19baf5fc2db9a5f4f61",
        "source_commit": "a15c3326fcc07844e06cc02ff176cf39ab7c0bbb",
    },
    "control": {
        "path": (
            "data/2d_toric_code/without_measurement_noise/"
            "q0_control_extension_nd3_20260421_225303/"
            "scan_result_multi_L_q0_control_extension.npz"
        ),
        "sha256": "06254aa73b3e5c4596bdaf94d076e2c26c7427e43e8ba3b70789b49d199094ee",
        "source_commit": "a197215bd18e9ffc160b4864b7f54239ff4e39da",
    },
}


def _timestamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_text_atomic(path, text):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_json_atomic(path, value):
    _write_text_atomic(
        path,
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )


def _load_q0_npz(path, expected_source_commit=None):
    with np.load(path, allow_pickle=False) as data:
        p_values = np.asarray(
            data["data_error_probability_list"], dtype=np.float64
        )
        lattice_sizes = np.asarray(data["lattice_size_list"], dtype=np.int64)
        per_disorder = np.asarray(
            data["disorder_q_top_values_tensor"], dtype=np.float64
        )
        per_start = np.asarray(
            data["q0_q_top_values_per_disorder_per_start_tensor"],
            dtype=np.float64,
        )
        curve = np.asarray(data["q_top_curve_matrix"], dtype=np.float64)
        spread = np.asarray(
            data["q0_mean_q_top_spread_curve_matrix"], dtype=np.float64
        )
        source_commit = str(data["git_commit_sha"].item())
        num_disorder = int(data["num_disorder_samples"].item())
        if expected_source_commit and source_commit != expected_source_commit:
            raise ValueError(f"source commit mismatch for {path}")
        if not all(
            np.all(np.isfinite(array))
            for array in (per_disorder, per_start, curve, spread)
        ):
            raise ValueError(f"non-finite q=0 values in {path}")
        if not np.array_equal(np.mean(per_disorder, axis=2), curve):
            raise ValueError(f"q_top aggregate mismatch in {path}")
        if not np.array_equal(
            np.mean(np.ptp(per_start, axis=3), axis=2), spread
        ):
            raise ValueError(f"four-start spread mismatch in {path}")
        return {
            "p": p_values,
            "lattice_sizes": lattice_sizes,
            "per_disorder": per_disorder,
            "curve": curve,
            "spread": spread,
            "num_disorder": num_disorder,
            "source_commit": source_commit,
        }


def _series_from_run(run, lattice_size):
    indices = np.flatnonzero(run["lattice_sizes"] == int(lattice_size))
    if indices.size != 1:
        raise ValueError(f"expected one L={lattice_size} series")
    index = int(indices[0])
    per_disorder = run["per_disorder"][index]
    return {
        "p": run["p"].copy(),
        "mean": run["curve"][index].copy(),
        "sem": np.std(per_disorder, axis=1, ddof=1)
        / np.sqrt(run["num_disorder"]),
        "spread": run["spread"][index].copy(),
        "num_disorder": np.full(
            run["p"].shape, run["num_disorder"], dtype=np.int64
        ),
    }


def _join_series(first, second):
    p_values = np.concatenate((first["p"], second["p"]))
    if np.unique(np.round(p_values, 12)).size != p_values.size:
        raise ValueError("q=0 source grids overlap unexpectedly")
    order = np.argsort(p_values)
    return {
        key: np.concatenate((first[key], second[key]))[order]
        for key in ("p", "mean", "sem", "spread", "num_disorder")
    }


def _classify_pair(lower_size, upper_size, lower, upper):
    common = sorted(
        set(np.round(lower["p"], 12)).intersection(
            np.round(upper["p"], 12)
        )
    )
    lower_indices = [
        int(np.flatnonzero(np.isclose(lower["p"], value))[0])
        for value in common
    ]
    upper_indices = [
        int(np.flatnonzero(np.isclose(upper["p"], value))[0])
        for value in common
    ]
    p_values = np.asarray(common, dtype=np.float64)
    difference = upper["mean"][upper_indices] - lower["mean"][lower_indices]
    sem = np.sqrt(
        upper["sem"][upper_indices] ** 2
        + lower["sem"][lower_indices] ** 2
    )
    ci_low = difference - CI_Z * sem
    ci_high = difference + CI_Z * sem
    brackets = []
    for index in range(len(p_values) - 1):
        left = float(difference[index])
        right = float(difference[index + 1])
        if left == 0.0:
            crossing = float(p_values[index])
        elif left * right < 0.0:
            crossing = float(
                p_values[index]
                - left
                * (p_values[index + 1] - p_values[index])
                / (right - left)
            )
        else:
            continue
        brackets.append({
            "p_low": float(p_values[index]),
            "p_high": float(p_values[index + 1]),
            "linear_point_estimate": crossing,
        })

    if brackets:
        classification = "POINT_ESTIMATE_CROSSING_BRACKETED"
        boundary = None
    elif np.all(difference > 0.0):
        classification = "CROSSING_NOT_BRACKETED_POSITIVE_THROUGH_UPPER_BOUNDARY"
        boundary = {
            "direction": "p_crossing_above_sampled_max_if_sign_trend_continues",
            "p": float(p_values[-1]),
            "pointwise_95pct_sign_excludes_zero": bool(ci_low[-1] > 0.0),
        }
    elif np.all(difference < 0.0):
        classification = "CROSSING_NOT_BRACKETED_NEGATIVE_THROUGH_LOWER_BOUNDARY"
        boundary = {
            "direction": "p_crossing_below_sampled_min_if_sign_trend_continues",
            "p": float(p_values[0]),
            "pointwise_95pct_sign_excludes_zero": bool(ci_high[0] < 0.0),
        }
    else:
        classification = "ZERO_OR_NONFINITE_SIGN_PATTERN_REQUIRES_MANUAL_REVIEW"
        boundary = None
    return {
        "pair": f"L{upper_size}_minus_L{lower_size}",
        "classification": classification,
        "brackets": brackets,
        "one_sided_boundary_diagnostic": boundary,
        "p": p_values.tolist(),
        "difference": difference.tolist(),
        "ci95_low": ci_low.tolist(),
        "ci95_high": ci_high.tolist(),
    }


def analyze(stage_dir):
    stage_dir = Path(stage_dir).resolve()
    try:
        stage_relpath = stage_dir.relative_to(REPO_ROOT).as_posix()
    except ValueError as exc:
        raise ValueError("stage-dir must be inside this repository") from exc
    formal_audit_path = stage_dir / "formal_audit.json"
    pilot_audit_path = stage_dir / "pilot_audit.json"
    config_path = stage_dir / "control/nd2_staged_experiment_config.json"
    formal_audit = _load_json(formal_audit_path)
    pilot_audit = _load_json(pilot_audit_path)
    config = _load_json(config_path)
    if not formal_audit["passed"]:
        raise ValueError("remote formal audit did not pass")
    source_commit = config["source_commit"]
    if formal_audit["source_commit"] != source_commit:
        raise ValueError("formal audit/config source commit mismatch")

    retained = {}
    provenance = []
    for name, spec in RETAINED_INPUTS.items():
        input_path = REPO_ROOT / spec["path"]
        actual_hash = _sha256_file(input_path)
        if actual_hash != spec["sha256"]:
            raise ValueError(f"retained input SHA mismatch: {spec['path']}")
        retained[name] = _load_q0_npz(
            input_path, expected_source_commit=spec["source_commit"]
        )
        provenance.append({
            "role": f"retained_{name}",
            "path": spec["path"],
            "sha256": actual_hash,
            "source_commit": spec["source_commit"],
        })

    formal_paths = {
        9: stage_dir / "runs/q0_formal_L9/q0_formal_L9.npz",
        11: stage_dir / "runs/q0_formal_L11/q0_formal_L11.npz",
    }
    formal_runs = {}
    for lattice_size, input_path in formal_paths.items():
        expected_hash = formal_audit["lattice_sizes"][str(lattice_size)][
            "npz_sha256"
        ]
        actual_hash = _sha256_file(input_path)
        if actual_hash != expected_hash:
            raise ValueError(f"formal L={lattice_size} SHA mismatch")
        formal_runs[lattice_size] = _load_q0_npz(
            input_path, expected_source_commit=source_commit
        )
        provenance.append({
            "role": f"formal_L{lattice_size}",
            "path": input_path.relative_to(REPO_ROOT).as_posix(),
            "sha256": actual_hash,
            "source_commit": source_commit,
        })

    series = {
        7: _series_from_run(retained["deep"], 7),
        9: _join_series(
            _series_from_run(retained["control"], 9),
            _series_from_run(formal_runs[9], 9),
        ),
        11: _join_series(
            _series_from_run(retained["control"], 11),
            _series_from_run(formal_runs[11], 11),
        ),
    }
    pair_results = {
        "L9_minus_L7": _classify_pair(7, 9, series[7], series[9]),
        "L11_minus_L9": _classify_pair(9, 11, series[9], series[11]),
    }

    report = {
        "schema_version": 1,
        "created_at": _timestamp(),
        "scope": "q=0 formal extension audit; no paper figure",
        "paper_figure": False,
        "q_positive_curve_included": False,
        "source_commit": source_commit,
        "stage": stage_relpath,
        "selected_l11_schedule": formal_audit["selected_l11_schedule"],
        "q0_pilot_gate": pilot_audit["q0"]["gate"],
        "q_positive_sentinel_decision": pilot_audit["q_positive"][
            "decision"
        ],
        "inputs": provenance,
        "pair_classifications": pair_results,
        "maximum_mean_four_start_spread": {
            str(lattice_size): float(np.max(item["spread"]))
            for lattice_size, item in series.items()
        },
    }
    _write_json_atomic(
        OUTPUT_DIR / "q0_formal_extension_audit.json", report
    )

    csv_path = OUTPUT_DIR / "q0_formal_extension_summary.csv"
    temporary_csv = csv_path.with_name(csv_path.name + ".tmp")
    with temporary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "lattice_size",
                "p",
                "num_disorder",
                "mean_q_top",
                "standard_error",
                "ci95_low",
                "ci95_high",
                "mean_four_start_spread",
            ),
        )
        writer.writeheader()
        for lattice_size, item in series.items():
            for index, p_value in enumerate(item["p"]):
                writer.writerow({
                    "lattice_size": lattice_size,
                    "p": float(p_value),
                    "num_disorder": int(item["num_disorder"][index]),
                    "mean_q_top": float(item["mean"][index]),
                    "standard_error": float(item["sem"][index]),
                    "ci95_low": float(
                        item["mean"][index] - CI_Z * item["sem"][index]
                    ),
                    "ci95_high": float(
                        item["mean"][index] + CI_Z * item["sem"][index]
                    ),
                    "mean_four_start_spread": float(item["spread"][index]),
                })
    temporary_csv.replace(csv_path)

    lines = [
        "# `q=0` nd-2 正式补点审计",
        "",
        "本说明只整理 `q=0` 正式补点与 crossing/边界分类；不生成最终论文图，也不把 `q>0` 哨兵升级为论文证据。",
        "",
        f"- 固定源码：`{source_commit}`",
        f"- nd-2 收集目录：`{stage_relpath}`",
        f"- L11 pilot 选择：`{formal_audit['selected_l11_schedule']}`",
        f"- q>0 哨兵结论：`{pilot_audit['q_positive']['decision']}`",
        "",
        "## 相邻尺寸分类",
        "",
    ]
    for key in ("L9_minus_L7", "L11_minus_L9"):
        item = pair_results[key]
        lines.append(f"- `{key}`：`{item['classification']}`")
        for bracket in item["brackets"]:
            lines.append(
                "  - 网格 bracket "
                f"`[{bracket['p_low']:.4f}, {bracket['p_high']:.4f}]`；"
                f"线性点估计 `{bracket['linear_point_estimate']:.6f}`，"
                "不作为最终临界值。"
            )
        if item["one_sided_boundary_diagnostic"] is not None:
            boundary = item["one_sided_boundary_diagnostic"]
            lines.append(
                f"  - 边界 `p={boundary['p']:.4f}` 的点态 95% 区间"
                f"排除零：`{boundary['pointwise_95pct_sign_excludes_zero']}`；"
                "不得把边界诊断画成普通 crossing 点。"
            )
    lines.extend([
        "",
        "## 交付",
        "",
        "- `q0_formal_extension_audit.json`：输入哈希、pilot 决策和相邻尺寸分类。",
        "- `q0_formal_extension_summary.csv`：逐 `(L,p)` 的 disorder 统计量与四起点 spread。",
        "",
    ])
    _write_text_atomic(
        OUTPUT_DIR / "Q0_FORMAL_EXTENSION_AUDIT.md", "\n".join(lines)
    )
    print(json.dumps({
        key: value["classification"]
        for key, value in pair_results.items()
    }, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-dir", required=True)
    args = parser.parse_args()
    analyze(args.stage_dir)


if __name__ == "__main__":
    main()
