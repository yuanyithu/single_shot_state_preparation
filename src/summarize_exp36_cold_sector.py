"""Summarize exp36 cold-sector histogram convergence diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _as_scalar(data, key, default=None):
    if key not in data:
        return default
    value = data[key]
    if np.asarray(value).shape == ():
        return np.asarray(value).item()
    return value


def _probabilities_from_counts(counts):
    counts = np.asarray(counts, dtype=np.float64)
    total = float(np.sum(counts))
    if total <= 0.0:
        return np.zeros_like(counts, dtype=np.float64)
    return counts / total


def _tv_from_counts(left_counts, right_counts):
    left_probabilities = _probabilities_from_counts(left_counts)
    right_probabilities = _probabilities_from_counts(right_counts)
    return float(0.5 * np.sum(np.abs(
        left_probabilities - right_probabilities
    )))


def _decode_sector(index, num_masks):
    signs = []
    for mask_index in range(num_masks):
        signs.append("-" if ((int(index) >> mask_index) & 1) else "+")
    return "".join(signs)


def _top_sector_rows(counts, top_k):
    counts = np.asarray(counts, dtype=np.int64)
    probabilities = _probabilities_from_counts(counts)
    if counts.size == 0:
        return []
    num_masks = int(np.log2(counts.size))
    top_indices = np.argsort(probabilities)[::-1][:top_k]
    rows = []
    for sector_index in top_indices:
        if counts[sector_index] <= 0:
            continue
        rows.append((
            int(sector_index),
            _decode_sector(sector_index, num_masks),
            int(counts[sector_index]),
            float(probabilities[sector_index]),
        ))
    return rows


def _load_histogram_payload(path):
    data = np.load(path, allow_pickle=True)
    if "cold_sector_histogram_counts_per_disorder_tensor" in data:
        full_counts_tensor = np.asarray(
            data["cold_sector_histogram_counts_per_disorder_tensor"],
            dtype=np.int64,
        )
        block_counts_tensor = np.asarray(
            data["cold_sector_histogram_block_counts_per_disorder_tensor"],
            dtype=np.int64,
        )
        chain_counts_tensor = np.asarray(
            data[
                "chain_cold_sector_histogram_counts_per_disorder_per_start_replica_tensor"
            ],
            dtype=np.int64,
        )
        first_second_tv_tensor = np.asarray(
            data[
                "chain_cold_sector_histogram_first_second_tv_per_disorder_per_start_replica_tensor"
            ],
            dtype=np.float64,
        )
        q_top_tensor = np.asarray(
            data.get("disorder_q_top_values_tensor", []),
            dtype=np.float64,
        )
        chain_q_top_tensor = np.asarray(
            data.get(
                "chain_q_top_values_per_disorder_per_start_replica_tensor",
                [],
            ),
            dtype=np.float64,
        )
    elif "cold_sector_histogram_counts_per_disorder" in data:
        full_counts_tensor = np.asarray(
            data["cold_sector_histogram_counts_per_disorder"],
            dtype=np.int64,
        )
        block_counts_tensor = np.asarray(
            data["cold_sector_histogram_block_counts_per_disorder"],
            dtype=np.int64,
        )
        chain_counts_tensor = np.asarray(
            data[
                "chain_cold_sector_histogram_counts_per_disorder_per_start_replica"
            ],
            dtype=np.int64,
        )
        first_second_tv_tensor = np.asarray(
            data[
                "chain_cold_sector_histogram_first_second_tv_per_disorder_per_start_replica"
            ],
            dtype=np.float64,
        )
        q_top_tensor = np.asarray(
            data.get("disorder_q_top_values", []),
            dtype=np.float64,
        )
        chain_q_top_tensor = np.asarray(
            data.get("chain_q_top_values_per_disorder_per_start_replica", []),
            dtype=np.float64,
        )
    else:
        raise KeyError(
            f"{path} does not contain cold-sector histogram diagnostics"
        )

    return {
        "path": Path(path),
        "mode": str(_as_scalar(data, "q_positive_initial_chain_mode", "")),
        "full_counts_tensor": full_counts_tensor,
        "block_counts_tensor": block_counts_tensor,
        "q_top_tensor": q_top_tensor,
        "full_counts": np.sum(full_counts_tensor, axis=tuple(
            range(full_counts_tensor.ndim - 1)
        )),
        "block_counts": block_counts_tensor,
        "chain_counts": chain_counts_tensor,
        "first_second_tv": first_second_tv_tensor,
        "q_top": q_top_tensor,
        "chain_q_top": chain_q_top_tensor,
    }


def _flatten_disorder_counts(full_counts_tensor):
    full_counts_tensor = np.asarray(full_counts_tensor, dtype=np.int64)
    if full_counts_tensor.ndim < 2:
        raise ValueError("full_counts_tensor must include a sector-bin axis")
    return full_counts_tensor.reshape(-1, full_counts_tensor.shape[-1])


def _flatten_q_top(q_top_tensor):
    q_top_tensor = np.asarray(q_top_tensor, dtype=np.float64)
    if q_top_tensor.size == 0:
        return np.empty(0, dtype=np.float64)
    return q_top_tensor.reshape(-1)


def _write_report(payloads, labels, top_k):
    lines = []
    lines.append("# exp36 cold-sector histogram convergence summary")
    lines.append("")
    lines.append("## Runs")
    aggregate_counts = []
    for label, payload in zip(labels, payloads):
        full_counts = payload["full_counts"]
        aggregate_counts.append(full_counts)
        lines.append("")
        lines.append(f"### {label}")
        lines.append("")
        lines.append(f"- file: `{payload['path']}`")
        if payload["mode"]:
            lines.append(f"- initial mode: `{payload['mode']}`")
        if payload["q_top"].size:
            lines.append(
                f"- q_top mean: {float(np.mean(payload['q_top'])):.6f}"
            )
        if payload["chain_q_top"].size:
            chain_values = np.ravel(payload["chain_q_top"])
            lines.append(
                "- chain q_top range: "
                f"{float(np.min(chain_values)):.6f} .. "
                f"{float(np.max(chain_values)):.6f}"
            )
        tv_values = np.ravel(payload["first_second_tv"])
        tv_values = tv_values[np.isfinite(tv_values)]
        if tv_values.size:
            lines.append(
                "- chain first-half vs second-half TV: "
                f"mean={float(np.mean(tv_values)):.4f}, "
                f"max={float(np.max(tv_values)):.4f}"
            )
        lines.append("- top sectors:")
        for sector_index, signs, count, probability in _top_sector_rows(
                full_counts,
                top_k):
            lines.append(
                f"  - {sector_index:3d} `{signs}` "
                f"count={count} prob={probability:.4f}"
            )

    if len(payloads) >= 2:
        lines.append("")
        lines.append("## Pairwise TV Between Runs")
        lines.append("")
        header = "| run | " + " | ".join(labels) + " |"
        separator = "|---|" + "|".join(["---"] * len(labels)) + "|"
        lines.append(header)
        lines.append(separator)
        for row_label, row_counts in zip(labels, aggregate_counts):
            row = [row_label]
            for col_counts in aggregate_counts:
                row.append(f"{_tv_from_counts(row_counts, col_counts):.4f}")
            lines.append("| " + " | ".join(row) + " |")

        per_run_disorder_counts = [
            _flatten_disorder_counts(payload["full_counts_tensor"])
            for payload in payloads
        ]
        per_run_q_top = [
            _flatten_q_top(payload["q_top_tensor"])
            for payload in payloads
        ]
        num_disorders = min(counts.shape[0] for counts in per_run_disorder_counts)
        if num_disorders > 0:
            lines.append("")
            lines.append("## Per-Disorder Gate")
            for disorder_index in range(num_disorders):
                lines.append("")
                lines.append(f"### disorder {disorder_index}")
                q_top_values = []
                for label, q_top_values_for_run in zip(labels, per_run_q_top):
                    if disorder_index < q_top_values_for_run.size:
                        q_top_values.append(
                            (label, float(q_top_values_for_run[disorder_index]))
                        )
                if q_top_values:
                    formatted_q_top = ", ".join(
                        f"{label}={value:.6f}"
                        for label, value in q_top_values
                    )
                    spread = max(value for _, value in q_top_values) - min(
                        value for _, value in q_top_values
                    )
                    lines.append(f"- q_top: {formatted_q_top}")
                    lines.append(f"- q_top spread: {spread:.6f}")
                lines.append("- pairwise cold-sector TV:")
                lines.append("| run | " + " | ".join(labels) + " |")
                lines.append("|---|" + "|".join(["---"] * len(labels)) + "|")
                for row_label, row_counts in zip(
                        labels,
                        per_run_disorder_counts):
                    row = [row_label]
                    for col_counts in per_run_disorder_counts:
                        row.append(
                            f"{_tv_from_counts(row_counts[disorder_index], col_counts[disorder_index]):.4f}"
                        )
                    lines.append("| " + " | ".join(row) + " |")
                lines.append("- top sectors by run:")
                for label, counts in zip(labels, per_run_disorder_counts):
                    top_rows = _top_sector_rows(
                        counts[disorder_index],
                        min(top_k, 5),
                    )
                    top_text = "; ".join(
                        f"{sector_index} `{signs}` p={probability:.4f}"
                        for sector_index, signs, _, probability in top_rows
                    )
                    lines.append(f"  - {label}: {top_text}")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz_paths", nargs="+", type=Path)
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    payloads = [_load_histogram_payload(path) for path in args.npz_paths]
    if args.labels is None or len(args.labels) == 0:
        labels = [path.parent.name for path in args.npz_paths]
    else:
        if len(args.labels) != len(args.npz_paths):
            raise ValueError("--labels length must match npz_paths length")
        labels = list(args.labels)
    report = _write_report(payloads, labels, args.top_k)
    if args.output is None:
        print(report, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
