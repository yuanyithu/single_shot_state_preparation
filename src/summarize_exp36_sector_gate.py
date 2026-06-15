"""Summarize exp36 cold-sector histogram gates.

The gate is aimed at the physical question: for a fixed disorder sample,
do independent chains started from different initial sectors produce the
same long-time distribution over measured Wilson-loop sector vectors?
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


RUN_Q_RE = re.compile(r"run_q([0-9]+p[0-9]+)")


@dataclass
class SectorEntry:
    path: Path
    label: str
    lattice_size: int
    q_value: float
    disorder_index: int
    q_top: float
    q_top_spread: float
    chain_counts: np.ndarray
    pooled_counts: np.ndarray
    first_second_tv: np.ndarray
    adjacent_block_tv: np.ndarray
    chain_block_range: np.ndarray
    never_flipped_count: int | None
    start_sector_labels: tuple[str, ...] | None


def _scalar(data: np.lib.npyio.NpzFile, key: str, default=None):
    if key not in data:
        return default
    value = np.asarray(data[key])
    if value.shape == ():
        return value.item()
    return value


def _probabilities(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    total = float(np.sum(counts))
    if total <= 0.0:
        return np.zeros_like(counts, dtype=np.float64)
    return counts / total


def _tv(left_counts: np.ndarray, right_counts: np.ndarray) -> float:
    return float(0.5 * np.sum(np.abs(
        _probabilities(left_counts) - _probabilities(right_counts)
    )))


def _max_pairwise_tv(counts_by_chain: np.ndarray) -> float:
    counts_by_chain = np.asarray(counts_by_chain, dtype=np.int64)
    if counts_by_chain.shape[0] < 2:
        return 0.0
    values = []
    for left_index in range(counts_by_chain.shape[0]):
        for right_index in range(left_index + 1, counts_by_chain.shape[0]):
            values.append(_tv(
                counts_by_chain[left_index],
                counts_by_chain[right_index],
            ))
    return float(max(values))


def _bootstrap_max_pairwise_tv_quantiles(
        counts_by_chain: np.ndarray,
        rng: np.random.Generator,
        num_replicates: int) -> tuple[float, float]:
    counts_by_chain = np.asarray(counts_by_chain, dtype=np.int64)
    if num_replicates <= 0 or counts_by_chain.shape[0] < 2:
        return math.nan, math.nan
    pooled_counts = np.sum(counts_by_chain, axis=0).astype(np.float64)
    total = float(np.sum(pooled_counts))
    if total <= 0.0:
        return math.nan, math.nan
    probabilities = pooled_counts / total
    chain_sample_sizes = [
        int(np.sum(chain_counts))
        for chain_counts in counts_by_chain
    ]
    bootstrap_values = np.empty(num_replicates, dtype=np.float64)
    for replicate_index in range(num_replicates):
        sampled = np.vstack([
            rng.multinomial(sample_size, probabilities)
            for sample_size in chain_sample_sizes
        ])
        bootstrap_values[replicate_index] = _max_pairwise_tv(sampled)
    return (
        float(np.quantile(bootstrap_values, 0.95)),
        float(np.quantile(bootstrap_values, 0.99)),
    )


def _num_masks_from_counts(counts: np.ndarray) -> int:
    num_bins = int(np.asarray(counts).shape[-1])
    return int(round(math.log2(num_bins)))


def _decode_sector(index: int, num_masks: int) -> str:
    return "".join(
        "-" if ((int(index) >> bit_index) & 1) else "+"
        for bit_index in range(num_masks)
    )


def _top_sector_text(counts: np.ndarray, top_k: int = 3) -> str:
    counts = np.asarray(counts, dtype=np.int64)
    probabilities = _probabilities(counts)
    num_masks = _num_masks_from_counts(counts)
    rows = []
    for sector_index in np.argsort(probabilities)[::-1][:top_k]:
        if counts[sector_index] <= 0:
            continue
        rows.append(
            f"{_decode_sector(int(sector_index), num_masks)}:"
            f"{probabilities[sector_index]:.3f}"
        )
    return ", ".join(rows)


def _start_sector_labels_from_npz(
        data: np.lib.npyio.NpzFile) -> tuple[str, ...] | None:
    if "start_sector_labels" not in data:
        return None
    labels = np.asarray(data["start_sector_labels"]).reshape(-1)
    return tuple(str(label) for label in labels)


def _start_label_text(labels: tuple[str, ...] | None) -> str:
    if not labels:
        return "n/a"
    return ", ".join(labels)


def _start_coverage_text(
        labels: tuple[str, ...] | None,
        num_chains: int) -> str:
    if labels and all(set(label) <= {"0", "1"} for label in labels):
        label_lengths = {len(label) for label in labels}
        if len(label_lengths) == 1:
            full_count = 1 << next(iter(label_lengths))
            return f"{len(labels)}/{full_count}"
    return f"{num_chains}/?"


def _q_from_path(path: Path, default: float = math.nan) -> float:
    for parent in [path, *path.parents]:
        match = RUN_Q_RE.search(parent.name)
        if match:
            return float(match.group(1).replace("p", "."))
    return default


def _label_from_path(path: Path) -> str:
    for parent in [path, *path.parents]:
        if RUN_Q_RE.search(parent.name):
            return parent.name
    return path.stem


def _entry_from_chunk_npz(path: Path, data: np.lib.npyio.NpzFile) -> list[SectorEntry]:
    if "chain_cold_sector_histogram_counts_per_disorder_per_start_replica" not in data:
        return []
    chain_counts = np.asarray(
        data["chain_cold_sector_histogram_counts_per_disorder_per_start_replica"],
        dtype=np.int64,
    )
    pooled_counts = np.asarray(
        data["cold_sector_histogram_counts_per_disorder"],
        dtype=np.int64,
    )
    num_disorders = chain_counts.shape[0]
    chain_counts = chain_counts.reshape(
        num_disorders,
        chain_counts.shape[1] * chain_counts.shape[2],
        chain_counts.shape[-1],
    )
    first_second_tv = np.asarray(
        data["chain_cold_sector_histogram_first_second_tv_per_disorder_per_start_replica"],
        dtype=np.float64,
    ).reshape(num_disorders, -1)
    adjacent_block_tv = np.asarray(
        data["chain_cold_sector_histogram_adjacent_block_tv_per_disorder_per_start_replica"],
        dtype=np.float64,
    ).reshape(num_disorders, -1)
    chain_block_range = np.asarray(
        data["chain_q_top_block_range_per_disorder_per_start_replica"],
        dtype=np.float64,
    ).reshape(num_disorders, -1)
    q_top_values = np.asarray(data["disorder_q_top_values"], dtype=np.float64)
    q_top_spread_values = np.asarray(
        data.get("q_top_spread_per_disorder", np.full(num_disorders, math.nan)),
        dtype=np.float64,
    )
    never_flipped = np.asarray(
        data.get("num_chains_that_never_flipped_sector_per_disorder", []),
        dtype=np.int64,
    )
    lattice_size = int(_scalar(data, "lattice_size", -1))
    q_value = float(_scalar(data, "syndrome_error_probability", _q_from_path(path)))
    label = _label_from_path(path)
    start_sector_labels = _start_sector_labels_from_npz(data)
    entries = []
    disorder_offset = int(_scalar(data, "disorder_offset", 0))
    for disorder_index in range(num_disorders):
        entries.append(SectorEntry(
            path=path,
            label=label,
            lattice_size=lattice_size,
            q_value=q_value,
            disorder_index=disorder_offset + disorder_index,
            q_top=float(q_top_values[disorder_index]),
            q_top_spread=float(q_top_spread_values[disorder_index]),
            chain_counts=chain_counts[disorder_index],
            pooled_counts=pooled_counts[disorder_index],
            first_second_tv=first_second_tv[disorder_index],
            adjacent_block_tv=adjacent_block_tv[disorder_index],
            chain_block_range=chain_block_range[disorder_index],
            never_flipped_count=(
                int(never_flipped[disorder_index])
                if never_flipped.size > disorder_index else None
            ),
            start_sector_labels=start_sector_labels,
        ))
    return entries


def _entries_from_tensor_npz(
        path: Path,
        data: np.lib.npyio.NpzFile) -> list[SectorEntry]:
    if "chain_cold_sector_histogram_counts_per_disorder_per_start_replica_tensor" not in data:
        return []
    chain_tensor = np.asarray(
        data["chain_cold_sector_histogram_counts_per_disorder_per_start_replica_tensor"],
        dtype=np.int64,
    )
    pooled_tensor = np.asarray(
        data["cold_sector_histogram_counts_per_disorder_tensor"],
        dtype=np.int64,
    )
    # Expected layout:
    # lattice, p, disorder, start, replica, sector_bin.
    if chain_tensor.ndim != 6:
        raise ValueError(
            "expected chain cold-sector tensor with 6 dimensions "
            "(L,p,disorder,start,replica,sector)"
        )
    lattice_sizes = np.asarray(data.get("lattice_size_list", []), dtype=np.int64)
    q_value = float(_scalar(data, "syndrome_error_probability", _q_from_path(path)))
    q_top_tensor = np.asarray(
        data.get("disorder_q_top_values_tensor", np.full(chain_tensor.shape[:3], math.nan)),
        dtype=np.float64,
    )
    q_top_spread_tensor = np.asarray(
        data.get("q_top_spread_per_disorder_tensor", np.full(chain_tensor.shape[:3], math.nan)),
        dtype=np.float64,
    )
    first_second_tv_tensor = np.asarray(
        data["chain_cold_sector_histogram_first_second_tv_per_disorder_per_start_replica_tensor"],
        dtype=np.float64,
    )
    adjacent_block_tv_tensor = np.asarray(
        data["chain_cold_sector_histogram_adjacent_block_tv_per_disorder_per_start_replica_tensor"],
        dtype=np.float64,
    )
    chain_block_range_tensor = np.asarray(
        data["chain_q_top_block_range_per_disorder_per_start_replica_tensor"],
        dtype=np.float64,
    )
    never_tensor = np.asarray(
        data.get("num_chains_that_never_flipped_sector_per_disorder_tensor", []),
        dtype=np.int64,
    )
    entries = []
    label = _label_from_path(path)
    start_sector_labels = _start_sector_labels_from_npz(data)
    for lattice_index in range(chain_tensor.shape[0]):
        lattice_size = (
            int(lattice_sizes[lattice_index])
            if lattice_sizes.size > lattice_index else lattice_index
        )
        for p_index in range(chain_tensor.shape[1]):
            for disorder_index in range(chain_tensor.shape[2]):
                chain_counts = chain_tensor[
                    lattice_index,
                    p_index,
                    disorder_index,
                ].reshape(-1, chain_tensor.shape[-1])
                entries.append(SectorEntry(
                    path=path,
                    label=label,
                    lattice_size=lattice_size,
                    q_value=q_value,
                    disorder_index=disorder_index,
                    q_top=float(q_top_tensor[
                        lattice_index,
                        p_index,
                        disorder_index,
                    ]),
                    q_top_spread=float(q_top_spread_tensor[
                        lattice_index,
                        p_index,
                        disorder_index,
                    ]),
                    chain_counts=chain_counts,
                    pooled_counts=pooled_tensor[
                        lattice_index,
                        p_index,
                        disorder_index,
                    ],
                    first_second_tv=first_second_tv_tensor[
                        lattice_index,
                        p_index,
                        disorder_index,
                    ].reshape(-1),
                    adjacent_block_tv=adjacent_block_tv_tensor[
                        lattice_index,
                        p_index,
                        disorder_index,
                    ].reshape(-1),
                    chain_block_range=chain_block_range_tensor[
                        lattice_index,
                        p_index,
                        disorder_index,
                    ].reshape(-1),
                    never_flipped_count=(
                        int(never_tensor[
                            lattice_index,
                            p_index,
                            disorder_index,
                        ])
                        if never_tensor.size else None
                    ),
                    start_sector_labels=start_sector_labels,
                ))
    return entries


def _npz_paths_from_inputs(inputs: list[Path]) -> list[Path]:
    paths = []
    for input_path in inputs:
        if input_path.is_dir():
            chunk_paths = sorted(input_path.glob("run_q*/chunks/*.npz"))
            if chunk_paths:
                paths.extend(chunk_paths)
                continue
            chunk_paths = sorted(input_path.glob("chunks/*.npz"))
            if chunk_paths:
                paths.extend(chunk_paths)
                continue
            paths.extend(sorted(input_path.glob("*.npz")))
        else:
            paths.append(input_path)
    return paths


def _load_entries(inputs: list[Path]) -> list[SectorEntry]:
    entries = []
    for path in _npz_paths_from_inputs(inputs):
        with np.load(path, allow_pickle=False) as data:
            chunk_entries = _entry_from_chunk_npz(path, data)
            if chunk_entries:
                entries.extend(chunk_entries)
                continue
            tensor_entries = _entries_from_tensor_npz(path, data)
            if tensor_entries:
                entries.extend(tensor_entries)
                continue
            raise KeyError(f"{path} has no cold-sector histogram diagnostics")
    return entries


def _mean_sem(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return math.nan, math.nan
    mean = float(np.mean(values))
    if values.size == 1:
        return mean, 0.0
    return mean, float(np.std(values, ddof=1) / math.sqrt(values.size))


def _write_report(
        entries: list[SectorEntry],
        bootstrap_replicates: int,
        seed: int,
        tv_epsilon: float) -> str:
    rng = np.random.default_rng(seed)
    rows = []
    for entry in entries:
        observed_tv = _max_pairwise_tv(entry.chain_counts)
        p95, p99 = _bootstrap_max_pairwise_tv_quantiles(
            entry.chain_counts,
            rng,
            bootstrap_replicates,
        )
        gate_failed = (
            np.isfinite(p99)
            and observed_tv > p99 + tv_epsilon
        )
        rows.append({
            "entry": entry,
            "observed_tv": observed_tv,
            "bootstrap_p95": p95,
            "bootstrap_p99": p99,
            "gate_failed": gate_failed,
            "top_sector": _top_sector_text(entry.pooled_counts, top_k=3),
            "top_sector_index": int(np.argmax(entry.pooled_counts)),
            "top_sector_mass": float(np.max(_probabilities(entry.pooled_counts))),
            "first_second_tv_max": float(np.nanmax(entry.first_second_tv)),
            "first_second_tv_mean": float(np.nanmean(entry.first_second_tv)),
            "adjacent_block_tv_max": float(np.nanmax(entry.adjacent_block_tv)),
            "chain_block_range_max": float(np.nanmax(entry.chain_block_range)),
        })

    lines = []
    lines.append("# exp36 sector histogram gate")
    lines.append("")
    lines.append("## Gate Definition")
    lines.append("")
    lines.append(
        "- For each fixed disorder, compare cold-chain sector histograms "
        "from different initial states."
    )
    lines.append(
        "- The statistic is the maximum pairwise total-variation distance "
        "between start-chain histograms."
    )
    lines.append(
        "- The reference scale is a parametric bootstrap from the pooled "
        "sector histogram with the same per-chain sample counts."
    )
    lines.append(
        "- A disorder is flagged when observed max TV is larger than the "
        "bootstrap p99 plus the configured epsilon."
    )
    lines.append("")
    lines.append(f"- bootstrap replicates: {bootstrap_replicates}")
    lines.append(f"- TV epsilon: {tv_epsilon:g}")
    lines.append("")

    groups = {}
    for row in rows:
        entry = row["entry"]
        groups.setdefault((entry.q_value, entry.lattice_size), []).append(row)

    lines.append("## Initial-State Coverage")
    lines.append("")
    lines.append(
        "| q | L | compared chains | logical-sector start coverage | start labels |"
    )
    lines.append("|---:|---:|---:|---:|---|")
    for (q_value, lattice_size), group_rows in sorted(groups.items()):
        label_sets = {
            row["entry"].start_sector_labels
            for row in group_rows
        }
        num_chains = int(group_rows[0]["entry"].chain_counts.shape[0])
        label_texts = []
        coverage_texts = []
        for labels in sorted(label_sets, key=lambda item: _start_label_text(item)):
            label_texts.append(_start_label_text(labels))
            coverage_texts.append(_start_coverage_text(labels, num_chains))
        lines.append(
            f"| {q_value:.3f} | {lattice_size} | {num_chains} | "
            f"{'; '.join(coverage_texts)} | {'; '.join(label_texts)} |"
        )
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append(
        "| q | L | disorders | q_top mean | q_top SEM | "
        "start-TV mean | start-TV max | boot-p99 median | boot-p99 max | "
        "TV fails | q_top spread max | first/second TV max | "
        "block q_top range max | never-flipped mean/max | dominant sectors |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for (q_value, lattice_size), group_rows in sorted(groups.items()):
        q_top_values = np.array([row["entry"].q_top for row in group_rows])
        q_top_mean, q_top_sem = _mean_sem(q_top_values)
        observed_tvs = np.array([row["observed_tv"] for row in group_rows])
        bootstrap_p99 = np.array([row["bootstrap_p99"] for row in group_rows])
        q_top_spreads = np.array([
            row["entry"].q_top_spread
            for row in group_rows
        ])
        first_second_max = np.array([
            row["first_second_tv_max"]
            for row in group_rows
        ])
        block_range_max = np.array([
            row["chain_block_range_max"]
            for row in group_rows
        ])
        never_values = np.array([
            row["entry"].never_flipped_count
            for row in group_rows
            if row["entry"].never_flipped_count is not None
        ], dtype=np.float64)
        sector_counts = {}
        for row in group_rows:
            sector_counts[row["top_sector"]] = (
                sector_counts.get(row["top_sector"], 0) + 1
            )
        dominant = "; ".join(
            f"{sector} ({count})"
            for sector, count in sorted(
                sector_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )[:3]
        )
        if never_values.size:
            never_text = (
                f"{float(np.mean(never_values)):.2f}/"
                f"{int(np.max(never_values))}"
            )
        else:
            never_text = "n/a"
        lines.append(
            f"| {q_value:.3f} | {lattice_size} | {len(group_rows)} | "
            f"{q_top_mean:.6f} | {q_top_sem:.6f} | "
            f"{float(np.mean(observed_tvs)):.4f} | "
            f"{float(np.max(observed_tvs)):.4f} | "
            f"{float(np.nanmedian(bootstrap_p99)):.4f} | "
            f"{float(np.nanmax(bootstrap_p99)):.4f} | "
            f"{sum(row['gate_failed'] for row in group_rows)} | "
            f"{float(np.nanmax(q_top_spreads)):.6f} | "
            f"{float(np.nanmax(first_second_max)):.4f} | "
            f"{float(np.nanmax(block_range_max)):.4f} | "
            f"{never_text} | {dominant} |"
        )

    failing_rows = [row for row in rows if row["gate_failed"]]
    lines.append("")
    lines.append("## Flagged Disorders")
    lines.append("")
    if not failing_rows:
        lines.append("No disorder exceeded the bootstrap p99 start-sector TV gate.")
    else:
        lines.append(
            "| q | L | disorder | observed TV | boot p99 | q_top | "
            "q_top spread | top sectors | file |"
        )
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---|---|")
        for row in failing_rows:
            entry = row["entry"]
            lines.append(
                f"| {entry.q_value:.3f} | {entry.lattice_size} | "
                f"{entry.disorder_index} | {row['observed_tv']:.4f} | "
                f"{row['bootstrap_p99']:.4f} | {entry.q_top:.6f} | "
                f"{entry.q_top_spread:.6f} | {row['top_sector']} | "
                f"`{entry.path}` |"
            )

    lines.append("")
    lines.append("## Sample Disorder Rows")
    lines.append("")
    lines.append(
        "| q | L | disorder | q_top | observed TV | boot p99 | "
        "first/second TV max | top sectors | file |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---|---|")
    for row in rows[: min(24, len(rows))]:
        entry = row["entry"]
        lines.append(
            f"| {entry.q_value:.3f} | {entry.lattice_size} | "
            f"{entry.disorder_index} | {entry.q_top:.6f} | "
            f"{row['observed_tv']:.4f} | {row['bootstrap_p99']:.4f} | "
            f"{row['first_second_tv_max']:.4f} | {row['top_sector']} | "
            f"`{entry.path}` |"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=200)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--tv-epsilon", type=float, default=1e-12)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    entries = _load_entries(args.inputs)
    report = _write_report(
        entries=entries,
        bootstrap_replicates=int(args.bootstrap_replicates),
        seed=int(args.seed),
        tv_epsilon=float(args.tv_epsilon),
    )
    if args.output is None:
        print(report, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
