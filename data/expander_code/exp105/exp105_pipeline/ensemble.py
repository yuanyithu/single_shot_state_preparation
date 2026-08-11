"""Seed-derived expander-code ensemble and its composition census.

The ensemble is a rule, not a directory of files. A code is fully determined by
its graph seed, so the registry records provenance and structure only; the
parity-check matrix and the logical frame are rebuilt on demand and checked
against the recorded hashes.

Acceptance uses two algebraic criteria and nothing else: the bipartite graph
must be simple, and H must have full row rank so that every accepted code has
k = m^2. Neither criterion looks at decoder behaviour, and no code is ever
dropped after its failure rate is known.

exp105 differs from exp104 in two mechanical ways. Panels are unequal across m,
because the variance of the primary contrast is minimised that way at fixed
cost. And the registry is a compact columnar NPZ rather than JSON: exp104's
12,000-row JSON is already 3.3 MB, and exp105 holds of order 10^5 rows.
"""

import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np

from . import CENSUS_SCHEMA, EXPERIMENT_ID, REGISTRY_SCHEMA
from .config import (
    ANCHOR_M_VALUES,
    ENSEMBLE,
    MASTER_SEED_HEX,
    NAMESPACES,
    code_id,
)
from .exp101_bridge import load_exp101
from .io import arrays_sha256, atomic_json, atomic_npz, canonical_json, sha256_json
from .seeds import _digest_seed


def _exp101():
    load_exp101()
    from exp101_certified_src.gf2 import gf2_rank
    from exp101_certified_src.graphs import random_biregular_graph_from_m
    from exp101_certified_src.hgp import classical_parity_check_matrix
    from exp101_certified_src.params import classical_code_distance

    return (
        random_biregular_graph_from_m,
        classical_parity_check_matrix,
        gf2_rank,
        classical_code_distance,
    )


def matrix_sha256(matrix):
    matrix = np.ascontiguousarray(matrix, dtype=np.uint8)
    header = np.asarray(matrix.shape, dtype=np.int64).tobytes()
    return hashlib.sha256(header + matrix.tobytes()).hexdigest()


def _candidate_seed_from_parts(master_seed_hex, namespace, m, candidate_index):
    return _digest_seed(
        ":".join([master_seed_hex, namespace, str(int(m)), str(int(candidate_index))])
    )


def build_candidate(master_seed_hex, namespace, m, candidate_index):
    """Construct one candidate and report why it was accepted or rejected."""
    random_graph, parity_check, rank_of, _ = _exp101()
    seed = _candidate_seed_from_parts(master_seed_hex, namespace, m, candidate_index)
    graph = random_graph(
        m, ENSEMBLE["d_A"], ENSEMBLE["d_B"], seed,
        max_attempts=ENSEMBLE["max_attempts"],
    )
    H = parity_check(graph)
    rank = int(rank_of(H))
    return {
        "candidate_index": int(candidate_index),
        "graph_seed": int(seed),
        "construction_attempts": int(graph.construction_attempts),
        "classical_rank": rank,
        "full_row_rank": rank == 3 * m,
        "classical_H_sha256": matrix_sha256(H),
        "H": H,
    }


def rebuild_code(row):
    """Rebuild an accepted code from its frozen seed and verify every hash."""
    random_graph, parity_check, rank_of, _ = _exp101()
    graph = random_graph(
        int(row["m"]), ENSEMBLE["d_A"], ENSEMBLE["d_B"], int(row["graph_seed"]),
        max_attempts=ENSEMBLE["max_attempts"],
    )
    if int(graph.construction_attempts) != int(row["construction_attempts"]):
        raise ValueError(f"seed reconstruction mismatch: {row['code_id']}")
    H = parity_check(graph)
    if matrix_sha256(H) != row["classical_H_sha256"]:
        raise ValueError(f"parity-check matrix hash mismatch: {row['code_id']}")
    rank = int(rank_of(H))
    if rank != int(row["classical_rank"]) or rank != 3 * int(row["m"]):
        raise ValueError(f"classical rank mismatch: {row['code_id']}")
    return H


def generate_codes(m, count, master_seed_hex=MASTER_SEED_HEX,
                   namespace=NAMESPACES["ensemble"], with_distance=True):
    """Scan candidates in order and accept the first `count` legal codes."""
    _, _, _, distance_of = _exp101()
    accepted = []
    seen = set()
    candidate_index = 0
    rejected = collections.Counter()
    while len(accepted) < count:
        candidate = build_candidate(master_seed_hex, namespace, m, candidate_index)
        candidate_index += 1
        if not candidate["full_row_rank"]:
            rejected["rank_deficient"] += 1
            continue
        if candidate["classical_H_sha256"] in seen:
            rejected["duplicate_matrix"] += 1
            continue
        seen.add(candidate["classical_H_sha256"])
        row = {
            "code_id": code_id(m, len(accepted)),
            "m": int(m),
            "code_index": len(accepted),
            "candidate_index": candidate["candidate_index"],
            "graph_seed": candidate["graph_seed"],
            "construction_attempts": candidate["construction_attempts"],
            "classical_H_sha256": candidate["classical_H_sha256"],
            "classical_rank": candidate["classical_rank"],
            "n": 25 * int(m) ** 2,
            "k": int(m) ** 2,
        }
        if with_distance:
            row["classical_distance"] = int(distance_of(candidate["H"]))
        accepted.append(row)
    return accepted, {
        "candidates_scanned": candidate_index,
        "accepted": len(accepted),
        "acceptance_rate": len(accepted) / candidate_index,
        "rejected": dict(rejected),
    }


# Columnar layout of the registry NPZ. The order is part of the hashed
# identity, so it is written once here and never derived from a dict order.
_INT_COLUMNS = (
    "m", "code_index", "candidate_index", "graph_seed",
    "construction_attempts", "classical_rank", "n", "k", "classical_distance",
)
_STR_COLUMNS = ("code_id", "classical_H_sha256")
REGISTRY_COLUMNS = _INT_COLUMNS + _STR_COLUMNS
ROW_FIELDS = tuple(sorted(REGISTRY_COLUMNS))


def _columns_from_rows(rows):
    columns = {
        name: np.asarray([int(row[name]) for row in rows], dtype=np.int64)
        for name in _INT_COLUMNS
    }
    columns.update({
        name: np.asarray([str(row[name]) for row in rows], dtype="U64")
        for name in _STR_COLUMNS
    })
    return columns


def _registry_digest(metadata, columns):
    return hashlib.sha256(
        canonical_json(metadata).encode("ascii")
        + b"\0"
        + arrays_sha256(columns, list(REGISTRY_COLUMNS)).encode("ascii")
    ).hexdigest()


def build_registry(output_path, codes_per_m, master_seed_hex=MASTER_SEED_HEX,
                   namespace=NAMESPACES["ensemble"], progress=None):
    """Build the frozen registry with a per-m panel size.

    `codes_per_m` maps m to its panel size; sizes are unequal by design.

    `namespace` separates ensembles that must not share codes. The locating
    pilot draws from `NAMESPACES["pilot"]` rather than from the production
    namespace, so no code that helps choose the frozen grid is also a code the
    frozen grid is later measured on.
    """
    codes_per_m = {int(m): int(count) for m, count in codes_per_m.items()}
    rows = []
    audit = {}
    for m in sorted(codes_per_m):
        panel, stats = generate_codes(
            m, codes_per_m[m], master_seed_hex, namespace=namespace,
        )
        rows.extend(panel)
        audit[str(m)] = stats
        if progress is not None:
            progress(m, stats)
    metadata = {
        "schema_version": REGISTRY_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "master_seed_hex": master_seed_hex,
        "seed_namespace": namespace,
        "ensemble": ENSEMBLE,
        "m_values": sorted(codes_per_m),
        "codes_per_m": {str(m): codes_per_m[m] for m in sorted(codes_per_m)},
        "columns": list(REGISTRY_COLUMNS),
        "acceptance_audit": audit,
    }
    columns = _columns_from_rows(rows)
    digest = _registry_digest(metadata, columns)
    if output_path is not None:
        atomic_npz(output_path, dict(
            columns,
            metadata_json=np.asarray(canonical_json(metadata)),
            registry_sha256=np.asarray(digest),
        ))
    return {"metadata": metadata, "columns": columns, "registry_sha256": digest,
            "codes": rows}


def load_registry(path, verify_seeds=False):
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
        claimed = str(data["registry_sha256"])
        columns = {name: np.asarray(data[name]) for name in REGISTRY_COLUMNS}
    if metadata["schema_version"] != REGISTRY_SCHEMA:
        raise ValueError("unexpected exp105 registry schema")
    if metadata["columns"] != list(REGISTRY_COLUMNS):
        raise ValueError("exp105 registry column order is not the frozen order")
    if _registry_digest(metadata, columns) != claimed:
        raise ValueError("exp105 registry SHA256 mismatch")
    counts = {int(m): int(c) for m, c in metadata["codes_per_m"].items()}
    expected = [
        code_id(m, index)
        for m in sorted(counts)
        for index in range(counts[m])
    ]
    ids = [str(value) for value in columns["code_id"]]
    if ids != expected:
        raise ValueError("exp105 registry code IDs are incomplete, reordered or duplicated")
    registry = {
        "metadata": metadata,
        "columns": columns,
        "registry_sha256": claimed,
        "codes": rows_from_columns(columns),
    }
    if verify_seeds:
        for row in registry["codes"]:
            rebuild_code(row)
    return registry


def rows_from_columns(columns):
    total = len(columns["code_id"])
    rows = []
    for index in range(total):
        row = {name: int(columns[name][index]) for name in _INT_COLUMNS}
        row.update({name: str(columns[name][index]) for name in _STR_COLUMNS})
        rows.append(row)
    return rows


def registry_index(registry):
    return {row["code_id"]: row for row in registry["codes"]}


def census(m, accepted_target, master_seed_hex, namespace):
    """Exact classical-distance distribution of the accepted ensemble at one m.

    This costs a 2^(n_v - rank) codeword enumeration per code and it is what
    makes the primary result interpretable: the distance-2 fraction falls with
    m, so a small panel estimates the composition to no useful accuracy.
    """
    _, _, _, distance_of = _exp101()
    counts = collections.Counter()
    attempts = []
    seen = set()
    accepted = 0
    candidate_index = 0
    rejected = collections.Counter()
    while accepted < accepted_target:
        candidate = build_candidate(master_seed_hex, namespace, m, candidate_index)
        candidate_index += 1
        if not candidate["full_row_rank"]:
            rejected["rank_deficient"] += 1
            continue
        if candidate["classical_H_sha256"] in seen:
            rejected["duplicate_matrix"] += 1
            continue
        seen.add(candidate["classical_H_sha256"])
        counts[int(distance_of(candidate["H"]))] += 1
        attempts.append(candidate["construction_attempts"])
        accepted += 1
    return {
        "m": int(m),
        "accepted": accepted,
        "candidates_scanned": candidate_index,
        "acceptance_rate": accepted / candidate_index,
        "rejected": dict(rejected),
        "mean_simple_graph_attempts": float(np.mean(attempts)),
        "distance_counts": {str(d): int(c) for d, c in sorted(counts.items())},
        "distance_fractions": {
            str(d): float(c) / accepted for d, c in sorted(counts.items())
        },
    }


def run_census(output_path, accepted_target, master_seed_hex=MASTER_SEED_HEX,
               namespace=NAMESPACES["ensemble"], m_values=None, progress=None):
    if m_values is None:
        m_values = sorted(set(ANCHOR_M_VALUES) | {3, 4, 5, 6, 7, 8})
    per_m = {}
    for m in m_values:
        per_m[str(m)] = census(m, accepted_target, master_seed_hex, namespace)
        if progress is not None:
            progress(m, per_m[str(m)])
    core = {
        "schema_version": CENSUS_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "master_seed_hex": master_seed_hex,
        "seed_namespace": namespace,
        "ensemble": ENSEMBLE,
        "accepted_per_m": int(accepted_target),
        "per_m": per_m,
    }
    report = dict(core, census_sha256=sha256_json(core))
    if output_path is not None:
        atomic_json(output_path, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="exp105 ensemble construction")
    sub = parser.add_subparsers(dest="command", required=True)

    registry = sub.add_parser("registry", help="build the frozen code registry")
    registry.add_argument("output")
    registry.add_argument(
        "--codes-per-m", required=True,
        help='JSON object mapping m to its panel size, e.g. \'{"3": 70000}\'',
    )
    registry.add_argument("--namespace", default=NAMESPACES["ensemble"])

    counting = sub.add_parser("census", help="measure the ensemble composition")
    counting.add_argument("output")
    counting.add_argument("--accepted-per-m", type=int, default=20000)
    counting.add_argument("--master-seed-hex", default=MASTER_SEED_HEX)
    counting.add_argument("--namespace", default=NAMESPACES["ensemble"])
    counting.add_argument("--m-values", type=int, nargs="+", default=None)

    args = parser.parse_args(argv)
    if args.command == "registry":
        result = build_registry(
            args.output, json.loads(args.codes_per_m), namespace=args.namespace,
            progress=lambda m, stats: print(
                f"m={m} accepted={stats['accepted']} "
                f"acceptance_rate={stats['acceptance_rate']:.4f} "
                f"rejected={stats['rejected']}", flush=True,
            ),
        )
        print(result["registry_sha256"])
    else:
        result = run_census(
            args.output, args.accepted_per_m, args.master_seed_hex, args.namespace,
            m_values=args.m_values,
            progress=lambda m, row: print(
                f"m={m} accept_rate={row['acceptance_rate']:.4f} "
                f"d_fractions={row['distance_fractions']}", flush=True,
            ),
        )
        print(result["census_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
