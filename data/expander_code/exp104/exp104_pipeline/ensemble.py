"""Seed-derived expander-code ensemble and its composition census.

The ensemble is a rule, not a directory of files. A code is fully determined by
its graph seed, so the registry manifest records provenance and structure only;
the parity-check matrix and the logical frame are rebuilt on demand and checked
against the recorded hashes.

Acceptance uses two algebraic criteria and nothing else: the bipartite graph
must be simple, and H must have full row rank so that every accepted code has
k = m^2. Neither criterion looks at decoder behaviour, and no code is ever
dropped after its failure rate is known.
"""

import argparse
import hashlib
import collections
from pathlib import Path

import numpy as np

from data.expander_code.exp102.exp102_pipeline.exp101_bridge import load_exp101

from . import CENSUS_SCHEMA, EXPERIMENT_ID, REGISTRY_SCHEMA
from .config import (
    CODES_PER_M,
    ENSEMBLE,
    M_VALUES,
    MASTER_SEED_HEX,
    NAMESPACES,
    code_id,
)
from .io import atomic_json, sha256_json
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


def build_registry(output_path, master_seed_hex=MASTER_SEED_HEX, codes_per_m=CODES_PER_M,
                   m_values=M_VALUES, progress=None):
    codes = []
    audit = {}
    for m in m_values:
        rows, stats = generate_codes(m, codes_per_m, master_seed_hex)
        codes.extend(rows)
        audit[str(m)] = stats
        if progress is not None:
            progress(m, stats)
    core = {
        "schema_version": REGISTRY_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "master_seed_hex": master_seed_hex,
        "seed_namespace": NAMESPACES["ensemble"],
        "ensemble": ENSEMBLE,
        "m_values": list(m_values),
        "codes_per_m": int(codes_per_m),
        "acceptance_audit": audit,
        "codes": codes,
    }
    registry = dict(core, registry_sha256=sha256_json(core))
    if output_path is not None:
        atomic_json(output_path, registry)
    return registry


def load_registry(path, verify_seeds=False):
    import json

    path = Path(path)
    registry = json.loads(path.read_text(encoding="ascii"))
    claimed = registry.pop("registry_sha256", None)
    actual = sha256_json(registry)
    registry["registry_sha256"] = claimed
    if claimed != actual:
        raise ValueError("exp104 registry SHA256 mismatch")
    if registry["schema_version"] != REGISTRY_SCHEMA:
        raise ValueError("unexpected exp104 registry schema")
    expected = {
        code_id(m, index)
        for m in registry["m_values"]
        for index in range(registry["codes_per_m"])
    }
    if {row["code_id"] for row in registry["codes"]} != expected:
        raise ValueError("exp104 registry code IDs are incomplete or duplicated")
    if len(registry["codes"]) != len(expected):
        raise ValueError("exp104 registry contains duplicate rows")
    if verify_seeds:
        for row in registry["codes"]:
            rebuild_code(row)
    return registry


def registry_index(registry):
    return {row["code_id"]: row for row in registry["codes"]}


def census(m, accepted_target, master_seed_hex, namespace):
    """Exact classical-distance distribution of the accepted ensemble at one m.

    This costs nothing on the scale of the experiment (2^m codeword enumeration
    per code) and it is what explains exp103: the distance-2 fraction falls with
    m, so an eight-code panel estimates the composition to no useful accuracy.
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
               namespace=NAMESPACES["ensemble"], m_values=M_VALUES, progress=None):
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
    parser = argparse.ArgumentParser(description="exp104 ensemble construction")
    sub = parser.add_subparsers(dest="command", required=True)

    registry = sub.add_parser("registry", help="build the frozen code registry")
    registry.add_argument("output")
    registry.add_argument("--codes-per-m", type=int, default=CODES_PER_M)

    counting = sub.add_parser("census", help="measure the ensemble composition")
    counting.add_argument("output")
    counting.add_argument("--accepted-per-m", type=int, default=200000)
    counting.add_argument("--master-seed-hex", default=MASTER_SEED_HEX)
    counting.add_argument("--namespace", default=NAMESPACES["ensemble"])

    args = parser.parse_args(argv)
    if args.command == "registry":
        result = build_registry(
            args.output, codes_per_m=args.codes_per_m,
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
            progress=lambda m, row: print(
                f"m={m} accept_rate={row['acceptance_rate']:.4f} "
                f"d_fractions={row['distance_fractions']}", flush=True,
            ),
        )
        print(result["census_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
