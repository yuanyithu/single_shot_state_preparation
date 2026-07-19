import argparse
import csv
import hashlib
import json
import secrets
from pathlib import Path

import numpy as np

from .exp101_bridge import load_exp101
from .io import atomic_json, atomic_npz, canonical_json, sha256_json


def candidate_seed(master_seed_hex, m, candidate_index):
    payload = f"exp102-registry-v1:{master_seed_hex}:{int(m)}:{int(candidate_index)}"
    return int.from_bytes(hashlib.sha256(payload.encode("ascii")).digest()[:8], "big") & ((1 << 63) - 1)


def matrix_hash(matrix):
    matrix = np.ascontiguousarray(matrix, dtype=np.uint8)
    header = np.asarray(matrix.shape, dtype=np.int64).tobytes()
    return hashlib.sha256(header + matrix.tobytes()).hexdigest()


def code_fingerprints(H):
    load_exp101()
    from exp101_certified_src.gf2 import gf2_rank
    from exp101_certified_src.hgp import hgp_from_H
    from exp101_certified_src.logicals import logical_pauli_operators
    from exp101_certified_src.section import build_linear_section

    H_Z, H_X = hgp_from_H(H)
    logicals = logical_pauli_operators(H_X, H_Z)
    section = build_linear_section(H_Z)
    digest = hashlib.sha256()
    for array in (H_Z, H_X, logicals.logical_X, logicals.logical_Z):
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(array, dtype=np.uint8).tobytes())
    digest.update(section.fingerprint().encode("ascii"))
    return {
        "n": int(H_Z.shape[1]), "k": int(logicals.k),
        "rank_H_X": int(gf2_rank(H_X)), "rank_H_Z": int(gf2_rank(H_Z)),
        "section_fingerprint": section.fingerprint(),
        "logical_frame_fingerprint": digest.hexdigest(),
    }


def build_registry(output_dir, master_seed_hex=None, include_frames=True):
    load_exp101()
    from exp101_certified_src.gf2 import gf2_rank
    from exp101_certified_src.graphs import random_biregular_graph_from_m
    from exp101_certified_src.hgp import classical_parity_check_matrix
    from exp101_certified_src.params import classical_code_distance

    output_dir = Path(output_dir)
    codes_dir = output_dir / "codes"
    codes_dir.mkdir(parents=True, exist_ok=True)
    if master_seed_hex is None:
        master_seed_hex = secrets.token_hex(32)
    if len(master_seed_hex) != 64:
        raise ValueError("master seed must be 32 bytes encoded as 64 hex characters")
    int(master_seed_hex, 16)
    records, audit, seen = [], [], set()
    for m in range(3, 9):
        accepted = 0
        candidate_index = 0
        while accepted < 8:
            seed = candidate_seed(master_seed_hex, m, candidate_index)
            graph = random_biregular_graph_from_m(m, 3, 4, seed, max_attempts=10000)
            H = classical_parity_check_matrix(graph)
            rank = gf2_rank(H)
            H_hash = matrix_hash(H)
            reason = "accepted"
            if rank != 3 * m:
                reason = "rank_deficient"
            elif H_hash in seen:
                reason = "duplicate_matrix"
            audit.append({"m": m, "candidate_index": candidate_index, "graph_seed": seed,
                          "construction_attempts": graph.construction_attempts,
                          "rank": rank, "classical_H_sha256": H_hash, "decision": reason})
            if reason == "accepted":
                code_id = f"m{m:02d}_c{accepted:02d}"
                edge_a = np.array([a for a, b in sorted(graph.edge_set())], dtype=np.int16)
                edge_b = np.array([b for a, b in sorted(graph.edge_set())], dtype=np.int16)
                metadata = {
                    "code_id": code_id, "m": m, "candidate_index": candidate_index,
                    "graph_seed": seed, "construction_attempts": graph.construction_attempts,
                    "classical_H_sha256": H_hash, "classical_rank": rank,
                    "classical_distance": int(classical_code_distance(H)),
                }
                if include_frames:
                    metadata.update(code_fingerprints(H))
                atomic_npz(codes_dir / f"{code_id}.npz", H=H, edge_a=edge_a, edge_b=edge_b)
                metadata["code_npz_sha256"] = hashlib.sha256((codes_dir / f"{code_id}.npz").read_bytes()).hexdigest()
                records.append(metadata)
                seen.add(H_hash)
                accepted += 1
            candidate_index += 1
    core = {"registry_version": "exp102.registry.v1", "master_seed_hex": master_seed_hex,
            "selection_rule": "first_8_simple_full_row_rank_unique_H_per_m", "codes": records}
    registry_hash = sha256_json(core)
    registry = dict(core, registry_sha256=registry_hash)
    atomic_json(output_dir / "registry.json", registry)
    _write_csv(output_dir / "code_registry.csv", records)
    _write_csv(output_dir / "candidate_audit.csv", audit)
    return registry


def _write_csv(path, rows):
    with open(path, "w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_registry(path, verify_files=True):
    path = Path(path)
    registry = json.loads(path.read_text(encoding="ascii"))
    claimed = registry.pop("registry_sha256", None)
    actual = sha256_json(registry)
    registry["registry_sha256"] = claimed
    if claimed != actual:
        raise ValueError("registry SHA256 mismatch")
    if len(registry["codes"]) != 48:
        raise ValueError("registry must contain exactly 48 codes")
    expected = {f"m{m:02d}_c{c:02d}" for m in range(3, 9) for c in range(8)}
    if {row["code_id"] for row in registry["codes"]} != expected:
        raise ValueError("registry code IDs are incomplete or duplicated")
    if verify_files:
        for row in registry["codes"]:
            code_path = path.parent / "codes" / f'{row["code_id"]}.npz'
            if hashlib.sha256(code_path.read_bytes()).hexdigest() != row["code_npz_sha256"]:
                raise ValueError(f"code file hash mismatch: {row['code_id']}")
            with np.load(code_path, allow_pickle=False) as data:
                if matrix_hash(data["H"]) != row["classical_H_sha256"]:
                    raise ValueError(f"matrix hash mismatch: {row['code_id']}")
    return registry


def load_frozen_code(registry_path, code_id):
    """Rebuild exactly one graph from its frozen seed and verify every stored representation."""
    load_exp101()
    from exp101_certified_src.graphs import random_biregular_graph_from_m
    from exp101_certified_src.hgp import classical_parity_check_matrix
    registry_path = Path(registry_path)
    registry = load_registry(registry_path, verify_files=False)
    row = next((item for item in registry["codes"] if item["code_id"] == code_id), None)
    if row is None:
        raise ValueError(f"unknown frozen code ID {code_id!r}")
    graph = random_biregular_graph_from_m(row["m"], 3, 4, row["graph_seed"], max_attempts=10000)
    rebuilt = classical_parity_check_matrix(graph)
    if graph.construction_attempts != row["construction_attempts"] or matrix_hash(rebuilt) != row["classical_H_sha256"]:
        raise ValueError(f"seed reconstruction mismatch: {code_id}")
    code_path = registry_path.parent / "codes" / f"{code_id}.npz"
    if hashlib.sha256(code_path.read_bytes()).hexdigest() != row["code_npz_sha256"]:
        raise ValueError(f"code file hash mismatch: {code_id}")
    with np.load(code_path, allow_pickle=False) as data:
        stored = data["H"].copy()
    if not np.array_equal(stored, rebuilt):
        raise ValueError(f"stored H differs from seed reconstruction: {code_id}")
    return registry, row, stored


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    parser.add_argument("--master-seed-hex")
    parser.add_argument("--skip-frame-fingerprints", action="store_true")
    args = parser.parse_args(argv)
    result = build_registry(args.output_dir, args.master_seed_hex, not args.skip_frame_fingerprints)
    print(result["registry_sha256"])


if __name__ == "__main__":
    main()
