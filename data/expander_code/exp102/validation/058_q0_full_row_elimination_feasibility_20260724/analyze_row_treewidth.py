"""Compute deterministic elimination-width bounds for the m8 row conditional."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
CONTROL = (
    ROOT.parent
    / "056_q0_random_full_column_direct_block_t1_m8_v2_20260724"
    / "control/control.npz"
)
OUTPUT = ROOT / "row_elimination_width.json"


def _interaction_graph(H):
    adjacency = [set() for _ in range(H.shape[0])]
    scopes = []
    for column in range(H.shape[1]):
        scope = tuple(int(value) for value in np.flatnonzero(H[:, column]))
        scopes.append(scope)
        for left in scope:
            adjacency[left].update(value for value in scope if value != left)
    return adjacency, scopes


def _eliminate(adjacency, mode):
    graph = [set(values) for values in adjacency]
    remaining = set(range(len(graph)))
    order = []
    widths = []
    table_entries = []
    while remaining:
        candidates = []
        for variable in sorted(remaining):
            neighbors = sorted(graph[variable] & remaining)
            missing = sum(
                right not in graph[left]
                for index, left in enumerate(neighbors)
                for right in neighbors[index + 1:]
            )
            if mode == "min_fill":
                key = (missing, len(neighbors), variable)
            elif mode == "min_degree":
                key = (len(neighbors), missing, variable)
            else:
                raise ValueError("unknown elimination heuristic")
            candidates.append((key, variable, neighbors))
        _, variable, neighbors = min(candidates)
        order.append(variable)
        widths.append(len(neighbors))
        table_entries.append(1 << (len(neighbors) + 1))
        for index, left in enumerate(neighbors):
            for right in neighbors[index + 1:]:
                graph[left].add(right)
                graph[right].add(left)
        remaining.remove(variable)
    return {
        "heuristic": mode,
        "induced_width": max(widths, default=0),
        "largest_factor_entries": max(table_entries, default=1),
        "order": order,
        "sum_factor_entries": int(sum(table_entries)),
        "width_by_step": widths,
    }


def main():
    with np.load(CONTROL, allow_pickle=False) as archive:
        H = archive["H"].copy()
        metadata = json.loads(str(archive["metadata_json"].item()))
    adjacency, scopes = _interaction_graph(H)
    results = [_eliminate(adjacency, mode) for mode in ("min_fill", "min_degree")]
    best = min(results, key=lambda value: (
        value["induced_width"], value["sum_factor_entries"], value["order"],
    ))
    payload = {
        "cell_fingerprint": metadata["cell_fingerprint"],
        "code_id": metadata["cell"]["code_id"],
        "factor_count": len(scopes),
        "factor_scopes": [list(scope) for scope in scopes],
        "gate": {
            "max_induced_width": 18,
            "max_largest_factor_entries": 1 << 19,
        },
        "gate_pass": bool(
            best["induced_width"] <= 18
            and best["largest_factor_entries"] <= 1 << 19
        ),
        "h_sha256": hashlib.sha256(
            np.asarray(H.shape, dtype=">u8").tobytes() + H.tobytes()
        ).hexdigest(),
        "heuristics": results,
        "selected_order": best["order"],
        "selected_order_sha256": hashlib.sha256(
            np.asarray(best["order"], dtype=">u4").tobytes()
        ).hexdigest(),
        "version": "exp102.q0_full_row_elimination.width.v0",
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["report_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    OUTPUT.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
