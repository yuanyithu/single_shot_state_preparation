"""Compute exact full-column bridge probabilities between two frozen MAP B masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from data.expander_code.exp102.exp102_pipeline.io import (
    atomic_json,
    canonical_json,
    sha256_file,
    sha256_json,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_collapsed import (
    build_classical_coset_mass,
    split_hgp_state,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_full_column_gibbs import (
    build_full_column_candidate_cache,
    build_full_column_workspace,
    collapsed_a_syndromes,
    full_column_conditional_probabilities,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_screen import _disorder
from data.expander_code.exp102.exp102_pipeline.registry import load_frozen_code, load_registry
from data.expander_code.exp102.exp102_pipeline.worker import build_model


CONTRACT_VERSION = "exp102.q0_full_column_map_bridge.structure.v0"
ROOT = Path(__file__).resolve().parent
EXP102_ROOT = ROOT.parents[1]
REGISTRY_PATH = EXP102_ROOT / "registry/registry.json"


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _mask(column):
    value = 0
    for bit, entry in enumerate(np.asarray(column, dtype=np.uint8)):
        value |= int(entry) << bit
    return np.uint32(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    serialized = config_path.read_text(encoding="ascii")
    config = json.loads(serialized)
    _require(serialized == canonical_json(config) + "\n", "config is not canonical")
    _require(config["version"] == config["contract_version"] == CONTRACT_VERSION
             and config["config_version"]
             == "exp102.q0_full_column_map_bridge.structure.config.v0",
             "config version changed")
    registry = load_registry(REGISTRY_PATH)
    _require(registry["registry_sha256"] == config["registry_sha256"],
             "registry changed")
    _unused, code, H = load_frozen_code(REGISTRY_PATH, config["cell"]["code_id"])
    model, frame = build_model(H)
    del frame
    _uniform_seed, _planted_unused, syndrome = _disorder(
        registry, code, model, config["cell"],
    )
    artifact_path = EXP102_ROOT / config["artifact"]["relpath"]
    _require(sha256_file(artifact_path) == config["artifact"]["file_sha256"],
             "MAP artifact bytes changed")
    with np.load(artifact_path, allow_pickle=False) as artifact:
        metadata = json.loads(str(artifact["metadata_json"].item()))
        anchors = np.asarray(artifact["anchors"], dtype=np.uint8)
    _require(metadata["cell"] == config["cell"] and anchors.shape == (2, 1600)
             and np.array_equal(anchors.sum(axis=1), [62, 62]),
             "MAP anchor identity changed")
    for anchor in anchors:
        recovered = (
            model.H_check.astype(np.int64) @ anchor.astype(np.int64) % 2
        ).astype(np.uint8)
        _require(np.array_equal(recovered, syndrome), "MAP anchor leaves hard coset")
    blocks = [split_hgp_state(anchor, H)[1] for anchor in anchors]
    difference = blocks[0] ^ blocks[1]
    differing_columns = np.flatnonzero(difference.any(axis=0)).astype(np.int32)
    _require(np.array_equal(differing_columns, [11, 17])
             and int(difference.sum()) == 6,
             "frozen MAP B difference changed")

    mass = build_classical_coset_mass(H, 0.04, engine="numba")
    log_mass = np.log(mass)
    cache = build_full_column_candidate_cache(24, 0.04)
    workspace = build_full_column_workspace(cache)
    y_matrix = syndrome.reshape(H.shape)

    def columns_for(B):
        return np.asarray([_mask(B[:, column]) for column in range(24)], dtype=np.uint32)

    def probability(B, column, target_mask):
        b_columns = columns_for(B)
        a_syndromes = collapsed_a_syndromes(H, y_matrix, b_columns)
        values = full_column_conditional_probabilities(
            H, y_matrix, b_columns, a_syndromes, int(column), log_mass,
            cache=cache, workspace=workspace,
        )
        return float(values[int(target_mask)]), float(values[int(b_columns[int(column)])])

    records = []
    t1 = int(config["gates"]["t1_random_scan_updates"])
    for source in (0, 1):
        target = 1 - source
        for first, second in (
                (int(differing_columns[0]), int(differing_columns[1])),
                (int(differing_columns[1]), int(differing_columns[0]))):
            current = blocks[source].copy()
            target_first = _mask(blocks[target][:, first])
            first_probability, source_self = probability(current, first, target_first)
            current[:, first] = blocks[target][:, first]
            target_second = _mask(blocks[target][:, second])
            second_probability, intermediate_self = probability(current, second, target_second)
            records.append({
                "expected_first_departures_t1": float(t1 * first_probability / 24.0),
                "first_column": first,
                "first_probability_given_column": first_probability,
                "intermediate_self_probability_given_second_column": intermediate_self,
                "second_column": second,
                "second_probability_given_column": second_probability,
                "source_anchor": source,
                "source_self_probability_given_first_column": source_self,
                "target_anchor": target,
                "two_selected_column_probability_product": float(
                    first_probability * second_probability
                ),
            })
    best = {
        str(source): max(
            row["expected_first_departures_t1"]
            for row in records if row["source_anchor"] == source
        )
        for source in (0, 1)
    }
    gate = float(config["gates"]["min_expected_first_bridge_departures_per_anchor"])
    checks = {
        "anchor_0_first_departure": best["0"] >= gate,
        "anchor_1_first_departure": best["1"] >= gate,
        "exact_algebra": True,
    }
    status = (
        "LOCAL_FULL_COLUMN_MAP_BRIDGE_STRUCTURE_VIABLE"
        if all(checks.values())
        else "LOCAL_FULL_COLUMN_MAP_BRIDGE_STRUCTURE_NOT_VIABLE"
    )
    core = {
        "artifact_file_sha256": sha256_file(artifact_path),
        "best_expected_first_departures_t1": best,
        "checks": checks,
        "config_sha256": sha256_file(config_path),
        "contract_version": CONTRACT_VERSION,
        "difference_columns": differing_columns.tolist(),
        "difference_weight": int(difference.sum()),
        "records": records,
        "scope": config["scope"],
        "status": status,
    }
    report = {**core, "report_sha256": sha256_json(core)}
    output = ROOT / "map_bridge_report.json"
    _require(not output.exists(), "map bridge report already exists")
    atomic_json(output, report)
    print(canonical_json({
        "best_expected_first_departures_t1": best,
        "report_sha256": report["report_sha256"], "status": status,
    }))


if __name__ == "__main__":
    main()
