#!/usr/bin/env python3
"""Interpret the audited V0 raw without producing new MCMC samples.

The local leave-return gate is useful for detecting common freezing, but a
low-temperature posterior can legitimately have nearly constant logical
characters.  This diagnostic therefore adds a target-support check: compare
the U-family measurement weights with a known legal low-weight coset state.
It is deliberately diagnostic only and never grants formal authorization.
"""

from __future__ import annotations

import collections
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parent
EXP102_ROOT = ROOT.parents[1]
PROJECT_ROOT = ROOT.parents[4]
RUN_ROOT = ROOT / "local_hard_viability_001"
REGISTRY_PATH = EXP102_ROOT / "registry" / "registry.json"

sys.path.insert(0, str(PROJECT_ROOT))

from data.expander_code.exp102.exp102_pipeline.exp101_bridge import load_exp101
from data.expander_code.exp102.exp102_pipeline.registry import load_frozen_code


DIAGNOSTIC_VERSION = "exp102.q0_hgp_full_row_gibbs.v0.convergence_diagnostic.v1"


class DiagnosticConflict(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise DiagnosticConflict(message)


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value):
    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="ascii"))


def read_raw(path):
    with np.load(path, allow_pickle=False) as archive:
        values = {name: archive[name].copy() for name in archive.files}
    require(not any(value.dtype.hasobject for value in values.values()),
            f"object dtype in {Path(path).name}")
    return values


def atomic_json(path, value):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="ascii") as handle:
            handle.write(canonical_json(value) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def label_summary(labels):
    labels = [int(value) for value in labels]
    counts = collections.Counter(labels)
    total = len(labels)
    collision = sum(value * (value - 1) for value in counts.values()) / (total * (total - 1))
    entropy = -sum((value / total) * math.log2(value / total) for value in counts.values())
    return {
        "sample_count": total,
        "unique_label_count": len(counts),
        "raw_collision_diagnostic": collision,
        "raw_label_entropy_bits": entropy,
        "top_labels": [
            {"label_hex": f"{label:016x}", "count": count}
            for label, count in counts.most_common(8)
        ],
    }


def state_weight(packed, num_qubits):
    return int(np.unpackbits(
        np.asarray(packed, dtype=np.uint8), count=num_qubits, bitorder="little",
    ).sum())


def build_model(H):
    load_exp101()
    from exp101_certified_src.hgp import hgp_from_H
    from exp101_certified_src.logicals import logical_pauli_operators
    from exp101_certified_src.model import assemble_sector_model

    H_Z, H_X = hgp_from_H(H)
    logicals = logical_pauli_operators(H_X, H_Z)
    return assemble_sector_model(H_X, H_Z, logicals, sector="x_error")


def summarize_family(family, raw_paths, num_qubits):
    labels, weights = [], []
    initial, burn, final = [], [], []
    first_b, last_b = [], []
    for path in raw_paths:
        raw = read_raw(path)
        labels.extend(int(value) for value in raw["measurement_labels"])
        weights.extend(int(value) for value in raw["measurement_weights"])
        initial.append({
            "label_hex": f"{int(raw['initial_label']):016x}",
            "weight": state_weight(raw["initial_state_packed"], num_qubits),
        })
        burn.append({
            "label_hex": f"{int(raw['burn_label']):016x}",
            "weight": state_weight(raw["burn_state_packed"], num_qubits),
        })
        final.append({
            "label_hex": f"{int(raw['final_label']):016x}",
            "weight": state_weight(raw["final_state_packed"], num_qubits),
        })
        first_b.append(tuple(int(value) for value in raw["measurement_b_columns"][0]))
        last_b.append(tuple(int(value) for value in raw["measurement_b_columns"][-1]))
    return {
        "family": family,
        "measurement_weight": {
            "minimum": int(min(weights)),
            "maximum": int(max(weights)),
            "mean": float(np.mean(weights)),
            "standard_deviation": float(np.std(weights)),
        },
        "initial": initial,
        "burn": burn,
        "final": final,
        "distinct_B_at_first_measurement": len(set(first_b)),
        "distinct_B_at_final_measurement": len(set(last_b)),
        "distinct_B_across_endpoints": len(set(first_b) | set(last_b)),
        "labels": label_summary(labels),
    }


def main():
    output = RUN_ROOT / "CONVERGENCE_DIAGNOSTIC.json"
    require(not output.exists(), "convergence diagnostic already exists")
    manifest = read_json(RUN_ROOT / "MANIFEST.json")
    report = read_json(RUN_ROOT / "REPORT.json")
    audit = read_json(RUN_ROOT / "INDEPENDENT_AUDIT.json")
    require(report["status"] == "LOCAL_LOGICAL_TRANSPORT_NOT_VIABLE",
            "unexpected local transport result")
    require(audit["status"] == "INDEPENDENT_RAW_AUDIT_PASS",
            "independent audit did not pass")
    require(audit["transport_status"] == report["status"],
            "audit and report transport status differ")
    expected_raw_hashes = report["raw_sha256"]
    raw_paths = {
        family: sorted((RUN_ROOT / "raw").glob(f"{family}_*.npz"))
        for family in ("P", "U", "L")
    }
    for paths in raw_paths.values():
        require(len(paths) == 8, "raw trajectory count changed")
        for path in paths:
            require(sha256_file(path) == expected_raw_hashes[path.name],
                    f"raw hash changed: {path.name}")
    _, _, H = load_frozen_code(REGISTRY_PATH, manifest["cell"]["code_id"])
    model = build_model(H)
    load_exp101()
    from exp101_certified_src.gf2 import gf2_rank

    hard_coset_dimension = int(model.num_qubits - gf2_rank(model.H_check))
    summaries = {
        family: summarize_family(family, paths, model.num_qubits)
        for family, paths in raw_paths.items()
    }
    low_weight = min(
        item["weight"] for family in ("P", "L") for item in summaries[family]["burn"]
    )
    high_weight = summaries["U"]["measurement_weight"]["minimum"]
    p = float(manifest["cell"]["p"])
    odds = p / (1.0 - p)
    log10_tail_bound = (
        hard_coset_dimension * math.log10(2.0)
        + (high_weight - low_weight) * math.log10(odds)
    )
    tail_bound = min(1.0, 10.0 ** log10_tail_bound)
    same_low_burn_label = {
        item["label_hex"]
        for family in ("P", "L") for item in summaries[family]["burn"]
    }
    require(len(same_low_burn_label) == 1, "P/L did not collapse to one low burn label")
    require(all(
        item["weight"] >= high_weight for item in summaries["U"]["burn"]
        + summaries["U"]["final"]
    ), "U endpoint unexpectedly falls below its measurement threshold")
    core = {
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "manifest_sha256": manifest["manifest_sha256"],
        "report_sha256": report["report_sha256"],
        "independent_audit_sha256": audit["audit_sha256"],
        "formal_authorization": False,
        "new_mcmc_samples_generated": False,
        "families": summaries,
        "target_support_check": {
            "posterior_weight_model": "pi(e|y)_proportional_to_(p/(1-p))^weight",
            "known_legal_low_weight_state": low_weight,
            "U_measurement_minimum_weight": high_weight,
            "hard_coset_dimension": hard_coset_dimension,
            "upper_bound_formula": "min(1,2^d*(p/(1-p))^(w_high-w_low))",
            "log10_upper_bound": log10_tail_bound,
            "upper_bound": tail_bound,
            "interpretation": (
                "Every U measurement lies in weight >= w_high. Its total target "
                "mass is bounded using one known legal weight-w_low state and the "
                "entire hard-coset cardinality; this is a target-support sanity "
                "check, not a Monte Carlo confidence interval."
            ),
        },
        "gate_alignment": {
            "leave_return_gate_alone_is_not_a_stationarity_proof": True,
            "P_and_L_low_local_variability_can_be_physically_consistent": True,
            "U_family_remains_in_a_bounded_negligible_weight_region": True,
            "conclusion": "FROZEN_FULL_ROW_CONFIGURATION_HAS_ADVERSARIAL_INIT_NONCONVERGENCE",
        },
    }
    diagnostic = {**core, "diagnostic_sha256": sha256_json(core)}
    atomic_json(output, diagnostic)
    print(canonical_json(diagnostic))


if __name__ == "__main__":
    main()
