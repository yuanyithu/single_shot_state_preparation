"""Exact conditional-stay diagnostic for the full-row Gibbs hard sentinel.

No trajectory is run here.  For each adversarial but legal initialization it
evaluates the exact conditional probability that each B-row heatbath retains
its current row.  This distinguishes a short zero-change trace from a
conditional kernel that is already numerically pinned at the start.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

import numpy as np

from data.expander_code.exp102.exp102_pipeline.io import (
    atomic_json,
    sha256_file,
    sha256_json,
)
from data.expander_code.exp102.exp102_pipeline.q0_global import (
    state_label,
    uniform_hard_coset_state,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_collapsed import (
    _bits_to_mask,
    _initial_collapsed_masks,
    _qubit_signatures,
    _section_and_kernel_masks,
    build_classical_coset_mass,
    split_hgp_state,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_full_row_gibbs_v0 import (
    FULL_ROW_GIBBS_VERSION,
    FullRowGibbsConfig,
    FullRowGibbsSeedIdentity,
    build_full_row_elimination_plan,
    full_row_current_assignment_log_probability,
    select_low_energy_logical_start,
)
from data.expander_code.exp102.exp102_pipeline.registry import (
    load_frozen_code,
    load_registry,
)
from data.expander_code.exp102.exp102_pipeline.seeds import derive_seed
from data.expander_code.exp102.exp102_pipeline.worker import build_model


DIAGNOSTIC_VERSION = "exp102.q0_hgp_full_row_gibbs.row_pinning.v0"
CELL = {
    "code_id": "m08_c06",
    "disorder_index": 0,
    "disorder_source": "attempt022",
    "p": 0.04,
}
ROOT = Path(__file__).resolve().parent
REGISTRY_PATH = ROOT.parents[1] / "registry" / "registry.json"


def _source_commit():
    return subprocess.run(
        ("git", "rev-parse", "HEAD"), check=True, capture_output=True, text=True,
    ).stdout.strip()


def _context():
    registry = load_registry(REGISTRY_PATH)
    _, code, H = load_frozen_code(REGISTRY_PATH, CELL["code_id"])
    model, frame = build_model(H)
    uniform_seed = derive_seed(
        "pilot_ladder_m8_attempt22", registry["registry_sha256"],
        code["code_id"], CELL["disorder_index"], "uniforms",
    )
    epsilon = (
        np.random.Generator(np.random.PCG64(uniform_seed)).random(model.num_qubits)
        < CELL["p"]
    ).astype(np.uint8)
    syndrome = (
        model.H_check.astype(np.int64) @ epsilon.astype(np.int64) % 2
    ).astype(np.uint8)
    if not syndrome.any():
        raise RuntimeError("hard sentinel syndrome unexpectedly vanished")
    return registry, code, H, model, frame, uniform_seed, epsilon, syndrome


def _identity(registry, config, family):
    return FullRowGibbsSeedIdentity(
        source_commit=_source_commit(),
        config_sha256=sha256_json(config.as_dict()),
        registry_sha256=registry["registry_sha256"],
        cell_fingerprint=sha256_json(CELL),
        method_id=config.method_id,
        resource_tier="PINNING_DIAGNOSTIC",
        init_family=family,
        trajectory_index=0,
        trajectory_namespace="q0_full_row_pinning_diagnostic_v0",
    )


def _family_report(family, state, model, frame, H, syndrome, plan, log_mass, log_odds):
    b_columns, a_syndromes, _ = _initial_collapsed_masks(state, syndrome, H)
    logs = np.asarray([
        full_row_current_assignment_log_probability(
            H, plan, b_columns, a_syndromes, row, log_mass, log_odds,
        )
        for row in range(H.shape[0])
    ], dtype=np.float64)
    leaves = -np.expm1(logs)
    if not np.all(np.isfinite(logs)) or np.any(logs > 1e-14) or np.any(leaves < 0.0):
        raise RuntimeError("row conditional stay probabilities are invalid")
    residual = (
        model.H_check.astype(np.int64) @ state.astype(np.int64) % 2
    ).astype(np.uint8) ^ syndrome
    if residual.any():
        raise RuntimeError(f"{family} initialization left the hard coset")
    return {
        "family": family,
        "initial_weight": int(state.sum()),
        "initial_label": int(state_label(frame, state)),
        "b_weight": int(sum(int(value).bit_count() for value in b_columns)),
        "row_log_stay_probability": logs.tolist(),
        "row_leave_probability": leaves.tolist(),
        "sum_row_leave_probability": float(leaves.sum(dtype=np.float64)),
        "max_row_leave_probability": float(leaves.max()),
        "zero_rows_exactly_one_in_float64": int(np.count_nonzero(logs == 0.0)),
    }


def _a_redraw_diagnostics(state, frame, H, syndrome, p):
    """Exact A|B physical and basis-character change probabilities."""
    rows, columns = H.shape
    A, _ = split_hgp_state(state, H)
    _, a_syndromes, _ = _initial_collapsed_masks(state, syndrome, H)
    section_masks, kernel_combinations = _section_and_kernel_masks(H)
    odds = p / (1.0 - p)
    log_stay = 0.0
    max_column_leave = 0.0
    character_means = np.ones(64, dtype=np.float64)
    signatures = _qubit_signatures(frame)
    for column in range(columns):
        section = np.uint32(0)
        for variable in range(columns):
            if int(section_masks[variable] & a_syndromes[column]).bit_count() & 1:
                section |= np.uint32(1) << np.uint32(variable)
        current = _bits_to_mask(A[:, column])
        total = 0.0
        current_weight = None
        signed_weights = np.zeros(64, dtype=np.float64)
        for combination in kernel_combinations:
            candidate = section ^ combination
            weight = odds ** int(candidate).bit_count()
            total += weight
            if candidate == current:
                current_weight = weight
            delta_signature = 0
            for variable in range(columns):
                if ((int(candidate) >> variable) & 1) != ((int(current) >> variable) & 1):
                    delta_signature ^= int(signatures[variable * columns + column])
            for bit in range(64):
                signed_weights[bit] += weight * (-1.0 if (delta_signature >> bit) & 1 else 1.0)
        if current_weight is None or not total > 0.0:
            raise RuntimeError("A conditional does not contain the current hard-coset column")
        probability = current_weight / total
        if not 0.0 < probability <= 1.0:
            raise RuntimeError("A conditional probability is invalid")
        log_stay += float(np.log(probability))
        max_column_leave = max(max_column_leave, 1.0 - probability)
        character_means *= signed_weights / total
    any_change = -float(np.expm1(log_stay))
    basis_flip = (1.0 - character_means) / 2.0
    if np.any(basis_flip < -1e-12) or np.any(basis_flip > 1.0 + 1e-12):
        raise RuntimeError("A conditional basis-character probability is invalid")
    return {
        "a_redraw_log_probability_all_columns_unchanged": log_stay,
        "a_redraw_probability_any_physical_change": any_change,
        "a_redraw_logical_change_upper_bound": any_change,
        "a_redraw_max_single_column_leave_probability": max_column_leave,
        "a_redraw_basis_flip_probability": basis_flip.tolist(),
        "a_redraw_max_basis_flip_probability": float(basis_flip.max()),
        "a_redraw_sum_basis_flip_probability": float(basis_flip.sum(dtype=np.float64)),
        "a_redraw_basis_flips_at_least_1e-6": int(np.count_nonzero(basis_flip >= 1e-6)),
    }


def diagnose():
    registry, code, H, model, frame, uniform_seed, epsilon, syndrome = _context()
    config = FullRowGibbsConfig(CELL["p"], 1, 8)
    l_start, l_metadata = select_low_energy_logical_start(epsilon, model, frame)
    u_start = uniform_hard_coset_state(
        model, syndrome, _identity(registry, config, "U").seed("initialize", "hard_coset"),
    )
    mass = build_classical_coset_mass(H, CELL["p"], engine="reference")
    plan = build_full_row_elimination_plan(H)
    report = {
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "sampler_version": FULL_ROW_GIBBS_VERSION,
        "source_commit": _source_commit(),
        "source_files": {
            "q0_hgp_full_row_gibbs.py": sha256_file(
                ROOT.parents[1] / "exp102_pipeline" / "q0_hgp_full_row_gibbs_v0.py",
            ),
            "diagnose_row_pinning.py": sha256_file(__file__),
        },
        "cell": CELL,
        "registry_sha256": registry["registry_sha256"],
        "code_npz_sha256": code["code_npz_sha256"],
        "uniform_seed": int(uniform_seed),
        "syndrome_weight": int(syndrome.sum()),
        "physical_zero_is_legal": False,
        "physical_zero_residual_weight": int(syndrome.sum()),
        "plan_sha256": plan.sha256,
        "plan_max_width": plan.max_width,
        "plan_structural_table_cells": plan.structural_table_cells,
        "mass_sha256": hashlib.sha256(np.asarray(mass, dtype=">f8").tobytes()).hexdigest(),
        "l_start": l_metadata,
        "families": [
            {
                **_family_report("P", epsilon, model, frame, H, syndrome, plan, np.log(mass), np.log(CELL["p"] / (1.0 - CELL["p"]))),
                **_a_redraw_diagnostics(epsilon, frame, H, syndrome, CELL["p"]),
            },
            {
                **_family_report("U", u_start, model, frame, H, syndrome, plan, np.log(mass), np.log(CELL["p"] / (1.0 - CELL["p"]))),
                **_a_redraw_diagnostics(u_start, frame, H, syndrome, CELL["p"]),
            },
            {
                **_family_report("L", l_start, model, frame, H, syndrome, plan, np.log(mass), np.log(CELL["p"] / (1.0 - CELL["p"]))),
                **_a_redraw_diagnostics(l_start, frame, H, syndrome, CELL["p"]),
            },
        ],
        "trajectory_raw_produced": False,
        "purpose": "exact_conditional_pinning_diagnostic_only",
    }
    report["report_sha256"] = sha256_json(report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "row_pinning.json")
    args = parser.parse_args(argv)
    report = diagnose()
    atomic_json(args.output, report)
    print(report["report_sha256"])


if __name__ == "__main__":
    main()
