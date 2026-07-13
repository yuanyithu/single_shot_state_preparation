"""Generate authoritative exp101.physics.v2 certification evidence."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
EXP101_ROOT = HERE.parents[1]
if str(EXP101_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP101_ROOT))

from src.enumerate_exact import exact_reference  # noqa: E402
from src.gf2 import gf2_matmul  # noqa: E402
from src.graphs import repetition_parity_check_matrix  # noqa: E402
from src.hgp import hgp_from_H  # noqa: E402
from src.logicals import logical_pauli_operators  # noqa: E402
from src.model import (  # noqa: E402
    DisorderRealization,
    PHYSICS_CONTRACT_VERSION,
    assemble_sector_model,
    wire_ensemble,
)
from src.observables import (  # noqa: E402
    build_observable_frame,
    posterior_statistics,
)
from src.run_scan import (  # noqa: E402
    PROTOCOL_VERSION,
    _build_specs,
    implementation_fingerprint,
    merge,
    resolve_engine,
    scan,
    task_seed,
)


def bits(value, length):
    return np.asarray(
        [(value >> bit) & 1 for bit in range(length)], dtype=np.uint8
    )


def bernoulli_mass(vector, probability):
    vector = np.asarray(vector, dtype=np.uint8)
    weight = int(vector.sum())
    return probability**weight * (1.0 - probability) ** (
        vector.size - weight
    )


def label_bits(frame, vector):
    return sum(
        int(enabled) << bit
        for bit, enabled in enumerate(frame.label_of(vector))
    )


def build_small_css():
    H_Z, H_X = hgp_from_H(repetition_parity_check_matrix(2))
    logicals = logical_pauli_operators(H_X, H_Z)
    model = assemble_sector_model(H_X, H_Z, logicals, sector="x_error")
    return model, build_observable_frame(model)


def preparation_chain_representative(model, sigma_prep):
    """Choose a preparation representative independently of the production section."""
    base = model.logical_sector_section.apply(sigma_prep)
    if np.any(sigma_prep) and model.k:
        base = base ^ model.logical_move_basis[0]
    recovered = gf2_matmul(model.H_check, base[:, None])[:, 0]
    assert np.array_equal(recovered, sigma_prep)
    return base


def jsonable(value):
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_json(path, value):
    path.write_text(
        json.dumps(jsonable(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def exact_reduction_evidence():
    model, frame = build_small_css()
    p, q = 0.17, 0.13
    physical_syndromes = {
        tuple(gf2_matmul(model.H_check, bits(value, model.num_qubits)[:, None])[:, 0])
        for value in range(1 << model.num_qubits)
    }
    maxima = {
        "pointwise_weight_abs_error": 0.0,
        "partition_function_abs_error": 0.0,
        "sector_weight_abs_error": 0.0,
        "q_top_abs_error": 0.0,
        "map_success_abs_error": 0.0,
    }
    contexts = 0
    configurations = 0
    preparation_representative_difference_count = 0
    for sigma_tuple in sorted(physical_syndromes):
        sigma_prep = np.asarray(sigma_tuple, dtype=np.uint8)
        c_prep = preparation_chain_representative(model, sigma_prep)
        if not np.array_equal(
            c_prep, model.logical_sector_section.apply(sigma_prep)
        ):
            preparation_representative_difference_count += 1
        for measurement_value in range(1 << model.num_checks):
            measurement_error = bits(measurement_value, model.num_checks)
            s_prep = sigma_prep ^ measurement_error
            for epsilon_value in range(1 << model.num_qubits):
                contexts += 1
                epsilon_data_true = bits(epsilon_value, model.num_qubits)
                F_total = c_prep ^ epsilon_data_true
                sigma_final = gf2_matmul(
                    model.H_check, F_total[:, None]
                )[:, 0]
                effective_syndrome = s_prep ^ sigma_final
                assert np.array_equal(
                    effective_syndrome,
                    gf2_matmul(
                        model.H_check, epsilon_data_true[:, None]
                    )[:, 0] ^ measurement_error,
                )
                raw_sector = np.zeros(1 << model.k)
                reduced_sector = np.zeros_like(raw_sector)
                for a_value in range(1 << model.num_qubits):
                    configurations += 1
                    a = bits(a_value, model.num_qubits)
                    e = a ^ F_total
                    raw_residual = (
                        gf2_matmul(model.H_check, a[:, None])[:, 0]
                        ^ s_prep
                    )
                    reduced_residual = (
                        gf2_matmul(model.H_check, e[:, None])[:, 0]
                        ^ effective_syndrome
                    )
                    raw_weight = bernoulli_mass(
                        raw_residual, q
                    ) * bernoulli_mass(a ^ F_total, p)
                    reduced_weight = bernoulli_mass(
                        e, p
                    ) * bernoulli_mass(reduced_residual, q)
                    maxima["pointwise_weight_abs_error"] = max(
                        maxima["pointwise_weight_abs_error"],
                        abs(raw_weight - reduced_weight),
                    )
                    logical_class = label_bits(frame, e)
                    raw_sector[logical_class] += raw_weight
                    reduced_sector[logical_class] += reduced_weight

                raw_Z = float(raw_sector.sum())
                reduced_Z = float(reduced_sector.sum())
                maxima["partition_function_abs_error"] = max(
                    maxima["partition_function_abs_error"],
                    abs(raw_Z - reduced_Z),
                )
                raw_probability = raw_sector / raw_Z
                reduced_probability = reduced_sector / reduced_Z
                maxima["sector_weight_abs_error"] = max(
                    maxima["sector_weight_abs_error"],
                    float(np.max(np.abs(
                        raw_probability - reduced_probability
                    ))),
                )
                raw_stats = posterior_statistics(
                    raw_probability, planted_class=label_bits(
                        frame, epsilon_data_true
                    )
                )
                reduced_stats = posterior_statistics(
                    reduced_probability, planted_class=label_bits(
                        frame, epsilon_data_true
                    )
                )
                maxima["q_top_abs_error"] = max(
                    maxima["q_top_abs_error"],
                    abs(raw_stats["q_top"] - reduced_stats["q_top"]),
                )
                maxima["map_success_abs_error"] = max(
                    maxima["map_success_abs_error"],
                    abs(
                        raw_stats["map_success_probability"]
                        - reduced_stats["map_success_probability"]
                    ),
                )

    assert max(maxima.values()) < 1e-13, maxima
    assert preparation_representative_difference_count > 0

    epsilon = bits(7, model.num_qubits)
    measurement = bits(2, model.num_checks)
    effective = (
        gf2_matmul(model.H_check, epsilon[:, None])[:, 0] ^ measurement
    )
    disorder = DisorderRealization(
        epsilon_data_true=epsilon,
        measurement_error=measurement,
        effective_syndrome=effective,
        p=0.19,
        q=0.11,
    )
    wiring = wire_ensemble(model, disorder, "true_posterior", frame)
    oracle = exact_reference(model, frame, wiring, force_python=True)
    oracle_fields = [
        "weights_absolute", "weights_relative", "characters_absolute",
        "characters_relative", "posterior_purity",
        "posterior_mass_on_planted_class", "map_success_probability",
        "map_success_lower_bound", "map_success_upper_bound", "q_top",
    ]
    assert all(field in oracle for field in oracle_fields)
    assert oracle["posterior_purity"] <= oracle["map_success_probability"]
    assert (
        oracle["map_success_probability"]
        <= oracle["map_success_upper_bound"]
    )

    support_qubit = int(np.flatnonzero(model.H_check.any(axis=0))[0])
    alternative_truth = epsilon.copy()
    alternative_truth[support_qubit] ^= 1
    alternative_truth_syndrome = gf2_matmul(
        model.H_check, alternative_truth[:, None]
    )[:, 0]
    original_truth_syndrome = gf2_matmul(
        model.H_check, epsilon[:, None]
    )[:, 0]
    assert not np.array_equal(
        original_truth_syndrome, alternative_truth_syndrome
    )
    alternative_measurement = effective ^ alternative_truth_syndrome
    alternative = DisorderRealization(
        epsilon_data_true=alternative_truth,
        measurement_error=alternative_measurement,
        effective_syndrome=effective,
        p=0.19,
        q=0.11,
    )
    alternative_wiring = wire_ensemble(
        model, alternative, "true_posterior", frame
    )
    energy_difference = max(
        abs(
            wiring.total_energy(model, bits(value, model.num_qubits))
            - alternative_wiring.total_energy(
                model, bits(value, model.num_qubits)
            )
        )
        for value in range(1 << model.num_qubits)
    )
    assert energy_difference == 0.0

    shifted_coordinate_max_abs_difference = 0.0
    for value in range(1 << model.num_qubits):
        e = bits(value, model.num_qubits)
        x = e ^ epsilon
        canonical_energy = (
            wiring.K_p * float(e.sum())
            + wiring.K_q * float((
                gf2_matmul(model.H_check, e[:, None])[:, 0] ^ effective
            ).sum())
        )
        shifted_energy = (
            wiring.K_p * float((x ^ epsilon).sum())
            + wiring.K_q * float((
                gf2_matmul(model.H_check, x[:, None])[:, 0] ^ measurement
            ).sum())
        )
        shifted_coordinate_max_abs_difference = max(
            shifted_coordinate_max_abs_difference,
            abs(canonical_energy - shifted_energy),
        )
    assert shifted_coordinate_max_abs_difference < 1e-13

    nontrivial_truth = np.zeros(model.num_qubits, dtype=np.uint8)
    nontrivial_truth[support_qubit] = 1
    zero_measurement = np.zeros(model.num_checks, dtype=np.uint8)
    nontrivial_effective = gf2_matmul(
        model.H_check, nontrivial_truth[:, None]
    )[:, 0]
    distinguishing_disorder = DisorderRealization(
        epsilon_data_true=nontrivial_truth,
        measurement_error=zero_measurement,
        effective_syndrome=nontrivial_effective,
        p=0.17,
        q=0.13,
    )
    true_wiring = wire_ensemble(
        model, distinguishing_disorder, "true_posterior", frame
    )
    legacy_wiring = wire_ensemble(
        model, distinguishing_disorder, "legacy_delta_only", frame
    )
    zero_error = np.zeros(model.num_qubits, dtype=np.uint8)
    true_legacy_energy_gap = abs(
        true_wiring.total_energy(model, zero_error)
        - legacy_wiring.total_energy(model, zero_error)
    )
    assert true_legacy_energy_gap > 0.0

    q0_disorder = DisorderRealization(
        epsilon_data_true=nontrivial_truth,
        measurement_error=zero_measurement,
        effective_syndrome=nontrivial_effective,
        p=0.17,
        q=0.0,
    )
    q0_true = wire_ensemble(model, q0_disorder, "true_posterior", frame)
    q0_legacy = wire_ensemble(
        model, q0_disorder, "legacy_delta_only", frame
    )
    q0_true_exact = exact_reference(
        model, frame, q0_true, force_python=True
    )
    q0_legacy_exact = exact_reference(
        model, frame, q0_legacy, force_python=True
    )
    q0_tables_differ = not np.array_equal(
        q0_true_exact["table"].table, q0_legacy_exact["table"].table
    )
    assert q0_tables_differ
    assert q0_legacy_exact["map_success_probability"] is None

    artificial = posterior_statistics(np.asarray([0.1, 0.9]), planted_class=0)
    assert artificial["posterior_mass_on_planted_class"] == 0.1
    assert artificial["map_success_probability"] == 0.9

    alias_seed_equal = task_seed(
        "family", "x_error", "repo_compat", 0.1, 0.05, 3, "stream"
    ) == task_seed(
        "family", "x_error", "legacy_delta_only", 0.1, 0.05, 3, "stream"
    )
    assert alias_seed_equal
    routing = {
        "k10_q_positive": resolve_engine("auto", 10, 0.1),
        "k11_q_positive": resolve_engine("auto", 11, 0.1),
        "k11_q_zero": resolve_engine("auto", 11, 0.0),
    }

    evidence = {
        "physics_contract_version": PHYSICS_CONTRACT_VERSION,
        "scan_contract_version": PROTOCOL_VERSION,
        "small_css": {
            "n": model.num_qubits,
            "k": model.k,
            "num_checks": model.num_checks,
        },
        "enumeration": {
            "physical_sigma_prep_count": len(physical_syndromes),
            "independent_preparation_representative_difference_count": (
                preparation_representative_difference_count
            ),
            "disorder_context_count": contexts,
            "configuration_weight_comparisons": configurations,
            **maxima,
        },
        "true_energy_fixed_effective_syndrome_max_abs_difference": (
            energy_difference
        ),
        "fixed_y_truth_syndrome_changed": True,
        "shifted_coordinate_energy_max_abs_difference": (
            shifted_coordinate_max_abs_difference
        ),
        "true_vs_legacy_nontrivial_energy_gap": true_legacy_energy_gap,
        "q_zero_semantics": {
            "true_gibbs_argument": q0_true.gibbs_syndrome_argument,
            "legacy_gibbs_argument": q0_legacy.gibbs_syndrome_argument,
            "true_quenched_coset_differs_from_legacy_clean_kernel": (
                q0_tables_differ
            ),
            "legacy_map_success_probability": q0_legacy_exact[
                "map_success_probability"
            ],
        },
        "alias_and_routing": {
            "deprecated_alias_seed_equals_canonical": alias_seed_equal,
            "resolved_engines": routing,
        },
        "exact_oracle": {
            "fields": oracle_fields,
            "weights_absolute": oracle["weights_absolute"],
            "weights_relative": oracle["weights_relative"],
            "characters_absolute": oracle["characters_absolute"],
            "characters_relative": oracle["characters_relative"],
            "posterior_purity": oracle["posterior_purity"],
            "posterior_mass_on_planted_class": oracle[
                "posterior_mass_on_planted_class"
            ],
            "map_success_probability": oracle["map_success_probability"],
            "map_success_bounds": [
                oracle["map_success_lower_bound"],
                oracle["map_success_upper_bound"],
            ],
            "q_top_absolute": oracle["q_top_absolute"],
            "q_top_relative": oracle["q_top_relative"],
        },
        "artificial_posterior_0p1_0p9": artificial,
        "passed": True,
    }
    write_json(HERE / "exact_reduction_evidence.json", evidence)
    (HERE / "exact_reduction_evidence.md").write_text(
        "# Exact reduction evidence\n\n"
        f"- Contract: `{PHYSICS_CONTRACT_VERSION}` / `{PROTOCOL_VERSION}`\n"
        f"- Enumerated contexts: `{contexts}`; pointwise weights: "
        f"`{configurations}`\n"
        "- Independent preparation representative differs from the "
        "production logical-sector section: PASS\n"
        f"- Max pointwise error: `{maxima['pointwise_weight_abs_error']:.3e}`\n"
        f"- Max partition error: `{maxima['partition_function_abs_error']:.3e}`\n"
        f"- Max sector-probability error: `{maxima['sector_weight_abs_error']:.3e}`\n"
        f"- Max q_top error: `{maxima['q_top_abs_error']:.3e}`\n"
        f"- Max MAP error: `{maxima['map_success_abs_error']:.3e}`\n"
        f"- Fixed-y true-energy truth dependence: `{energy_difference:.3e}`\n"
        "- Fixed-y comparison changes `H_check epsilon_data_true`: PASS\n"
        "- Shifted-coordinate/canonical pointwise energy identity: "
        f"`{shifted_coordinate_max_abs_difference:.3e}`\n"
        "- q=0 true quenched coset differs from legacy clean kernel: PASS\n"
        "- Exact oracle purity/MAP bounds: PASS\n"
        "- Alias normalization and three-way auto routing: PASS\n",
        encoding="utf-8",
    )
    return evidence


def write_synthetic_chunk(spec, result):
    path = Path(spec["chunk_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "protocol": PROTOCOL_VERSION,
        "task_fingerprint": spec["task_fingerprint"],
        "implementation_fingerprint": spec["implementation_fingerprint"],
        "result": result,
    }), encoding="utf-8")


def synthetic_result(spec, q_top, valid):
    return {
        "task_fingerprint": spec["task_fingerprint"],
        "implementation_fingerprint": spec["implementation_fingerprint"],
        "git_commit_sha": "synthetic-validation-fixture",
        "git_worktree_dirty": False,
        "family": {"family": "surface", "size": 2},
        "k": 1,
        "code_fingerprint": "synthetic-code",
        "section_fingerprint": "synthetic-section",
        "observable_frame_fingerprint": "synthetic-frame",
        "observable_set_fingerprint": "synthetic-observable-set",
        "resolved_engine": spec["resolved_engine"],
        "resolved_engine_config": spec["engine_config"],
        "character_count": 1,
        "u_bitmasks": [1],
        "u_rand_seed": None,
        "character_means_absolute": [0.2],
        "character_means_relative": [0.2],
        "m2_u_pooled_square_raw": [0.04],
        "m2_u_debiased": [q_top],
        "m2_u_debiased_jackknife_se": [0.01],
        "q_top_estimate": q_top,
        "q_top_absolute": q_top,
        "q_top_relative": q_top,
        "q_top_estimator_name": "synthetic_validation_fixture",
        "valid_for_aggregation": valid,
        "failure_reasons": [] if valid else ["synthetic_invalid"],
        "flags": "PASS" if valid else "INVALID:synthetic_invalid",
    }


def pt_and_aggregation_evidence():
    pt_dir = HERE / "pt_invalid_scan"
    pt_config = {
        "num_temperatures": 8,
        "q_hot": 0.40,
        "num_burn_in_rounds": 0,
        "num_measurement_rounds": 4,
        "num_instances": 4,
    }
    pt_npz, pt_report = scan(
        pt_dir, "surface", [2], 0.12, [0.08], 1,
        engine="pt", engine_config=pt_config, force_recompute=True,
    )
    assert pt_report["failed"] == []
    assert pt_report["computed"] == 1
    assert pt_report["reused"] == 0
    with np.load(pt_npz) as data:
        pt_manifest = json.loads(str(data["manifest_json"]))
        pt_valid = bool(data["valid_for_aggregation"][0, 0, 0])
        pt_mean = float(data["mean_q_top_estimate"][0, 0])
        round_trips = data["pt_round_trips_per_disorder"][0, 0, 0].tolist()
        burn_in_round_trips = data[
            "pt_burn_in_round_trips_per_disorder"
        ][0, 0, 0].tolist()
        measurement_round_trips = data[
            "pt_measurement_round_trips_per_disorder"
        ][0, 0, 0].tolist()
        pt_task_fingerprint = str(
            data["task_fingerprint_per_disorder"][0, 0, 0]
        )
        pt_implementation_fingerprint = str(data["implementation_fingerprint"])
        failure_reasons = str(data[
            "failure_reasons_per_disorder"
        ][0, 0, 0])
        pt_schema_fields = [
            "pt_ladder_p_per_disorder", "pt_ladder_q_per_disorder",
            "pt_swap_attempts_per_disorder",
            "pt_swap_accepts_per_disorder", "pt_swap_rates_per_disorder",
            "pt_round_trips_per_disorder",
            "pt_burn_in_round_trips_per_disorder",
            "pt_measurement_round_trips_per_disorder",
            "pt_cold_logical_acceptance_per_disorder",
            "gate_diagnostics_json_per_disorder",
            "task_fingerprint_per_disorder",
            "implementation_fingerprint_per_disorder",
            "implementation_fingerprint", "git_commit_sha",
            "git_worktree_dirty",
        ]
        missing_pt_fields = [
            field for field in pt_schema_fields if field not in data.files
        ]
    assert not pt_valid
    assert np.isnan(pt_mean)
    assert round_trips == [0, 0, 0, 0]
    assert measurement_round_trips == round_trips
    assert "pt_instance_round_trips_insufficient" in failure_reasons
    assert not missing_pt_fields
    assert len(pt_task_fingerprint) == 64
    assert pt_implementation_fingerprint == implementation_fingerprint()
    assert pt_manifest["implementation_fingerprint"] \
        == pt_implementation_fingerprint
    assert pt_manifest["git_commit_sha"]
    assert isinstance(pt_manifest["git_worktree_dirty"], bool)

    aggregation_dir = HERE / "aggregation_validity_scan"
    specs = _build_specs(
        aggregation_dir, "surface", [2], 0.1, [0.05], 3,
        "x_error", "true_posterior", "direct", {}, "full_rank", None, 64,
    )
    write_synthetic_chunk(specs[0], synthetic_result(specs[0], 0.2, True))
    write_synthetic_chunk(specs[1], synthetic_result(specs[1], 0.4, True))
    write_synthetic_chunk(specs[2], synthetic_result(specs[2], 0.95, False))
    aggregation_npz = merge(
        aggregation_dir, "surface", [2], 0.1, [0.05], 3,
        "x_error", "true_posterior", "direct", {}, "full_rank",
        expected_specs=specs,
    )
    with np.load(aggregation_npz) as data:
        aggregate_mean = float(data["mean_q_top_estimate"][0, 0])
        aggregate_sem = float(data[
            "disorder_sem_q_top_estimate"
        ][0, 0])
        crossing_input = data[
            "q_top_crossing_input_per_disorder"
        ][0, 0].tolist()
        counts = {
            "valid": int(data["valid_disorder_count"][0, 0]),
            "invalid": int(data["invalid_disorder_count"][0, 0]),
            "missing": int(data["missing_disorder_count"][0, 0]),
        }
    assert np.isclose(aggregate_mean, 0.3)
    assert np.isclose(aggregate_sem, 0.1)
    assert np.allclose(crossing_input[:2], [0.2, 0.4])
    assert np.isnan(crossing_input[2])
    assert counts == {"valid": 2, "invalid": 1, "missing": 0}

    evidence = {
        "physics_contract_version": PHYSICS_CONTRACT_VERSION,
        "scan_contract_version": PROTOCOL_VERSION,
        "real_pt_invalid_scan": {
            "computed_tasks": pt_report["computed"],
            "reused_tasks": pt_report["reused"],
            "computed_or_reused_tasks": (
                pt_report["computed"] + pt_report["reused"]
            ),
            "valid_for_aggregation": pt_valid,
            "mean_q_top_estimate": None,
            "round_trips_per_instance": round_trips,
            "burn_in_round_trips_per_instance": burn_in_round_trips,
            "measurement_round_trips_per_instance": measurement_round_trips,
            "failure_reasons": failure_reasons,
            "task_fingerprint": pt_task_fingerprint,
            "implementation_fingerprint": pt_implementation_fingerprint,
            "git_commit_sha": pt_manifest["git_commit_sha"],
            "git_worktree_dirty": pt_manifest["git_worktree_dirty"],
            "merge_git_commit_sha": pt_manifest["merge_git_commit_sha"],
            "merge_git_worktree_dirty": pt_manifest[
                "merge_git_worktree_dirty"
            ],
            "complete_pt_schema_fields_present": not missing_pt_fields,
            "missing_pt_schema_fields": missing_pt_fields,
        },
        "synthetic_validity_aggregation": {
            "valid_q_top": [0.2, 0.4],
            "invalid_q_top": 0.95,
            "reported_mean_q_top_estimate": aggregate_mean,
            "reported_disorder_sem_q_top_estimate": aggregate_sem,
            "crossing_input": crossing_input,
            "counts": counts,
        },
        "passed": True,
    }
    write_json(HERE / "pt_aggregation_evidence.json", evidence)
    (HERE / "pt_aggregation_evidence.md").write_text(
        "# PT and aggregation integration evidence\n\n"
        "- Real four-instance PT task: `INVALID` as designed.\n"
        "- Forced fresh execution: `computed=1`, `reused=0`.\n"
        f"- Round trips per instance: `{round_trips}`.\n"
        f"- Burn-in/measurement round trips: `{burn_in_round_trips}` / "
        f"`{measurement_round_trips}`.\n"
        "- Failure includes `pt_instance_round_trips_insufficient`.\n"
        "- Invalid-only PT mean: `NaN` (serialized as `null`).\n"
        "- Complete ladder/swap/round-trip/cold/gate schema: PASS.\n"
        "- Synthetic valid/invalid aggregate counts: "
        f"`{counts}`.\n"
        f"- Valid-only mean/SEM: `{aggregate_mean}` / `{aggregate_sem}` "
        "(invalid value `0.95` excluded).\n"
        f"- Crossing input: `{crossing_input}` (invalid entry is NaN).\n"
        f"- Task/source fingerprints: `{pt_task_fingerprint}` / "
        f"`{pt_implementation_fingerprint}`.\n",
        encoding="utf-8",
    )
    return evidence


EVIDENCE_TESTS = [
    (
        "CSS move completeness prevents exact/MCMC q=0 support mismatch",
        "test_physics_golden.py",
        "test_incomplete_q_zero_move_set_is_rejected_at_model_assembly",
    ),
    (
        "raw/reduced, shifted-coordinate, fixed-y, and x/z convention",
        "test_paper_reduction_golden.py",
        "test_fixed_effective_syndrome_is_truth_independent_off_kernel",
    ),
    (
        "canonical aliases, posterior statistics, and unphysical bounds",
        "test_model_observables.py",
        "test_unphysical_debiased_purity_is_retained_without_success_bounds",
    ),
    (
        "boundary-only invariance and logical-shift counterexample",
        "test_section_frames.py",
        "test_logical_section_shift_is_not_claimed_as_gauge",
    ),
    (
        "population weighting, U-statistic, jackknife, and FPC",
        "test_model_observables.py",
        "test_delete_one_jackknife_tracks_repeated_sampling_error",
    ),
    (
        "large-k TI rejection and gap-only diagnostics",
        "test_sector_ti.py",
        "test_diagnostics_match_exact_gaps_and_expose_no_purity",
    ),
    (
        "independent-sector TI bootstrap uncertainty",
        "test_sector_ti.py",
        "test_independent_sector_resampling_avoids_false_zero_gap_error",
    ),
    (
        "analytic p=0 and p=0.5 full-sector TI endpoints",
        "test_run_scan.py",
        "test_auto_full_ti_endpoints_are_analytic_end_to_end",
    ),
    (
        "three-way auto routing and large-k preflight refusal",
        "test_run_scan.py",
        "test_auto_routes_all_three_production_paths",
    ),
    (
        "actual k=16 observable construction preserves all 80 characters",
        "test_run_scan.py",
        "test_actual_k16_observable_set_keeps_16_plus_64_characters",
    ),
    (
        "manifest/NPZ schema and source/config/cache identity isolation",
        "test_run_scan.py",
        "test_source_fingerprint_participates_in_chunk_identity",
    ),
    (
        "chunk outer/inner task identity tamper rejection",
        "test_run_scan.py",
        "test_mismatched_inner_task_fingerprint_recomputes_chunk",
    ),
    (
        "Git dirty provenance uses boolean value plus known marker",
        "test_scan_estimators.py",
        "test_unknown_git_dirty_state_uses_known_marker",
    ),
    (
        "deprecated aliases normalize before seed/result/manifest storage",
        "test_run_scan.py",
        "test_alias_is_canonical_before_seed_and_manifest",
    ),
    (
        "legacy sampled output is formal-only and excluded from aggregation",
        "test_run_scan.py",
        "test_legacy_sampled_scan_is_formal_only_end_to_end",
    ),
    (
        "INVALID-safe mean/SEM/crossing aggregation",
        "test_scan_estimators.py",
        "test_sem_uses_two_valid_samples_and_excludes_invalid",
    ),
    (
        "PT endpoint state machine and fresh-phase round trips",
        "test_pt.py",
        "test_new_phase_does_not_inherit_partial_transit",
    ),
    (
        "current narrative and PRE_ALIGNMENT overwrite guards",
        "test_contract_text.py",
        "test_historical_runners_cannot_overwrite_alignment_warning",
    ),
]


def evidence_test_inventory():
    inventory = []
    for topic, filename, test_name in EVIDENCE_TESTS:
        path = EXP101_ROOT / "tests" / filename
        source = path.read_text(encoding="utf-8")
        assert test_name in source, (filename, test_name)
        inventory.append({
            "topic": topic,
            "test_file": f"tests/{filename}",
            "test_name": test_name,
        })
    return inventory


def run_pytest_suite():
    command = [
        sys.executable, "-m", "pytest", "-q", str(EXP101_ROOT / "tests")
    ]
    completed = subprocess.run(
        command,
        cwd=EXP101_ROOT.parents[2],
        capture_output=True,
        text=True,
        check=False,
    )
    log = (
        "command: " + json.dumps(command) + "\n\n"
        "[stdout]\n" + completed.stdout + "\n"
        "[stderr]\n" + completed.stderr
    )
    (HERE / "pytest_full_output.txt").write_text(log, encoding="utf-8")
    (HERE / "pytest_exit_code.txt").write_text(
        f"{completed.returncode}\n", encoding="ascii"
    )
    return {
        "command": command,
        "exit_code": completed.returncode,
        "passed": completed.returncode == 0,
        "log_path": "pytest_full_output.txt",
        "exit_code_path": "pytest_exit_code.txt",
        "log_sha256": hashlib.sha256(log.encode("utf-8")).hexdigest(),
        "stdout_last_lines": completed.stdout.splitlines()[-5:],
        "stderr_last_lines": completed.stderr.splitlines()[-5:],
    }


def write_summary(exact, integration, pytest_result, inventory, environment):
    suite_status = "PASS" if pytest_result["passed"] else "NOT RUN/FAIL"
    rows = []
    for item in inventory:
        rows.append(
            f"| {item['topic']} | `{item['test_file']}::{item['test_name']}` "
            f"| {suite_status} |"
        )
    overall = bool(environment["overall_passed"])
    (HERE / "summary.md").write_text(
        "# exp101.physics.v2 validation 014 evidence index\n\n"
        f"Overall certification evidence: `{'PASS' if overall else 'INCOMPLETE'}`. "
        "This file does not itself change `status.md`.\n\n"
        "## Reproducibility\n\n"
        f"- Contracts: `{PHYSICS_CONTRACT_VERSION}` / `{PROTOCOL_VERSION}`.\n"
        f"- Python/NumPy: `{environment['python']}` / `{environment['numpy']}`.\n"
        f"- Conda environment: `{environment['conda_default_env']}`.\n"
        f"- Git SHA / dirty: `{environment['git_commit_sha']}` / "
        f"`{environment['git_worktree_dirty']}`.\n"
        f"- Implementation fingerprint: "
        f"`{environment['implementation_fingerprint']}`.\n"
        f"- Full pytest exit/log SHA256: `{pytest_result['exit_code']}` / "
        f"`{pytest_result['log_sha256']}`.\n\n"
        "## Decisive evidence\n\n"
        f"- Exact raw-paper/reduced enumeration: "
        f"`{'PASS' if exact['passed'] else 'FAIL'}`; "
        "see `exact_reduction_evidence.json` and `.md`.\n"
        "  This artifact also records fixed-y truth independence off the "
        "kernel, shifted-coordinate equality, q=0 true/legacy separation, "
        "alias routing, absolute/relative characters, and posterior bounds.\n"
        f"- Fresh PT/aggregation integration: "
        f"`{'PASS' if integration['passed'] else 'FAIL'}`; "
        "see `pt_aggregation_evidence.json` and `.md`.\n"
        f"- Complete exp101 pytest suite: `{suite_status}`; see "
        "`pytest_full_output.txt` and `pytest_exit_code.txt`.\n\n"
        "## Coverage map\n\n"
        "| Contract area | Decisive test | Suite status |\n"
        "|---|---|---|\n"
        + "\n".join(rows)
        + "\n\nMachine-readable inventory and hashes are in `environment.json`.\n",
        encoding="utf-8",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate exp101.physics.v2 validation 014 evidence"
    )
    parser.add_argument(
        "--skip-pytest", action="store_true",
        help="refresh exact/PT evidence only; leaves certification incomplete",
    )
    args = parser.parse_args(argv)
    HERE.mkdir(parents=True, exist_ok=True)
    conda_env = os.environ.get("CONDA_DEFAULT_ENV")
    if conda_env != "12":
        raise RuntimeError(
            "validation 014 must run in conda environment 12; "
            f"got {conda_env!r}"
        )
    exact = exact_reduction_evidence()
    integration = pt_and_aggregation_evidence()
    inventory = evidence_test_inventory()
    pytest_result = (
        {
            "command": [], "exit_code": None, "passed": False,
            "log_path": None, "exit_code_path": None,
            "log_sha256": None, "stdout_last_lines": [],
            "stderr_last_lines": [],
        }
        if args.skip_pytest else run_pytest_suite()
    )
    pt_metadata = integration["real_pt_invalid_scan"]
    environment = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "conda_default_env": conda_env,
        "physics_contract_version": PHYSICS_CONTRACT_VERSION,
        "scan_contract_version": PROTOCOL_VERSION,
        "git_commit_sha": pt_metadata["git_commit_sha"],
        "git_worktree_dirty": pt_metadata["git_worktree_dirty"],
        "merge_git_commit_sha": pt_metadata["merge_git_commit_sha"],
        "merge_git_worktree_dirty": pt_metadata[
            "merge_git_worktree_dirty"
        ],
        "implementation_fingerprint": pt_metadata[
            "implementation_fingerprint"
        ],
        "task_fingerprint": pt_metadata["task_fingerprint"],
        "exact_passed": exact["passed"],
        "integration_passed": integration["passed"],
        "pytest": pytest_result,
        "evidence_test_inventory": inventory,
        "overall_passed": bool(
            exact["passed"] and integration["passed"]
            and pytest_result["passed"]
        ),
    }
    write_json(HERE / "environment.json", environment)
    write_summary(exact, integration, pytest_result, inventory, environment)
    print(json.dumps(environment, sort_keys=True))
    return 0 if (
        exact["passed"] and integration["passed"]
        and (args.skip_pytest or pytest_result["passed"])
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
