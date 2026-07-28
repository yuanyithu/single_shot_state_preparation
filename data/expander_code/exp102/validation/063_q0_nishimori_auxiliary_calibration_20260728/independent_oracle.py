"""Independent physics.v2 enumeration for validation 063.

This module deliberately does not import the calibration runner or exp102
``worker``.  It reconstructs the HGP model directly from the certified exp101
physics implementation and recomputes every reported scientific quantity.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.expander_code.exp101.src.hgp import hgp_from_H
from data.expander_code.exp101.src.logicals import logical_pauli_operators
from data.expander_code.exp101.src.model import assemble_sector_model
from data.expander_code.exp101.src.observables import build_observable_frame


ORACLE_VERSION = "exp102.q0_nishimori_auxiliary.independent_oracle.v1"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def derive_seed(namespace, *parts):
    payload = canonical([namespace, *parts]).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _all_states(num_qubits):
    masks = np.arange(1 << int(num_qubits), dtype=np.uint64)
    shifts = np.arange(num_qubits, dtype=np.uint64)
    return ((masks[:, None] >> shifts) & np.uint64(1)).astype(np.uint8)


def _build_model(H):
    H_Z, H_X = hgp_from_H(np.asarray(H, dtype=np.uint8))
    logicals = logical_pauli_operators(H_X, H_Z)
    model = assemble_sector_model(H_X, H_Z, logicals, sector="x_error")
    return model, build_observable_frame(model)


def _syndrome_bytes(H_check, state):
    syndrome = (
        np.asarray(H_check, dtype=np.int64) @ np.asarray(state, dtype=np.int64) % 2
    ).astype(np.uint8)
    return np.packbits(syndrome, bitorder="little").tobytes()


def _label_integer(frame, state):
    bits = frame.label_of(np.asarray(state, dtype=np.uint8))
    return sum(int(bit) << index for index, bit in enumerate(bits))


def _state_table_sha(states, syndromes, labels):
    digest = hashlib.sha256(b"exp102.nishimori.physical_state_table.v1\0")
    for index, (state, syndrome, label) in enumerate(zip(states, syndromes, labels)):
        digest.update(int(index).to_bytes(8, "big"))
        digest.update(int(state.sum()).to_bytes(4, "big"))
        digest.update(len(syndrome).to_bytes(4, "big"))
        digest.update(syndrome)
        digest.update(int(label).to_bytes(8, "big"))
    return digest.hexdigest()


def enumerate_physics_v2(H, p, model_id):
    p = float(p)
    if not 0.0 < p < 0.5:
        raise ValueError("oracle p must lie in (0,.5)")
    model, frame = _build_model(H)
    states = _all_states(model.num_qubits)
    syndromes = [_syndrome_bytes(model.H_check, state) for state in states]
    labels = np.asarray([_label_integer(frame, state) for state in states], dtype=np.int64)
    physical_weights = np.asarray(
        [p ** int(state.sum()) * (1.0 - p) ** (model.num_qubits - int(state.sum())) for state in states],
        dtype=np.float64,
    )
    physical_weights /= physical_weights.sum()
    weights = states.sum(axis=1).astype(np.int64)
    keys = sorted(set(syndromes))
    label_count = 1 << int(model.k)
    y_probabilities = []
    logical_posteriors = []
    support_counts = []
    max_log_ratio_error = 0.0
    log_b = math.log(p / (1.0 - p))
    for key in keys:
        indices = np.asarray([i for i, value in enumerate(syndromes) if value == key], dtype=np.int64)
        if indices.size == 0:
            raise AssertionError("empty hard-coset support")
        if any(syndromes[int(index)] != key for index in indices):
            raise AssertionError("state escaped its hard coset")
        mass = float(physical_weights[indices].sum())
        if not math.isfinite(mass) or mass <= 0.0:
            raise AssertionError("nonpositive syndrome mass")
        posterior = np.zeros(label_count, dtype=np.float64)
        for index in indices:
            posterior[int(labels[index])] += physical_weights[index] / mass
        if not np.isclose(posterior.sum(), 1.0, atol=2e-15, rtol=0.0):
            raise AssertionError("logical posterior is not normalized")
        reference = int(indices[0])
        for index in indices:
            index = int(index)
            actual = math.log(float(physical_weights[index])) - math.log(float(physical_weights[reference]))
            expected = int(weights[index] - weights[reference]) * log_b
            max_log_ratio_error = max(max_log_ratio_error, abs(actual - expected))
        y_probabilities.append(mass)
        logical_posteriors.append(posterior)
        support_counts.append(int(indices.size))
    y_probabilities = np.asarray(y_probabilities, dtype=np.float64)
    logical_posteriors = np.asarray(logical_posteriors, dtype=np.float64)
    if not np.isclose(y_probabilities.sum(), 1.0, atol=2e-15, rtol=0.0):
        raise AssertionError("syndrome masses are not normalized")
    if max_log_ratio_error > 2e-13:
        raise AssertionError("hard-coset b^Delta-weight ratio failed")
    golden = {
        "frame_fingerprint": frame.fingerprint(),
        "hard_coset_support_verified": True,
        "k": int(model.k),
        "logical_posteriors": logical_posteriors.tolist(),
        "max_log_b_weight_ratio_error": float(max_log_ratio_error),
        "model_fingerprint": model.fingerprint(),
        "model_id": str(model_id),
        "n": int(model.num_qubits),
        "num_checks": int(model.num_checks),
        "p": p,
        "physics_contract_version": "exp101.physics.v2",
        "posterior_ensemble": "true_posterior",
        "section_fingerprint": model.logical_sector_section.fingerprint(),
        "sector": "x_error",
        "state_table_sha256": _state_table_sha(states, syndromes, labels),
        "support_counts": support_counts,
        "syndrome_keys_hex": [key.hex() for key in keys],
        "syndrome_probabilities": y_probabilities.tolist(),
    }
    return golden, logical_posteriors, y_probabilities


def _masks_and_groups(k):
    masks = np.arange(1, 1 << int(k), dtype=np.uint64)
    basis = np.asarray([(int(mask) & (int(mask) - 1)) == 0 for mask in masks], dtype=bool)
    return masks, {
        "basis_max": np.flatnonzero(basis),
        "nonbasis_max": np.flatnonzero(~basis),
        "omnibus": np.arange(masks.size, dtype=np.int64),
    }


def _walsh(distributions):
    distributions = np.asarray(distributions, dtype=np.float64)
    label_count = distributions.shape[1]
    labels = np.arange(label_count, dtype=np.uint64)
    masks, groups = _masks_and_groups(int(round(math.log2(label_count))))
    signs = 1.0 - 2.0 * (
        np.bitwise_count(labels[:, None] & masks[None, :]) & np.uint8(1)
    ).astype(np.float64)
    return distributions @ signs, signs, masks, groups


def oracle_candidate_metrics(target, candidate, y_probabilities):
    target = np.asarray(target, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    y_probabilities = np.asarray(y_probabilities, dtype=np.float64)
    target_means, _, masks, groups = _walsh(target)
    candidate_means, _, _, _ = _walsh(candidate)
    differences = np.sum(
        y_probabilities[:, None]
        * (np.square(candidate_means) - target_means * candidate_means),
        axis=0,
    )
    collision_by_y = np.sum(np.square(candidate), axis=1)
    planted_by_y = np.sum(target * candidate, axis=1)
    label_count = target.shape[1]
    normalization = 1.0 - 1.0 / label_count
    target_purity = float(np.sum(y_probabilities * np.sum(np.square(target), axis=1)))
    candidate_purity = float(np.sum(y_probabilities * collision_by_y))
    group_metrics = {}
    for name, indices in groups.items():
        if name == "omnibus":
            group_metrics[name] = {
                "applicable": True,
                "exact_effect": float(np.mean(differences)),
                "max_abs_exact_effect": float(np.max(np.abs(differences))),
            }
        elif indices.size == 0:
            group_metrics[name] = {
                "applicable": False,
                "exact_effect": None,
                "max_abs_exact_effect": None,
            }
        else:
            group_metrics[name] = {
                "applicable": True,
                "exact_effect": None,
                "max_abs_exact_effect": float(np.max(np.abs(differences[indices]))),
            }
    return {
        "candidate_q_top": float((candidate_purity - 1.0 / label_count) / normalization),
        "character_masks": [int(mask) for mask in masks],
        "collision_mass": candidate_purity,
        "group_exact_metrics": group_metrics,
        "max_abs_per_character_identity_difference": float(np.max(np.abs(differences))),
        "mean_per_character_identity_difference": float(np.mean(differences)),
        "per_character_identity_difference": differences.tolist(),
        "planted_hit": float(np.sum(y_probabilities * planted_by_y)),
        "scalar_identity_difference": float(np.sum(y_probabilities * (collision_by_y - planted_by_y))),
        "target_q_top": float((target_purity - 1.0 / label_count) / normalization),
    }


def oracle_controls(target, wrong_temperature):
    label_count = target.shape[1]
    permuted = np.empty_like(target)
    for label in range(label_count):
        permuted[:, label] = target[:, label ^ 1]
    map_delta = np.zeros_like(target)
    map_delta[np.arange(target.shape[0]), np.argmax(target, axis=1)] = 1.0
    return {
        "correct_posterior": target.copy(),
        "label_permutation": permuted,
        "truth_blind_map_delta": map_delta,
        "uniform_logical": np.full_like(target, 1.0 / label_count),
        "wrong_temperature": wrong_temperature,
    }


def _score_population(target, candidate, y_probabilities):
    candidate_means, signs, _, groups = _walsh(candidate)
    scores = []
    probabilities = []
    for y_index in range(target.shape[0]):
        for label in range(target.shape[1]):
            scores.append(
                np.square(candidate_means[y_index])
                - signs[label] * candidate_means[y_index]
            )
            probabilities.append(float(y_probabilities[y_index] * target[y_index, label]))
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.asarray(scores, dtype=np.float64), probabilities, groups


def _rate_mcse(rate, replications):
    return float(math.sqrt(max(rate * (1.0 - rate), 0.0) / replications))


def oracle_power_curve(target, candidate, y_probabilities, config, identity):
    scores, probabilities, groups = _score_population(target, candidate, y_probabilities)
    reps = int(config["power"]["replications"])
    margin = float(config["diagnostic_equivalence"]["margin"])
    sigma = float(config["diagnostic_equivalence"]["sigma_multiplier"])
    slack = float(config["diagnostic_equivalence"]["sigma_slack"])
    rows = []
    for ensemble_size in config["power"]["ensemble_sizes"]:
        ensemble_size = int(ensemble_size)
        rng = np.random.default_rng(derive_seed(config["seed_namespace"], "power", *identity, ensemble_size))
        indices = rng.choice(scores.shape[0], size=(reps, ensemble_size), p=probabilities)
        sampled = scores[indices]
        per_character_means = sampled.mean(axis=1)
        per_character_ses = sampled.std(axis=1, ddof=1) / math.sqrt(ensemble_size)
        statistics = {}
        for group_name, group_indices in groups.items():
            if group_name == "omnibus":
                scalar_samples = sampled.mean(axis=2)
                means = scalar_samples.mean(axis=1)
                ses = scalar_samples.std(axis=1, ddof=1) / math.sqrt(ensemble_size)
                equivalence = np.abs(means) + sigma * ses <= margin
                rejection = np.abs(means) > sigma * ses + slack
            elif group_indices.size == 0:
                statistics[group_name] = {
                    "applicable": False,
                    "diagnostic_equivalence_pass_rate": None,
                    "equality_rejection_rate": None,
                    "rate_mcse_upper_bound": None,
                }
                continue
            else:
                bounds = np.abs(per_character_means[:, group_indices]) + sigma * per_character_ses[:, group_indices]
                rejection_bounds = np.abs(per_character_means[:, group_indices]) - sigma * per_character_ses[:, group_indices]
                equivalence = np.all(bounds <= margin, axis=1)
                rejection = np.any(rejection_bounds > slack, axis=1)
            equivalence_rate = float(np.mean(equivalence))
            rejection_rate = float(np.mean(rejection))
            statistics[group_name] = {
                "applicable": True,
                "diagnostic_equivalence_pass_rate": equivalence_rate,
                "equality_rejection_rate": rejection_rate,
                "rate_mcse_upper_bound": max(
                    _rate_mcse(equivalence_rate, reps), _rate_mcse(rejection_rate, reps)
                ),
            }
        rows.append({
            "ensemble_size": ensemble_size,
            "optimistic_no_sampler_noise": True,
            "replications": reps,
            "statistics": statistics,
        })
    return rows


def chain_character_metrics(k, chain_labels, truth_label):
    chain_labels = [int(value) for value in chain_labels]
    truth_label = int(truth_label)
    if len(chain_labels) < 2:
        raise ValueError("chain control needs at least two trajectories")
    label_count = 1 << int(k)
    if any(not 0 <= value < label_count for value in [*chain_labels, truth_label]):
        raise ValueError("chain-control label outside range")
    masks, groups = _masks_and_groups(k)
    labels = np.asarray(chain_labels, dtype=np.uint64)
    signs = 1.0 - 2.0 * (
        np.bitwise_count(labels[:, None] & masks[None, :]) & np.uint8(1)
    ).astype(np.float64)
    truth_signs = 1.0 - 2.0 * (
        np.bitwise_count(np.uint64(truth_label) & masks) & np.uint8(1)
    ).astype(np.float64)
    count = signs.shape[0]
    pair_sum = np.square(signs.sum(axis=0)) - np.square(signs).sum(axis=0)
    m2 = pair_sum / (count * (count - 1))
    planted = truth_signs * signs.mean(axis=0)
    differences = m2 - planted
    equal_pairs = sum(
        chain_labels[left] == chain_labels[right]
        for left in range(count) for right in range(count) if left != right
    )
    collision = equal_pairs / (count * (count - 1))
    planted_hit = sum(value == truth_label for value in chain_labels) / count
    group_metrics = {}
    for name, indices in groups.items():
        if name == "omnibus":
            group_metrics[name] = {
                "applicable": True,
                "exact_effect": float(np.mean(differences)),
                "max_abs_exact_effect": float(np.max(np.abs(differences))),
            }
        elif indices.size == 0:
            group_metrics[name] = {
                "applicable": False,
                "exact_effect": None,
                "max_abs_exact_effect": None,
            }
        else:
            group_metrics[name] = {
                "applicable": True,
                "exact_effect": None,
                "max_abs_exact_effect": float(np.max(np.abs(differences[indices]))),
            }
    return {
        "chain_labels": chain_labels,
        "character_masks": [int(mask) for mask in masks],
        "collision_mass_u_statistic": float(collision),
        "group_exact_metrics": group_metrics,
        "m2_debiased_per_character": m2.tolist(),
        "per_character_identity_difference": differences.tolist(),
        "planted_cross_moment_per_character": planted.tolist(),
        "planted_hit": float(planted_hit),
        "scalar_identity_difference": float(collision - planted_hit),
        "truth_label": truth_label,
    }


def oracle_chain_controls():
    p = np.asarray([0.9, 0.1], dtype=np.float64)
    q = np.asarray([0.5, 0.5], dtype=np.float64)
    return {
        "common_planted_freeze": chain_character_metrics(4, [11, 11, 11, 11], 11),
        "four_distinct_label_freeze": chain_character_metrics(4, [0, 1, 2, 3], 15),
        "two_label_equal_moment_counterexample": {
            "candidate_q_top": 0.0,
            "collision_mass": float(np.dot(q, q)),
            "planted_hit": float(np.dot(p, q)),
            "scalar_identity_difference": float(np.dot(q, q) - np.dot(p, q)),
            "target_q_top": 0.64,
        },
    }


def validate_control_catalog(config):
    exact_names = set(config["controls"]["exact"])
    chain_names = set(config["controls"]["chain_level"])
    expected_exact = {
        "correct_posterior", "label_permutation", "truth_blind_map_delta",
        "uniform_logical", "wrong_temperature",
    }
    expected_chain = {
        "common_planted_freeze", "four_distinct_label_freeze",
        "two_label_equal_moment_counterexample",
    }
    if exact_names != expected_exact or chain_names != expected_chain:
        raise RuntimeError("control catalog changed")


def evaluate_calibration_gate(config, exact_rows, power_rows, chain_controls):
    validate_control_catalog(config)
    tolerance = float(config["calibration_gate"]["exact_tolerance"])
    effect_floor = float(config["calibration_gate"]["detected_effect_floor"])
    min_detection = float(config["calibration_gate"]["minimum_detection_rate"])
    min_equivalence = float(config["calibration_gate"]["minimum_equivalence_rate"])
    gate_size = int(config["calibration_gate"]["power_gate_ensemble_size"])
    failures = []
    expectations = config["controls"]["exact"]
    power_by_key = {
        (row["model_id"], float(row["p"]), row["control"]): row for row in power_rows
    }
    for row in exact_rows:
        expectation = expectations[row["control"]]
        power = power_by_key[(row["model_id"], float(row["p"]), row["control"])]
        gate_rows = [item for item in power["rows"] if int(item["ensemble_size"]) == gate_size]
        if len(gate_rows) != 1:
            failures.append(f"missing power gate row: {row['model_id']}/{row['p']}/{row['control']}")
            continue
        statistics = gate_rows[0]["statistics"]
        for group, outcome in expectation["expected_power_outcome"].items():
            stat = statistics[group]
            exact_group = row["group_exact_metrics"][group]
            if not stat["applicable"]:
                if group == "nonbasis_max" and int(row["k"]) == 1:
                    continue
                failures.append(f"unexpected NA group: {row['model_id']}/{row['p']}/{row['control']}/{group}")
                continue
            if outcome == "equivalent":
                if stat["diagnostic_equivalence_pass_rate"] < min_equivalence:
                    failures.append(f"equivalence power failed: {row['model_id']}/{row['p']}/{row['control']}/{group}")
                exact_value = (
                    abs(exact_group["exact_effect"]) if group == "omnibus"
                    else exact_group["max_abs_exact_effect"]
                )
                if exact_value > tolerance:
                    failures.append(f"equivalent exact effect failed: {row['model_id']}/{row['p']}/{row['control']}/{group}")
            elif outcome == "detected":
                if stat["equality_rejection_rate"] < min_detection:
                    failures.append(f"detection power failed: {row['model_id']}/{row['p']}/{row['control']}/{group}")
                exact_value = (
                    abs(exact_group["exact_effect"]) if group == "omnibus"
                    else exact_group["max_abs_exact_effect"]
                )
                if exact_value < effect_floor:
                    failures.append(f"detected exact effect too small: {row['model_id']}/{row['p']}/{row['control']}/{group}")
            else:
                failures.append(f"unknown expected outcome {outcome!r}")

    common = chain_controls["common_planted_freeze"]
    four = chain_controls["four_distinct_label_freeze"]
    two = chain_controls["two_label_equal_moment_counterexample"]
    if abs(common["scalar_identity_difference"]) > tolerance or max(
        abs(value) for value in common["per_character_identity_difference"]
    ) > tolerance:
        failures.append("common planted freeze did not remain a full blind control")
    if abs(four["scalar_identity_difference"]) > tolerance:
        failures.append("four-distinct scalar identity was not blind")
    if abs(four["group_exact_metrics"]["omnibus"]["exact_effect"]) > tolerance:
        failures.append("four-distinct omnibus character identity was not blind")
    for group in ("basis_max", "nonbasis_max"):
        if four["group_exact_metrics"][group]["max_abs_exact_effect"] < effect_floor:
            failures.append(f"four-distinct {group} did not expose the sparse discrepancy")
    if abs(two["scalar_identity_difference"]) > tolerance:
        failures.append("two-label counterexample stopped being identity-blind")
    if abs(two["target_q_top"] - two["candidate_q_top"]) < 0.5:
        failures.append("two-label counterexample lost its q_top discrepancy")
    return {
        "failures": failures,
        "passed": not failures,
        "power_is_optimistic_no_sampler_noise": True,
        "universal_q_top_bias_bound": None,
    }


def build_oracle_calibration(config, *, include_power=True):
    validate_control_catalog(config)
    golden_rows = []
    exact_rows = []
    power_rows = []
    for model_spec in config["exact_models"]:
        cached = {}
        for p in config["p_values"]:
            golden, target, y_probabilities = enumerate_physics_v2(
                model_spec["H"], p, model_spec["id"]
            )
            cached[float(p)] = (target, y_probabilities)
            golden_rows.append(golden)
            wrong_p = float(config["wrong_temperature_map"][str(p)])
            if wrong_p not in cached:
                _, wrong, wrong_y = enumerate_physics_v2(model_spec["H"], wrong_p, model_spec["id"])
                cached[wrong_p] = (wrong, wrong_y)
            wrong = cached[wrong_p][0]
            for control, candidate in oracle_controls(target, wrong).items():
                metrics = oracle_candidate_metrics(target, candidate, y_probabilities)
                exact_rows.append({
                    "control": control,
                    "k": int(golden["k"]),
                    "model_id": model_spec["id"],
                    "p": float(p),
                    "wrong_temperature_p": wrong_p if control == "wrong_temperature" else None,
                    **metrics,
                })
                if include_power:
                    power_rows.append({
                        "control": control,
                        "model_id": model_spec["id"],
                        "p": float(p),
                        "rows": oracle_power_curve(
                            target, candidate, y_probabilities, config,
                            (model_spec["id"], float(p), control),
                        ),
                    })
    chain_controls = oracle_chain_controls()
    gate = (
        evaluate_calibration_gate(config, exact_rows, power_rows, chain_controls)
        if include_power else None
    )
    return {
        "calibration_gate": gate,
        "chain_level_control_metrics": chain_controls,
        "exact_control_rows": exact_rows,
        "golden_rows": golden_rows,
        "oracle_version": ORACLE_VERSION,
        "power_rows": power_rows,
    }
