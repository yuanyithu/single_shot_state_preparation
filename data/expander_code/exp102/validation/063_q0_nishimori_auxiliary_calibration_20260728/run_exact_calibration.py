"""Exact q=0 Nishimori identities, controls, and optimistic power."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.expander_code.exp102.exp102_pipeline.worker import build_model


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "nishimori_config.json"
SCHEMA_PATH = ROOT / "nishimori_raw_schema.v1.json"
OUTPUT_PATH = ROOT / "exact_calibration_report.json"
EXPECTED_AUTHORITY = {
    "formal_authorization": False,
    "maximum_status": "NISHIMORI_AUXILIARY_AUDIT_CALIBRATED_WITH_KNOWN_BLIND_CONTROLS",
    "posterior_estimation": False,
    "production_authorization": False,
    "remote_authorization": False,
    "sole_confirmer_authorization": False,
}
CALIBRATED_STATUS = "NISHIMORI_AUXILIARY_AUDIT_CALIBRATED_WITH_KNOWN_BLIND_CONTROLS"
INSUFFICIENT_STATUS = "NISHIMORI_AUXILIARY_CALIBRATION_INSUFFICIENT"
# Kept as the maximum-success status for callers that audit the authority cap.
EXPECTED_STATUS = CALIBRATED_STATUS


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def derive_seed(namespace, *parts):
    payload = canonical([namespace, *parts]).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def all_binary_states(size):
    masks = np.arange(1 << int(size), dtype=np.uint64)
    bits = ((masks[:, None] >> np.arange(size, dtype=np.uint64)) & 1)
    return bits.astype(np.uint8)


def packed_syndromes(model, states):
    syndromes = (
        model.H_check.astype(np.int64) @ states.T.astype(np.int64) % 2
    ).T.astype(np.uint8)
    return [np.packbits(row, bitorder="little").tobytes() for row in syndromes]


def integer_labels(frame, states):
    result = []
    for state in states:
        label = frame.label_of(state)
        result.append(sum(int(bit) << index for index, bit in enumerate(label)))
    return np.asarray(result, dtype=np.int64)


def bernoulli_weights(states, p):
    weights = np.power(float(p), states.sum(axis=1)) * np.power(
        1.0 - float(p), states.shape[1] - states.sum(axis=1)
    )
    weights = np.asarray(weights, dtype=np.float64)
    weights /= weights.sum()
    return weights


def conditional_ensemble(model, frame, states, syndromes, labels, p):
    weights = bernoulli_weights(states, p)
    label_count = 1 << model.k
    keys = sorted(set(syndromes))
    y_probabilities = []
    posteriors = []
    for key in keys:
        indices = np.asarray([index for index, value in enumerate(syndromes) if value == key])
        mass = float(weights[indices].sum())
        posterior = np.zeros(label_count, dtype=np.float64)
        for index in indices:
            posterior[int(labels[index])] += weights[index]
        posterior /= mass
        y_probabilities.append(mass)
        posteriors.append(posterior)
    y_probabilities = np.asarray(y_probabilities, dtype=np.float64)
    y_probabilities /= y_probabilities.sum()
    return keys, y_probabilities, np.asarray(posteriors), weights


def _state_table_sha(states, syndromes, labels):
    digest = hashlib.sha256(b"exp102.nishimori.physical_state_table.v1\0")
    for index, (state, syndrome, label) in enumerate(zip(states, syndromes, labels)):
        digest.update(int(index).to_bytes(8, "big"))
        digest.update(int(state.sum()).to_bytes(4, "big"))
        digest.update(len(syndrome).to_bytes(4, "big"))
        digest.update(syndrome)
        digest.update(int(label).to_bytes(8, "big"))
    return digest.hexdigest()


def runner_golden(model, frame, states, syndromes, labels, p, model_id):
    keys, y_probabilities, posterior, physical_weights = conditional_ensemble(
        model, frame, states, syndromes, labels, p
    )
    state_weights = states.sum(axis=1).astype(np.int64)
    log_b = math.log(float(p) / (1.0 - float(p)))
    support_counts = []
    max_error = 0.0
    for key in keys:
        indices = np.asarray([index for index, value in enumerate(syndromes) if value == key])
        support_counts.append(int(indices.size))
        reference = int(indices[0])
        for index in indices:
            index = int(index)
            actual = math.log(float(physical_weights[index])) - math.log(float(physical_weights[reference]))
            expected = int(state_weights[index] - state_weights[reference]) * log_b
            max_error = max(max_error, abs(actual - expected))
    if max_error > 2e-13:
        raise RuntimeError("runner hard-coset b^Delta-weight ratio failed")
    return {
        "frame_fingerprint": frame.fingerprint(),
        "hard_coset_support_verified": True,
        "k": int(model.k),
        "logical_posteriors": posterior.tolist(),
        "max_log_b_weight_ratio_error": float(max_error),
        "model_fingerprint": model.fingerprint(),
        "model_id": str(model_id),
        "n": int(model.num_qubits),
        "num_checks": int(model.num_checks),
        "p": float(p),
        "physics_contract_version": "exp101.physics.v2",
        "posterior_ensemble": "true_posterior",
        "section_fingerprint": model.logical_sector_section.fingerprint(),
        "sector": "x_error",
        "state_table_sha256": _state_table_sha(states, syndromes, labels),
        "support_counts": support_counts,
        "syndrome_keys_hex": [key.hex() for key in keys],
        "syndrome_probabilities": y_probabilities.tolist(),
    }, posterior, y_probabilities


def character_masks_and_groups(k):
    masks = np.arange(1, 1 << int(k), dtype=np.uint64)
    basis = np.asarray([(int(mask) & (int(mask) - 1)) == 0 for mask in masks], dtype=bool)
    return masks, {
        "basis_max": np.flatnonzero(basis),
        "nonbasis_max": np.flatnonzero(~basis),
        "omnibus": np.arange(masks.size, dtype=np.int64),
    }


def walsh_means(distributions):
    distributions = np.asarray(distributions, dtype=np.float64)
    label_count = distributions.shape[1]
    labels = np.arange(label_count, dtype=np.uint64)
    masks, groups = character_masks_and_groups(int(round(math.log2(label_count))))
    signs = 1.0 - 2.0 * (
        np.bitwise_count(labels[:, None] & masks[None, :]) & np.uint8(1)
    ).astype(np.float64)
    return distributions @ signs, signs, masks, groups


def candidate_metrics(target, candidate, y_probabilities):
    target_means, _, masks, groups = walsh_means(target)
    candidate_means, _, _, _ = walsh_means(candidate)
    differences = np.sum(
        y_probabilities[:, None]
        * (np.square(candidate_means) - target_means * candidate_means),
        axis=0,
    )
    collision_by_y = np.square(candidate).sum(axis=1)
    planted_by_y = (target * candidate).sum(axis=1)
    target_purity = float(np.sum(y_probabilities * np.square(target).sum(axis=1)))
    candidate_purity = float(np.sum(y_probabilities * collision_by_y))
    label_count = target.shape[1]
    normalization = 1.0 - 1.0 / label_count
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


def control_candidates(target, wrong_temperature):
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


def score_population(target, candidate, y_probabilities):
    candidate_means, signs, _, groups = walsh_means(candidate)
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


def power_curve(target, candidate, y_probabilities, config, identity):
    scores, probabilities, groups = score_population(target, candidate, y_probabilities)
    rows = []
    reps = int(config["power"]["replications"])
    margin = float(config["diagnostic_equivalence"]["margin"])
    sigma = float(config["diagnostic_equivalence"]["sigma_multiplier"])
    slack = float(config["diagnostic_equivalence"]["sigma_slack"])
    for ensemble_size in config["power"]["ensemble_sizes"]:
        ensemble_size = int(ensemble_size)
        rng = np.random.default_rng(derive_seed(
            config["seed_namespace"], "power", *identity, ensemble_size,
        ))
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
    masks, groups = character_masks_and_groups(k)
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
            group_metrics[name] = {"applicable": False, "exact_effect": None, "max_abs_exact_effect": None}
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


def chain_level_controls():
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
    if set(config["controls"]["exact"]) != {
        "correct_posterior", "label_permutation", "truth_blind_map_delta",
        "uniform_logical", "wrong_temperature",
    }:
        raise RuntimeError("exact control catalog changed")
    if set(config["controls"]["chain_level"]) != {
        "common_planted_freeze", "four_distinct_label_freeze",
        "two_label_equal_moment_counterexample",
    }:
        raise RuntimeError("chain-level control catalog changed")


def evaluate_calibration_gate(config, exact_rows, power_rows, chain_controls):
    validate_control_catalog(config)
    tolerance = float(config["calibration_gate"]["exact_tolerance"])
    effect_floor = float(config["calibration_gate"]["detected_effect_floor"])
    min_detection = float(config["calibration_gate"]["minimum_detection_rate"])
    min_equivalence = float(config["calibration_gate"]["minimum_equivalence_rate"])
    gate_size = int(config["calibration_gate"]["power_gate_ensemble_size"])
    failures = []
    power_by_key = {
        (row["model_id"], float(row["p"]), row["control"]): row for row in power_rows
    }
    for row in exact_rows:
        expectation = config["controls"]["exact"][row["control"]]
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
            exact_value = (
                abs(exact_group["exact_effect"]) if group == "omnibus"
                else exact_group["max_abs_exact_effect"]
            )
            if outcome == "equivalent":
                if stat["diagnostic_equivalence_pass_rate"] < min_equivalence or exact_value > tolerance:
                    failures.append(f"equivalence gate failed: {row['model_id']}/{row['p']}/{row['control']}/{group}")
            elif outcome == "detected":
                if stat["equality_rejection_rate"] < min_detection or exact_value < effect_floor:
                    failures.append(f"detection gate failed: {row['model_id']}/{row['p']}/{row['control']}/{group}")
            else:
                failures.append(f"unknown expected outcome {outcome!r}")
    common = chain_controls["common_planted_freeze"]
    four = chain_controls["four_distinct_label_freeze"]
    two = chain_controls["two_label_equal_moment_counterexample"]
    if abs(common["scalar_identity_difference"]) > tolerance or max(
        abs(value) for value in common["per_character_identity_difference"]
    ) > tolerance:
        failures.append("common planted freeze is not fully blind")
    if abs(four["scalar_identity_difference"]) > tolerance or abs(
        four["group_exact_metrics"]["omnibus"]["exact_effect"]
    ) > tolerance:
        failures.append("four-distinct omnibus is not blind")
    for group in ("basis_max", "nonbasis_max"):
        if four["group_exact_metrics"][group]["max_abs_exact_effect"] < effect_floor:
            failures.append(f"four-distinct {group} did not detect sparse freezing")
    if abs(two["scalar_identity_difference"]) > tolerance:
        failures.append("two-label identity is no longer blind")
    if abs(two["target_q_top"] - two["candidate_q_top"]) < 0.5:
        failures.append("two-label q_top discrepancy disappeared")
    return {
        "failures": failures,
        "passed": not failures,
        "power_is_optimistic_no_sampler_noise": True,
        "universal_q_top_bias_bound": None,
    }


def build_calibration_payload(config, *, include_power=True):
    validate_control_catalog(config)
    golden_rows = []
    exact_rows = []
    power_rows = []
    for model_spec in config["exact_models"]:
        H = np.asarray(model_spec["H"], dtype=np.uint8)
        model, frame = build_model(H)
        states = all_binary_states(model.num_qubits)
        syndromes = packed_syndromes(model, states)
        labels = integer_labels(frame, states)
        cached = {}
        for p in config["p_values"]:
            golden, target, y_probabilities = runner_golden(
                model, frame, states, syndromes, labels, p, model_spec["id"]
            )
            cached[float(p)] = target
            golden_rows.append(golden)
            wrong_p = float(config["wrong_temperature_map"][str(p)])
            if wrong_p not in cached:
                _, _, wrong, _ = conditional_ensemble(
                    model, frame, states, syndromes, labels, wrong_p
                )
                cached[wrong_p] = wrong
            wrong = cached[wrong_p]
            for control, candidate in control_candidates(target, wrong).items():
                metrics = candidate_metrics(target, candidate, y_probabilities)
                exact_rows.append({
                    "control": control,
                    "k": int(model.k),
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
                        "rows": power_curve(
                            target, candidate, y_probabilities, config,
                            (model_spec["id"], float(p), control),
                        ),
                    })
    chain_controls = chain_level_controls()
    gate = (
        evaluate_calibration_gate(config, exact_rows, power_rows, chain_controls)
        if include_power else None
    )
    return {
        "calibration_gate": gate,
        "chain_level_control_metrics": chain_controls,
        "exact_control_rows": exact_rows,
        "golden_rows": golden_rows,
        "power_rows": power_rows,
    }


def _git(*args):
    env = os.environ.copy()
    env["GIT_OPTIONAL_LOCKS"] = "0"
    return subprocess.run(
        ["git", *args], cwd=PROJECT_ROOT, check=True,
        capture_output=True, text=True, env=env,
    ).stdout


def source_identity(config, *, require_clean):
    if config["authority"] != EXPECTED_AUTHORITY:
        raise RuntimeError("auxiliary authority changed")
    bound = config["implementation"]["bound_files"]
    resolved = {}
    for role, descriptor in sorted(bound.items()):
        path = PROJECT_ROOT / descriptor["path"]
        if not path.is_file() or sha256_file(path) != descriptor["sha256"]:
            raise RuntimeError(f"bound source changed: {role}")
        if require_clean:
            tracked = subprocess.run(
                ["git", "ls-files", "--error-unmatch", descriptor["path"]],
                cwd=PROJECT_ROOT, capture_output=True, text=True,
            )
            if tracked.returncode != 0:
                raise RuntimeError(f"untracked bound source: {role}")
        resolved[role] = dict(descriptor)
    config_relative = CONFIG_PATH.relative_to(PROJECT_ROOT).as_posix()
    if require_clean:
        if subprocess.run(
            ["git", "ls-files", "--error-unmatch", config_relative],
            cwd=PROJECT_ROOT, capture_output=True, text=True,
        ).returncode != 0:
            raise RuntimeError("untracked Nishimori config")
        status = _git("status", "--porcelain=v1", "--untracked-files=all")
        if status:
            raise RuntimeError("entire Nishimori source worktree is not clean")
        bytecode = [
            path for path in PROJECT_ROOT.rglob("*")
            if path.name == "__pycache__" or (path.is_file() and path.suffix == ".pyc")
        ]
        if bytecode:
            raise RuntimeError("source worktree contains Python bytecode")
    source_commit = _git("rev-parse", "HEAD").strip()
    config_sha256 = sha256_file(CONFIG_PATH)
    tree_core = {
        "bound_files": resolved,
        "config_sha256": config_sha256,
        "source_commit": source_commit,
    }
    return {
        **tree_core,
        "source_tree_sha256": hashlib.sha256(canonical(tree_core).encode("ascii")).hexdigest(),
    }


def terminal_status(calibration_gate):
    if not isinstance(calibration_gate, dict):
        raise RuntimeError("calibration gate is missing")
    passed = calibration_gate.get("passed")
    failures = calibration_gate.get("failures")
    if not isinstance(passed, bool) or not isinstance(failures, list):
        raise RuntimeError("calibration gate fields changed")
    if any(not isinstance(failure, str) or not failure for failure in failures):
        raise RuntimeError("calibration gate failure is invalid")
    if passed != (len(failures) == 0):
        raise RuntimeError("calibration gate result contradicts its failures")
    return CALIBRATED_STATUS if passed else INSUFFICIENT_STATUS


def build_report_core(config, identity, payload):
    core = {
        "authority": EXPECTED_AUTHORITY,
        **identity,
        **payload,
        "runner_sha256": sha256_file(Path(__file__)),
        "schema_sha256": sha256_file(SCHEMA_PATH),
        "status": terminal_status(payload.get("calibration_gate")),
        "universal_q_top_bias_bound_from_identity": None,
        "version": config["version"],
    }
    core["report_sha256"] = hashlib.sha256(canonical(core).encode("ascii")).hexdigest()
    return core


def main():
    if OUTPUT_PATH.exists():
        raise RuntimeError("exact calibration report already exists")
    config = json.loads(CONFIG_PATH.read_text(encoding="ascii"))
    identity = source_identity(config, require_clean=True)
    payload = build_calibration_payload(config, include_power=True)
    core = build_report_core(config, identity, payload)
    with OUTPUT_PATH.open("x", encoding="ascii") as handle:
        handle.write(canonical(core) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps({
        "exact_control_row_count": len(core["exact_control_rows"]),
        "report_sha256": core["report_sha256"],
        "status": core["status"],
    }, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
