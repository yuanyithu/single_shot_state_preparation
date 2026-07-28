"""Calibrate one deployable character-equivalence gate on IID oracle draws."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from statistics import NormalDist
import subprocess
import sys

import numpy as np


# Importing the frozen model builder must not dirty a clean source worktree.
sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.expander_code.exp102.exp102_pipeline.worker import build_model


ROOT = Path(__file__).resolve().parent
EXP102_ROOT = ROOT.parents[1]
CONFIG_PATH = ROOT / "calibration_config.json"
OUTPUT_PATH = ROOT / "calibration_report.json"
AUDIT_PATH = ROOT / "independent_audit.json"
COMMIT_RE = re.compile(r"[0-9a-f]{40}")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def load_json_strict(path):
    def reject_constant(value):
        raise ValueError(f"non-finite JSON constant: {value}")

    return json.loads(
        Path(path).read_text(encoding="ascii"), parse_constant=reject_constant,
    )


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_config_self_hash(config):
    expected = config["config_self_sha256"]
    unsigned = dict(config)
    unsigned.pop("config_self_sha256")
    actual = sha256_bytes(canonical(unsigned).encode("ascii"))
    if actual != expected:
        raise RuntimeError("calibration config self hash changed")


def validate_config(config):
    verify_config_self_hash(config)
    rule = config["candidate_rule"]
    if float(rule["tolerance"]) != 0.04 or float(rule["minimum_bad_margin"]) != 0.02:
        raise RuntimeError("character tolerance or bad-shift margin changed")
    if set(rule["required_roles"]) != {"collapsed_B", "logical"}:
        raise RuntimeError("simultaneous role registry changed")
    if set(config["injected_character_shifts"]) != {0.0, 0.02, 0.04, 0.06}:
        raise RuntimeError("exact injected-shift registry changed")
    synthetic = config["synthetic_multiplicity_stress"]
    catalogs = synthetic["catalogs"]
    if {int(row["size"]) for row in catalogs} != {15, 163, 511, 688, 4160}:
        raise RuntimeError("synthetic catalog-size registry changed")
    if len({row["id"] for row in catalogs}) != len(catalogs):
        raise RuntimeError("duplicate synthetic catalog id")
    if any(row["role"] not in rule["required_roles"] for row in catalogs):
        raise RuntimeError("synthetic catalog has an unknown simultaneous role")
    calibrated = {
        (row["role"], int(row["size"])) for row in catalogs
        if row["calibrate_multiplier"]
    }
    if calibrated != {("collapsed_B", 688), ("logical", 4160)}:
        raise RuntimeError("role-multiplier calibration catalogs changed")
    if set(synthetic["character_means"]) != {0.0, 0.8}:
        raise RuntimeError("synthetic base-mean registry changed")
    if 0.0 not in synthetic["single_character_shifts"] or 0.06 not in synthetic[
        "single_character_shifts"
    ]:
        raise RuntimeError("synthetic single-character null or bad shift missing")
    if 0.04 not in synthetic["distributed_character_shifts"]:
        raise RuntimeError("distributed .04 purity witness is missing")
    points = ordered_operating_points(config)
    if not points or any(trajectories < 2 or draws <= 0 for trajectories, draws in points):
        raise RuntimeError("common operating grid is invalid")
    for name in ("calibration_trials", "selection_trials", "confirmation_trials"):
        if int(config["replications"][name]) <= 0:
            raise RuntimeError(f"invalid replication count: {name}")
    required_artifacts = {"auditor", "readme", "red_team", "runner", "tests"}
    if set(config["source_artifacts"]) != required_artifacts:
        raise RuntimeError("source-artifact registry changed")
    if config["authority"]["maximum_status"] != (
        "CHARACTER_GATE_COMMON_OPERATING_POINT_CONFIRMED"
    ):
        raise RuntimeError("calibration authority changed")


def derive_seed(namespace, *parts):
    payload = canonical([namespace, *parts]).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _exp102_path(relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError("configured source path escapes exp102 root")
    path = (EXP102_ROOT / relative).resolve()
    try:
        path.relative_to(EXP102_ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError("configured source path escapes exp102 root") from exc
    if path.is_symlink():
        raise RuntimeError(f"frozen source may not be a symlink: {path}")
    return path


def configured_source_artifacts(config):
    rows = []
    for name, spec in sorted(config["source_artifacts"].items()):
        path = _exp102_path(spec["path"])
        expected = spec["sha256"]
        if re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise RuntimeError(f"invalid source SHA for {name}")
        rows.append((path, expected))
    if len({path for path, _ in rows}) != len(rows):
        raise RuntimeError("duplicate configured source artifact")
    return rows


def reject_validation_bytecode(root=ROOT):
    root = Path(root)
    offenders = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.name == "__pycache__" or path.suffix in {".pyc", ".pyo"}
    )
    if offenders:
        raise RuntimeError(f"validation source contains bytecode: {offenders[0]}")


def _git(args, *, text=True):
    return subprocess.run(
        ["git", *args], cwd=PROJECT_ROOT, check=True,
        capture_output=True, text=text,
    ).stdout


def require_completely_clean_worktree():
    if _git(["status", "--porcelain=v1", "--untracked-files=all"]):
        raise RuntimeError("validation 062 requires a completely clean worktree")


def verify_committed_clean_source(config):
    validate_config(config)
    reject_validation_bytecode()
    if OUTPUT_PATH.exists() or AUDIT_PATH.exists():
        raise RuntimeError("validation 062 output already exists")
    repository = Path(_git(["rev-parse", "--show-toplevel"]).strip()).resolve()
    if repository != PROJECT_ROOT.resolve():
        raise RuntimeError("validation is outside its bound repository")
    source_commit = _git(["rev-parse", "HEAD"]).strip()
    if COMMIT_RE.fullmatch(source_commit) is None:
        raise RuntimeError("HEAD is not a full source identity")
    require_completely_clean_worktree()

    artifacts = [(CONFIG_PATH.resolve(), sha256_file(CONFIG_PATH))]
    artifacts.extend(configured_source_artifacts(config))
    rows = []
    for path, expected in artifacts:
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"configured source SHA changed: {path}")
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        _git(["ls-files", "--error-unmatch", "--", relative])
        committed = _git(["show", f"{source_commit}:{relative}"], text=False)
        if sha256_bytes(committed) != expected:
            raise RuntimeError(f"source commit bytes changed: {relative}")
        rows.append([relative, expected])
    rows.sort()
    return {
        "source_commit": source_commit,
        "source_file_count": len(rows),
        "source_tree_sha256": sha256_bytes(canonical(rows).encode("ascii")),
    }


def gf2_rank(matrix):
    work = np.asarray(matrix, dtype=np.uint8).copy()
    rank = 0
    for column in range(work.shape[1]):
        pivots = np.flatnonzero(work[rank:, column])
        if not pivots.size:
            continue
        pivot = rank + int(pivots[0])
        work[[rank, pivot]] = work[[pivot, rank]]
        for row in range(work.shape[0]):
            if row != rank and work[row, column]:
                work[row] ^= work[rank]
        rank += 1
        if rank == work.shape[0]:
            break
    return rank


def state_key(state):
    return np.packbits(state, bitorder="little").tobytes()


def coset_states(model, syndrome):
    generators = np.vstack((model.stabilizer_rows, model.logical_move_basis))
    expected_nullity = model.num_qubits - gf2_rank(model.H_check)
    if generators.shape[0] != expected_nullity or gf2_rank(generators) != expected_nullity:
        raise RuntimeError("hard-coset generators do not span the full kernel")
    base = model.logical_sector_section.apply(syndrome, strict=True)
    states = np.repeat(base[None, :], 1 << expected_nullity, axis=0)
    for coefficient in range(states.shape[0]):
        for bit, row in enumerate(generators):
            if (coefficient >> bit) & 1:
                states[coefficient] ^= row
    if len({state_key(state) for state in states}) != states.shape[0]:
        raise RuntimeError("affine hard-coset enumeration is not bijective")
    residual = (
        model.H_check.astype(np.int64) @ states.T.astype(np.int64) % 2
    ).T.astype(np.uint8)
    if not np.all(residual == syndrome[None, :]):
        raise RuntimeError("enumerated state left the hard coset")
    return states


def posterior_probabilities(states, p):
    ratio = float(p) / (1.0 - float(p))
    log_weights = states.sum(axis=1, dtype=np.int64) * math.log(ratio)
    weights = np.exp(log_weights - log_weights.max())
    return weights / weights.sum()


def integer_labels(model, frame, states):
    values = []
    for state in states:
        label = frame.label_of(state)
        values.append(sum(int(bit) << index for index, bit in enumerate(label)))
    return np.asarray(values, dtype=np.int64)


def b_labels(classical_H, states):
    r, n = classical_H.shape
    dimension = r * r
    if dimension >= 63:
        raise RuntimeError("exact small-HGP B label exceeds int64 boundary")
    bits = states[:, n * n:n * n + dimension]
    powers = np.left_shift(np.int64(1), np.arange(dimension, dtype=np.int64))
    return bits.astype(np.int64) @ powers


def character_signs(values, dimension):
    masks = np.arange(1, 1 << int(dimension), dtype=np.int64)
    parity = np.bitwise_count(
        values[:, None].astype(np.uint64) & masks[None, :].astype(np.uint64)
    ) & np.uint8(1)
    return 1.0 - 2.0 * parity.astype(np.float64), masks


def build_exact_scenarios(config):
    scenarios = []
    for model_spec in config["exact_hgp"]["models"]:
        H = np.asarray(model_spec["H"], dtype=np.uint8)
        model, frame = build_model(H)
        for syndrome_kind in config["exact_hgp"]["syndromes"]:
            epsilon = np.zeros(model.num_qubits, dtype=np.uint8)
            if syndrome_kind == "nonzero":
                epsilon[[0, model.num_qubits - 1]] = 1
            syndrome = (
                model.H_check.astype(np.int64) @ epsilon.astype(np.int64) % 2
            ).astype(np.uint8)
            if bool(np.any(syndrome)) != (syndrome_kind == "nonzero"):
                raise RuntimeError("named exact syndrome has the wrong zero status")
            states = coset_states(model, syndrome)
            logical_values = integer_labels(model, frame, states)
            B_values = b_labels(H, states)
            logical_signs, _ = character_signs(logical_values, model.k)
            B_signs, _ = character_signs(B_values, H.shape[0] ** 2)
            for p in config["exact_hgp"]["p_values"]:
                probabilities = posterior_probabilities(states, p)
                for catalog, signs, dimension in (
                    ("logical", logical_signs, model.k),
                    ("collapsed_B", B_signs, H.shape[0] ** 2),
                ):
                    role = "logical" if catalog == "logical" else "collapsed_B"
                    scenarios.append({
                        "catalog": catalog,
                        "catalog_size": int(signs.shape[1]),
                        "dimension": int(dimension),
                        "logical_signs": logical_signs,
                        "model_id": model_spec["id"],
                        "p": float(p),
                        "probabilities": probabilities,
                        "requirement_categories": [f"exact_{catalog}"],
                        "signs": signs,
                        "simultaneous_role": role,
                        "syndrome": syndrome.tolist(),
                        "syndrome_kind": syndrome_kind,
                    })
    return scenarios


def tilted_probabilities(probabilities, signs, requested_shift):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    signs = np.asarray(signs, dtype=np.float64)
    means = probabilities @ signs
    selected = int(np.argmin(np.abs(means)))
    current = float(means[selected])
    direction = -1.0 if current >= 0.0 else 1.0
    target = current + direction * float(requested_shift)
    plus = signs[:, selected] > 0.0
    plus_mass = float(probabilities[plus].sum())
    target_plus = (1.0 + target) / 2.0
    if not 0.0 < plus_mass < 1.0 or not 0.0 <= target_plus <= 1.0:
        raise RuntimeError("selected exact character has no valid shifted support")
    result = probabilities.copy()
    result[plus] *= target_plus / plus_mass
    result[~plus] *= (1.0 - target_plus) / (1.0 - plus_mass)
    result /= result.sum()
    shifted_means = result @ signs
    if abs(abs(shifted_means[selected] - current) - requested_shift) > 1e-12:
        raise RuntimeError("character tilt did not realize the requested shift")
    return result, selected, means, shifted_means


def sample_chain_means(rng, probabilities, signs, trials, trajectories, draws):
    counts = rng.multinomial(
        int(draws), probabilities, size=(int(trials), int(trajectories)),
    )
    return np.einsum("...s,su->...u", counts, signs, optimize=True) / float(draws)


def signed_differences_and_se(left, right):
    signed = left.mean(axis=1) - right.mean(axis=1)
    se = np.sqrt(
        left.var(axis=1, ddof=1) / left.shape[1]
        + right.var(axis=1, ddof=1) / right.shape[1]
    )
    return signed, se


def safe_standardized_max(error, se):
    ratio = np.zeros_like(error, dtype=np.float64)
    positive = se > 0.0
    ratio[positive] = np.abs(error[positive]) / se[positive]
    ratio[~positive & (np.abs(error) > 1e-15)] = np.inf
    return np.max(ratio, axis=1)


def higher_quantile(values, probability):
    values = np.sort(np.asarray(values, dtype=np.float64))
    index = int(math.ceil(float(probability) * values.size) - 1)
    return float(values[min(max(index, 0), values.size - 1)])


def wilson_lower(successes, trials, confidence):
    successes = int(successes)
    trials = int(trials)
    if not 0 <= successes <= trials or trials <= 0:
        raise ValueError("invalid binomial count")
    z = NormalDist().inv_cdf(float(confidence))
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = p + z * z / (2.0 * trials)
    radius = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials))
    return max(0.0, (center - radius) / denominator)


def _new_trial_accumulator(trials):
    return {
        "coverage": np.ones(trials, dtype=bool),
        "historical": np.ones(trials, dtype=bool),
        "max_estimated": np.zeros(trials, dtype=np.float64),
        "max_lower": np.zeros(trials, dtype=np.float64),
        "max_upper": np.zeros(trials, dtype=np.float64),
    }


def _update_trial_accumulator(accumulator, signed, se, true_signed, multiplier, config):
    delta = np.abs(signed)
    tolerance = float(config["candidate_rule"]["tolerance"])
    accumulator["max_upper"] = np.maximum(
        accumulator["max_upper"], np.max(delta + multiplier * se, axis=1),
    )
    accumulator["max_lower"] = np.maximum(
        accumulator["max_lower"],
        np.max(np.maximum(delta - multiplier * se, 0.0), axis=1),
    )
    accumulator["max_estimated"] = np.maximum(
        accumulator["max_estimated"], np.max(delta, axis=1),
    )
    accumulator["coverage"] &= np.all(
        np.abs(signed - true_signed[None, :]) <= multiplier * se + 1e-15,
        axis=1,
    )
    historical = config["historical_rule"]
    accumulator["historical"] &= np.all(
        (delta <= historical["absolute_tolerance"])
        & (delta <= historical["sigma_multiplier"] * se + historical["sigma_slack"]),
        axis=1,
    )
    if not np.isfinite(tolerance):
        raise RuntimeError("non-finite character tolerance")


def _finish_trial_accumulator(accumulator, config):
    tolerance = float(config["candidate_rule"]["tolerance"])
    passed = accumulator["max_upper"] <= tolerance
    failed = accumulator["max_lower"] > tolerance
    if np.any(passed & failed):
        raise RuntimeError("character decisions overlap")
    return {
        "candidate_fail": failed,
        "candidate_inconclusive": ~(passed | failed),
        "candidate_pass": passed,
        "coverage": accumulator["coverage"],
        "historical_pass": accumulator["historical"],
        "mean_max_abs_estimated_shift": float(np.mean(accumulator["max_estimated"])),
    }


def _rate_fields(statistics, config):
    trials = int(statistics["candidate_pass"].size)
    confidence = float(config["candidate_rule"]["binomial_lower_confidence"])
    fields = {"evaluation_trials": trials}
    for name in (
        "candidate_pass", "candidate_fail", "candidate_inconclusive",
        "coverage", "historical_pass",
    ):
        count = int(np.count_nonzero(statistics[name]))
        fields[f"{name}_count"] = count
        fields[f"{name}_rate"] = float(count / trials)
        fields[f"{name}_wilson_lower"] = float(
            wilson_lower(count, trials, confidence)
        )
    fields["historical_fail_count"] = trials - fields["historical_pass_count"]
    fields["historical_fail_rate"] = float(
        fields["historical_fail_count"] / trials
    )
    fields["mean_max_abs_estimated_shift"] = statistics[
        "mean_max_abs_estimated_shift"
    ]
    return fields


def _distribution_fields(base_means, shifted_means, base_logical, shifted_logical):
    true_shift = np.asarray(base_means) - np.asarray(shifted_means)
    base_logical = np.asarray(base_logical, dtype=np.float64)
    shifted_logical = np.asarray(shifted_logical, dtype=np.float64)
    left_purity = float(np.mean(np.asarray(base_means) ** 2))
    right_purity = float(np.mean(np.asarray(shifted_means) ** 2))
    left_q_top = float(np.mean(base_logical ** 2))
    right_q_top = float(np.mean(shifted_logical ** 2))
    return {
        "base_character_means": np.asarray(base_means).tolist(),
        "shifted_character_means": np.asarray(shifted_means).tolist(),
        "true_character_purity_delta": abs(left_purity - right_purity),
        "true_character_purity_left": left_purity,
        "true_character_purity_right": right_purity,
        "true_character_shifts": true_shift.tolist(),
        "true_max_abs_character_shift": float(np.max(np.abs(true_shift))),
        "true_q_top_delta": abs(left_q_top - right_q_top),
        "true_q_top_left": left_q_top,
        "true_q_top_right": right_q_top,
        "base_logical_means": base_logical.tolist(),
        "shifted_logical_means": shifted_logical.tolist(),
    }


def vector_sha256(values):
    values = np.asarray(values, dtype=np.float64)
    return sha256_bytes(canonical(values.tolist()).encode("ascii"))


def vector_summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "maximum": float(np.max(values)),
        "mean_square": float(np.mean(values ** 2)),
        "minimum": float(np.min(values)),
        "nonzero_count": int(np.count_nonzero(np.abs(values) > 1e-14)),
    }


def synthetic_chain_means(rng, means, trials, trajectories, draws):
    probabilities = (1.0 + np.asarray(means, dtype=np.float64)) / 2.0
    plus = rng.binomial(
        int(draws), probabilities,
        size=(int(trials), int(trajectories), probabilities.size),
    )
    return 2.0 * plus / float(draws) - 1.0


def synthetic_null_maxima(
        rng, catalog_size, base_mean, trials, trajectories, draws, batch_size):
    maxima = np.zeros(int(trials), dtype=np.float64)
    for start in range(0, int(catalog_size), int(batch_size)):
        width = min(int(batch_size), int(catalog_size) - start)
        means = np.full(width, float(base_mean))
        left = synthetic_chain_means(rng, means, trials, trajectories, draws)
        right = synthetic_chain_means(rng, means, trials, trajectories, draws)
        signed, se = signed_differences_and_se(left, right)
        maxima = np.maximum(maxima, safe_standardized_max(signed, se))
    return maxima


def common_multipliers(config, config_sha, scenarios, trajectories, draws):
    namespace = config["seed_namespace"]
    calibration_trials = int(config["replications"]["calibration_trials"])
    quantile = float(config["candidate_rule"]["calibration_quantile"])
    scenario_quantiles = []
    for index, scenario in enumerate(scenarios):
        seed = derive_seed(
            namespace, config_sha, "common-z-exact", trajectories, draws, index,
        )
        rng = np.random.default_rng(seed)
        left = sample_chain_means(
            rng, scenario["probabilities"], scenario["signs"],
            calibration_trials, trajectories, draws,
        )
        right = sample_chain_means(
            rng, scenario["probabilities"], scenario["signs"],
            calibration_trials, trajectories, draws,
        )
        signed, se = signed_differences_and_se(left, right)
        scenario_quantiles.append({
            "catalog": scenario["catalog"],
            "catalog_size": scenario["catalog_size"],
            "model_id": scenario["model_id"],
            "p": scenario["p"],
            "quantile": higher_quantile(safe_standardized_max(signed, se), quantile),
            "scenario_index": int(index),
            "seed": int(seed),
            "simultaneous_role": scenario["simultaneous_role"],
            "source": "exact_hgp_iid",
            "syndrome_kind": scenario["syndrome_kind"],
        })
    synthetic = config["synthetic_multiplicity_stress"]
    for catalog in synthetic["catalogs"]:
        if not catalog["calibrate_multiplier"]:
            continue
        for base_mean in synthetic["character_means"]:
            seed = derive_seed(
                namespace, config_sha, "common-z-synthetic", trajectories, draws,
                catalog["size"], base_mean,
            )
            rng = np.random.default_rng(seed)
            maxima = synthetic_null_maxima(
                rng, catalog["size"], base_mean, calibration_trials,
                trajectories, draws, synthetic["character_batch_size"],
            )
            scenario_quantiles.append({
                "base_character_mean": float(base_mean),
                "catalog": catalog["id"],
                "catalog_size": int(catalog["size"]),
                "model_id": None,
                "p": None,
                "quantile": higher_quantile(maxima, quantile),
                "scenario_index": None,
                "seed": int(seed),
                "simultaneous_role": catalog["role"],
                "source": "synthetic_independent_character_stress",
                "syndrome_kind": None,
            })
    required_roles = list(config["candidate_rule"]["required_roles"])
    role_multipliers = {}
    for role in required_roles:
        values = [
            row["quantile"] for row in scenario_quantiles
            if row["simultaneous_role"] == role
        ]
        if not values:
            raise RuntimeError(f"simultaneous role has no calibration rows: {role}")
        role_multipliers[role] = float(max(values))
    if set(row["simultaneous_role"] for row in scenario_quantiles) != set(required_roles):
        raise RuntimeError("calibration produced an unregistered simultaneous role")
    if not all(np.isfinite(value) for value in role_multipliers.values()):
        raise RuntimeError("common simultaneous multiplier is non-finite")
    return {
        "draws_per_trajectory": int(draws),
        "role_multipliers": role_multipliers,
        "scenario_quantiles": scenario_quantiles,
        "trajectory_count": int(trajectories),
    }


def evaluate_exact_row(
        scenario, requested_shift, trials, trajectories, draws, multiplier,
        config, config_sha, stage, index):
    shifted, selected, base_means, shifted_means = tilted_probabilities(
        scenario["probabilities"], scenario["signs"], requested_shift,
    )
    base_logical = scenario["probabilities"] @ scenario["logical_signs"]
    shifted_logical = shifted @ scenario["logical_signs"]
    seed = derive_seed(
        config["seed_namespace"], config_sha, stage, "exact", trajectories,
        draws, index, requested_shift,
    )
    rng = np.random.default_rng(seed)
    left = sample_chain_means(
        rng, scenario["probabilities"], scenario["signs"],
        trials, trajectories, draws,
    )
    right = sample_chain_means(
        rng, shifted, scenario["signs"], trials, trajectories, draws,
    )
    signed, se = signed_differences_and_se(left, right)
    true_signed = base_means - shifted_means
    accumulator = _new_trial_accumulator(trials)
    _update_trial_accumulator(
        accumulator, signed, se, true_signed, multiplier, config,
    )
    statistics = _finish_trial_accumulator(accumulator, config)
    distribution = _distribution_fields(
        base_means, shifted_means, base_logical, shifted_logical,
    )
    if scenario["catalog"] == "logical" and (
        abs(
            distribution["true_character_purity_left"]
            - distribution["true_q_top_left"]
        ) > 1e-13
        or abs(
            distribution["true_character_purity_right"]
            - distribution["true_q_top_right"]
        ) > 1e-13
    ):
        raise RuntimeError("complete logical character purity is not q_top")
    return {
        **_rate_fields(statistics, config),
        **distribution,
        "catalog": scenario["catalog"],
        "catalog_size": scenario["catalog_size"],
        "draws_per_trajectory": int(draws),
        "model_id": scenario["model_id"],
        "p": scenario["p"],
        "physical_distribution": True,
        "q_top_accounting": (
            "exact_full_logical_catalog"
            if scenario["catalog"] == "logical"
            else "induced_from_retained_conditional_logical_signs"
        ),
        "requested_shift": float(requested_shift),
        "requirement_categories": scenario["requirement_categories"],
        "scenario_index": int(index),
        "selected_character_index": int(selected),
        "seed": int(seed),
        "shift_pattern": "single_character_reweight",
        "simultaneous_multiplier": float(multiplier),
        "simultaneous_role": scenario["simultaneous_role"],
        "source": "exact_hgp_iid",
        "stage": stage,
        "syndrome_kind": scenario["syndrome_kind"],
        "trajectory_count": int(trajectories),
    }


def evaluate_synthetic_row(
        catalog, base_mean, requested_shift, pattern, trials, trajectories,
        draws, multiplier, config, config_sha, stage):
    size = int(catalog["size"])
    left_means = np.full(size, float(base_mean))
    right_means = left_means.copy()
    direction = -1.0 if base_mean >= 0.0 else 1.0
    if pattern == "single_character":
        right_means[0] += direction * float(requested_shift)
        selected = 0
    elif pattern == "distributed_all_characters":
        right_means += direction * float(requested_shift)
        selected = None
    else:
        raise RuntimeError("unknown synthetic shift pattern")
    if np.any(np.abs(right_means) > 1.0):
        raise RuntimeError("synthetic character mean left its algebraic range")
    true_signed = left_means - right_means
    seed = derive_seed(
        config["seed_namespace"], config_sha, stage, "synthetic", trajectories,
        draws, catalog["id"], base_mean, pattern, requested_shift,
    )
    rng = np.random.default_rng(seed)
    accumulator = _new_trial_accumulator(trials)
    batch_size = int(config["synthetic_multiplicity_stress"]["character_batch_size"])
    for start in range(0, size, batch_size):
        stop = min(start + batch_size, size)
        left = synthetic_chain_means(
            rng, left_means[start:stop], trials, trajectories, draws,
        )
        right = synthetic_chain_means(
            rng, right_means[start:stop], trials, trajectories, draws,
        )
        signed, se = signed_differences_and_se(left, right)
        _update_trial_accumulator(
            accumulator, signed, se, true_signed[start:stop], multiplier, config,
        )
    statistics = _finish_trial_accumulator(accumulator, config)
    left_purity = float(np.mean(left_means ** 2))
    right_purity = float(np.mean(right_means ** 2))
    true_shift_summary = vector_summary(true_signed)
    categories = ["synthetic_all_registered_sizes", *catalog["roles"]]
    return {
        **_rate_fields(statistics, config),
        "base_character_mean": float(base_mean),
        "catalog": catalog["id"],
        "catalog_size": size,
        "draws_per_trajectory": int(draws),
        "model_id": None,
        "p": None,
        "physical_distribution": False,
        "q_top_accounting": "not_defined_for_nonphysical_synthetic_stress",
        "requested_shift": float(requested_shift),
        "requirement_categories": categories,
        "scenario_index": None,
        "selected_character_index": selected,
        "seed": int(seed),
        "shift_pattern": pattern,
        "shifted_character_mean_maximum": float(np.max(right_means)),
        "shifted_character_mean_minimum": float(np.min(right_means)),
        "simultaneous_multiplier": float(multiplier),
        "simultaneous_role": catalog["role"],
        "source": "synthetic_independent_character_stress",
        "stage": stage,
        "syndrome_kind": None,
        "trajectory_count": int(trajectories),
        "true_character_purity_delta": abs(left_purity - right_purity),
        "true_character_purity_left": left_purity,
        "true_character_purity_right": right_purity,
        "true_character_shift_summary": true_shift_summary,
        "true_character_shift_vector_sha256": vector_sha256(true_signed),
        "true_max_abs_character_shift": float(np.max(np.abs(true_signed))),
        "true_q_top_delta": None,
        "true_q_top_left": None,
        "true_q_top_right": None,
    }


def evaluate_point(
        config, config_sha, scenarios, trajectories, draws, role_multipliers,
        stage):
    trials = int(config["replications"][f"{stage}_trials"])
    rows = []
    for index, scenario in enumerate(scenarios):
        for requested_shift in config["injected_character_shifts"]:
            rows.append(evaluate_exact_row(
                scenario, requested_shift, trials, trajectories, draws,
                role_multipliers[scenario["simultaneous_role"]], config,
                config_sha, stage, index,
            ))
    synthetic = config["synthetic_multiplicity_stress"]
    for catalog in synthetic["catalogs"]:
        for base_mean in synthetic["character_means"]:
            for requested_shift in synthetic["single_character_shifts"]:
                rows.append(evaluate_synthetic_row(
                    catalog, base_mean, requested_shift, "single_character",
                    trials, trajectories, draws,
                    role_multipliers[catalog["role"]], config, config_sha, stage,
                ))
            for requested_shift in synthetic["distributed_character_shifts"]:
                rows.append(evaluate_synthetic_row(
                    catalog, base_mean, requested_shift,
                    "distributed_all_characters", trials, trajectories, draws,
                    role_multipliers[catalog["role"]], config, config_sha, stage,
                ))
    return rows


def summarize_operating_point(
        rows, config, trajectories, draws, role_multipliers, stage):
    required = config["candidate_rule"]["required_categories"]
    tolerance = float(config["candidate_rule"]["tolerance"])
    minimum_bad = tolerance + float(config["candidate_rule"]["minimum_bad_margin"])
    category_summaries = {}
    for category in required:
        subset = [row for row in rows if category in row["requirement_categories"]]
        null = [
            row for row in subset
            if row["true_max_abs_character_shift"] <= 1e-14
        ]
        bad = [
            row for row in subset
            if row["true_max_abs_character_shift"] >= minimum_bad - 1e-12
        ]
        if not subset or not null or not bad:
            raise RuntimeError(f"operating-point category is incomplete: {category}")
        null_lower = min(row["candidate_pass_wilson_lower"] for row in null)
        bad_lower = min(row["candidate_fail_wilson_lower"] for row in bad)
        coverage_lower = min(row["coverage_wilson_lower"] for row in subset)
        category_summaries[category] = {
            "eligible": bool(
                null_lower >= config["candidate_rule"]["null_pass_min"]
                and bad_lower >= config["candidate_rule"]["fail_power_min"]
                and coverage_lower >= config["candidate_rule"]["empirical_coverage_min"]
            ),
            "minimum_bad_shift_fail_wilson_lower": float(bad_lower),
            "minimum_coverage_wilson_lower": float(coverage_lower),
            "minimum_null_pass_wilson_lower": float(null_lower),
            "row_count": len(subset),
        }
    witnesses = [
        row for row in rows
        if row["shift_pattern"] == "distributed_all_characters"
        and row["true_max_abs_character_shift"] <= tolerance + 1e-12
        and row["true_character_purity_delta"] > tolerance
    ]
    distributed_witness = bool(witnesses)
    for row in rows:
        expected = role_multipliers[row["simultaneous_role"]]
        if row["simultaneous_multiplier"] != expected:
            raise RuntimeError("row did not use its frozen role multiplier")
    eligible = all(row["eligible"] for row in category_summaries.values())
    eligible = eligible and distributed_witness
    return {
        "category_summaries": category_summaries,
        "cost_independent_draws": int(trajectories) * int(draws),
        "distributed_bias_witness": distributed_witness,
        "draws_per_trajectory": int(draws),
        "eligible": bool(eligible),
        "role_multipliers": {
            role: float(value) for role, value in sorted(role_multipliers.items())
        },
        "stage": stage,
        "trajectory_count": int(trajectories),
    }


def ordered_operating_points(config):
    points = [
        (int(point["trajectory_count"]), int(point["draws_per_trajectory"]))
        for point in config["common_operating_grid"]["points"]
    ]
    if len(set(points)) != len(points):
        raise RuntimeError("common operating grid contains duplicate points")
    return sorted(points, key=lambda row: (row[0] * row[1], row[0], row[1]))


def first_eligible_operating_point(points):
    for point in points:
        if point["eligible"]:
            return point
    return None


def confirmation_matches_selection(selected, confirmation):
    if selected is None or confirmation is None or not confirmation["eligible"]:
        return False
    identity_fields = (
        "trajectory_count", "draws_per_trajectory", "role_multipliers",
    )
    return all(selected[field] == confirmation[field] for field in identity_fields)


def main():
    if OUTPUT_PATH.exists():
        raise RuntimeError("calibration report already exists")
    config = load_json_strict(CONFIG_PATH)
    provenance = verify_committed_clean_source(config)
    config_sha = sha256_file(CONFIG_PATH)
    scenarios = build_exact_scenarios(config)

    calibration_rows = []
    selection_rows = []
    selection_points = []
    selected = None
    for trajectories, draws in ordered_operating_points(config):
        calibration = common_multipliers(
            config, config_sha, scenarios, trajectories, draws,
        )
        calibration_rows.append(calibration)
        rows = evaluate_point(
            config, config_sha, scenarios, trajectories, draws,
            calibration["role_multipliers"], "selection",
        )
        selection_rows.extend(rows)
        point = summarize_operating_point(
            rows, config, trajectories, draws,
            calibration["role_multipliers"], "selection",
        )
        selection_points.append(point)
        if point["eligible"]:
            selected = point
            break

    confirmation_rows = []
    confirmation_point = None
    if selected is not None:
        confirmation_rows = evaluate_point(
            config, config_sha, scenarios, selected["trajectory_count"],
            selected["draws_per_trajectory"], selected["role_multipliers"],
            "confirmation",
        )
        confirmation_point = summarize_operating_point(
            confirmation_rows, config, selected["trajectory_count"],
            selected["draws_per_trajectory"], selected["role_multipliers"],
            "confirmation",
        )

    if selected != first_eligible_operating_point(selection_points):
        raise RuntimeError("selection did not freeze the first eligible cost point")
    passed = confirmation_matches_selection(selected, confirmation_point)
    tolerance = float(config["candidate_rule"]["tolerance"])
    core = {
        "authority": config["authority"],
        "calibration_points": calibration_rows,
        "config_sha256": config_sha,
        "confirmation_point": confirmation_point,
        "confirmation_rows": confirmation_rows,
        "deterministic_interpretation": {
            "frozen_catalog_character_purity_bound": 2.0 * tolerance,
            "full_q_top_bound_only_when_logical_catalog_is_complete": True,
            "maximum_character_mean_tolerance": tolerance,
            "unobserved_characters_covered": False,
        },
        "exact_hgp_scenario_count": len(scenarios),
        "selected_common_operating_point": selected,
        "selection_points": selection_points,
        "selection_rows": selection_rows,
        "source_commit": provenance["source_commit"],
        "source_file_count": provenance["source_file_count"],
        "source_tree_sha256": provenance["source_tree_sha256"],
        "status": (
            "CHARACTER_GATE_COMMON_OPERATING_POINT_CONFIRMED"
            if passed else "CHARACTER_GATE_REDESIGN_REQUIRED"
        ),
        "version": config["version"],
    }
    core["report_sha256"] = sha256_bytes(canonical(core).encode("ascii"))
    with OUTPUT_PATH.open("x", encoding="ascii") as handle:
        handle.write(canonical(core) + "\n")
    print(json.dumps({
        "report_sha256": core["report_sha256"],
        "selected_common_operating_point": selected,
        "status": core["status"],
    }, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
