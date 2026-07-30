"""Independent raw-decision and provenance audit for validation 066."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from statistics import NormalDist
import subprocess

import numpy as np


ROOT = Path(__file__).resolve().parent
EXP102_ROOT = ROOT.parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[5]
CONFIG_PATH = ROOT / "delivery_gate_config.json"
REPORT_PATH = ROOT / "delivery_gate_report.json"
OUTPUT_PATH = ROOT / "independent_audit.json"
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


REPLAY_RECEIPT_SCHEMA = "exp102.q0_delivery_gate.replay_receipt.le.v1"


def digest_named_arrays(domain, arrays):
    digest = hashlib.sha256()
    digest.update((REPLAY_RECEIPT_SCHEMA + "\0" + domain + "\0").encode("ascii"))
    for name, (values, dtype) in sorted(arrays.items()):
        array = np.ascontiguousarray(np.asarray(values, dtype=np.dtype(dtype)))
        header = canonical({
            "dtype": np.dtype(dtype).str,
            "name": name,
            "shape": list(array.shape),
        }).encode("ascii")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def replay_receipt(labels, left_counts, right_counts, metrics):
    return {
        "all_trial_metrics_sha256": digest_named_arrays(
            "all_trial_metrics",
            {name: (values, "<f8") for name, values in metrics.items()},
        ),
        "histogram_counts_sha256": digest_named_arrays("histogram_counts", {
            "labels": (labels, "<u8"),
            "left_counts": (left_counts, "<i8"),
            "right_counts": (right_counts, "<i8"),
        }),
        "schema": REPLAY_RECEIPT_SCHEMA,
    }


def verify_self_hash(payload, field):
    expected = payload[field]
    unsigned = dict(payload)
    unsigned.pop(field)
    if sha256_bytes(canonical(unsigned).encode("ascii")) != expected:
        raise RuntimeError(f"self hash changed: {field}")


def validate_config_contract(config):
    verify_self_hash(config, "config_self_sha256")
    expected_rule = {
        "bad_d2_minimum": 0.06,
        "bad_false_pass_maximum": 0.02,
        "bad_q_top_delta_minimum": 0.06,
        "binomial_lower_confidence": 0.95,
        "fail_familywise_confidence": 0.95,
        "fail_multiplicity_scope": (
            "all_grid_selection_and_potential_confirmation_fail_hypotheses"
        ),
        "d2_upper_tolerance": 0.04,
        "fail_power_minimum": 0.95,
        "good_alternative_pass_minimum": 0.95,
        "interval_coverage_minimum": 0.98,
        "known_blind_pass_minimum": 0.95,
        "null_pass_minimum": 0.95,
        "q_top_delta_tolerance": 0.04,
        "states": ["PASS", "FAIL", "INCONCLUSIVE"],
    }
    if config["decision_rule"] != expected_rule:
        raise RuntimeError("decision-rule contract changed")
    if config["replications"] != {
            "calibration_trials": 600, "confirmation_trials": 300,
            "selection_trials": 300}:
        raise RuntimeError("replication contract changed")
    if config["uncertainty_calibration"] != {
            "higher_quantile": 0.995,
            "joint_scalar_estimands": ["signed_q_top_delta", "d2_norm"],
            "multiplier_shared_across_scenarios": True}:
        raise RuntimeError("uncertainty contract changed")
    if config["fail_multiplicity_contract"] != {
            "expected_hypotheses_per_stage_point": 139,
            "expected_total_hypothesis_count": 1390,
            "strict_familywise_guarantee": False,
            "wilson_role": "OPERATIONAL_ONLY"}:
        raise RuntimeError("FAIL multiplicity contract changed")
    if config["report_contract"] != {
            "maximum_bytes": 10485760,
            "persistent_per_trial_arrays": False}:
        raise RuntimeError("compact report contract changed")
    expected_rng = {
        "bit_generator": "PCG64",
        "generator_constructor": "numpy.random.default_rng",
        "numpy_version": "2.4.1",
        "scope": "same_environment_bit_replay",
    }
    if config["rng_replay_contract"] != expected_rng:
        raise RuntimeError("RNG replay contract changed")
    if np.__version__ != expected_rng["numpy_version"]:
        raise RuntimeError("NumPy version cannot satisfy frozen bit replay")
    if type(np.random.default_rng(0).bit_generator).__name__ != expected_rng[
            "bit_generator"]:
        raise RuntimeError("NumPy bit generator cannot satisfy frozen replay")


def derive_seed(namespace, *parts):
    payload = canonical([namespace, *parts]).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _exp102_path(relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError("configured path escapes exp102 root")
    unresolved = EXP102_ROOT / relative
    if unresolved.is_symlink():
        raise RuntimeError(f"frozen input may not be a symlink: {unresolved}")
    path = unresolved.resolve()
    try:
        path.relative_to(EXP102_ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError("configured path escapes exp102 root") from exc
    return path


def _git(args, *, text=True):
    return subprocess.run(
        ["git", *args], cwd=PROJECT_ROOT, check=True,
        capture_output=True, text=text,
    ).stdout


def reject_validation_bytecode(root=ROOT):
    offenders = sorted(
        path.relative_to(root).as_posix()
        for path in Path(root).rglob("*")
        if path.name == "__pycache__" or path.suffix in {".pyc", ".pyo"}
    )
    if offenders:
        raise RuntimeError(f"validation source contains bytecode: {offenders[0]}")


def verify_clean_audit_worktree():
    allowed = {
        REPORT_PATH.relative_to(PROJECT_ROOT).as_posix(),
        OUTPUT_PATH.relative_to(PROJECT_ROOT).as_posix(),
    }
    dirty = []
    for line in _git(["status", "--porcelain=v1", "--untracked-files=all"]).splitlines():
        if line.startswith("?? ") and line[3:] in allowed:
            continue
        dirty.append(line)
    if dirty:
        raise RuntimeError(f"audit found source-worktree changes: {dirty[0]}")


def verify_source_provenance(config, report):
    verify_self_hash(config, "config_self_sha256")
    verify_self_hash(report, "report_sha256")
    reject_validation_bytecode()
    verify_clean_audit_worktree()
    repository = Path(_git(["rev-parse", "--show-toplevel"]).strip()).resolve()
    if repository != PROJECT_ROOT.resolve():
        raise RuntimeError("validation is outside its bound repository")
    source_commit = _git(["rev-parse", "HEAD"]).strip()
    if COMMIT_RE.fullmatch(source_commit) is None or source_commit != report["source_commit"]:
        raise RuntimeError("report does not bind the current source commit")
    config_sha = sha256_file(CONFIG_PATH)
    if report["config_sha256"] != config_sha:
        raise RuntimeError("report config hash changed")

    artifacts = [(CONFIG_PATH.resolve(), config_sha)]
    for name, spec in sorted(config["source_artifacts"].items()):
        expected = spec["sha256"]
        if re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise RuntimeError(f"invalid source SHA for {name}")
        artifacts.append((_exp102_path(spec["path"]), expected))
    if len({path for path, _ in artifacts}) != len(artifacts):
        raise RuntimeError("duplicate source artifact")
    rows = []
    for path, expected in artifacts:
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"source artifact changed: {path}")
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        _git(["ls-files", "--error-unmatch", "--", relative])
        committed = _git(["show", f"{source_commit}:{relative}"], text=False)
        if sha256_bytes(committed) != expected:
            raise RuntimeError(f"source commit bytes changed: {relative}")
        rows.append([relative, expected])
    rows.sort()
    if report["source_file_count"] != len(rows):
        raise RuntimeError("report source-file count changed")
    if report["source_tree_sha256"] != sha256_bytes(
            canonical(rows).encode("ascii")):
        raise RuntimeError("report source-tree identity changed")


def verify_bound_validation_062(config):
    bound = config["bound_validation_062"]
    report_spec = bound["report"]
    audit_spec = bound["audit"]
    report_path = _exp102_path(report_spec["path"])
    audit_path = _exp102_path(audit_spec["path"])
    if sha256_file(report_path) != report_spec["file_sha256"]:
        raise RuntimeError("bound validation-062 report bytes changed")
    if sha256_file(audit_path) != audit_spec["file_sha256"]:
        raise RuntimeError("bound validation-062 audit bytes changed")
    report = load_json_strict(report_path)
    audit = load_json_strict(audit_path)
    verify_self_hash(report, "report_sha256")
    verify_self_hash(audit, "audit_sha256")
    if report["report_sha256"] != report_spec["report_sha256"]:
        raise RuntimeError("validation-062 report identity changed")
    if report["status"] != report_spec["required_status"]:
        raise RuntimeError("validation-062 report status changed")
    if report["source_commit"] != report_spec["required_source_commit"]:
        raise RuntimeError("validation-062 source identity changed")
    if audit["audit_sha256"] != audit_spec["audit_sha256"]:
        raise RuntimeError("validation-062 audit identity changed")
    if audit["status"] != audit_spec["required_status"]:
        raise RuntimeError("validation-062 audit status changed")
    if audit["report_sha256"] != report["report_sha256"]:
        raise RuntimeError("validation-062 audit/report binding changed")
    if audit["source_commit"] != report["source_commit"]:
        raise RuntimeError("validation-062 audit/report source commit changed")
    return report


def labels_hex(labels):
    return [f"0x{int(value):016x}" for value in np.asarray(labels, dtype=np.uint64)]


def normalize_collision(collision, k):
    size = 1 << int(k)
    collision = np.asarray(collision, dtype=np.float64)
    return collision + (collision - 1.0) / (size - 1)


def normalize_d2(raw_d2, k):
    size = 1 << int(k)
    raw_d2 = np.asarray(raw_d2, dtype=np.float64)
    return raw_d2 + raw_d2 / (size - 1)


def character_means(labels, probabilities, masks):
    labels = np.asarray(labels, dtype=np.uint64)
    masks = np.asarray(masks, dtype=np.uint64)
    parity = np.bitwise_count(labels[:, None] & masks[None, :]) & np.uint8(1)
    return np.asarray(probabilities, dtype=np.float64) @ (
        1.0 - 2.0 * parity.astype(np.float64)
    )


def inverse_walsh(nonzero_means):
    nonzero_means = np.asarray(nonzero_means, dtype=np.float64)
    size = nonzero_means.size + 1
    if size <= 1 or size & (size - 1):
        raise RuntimeError("complete character catalog size changed")
    k = size.bit_length() - 1
    coefficients = np.concatenate((np.ones(1), nonzero_means))
    probabilities = np.empty(size, dtype=np.float64)
    for label in range(size):
        total = 0.0
        for mask, mean in enumerate(coefficients):
            sign = -1.0 if ((mask & label).bit_count() & 1) else 1.0
            total += float(mean) * sign
        probabilities[label] = total / size
    if np.any(probabilities < 0.0) or abs(float(probabilities.sum()) - 1.0) > 2e-13:
        raise RuntimeError("inverse-Walsh probabilities changed")
    labels = np.arange(size, dtype=np.uint64)
    recovered = character_means(labels, probabilities, np.arange(1, size, dtype=np.uint64))
    if not np.allclose(recovered, nonzero_means, atol=2e-13, rtol=0.0):
        raise RuntimeError("inverse-Walsh round trip changed")
    return labels, probabilities, k


def q_top(probabilities, k):
    return float(normalize_collision(np.dot(probabilities, probabilities), k))


def shift_q_top(labels, probabilities, k, requested_delta):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    base = q_top(probabilities, k)
    if float(requested_delta) == 0.0:
        return labels.copy(), probabilities.copy()
    if base + float(requested_delta) <= 1.0 - 1e-13:
        target = base + float(requested_delta)
        endpoint = np.zeros(probabilities.size, dtype=np.float64)
        endpoint[int(np.argmax(probabilities))] = 1.0
        low, high = 0.0, 1.0
        for _ in range(100):
            middle = (low + high) / 2.0
            candidate = (1.0 - middle) * probabilities + middle * endpoint
            if q_top(candidate, k) < target:
                low = middle
            else:
                high = middle
        shifted = (1.0 - high) * probabilities + high * endpoint
    else:
        target = base - float(requested_delta)
        size = 1 << int(k)
        if labels.size != size:
            raise RuntimeError("downward exact stress lost complete support")
        alpha = 1.0 - math.sqrt(target / base)
        shifted = (1.0 - alpha) * probabilities + alpha / size
    if abs(abs(q_top(shifted, k) - base) - float(requested_delta)) > 2e-13:
        raise RuntimeError("q_top stress truth changed")
    return labels.copy(), shifted


def aligned_truth(labels_a, probabilities_a, labels_b, probabilities_b, k):
    union = sorted({int(value) for value in labels_a} | {int(value) for value in labels_b})
    positions = {value: index for index, value in enumerate(union)}
    left = np.zeros(len(union), dtype=np.float64)
    right = np.zeros(len(union), dtype=np.float64)
    for label, probability in zip(labels_a, probabilities_a):
        left[positions[int(label)]] = float(probability)
    for label, probability in zip(labels_b, probabilities_b):
        right[positions[int(label)]] = float(probability)
    q_left = q_top(left, k)
    q_right = q_top(right, k)
    metrics = {
        "d2_norm": float(normalize_d2(np.dot(left - right, left - right), k)),
        "q_top_delta_abs": abs(q_left - q_right),
        "q_top_delta_signed": q_left - q_right,
        "q_top_left": q_left,
        "q_top_right": q_right,
    }
    return metrics


def diagnostic_masks(k, config):
    diagnostic = config["character_diagnostic"]
    if int(k) <= int(diagnostic["include_all_nonzero_when_k_at_most"]):
        return np.arange(1, 1 << int(k), dtype=np.uint64)
    values = [1 << index for index in range(int(k))]
    seen = set(values)
    target = len(values) + int(diagnostic["nonbasis_mask_count"])
    counter = 0
    while len(values) < target:
        payload = canonical([
            config["seed_namespace"], "character-diagnostic-mask", int(k), counter,
        ]).encode("ascii")
        candidate = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        candidate &= (1 << int(k)) - 1
        counter += 1
        if candidate and candidate not in seen:
            values.append(candidate)
            seen.add(candidate)
    return np.asarray([np.uint64(value) for value in values], dtype=np.uint64)


def sparse_profile(k, requested_q_top, support_size):
    support_size = int(support_size)
    size = 1 << int(k)
    purity = 1.0 / size + (1.0 - 1.0 / size) * float(requested_q_top)
    major = (
        1.0 + math.sqrt((support_size - 1.0) * (support_size * purity - 1.0))
    ) / support_size
    probabilities = np.full(
        support_size, (1.0 - major) / (support_size - 1), dtype=np.float64,
    )
    probabilities[0] = major
    labels = np.arange(support_size, dtype=np.uint64)
    if int(k) == 64:
        labels[-1] = np.uint64(1 << 63)
    return labels, probabilities


def controlled_d2(k, requested_d2, support_size=64):
    labels = np.arange(int(support_size), dtype=np.uint64)
    if int(k) == 64:
        labels[-1] = np.uint64(1 << 63)
    size = 1 << int(k)
    difference = math.sqrt((float(requested_d2) - float(requested_d2) / size) / 2.0)
    left = np.full(int(support_size), 0.6 / (int(support_size) - 2), dtype=np.float64)
    left[0], left[1] = 0.2 + difference / 2.0, 0.2 - difference / 2.0
    right = left.copy()
    right[0], right[1] = left[1], left[0]
    return labels, left, labels.copy(), right


def disjoint_pair(k, support_size):
    support_size = int(support_size)
    left = np.arange(support_size, dtype=np.uint64)
    right = (
        np.uint64(1 << 63) | np.arange(support_size, dtype=np.uint64)
        if int(k) == 64
        else np.arange(support_size, 2 * support_size, dtype=np.uint64)
    )
    probabilities = np.full(support_size, 1.0 / support_size, dtype=np.float64)
    return left, probabilities, right, probabilities.copy()


def _token(value):
    return f"{float(value):.2f}".replace(".", "p")


def public_scenario(
        scenario_id, source, classification, k, labels_left, probabilities_left,
        labels_right, probabilities_right, config, *, observation_mode="iid_target",
        requested_q_top_delta=None):
    masks = diagnostic_masks(k, config)
    truth = aligned_truth(
        labels_left, probabilities_left, labels_right, probabilities_right, k,
    )
    return {
        "character_diagnostic_mask_count": int(masks.size),
        "character_diagnostic_mask_sha256": sha256_bytes(
            canonical(labels_hex(masks)).encode("ascii")
        ),
        "classification": classification,
        "id": scenario_id,
        "k": int(k),
        "labels_left": labels_hex(labels_left),
        "labels_right": labels_hex(labels_right),
        "observation_mode": observation_mode,
        "probabilities_left": np.asarray(probabilities_left).tolist(),
        "probabilities_right": np.asarray(probabilities_right).tolist(),
        "requested_q_top_delta": requested_q_top_delta,
        "source": source,
        "truth": {name: float(value) for name, value in sorted(truth.items())},
        "uint64_bit63_exercised": bool(
            int(k) == 64 and any(
                int(value) & (1 << 63) for value in np.concatenate((
                    labels_left, labels_right, masks,
                ))
            )
        ),
    }


def expected_scenario_registry(config, bound_report):
    result = []
    row_filter = config["bound_validation_062"]["source_row_filter"]
    rows = [
        row for row in bound_report["selection_rows"]
        if all(row.get(name) == value for name, value in row_filter.items())
    ]
    if len(rows) != int(config["bound_validation_062"]["unique_distribution_count"]):
        raise RuntimeError("bound exact distribution count changed")
    for row in rows:
        labels, base, k = inverse_walsh(row["base_logical_means"])
        if abs(q_top(base, k) - float(row["true_q_top_left"])) > 2e-13:
            raise RuntimeError("bound exact q_top changed")
        for delta in config["sparse_label_stress"]["q_top_deltas"]:
            shifted_labels, shifted = shift_q_top(labels, base, k, delta)
            classification = {
                0.0: "null", 0.02: "good_q_top", 0.04: "boundary_q_top",
                0.06: "bad_q_top",
            }[float(delta)]
            result.append(public_scenario(
                f"exact_{row['model_id']}_{row['syndrome_kind']}_"
                f"p{_token(row['p'])}_qdelta_{_token(delta)}",
                "validation_062_inverse_walsh", classification, k,
                labels, base, shifted_labels, shifted, config,
                requested_q_top_delta=float(delta),
            ))
    sparse = config["sparse_label_stress"]
    for k in sparse["dimensions"]:
        for base_q_top in sparse["q_top_bases"]:
            labels, base = sparse_profile(
                k, base_q_top, sparse["sparse_q_top_support_size"],
            )
            for delta in sparse["q_top_deltas"]:
                shifted_labels, shifted = shift_q_top(labels, base, k, delta)
                classification = {
                    0.0: "null", 0.02: "good_q_top", 0.04: "boundary_q_top",
                    0.06: "bad_q_top",
                }[float(delta)]
                result.append(public_scenario(
                    f"sparse_k{k}_base_{_token(base_q_top)}_qdelta_{_token(delta)}",
                    "outcome_blind_sparse_label_distribution", classification, k,
                    labels, base, shifted_labels, shifted, config,
                    requested_q_top_delta=float(delta),
                ))
        for d2_delta in sparse["d2_deltas"]:
            left, p_left, right, p_right = controlled_d2(k, d2_delta)
            classification = {
                0.0: "d2_null", 0.02: "good_d2", 0.04: "boundary_d2",
                0.06: "bad_d2",
            }[float(d2_delta)]
            result.append(public_scenario(
                f"sparse_k{k}_same_purity_d2_{_token(d2_delta)}",
                "outcome_blind_sparse_label_distribution", classification, k,
                left, p_left, right, p_right, config,
            ))
        left, p_left, right, p_right = disjoint_pair(
            k, sparse["same_purity_disjoint_support_size"],
        )
        result.append(public_scenario(
            f"sparse_k{k}_same_purity_disjoint_d2_stress",
            "outcome_blind_sparse_label_distribution", "bad_d2", k,
            left, p_left, right, p_right, config,
        ))
    left, p_left, right, p_right = disjoint_pair(
        64, sparse["same_purity_disjoint_support_size"],
    )
    result.append(public_scenario(
        "control_k64_common_freeze", "known_transport_blind_control",
        "known_blind", 64, left, p_left, right, p_right, config,
        observation_mode="common_freeze",
    ))
    result.append(public_scenario(
        "control_k64_distinct_freeze_same_set",
        "known_transport_blind_control", "known_blind", 64,
        left, p_left, right, p_right, config,
        observation_mode="distinct_freeze_same_set",
    ))
    if len({row["id"] for row in result}) != len(result):
        raise RuntimeError("expected scenario registry contains duplicates")
    return result


def ordered_points(config):
    points = [
        (int(row["trajectory_count"]), int(row["draws_per_trajectory"]))
        for row in config["common_operating_grid"]["points"]
    ]
    expected = sorted(points, key=lambda row: (row[0] * row[1], row[0], row[1]))
    if points != expected or len(points) != len(set(points)):
        raise RuntimeError("configured operating order changed")
    return points


def higher_quantile(values, probability):
    values = sorted(math.inf if value is None else float(value) for value in values)
    index = int(math.ceil(float(probability) * len(values)) - 1)
    selected = values[min(max(index, 0), len(values) - 1)]
    return None if not math.isfinite(selected) else float(selected)


def expected_seeds(config, config_sha, stage, trajectories, draws, scenario_id):
    common = (
        config["seed_namespace"], config_sha, stage, int(trajectories), int(draws),
        scenario_id,
    )
    return derive_seed(*common, "family_A"), derive_seed(*common, "family_B")


def assert_float(actual, expected, message, tolerance=2e-13):
    if not math.isfinite(float(actual)) or abs(float(actual) - float(expected)) > tolerance:
        raise RuntimeError(message)


def _aligned_sampling_probabilities(metadata):
    left_labels = [int(value, 16) for value in metadata["labels_left"]]
    right_labels = [int(value, 16) for value in metadata["labels_right"]]
    labels = sorted(set(left_labels) | set(right_labels))
    positions = {label: index for index, label in enumerate(labels)}
    left = np.zeros(len(labels), dtype=np.float64)
    right = np.zeros(len(labels), dtype=np.float64)
    for label, probability in zip(left_labels, metadata["probabilities_left"]):
        left[positions[label]] = float(probability)
    for label, probability in zip(right_labels, metadata["probabilities_right"]):
        right[positions[label]] = float(probability)
    return np.asarray([np.uint64(label) for label in labels]), left, right


def replay_histogram_counts(
        metadata, config, trials, trajectories, draws, seed_left, seed_right):
    mode = metadata["observation_mode"]
    if mode == "iid_target":
        labels, left_probability, right_probability = _aligned_sampling_probabilities(
            metadata,
        )
        left = np.random.default_rng(int(seed_left)).multinomial(
            int(draws), left_probability, size=(int(trials), int(trajectories)),
        )
        right = np.random.default_rng(int(seed_right)).multinomial(
            int(draws), right_probability, size=(int(trials), int(trajectories)),
        )
        return labels, left, right
    if mode == "common_freeze":
        label = int(config["sparse_label_stress"]["common_freeze_label"], 16)
        labels = np.asarray([np.uint64(label)], dtype=np.uint64)
        left = np.full(
            (int(trials), int(trajectories), 1), int(draws), dtype=np.int64,
        )
        return labels, left, left.copy()
    if mode == "distinct_freeze_same_set":
        values = list(range(int(trajectories)))
        values[-2] = 1 << 63
        values[-1] = (1 << 64) - 1
        labels = np.asarray([np.uint64(value) for value in values], dtype=np.uint64)
        left = np.zeros(
            (int(trials), int(trajectories), int(trajectories)), dtype=np.int64,
        )
        for trajectory in range(int(trajectories)):
            left[:, trajectory, trajectory] = int(draws)
        return labels, left, left.copy()
    raise RuntimeError("unknown replay observation mode")


def _replay_collision(histograms):
    count = histograms.shape[1]
    summed = np.sum(histograms, axis=1)
    all_pairs = np.einsum("ts,ts->t", summed, summed, optimize=True)
    self_pairs = np.einsum("tis,tis->t", histograms, histograms, optimize=True)
    return (all_pairs - self_pairs) / (count * (count - 1))


def _replay_metric_core(left, right, k):
    collision_left = _replay_collision(left)
    collision_right = _replay_collision(right)
    cross = np.einsum(
        "ts,ts->t", np.sum(left, axis=1), np.sum(right, axis=1), optimize=True,
    ) / (left.shape[1] * right.shape[1])
    q_left = normalize_collision(collision_left, k)
    q_right = normalize_collision(collision_right, k)
    d2 = normalize_d2(collision_left + collision_right - 2.0 * cross, k)
    return q_left, q_right, d2


def replay_trial_metrics(
        metadata, config, trials, trajectories, draws, seed_left, seed_right):
    labels, left_counts, right_counts = replay_histogram_counts(
        metadata, config, trials, trajectories, draws, seed_left, seed_right,
    )
    left = left_counts.astype(np.float64) / left_counts.sum(
        axis=2, keepdims=True, dtype=np.int64,
    )
    right = right_counts.astype(np.float64) / right_counts.sum(
        axis=2, keepdims=True, dtype=np.int64,
    )
    q_left, q_right, d2 = _replay_metric_core(left, right, metadata["k"])
    delete_q_left = []
    delete_q_right = []
    delete_d2_left = []
    delete_d2_right = []
    for omitted in range(int(trajectories)):
        q_a, q_b, value = _replay_metric_core(
            np.delete(left, omitted, axis=1), right, metadata["k"],
        )
        delete_q_left.append(q_a - q_b)
        delete_d2_left.append(value)
        q_a, q_b, value = _replay_metric_core(
            left, np.delete(right, omitted, axis=1), metadata["k"],
        )
        delete_q_right.append(q_a - q_b)
        delete_d2_right.append(value)
    delete_q_left = np.stack(delete_q_left, axis=1)
    delete_q_right = np.stack(delete_q_right, axis=1)
    delete_d2_left = np.stack(delete_d2_left, axis=1)
    delete_d2_right = np.stack(delete_d2_right, axis=1)
    masks = diagnostic_masks(metadata["k"], config)
    parity = np.bitwise_count(labels[:, None] & masks[None, :]) & np.uint8(1)
    signs = 1.0 - 2.0 * parity.astype(np.float64)
    character_delta = (left.mean(axis=1) - right.mean(axis=1)) @ signs
    metrics = {
        "character_max_abs_delta_diagnostic": np.max(
            np.abs(character_delta), axis=1,
        ),
        "delete_one_d2_left": delete_d2_left,
        "delete_one_d2_right": delete_d2_right,
        "delete_one_q_top_delta_left": delete_q_left,
        "delete_one_q_top_delta_right": delete_q_right,
        "d2_norm": d2,
        "d2_se": groupwise_se(delete_d2_left, delete_d2_right),
        "q_top_delta_abs": np.abs(q_left - q_right),
        "q_top_delta_se": groupwise_se(delete_q_left, delete_q_right),
        "q_top_delta_signed": q_left - q_right,
        "q_top_left": q_left,
        "q_top_right": q_right,
    }
    return metrics, replay_receipt(labels, left_counts, right_counts, metrics)


def verify_calibration_point(point, config, config_sha, scenario_registry):
    trajectories = int(point["trajectory_count"])
    draws = int(point["draws_per_trajectory"])
    trials = int(config["replications"]["calibration_trials"])
    quantile = float(config["uncertainty_calibration"]["higher_quantile"])
    if point["quantile_probability"] != quantile:
        raise RuntimeError("calibration quantile probability changed")
    expected_ids = {
        row["id"] for row in scenario_registry if row["classification"] != "known_blind"
    }
    metadata_by_id = {row["id"]: row for row in scenario_registry}
    rows = point["scenario_quantiles"]
    if {row["scenario_id"] for row in rows} != expected_ids or len(rows) != len(expected_ids):
        raise RuntimeError("calibration scenario registry changed")
    arrays = []
    for row in rows:
        if row["trial_count"] != trials:
            raise RuntimeError("calibration trial count changed")
        expected_left, expected_right = expected_seeds(
            config, config_sha, "calibration", trajectories, draws, row["scenario_id"],
        )
        if (row["seed_left"], row["seed_right"]) != (expected_left, expected_right):
            raise RuntimeError("calibration family seed changed")
        metadata = metadata_by_id[row["scenario_id"]]
        replayed, receipt = replay_trial_metrics(
            metadata, config, trials, trajectories, draws,
            expected_left, expected_right,
        )
        if row["replay_receipt"] != receipt:
            raise RuntimeError("calibration replay receipt changed")

        def standardized(estimate, truth, se):
            error = np.abs(estimate - float(truth))
            result = np.zeros(error.shape, dtype=np.float64)
            positive = se > 0.0
            result[positive] = error[positive] / se[positive]
            result[(~positive) & (error > 2e-13)] = np.inf
            return result

        q_error = standardized(
            replayed["q_top_delta_signed"],
            metadata["truth"]["q_top_delta_signed"],
            replayed["q_top_delta_se"],
        )
        d2_error = standardized(
            replayed["d2_norm"], metadata["truth"]["d2_norm"],
            replayed["d2_se"],
        )
        values = np.maximum(q_error, d2_error)
        expected_joint_digest = digest_named_arrays(
            "calibration_joint_standardized_errors", {"values": (values, "<f8")}
        )
        if row["joint_standardized_errors_sha256"] != expected_joint_digest:
            raise RuntimeError("calibration standardized-error digest changed")
        finite_values = [
            float(value) if math.isfinite(float(value)) else None for value in values
        ]
        if row["marginal_quantile_diagnostic"] != higher_quantile(
                finite_values, quantile):
            raise RuntimeError("marginal calibration quantile changed")
        arrays.append(values)
    outer = np.max(np.stack(arrays, axis=0), axis=0)
    expected_outer_digest = digest_named_arrays(
        "calibration_outer_max_standardized_errors", {"values": (outer, "<f8")}
    )
    if point["outer_max_standardized_errors_sha256"] != expected_outer_digest:
        raise RuntimeError("outer simultaneous calibration digest changed")
    expected_multiplier = higher_quantile(
        [float(value) if math.isfinite(float(value)) else None for value in outer],
        quantile,
    )
    if point["multiplier"] != expected_multiplier:
        raise RuntimeError("simultaneous calibration multiplier changed")
    if point["valid"] != (expected_multiplier is not None):
        raise RuntimeError("calibration validity changed")


def wilson_lower(successes, trials, confidence):
    z = NormalDist().inv_cdf(float(confidence))
    p = int(successes) / int(trials)
    denominator = 1.0 + z * z / int(trials)
    center = p + z * z / (2.0 * int(trials))
    radius = z * math.sqrt(
        p * (1.0 - p) / int(trials) + z * z / (4.0 * int(trials) ** 2)
    )
    return max(0.0, (center - radius) / denominator)


def wilson_upper(successes, trials, confidence):
    return 1.0 - wilson_lower(int(trials) - int(successes), trials, confidence)


def groupwise_se(left, right):
    variance = np.zeros(left.shape[0], dtype=np.float64)
    for values in (left, right):
        centered = values - values.mean(axis=1, keepdims=True)
        variance += (values.shape[1] - 1.0) / values.shape[1] * np.sum(
            centered * centered, axis=1,
        )
    return np.sqrt(variance)


def decision_arrays(raw, multiplier, config):
    rule = config["decision_rule"]
    q_delta = np.asarray(raw["q_top_delta_abs"], dtype=np.float64)
    q_se = np.asarray(raw["q_top_delta_se"], dtype=np.float64)
    d2 = np.asarray(raw["d2_norm"], dtype=np.float64)
    d2_se = np.asarray(raw["d2_se"], dtype=np.float64)
    q_pass = q_delta + float(multiplier) * q_se <= rule["q_top_delta_tolerance"]
    q_fail = np.maximum(q_delta - float(multiplier) * q_se, 0.0) > rule[
        "q_top_delta_tolerance"]
    d2_pass = np.maximum(d2, 0.0) + float(multiplier) * d2_se <= rule[
        "d2_upper_tolerance"]
    d2_fail = np.maximum(d2 - float(multiplier) * d2_se, 0.0) > rule[
        "d2_upper_tolerance"]
    passed = q_pass & d2_pass
    failed = q_fail | d2_fail
    return {
        "d2_fail": d2_fail,
        "d2_inconclusive": ~(d2_pass | d2_fail),
        "d2_pass": d2_pass,
        "fail": failed,
        "inconclusive": ~(passed | failed),
        "pass": passed,
        "q_top_fail": q_fail,
        "q_top_inconclusive": ~(q_pass | q_fail),
        "q_top_pass": q_pass,
    }


def fail_hypotheses_per_stage_point(scenarios):
    classifications = [row["classification"] for row in scenarios]
    boundary = {"boundary_q_top", "boundary_d2"}
    base = [name for name in classifications if name not in boundary]
    bad = [name for name in classifications if name in {"bad_q_top", "bad_d2"}]
    required = {
        "null", "d2_null", "good_q_top", "good_d2", "bad_q_top", "bad_d2",
        "known_blind", "boundary_q_top", "boundary_d2",
    }
    if set(classifications) != required or not base or not bad:
        raise RuntimeError("rowwise FAIL claim registry changed")
    count = len(base) + len(bad) + 1
    if count != 139:
        raise RuntimeError("FAIL hypotheses per stage point changed")
    return count


def fail_hypothesis_count(config, scenarios):
    count = len(ordered_points(config)) * 2 * fail_hypotheses_per_stage_point(
        scenarios
    )
    if count != 1390:
        raise RuntimeError("full-protocol FAIL hypothesis count changed")
    return count


def fail_adjusted_confidence(config, scenarios):
    family = float(config["decision_rule"]["fail_familywise_confidence"])
    return 1.0 - (1.0 - family) / fail_hypothesis_count(config, scenarios)


def verify_rate(row, values, prefix, pass_confidence, fail_confidence):
    count = int(np.count_nonzero(values))
    trials = int(values.size)
    expected = {
        f"{prefix}_count": count,
        f"{prefix}_rate": float(count / trials),
        f"{prefix}_wilson_lower": float(
            wilson_lower(count, trials, pass_confidence)
        ),
        f"{prefix}_wilson_upper": float(
            wilson_upper(count, trials, pass_confidence)
        ),
        f"{prefix}_fail_adjusted_wilson_lower": float(
            wilson_lower(count, trials, fail_confidence)
        ),
        f"{prefix}_fail_adjusted_wilson_upper": float(
            wilson_upper(count, trials, fail_confidence)
        ),
    }
    for name, value in expected.items():
        if isinstance(value, int):
            if row[name] != value:
                raise RuntimeError(f"stored rate count changed: {name}")
        else:
            assert_float(row[name], value, f"stored rate changed: {name}")


def verify_row(
        row, config, config_sha, metadata, stage, trajectories, draws, multiplier,
        scenario_registry):
    trials = int(config["replications"][f"{stage}_trials"])
    if row["scenario_id"] != metadata["id"] or row["classification"] != metadata[
            "classification"]:
        raise RuntimeError("row scenario identity changed")
    if row["stage"] != stage or row["evaluation_trials"] != trials:
        raise RuntimeError("row stage or trial count changed")
    if (row["trajectory_count"], row["draws_per_trajectory"], row["multiplier"]) != (
            trajectories, draws, multiplier):
        raise RuntimeError("row operating-point identity changed")
    expected_left, expected_right = expected_seeds(
        config, config_sha, stage, trajectories, draws, metadata["id"],
    )
    if (row["seed_left"], row["seed_right"]) != (expected_left, expected_right):
        raise RuntimeError("fresh family seed changed")
    if row["character_diagnostic_participates_in_decision"]:
        raise RuntimeError("character diagnostic acquired decision authority")
    expected_interpretation = (
        "EXPECTED_KNOWN_BLIND"
        if metadata["classification"] == "known_blind"
        else "IID_DELIVERY_GATE_CALIBRATION"
    )
    if row["expected_interpretation"] != expected_interpretation:
        raise RuntimeError("known-blind interpretation changed")
    assert_float(row["true_d2_norm"], metadata["truth"]["d2_norm"], "row D2 truth changed")
    assert_float(
        row["true_q_top_delta_signed"], metadata["truth"]["q_top_delta_signed"],
        "row q_top truth changed",
    )
    arrays, receipt = replay_trial_metrics(
        metadata, config, trials, trajectories, draws,
        expected_left, expected_right,
    )
    if row["replay_receipt"] != receipt:
        raise RuntimeError("seed replay receipt changed")
    decisions = decision_arrays(arrays, multiplier, config)
    confidence = float(config["decision_rule"]["binomial_lower_confidence"])
    fail_confidence = fail_adjusted_confidence(config, scenario_registry)
    for name, values in decisions.items():
        verify_rate(
            row, values, f"candidate_{name}", confidence, fail_confidence,
        )
    coverage_applicable = metadata["classification"] != "known_blind"
    if row["coverage_applicable"] != coverage_applicable:
        raise RuntimeError("row coverage authority changed")
    if coverage_applicable:
        covered = (
            np.abs(arrays["q_top_delta_signed"] - metadata["truth"]["q_top_delta_signed"])
            <= float(multiplier) * arrays["q_top_delta_se"] + 2e-13
        ) & (
            np.abs(arrays["d2_norm"] - metadata["truth"]["d2_norm"])
            <= float(multiplier) * arrays["d2_se"] + 2e-13
        )
        verify_rate(row, covered, "joint_coverage", confidence, fail_confidence)
    elif any(row[name] is not None for name in (
            "joint_coverage_count", "joint_coverage_rate",
            "joint_coverage_wilson_lower", "joint_coverage_wilson_upper",
            "joint_coverage_fail_adjusted_wilson_lower",
            "joint_coverage_fail_adjusted_wilson_upper")):
        raise RuntimeError("known-blind row acquired target coverage")
    augmented = dict(row)
    augmented["_raw_trial_metrics"] = arrays
    return augmented


def rate_verdict(lower, upper, target):
    if float(lower) >= float(target):
        return "PASS"
    if float(upper) < float(target):
        return "FAIL"
    return "INCONCLUSIVE"


def summarize(rows, config, trajectories, draws, multiplier, stage):
    rule = config["decision_rule"]
    groups = {
        "null": [row for row in rows if row["classification"] in {"null", "d2_null"}],
        "good": [row for row in rows if row["classification"] in {"good_q_top", "good_d2"}],
        "bad_q_top": [row for row in rows if row["classification"] == "bad_q_top"],
        "bad_d2": [row for row in rows if row["classification"] == "bad_d2"],
        "coverage": [row for row in rows if row["coverage_applicable"]],
        "known_blind": [row for row in rows if row["classification"] == "known_blind"],
    }
    if any(not values for values in groups.values()):
        raise RuntimeError("summary requirement group changed")

    def rowwise(name, subset, prefix, target):
        lower = min(row[f"{prefix}_wilson_lower"] for row in subset)
        upper = min(
            row[f"{prefix}_fail_adjusted_wilson_upper"] for row in subset
        )
        return {
            "fail_adjusted_worst_wilson_upper": float(upper),
            "name": name, "row_count": len(subset), "target_rate": float(target),
            "pass_worst_wilson_lower": float(lower),
            "verdict": rate_verdict(lower, upper, target),
        }

    def maximum_rate(name, subset, prefix, limit):
        lower = max(
            row[f"{prefix}_fail_adjusted_wilson_lower"] for row in subset
        )
        upper = max(row[f"{prefix}_wilson_upper"] for row in subset)
        verdict = (
            "PASS" if upper <= float(limit)
            else "FAIL" if lower > float(limit)
            else "INCONCLUSIVE"
        )
        return {
            "limit_rate": float(limit), "name": name,
            "row_count": len(subset), "verdict": verdict,
            "fail_adjusted_worst_wilson_lower": float(lower),
            "pass_worst_wilson_upper": float(upper),
        }

    requirements = [
        rowwise("null_pass_power", groups["null"], "candidate_pass", rule["null_pass_minimum"]),
        rowwise(
            "good_alternative_pass_power", groups["good"], "candidate_pass",
            rule["good_alternative_pass_minimum"],
        ),
        rowwise(
            "bad_q_top_fail_power", groups["bad_q_top"], "candidate_q_top_fail",
            rule["fail_power_minimum"],
        ),
        rowwise(
            "bad_d2_fail_power", groups["bad_d2"], "candidate_d2_fail",
            rule["fail_power_minimum"],
        ),
        maximum_rate(
            "bad_q_top_false_pass_control", groups["bad_q_top"],
            "candidate_q_top_pass", rule["bad_false_pass_maximum"],
        ),
        maximum_rate(
            "bad_d2_false_pass_control", groups["bad_d2"],
            "candidate_d2_pass", rule["bad_false_pass_maximum"],
        ),
        rowwise(
            "known_blind_expected_pass", groups["known_blind"], "candidate_pass",
            rule["known_blind_pass_minimum"],
        ),
    ]
    simultaneous = None
    for row in groups["coverage"]:
        raw = row["_raw_trial_metrics"]
        covered = (
            np.abs(np.asarray(raw["q_top_delta_signed"]) - row["true_q_top_delta_signed"])
            <= float(multiplier) * np.asarray(raw["q_top_delta_se"]) + 2e-13
        ) & (
            np.abs(np.asarray(raw["d2_norm"]) - row["true_d2_norm"])
            <= float(multiplier) * np.asarray(raw["d2_se"]) + 2e-13
        )
        simultaneous = covered if simultaneous is None else simultaneous & covered
    count = int(np.count_nonzero(simultaneous))
    trials = int(simultaneous.size)
    confidence = float(rule["binomial_lower_confidence"])
    lower = wilson_lower(count, trials, confidence)
    upper = wilson_upper(count, trials, confidence)
    fail_confidence = fail_adjusted_confidence(config, rows)
    fail_upper = wilson_upper(count, trials, fail_confidence)
    requirements.append({
        "name": "simultaneous_registered_interval_coverage",
        "row_count": len(groups["coverage"]), "success_count": count,
        "trial_count": trials, "target_rate": float(rule["interval_coverage_minimum"]),
        "fail_adjusted_worst_wilson_upper": float(fail_upper),
        "pass_worst_wilson_lower": float(lower),
        "verdict": rate_verdict(
            lower, fail_upper, rule["interval_coverage_minimum"]
        ),
    })
    verdicts = {row["verdict"] for row in requirements}
    decision = "PASS" if verdicts == {"PASS"} else (
        "FAIL" if "FAIL" in verdicts else "INCONCLUSIVE"
    )
    return {
        "cost_independent_draws_per_side": int(trajectories) * int(draws),
        "decision": decision,
        "draws_per_trajectory": int(draws),
        "eligible": decision == "PASS",
        "fail_adjusted_confidence": float(fail_confidence),
        "fail_hypothesis_count": fail_hypothesis_count(config, rows),
        "known_blind_controls_preserved": all(
            row["expected_interpretation"] == "EXPECTED_KNOWN_BLIND"
            for row in groups["known_blind"]
        ),
        "multiplier": float(multiplier),
        "requirements": requirements,
        "row_count": len(rows),
        "stage": stage,
        "trajectory_count": int(trajectories),
    }


def verify_report_schema(report):
    top_level = {
        "authority", "bound_validation_062", "calibration_points",
        "confirmation_point", "confirmation_rows", "config_sha256",
        "estimator_contract", "known_blind_interpretation",
        "multiplicity_contract", "project_transition_status", "remaining_blockers",
        "replay_receipt_contract", "report_sha256", "rng_replay_contract",
        "scenario_registry",
        "selected_common_operating_point", "selection_points", "selection_rows",
        "source_commit", "source_file_count", "source_tree_sha256", "status",
        "version",
    }
    if set(report) != top_level:
        raise RuntimeError("delivery-gate report top-level schema changed")
    if set(report["bound_validation_062"]) != {
            "audit_sha256", "report_sha256", "source_commit"}:
        raise RuntimeError("bound-input report schema changed")
    calibration_point_keys = {
        "draws_per_trajectory", "multiplier",
        "outer_max_standardized_errors_sha256",
        "quantile_probability", "scenario_quantiles", "trajectory_count", "valid",
    }
    calibration_row_keys = {
        "joint_standardized_errors_sha256", "marginal_quantile_diagnostic",
        "replay_receipt", "scenario_id", "seed_left", "seed_right",
        "trial_count",
    }
    for point in report["calibration_points"]:
        if set(point) != calibration_point_keys:
            raise RuntimeError("calibration point schema changed")
        for row in point["scenario_quantiles"]:
            if set(row) != calibration_row_keys:
                raise RuntimeError("calibration row schema changed")
    decision_names = {
        "pass", "fail", "inconclusive", "q_top_pass", "q_top_fail",
        "q_top_inconclusive", "d2_pass", "d2_fail", "d2_inconclusive",
    }
    rate_keys = {
        f"candidate_{decision}_{suffix}"
        for decision in decision_names
        for suffix in ("count", "rate", "wilson_lower", "wilson_upper")
    }
    rate_keys |= {
        f"candidate_{decision}_fail_adjusted_wilson_{side}"
        for decision in decision_names for side in ("lower", "upper")
    }
    selection_row_keys = rate_keys | {
        "character_diagnostic_participates_in_decision", "classification",
        "coverage_applicable", "draws_per_trajectory", "evaluation_trials",
        "expected_interpretation", "joint_coverage_count", "joint_coverage_rate",
        "joint_coverage_wilson_lower", "joint_coverage_wilson_upper", "multiplier",
        "joint_coverage_fail_adjusted_wilson_lower",
        "joint_coverage_fail_adjusted_wilson_upper", "replay_receipt",
        "scenario_id", "seed_left", "seed_right", "stage",
        "trajectory_count", "true_d2_norm", "true_q_top_delta_signed",
    }
    for row in [*report["selection_rows"], *report["confirmation_rows"]]:
        if set(row) != selection_row_keys:
            raise RuntimeError("selection/confirmation row schema changed")
    valid_point_keys = {
        "cost_independent_draws_per_side", "decision", "draws_per_trajectory",
        "eligible", "fail_adjusted_confidence", "fail_hypothesis_count",
        "known_blind_controls_preserved", "multiplier",
        "requirements", "row_count", "stage", "trajectory_count",
    }
    invalid_point_keys = {
        "decision", "draws_per_trajectory", "eligible", "invalid_reason",
        "multiplier", "stage", "trajectory_count",
    }
    for point in report["selection_points"]:
        if set(point) not in (valid_point_keys, invalid_point_keys):
            raise RuntimeError("selection point schema changed")
    if report["confirmation_point"] is not None and set(
            report["confirmation_point"]) != valid_point_keys:
        raise RuntimeError("confirmation point schema changed")
    scenario_keys = {
        "character_diagnostic_mask_count", "character_diagnostic_mask_sha256",
        "classification", "id", "k", "labels_left", "labels_right",
        "observation_mode", "probabilities_left", "probabilities_right",
        "requested_q_top_delta", "source", "truth", "uint64_bit63_exercised",
    }
    for row in report["scenario_registry"]:
        if set(row) != scenario_keys:
            raise RuntimeError("scenario registry schema changed")


def verify_decisions(config, report, scenario_registry):
    expected_authority = {
        "cell_certification": False,
        "formal_authorization": False,
        "maximum_status": "LOCAL_DELIVERY_GATE_COMMON_OPERATING_POINT_CONFIRMED",
        "mixing_certification": False,
        "production_authorization": False,
        "remote_authorization": False,
        "target_basin_mass_certification": False,
    }
    if report["authority"] != expected_authority or report["version"] != config["version"]:
        raise RuntimeError("report authority or version changed")
    expected_estimator = {
        "character_maximum_role": "DIAGNOSTIC_ONLY",
        "d2_is_total_variation": False,
        "d2_raw_clipped": False,
        "d2_upper_decision_projects_estimate_at_zero": True,
        "full_label_collision_u_statistic": True,
        "fixed_strata_must_be_analyzed_separately": True,
        "groupwise_delete_one_variances_added": True,
        "iid_same_distribution_assumption_required_within_family": True,
        "jackknife_interval_is_strict_confidence_interval": False,
        "q_top_raw_clipped": False,
        "time_blocks_may_be_treated_as_trajectories": False,
    }
    if report["estimator_contract"] != expected_estimator:
        raise RuntimeError("estimator authority changed")
    expected_blind = {
        "adversarial_initialization_still_required": True,
        "direct_full_labels_certify_unvisited_target_mass": False,
        "orthogonal_confirmation_still_required": True,
        "status": "EXPECTED_KNOWN_BLIND",
        "transport_gates_still_required": True,
    }
    if report["known_blind_interpretation"] != expected_blind:
        raise RuntimeError("known-blind authority changed")
    expected_blockers = [
        "LARGE_K_ORTHOGONAL_CONFIRMER_PORTFOLIO_UNFROZEN",
        "FUTURE_SCHEMA_RUNTIME_COVERAGE_INCOMPLETE",
        "CAMPAIGN_BUDGET_UNAPPROVED",
        "STAGE3_MULTI_COMPARISON_MULTIPLICITY_UNFROZEN",
    ]
    if report["remaining_blockers"] != expected_blockers:
        raise RuntimeError("remaining blocker registry changed")
    if report["rng_replay_contract"] != config["rng_replay_contract"]:
        raise RuntimeError("report RNG replay contract changed")
    if report["project_transition_status"] != "BLOCKED_BEFORE_REMOTE":
        raise RuntimeError("project transition status changed")
    fail_confidence = fail_adjusted_confidence(config, scenario_registry)
    expected_multiplicity = {
        "fail_adjusted_confidence": fail_confidence,
        "fail_familywise_confidence": config["decision_rule"][
            "fail_familywise_confidence"
        ],
        "fail_hypothesis_count": fail_hypothesis_count(config, scenario_registry),
        "pass_confidence": config["decision_rule"]["binomial_lower_confidence"],
        "fail_hypotheses_per_stage_point": fail_hypotheses_per_stage_point(
            scenario_registry
        ),
        "scope": config["decision_rule"]["fail_multiplicity_scope"],
        "strict_familywise_guarantee": False,
        "wilson_role": "OPERATIONAL_ONLY",
    }
    if report["multiplicity_contract"] != expected_multiplicity:
        raise RuntimeError("FAIL multiplicity contract changed")
    if report["replay_receipt_contract"] != {
            "maximum_report_bytes": config["report_contract"]["maximum_bytes"],
            "persistent_raw": False, "schema": REPLAY_RECEIPT_SCHEMA,
            "seed_replay_required": True}:
        raise RuntimeError("replay receipt contract changed")
    points = ordered_points(config)
    calibrations = report["calibration_points"]
    selection_points = report["selection_points"]
    if len(calibrations) != len(selection_points):
        raise RuntimeError("calibration/selection point count changed")
    config_sha = sha256_file(CONFIG_PATH)
    metadata_by_id = {row["id"]: row for row in scenario_registry}
    recomputed_points = []
    selected = None
    for index, (calibration, stored) in enumerate(zip(calibrations, selection_points)):
        trajectories, draws = points[index]
        if (calibration["trajectory_count"], calibration["draws_per_trajectory"]) != (
                trajectories, draws):
            raise RuntimeError("calibration operating order changed")
        verify_calibration_point(calibration, config, config_sha, scenario_registry)
        if not calibration["valid"]:
            expected = {
                "decision": "INCONCLUSIVE", "draws_per_trajectory": draws,
                "eligible": False, "invalid_reason": "NO_FINITE_CALIBRATED_MULTIPLIER",
                "multiplier": None, "stage": "selection",
                "trajectory_count": trajectories,
            }
            if canonical(stored) != canonical(expected):
                raise RuntimeError("invalid calibration point decision changed")
            recomputed_points.append(expected)
            continue
        rows = [
            row for row in report["selection_rows"]
            if (row["trajectory_count"], row["draws_per_trajectory"]) == (
                trajectories, draws)
        ]
        if {row["scenario_id"] for row in rows} != set(metadata_by_id) or len(rows) != len(metadata_by_id):
            raise RuntimeError("selection row registry changed")
        replay_rows = []
        for row in rows:
            replay_rows.append(verify_row(
                row, config, config_sha, metadata_by_id[row["scenario_id"]],
                "selection", trajectories, draws, calibration["multiplier"],
                scenario_registry,
            ))
        expected = summarize(
            replay_rows, config, trajectories, draws, calibration["multiplier"],
            "selection",
        )
        if canonical(stored) != canonical(expected):
            raise RuntimeError("selection point decision changed")
        recomputed_points.append(expected)
        if expected["eligible"]:
            selected = expected
            if index != len(selection_points) - 1:
                raise RuntimeError("selection continued after first PASS")
            break
    if selected is None and len(selection_points) != len(points):
        raise RuntimeError("selection stopped without PASS")
    expected_selection_rows = sum(
        len(metadata_by_id) for calibration in calibrations if calibration["valid"]
    )
    if len(report["selection_rows"]) != expected_selection_rows:
        raise RuntimeError("selection report contains extra or missing rows")
    if canonical(report["selected_common_operating_point"]) != canonical(selected):
        raise RuntimeError("selected operating point changed")

    confirmation = report["confirmation_point"]
    confirmation_rows = report["confirmation_rows"]
    if selected is None:
        if confirmation is not None or confirmation_rows:
            raise RuntimeError("confirmation ran without selection PASS")
    else:
        trajectories = selected["trajectory_count"]
        draws = selected["draws_per_trajectory"]
        if (
            {row["scenario_id"] for row in confirmation_rows} != set(metadata_by_id)
            or len(confirmation_rows) != len(metadata_by_id)
        ):
            raise RuntimeError("confirmation row registry changed")
        replay_confirmation_rows = []
        for row in confirmation_rows:
            replay_confirmation_rows.append(verify_row(
                row, config, config_sha, metadata_by_id[row["scenario_id"]],
                "confirmation", trajectories, draws, selected["multiplier"],
                scenario_registry,
            ))
        expected_confirmation = summarize(
            replay_confirmation_rows, config, trajectories, draws,
            selected["multiplier"],
            "confirmation",
        )
        if canonical(confirmation) != canonical(expected_confirmation):
            raise RuntimeError("confirmation decision changed")

    if selected is not None and confirmation["decision"] == "PASS":
        expected_status = "LOCAL_DELIVERY_GATE_COMMON_OPERATING_POINT_CONFIRMED"
    elif confirmation is not None:
        expected_status = (
            "SELECTED_POINT_CONFIRMATION_FAILED_REDESIGN_REQUIRED"
            if confirmation["decision"] == "FAIL"
            else "DELIVERY_GATE_CALIBRATION_INCONCLUSIVE"
        )
    elif selection_points and all(point["decision"] == "FAIL" for point in recomputed_points):
        expected_status = "DELIVERY_GATE_REDESIGN_REQUIRED"
    else:
        expected_status = "DELIVERY_GATE_CALIBRATION_INCONCLUSIVE"
    if report["status"] != expected_status:
        raise RuntimeError("terminal statistical taxonomy changed")


def main():
    if OUTPUT_PATH.exists():
        raise RuntimeError("independent audit already exists")
    if not REPORT_PATH.is_file():
        raise RuntimeError("delivery-gate report does not exist")
    if REPORT_PATH.is_symlink():
        raise RuntimeError("delivery-gate report may not be a symlink")
    config = load_json_strict(CONFIG_PATH)
    if REPORT_PATH.stat().st_size > int(config["report_contract"]["maximum_bytes"]):
        raise RuntimeError("delivery-gate report exceeds frozen size limit")
    report = load_json_strict(REPORT_PATH)
    verify_report_schema(report)
    validate_config_contract(config)
    verify_source_provenance(config, report)
    bound_report = verify_bound_validation_062(config)
    expected_registry = expected_scenario_registry(config, bound_report)
    if canonical(report["scenario_registry"]) != canonical(expected_registry):
        raise RuntimeError("delivery-gate scenario registry changed")
    if report["bound_validation_062"] != {
            "audit_sha256": config["bound_validation_062"]["audit"]["audit_sha256"],
            "report_sha256": bound_report["report_sha256"],
            "source_commit": bound_report["source_commit"]}:
        raise RuntimeError("report validation-062 binding changed")
    verify_decisions(config, report, expected_registry)
    core = {
        "config_sha256": report["config_sha256"],
        "report_sha256": report["report_sha256"],
        "source_commit": report["source_commit"],
        "status": "INDEPENDENT_AUDIT_PASS_" + report["status"],
        "version": "exp102.q0_delivery_gate_calibration.audit.v1",
    }
    core["audit_sha256"] = sha256_bytes(canonical(core).encode("ascii"))
    with OUTPUT_PATH.open("x", encoding="ascii") as handle:
        handle.write(canonical(core) + "\n")
    print(json.dumps(core, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
