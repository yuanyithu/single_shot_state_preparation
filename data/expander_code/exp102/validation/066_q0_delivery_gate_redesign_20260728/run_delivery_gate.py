"""Calibrate direct full-label q_top and D2 delivery gates on local IID draws."""

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


sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parent
EXP102_ROOT = ROOT.parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[5]
CONFIG_PATH = ROOT / "delivery_gate_config.json"
OUTPUT_PATH = ROOT / "delivery_gate_report.json"
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


REPLAY_RECEIPT_SCHEMA = "exp102.q0_delivery_gate.replay_receipt.le.v1"


def digest_named_arrays(domain, arrays):
    """Commit arrays in a dtype/endianness/shape-explicit binary format."""
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
    counts_digest = digest_named_arrays("histogram_counts", {
        "labels": (labels, "<u8"),
        "left_counts": (left_counts, "<i8"),
        "right_counts": (right_counts, "<i8"),
    })
    metric_digest = digest_named_arrays("all_trial_metrics", {
        name: (values, "<f8") for name, values in metrics.items()
    })
    return {
        "all_trial_metrics_sha256": metric_digest,
        "histogram_counts_sha256": counts_digest,
        "schema": REPLAY_RECEIPT_SCHEMA,
    }


def verify_self_hash(payload, field):
    expected = payload[field]
    unsigned = dict(payload)
    unsigned.pop(field)
    actual = sha256_bytes(canonical(unsigned).encode("ascii"))
    if actual != expected:
        raise RuntimeError(f"self hash changed: {field}")


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


def ordered_operating_points(config):
    points = [
        (int(row["trajectory_count"]), int(row["draws_per_trajectory"]))
        for row in config["common_operating_grid"]["points"]
    ]
    if len(points) != len(set(points)):
        raise RuntimeError("duplicate operating point")
    ordered = sorted(points, key=lambda row: (row[0] * row[1], row[0], row[1]))
    if points != ordered:
        raise RuntimeError("operating points are not frozen in cost order")
    return ordered


def validate_config(config):
    verify_self_hash(config, "config_self_sha256")
    if config["version"] != "exp102.q0_delivery_gate_calibration.v1":
        raise RuntimeError("delivery-gate config version changed")
    rule = config["decision_rule"]
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
    if rule != expected_rule:
        raise RuntimeError("delivery-gate state registry changed")
    sparse = config["sparse_label_stress"]
    if sparse["dimensions"] != [9, 16, 36, 64]:
        raise RuntimeError("sparse logical-dimension registry changed")
    if sparse["q_top_deltas"] != [0.0, 0.02, 0.04, 0.06]:
        raise RuntimeError("sparse q_top stress registry changed")
    if sparse["d2_deltas"] != [0.0, 0.02, 0.04, 0.06]:
        raise RuntimeError("sparse D2 stress registry changed")
    if sparse["q_top_bases"] != [0.05, 0.15, 0.55, 0.9]:
        raise RuntimeError("sparse q_top profile registry changed")
    if int(sparse["sparse_q_top_support_size"]) != 256:
        raise RuntimeError("sparse q_top support size changed")
    if sparse["distinct_freeze_label_rule"] != (
            "trajectory_index_with_last_two_uint64_boundaries"):
        raise RuntimeError("distinct-freeze label rule changed")
    if int(sparse["same_purity_disjoint_support_size"]) != 32:
        raise RuntimeError("same-purity D2 stress changed")
    diagnostic = config["character_diagnostic"]
    if diagnostic["participates_in_decision"] or diagnostic[
            "participates_in_coverage"]:
        raise RuntimeError("character diagnostic acquired decision authority")
    if config["authority"] != {
        "cell_certification": False,
        "formal_authorization": False,
        "maximum_status": "LOCAL_DELIVERY_GATE_COMMON_OPERATING_POINT_CONFIRMED",
        "mixing_certification": False,
        "production_authorization": False,
        "remote_authorization": False,
        "target_basin_mass_certification": False,
    }:
        raise RuntimeError("delivery-gate authority changed")
    if set(config["source_artifacts"]) != {
            "auditor", "readme", "red_team", "runner", "tests"}:
        raise RuntimeError("source artifact registry changed")
    if config["replications"] != {
            "calibration_trials": 600, "confirmation_trials": 300,
            "selection_trials": 300}:
        raise RuntimeError("delivery-gate replication registry changed")
    if config["uncertainty_calibration"] != {
            "higher_quantile": 0.995,
            "joint_scalar_estimands": ["signed_q_top_delta", "d2_norm"],
            "multiplier_shared_across_scenarios": True}:
        raise RuntimeError("uncertainty calibration registry changed")
    if config["fail_multiplicity_contract"] != {
            "expected_hypotheses_per_stage_point": 139,
            "expected_total_hypothesis_count": 1390,
            "strict_familywise_guarantee": False,
            "wilson_role": "OPERATIONAL_ONLY"}:
        raise RuntimeError("FAIL multiplicity registry changed")
    if config["report_contract"] != {
            "maximum_bytes": 10485760,
            "persistent_per_trial_arrays": False}:
        raise RuntimeError("compact report registry changed")
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
    ordered_operating_points(config)


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
        raise RuntimeError("validation 066 requires a completely clean worktree")


def configured_source_artifacts(config):
    rows = []
    for name, spec in sorted(config["source_artifacts"].items()):
        expected = spec["sha256"]
        if re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise RuntimeError(f"invalid source SHA for {name}")
        rows.append((_exp102_path(spec["path"]), expected))
    if len({path for path, _ in rows}) != len(rows):
        raise RuntimeError("duplicate configured source artifact")
    return rows


def verify_committed_clean_source(config):
    validate_config(config)
    reject_validation_bytecode()
    if OUTPUT_PATH.exists() or AUDIT_PATH.exists():
        raise RuntimeError("validation 066 output already exists")
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
        raise RuntimeError("bound validation-062 report self identity changed")
    if report["status"] != report_spec["required_status"]:
        raise RuntimeError("bound validation-062 report status changed")
    if report["source_commit"] != report_spec["required_source_commit"]:
        raise RuntimeError("bound validation-062 source commit changed")
    if audit["status"] != audit_spec["required_status"]:
        raise RuntimeError("bound validation-062 audit status changed")
    if audit["audit_sha256"] != audit_spec["audit_sha256"]:
        raise RuntimeError("bound validation-062 audit self identity changed")
    if audit["report_sha256"] != report["report_sha256"]:
        raise RuntimeError("validation-062 audit does not bind the report")
    if audit["source_commit"] != report["source_commit"]:
        raise RuntimeError("validation-062 audit source identity changed")
    return report


def labels_hex(labels):
    return [f"0x{int(value):016x}" for value in np.asarray(labels, dtype=np.uint64)]


def parse_label(value):
    parsed = int(value, 16) if isinstance(value, str) else int(value)
    if not 0 <= parsed < (1 << 64):
        raise RuntimeError("logical label is outside uint64")
    return np.uint64(parsed)


def validate_distribution(labels, probabilities, k):
    labels = np.asarray(labels, dtype=np.uint64).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if labels.size == 0 or labels.size != probabilities.size:
        raise RuntimeError("logical distribution shape changed")
    if len({int(value) for value in labels}) != labels.size:
        raise RuntimeError("logical distribution contains duplicate labels")
    if any(int(value) >= (1 << int(k)) for value in labels):
        raise RuntimeError("logical label exceeds its registered dimension")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
        raise RuntimeError("logical distribution has invalid probabilities")
    if abs(float(probabilities.sum()) - 1.0) > 2e-13:
        raise RuntimeError("logical distribution is not normalized")
    return labels, probabilities


def normalize_collision(collision, k):
    """Use Python-int 2**k without losing the k=64 correction to 1.0."""
    size = 1 << int(k)
    collision = np.asarray(collision, dtype=np.float64)
    return collision + (collision - 1.0) / (size - 1)


def normalize_d2(raw_d2, k):
    size = 1 << int(k)
    raw_d2 = np.asarray(raw_d2, dtype=np.float64)
    return raw_d2 + raw_d2 / (size - 1)


def normalized_q_top(probabilities, k):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    return float(normalize_collision(np.dot(probabilities, probabilities), k))


def aligned_probabilities(labels_a, probabilities_a, labels_b, probabilities_b):
    labels = np.asarray(sorted(
        {int(value) for value in labels_a} | {int(value) for value in labels_b}
    ), dtype=np.uint64)
    positions = {int(value): index for index, value in enumerate(labels)}
    left = np.zeros(labels.size, dtype=np.float64)
    right = np.zeros(labels.size, dtype=np.float64)
    for label, probability in zip(labels_a, probabilities_a):
        left[positions[int(label)]] = float(probability)
    for label, probability in zip(labels_b, probabilities_b):
        right[positions[int(label)]] = float(probability)
    return labels, left, right


def true_distribution_metrics(labels_a, probabilities_a, labels_b, probabilities_b, k):
    labels, left, right = aligned_probabilities(
        labels_a, probabilities_a, labels_b, probabilities_b,
    )
    del labels
    q_left = float(normalize_collision(np.dot(left, left), k))
    q_right = float(normalize_collision(np.dot(right, right), k))
    return {
        "d2_norm": float(normalize_d2(np.dot(left - right, left - right), k)),
        "q_top_delta_abs": abs(q_left - q_right),
        "q_top_delta_signed": q_left - q_right,
        "q_top_left": q_left,
        "q_top_right": q_right,
    }


def inverse_walsh(nonzero_means):
    nonzero_means = np.asarray(nonzero_means, dtype=np.float64).reshape(-1)
    size = nonzero_means.size + 1
    if size <= 1 or size & (size - 1):
        raise RuntimeError("complete logical character catalog has invalid size")
    k = size.bit_length() - 1
    coefficients = np.concatenate((np.ones(1), nonzero_means))
    probabilities = np.empty(size, dtype=np.float64)
    for label in range(size):
        total = 0.0
        for mask, mean in enumerate(coefficients):
            sign = -1.0 if ((mask & label).bit_count() & 1) else 1.0
            total += float(mean) * sign
        probabilities[label] = total / size
    if np.min(probabilities) < -2e-13:
        raise RuntimeError("inverse Walsh produced negative target mass")
    if np.any(probabilities < 0.0):
        raise RuntimeError("inverse Walsh requires numerical clipping")
    if abs(float(probabilities.sum()) - 1.0) > 2e-13:
        raise RuntimeError("inverse Walsh distribution is not normalized")
    recovered = character_means_from_distribution(
        np.arange(size, dtype=np.uint64), probabilities,
        np.arange(1, size, dtype=np.uint64),
    )
    if not np.allclose(recovered, nonzero_means, atol=2e-13, rtol=0.0):
        raise RuntimeError("forward Walsh does not recover bound character means")
    return np.arange(size, dtype=np.uint64), probabilities, k


def character_means_from_distribution(labels, probabilities, masks):
    labels = np.asarray(labels, dtype=np.uint64)
    masks = np.asarray(masks, dtype=np.uint64)
    parity = np.bitwise_count(labels[:, None] & masks[None, :]) & np.uint8(1)
    signs = 1.0 - 2.0 * parity.astype(np.float64)
    return np.asarray(probabilities, dtype=np.float64) @ signs


def shift_distribution_q_top(labels, probabilities, k, requested_delta):
    labels, probabilities = validate_distribution(labels, probabilities, k)
    requested_delta = float(requested_delta)
    base = normalized_q_top(probabilities, k)
    if requested_delta == 0.0:
        return labels.copy(), probabilities.copy()
    if base + requested_delta <= 1.0 - 1e-13:
        target = base + requested_delta
        modal = int(np.argmax(probabilities))
        endpoint = np.zeros(probabilities.size, dtype=np.float64)
        endpoint[modal] = 1.0
        low, high = 0.0, 1.0
        for _ in range(100):
            middle = (low + high) / 2.0
            candidate = (1.0 - middle) * probabilities + middle * endpoint
            if normalized_q_top(candidate, k) < target:
                low = middle
            else:
                high = middle
        shifted = (1.0 - high) * probabilities + high * endpoint
    else:
        if base < requested_delta:
            raise RuntimeError("q_top stress has no legal direction")
        target = base - requested_delta
        full_size = 1 << int(k)
        if labels.size != full_size or any(
                int(label) != index for index, label in enumerate(labels)):
            raise RuntimeError("downward q_top shift requires complete support")
        alpha = 1.0 - math.sqrt(target / base)
        shifted = (1.0 - alpha) * probabilities + alpha / full_size
    achieved = abs(normalized_q_top(shifted, k) - base)
    if abs(achieved - requested_delta) > 2e-13:
        raise RuntimeError("legal distribution did not realize requested q_top delta")
    return labels.copy(), shifted


def _p_token(value):
    return f"{float(value):.2f}".replace(".", "p")


def _d_token(value):
    return f"{float(value):.2f}".replace(".", "p")


def _scenario(
        scenario_id, source, classification, k, labels_left, probabilities_left,
        labels_right, probabilities_right, *, observation_mode="iid_target",
        requested_q_top_delta=None):
    labels_left, probabilities_left = validate_distribution(
        labels_left, probabilities_left, k,
    )
    labels_right, probabilities_right = validate_distribution(
        labels_right, probabilities_right, k,
    )
    truths = true_distribution_metrics(
        labels_left, probabilities_left, labels_right, probabilities_right, k,
    )
    return {
        "classification": classification,
        "id": scenario_id,
        "k": int(k),
        "labels_left": labels_left,
        "labels_right": labels_right,
        "observation_mode": observation_mode,
        "probabilities_left": probabilities_left,
        "probabilities_right": probabilities_right,
        "requested_q_top_delta": requested_q_top_delta,
        "source": source,
        "truth": truths,
    }


def reconstruct_small_hgp_scenarios(config, report):
    row_filter = config["bound_validation_062"]["source_row_filter"]
    rows = [
        row for row in report["selection_rows"]
        if all(row.get(name) == value for name, value in row_filter.items())
    ]
    expected = int(config["bound_validation_062"]["unique_distribution_count"])
    if len(rows) != expected:
        raise RuntimeError("bound validation-062 complete-logical row count changed")
    identities = set()
    scenarios = []
    for row in rows:
        identity = (row["model_id"], row["syndrome_kind"], float(row["p"]))
        if identity in identities:
            raise RuntimeError("duplicate bound small-HGP distribution")
        identities.add(identity)
        labels, base, k = inverse_walsh(row["base_logical_means"])
        source_q_top = float(row["true_q_top_left"])
        if abs(normalized_q_top(base, k) - source_q_top) > 2e-13:
            raise RuntimeError("inverse-Walsh purity disagrees with validation 062")
        for delta in config["sparse_label_stress"]["q_top_deltas"]:
            shifted_labels, shifted = shift_distribution_q_top(
                labels, base, k, delta,
            )
            classification = {
                0.0: "null",
                0.02: "good_q_top",
                0.04: "boundary_q_top",
                0.06: "bad_q_top",
            }[float(delta)]
            scenario_id = (
                f"exact_{row['model_id']}_{row['syndrome_kind']}_"
                f"p{_p_token(row['p'])}_qdelta_{_d_token(delta)}"
            )
            scenarios.append(_scenario(
                scenario_id, "validation_062_inverse_walsh", classification, k,
                labels, base, shifted_labels, shifted,
                requested_q_top_delta=float(delta),
            ))
    return scenarios


def sparse_profile_distribution(k, q_top, support_size):
    support_size = int(support_size)
    size = 1 << int(k)
    purity = 1.0 / size + (1.0 - 1.0 / size) * float(q_top)
    if not 1.0 / support_size <= purity <= 1.0:
        raise RuntimeError("sparse q_top profile is infeasible")
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
    if abs(normalized_q_top(probabilities, k) - float(q_top)) > 2e-13:
        raise RuntimeError("sparse profile did not realize requested q_top")
    return labels, probabilities


def disjoint_uniform_pair(k, support_size):
    support_size = int(support_size)
    left = np.arange(support_size, dtype=np.uint64)
    if int(k) == 64:
        high_bit = np.uint64(1 << 63)
        right = high_bit | np.arange(support_size, dtype=np.uint64)
    else:
        right = np.arange(support_size, 2 * support_size, dtype=np.uint64)
    probabilities = np.full(support_size, 1.0 / support_size, dtype=np.float64)
    return left, probabilities, right, probabilities.copy()


def controlled_same_purity_d2_pair(k, requested_d2, support_size=64):
    support_size = int(support_size)
    labels = np.arange(support_size, dtype=np.uint64)
    if int(k) == 64:
        labels[-1] = np.uint64(1 << 63)
    size = 1 << int(k)
    raw_d2 = float(requested_d2) - float(requested_d2) / size
    difference = math.sqrt(raw_d2 / 2.0)
    center = 0.2
    left = np.full(support_size, 0.6 / (support_size - 2), dtype=np.float64)
    left[0] = center + difference / 2.0
    left[1] = center - difference / 2.0
    right = left.copy()
    right[0], right[1] = left[1], left[0]
    metrics = true_distribution_metrics(labels, left, labels, right, k)
    if abs(metrics["q_top_delta_abs"]) > 2e-13:
        raise RuntimeError("controlled D2 stress changed purity")
    if abs(metrics["d2_norm"] - float(requested_d2)) > 2e-13:
        raise RuntimeError("controlled D2 stress missed requested distance")
    return labels, left, labels.copy(), right


def build_sparse_scenarios(config):
    sparse = config["sparse_label_stress"]
    scenarios = []
    for k in sparse["dimensions"]:
        for base_q_top in sparse["q_top_bases"]:
            labels, base = sparse_profile_distribution(
                k, base_q_top, sparse["sparse_q_top_support_size"],
            )
            for delta in sparse["q_top_deltas"]:
                shifted_labels, shifted = shift_distribution_q_top(
                    labels, base, k, delta,
                )
                classification = {
                    0.0: "null",
                    0.02: "good_q_top",
                    0.04: "boundary_q_top",
                    0.06: "bad_q_top",
                }[float(delta)]
                scenarios.append(_scenario(
                    f"sparse_k{k}_base_{_d_token(base_q_top)}_"
                    f"qdelta_{_d_token(delta)}",
                    "outcome_blind_sparse_label_distribution", classification, k,
                    labels, base, shifted_labels, shifted,
                    requested_q_top_delta=float(delta),
                ))
        for d2_delta in sparse["d2_deltas"]:
            left, p_left, right, p_right = controlled_same_purity_d2_pair(
                k, d2_delta,
            )
            classification = {
                0.0: "d2_null",
                0.02: "good_d2",
                0.04: "boundary_d2",
                0.06: "bad_d2",
            }[float(d2_delta)]
            scenarios.append(_scenario(
                f"sparse_k{k}_same_purity_d2_{_d_token(d2_delta)}",
                "outcome_blind_sparse_label_distribution", classification, k,
                left, p_left, right, p_right,
            ))
        left, p_left, right, p_right = disjoint_uniform_pair(
            k, sparse["same_purity_disjoint_support_size"],
        )
        scenarios.append(_scenario(
            f"sparse_k{k}_same_purity_disjoint_d2_stress",
            "outcome_blind_sparse_label_distribution", "bad_d2", k,
            left, p_left, right, p_right,
        ))

    left, p_left, right, p_right = disjoint_uniform_pair(
        64, sparse["same_purity_disjoint_support_size"],
    )
    scenarios.append(_scenario(
        "control_k64_common_freeze", "known_transport_blind_control",
        "known_blind", 64, left, p_left, right, p_right,
        observation_mode="common_freeze",
    ))
    scenarios.append(_scenario(
        "control_k64_distinct_freeze_same_set",
        "known_transport_blind_control", "known_blind", 64,
        left, p_left, right, p_right,
        observation_mode="distinct_freeze_same_set",
    ))
    return scenarios


def build_scenarios(config, report):
    scenarios = reconstruct_small_hgp_scenarios(config, report)
    scenarios.extend(build_sparse_scenarios(config))
    if len({row["id"] for row in scenarios}) != len(scenarios):
        raise RuntimeError("duplicate delivery-gate scenario id")
    return scenarios


def diagnostic_masks(k, config):
    diagnostic = config["character_diagnostic"]
    if int(k) <= int(diagnostic["include_all_nonzero_when_k_at_most"]):
        return np.arange(1, 1 << int(k), dtype=np.uint64)
    values = [1 << index for index in range(int(k))]
    seen = set(values)
    counter = 0
    target = len(values) + int(diagnostic["nonbasis_mask_count"])
    while len(values) < target:
        payload = canonical([
            config["seed_namespace"], "character-diagnostic-mask", int(k), counter,
        ]).encode("ascii")
        candidate = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        candidate &= (1 << int(k)) - 1
        counter += 1
        if candidate and candidate not in seen:
            seen.add(candidate)
            values.append(candidate)
    return np.asarray([np.uint64(value) for value in values], dtype=np.uint64)


def scenario_public_metadata(scenario, config):
    masks = diagnostic_masks(scenario["k"], config)
    truth = {name: float(value) for name, value in sorted(scenario["truth"].items())}
    return {
        "character_diagnostic_mask_count": int(masks.size),
        "character_diagnostic_mask_sha256": sha256_bytes(
            canonical(labels_hex(masks)).encode("ascii")
        ),
        "classification": scenario["classification"],
        "id": scenario["id"],
        "k": int(scenario["k"]),
        "labels_left": labels_hex(scenario["labels_left"]),
        "labels_right": labels_hex(scenario["labels_right"]),
        "observation_mode": scenario["observation_mode"],
        "probabilities_left": scenario["probabilities_left"].tolist(),
        "probabilities_right": scenario["probabilities_right"].tolist(),
        "requested_q_top_delta": scenario["requested_q_top_delta"],
        "source": scenario["source"],
        "truth": truth,
        "uint64_bit63_exercised": bool(
            int(scenario["k"]) == 64
            and any(int(value) & (1 << 63) for value in np.concatenate((
                scenario["labels_left"], scenario["labels_right"], masks,
            )))
        ),
    }


def _sampling_support(scenario, config):
    mode = scenario["observation_mode"]
    if mode == "iid_target":
        return aligned_probabilities(
            scenario["labels_left"], scenario["probabilities_left"],
            scenario["labels_right"], scenario["probabilities_right"],
        )
    sparse = config["sparse_label_stress"]
    if mode == "common_freeze":
        label = parse_label(sparse["common_freeze_label"])
        return (
            np.asarray([label], dtype=np.uint64),
            np.ones(1, dtype=np.float64), np.ones(1, dtype=np.float64),
        )
    if mode == "distinct_freeze_same_set":
        return None, None, None
    raise RuntimeError("unknown observation mode")


def draw_trajectory_histograms(
        scenario, config, trials, trajectories, draws, seed_left, seed_right):
    labels, probabilities_left, probabilities_right = _sampling_support(
        scenario, config,
    )
    mode = scenario["observation_mode"]
    if mode == "iid_target":
        left_rng = np.random.default_rng(int(seed_left))
        right_rng = np.random.default_rng(int(seed_right))
        left = left_rng.multinomial(
            int(draws), probabilities_left, size=(int(trials), int(trajectories)),
        )
        right = right_rng.multinomial(
            int(draws), probabilities_right, size=(int(trials), int(trajectories)),
        )
    elif mode == "common_freeze":
        left = np.full((int(trials), int(trajectories), 1), int(draws), dtype=np.int64)
        right = left.copy()
    elif mode == "distinct_freeze_same_set":
        values = list(range(int(trajectories)))
        values[-2] = 1 << 63
        values[-1] = (1 << 64) - 1
        labels = np.asarray([np.uint64(value) for value in values], dtype=np.uint64)
        width = int(trajectories)
        left = np.zeros((int(trials), int(trajectories), width), dtype=np.int64)
        for trajectory in range(int(trajectories)):
            left[:, trajectory, trajectory] = int(draws)
        right = left.copy()
    else:
        raise RuntimeError("unknown observation mode")
    return labels, left, right


def _collision(histograms):
    trajectories = histograms.shape[1]
    summed = histograms.sum(axis=1)
    total_square = np.einsum("ts,ts->t", summed, summed, optimize=True)
    self_square = np.einsum("tis,tis->t", histograms, histograms, optimize=True)
    return (total_square - self_square) / (trajectories * (trajectories - 1))


def _metric_core(left, right, k):
    collision_left = _collision(left)
    collision_right = _collision(right)
    cross = np.einsum(
        "ts,ts->t", left.sum(axis=1), right.sum(axis=1), optimize=True,
    ) / (left.shape[1] * right.shape[1])
    q_left = normalize_collision(collision_left, k)
    q_right = normalize_collision(collision_right, k)
    d2 = normalize_d2(collision_left + collision_right - 2.0 * cross, k)
    return q_left, q_right, d2


def _groupwise_jackknife_se(delete_left, delete_right):
    variance = np.zeros(delete_left.shape[0], dtype=np.float64)
    for values in (delete_left, delete_right):
        centered = values - values.mean(axis=1, keepdims=True)
        variance += (values.shape[1] - 1.0) / values.shape[1] * np.sum(
            centered * centered, axis=1,
        )
    return np.sqrt(variance)


def estimate_trial_metrics(labels, left_counts, right_counts, k, masks):
    draws_left = left_counts.sum(axis=2, keepdims=True).astype(np.float64)
    draws_right = right_counts.sum(axis=2, keepdims=True).astype(np.float64)
    left = left_counts.astype(np.float64) / draws_left
    right = right_counts.astype(np.float64) / draws_right
    q_left, q_right, d2 = _metric_core(left, right, k)
    trajectories = left.shape[1]
    delete_q_left = []
    delete_q_right = []
    delete_d2_left = []
    delete_d2_right = []
    for omitted in range(trajectories):
        ql, qr, value = _metric_core(np.delete(left, omitted, axis=1), right, k)
        delete_q_left.append(ql - qr)
        delete_d2_left.append(value)
        ql, qr, value = _metric_core(left, np.delete(right, omitted, axis=1), k)
        delete_q_right.append(ql - qr)
        delete_d2_right.append(value)
    delete_q_left = np.stack(delete_q_left, axis=1)
    delete_q_right = np.stack(delete_q_right, axis=1)
    delete_d2_left = np.stack(delete_d2_left, axis=1)
    delete_d2_right = np.stack(delete_d2_right, axis=1)
    mean_left = left.mean(axis=1)
    mean_right = right.mean(axis=1)
    signs = 1.0 - 2.0 * (
        np.bitwise_count(
            np.asarray(labels, dtype=np.uint64)[:, None]
            & np.asarray(masks, dtype=np.uint64)[None, :]
        ) & np.uint8(1)
    ).astype(np.float64)
    character_delta = (mean_left - mean_right) @ signs
    metrics = {
        "character_max_abs_delta_diagnostic": np.max(
            np.abs(character_delta), axis=1,
        ),
        "delete_one_d2_left": delete_d2_left,
        "delete_one_d2_right": delete_d2_right,
        "delete_one_q_top_delta_left": delete_q_left,
        "delete_one_q_top_delta_right": delete_q_right,
        "d2_norm": d2,
        "d2_se": _groupwise_jackknife_se(delete_d2_left, delete_d2_right),
        "q_top_delta_abs": np.abs(q_left - q_right),
        "q_top_delta_se": _groupwise_jackknife_se(
            delete_q_left, delete_q_right,
        ),
        "q_top_delta_signed": q_left - q_right,
        "q_top_left": q_left,
        "q_top_right": q_right,
    }
    for name, values in metrics.items():
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"non-finite raw delivery metric: {name}")
    return metrics


def _scenario_seeds(config, config_sha, stage, trajectories, draws, scenario_id):
    common = (
        config["seed_namespace"], config_sha, stage, int(trajectories), int(draws),
        scenario_id,
    )
    return (
        derive_seed(*common, "family_A"),
        derive_seed(*common, "family_B"),
    )


def simulate_metrics(
        scenario, config, config_sha, stage, trajectories, draws, trials):
    seed_left, seed_right = _scenario_seeds(
        config, config_sha, stage, trajectories, draws, scenario["id"],
    )
    labels, left, right = draw_trajectory_histograms(
        scenario, config, trials, trajectories, draws, seed_left, seed_right,
    )
    metrics = estimate_trial_metrics(
        labels, left, right, scenario["k"],
        diagnostic_masks(scenario["k"], config),
    )
    return metrics, seed_left, seed_right, replay_receipt(labels, left, right, metrics)


def _standardized_error(estimate, truth, se):
    error = np.abs(np.asarray(estimate, dtype=np.float64) - float(truth))
    se = np.asarray(se, dtype=np.float64)
    values = np.zeros(error.shape, dtype=np.float64)
    positive = se > 0.0
    values[positive] = error[positive] / se[positive]
    nonfinite = (~positive) & (error > 2e-13)
    values[nonfinite] = np.inf
    return values


def higher_quantile_with_infinity(values, probability):
    values = np.sort(np.asarray(values, dtype=np.float64))
    if values.size == 0:
        raise RuntimeError("empty uncertainty calibration")
    index = int(math.ceil(float(probability) * values.size) - 1)
    selected = float(values[min(max(index, 0), values.size - 1)])
    return None if not math.isfinite(selected) else selected


def calibration_point(config, config_sha, scenarios, trajectories, draws):
    trials = int(config["replications"]["calibration_trials"])
    quantile_probability = float(
        config["uncertainty_calibration"]["higher_quantile"]
    )
    rows = []
    joint_by_scenario = []
    for scenario in scenarios:
        if scenario["classification"] == "known_blind":
            continue
        metrics, seed_left, seed_right, receipt = simulate_metrics(
            scenario, config, config_sha, "calibration", trajectories, draws, trials,
        )
        q_error = _standardized_error(
            metrics["q_top_delta_signed"],
            scenario["truth"]["q_top_delta_signed"], metrics["q_top_delta_se"],
        )
        d2_error = _standardized_error(
            metrics["d2_norm"], scenario["truth"]["d2_norm"], metrics["d2_se"],
        )
        joint = np.maximum(q_error, d2_error)
        joint_by_scenario.append(joint)
        rows.append({
            "joint_standardized_errors_sha256": digest_named_arrays(
                "calibration_joint_standardized_errors",
                {"values": (joint, "<f8")},
            ),
            "marginal_quantile_diagnostic": higher_quantile_with_infinity(
                joint, quantile_probability,
            ),
            "replay_receipt": receipt,
            "scenario_id": scenario["id"],
            "seed_left": int(seed_left),
            "seed_right": int(seed_right),
            "trial_count": trials,
        })
    outer_max = np.max(np.stack(joint_by_scenario, axis=0), axis=0)
    multiplier = higher_quantile_with_infinity(outer_max, quantile_probability)
    return {
        "draws_per_trajectory": int(draws),
        "multiplier": multiplier,
        "outer_max_standardized_errors_sha256": digest_named_arrays(
            "calibration_outer_max_standardized_errors",
            {"values": (outer_max, "<f8")},
        ),
        "quantile_probability": quantile_probability,
        "scenario_quantiles": rows,
        "trajectory_count": int(trajectories),
        "valid": multiplier is not None,
    }


def wilson_lower(successes, trials, confidence):
    successes = int(successes)
    trials = int(trials)
    if not 0 <= successes <= trials or trials <= 0:
        raise ValueError("invalid binomial count")
    z = NormalDist().inv_cdf(float(confidence))
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = p + z * z / (2.0 * trials)
    radius = z * math.sqrt(
        p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)
    )
    return max(0.0, (center - radius) / denominator)


def wilson_upper(successes, trials, confidence):
    return 1.0 - wilson_lower(int(trials) - int(successes), trials, confidence)


def decision_arrays(metrics, multiplier, config):
    multiplier = float(multiplier)
    rule = config["decision_rule"]
    q_delta = np.asarray(metrics["q_top_delta_abs"], dtype=np.float64)
    q_se = np.asarray(metrics["q_top_delta_se"], dtype=np.float64)
    d2 = np.asarray(metrics["d2_norm"], dtype=np.float64)
    d2_se = np.asarray(metrics["d2_se"], dtype=np.float64)
    q_upper = q_delta + multiplier * q_se
    q_lower = np.maximum(q_delta - multiplier * q_se, 0.0)
    d2_upper = np.maximum(d2, 0.0) + multiplier * d2_se
    d2_lower = np.maximum(d2 - multiplier * d2_se, 0.0)
    q_pass = q_upper <= float(rule["q_top_delta_tolerance"])
    q_fail = q_lower > float(rule["q_top_delta_tolerance"])
    d2_pass = d2_upper <= float(rule["d2_upper_tolerance"])
    d2_fail = d2_lower > float(rule["d2_upper_tolerance"])
    passed = q_pass & d2_pass
    failed = q_fail | d2_fail
    if np.any(passed & failed):
        raise RuntimeError("delivery-gate PASS and FAIL overlap")
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
    # One base-rate claim for every non-boundary row, a second false-PASS
    # claim for every bad row, and one simultaneous-coverage claim.
    count = len(base) + len(bad) + 1
    if count != 139:
        raise RuntimeError("FAIL hypotheses per stage point changed")
    return count


def fail_hypothesis_count(config, scenarios):
    count = len(ordered_operating_points(config)) * 2 * fail_hypotheses_per_stage_point(
        scenarios,
    )
    if count != 1390:
        raise RuntimeError("full-protocol FAIL hypothesis count changed")
    return count


def fail_adjusted_confidence(config, scenarios):
    family_confidence = float(config["decision_rule"]["fail_familywise_confidence"])
    return 1.0 - (1.0 - family_confidence) / fail_hypothesis_count(config, scenarios)


def _rate_fields(values, pass_confidence, fail_confidence, prefix):
    values = np.asarray(values, dtype=bool)
    count = int(np.count_nonzero(values))
    trials = int(values.size)
    return {
        f"{prefix}_count": count,
        f"{prefix}_rate": float(count / trials),
        f"{prefix}_wilson_lower": float(wilson_lower(count, trials, pass_confidence)),
        f"{prefix}_wilson_upper": float(wilson_upper(count, trials, pass_confidence)),
        f"{prefix}_fail_adjusted_wilson_lower": float(
            wilson_lower(count, trials, fail_confidence)
        ),
        f"{prefix}_fail_adjusted_wilson_upper": float(
            wilson_upper(count, trials, fail_confidence)
        ),
    }


def rate_verdict(lower, upper, target):
    if float(lower) >= float(target):
        return "PASS"
    if float(upper) < float(target):
        return "FAIL"
    return "INCONCLUSIVE"


def evaluate_row(
        scenario, config, config_sha, stage, trajectories, draws, multiplier,
        scenarios):
    trials = int(config["replications"][f"{stage}_trials"])
    metrics, seed_left, seed_right, receipt = simulate_metrics(
        scenario, config, config_sha, stage, trajectories, draws, trials,
    )
    decisions = decision_arrays(metrics, multiplier, config)
    pass_confidence = float(config["decision_rule"]["binomial_lower_confidence"])
    fail_confidence = fail_adjusted_confidence(config, scenarios)
    fields = {"evaluation_trials": trials}
    for name in (
            "pass", "fail", "inconclusive", "q_top_pass", "q_top_fail",
            "q_top_inconclusive", "d2_pass", "d2_fail", "d2_inconclusive"):
        fields.update(_rate_fields(
            decisions[name], pass_confidence, fail_confidence, f"candidate_{name}",
        ))

    coverage_applicable = scenario["classification"] != "known_blind"
    if coverage_applicable:
        q_covered = np.abs(
            metrics["q_top_delta_signed"]
            - scenario["truth"]["q_top_delta_signed"]
        ) <= float(multiplier) * metrics["q_top_delta_se"] + 2e-13
        d2_covered = np.abs(
            metrics["d2_norm"] - scenario["truth"]["d2_norm"]
        ) <= float(multiplier) * metrics["d2_se"] + 2e-13
        fields.update(_rate_fields(
            q_covered & d2_covered, pass_confidence, fail_confidence,
            "joint_coverage",
        ))
    else:
        fields.update({
            "joint_coverage_count": None,
            "joint_coverage_rate": None,
            "joint_coverage_wilson_upper": None,
            "joint_coverage_wilson_lower": None,
            "joint_coverage_fail_adjusted_wilson_upper": None,
            "joint_coverage_fail_adjusted_wilson_lower": None,
        })
    return {
        **fields,
        "character_diagnostic_participates_in_decision": False,
        "classification": scenario["classification"],
        "coverage_applicable": coverage_applicable,
        "draws_per_trajectory": int(draws),
        "expected_interpretation": (
            "EXPECTED_KNOWN_BLIND"
            if scenario["classification"] == "known_blind"
            else "IID_DELIVERY_GATE_CALIBRATION"
        ),
        "multiplier": float(multiplier),
        "replay_receipt": receipt,
        "scenario_id": scenario["id"],
        "seed_left": int(seed_left),
        "seed_right": int(seed_right),
        "stage": stage,
        "trajectory_count": int(trajectories),
        "true_d2_norm": float(scenario["truth"]["d2_norm"]),
        "true_q_top_delta_signed": float(
            scenario["truth"]["q_top_delta_signed"]
        ),
        "_raw_trial_metrics": metrics,
    }


def compact_evaluation_row(row):
    result = dict(row)
    result.pop("_raw_trial_metrics")
    return result


def summarize_point(rows, config, trajectories, draws, multiplier, stage):
    rule = config["decision_rule"]
    groups = {
        "null": [
            row for row in rows if row["classification"] in {"null", "d2_null"}
        ],
        "good": [
            row for row in rows
            if row["classification"] in {"good_q_top", "good_d2"}
        ],
        "bad_q_top": [row for row in rows if row["classification"] == "bad_q_top"],
        "bad_d2": [row for row in rows if row["classification"] == "bad_d2"],
        "coverage": [row for row in rows if row["coverage_applicable"]],
        "known_blind": [row for row in rows if row["classification"] == "known_blind"],
    }
    if any(not values for values in groups.values()):
        raise RuntimeError("delivery-gate requirement group is empty")

    def rowwise_requirement(name, subset, rate_prefix, target):
        lower = min(row[f"{rate_prefix}_wilson_lower"] for row in subset)
        upper = min(
            row[f"{rate_prefix}_fail_adjusted_wilson_upper"] for row in subset
        )
        return {
            "fail_adjusted_worst_wilson_upper": float(upper),
            "name": name,
            "row_count": len(subset),
            "target_rate": float(target),
            "pass_worst_wilson_lower": float(lower),
            "verdict": rate_verdict(lower, upper, target),
        }

    def maximum_rate_requirement(name, subset, rate_prefix, limit):
        lower = max(
            row[f"{rate_prefix}_fail_adjusted_wilson_lower"] for row in subset
        )
        upper = max(row[f"{rate_prefix}_wilson_upper"] for row in subset)
        verdict = (
            "PASS" if upper <= float(limit)
            else "FAIL" if lower > float(limit)
            else "INCONCLUSIVE"
        )
        return {
            "limit_rate": float(limit),
            "name": name,
            "row_count": len(subset),
            "verdict": verdict,
            "fail_adjusted_worst_wilson_lower": float(lower),
            "pass_worst_wilson_upper": float(upper),
        }

    requirements = [
        rowwise_requirement(
            "null_pass_power", groups["null"], "candidate_pass",
            rule["null_pass_minimum"],
        ),
        rowwise_requirement(
            "good_alternative_pass_power", groups["good"], "candidate_pass",
            rule["good_alternative_pass_minimum"],
        ),
        rowwise_requirement(
            "bad_q_top_fail_power", groups["bad_q_top"], "candidate_q_top_fail",
            rule["fail_power_minimum"],
        ),
        rowwise_requirement(
            "bad_d2_fail_power", groups["bad_d2"], "candidate_d2_fail",
            rule["fail_power_minimum"],
        ),
        maximum_rate_requirement(
            "bad_q_top_false_pass_control", groups["bad_q_top"],
            "candidate_q_top_pass", rule["bad_false_pass_maximum"],
        ),
        maximum_rate_requirement(
            "bad_d2_false_pass_control", groups["bad_d2"],
            "candidate_d2_pass", rule["bad_false_pass_maximum"],
        ),
        rowwise_requirement(
            "known_blind_expected_pass", groups["known_blind"], "candidate_pass",
            rule["known_blind_pass_minimum"],
        ),
    ]
    simultaneous = None
    for row in groups["coverage"]:
        raw = row["_raw_trial_metrics"]
        covered = (
            np.abs(
                np.asarray(raw["q_top_delta_signed"], dtype=np.float64)
                - float(row["true_q_top_delta_signed"])
            ) <= float(multiplier) * np.asarray(
                raw["q_top_delta_se"], dtype=np.float64,
            ) + 2e-13
        ) & (
            np.abs(
                np.asarray(raw["d2_norm"], dtype=np.float64)
                - float(row["true_d2_norm"])
            ) <= float(multiplier) * np.asarray(
                raw["d2_se"], dtype=np.float64,
            ) + 2e-13
        )
        simultaneous = covered if simultaneous is None else simultaneous & covered
    confidence = float(rule["binomial_lower_confidence"])
    fail_confidence = fail_adjusted_confidence(config, rows)
    coverage_count = int(np.count_nonzero(simultaneous))
    coverage_trials = int(simultaneous.size)
    coverage_lower = wilson_lower(coverage_count, coverage_trials, confidence)
    coverage_upper = wilson_upper(coverage_count, coverage_trials, confidence)
    coverage_fail_upper = wilson_upper(
        coverage_count, coverage_trials, fail_confidence,
    )
    requirements.append({
        "name": "simultaneous_registered_interval_coverage",
        "row_count": len(groups["coverage"]),
        "success_count": coverage_count,
        "trial_count": coverage_trials,
        "target_rate": float(rule["interval_coverage_minimum"]),
        "fail_adjusted_worst_wilson_upper": float(coverage_fail_upper),
        "pass_worst_wilson_lower": float(coverage_lower),
        "verdict": rate_verdict(
            coverage_lower, coverage_fail_upper,
            rule["interval_coverage_minimum"],
        ),
    })
    verdicts = {row["verdict"] for row in requirements}
    if verdicts == {"PASS"}:
        decision = "PASS"
    elif "FAIL" in verdicts:
        decision = "FAIL"
    else:
        decision = "INCONCLUSIVE"
    known_blind_preserved = all(
        row["expected_interpretation"] == "EXPECTED_KNOWN_BLIND"
        for row in groups["known_blind"]
    )
    return {
        "cost_independent_draws_per_side": int(trajectories) * int(draws),
        "decision": decision,
        "draws_per_trajectory": int(draws),
        "eligible": decision == "PASS",
        "known_blind_controls_preserved": bool(known_blind_preserved),
        "fail_adjusted_confidence": float(fail_confidence),
        "fail_hypothesis_count": fail_hypothesis_count(config, rows),
        "multiplier": float(multiplier),
        "requirements": requirements,
        "row_count": len(rows),
        "stage": stage,
        "trajectory_count": int(trajectories),
    }


def evaluate_point(
        scenarios, config, config_sha, stage, trajectories, draws, multiplier):
    rows = [
        evaluate_row(
            scenario, config, config_sha, stage, trajectories, draws, multiplier,
            scenarios,
        )
        for scenario in scenarios
    ]
    point = summarize_point(
        rows, config, trajectories, draws, multiplier, stage,
    )
    return [compact_evaluation_row(row) for row in rows], point


def point_identity(point):
    if point is None:
        return None
    return (
        int(point["trajectory_count"]), int(point["draws_per_trajectory"]),
        float(point["multiplier"]),
    )


def confirmation_matches_selection(selected, confirmation):
    return bool(
        selected is not None and confirmation is not None
        and confirmation["eligible"]
        and point_identity(selected) == point_identity(confirmation)
    )


def invalid_calibration_selection_point(trajectories, draws):
    return {
        "decision": "INCONCLUSIVE",
        "draws_per_trajectory": int(draws),
        "eligible": False,
        "invalid_reason": "NO_FINITE_CALIBRATED_MULTIPLIER",
        "multiplier": None,
        "stage": "selection",
        "trajectory_count": int(trajectories),
    }


def terminal_status_for(selection_points, selected, confirmation):
    if confirmation_matches_selection(selected, confirmation):
        return "LOCAL_DELIVERY_GATE_COMMON_OPERATING_POINT_CONFIRMED"
    if confirmation is not None:
        return (
            "SELECTED_POINT_CONFIRMATION_FAILED_REDESIGN_REQUIRED"
            if confirmation["decision"] == "FAIL"
            else "DELIVERY_GATE_CALIBRATION_INCONCLUSIVE"
        )
    if selection_points and all(
            point["decision"] == "FAIL" for point in selection_points):
        return "DELIVERY_GATE_REDESIGN_REQUIRED"
    return "DELIVERY_GATE_CALIBRATION_INCONCLUSIVE"


def main():
    if OUTPUT_PATH.exists():
        raise RuntimeError("delivery-gate report already exists")
    config = load_json_strict(CONFIG_PATH)
    provenance = verify_committed_clean_source(config)
    bound_report = verify_bound_validation_062(config)
    scenarios = build_scenarios(config, bound_report)
    scenario_registry = [scenario_public_metadata(row, config) for row in scenarios]
    config_sha = sha256_file(CONFIG_PATH)

    calibration_points = []
    selection_points = []
    selection_rows = []
    selected = None
    for trajectories, draws in ordered_operating_points(config):
        calibration = calibration_point(
            config, config_sha, scenarios, trajectories, draws,
        )
        calibration_points.append(calibration)
        if not calibration["valid"]:
            selection_points.append(invalid_calibration_selection_point(
                trajectories, draws,
            ))
            continue
        rows, point = evaluate_point(
            scenarios, config, config_sha, "selection", trajectories, draws,
            calibration["multiplier"],
        )
        selection_rows.extend(rows)
        selection_points.append(point)
        if point["eligible"]:
            selected = point
            break

    confirmation_rows = []
    confirmation_point = None
    if selected is not None:
        confirmation_rows, confirmation_point = evaluate_point(
            scenarios, config, config_sha, "confirmation",
            selected["trajectory_count"], selected["draws_per_trajectory"],
            selected["multiplier"],
        )
    terminal_status = terminal_status_for(
        selection_points, selected, confirmation_point,
    )
    core = {
        "authority": config["authority"],
        "bound_validation_062": {
            "audit_sha256": config["bound_validation_062"]["audit"]["audit_sha256"],
            "report_sha256": bound_report["report_sha256"],
            "source_commit": bound_report["source_commit"],
        },
        "calibration_points": calibration_points,
        "confirmation_point": confirmation_point,
        "confirmation_rows": confirmation_rows,
        "config_sha256": config_sha,
        "estimator_contract": {
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
        },
        "known_blind_interpretation": {
            "adversarial_initialization_still_required": True,
            "direct_full_labels_certify_unvisited_target_mass": False,
            "orthogonal_confirmation_still_required": True,
            "status": "EXPECTED_KNOWN_BLIND",
            "transport_gates_still_required": True,
        },
        "multiplicity_contract": {
            "fail_adjusted_confidence": fail_adjusted_confidence(
                config, scenarios
            ),
            "fail_familywise_confidence": config["decision_rule"][
                "fail_familywise_confidence"
            ],
            "fail_hypothesis_count": fail_hypothesis_count(
                config, scenarios
            ),
            "pass_confidence": config["decision_rule"][
                "binomial_lower_confidence"
            ],
            "fail_hypotheses_per_stage_point": fail_hypotheses_per_stage_point(
                scenarios
            ),
            "scope": config["decision_rule"]["fail_multiplicity_scope"],
            "strict_familywise_guarantee": False,
            "wilson_role": "OPERATIONAL_ONLY",
        },
        "project_transition_status": "BLOCKED_BEFORE_REMOTE",
        "remaining_blockers": [
            "LARGE_K_ORTHOGONAL_CONFIRMER_PORTFOLIO_UNFROZEN",
            "FUTURE_SCHEMA_RUNTIME_COVERAGE_INCOMPLETE",
            "CAMPAIGN_BUDGET_UNAPPROVED",
            "STAGE3_MULTI_COMPARISON_MULTIPLICITY_UNFROZEN",
        ],
        "rng_replay_contract": config["rng_replay_contract"],
        "replay_receipt_contract": {
            "maximum_report_bytes": config["report_contract"]["maximum_bytes"],
            "persistent_raw": False,
            "schema": REPLAY_RECEIPT_SCHEMA,
            "seed_replay_required": True,
        },
        "scenario_registry": scenario_registry,
        "selected_common_operating_point": selected,
        "selection_points": selection_points,
        "selection_rows": selection_rows,
        "source_commit": provenance["source_commit"],
        "source_file_count": provenance["source_file_count"],
        "source_tree_sha256": provenance["source_tree_sha256"],
        "status": terminal_status,
        "version": config["version"],
    }
    core["report_sha256"] = sha256_bytes(canonical(core).encode("ascii"))
    encoded = (canonical(core) + "\n").encode("ascii")
    if len(encoded) > int(config["report_contract"]["maximum_bytes"]):
        raise RuntimeError("delivery-gate report exceeds frozen size limit")
    with OUTPUT_PATH.open("x", encoding="ascii") as handle:
        handle.write(encoded.decode("ascii"))
    print(json.dumps({
        "report_sha256": core["report_sha256"],
        "selected_common_operating_point": selected,
        "status": core["status"],
    }, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
