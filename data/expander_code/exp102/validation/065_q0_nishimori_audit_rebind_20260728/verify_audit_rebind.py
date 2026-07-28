#!/usr/bin/env python3
"""Independent verifier for the validation-065 audit-rebind output.

This file intentionally imports neither ``audit_rebind`` nor any 063 runner or
auditor code.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[4]
CONFIG_PATH = ROOT / "rebind_config.json"
AUDIT_PATH = ROOT / "audit_rebind_report.json"
OUTPUT_PATH = ROOT / "independent_verification.json"
AUDIT_VERSION = "exp102.q0_nishimori_audit_rebind.conflict.v1"
VERIFY_VERSION = "exp102.q0_nishimori_audit_rebind.independent_verification.v1"
AUDIT_STATUS = "CONFLICT_INDEPENDENT_NUMERICAL_RECOMPUTATION_MAP_TIE_SEMANTICS"
VERIFY_STATUS = "INDEPENDENT_VERIFICATION_PASS_OF_RECORDED_MAP_TIE_CONFLICT"
LEGACY_RE = re.compile(
    r"^(equivalence gate failed|equivalence power failed): "
    r"([^/]+)/([^/]+)/([^/]+)/([^/]+)$"
)


class VerificationError(RuntimeError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    )


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def bytes_sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def strict_json_bytes(value: bytes, label: str) -> dict[str, Any]:
    def reject(token: str) -> None:
        raise VerificationError(f"non-finite JSON constant in {label}: {token}")

    try:
        result = json.loads(value.decode("ascii"), parse_constant=reject)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"invalid ASCII JSON: {label}") from exc
    need(isinstance(result, dict), f"JSON root is not an object: {label}")
    return result


def strict_json(path: Path) -> dict[str, Any]:
    return strict_json_bytes(path.read_bytes(), str(path))


def check_self_hash(value: Mapping[str, Any], field: str) -> str:
    need(field in value, f"missing {field}")
    unsigned = {key: item for key, item in value.items() if key != field}
    actual = bytes_sha(canonical_json(unsigned).encode("ascii"))
    need(value[field] == actual, f"{field} mismatch")
    return actual


def git(*arguments: str, binary: bool = False, check: bool = True):
    result = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), *arguments], check=False,
        capture_output=True, text=not binary,
    )
    if check and result.returncode != 0:
        raise VerificationError(f"git verification failed: {' '.join(arguments)}")
    return result.stdout


def source_blob(commit: str, relative: str) -> bytes:
    return git("show", f"{commit}:{relative}", binary=True)


def local_path(relative: str) -> Path:
    part = Path(relative)
    need(not part.is_absolute() and ".." not in part.parts, "unsafe evidence path")
    result = (PROJECT_ROOT / part).resolve()
    try:
        result.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise VerificationError("evidence path escapes repository") from exc
    need(not result.is_symlink(), f"evidence path is a symlink: {relative}")
    return result


def verify_verifier_source(config: Mapping[str, Any]) -> dict[str, Any]:
    allowed = f"?? {AUDIT_PATH.relative_to(PROJECT_ROOT).as_posix()}"
    lines = set(filter(None, git("status", "--porcelain=v1", "--untracked-files=all").splitlines()))
    need(lines == {allowed}, "verifier requires exactly one untracked audit output")
    bytecode = [
        path for path in PROJECT_ROOT.rglob("*")
        if path.name == "__pycache__" or (path.is_file() and path.suffix in {".pyc", ".pyo"})
    ]
    need(not bytecode, "verifier source contains Python bytecode")
    head = git("rev-parse", "HEAD").strip()
    config_relative = CONFIG_PATH.relative_to(PROJECT_ROOT).as_posix()
    git("ls-files", "--error-unmatch", "--", config_relative)
    need(source_blob(head, config_relative) == CONFIG_PATH.read_bytes(), "committed rebind config changed")
    for role, item in sorted(config["implementation"]["bound_files"].items()):
        path = local_path(item["path"])
        need(file_sha(path) == item["sha256"], f"bound verifier source changed: {role}")
        git("ls-files", "--error-unmatch", "--", item["path"])
        need(source_blob(head, item["path"]) == path.read_bytes(), f"committed verifier source changed: {role}")
    core = {
        "bound_files": config["implementation"]["bound_files"],
        "config_sha256": file_sha(CONFIG_PATH),
        "source_commit": head,
    }
    return {**core, "source_tree_sha256": bytes_sha(canonical_json(core).encode("ascii"))}


def compare(actual: Any, expected: Any, location: str, tolerance: float) -> None:
    if isinstance(expected, dict):
        need(isinstance(actual, dict) and set(actual) == set(expected), f"keys changed at {location}")
        for key, value in expected.items():
            compare(actual[key], value, f"{location}.{key}", tolerance)
        return
    if isinstance(expected, list):
        need(isinstance(actual, list) and len(actual) == len(expected), f"shape changed at {location}")
        for index, value in enumerate(expected):
            compare(actual[index], value, f"{location}[{index}]", tolerance)
        return
    if isinstance(expected, float):
        need(not isinstance(actual, bool) and isinstance(actual, (int, float)), f"non-numeric at {location}")
        need(math.isfinite(float(actual)), f"nonfinite at {location}")
        need(math.isclose(float(actual), expected, rel_tol=tolerance, abs_tol=tolerance), f"numeric mismatch at {location}")
        return
    need(actual == expected, f"value mismatch at {location}")


def find_float_mismatches(
    actual: Any, expected: Any, location: str, tolerance: float,
    result: list[dict[str, Any]],
) -> None:
    if isinstance(expected, dict):
        need(isinstance(actual, dict) and set(actual) == set(expected), f"keys changed at {location}")
        for key, value in expected.items():
            find_float_mismatches(actual[key], value, f"{location}.{key}", tolerance, result)
        return
    if isinstance(expected, list):
        need(isinstance(actual, list) and len(actual) == len(expected), f"shape changed at {location}")
        for index, value in enumerate(expected):
            find_float_mismatches(actual[index], value, f"{location}[{index}]", tolerance, result)
        return
    if isinstance(expected, float):
        need(not isinstance(actual, bool) and isinstance(actual, (int, float)), f"non-numeric at {location}")
        need(math.isfinite(float(actual)), f"nonfinite at {location}")
        if not math.isclose(float(actual), expected, rel_tol=tolerance, abs_tol=tolerance):
            result.append({
                "absolute_difference": abs(float(actual) - expected),
                "oracle_value": expected,
                "path": location,
                "report_value": float(actual),
            })
        return
    need(actual == expected, f"value mismatch at {location}")


def p_text(value: Any) -> str:
    number = float(value)
    need(math.isfinite(number), "nonfinite p")
    return canonical_json(number)


def failure_id(model: str, p: Any, control: str, group: str) -> str:
    return f"EXACT_CONTROL_GATE|{model}|{p_text(p)}|{control}|{group}"


def independent_failures(
    old_config: Mapping[str, Any], exact: list[dict[str, Any]],
    powers: list[dict[str, Any]], chains: Mapping[str, Any],
) -> list[dict[str, Any]]:
    gate = old_config["calibration_gate"]
    size = int(gate["power_gate_ensemble_size"])
    exact_limit = float(gate["exact_tolerance"])
    effect_floor = float(gate["detected_effect_floor"])
    equivalence_min = float(gate["minimum_equivalence_rate"])
    detection_min = float(gate["minimum_detection_rate"])
    power_map: dict[tuple[str, float, str], dict[str, Any]] = {}
    for item in powers:
        key = (str(item["model_id"]), float(item["p"]), str(item["control"]))
        need(key not in power_map, "duplicate independent power key")
        power_map[key] = item
    result: list[dict[str, Any]] = []
    for row in exact:
        model = str(row["model_id"])
        p = float(row["p"])
        control = str(row["control"])
        power = power_map[(model, p, control)]
        selected = [entry for entry in power["rows"] if int(entry["ensemble_size"]) == size]
        need(len(selected) == 1, "independent gate row is not unique")
        outcomes = old_config["controls"]["exact"][control]["expected_power_outcome"]
        for group, outcome in outcomes.items():
            stat = selected[0]["statistics"][group]
            if not stat["applicable"]:
                need(group == "nonbasis_max" and int(row["k"]) == 1, "unexpected independent NA")
                continue
            metric = row["group_exact_metrics"][group]
            exact_value = abs(float(metric["exact_effect"])) if group == "omnibus" else float(metric["max_abs_exact_effect"])
            reasons: list[str] = []
            observed: dict[str, float] = {"absolute_exact_effect": exact_value}
            if outcome == "equivalent":
                rate = float(stat["diagnostic_equivalence_pass_rate"])
                observed["diagnostic_equivalence_pass_rate"] = rate
                thresholds = {"maximum_exact_effect": exact_limit, "minimum_equivalence_rate": equivalence_min}
                if rate < equivalence_min:
                    reasons.append("EQUIVALENCE_RATE_BELOW_MINIMUM")
                if exact_value > exact_limit:
                    reasons.append("EQUIVALENT_EXACT_EFFECT_ABOVE_TOLERANCE")
            elif outcome == "detected":
                rate = float(stat["equality_rejection_rate"])
                observed["equality_rejection_rate"] = rate
                thresholds = {"minimum_detection_rate": detection_min, "minimum_exact_effect": effect_floor}
                if rate < detection_min:
                    reasons.append("DETECTION_RATE_BELOW_MINIMUM")
                if exact_value < effect_floor:
                    reasons.append("DETECTED_EXACT_EFFECT_BELOW_FLOOR")
            else:
                raise VerificationError("unknown independent outcome")
            if reasons:
                result.append({
                    "character_group": group,
                    "control": control,
                    "ensemble_size": size,
                    "expected_outcome": outcome,
                    "failure_id": failure_id(model, p, control, group),
                    "model_id": model,
                    "observed": observed,
                    "p": p,
                    "reason_codes": reasons,
                    "scope": "EXACT_CONTROL_GATE",
                    "thresholds": thresholds,
                })
    common = chains["common_planted_freeze"]
    four = chains["four_distinct_label_freeze"]
    two = chains["two_label_equal_moment_counterexample"]
    need(abs(float(common["scalar_identity_difference"])) <= exact_limit, "independent common scalar control failed")
    need(max(abs(float(x)) for x in common["per_character_identity_difference"]) <= exact_limit, "independent common character control failed")
    need(abs(float(four["scalar_identity_difference"])) <= exact_limit, "independent four scalar control failed")
    need(abs(float(four["group_exact_metrics"]["omnibus"]["exact_effect"])) <= exact_limit, "independent four omnibus control failed")
    need(all(float(four["group_exact_metrics"][name]["max_abs_exact_effect"]) >= effect_floor for name in ("basis_max", "nonbasis_max")), "independent sparse-freeze control failed")
    need(abs(float(two["scalar_identity_difference"])) <= exact_limit, "independent two-label identity failed")
    need(abs(float(two["target_q_top"]) - float(two["candidate_q_top"])) >= 0.5, "independent two-label q_top control failed")
    result.sort(key=lambda row: row["failure_id"])
    need(len(result) == len({row["failure_id"] for row in result}), "duplicate independent failure")
    return result


def legacy_ids(messages: list[Any], aliases: Mapping[str, str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for message in messages:
        need(isinstance(message, str), "legacy message is not text")
        match = LEGACY_RE.fullmatch(message)
        need(match is not None, "unknown legacy grammar")
        prefix, model, p, control, group = match.groups()
        need(aliases.get(prefix) == "EXACT_CONTROL_GATE", "legacy alias changed")
        key = failure_id(model, p, control, group)
        need(key not in result, "duplicate legacy identity")
        result[key] = prefix
    return result


def frozen_oracle(config: Mapping[str, Any]):
    item = config["input_063"]
    path = local_path(item["independent_oracle_path"])
    need(file_sha(path) == item["independent_oracle_sha256"], "frozen oracle SHA changed")
    spec = importlib.util.spec_from_file_location("exp102_v065_verifier_oracle", path)
    need(spec is not None and spec.loader is not None, "cannot load verifier oracle")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def independent_tie_witnesses(
    oracle: Any, old_config: Mapping[str, Any], report: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> list[dict[str, Any]]:
    specifications = {row["id"]: row for row in old_config["exact_models"]}
    witnesses: list[dict[str, Any]] = []
    for left, right in zip(report["golden_rows"], expected["golden_rows"]):
        need(left["model_id"] == right["model_id"] and float(left["p"]) == float(right["p"]), "verifier golden order changed")
        report_distribution = oracle.np.asarray(left["logical_posteriors"], dtype=oracle.np.float64)
        oracle_distribution = oracle.np.asarray(right["logical_posteriors"], dtype=oracle.np.float64)
        report_choices = oracle.np.argmax(report_distribution, axis=1)
        oracle_choices = oracle.np.argmax(oracle_distribution, axis=1)
        changed = oracle.np.flatnonzero(report_choices != oracle_choices)
        if changed.size == 0:
            continue
        model, frame = oracle._build_model(specifications[left["model_id"]]["H"])
        states = oracle._all_states(model.num_qubits)
        syndromes = [oracle._syndrome_bytes(model.H_check, state) for state in states]
        labels = [oracle._label_integer(frame, state) for state in states]
        weights = states.sum(axis=1)
        syndrome_keys = sorted(set(syndromes))
        for value in changed:
            y_index = int(value)
            syndrome = syndrome_keys[y_index]
            report_label = int(report_choices[y_index])
            oracle_label = int(oracle_choices[y_index])
            support = [index for index, item in enumerate(syndromes) if item == syndrome]
            enumerators: dict[str, list[list[int]]] = {}
            minimum_states: dict[str, dict[str, Any]] = {}
            for label in (report_label, oracle_label):
                counts: dict[int, int] = {}
                candidates = []
                for index in support:
                    if int(labels[index]) == label:
                        weight = int(weights[index])
                        counts[weight] = counts.get(weight, 0) + 1
                        candidates.append((weight, index))
                need(candidates, "verifier tied label has empty support")
                minimum_weight, state_index = min(candidates)
                enumerators[str(label)] = [[weight, counts[weight]] for weight in sorted(counts)]
                minimum_states[str(label)] = {
                    "hamming_weight": int(minimum_weight),
                    "physical_state_index": int(state_index),
                    "state_bits_little_index_order": "".join(str(int(bit)) for bit in states[state_index]),
                }
            need(enumerators[str(report_label)] == enumerators[str(oracle_label)], "verifier found a non-tie argmax mismatch")
            witnesses.append({
                "canonical_minimum_physical_states": minimum_states,
                "logical_weight_enumerators": enumerators,
                "mathematical_weight_enumerator_tie": True,
                "model_id": left["model_id"],
                "oracle_choice": {
                    "floating_label_advantage_over_other": float(
                        oracle_distribution[y_index, oracle_label]
                        - oracle_distribution[y_index, report_label]
                    ),
                    "label": oracle_label,
                    "posterior_mass": float(oracle_distribution[y_index, oracle_label]),
                    "posterior_mass_at_report_label": float(oracle_distribution[y_index, report_label]),
                },
                "p": float(left["p"]),
                "report_choice": {
                    "floating_label_advantage_over_other": float(
                        report_distribution[y_index, report_label]
                        - report_distribution[y_index, oracle_label]
                    ),
                    "label": report_label,
                    "posterior_mass": float(report_distribution[y_index, report_label]),
                    "posterior_mass_at_oracle_label": float(report_distribution[y_index, oracle_label]),
                },
                "syndrome_hex": syndrome.hex(),
                "y_index": y_index,
            })
    witnesses.sort(key=lambda row: (row["model_id"], row["p"], row["syndrome_hex"]))
    return witnesses


def verify_inputs_and_recompute(config: Mapping[str, Any], audit: Mapping[str, Any]):
    item = config["input_063"]
    report_path = local_path(item["report_path"])
    conflict_path = local_path(item["conflict_evidence_path"])
    need(file_sha(report_path) == item["report_file_sha256"], "verifier report SHA changed")
    need(file_sha(conflict_path) == item["conflict_evidence_file_sha256"], "verifier conflict SHA changed")
    report = strict_json(report_path)
    conflict = strict_json(conflict_path)
    need(check_self_hash(report, "report_sha256") == item["report_self_sha256"], "verifier report self hash changed")
    check_self_hash(conflict, "evidence_sha256")
    commit = item["source_commit"]
    old_config_blob = source_blob(commit, item["config_path"])
    need(bytes_sha(old_config_blob) == item["config_sha256"], "verifier historical config blob changed")
    old_config = strict_json_bytes(old_config_blob, "historical config")
    need(report["bound_files"] == old_config["implementation"]["bound_files"], "verifier bound catalog changed")
    for role, descriptor in sorted(report["bound_files"].items()):
        need(bytes_sha(source_blob(commit, descriptor["path"])) == descriptor["sha256"], f"verifier historical blob changed: {role}")
    tree = {
        "bound_files": report["bound_files"],
        "config_sha256": item["config_sha256"],
        "source_commit": commit,
    }
    need(bytes_sha(canonical_json(tree).encode("ascii")) == item["source_tree_sha256"] == report["source_tree_sha256"], "verifier historical source tree changed")
    need(report["status"] == item["expected_scientific_status"], "verifier scientific status changed")
    need(report["universal_q_top_bias_bound_from_identity"] is None, "verifier found an invented q_top bound")
    need(conflict["independent_audit_created"] is False, "verifier conflict output boundary changed")
    oracle = frozen_oracle(config)
    expected = oracle.build_oracle_calibration(old_config, include_power=True)
    tolerance = float(config["numeric_comparison_tolerance"])
    mismatches: list[dict[str, Any]] = []
    for field in ("chain_level_control_metrics", "exact_control_rows", "golden_rows", "power_rows"):
        find_float_mismatches(report[field], expected[field], field, tolerance, mismatches)
    failures = independent_failures(
        old_config, expected["exact_control_rows"], expected["power_rows"],
        expected["chain_level_control_metrics"],
    )
    report_ids = legacy_ids(report["calibration_gate"]["failures"], config["legacy_failure_aliases"])
    oracle_ids = legacy_ids(expected["calibration_gate"]["failures"], config["legacy_failure_aliases"])
    numeric_ids = {row["failure_id"] for row in failures}
    need(set(report_ids) == set(oracle_ids) == numeric_ids, "verifier grammar rebind changed")
    compare(audit["grammar_rebind"]["structured_failures"], failures, "audit.structured_failures", tolerance)
    ties = independent_tie_witnesses(oracle, old_config, report, expected)
    conflict_payload = audit["independent_numerical_conflict"]
    need(conflict_payload["full_payload_match"] is False, "audit hides full-payload mismatch")
    need(conflict_payload["terminal_gate_invariant"] is True, "audit terminal-gate invariant changed")
    compare(conflict_payload["mismatches"], mismatches, "audit.mismatches", tolerance)
    compare(conflict_payload["map_tie_cases"], ties, "audit.map_tie_cases", tolerance)
    return report, conflict, expected, failures, mismatches, ties


def main() -> None:
    need(AUDIT_PATH.is_file(), "audit output is missing")
    need(not OUTPUT_PATH.exists(), "independent verification already exists")
    config = strict_json(CONFIG_PATH)
    need(config.get("version") == "exp102.q0_nishimori_audit_rebind.config.v1", "verifier config version changed")
    need(config.get("authority") and not any(config["authority"].values()), "verifier authority is not all false")
    source = verify_verifier_source(config)
    audit = strict_json(AUDIT_PATH)
    check_self_hash(audit, "audit_sha256")
    need(audit.get("version") == AUDIT_VERSION and audit.get("status") == AUDIT_STATUS, "audit terminal identity changed")
    need(audit.get("authority") == config["authority"], "audit authority changed")
    need(audit.get("audit_source") == source, "audit source binding changed")
    need(audit.get("original_exact_report_edited") is False, "audit claims original edit")
    need(audit.get("original_exact_report_rerun") is False, "audit claims original rerun")
    report, conflict, expected, failures, mismatches, ties = verify_inputs_and_recompute(config, audit)
    need(len(failures) == int(config["input_063"]["expected_calibration_failure_count"]), "verifier failure count changed")
    need(len(mismatches) == int(config["input_063"]["expected_full_payload_mismatch_count"]), "verifier mismatch count changed")
    need(len(ties) == int(config["input_063"]["expected_map_tie_case_count"]), "verifier tie count changed")
    core = {
        "audit_file_sha256": file_sha(AUDIT_PATH),
        "audit_sha256": audit["audit_sha256"],
        "authority": dict(config["authority"]),
        "config_sha256": file_sha(CONFIG_PATH),
        "independent_oracle_sha256": config["input_063"]["independent_oracle_sha256"],
        "input_conflict_evidence_sha256": conflict["evidence_sha256"],
        "input_report_sha256": report["report_sha256"],
        "numerical_payload_recomputed": True,
        "original_scientific_status": report["status"],
        "recorded_conflict_status": audit["status"],
        "status": VERIFY_STATUS,
        "structured_failure_count": len(failures),
        "structured_failures_sha256": bytes_sha(canonical_json(failures).encode("ascii")),
        "verified_full_payload_match": False,
        "verified_map_tie_case_count": len(ties),
        "verified_mismatch_count": len(mismatches),
        "verified_terminal_gate_invariant": True,
        "verifier_source": source,
        "version": VERIFY_VERSION,
    }
    core["verification_sha256"] = bytes_sha(canonical_json(core).encode("ascii"))
    with OUTPUT_PATH.open("x", encoding="ascii") as handle:
        handle.write(canonical_json(core) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(core, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
