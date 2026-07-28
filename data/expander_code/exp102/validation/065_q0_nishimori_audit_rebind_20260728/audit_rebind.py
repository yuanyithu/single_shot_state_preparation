#!/usr/bin/env python3
"""Fresh numerical audit rebind for the immutable validation-063 report."""

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
OUTPUT_PATH = ROOT / "audit_rebind_report.json"
AUDIT_VERSION = "exp102.q0_nishimori_audit_rebind.conflict.v1"
CONFLICT_STATUS = "CONFLICT_INDEPENDENT_NUMERICAL_RECOMPUTATION_MAP_TIE_SEMANTICS"
OLD_REPORT_VERSION = "exp102.q0_nishimori_auxiliary_calibration.v2"
OLD_MAXIMUM_STATUS = "NISHIMORI_AUXILIARY_AUDIT_CALIBRATED_WITH_KNOWN_BLIND_CONTROLS"
OLD_AUTHORITY = {
    "formal_authorization": False,
    "maximum_status": OLD_MAXIMUM_STATUS,
    "posterior_estimation": False,
    "production_authorization": False,
    "remote_authorization": False,
    "sole_confirmer_authorization": False,
}
FAILURE_RE = re.compile(
    r"^(equivalence gate failed|equivalence power failed): "
    r"([^/]+)/([^/]+)/([^/]+)/([^/]+)$"
)


class AuditError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def canonical(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_constant(token: str) -> None:
    raise AuditError(f"non-finite JSON constant: {token}")


def load_json_bytes(value: bytes, name: str) -> dict[str, Any]:
    try:
        result = json.loads(
            value.decode("ascii"), parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuditError(f"invalid ASCII JSON: {name}") from exc
    require(isinstance(result, dict), f"JSON root is not an object: {name}")
    return result


def load_json(path: Path) -> dict[str, Any]:
    return load_json_bytes(path.read_bytes(), str(path))


def verify_self_hash(payload: Mapping[str, Any], field: str) -> str:
    require(field in payload, f"missing self-hash field: {field}")
    unsigned = {key: value for key, value in payload.items() if key != field}
    actual = sha256_bytes(canonical(unsigned).encode("ascii"))
    require(payload[field] == actual, f"self hash changed: {field}")
    return actual


def _git(*args: str, text: bool = True, check: bool = True):
    result = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), *args], check=False,
        capture_output=True, text=text,
    )
    if check and result.returncode != 0:
        raise AuditError(f"git command failed: {' '.join(args)}")
    return result.stdout


def git_blob(commit: str, relative: str) -> bytes:
    return _git("show", f"{commit}:{relative}", text=False)


def project_path(relative: str) -> Path:
    candidate = Path(relative)
    require(not candidate.is_absolute() and ".." not in candidate.parts, "unsafe bound path")
    resolved = (PROJECT_ROOT / candidate).resolve()
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise AuditError("bound path escapes repository") from exc
    require(not resolved.is_symlink(), f"bound path is a symlink: {relative}")
    return resolved


def verify_source_freeze(config: Mapping[str, Any]) -> dict[str, Any]:
    require(config.get("version") == "exp102.q0_nishimori_audit_rebind.config.v1", "config version changed")
    require(config.get("authority") and not any(config["authority"].values()), "065 authority is not all false")
    status = [line for line in _git("status", "--porcelain=v1", "--untracked-files=all").splitlines() if line]
    require(not status, f"065 one-shot source is not clean: {status[0] if status else ''}")
    offenders = sorted(
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in PROJECT_ROOT.rglob("*")
        if path.name == "__pycache__" or (path.is_file() and path.suffix in {".pyc", ".pyo"})
    )
    require(not offenders, f"Python bytecode present: {offenders[0] if offenders else ''}")
    head = _git("rev-parse", "HEAD").strip()
    require(re.fullmatch(r"[0-9a-f]{40}", head) is not None, "invalid audit source commit")
    config_relative = CONFIG_PATH.relative_to(PROJECT_ROOT).as_posix()
    _git("ls-files", "--error-unmatch", "--", config_relative)
    require(git_blob(head, config_relative) == CONFIG_PATH.read_bytes(), "committed 065 config bytes changed")
    bound = config["implementation"]["bound_files"]
    for role, descriptor in sorted(bound.items()):
        path = project_path(descriptor["path"])
        require(path.is_file(), f"missing 065 bound source: {role}")
        require(sha256_file(path) == descriptor["sha256"], f"065 bound source SHA changed: {role}")
        _git("ls-files", "--error-unmatch", "--", descriptor["path"])
        require(git_blob(head, descriptor["path"]) == path.read_bytes(), f"committed 065 source changed: {role}")
    core = {
        "bound_files": bound,
        "config_sha256": sha256_file(CONFIG_PATH),
        "source_commit": head,
    }
    return {
        **core,
        "source_tree_sha256": sha256_bytes(canonical(core).encode("ascii")),
    }


def verify_original_evidence(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    spec = config["input_063"]
    report_path = project_path(spec["report_path"])
    conflict_path = project_path(spec["conflict_evidence_path"])
    require(sha256_file(report_path) == spec["report_file_sha256"], "063 report file SHA changed")
    require(sha256_file(conflict_path) == spec["conflict_evidence_file_sha256"], "063 conflict file SHA changed")
    report = load_json(report_path)
    conflict = load_json(conflict_path)
    require(verify_self_hash(report, "report_sha256") == spec["report_self_sha256"], "063 report identity changed")
    verify_self_hash(conflict, "evidence_sha256")
    commit = spec["source_commit"]
    require(_git("cat-file", "-t", commit).strip() == "commit", "063 source commit missing")
    require(subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "merge-base", "--is-ancestor", commit, "HEAD"],
        check=False, capture_output=True,
    ).returncode == 0, "063 source commit is not an ancestor")
    config_blob = git_blob(commit, spec["config_path"])
    require(sha256_bytes(config_blob) == spec["config_sha256"], "063 config blob SHA changed")
    old_config = load_json_bytes(config_blob, "063 source config blob")
    require(report.get("source_commit") == commit, "063 report source commit changed")
    require(report.get("config_sha256") == spec["config_sha256"], "063 report config SHA changed")
    require(report.get("source_tree_sha256") == spec["source_tree_sha256"], "063 report source-tree SHA changed")
    require(report.get("report_sha256") == spec["report_self_sha256"], "063 report self identity changed")
    require(report.get("schema_sha256") == spec["schema_sha256"], "063 schema identity changed")
    require(report.get("version") == OLD_REPORT_VERSION, "063 report version changed")
    require(report.get("status") == spec["expected_scientific_status"], "063 scientific status changed")
    require(report.get("authority") == OLD_AUTHORITY, "063 report authority changed")
    require(report.get("universal_q_top_bias_bound_from_identity") is None, "063 report invented a q_top bound")
    bound = old_config["implementation"]["bound_files"]
    require(report.get("bound_files") == bound, "063 report bound-file catalog changed")
    for role, descriptor in sorted(bound.items()):
        blob = git_blob(commit, descriptor["path"])
        require(sha256_bytes(blob) == descriptor["sha256"], f"063 historical source blob changed: {role}")
    tree_core = {
        "bound_files": bound,
        "config_sha256": spec["config_sha256"],
        "source_commit": commit,
    }
    require(sha256_bytes(canonical(tree_core).encode("ascii")) == spec["source_tree_sha256"], "063 source-tree binding changed")
    require(report.get("runner_sha256") == bound["runner"]["sha256"], "063 runner binding changed")
    require(sha256_bytes(git_blob(commit, bound["raw_schema"]["path"])) == spec["schema_sha256"], "063 schema blob changed")
    report_in_source = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "cat-file", "-e", f"{commit}:{spec['report_path']}"],
        check=False, capture_output=True,
    ).returncode == 0
    require(not report_in_source, "063 report unexpectedly exists in its source commit")
    require(conflict.get("status") == "CONFLICT_INDEPENDENT_AUDIT_MESSAGE_TAXONOMY_MISMATCH", "063 conflict status changed")
    require(conflict.get("report_file_sha256") == spec["report_file_sha256"], "063 conflict report file binding changed")
    require(conflict.get("report_sha256") == spec["report_self_sha256"], "063 conflict report identity changed")
    require(conflict.get("source_commit") == commit, "063 conflict source binding changed")
    require(conflict.get("independent_audit_created") is False, "063 conflict falsely claims an audit output")
    require(conflict.get("authority") and not any(conflict["authority"].values()), "063 conflict authority changed")
    return report, old_config, conflict


def assert_nested_close(actual: Any, expected: Any, path: str, tolerance: float) -> None:
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), f"mapping keys changed at {path}")
        for key in expected:
            assert_nested_close(actual[key], expected[key], f"{path}.{key}", tolerance)
        return
    if isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), f"list shape changed at {path}")
        for index, value in enumerate(expected):
            assert_nested_close(actual[index], value, f"{path}[{index}]", tolerance)
        return
    if isinstance(expected, float):
        require(not isinstance(actual, bool) and isinstance(actual, (int, float)), f"non-numeric value at {path}")
        require(math.isfinite(float(actual)), f"nonfinite value at {path}")
        require(math.isclose(float(actual), expected, rel_tol=tolerance, abs_tol=tolerance), f"numeric value changed at {path}")
        return
    require(actual == expected, f"value changed at {path}")


def collect_numeric_mismatches(
    actual: Any, expected: Any, path: str, tolerance: float,
    output: list[dict[str, Any]],
) -> None:
    """Collect float disagreements while keeping all non-float identity strict."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), f"mapping keys changed at {path}")
        for key in expected:
            collect_numeric_mismatches(actual[key], expected[key], f"{path}.{key}", tolerance, output)
        return
    if isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), f"list shape changed at {path}")
        for index, value in enumerate(expected):
            collect_numeric_mismatches(actual[index], value, f"{path}[{index}]", tolerance, output)
        return
    if isinstance(expected, float):
        require(not isinstance(actual, bool) and isinstance(actual, (int, float)), f"non-numeric value at {path}")
        require(math.isfinite(float(actual)), f"nonfinite value at {path}")
        if not math.isclose(float(actual), expected, rel_tol=tolerance, abs_tol=tolerance):
            output.append({
                "absolute_difference": abs(float(actual) - expected),
                "oracle_value": expected,
                "path": path,
                "report_value": float(actual),
            })
        return
    require(actual == expected, f"value changed at {path}")


def _p_token(value: Any) -> str:
    number = float(value)
    require(math.isfinite(number), "nonfinite p in failure identity")
    return canonical(number)


def _failure_id(model_id: str, p: Any, control: str, group: str) -> str:
    return f"EXACT_CONTROL_GATE|{model_id}|{_p_token(p)}|{control}|{group}"


def rebuild_structured_failures(
    config: Mapping[str, Any], exact_rows: list[dict[str, Any]],
    power_rows: list[dict[str, Any]], chain: Mapping[str, Any],
) -> list[dict[str, Any]]:
    gate = config["calibration_gate"]
    gate_size = int(gate["power_gate_ensemble_size"])
    tolerance = float(gate["exact_tolerance"])
    effect_floor = float(gate["detected_effect_floor"])
    min_detection = float(gate["minimum_detection_rate"])
    min_equivalence = float(gate["minimum_equivalence_rate"])
    power_index: dict[tuple[str, float, str], dict[str, Any]] = {}
    for row in power_rows:
        key = (str(row["model_id"]), float(row["p"]), str(row["control"]))
        require(key not in power_index, f"duplicate power row: {key}")
        power_index[key] = row
    failures: list[dict[str, Any]] = []
    for row in exact_rows:
        model_id, p, control = str(row["model_id"]), float(row["p"]), str(row["control"])
        expectation = config["controls"]["exact"][control]["expected_power_outcome"]
        power = power_index.get((model_id, p, control))
        require(power is not None, f"missing power payload: {model_id}/{p}/{control}")
        selected = [item for item in power["rows"] if int(item["ensemble_size"]) == gate_size]
        require(len(selected) == 1, f"missing or duplicate gate row: {model_id}/{p}/{control}")
        for group, outcome in expectation.items():
            stat = selected[0]["statistics"][group]
            exact_group = row["group_exact_metrics"][group]
            if not stat["applicable"]:
                require(group == "nonbasis_max" and int(row["k"]) == 1, "unexpected non-applicable group")
                continue
            exact_value = abs(float(exact_group["exact_effect"])) if group == "omnibus" else float(exact_group["max_abs_exact_effect"])
            reasons: list[str] = []
            observed: dict[str, float] = {"absolute_exact_effect": exact_value}
            thresholds: dict[str, float] = {}
            if outcome == "equivalent":
                rate = float(stat["diagnostic_equivalence_pass_rate"])
                observed["diagnostic_equivalence_pass_rate"] = rate
                thresholds = {"maximum_exact_effect": tolerance, "minimum_equivalence_rate": min_equivalence}
                if rate < min_equivalence:
                    reasons.append("EQUIVALENCE_RATE_BELOW_MINIMUM")
                if exact_value > tolerance:
                    reasons.append("EQUIVALENT_EXACT_EFFECT_ABOVE_TOLERANCE")
            elif outcome == "detected":
                rate = float(stat["equality_rejection_rate"])
                observed["equality_rejection_rate"] = rate
                thresholds = {"minimum_detection_rate": min_detection, "minimum_exact_effect": effect_floor}
                if rate < min_detection:
                    reasons.append("DETECTION_RATE_BELOW_MINIMUM")
                if exact_value < effect_floor:
                    reasons.append("DETECTED_EXACT_EFFECT_BELOW_FLOOR")
            else:
                raise AuditError(f"unknown expected outcome: {outcome!r}")
            if reasons:
                failures.append({
                    "character_group": group,
                    "control": control,
                    "ensemble_size": gate_size,
                    "expected_outcome": outcome,
                    "failure_id": _failure_id(model_id, p, control, group),
                    "model_id": model_id,
                    "observed": observed,
                    "p": p,
                    "reason_codes": reasons,
                    "scope": "EXACT_CONTROL_GATE",
                    "thresholds": thresholds,
                })
    # These controls are part of the numeric gate even though the immutable
    # report currently has no chain-level failure.
    common = chain["common_planted_freeze"]
    four = chain["four_distinct_label_freeze"]
    two = chain["two_label_equal_moment_counterexample"]
    require(abs(float(common["scalar_identity_difference"])) <= tolerance, "numeric chain gate failed: common scalar")
    require(max(abs(float(value)) for value in common["per_character_identity_difference"]) <= tolerance, "numeric chain gate failed: common character")
    require(abs(float(four["scalar_identity_difference"])) <= tolerance, "numeric chain gate failed: four scalar")
    require(abs(float(four["group_exact_metrics"]["omnibus"]["exact_effect"])) <= tolerance, "numeric chain gate failed: four omnibus")
    for group in ("basis_max", "nonbasis_max"):
        require(float(four["group_exact_metrics"][group]["max_abs_exact_effect"]) >= effect_floor, f"numeric chain gate failed: four {group}")
    require(abs(float(two["scalar_identity_difference"])) <= tolerance, "numeric chain gate failed: two scalar")
    require(abs(float(two["target_q_top"]) - float(two["candidate_q_top"])) >= 0.5, "numeric chain gate failed: two q_top")
    failures.sort(key=lambda item: item["failure_id"])
    require(len({item["failure_id"] for item in failures}) == len(failures), "duplicate structured failure identity")
    return failures


def parse_legacy_failures(values: list[Any], aliases: Mapping[str, str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        require(isinstance(value, str), "legacy failure is not text")
        match = FAILURE_RE.fullmatch(value)
        require(match is not None, f"unrecognized legacy failure grammar: {value!r}")
        prefix, model_id, p_token, control, group = match.groups()
        require(aliases.get(prefix) == "EXACT_CONTROL_GATE", f"unbound legacy prefix: {prefix}")
        failure_id = _failure_id(model_id, p_token, control, group)
        require(failure_id not in parsed, f"duplicate legacy failure identity: {failure_id}")
        parsed[failure_id] = prefix
    return parsed


def load_frozen_oracle(config: Mapping[str, Any]):
    spec = config["input_063"]
    path = project_path(spec["independent_oracle_path"])
    require(sha256_file(path) == spec["independent_oracle_sha256"], "current frozen oracle bytes changed")
    require(sha256_bytes(git_blob(spec["source_commit"], spec["independent_oracle_path"])) == spec["independent_oracle_sha256"], "historical frozen oracle blob changed")
    module_spec = importlib.util.spec_from_file_location("exp102_v065_frozen_oracle", path)
    require(module_spec is not None and module_spec.loader is not None, "cannot load frozen oracle")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def build_map_tie_diagnostics(
    oracle: Any, old_config: Mapping[str, Any], report: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> list[dict[str, Any]]:
    model_specs = {item["id"]: item for item in old_config["exact_models"]}
    diagnostics: list[dict[str, Any]] = []
    for report_row, oracle_row in zip(report["golden_rows"], expected["golden_rows"]):
        require(report_row["model_id"] == oracle_row["model_id"] and float(report_row["p"]) == float(oracle_row["p"]), "golden-row order changed")
        report_post = oracle.np.asarray(report_row["logical_posteriors"], dtype=oracle.np.float64)
        oracle_post = oracle.np.asarray(oracle_row["logical_posteriors"], dtype=oracle.np.float64)
        report_argmax = oracle.np.argmax(report_post, axis=1)
        oracle_argmax = oracle.np.argmax(oracle_post, axis=1)
        differing = oracle.np.flatnonzero(report_argmax != oracle_argmax)
        if differing.size == 0:
            continue
        spec = model_specs[report_row["model_id"]]
        model, frame = oracle._build_model(spec["H"])
        states = oracle._all_states(model.num_qubits)
        syndromes = [oracle._syndrome_bytes(model.H_check, state) for state in states]
        labels = [oracle._label_integer(frame, state) for state in states]
        weights = states.sum(axis=1)
        syndrome_keys = sorted(set(syndromes))
        for raw_index in differing:
            y_index = int(raw_index)
            key = syndrome_keys[y_index]
            report_label = int(report_argmax[y_index])
            oracle_label = int(oracle_argmax[y_index])
            supports = [index for index, syndrome in enumerate(syndromes) if syndrome == key]
            enumerators: dict[str, list[list[int]]] = {}
            canonical_states: dict[str, dict[str, Any]] = {}
            for label in (report_label, oracle_label):
                counts: dict[int, int] = {}
                candidates = []
                for state_index in supports:
                    if int(labels[state_index]) != label:
                        continue
                    weight = int(weights[state_index])
                    counts[weight] = counts.get(weight, 0) + 1
                    candidates.append((weight, int(state_index)))
                require(candidates, "empty tied-label support")
                minimum_weight, state_index = min(candidates)
                enumerators[str(label)] = [[weight, counts[weight]] for weight in sorted(counts)]
                canonical_states[str(label)] = {
                    "hamming_weight": minimum_weight,
                    "physical_state_index": state_index,
                    "state_bits_little_index_order": "".join(str(int(bit)) for bit in states[state_index]),
                }
            require(enumerators[str(report_label)] == enumerators[str(oracle_label)], "argmax mismatch is not an exact weight-enumerator tie")
            diagnostics.append({
                "canonical_minimum_physical_states": canonical_states,
                "logical_weight_enumerators": enumerators,
                "mathematical_weight_enumerator_tie": True,
                "model_id": report_row["model_id"],
                "oracle_choice": {
                    "floating_label_advantage_over_other": float(
                        oracle_post[y_index, oracle_label] - oracle_post[y_index, report_label]
                    ),
                    "label": oracle_label,
                    "posterior_mass": float(oracle_post[y_index, oracle_label]),
                    "posterior_mass_at_report_label": float(oracle_post[y_index, report_label]),
                },
                "p": float(report_row["p"]),
                "report_choice": {
                    "floating_label_advantage_over_other": float(
                        report_post[y_index, report_label] - report_post[y_index, oracle_label]
                    ),
                    "label": report_label,
                    "posterior_mass": float(report_post[y_index, report_label]),
                    "posterior_mass_at_oracle_label": float(report_post[y_index, oracle_label]),
                },
                "syndrome_hex": key.hex(),
                "y_index": y_index,
            })
    diagnostics.sort(key=lambda item: (item["model_id"], item["p"], item["syndrome_hex"]))
    return diagnostics


def recompute_and_compare(
    config: Mapping[str, Any], report: Mapping[str, Any], old_config: Mapping[str, Any],
) -> tuple[
    dict[str, Any], list[dict[str, Any]], dict[str, str], dict[str, str],
    list[dict[str, Any]], list[dict[str, Any]],
]:
    oracle = load_frozen_oracle(config)
    expected = oracle.build_oracle_calibration(old_config, include_power=True)
    tolerance = float(config["numeric_comparison_tolerance"])
    mismatches: list[dict[str, Any]] = []
    for field in ("chain_level_control_metrics", "exact_control_rows", "golden_rows", "power_rows"):
        collect_numeric_mismatches(report[field], expected[field], field, tolerance, mismatches)
    for field in ("passed", "power_is_optimistic_no_sampler_noise", "universal_q_top_bias_bound"):
        require(report["calibration_gate"][field] == expected["calibration_gate"][field], f"calibration gate metadata changed: {field}")
    report_structured = rebuild_structured_failures(
        old_config, report["exact_control_rows"], report["power_rows"],
        report["chain_level_control_metrics"],
    )
    oracle_structured = rebuild_structured_failures(
        old_config, expected["exact_control_rows"], expected["power_rows"],
        expected["chain_level_control_metrics"],
    )
    assert_nested_close(report_structured, oracle_structured, "structured_failures", tolerance)
    aliases = config["legacy_failure_aliases"]
    report_legacy = parse_legacy_failures(report["calibration_gate"]["failures"], aliases)
    oracle_legacy = parse_legacy_failures(expected["calibration_gate"]["failures"], aliases)
    numeric_ids = {item["failure_id"] for item in oracle_structured}
    require(set(report_legacy) == numeric_ids, "report legacy failures do not equal numeric failures")
    require(set(oracle_legacy) == numeric_ids, "oracle legacy failures do not equal numeric failures")
    ties = build_map_tie_diagnostics(oracle, old_config, report, expected)
    return expected, oracle_structured, report_legacy, oracle_legacy, mismatches, ties


def main() -> None:
    require(not OUTPUT_PATH.exists(), "065 audit output already exists")
    config = load_json(CONFIG_PATH)
    source = verify_source_freeze(config)
    report, old_config, conflict = verify_original_evidence(config)
    expected, failures, report_legacy, oracle_legacy, mismatches, ties = recompute_and_compare(config, report, old_config)
    require(len(failures) == int(config["input_063"]["expected_calibration_failure_count"]), "structured failure count changed")
    require(all(item["reason_codes"] == ["EQUIVALENCE_RATE_BELOW_MINIMUM"] for item in failures), "unexpected numeric failure reason")
    require(len(mismatches) == int(config["input_063"]["expected_full_payload_mismatch_count"]), "full-payload mismatch count changed")
    require(len(ties) == int(config["input_063"]["expected_map_tie_case_count"]), "MAP-tie count changed")
    maximum_mismatch = max(item["absolute_difference"] for item in mismatches)
    require(math.isclose(maximum_mismatch, float(config["input_063"]["expected_maximum_absolute_mismatch"]), rel_tol=0.0, abs_tol=1e-15), "maximum payload mismatch changed")
    for item in mismatches:
        match = re.match(r"^exact_control_rows\[(\d+)\]\.", item["path"])
        require(match is not None, "unexpected mismatch path family")
        require(report["exact_control_rows"][int(match.group(1))]["control"] == "truth_blind_map_delta", "mismatch escaped MAP-delta control")
    core = {
        "audit_source": source,
        "authority": dict(config["authority"]),
        "config_sha256": sha256_file(CONFIG_PATH),
        "grammar_rebind": {
            "numeric_failure_ids_sha256": sha256_bytes(canonical([item["failure_id"] for item in failures]).encode("ascii")),
            "oracle_legacy_prefixes": sorted(set(oracle_legacy.values())),
            "report_legacy_prefixes": sorted(set(report_legacy.values())),
            "structured_failure_count": len(failures),
            "structured_failures": failures,
        },
        "independent_numerical_conflict": {
            "full_payload_match": False,
            "map_tie_case_count": len(ties),
            "map_tie_cases": ties,
            "maximum_absolute_mismatch": maximum_mismatch,
            "mismatch_count": len(mismatches),
            "mismatches": mismatches,
            "terminal_gate_invariant": True,
        },
        "input_063": {
            "conflict_evidence_file_sha256": sha256_file(project_path(config["input_063"]["conflict_evidence_path"])),
            "conflict_evidence_sha256": conflict["evidence_sha256"],
            "original_audit_created": False,
            "report_file_sha256": sha256_file(project_path(config["input_063"]["report_path"])),
            "report_sha256": report["report_sha256"],
            "scientific_status": report["status"],
            "source_commit": report["source_commit"],
            "source_tree_sha256": report["source_tree_sha256"],
        },
        "numerical_recomputation": {
            "chain_control_count": len(expected["chain_level_control_metrics"]),
            "exact_control_row_count": len(expected["exact_control_rows"]),
            "golden_row_count": len(expected["golden_rows"]),
            "independent_oracle_sha256": config["input_063"]["independent_oracle_sha256"],
            "independent_oracle_version": expected["oracle_version"],
            "numeric_comparison_tolerance": config["numeric_comparison_tolerance"],
            "power_row_count": len(expected["power_rows"]),
            "report_payload_recomputed": True,
        },
        "original_exact_report_edited": False,
        "original_exact_report_rerun": False,
        "status": CONFLICT_STATUS,
        "version": AUDIT_VERSION,
    }
    core["audit_sha256"] = sha256_bytes(canonical(core).encode("ascii"))
    with OUTPUT_PATH.open("x", encoding="ascii") as handle:
        handle.write(canonical(core) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(core, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
