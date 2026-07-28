"""Independently reconstruct every decision in a completed 062 report."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from statistics import NormalDist
import subprocess


ROOT = Path(__file__).resolve().parent
EXP102_ROOT = ROOT.parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[5]
CONFIG_PATH = ROOT / "calibration_config.json"
REPORT_PATH = ROOT / "calibration_report.json"
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


def derive_seed(namespace, *parts):
    payload = canonical([namespace, *parts]).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def verify_self_hash(payload, field):
    expected = payload[field]
    unsigned = dict(payload)
    unsigned.pop(field)
    actual = sha256_bytes(canonical(unsigned).encode("ascii"))
    if actual != expected:
        raise RuntimeError(f"self hash changed: {field}")


def _git(args, *, text=True):
    return subprocess.run(
        ["git", *args], cwd=PROJECT_ROOT, check=True,
        capture_output=True, text=text,
    ).stdout


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


def reject_validation_bytecode(root=ROOT):
    root = Path(root)
    offenders = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
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
        raise RuntimeError("report does not bind the exact current source commit")
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
    expected_tree = sha256_bytes(canonical(rows).encode("ascii"))
    if report["source_file_count"] != len(rows):
        raise RuntimeError("report source-file count changed")
    if report["source_tree_sha256"] != expected_tree:
        raise RuntimeError("report source-tree hash changed")


def wilson_lower(successes, trials, confidence):
    successes = int(successes)
    trials = int(trials)
    if not 0 <= successes <= trials or trials <= 0:
        raise RuntimeError("invalid binomial count")
    z = NormalDist().inv_cdf(float(confidence))
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = p + z * z / (2.0 * trials)
    radius = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials))
    return max(0.0, (center - radius) / denominator)


def assert_float(actual, expected, message, tolerance=1e-14):
    if not math.isfinite(float(actual)) or abs(float(actual) - float(expected)) > tolerance:
        raise RuntimeError(message)


def vector_sha256(values):
    return sha256_bytes(canonical([float(value) for value in values]).encode("ascii"))


def vector_summary(values):
    values = [float(value) for value in values]
    return {
        "maximum": max(values),
        "mean_square": sum(value * value for value in values) / len(values),
        "minimum": min(values),
        "nonzero_count": sum(abs(value) > 1e-14 for value in values),
    }


def verify_rate_fields(row, config):
    trials = int(row["evaluation_trials"])
    expected_trials = int(config["replications"][f"{row['stage']}_trials"])
    if trials != expected_trials:
        raise RuntimeError("row uses the wrong fresh-stage replication count")
    confidence = float(config["candidate_rule"]["binomial_lower_confidence"])
    names = (
        "candidate_pass", "candidate_fail", "candidate_inconclusive",
        "coverage", "historical_pass",
    )
    for name in names:
        count = int(row[f"{name}_count"])
        if not 0 <= count <= trials:
            raise RuntimeError(f"invalid row count: {name}")
        assert_float(row[f"{name}_rate"], count / trials, f"row rate changed: {name}")
        assert_float(
            row[f"{name}_wilson_lower"],
            wilson_lower(count, trials, confidence),
            f"row Wilson lower bound changed: {name}",
        )
    if sum(int(row[f"candidate_{name}_count"]) for name in ("pass", "fail", "inconclusive")) != trials:
        raise RuntimeError("candidate decisions do not partition the trials")
    historical_fail = trials - int(row["historical_pass_count"])
    if row["historical_fail_count"] != historical_fail:
        raise RuntimeError("historical fail count changed")
    assert_float(
        row["historical_fail_rate"], historical_fail / trials,
        "historical fail rate changed",
    )


def verify_exact_distribution(row):
    size = int(row["catalog_size"])
    if size > 15 or not row["physical_distribution"]:
        raise RuntimeError("exact row has the wrong catalog size or physical flag")
    names = (
        "base_character_means", "shifted_character_means",
        "true_character_shifts",
    )
    if any(len(row[name]) != size for name in names):
        raise RuntimeError("exact row did not retain its complete character vectors")
    shifts = [
        float(left) - float(right)
        for left, right in zip(
            row["base_character_means"], row["shifted_character_means"],
        )
    ]
    for actual, expected in zip(row["true_character_shifts"], shifts):
        assert_float(actual, expected, "exact true shift vector changed")
    assert_float(
        row["true_max_abs_character_shift"], max(abs(value) for value in shifts),
        "exact maximum true shift changed",
    )
    left_purity = sum(float(value) ** 2 for value in row["base_character_means"]) / size
    right_purity = sum(float(value) ** 2 for value in row["shifted_character_means"]) / size
    assert_float(row["true_character_purity_left"], left_purity, "exact left purity changed")
    assert_float(row["true_character_purity_right"], right_purity, "exact right purity changed")
    assert_float(
        row["true_character_purity_delta"], abs(left_purity - right_purity),
        "exact purity delta changed",
    )
    logical_size = len(row["base_logical_means"])
    if logical_size == 0 or len(row["shifted_logical_means"]) != logical_size:
        raise RuntimeError("exact row lost retained conditional logical signs")
    left_q_top = sum(float(value) ** 2 for value in row["base_logical_means"]) / logical_size
    right_q_top = sum(float(value) ** 2 for value in row["shifted_logical_means"]) / logical_size
    assert_float(row["true_q_top_left"], left_q_top, "exact left q_top changed")
    assert_float(row["true_q_top_right"], right_q_top, "exact right q_top changed")
    assert_float(row["true_q_top_delta"], abs(left_q_top - right_q_top), "exact q_top delta changed")
    selected = int(row["selected_character_index"])
    if not 0 <= selected < size:
        raise RuntimeError("exact selected character is invalid")
    assert_float(
        abs(shifts[selected]), row["requested_shift"],
        "exact requested shift was not realized", tolerance=2e-12,
    )
    if row["catalog"] == "logical":
        if row["q_top_accounting"] != "exact_full_logical_catalog":
            raise RuntimeError("logical q_top accounting label changed")
        assert_float(left_purity, left_q_top, "complete logical purity is not q_top")
        assert_float(right_purity, right_q_top, "shifted logical purity is not q_top")
    elif row["catalog"] == "collapsed_B":
        if row["q_top_accounting"] != "induced_from_retained_conditional_logical_signs":
            raise RuntimeError("collapsed-B q_top accounting label changed")
    else:
        raise RuntimeError("unknown exact catalog")


def synthetic_vector(row):
    size = int(row["catalog_size"])
    base = float(row["base_character_mean"])
    direction = -1.0 if base >= 0.0 else 1.0
    shifted = [base] * size
    if row["shift_pattern"] == "single_character":
        shifted[0] += direction * float(row["requested_shift"])
        if row["selected_character_index"] != 0:
            raise RuntimeError("synthetic single-character index changed")
    elif row["shift_pattern"] == "distributed_all_characters":
        shifted = [value + direction * float(row["requested_shift"]) for value in shifted]
        if row["selected_character_index"] is not None:
            raise RuntimeError("distributed synthetic row selected one character")
    else:
        raise RuntimeError("unknown synthetic shift pattern")
    return [base - value for value in shifted], shifted


def verify_synthetic_distribution(row, catalog):
    if row["physical_distribution"]:
        raise RuntimeError("synthetic stress was mislabeled as physical")
    if row["q_top_accounting"] != "not_defined_for_nonphysical_synthetic_stress":
        raise RuntimeError("synthetic stress was assigned q_top authority")
    forbidden = {
        "base_character_means", "shifted_character_means",
        "true_character_shifts", "base_logical_means", "shifted_logical_means",
    }
    if forbidden.intersection(row):
        raise RuntimeError("synthetic row redundantly stored a full vector")
    shifts, shifted = synthetic_vector(row)
    expected_summary = vector_summary(shifts)
    if row["true_character_shift_summary"]["nonzero_count"] != expected_summary[
        "nonzero_count"
    ]:
        raise RuntimeError("synthetic true-shift nonzero count changed")
    for name in ("minimum", "maximum", "mean_square"):
        assert_float(
            row["true_character_shift_summary"][name], expected_summary[name],
            f"synthetic true-shift summary changed: {name}",
        )
    if row["true_character_shift_vector_sha256"] != vector_sha256(shifts):
        raise RuntimeError("synthetic true-shift vector hash changed")
    assert_float(
        row["true_max_abs_character_shift"], max(abs(value) for value in shifts),
        "synthetic maximum true shift changed",
    )
    left_purity = float(row["base_character_mean"]) ** 2
    right_purity = sum(value * value for value in shifted) / len(shifted)
    assert_float(row["true_character_purity_left"], left_purity, "synthetic left purity changed")
    assert_float(row["true_character_purity_right"], right_purity, "synthetic right purity changed")
    assert_float(
        row["true_character_purity_delta"], abs(left_purity - right_purity),
        "synthetic purity delta changed",
    )
    assert_float(
        row["shifted_character_mean_minimum"], min(shifted),
        "synthetic shifted minimum changed",
    )
    assert_float(
        row["shifted_character_mean_maximum"], max(shifted),
        "synthetic shifted maximum changed",
    )
    if any(row[name] is not None for name in (
        "true_q_top_delta", "true_q_top_left", "true_q_top_right",
    )):
        raise RuntimeError("synthetic stress contains a q_top value")
    expected_categories = ["synthetic_all_registered_sizes", *catalog["roles"]]
    if row["requirement_categories"] != expected_categories:
        raise RuntimeError("synthetic requirement categories changed")


def expected_row_identities(config):
    exact = set()
    for model in config["exact_hgp"]["models"]:
        for syndrome in config["exact_hgp"]["syndromes"]:
            for p in config["exact_hgp"]["p_values"]:
                for catalog in ("logical", "collapsed_B"):
                    for shift in config["injected_character_shifts"]:
                        exact.add((
                            "exact_hgp_iid", model["id"], syndrome, float(p),
                            catalog, None, "single_character_reweight", float(shift),
                        ))
    synthetic = set()
    spec = config["synthetic_multiplicity_stress"]
    for catalog in spec["catalogs"]:
        for base in spec["character_means"]:
            for shift in spec["single_character_shifts"]:
                synthetic.add((
                    "synthetic_independent_character_stress", None, None, None,
                    catalog["id"], float(base), "single_character", float(shift),
                ))
            for shift in spec["distributed_character_shifts"]:
                synthetic.add((
                    "synthetic_independent_character_stress", None, None, None,
                    catalog["id"], float(base), "distributed_all_characters",
                    float(shift),
                ))
    return exact | synthetic


def exact_scenario_indices(config):
    result = {}
    index = 0
    for model in config["exact_hgp"]["models"]:
        for syndrome in config["exact_hgp"]["syndromes"]:
            for p in config["exact_hgp"]["p_values"]:
                for catalog in ("logical", "collapsed_B"):
                    result[(model["id"], syndrome, float(p), catalog)] = index
                    index += 1
    return result


def row_identity(row):
    return (
        row["source"], row["model_id"], row["syndrome_kind"], row["p"],
        row["catalog"], row.get("base_character_mean"), row["shift_pattern"],
        row["requested_shift"],
    )


def verify_rows(rows, config, stage, trajectories, draws, role_multipliers):
    if {row_identity(row) for row in rows} != expected_row_identities(config):
        raise RuntimeError(f"{stage} row identity grid is incomplete or duplicated")
    if len(rows) != len(expected_row_identities(config)):
        raise RuntimeError(f"{stage} row identity grid contains duplicates")
    catalogs = {
        row["id"]: row for row in config["synthetic_multiplicity_stress"]["catalogs"]
    }
    scenario_indices = exact_scenario_indices(config)
    config_sha = sha256_file(CONFIG_PATH)
    for row in rows:
        if row["stage"] != stage:
            raise RuntimeError("row stage changed")
        if row["trajectory_count"] != trajectories or row["draws_per_trajectory"] != draws:
            raise RuntimeError("row does not use the common operating point")
        role = row["simultaneous_role"]
        if role not in role_multipliers:
            raise RuntimeError("row has an unknown simultaneous role")
        assert_float(
            row["simultaneous_multiplier"], role_multipliers[role],
            "row did not use its one frozen role multiplier",
        )
        verify_rate_fields(row, config)
        if row["source"] == "exact_hgp_iid":
            expected_role = "logical" if row["catalog"] == "logical" else "collapsed_B"
            if role != expected_role or row["requirement_categories"] != [f"exact_{row['catalog']}"]:
                raise RuntimeError("exact role or category changed")
            expected_index = scenario_indices[(
                row["model_id"], row["syndrome_kind"], row["p"], row["catalog"],
            )]
            if row["scenario_index"] != expected_index:
                raise RuntimeError("exact scenario index changed")
            expected_seed = derive_seed(
                config["seed_namespace"], config_sha, stage, "exact",
                trajectories, draws, expected_index, row["requested_shift"],
            )
            if row["seed"] != expected_seed:
                raise RuntimeError("exact fresh-stage seed changed")
            verify_exact_distribution(row)
        elif row["source"] == "synthetic_independent_character_stress":
            catalog = catalogs[row["catalog"]]
            if role != catalog["role"] or row["catalog_size"] != catalog["size"]:
                raise RuntimeError("synthetic catalog identity changed")
            if row["scenario_index"] is not None:
                raise RuntimeError("synthetic row has an exact scenario index")
            expected_seed = derive_seed(
                config["seed_namespace"], config_sha, stage, "synthetic",
                trajectories, draws, row["catalog"], row["base_character_mean"],
                row["shift_pattern"], row["requested_shift"],
            )
            if row["seed"] != expected_seed:
                raise RuntimeError("synthetic fresh-stage seed changed")
            verify_synthetic_distribution(row, catalog)
        else:
            raise RuntimeError("unknown calibration row source")


def summarize(rows, config, trajectories, draws, role_multipliers, stage):
    rule = config["candidate_rule"]
    tolerance = float(rule["tolerance"])
    minimum_bad = tolerance + float(rule["minimum_bad_margin"])
    categories = {}
    for category in rule["required_categories"]:
        subset = [row for row in rows if category in row["requirement_categories"]]
        null = [row for row in subset if row["true_max_abs_character_shift"] <= 1e-14]
        bad = [
            row for row in subset
            if row["true_max_abs_character_shift"] >= minimum_bad - 1e-12
        ]
        if not subset or not null or not bad:
            raise RuntimeError(f"operating category is incomplete: {category}")
        null_lower = min(row["candidate_pass_wilson_lower"] for row in null)
        bad_lower = min(row["candidate_fail_wilson_lower"] for row in bad)
        coverage_lower = min(row["coverage_wilson_lower"] for row in subset)
        categories[category] = {
            "eligible": bool(
                null_lower >= rule["null_pass_min"]
                and bad_lower >= rule["fail_power_min"]
                and coverage_lower >= rule["empirical_coverage_min"]
            ),
            "minimum_bad_shift_fail_wilson_lower": float(bad_lower),
            "minimum_coverage_wilson_lower": float(coverage_lower),
            "minimum_null_pass_wilson_lower": float(null_lower),
            "row_count": len(subset),
        }
    witness = any(
        row["shift_pattern"] == "distributed_all_characters"
        and row["true_max_abs_character_shift"] <= tolerance + 1e-12
        and row["true_character_purity_delta"] > tolerance
        for row in rows
    )
    return {
        "category_summaries": categories,
        "cost_independent_draws": int(trajectories) * int(draws),
        "distributed_bias_witness": bool(witness),
        "draws_per_trajectory": int(draws),
        "eligible": bool(all(row["eligible"] for row in categories.values()) and witness),
        "role_multipliers": {
            role: float(value) for role, value in sorted(role_multipliers.items())
        },
        "stage": stage,
        "trajectory_count": int(trajectories),
    }


def ordered_points(config):
    points = [
        (int(row["trajectory_count"]), int(row["draws_per_trajectory"]))
        for row in config["common_operating_grid"]["points"]
    ]
    if len(points) != len(set(points)):
        raise RuntimeError("duplicate configured operating point")
    return sorted(points, key=lambda row: (row[0] * row[1], row[0], row[1]))


def verify_calibration_point(point, config):
    roles = set(config["candidate_rule"]["required_roles"])
    rows = point["scenario_quantiles"]
    if set(row["simultaneous_role"] for row in rows) != roles:
        raise RuntimeError("calibration point omitted a simultaneous role")
    exact_count = len(config["exact_hgp"]["models"]) * len(
        config["exact_hgp"]["syndromes"]
    ) * len(config["exact_hgp"]["p_values"]) * 2
    synthetic_specs = [
        row for row in config["synthetic_multiplicity_stress"]["catalogs"]
        if row["calibrate_multiplier"]
    ]
    synthetic_count = len(synthetic_specs) * len(
        config["synthetic_multiplicity_stress"]["character_means"]
    )
    if len(rows) != exact_count + synthetic_count:
        raise RuntimeError("calibration scenario-quantile grid changed")
    config_sha = sha256_file(CONFIG_PATH)
    scenario_indices = exact_scenario_indices(config)
    synthetic_catalogs = {
        row["id"]: row for row in config["synthetic_multiplicity_stress"]["catalogs"]
    }
    for row in rows:
        if row["source"] == "exact_hgp_iid":
            expected_index = scenario_indices[(
                row["model_id"], row["syndrome_kind"], row["p"], row["catalog"],
            )]
            expected_seed = derive_seed(
                config["seed_namespace"], config_sha, "common-z-exact",
                point["trajectory_count"], point["draws_per_trajectory"],
                expected_index,
            )
            if row["scenario_index"] != expected_index or row["seed"] != expected_seed:
                raise RuntimeError("exact calibration seed identity changed")
        elif row["source"] == "synthetic_independent_character_stress":
            catalog = synthetic_catalogs[row["catalog"]]
            expected_seed = derive_seed(
                config["seed_namespace"], config_sha, "common-z-synthetic",
                point["trajectory_count"], point["draws_per_trajectory"],
                catalog["size"], row["base_character_mean"],
            )
            if row["scenario_index"] is not None or row["seed"] != expected_seed:
                raise RuntimeError("synthetic calibration seed identity changed")
        else:
            raise RuntimeError("unknown calibration-quantile source")
    for role in roles:
        expected = max(
            float(row["quantile"]) for row in rows
            if row["simultaneous_role"] == role
        )
        assert_float(
            point["role_multipliers"][role], expected,
            "frozen role multiplier is not the role maximum",
        )
    if set(point["role_multipliers"]) != roles:
        raise RuntimeError("calibration role-multiplier registry changed")


def point_key(point):
    return int(point["trajectory_count"]), int(point["draws_per_trajectory"])


def rows_at(rows, key):
    return [
        row for row in rows
        if (int(row["trajectory_count"]), int(row["draws_per_trajectory"])) == key
    ]


def verify_decisions(config, report):
    if report["authority"] != config["authority"] or report["version"] != config["version"]:
        raise RuntimeError("report authority or version changed")
    expected_interpretation = {
        "frozen_catalog_character_purity_bound": 0.08,
        "full_q_top_bound_only_when_logical_catalog_is_complete": True,
        "maximum_character_mean_tolerance": 0.04,
        "unobserved_characters_covered": False,
    }
    if canonical(report["deterministic_interpretation"]) != canonical(expected_interpretation):
        raise RuntimeError("deterministic character-to-purity interpretation changed")
    expected_scenarios = len(config["exact_hgp"]["models"]) * len(
        config["exact_hgp"]["syndromes"]
    ) * len(config["exact_hgp"]["p_values"]) * 2
    if report["exact_hgp_scenario_count"] != expected_scenarios:
        raise RuntimeError("exact HGP scenario count changed")

    configured = ordered_points(config)
    calibration = report["calibration_points"]
    selection_points = report["selection_points"]
    if len(calibration) != len(selection_points):
        raise RuntimeError("selection point lacks its calibration point")
    keys = [point_key(point) for point in selection_points]
    if keys != configured[:len(keys)]:
        raise RuntimeError("selection did not follow fixed cost order")
    if [point_key(point) for point in calibration] != keys:
        raise RuntimeError("calibration and selection point grids differ")

    recomputed_points = []
    for calibration_point, stored_point in zip(calibration, selection_points):
        verify_calibration_point(calibration_point, config)
        if calibration_point["role_multipliers"] != stored_point["role_multipliers"]:
            raise RuntimeError("selection did not use frozen calibration multipliers")
        key = point_key(stored_point)
        rows = rows_at(report["selection_rows"], key)
        verify_rows(
            rows, config, "selection", key[0], key[1],
            calibration_point["role_multipliers"],
        )
        recomputed = summarize(
            rows, config, key[0], key[1], calibration_point["role_multipliers"],
            "selection",
        )
        if canonical(recomputed) != canonical(stored_point):
            raise RuntimeError("stored selection-point decision changed")
        recomputed_points.append(recomputed)
    if len(report["selection_rows"]) != len(selection_points) * len(
        expected_row_identities(config)
    ):
        raise RuntimeError("selection report contains rows outside its point grid")

    eligible = [index for index, point in enumerate(recomputed_points) if point["eligible"]]
    if eligible:
        first = eligible[0]
        if first != len(recomputed_points) - 1:
            raise RuntimeError("selection continued after the first eligible point")
        expected_selected = recomputed_points[first]
    else:
        if len(recomputed_points) != len(configured):
            raise RuntimeError("selection stopped without finding an eligible point")
        expected_selected = None
    if canonical(report["selected_common_operating_point"]) != canonical(expected_selected):
        raise RuntimeError("selected common operating point changed")

    confirmation = report["confirmation_point"]
    confirmation_rows = report["confirmation_rows"]
    if expected_selected is None:
        if confirmation is not None or confirmation_rows:
            raise RuntimeError("confirmation ran without a selected point")
        confirmed = False
    else:
        key = point_key(expected_selected)
        if confirmation is None or point_key(confirmation) != key:
            raise RuntimeError("confirmation changed the selected common point")
        if confirmation["role_multipliers"] != expected_selected["role_multipliers"]:
            raise RuntimeError("confirmation changed a frozen role multiplier")
        verify_rows(
            confirmation_rows, config, "confirmation", key[0], key[1],
            expected_selected["role_multipliers"],
        )
        recomputed_confirmation = summarize(
            confirmation_rows, config, key[0], key[1],
            expected_selected["role_multipliers"], "confirmation",
        )
        if canonical(recomputed_confirmation) != canonical(confirmation):
            raise RuntimeError("stored confirmation decision changed")
        confirmed = bool(recomputed_confirmation["eligible"])

    expected_status = (
        "CHARACTER_GATE_COMMON_OPERATING_POINT_CONFIRMED"
        if confirmed else "CHARACTER_GATE_REDESIGN_REQUIRED"
    )
    if report["status"] != expected_status:
        raise RuntimeError("terminal status does not follow fresh confirmation")
    if confirmed and report["status"] != config["authority"]["maximum_status"]:
        raise RuntimeError("confirmed status exceeds or misses frozen authority")


def main():
    if OUTPUT_PATH.exists():
        raise RuntimeError("independent audit already exists")
    if not REPORT_PATH.is_file():
        raise RuntimeError("calibration report does not exist")
    config = load_json_strict(CONFIG_PATH)
    report = load_json_strict(REPORT_PATH)
    verify_source_provenance(config, report)
    verify_decisions(config, report)
    core = {
        "config_sha256": report["config_sha256"],
        "report_sha256": report["report_sha256"],
        "source_commit": report["source_commit"],
        "status": "INDEPENDENT_AUDIT_PASS_" + report["status"],
        "version": "exp102.q0_character_gate_calibration.audit.v2",
    }
    core["audit_sha256"] = sha256_bytes(canonical(core).encode("ascii"))
    with OUTPUT_PATH.open("x", encoding="ascii") as handle:
        handle.write(canonical(core) + "\n")
    print(json.dumps(core, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
