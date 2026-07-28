"""Independent raw-report audit for validation 060.

This module intentionally does not import ``analyze_structure`` or its source
preflight.  Factor scopes are reconstructed from direct B-coordinate
perturbations, and min-fill is reimplemented with integer bitsets.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[5]
ROOT = Path(__file__).resolve().parent
EXP102_ROOT = ROOT.parents[1]
CONFIG_PATH = ROOT / "structure_config.json"
REPORT_PATH = ROOT / "structure_report.json"
OUTPUT_PATH = ROOT / "independent_structure_audit.json"
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


def matrix_sha256(value):
    value = np.ascontiguousarray(value, dtype=np.uint8)
    digest = hashlib.sha256()
    digest.update(np.asarray(value.shape, dtype=">u8").tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


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
        raise RuntimeError("configured path escapes exp102 root")
    path = (EXP102_ROOT / relative).resolve()
    try:
        path.relative_to(EXP102_ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError("configured path escapes exp102 root") from exc
    if path.is_symlink():
        raise RuntimeError(f"frozen artifact may not be a symlink: {path}")
    return path


def _repo_relative(path):
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise RuntimeError(f"path is outside repository: {path}") from exc


def _artifact_rows(config, report):
    rows = [(CONFIG_PATH.resolve(), report["config_sha256"])]
    for section in ("implementation", "documentation"):
        values = config[section]
        for sha_key in sorted(key for key in values if key.endswith("_sha256")):
            key = sha_key[:-7]
            rows.append((_exp102_path(values[key]), values[sha_key]))
    inputs = config["inputs"]
    rows.extend((
        (_exp102_path(inputs["control"]), inputs["control_file_sha256"]),
        (
            _exp102_path(inputs["predecessor_report"]),
            inputs["predecessor_report_file_sha256"],
        ),
    ))
    unique = {}
    for path, expected in rows:
        if re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise RuntimeError(f"invalid artifact SHA256: {path}")
        if path in unique and unique[path] != expected:
            raise RuntimeError(f"conflicting artifact SHA256: {path}")
        unique[path] = expected
    return sorted(unique.items(), key=lambda row: str(row[0]))


def verify_report_source(config, report):
    source_commit = report["source_commit"]
    if COMMIT_RE.fullmatch(source_commit) is None:
        raise RuntimeError("report source commit is invalid")
    if _git(["rev-parse", "HEAD"]).strip() != source_commit:
        raise RuntimeError("audit must run at the report source commit")
    rows = []
    relative_paths = []
    for path, expected in _artifact_rows(config, report):
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"working artifact differs from config: {path}")
        relative = _repo_relative(path)
        relative_paths.append(relative)
        committed = _git(["show", f"{source_commit}:{relative}"], text=False)
        if sha256_bytes(committed) != expected:
            raise RuntimeError(f"source commit artifact differs: {relative}")
        rows.append([relative, expected])
    dirty = _git([
        "status", "--porcelain=v1", "--untracked-files=all", "--",
        *relative_paths,
    ])
    if dirty:
        raise RuntimeError("audited source artifacts changed after launch")
    rows.sort()
    source_tree_sha256 = sha256_bytes(canonical(rows).encode("ascii"))
    if report["source_file_count"] != len(rows):
        raise RuntimeError("source file count changed")
    if report["source_tree_sha256"] != source_tree_sha256:
        raise RuntimeError("source tree SHA changed")


def _coordinates_multirow(rank, selected_rows):
    return tuple(
        (int(row), column)
        for row in selected_rows
        for column in range(rank)
    )


def _coordinates_row_column(rank, selected_row, selected_column):
    return tuple(
        [(selected_row, column) for column in range(rank)]
        + [
            (row, selected_column)
            for row in range(rank) if row != selected_row
        ]
    )


def semantic_scopes(H, coordinates):
    """Derive factor membership only from direct single-coordinate B changes."""
    rank = H.shape[0]
    scopes = [(variable,) for variable in range(len(coordinates))]
    factor_variables = [[] for _ in range(H.shape[1])]
    for variable, (row, column) in enumerate(coordinates):
        B = np.zeros((rank, rank), dtype=np.uint8)
        B[row, column] = np.uint8(1)
        changed = np.asarray((B @ H) & np.uint8(1), dtype=np.uint8)
        for factor in range(H.shape[1]):
            if np.any(changed[:, factor]):
                factor_variables[factor].append(variable)
    scopes.extend(tuple(values) for values in factor_variables)
    return tuple(scopes)


def min_fill_bitset(variable_count, scopes):
    adjacency = [0] * variable_count
    for scope in scopes:
        if len(scope) != len(set(scope)) or any(
            variable < 0 or variable >= variable_count for variable in scope
        ):
            raise RuntimeError("invalid independently reconstructed factor scope")
        mask = sum(1 << variable for variable in scope)
        for variable in scope:
            adjacency[variable] |= mask ^ (1 << variable)
    remaining = (1 << variable_count) - 1
    order = []
    widths = []
    fill_edges = []
    while remaining:
        candidates = []
        live = remaining
        while live:
            least = live & -live
            variable = least.bit_length() - 1
            live ^= least
            neighbors = adjacency[variable] & remaining
            degree = neighbors.bit_count()
            existing_twice = 0
            scan = neighbors
            while scan:
                bit = scan & -scan
                neighbor = bit.bit_length() - 1
                scan ^= bit
                existing_twice += (adjacency[neighbor] & neighbors).bit_count()
            missing = degree * (degree - 1) // 2 - existing_twice // 2
            candidates.append(((missing, degree, variable), neighbors))
        (missing, _degree, variable), neighbors = min(candidates)
        order.append(variable)
        widths.append(neighbors.bit_count())
        fill_edges.append(missing)
        scan = neighbors
        while scan:
            bit = scan & -scan
            neighbor = bit.bit_length() - 1
            scan ^= bit
            adjacency[neighbor] |= neighbors ^ bit
        remaining ^= 1 << variable
    return tuple(order), tuple(widths), tuple(fill_edges)


def summarize_plan(H, candidate_id, coordinates, gates, parameter):
    scopes = semantic_scopes(H, coordinates)
    order, widths, fill_edges = min_fill_bitset(len(coordinates), scopes)
    width = max(widths, default=0)
    largest_entries = 1 << (width + 1)
    initial_scope = max(map(len, scopes), default=0)
    single_table_bytes = largest_entries * np.dtype(np.float64).itemsize
    gate_values = {
        "induced_width": width <= gates["max_induced_width"],
        "initial_factor_scope": initial_scope <= gates["max_initial_factor_scope"],
        "largest_factor_entries": (
            largest_entries <= gates["max_largest_factor_entries"]
        ),
        "scope_semantics": True,
        "single_table_bytes": single_table_bytes <= gates["max_single_table_bytes"],
    }
    gate_values["all"] = all(gate_values.values())
    order_payload = {
        "candidate_id": candidate_id,
        "factor_scopes": [list(scope) for scope in scopes],
        "order": list(order),
        "parameter": parameter,
    }
    return {
        "fill_edges_total": int(sum(fill_edges)),
        "gates": gate_values,
        "induced_width": int(width),
        "largest_factor_entries": int(largest_entries),
        "largest_initial_factor_scope": int(initial_scope),
        "order": list(order),
        "order_sha256": sha256_bytes(canonical(order_payload).encode("ascii")),
        "parameter": parameter,
        "single_table_bytes_lower_bound": int(single_table_bytes),
        "variable_count": len(coordinates),
        "width_by_step": list(widths),
    }


def reconstruct_report(config, report):
    inputs = config["inputs"]
    control_path = _exp102_path(inputs["control"])
    if sha256_file(control_path) != inputs["control_file_sha256"]:
        raise RuntimeError("control file SHA changed")
    with np.load(control_path, allow_pickle=False) as archive:
        if "H" not in archive.files or "metadata_json" not in archive.files:
            raise RuntimeError("frozen control schema changed")
        H = np.ascontiguousarray(archive["H"], dtype=np.uint8)
        metadata = json.loads(
            str(archive["metadata_json"].item()),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite metadata constant: {value}")
            ),
        )
    if H.shape != (24, 32) or matrix_sha256(H) != inputs["h_sha256"]:
        raise RuntimeError("frozen H identity changed")
    if metadata["cell"] != config["cell"]:
        raise RuntimeError("control cell changed")
    if metadata["control_content_sha256"] != inputs["control_content_sha256"]:
        raise RuntimeError("control content SHA changed")

    predecessor_path = _exp102_path(inputs["predecessor_report"])
    if sha256_file(predecessor_path) != inputs["predecessor_report_file_sha256"]:
        raise RuntimeError("predecessor report file SHA changed")
    predecessor = load_json_strict(predecessor_path)
    verify_self_hash(predecessor, "report_sha256")
    if (
        predecessor["report_sha256"] != inputs["predecessor_report_sha256"]
        or predecessor["status"] != "LOCAL_HYBRID_B_NECESSARY_GATES_FAIL"
    ):
        raise RuntimeError("predecessor result identity changed")

    gates = config["gates"]
    results = []
    rank = H.shape[0]
    for candidate in config["candidates"]:
        candidate_id = candidate["id"]
        if candidate["kind"] == "multirow":
            row_count = int(candidate["row_count"])
            coordinates = _coordinates_multirow(rank, range(row_count))
            plans = [summarize_plan(
                H, candidate_id, coordinates, gates, row_count,
            )]
            block_count = math.comb(rank, row_count)
        elif candidate["kind"] == "row_column_cross":
            plans = [
                summarize_plan(
                    H, candidate_id,
                    _coordinates_row_column(rank, 0, selected_column),
                    gates, selected_column,
                )
                for selected_column in range(rank)
            ]
            block_count = rank * rank
        else:
            raise RuntimeError("unknown frozen candidate kind")
        eligible = all(plan["gates"]["all"] for plan in plans)
        results.append({
            "block_count": int(block_count),
            "candidate": candidate,
            "eligible": bool(eligible),
            "maximum_induced_width": max(plan["induced_width"] for plan in plans),
            "maximum_single_table_bytes_lower_bound": max(
                plan["single_table_bytes_lower_bound"] for plan in plans
            ),
            "minimum_induced_width": min(plan["induced_width"] for plan in plans),
            "plans": plans,
        })
    eligible_ids = [row["candidate"]["id"] for row in results if row["eligible"]]
    candidate_order = {
        candidate["id"]: index for index, candidate in enumerate(config["candidates"])
    }
    eligible_results = [row for row in results if row["eligible"]]
    preferred = min(
        eligible_results,
        key=lambda row: (
            max(plan["variable_count"] for plan in row["plans"]),
            row["maximum_induced_width"],
            row["maximum_single_table_bytes_lower_bound"],
            candidate_order[row["candidate"]["id"]],
        ),
        default=None,
    )
    core = {
        "authority": config["authority"],
        "cell": config["cell"],
        "config_sha256": sha256_file(CONFIG_PATH),
        "control_content_sha256": metadata["control_content_sha256"],
        "eligible_candidate_ids": eligible_ids,
        "gates": {
            "at_least_one_candidate": bool(eligible_ids),
            "scope_semantics_all": True,
        },
        "h_sha256": matrix_sha256(H),
        "preferred_contingency_candidate_id": (
            None if preferred is None else preferred["candidate"]["id"]
        ),
        "results": results,
        "selection_policy": config["selection_policy"],
        "source_commit": report["source_commit"],
        "source_file_count": report["source_file_count"],
        "source_tree_sha256": report["source_tree_sha256"],
        "status": (
            "LOCAL_JOINT_BLOCK_STRUCTURE_CANDIDATE_FOUND"
            if eligible_ids else "LOCAL_JOINT_BLOCK_STRUCTURE_EXHAUSTED"
        ),
        "version": config["version"],
    }
    core["report_sha256"] = sha256_bytes(canonical(core).encode("ascii"))
    return core


def run_audit():
    config = load_json_strict(CONFIG_PATH)
    report = load_json_strict(REPORT_PATH)
    verify_self_hash(report, "report_sha256")
    verify_report_source(config, report)
    reconstructed = reconstruct_report(config, report)
    if canonical(reconstructed) != canonical(report):
        raise RuntimeError("independent reconstruction disagrees with structure report")
    core = {
        "audited_terminal_status": report["status"],
        "config_sha256": report["config_sha256"],
        "source_commit": report["source_commit"],
        "source_tree_sha256": report["source_tree_sha256"],
        "status": "INDEPENDENT_STRUCTURE_AUDIT_PASS",
        "structure_report_file_sha256": sha256_file(REPORT_PATH),
        "structure_report_sha256": report["report_sha256"],
        "version": "exp102.q0_multirow_joint_block.structure_audit.v0",
    }
    core["audit_sha256"] = sha256_bytes(canonical(core).encode("ascii"))
    return core


def main():
    if OUTPUT_PATH.exists():
        raise RuntimeError("independent structure audit already exists")
    try:
        core = run_audit()
    except BaseException as exc:
        core = {
            "error_message": str(exc),
            "error_type": type(exc).__name__,
            "status": "INDEPENDENT_STRUCTURE_AUDIT_CONFLICT",
            "version": "exp102.q0_multirow_joint_block.structure_audit.v0",
        }
        core["audit_sha256"] = sha256_bytes(canonical(core).encode("ascii"))
        with OUTPUT_PATH.open("x", encoding="ascii") as handle:
            handle.write(canonical(core) + "\n")
        raise
    with OUTPUT_PATH.open("x", encoding="ascii") as handle:
        handle.write(canonical(core) + "\n")
    print(json.dumps(core, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
