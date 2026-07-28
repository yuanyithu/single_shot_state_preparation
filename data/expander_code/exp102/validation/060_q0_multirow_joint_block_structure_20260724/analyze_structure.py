"""Analyze frozen exact joint collapsed-B factor scopes and VE widths."""

from __future__ import annotations

import hashlib
from importlib import import_module
import json
import math
from pathlib import Path
import sys

import numpy as np


# Importing the frozen 056 loader must not dirty the committed source tree.
sys.dont_write_bytecode = True


PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from preflight_structure import load_json_strict, verify_for_launch


ROOT = Path(__file__).resolve().parent
EXP102_ROOT = ROOT.parents[1]
CONFIG_PATH = ROOT / "structure_config.json"
OUTPUT = ROOT / "structure_report.json"
CONTROL_WORKFLOW = (
    "data.expander_code.exp102.validation."
    "056_q0_random_full_column_direct_block_t1_m8_v2_20260724.workflow"
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


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
    if hashlib.sha256(canonical(unsigned).encode("ascii")).hexdigest() != expected:
        raise RuntimeError(f"self hash changed: {field}")


def factor_scopes_multirow(H, row_count):
    rank = H.shape[0]
    scopes = [(variable,) for variable in range(rank * row_count)]
    for factor in range(H.shape[1]):
        support = [int(value) for value in np.flatnonzero(H[:, factor])]
        scopes.append(tuple(
            slot * rank + column
            for slot in range(row_count)
            for column in support
        ))
    return tuple(scopes)


def factor_scopes_row_column(H, selected_column):
    rank = H.shape[0]
    # Variables 0..r-1 are a selected row.  The rest are the selected column
    # outside its intersection with output row zero.
    column_variables = {
        output_row: rank + offset
        for offset, output_row in enumerate(range(1, rank))
    }
    scopes = [(variable,) for variable in range(2 * rank - 1)]
    for factor in range(H.shape[1]):
        support = [int(value) for value in np.flatnonzero(H[:, factor])]
        scope = list(support)
        if selected_column in support:
            scope.extend(column_variables.values())
        scopes.append(tuple(sorted(scope)))
    return tuple(scopes)


def interaction_graph(variable_count, scopes):
    graph = [set() for _ in range(variable_count)]
    for scope in scopes:
        if len(scope) != len(set(scope)) or any(
            variable < 0 or variable >= variable_count for variable in scope
        ):
            raise RuntimeError("invalid frozen factor scope")
        for left in scope:
            graph[left].update(right for right in scope if right != left)
    return graph


def min_fill_plan(variable_count, scopes):
    graph = interaction_graph(variable_count, scopes)
    remaining = set(range(variable_count))
    order = []
    widths = []
    fill_edges = []
    while remaining:
        candidates = []
        for variable in sorted(remaining):
            neighbors = sorted(graph[variable] & remaining)
            missing = sum(
                right not in graph[left]
                for index, left in enumerate(neighbors)
                for right in neighbors[index + 1:]
            )
            candidates.append(((missing, len(neighbors), variable), neighbors))
        (missing, _degree, variable), neighbors = min(candidates)
        order.append(variable)
        widths.append(len(neighbors))
        fill_edges.append(missing)
        for index, left in enumerate(neighbors):
            for right in neighbors[index + 1:]:
                graph[left].add(right)
                graph[right].add(left)
        remaining.remove(variable)
    if sorted(order) != list(range(variable_count)):
        raise RuntimeError("elimination order is not a permutation")
    return tuple(order), tuple(widths), tuple(fill_edges)


def scope_semantic_audit(H, kind, parameter, scopes):
    rank = H.shape[0]
    factor_scopes = scopes[-H.shape[1]:]
    variable_count = len(scopes) - H.shape[1]
    for variable in range(variable_count):
        B = np.zeros((rank, rank), dtype=np.uint8)
        if kind == "multirow":
            slot, column = divmod(variable, rank)
            B[slot, column] = 1
        else:
            selected_column = int(parameter)
            if variable < rank:
                B[0, variable] = 1
            else:
                B[1 + variable - rank, selected_column] = 1
        changed = np.asarray((B @ H) & np.uint8(1), dtype=np.uint8)
        for factor, scope in enumerate(factor_scopes):
            predicted = variable in scope
            actual = bool(np.any(changed[:, factor]))
            if predicted != actual:
                raise RuntimeError("factor scope does not match exact B H change")
    return True


def summarize_plan(H, candidate_id, variable_count, scopes, gates, parameter):
    order, widths, fill_edges = min_fill_plan(variable_count, scopes)
    semantic_pass = scope_semantic_audit(
        H,
        "multirow" if candidate_id.startswith("MR") else "row_column_cross",
        parameter,
        scopes,
    )
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
        "scope_semantics": semantic_pass,
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
        "order_sha256": hashlib.sha256(canonical(order_payload).encode("ascii")).hexdigest(),
        "parameter": parameter,
        "single_table_bytes_lower_bound": int(single_table_bytes),
        "variable_count": int(variable_count),
        "width_by_step": list(widths),
    }


def load_inputs():
    config = load_json_strict(CONFIG_PATH)
    analyzer = PROJECT_ROOT / "data/expander_code/exp102" / config["implementation"]["analyzer"]
    if analyzer.resolve() != Path(__file__).resolve():
        raise RuntimeError("analyzer path binding changed")
    if sha256_file(analyzer) != config["implementation"]["analyzer_sha256"]:
        raise RuntimeError("analyzer SHA changed")
    workflow = import_module(CONTROL_WORKFLOW)
    predecessor_config, predecessor_config_sha = workflow._load_config()
    context = workflow._load_control(
        workflow.SOURCE_CONTROL_DIR, predecessor_config, predecessor_config_sha,
    )
    H = np.ascontiguousarray(context["H"], dtype=np.uint8)
    if H.shape != (24, 32):
        raise RuntimeError("frozen m8 H shape changed")
    inputs = config["inputs"]
    if matrix_sha256(H) != inputs["h_sha256"]:
        raise RuntimeError("frozen m8 H SHA changed")
    if context["metadata"]["control_content_sha256"] != inputs[
        "control_content_sha256"
    ]:
        raise RuntimeError("frozen control content SHA changed")
    if context["metadata"]["cell"] != config["cell"]:
        raise RuntimeError("frozen cell binding changed")
    predecessor = load_json_strict(EXP102_ROOT / inputs["predecessor_report"])
    verify_self_hash(predecessor, "report_sha256")
    if predecessor["report_sha256"] != inputs["predecessor_report_sha256"]:
        raise RuntimeError("predecessor report binding changed")
    if predecessor["status"] != "LOCAL_HYBRID_B_NECESSARY_GATES_FAIL":
        raise RuntimeError("predecessor terminal status changed")
    return config, context, H


def main():
    if OUTPUT.exists():
        raise RuntimeError("structure report already exists")
    config = load_json_strict(CONFIG_PATH)
    provenance = verify_for_launch(config)
    config, context, H = load_inputs()
    gates = config["gates"]
    results = []
    for candidate in config["candidates"]:
        candidate_id = candidate["id"]
        if candidate["kind"] == "multirow":
            row_count = int(candidate["row_count"])
            scopes = factor_scopes_multirow(H, row_count)
            block_count = math.comb(H.shape[0], row_count)
            plans = [summarize_plan(
                H, candidate_id, H.shape[0] * row_count, scopes, gates,
                row_count,
            )]
        elif candidate["kind"] == "row_column_cross":
            block_count = H.shape[0] * H.shape[0]
            plans = [
                summarize_plan(
                    H, candidate_id, 2 * H.shape[0] - 1,
                    factor_scopes_row_column(H, selected_column), gates,
                    selected_column,
                )
                for selected_column in range(H.shape[0])
            ]
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
        "control_content_sha256": context["metadata"]["control_content_sha256"],
        "eligible_candidate_ids": eligible_ids,
        "gates": {
            "at_least_one_candidate": bool(eligible_ids),
            "scope_semantics_all": all(
                plan["gates"]["scope_semantics"]
                for result in results for plan in result["plans"]
            ),
        },
        "h_sha256": matrix_sha256(H),
        "preferred_contingency_candidate_id": (
            None if preferred is None else preferred["candidate"]["id"]
        ),
        "results": results,
        "selection_policy": config["selection_policy"],
        "source_commit": provenance["source_commit"],
        "source_file_count": provenance["source_file_count"],
        "source_tree_sha256": provenance["source_tree_sha256"],
        "status": (
            "LOCAL_JOINT_BLOCK_STRUCTURE_CANDIDATE_FOUND"
            if eligible_ids else "LOCAL_JOINT_BLOCK_STRUCTURE_EXHAUSTED"
        ),
        "version": config["version"],
    }
    core["report_sha256"] = hashlib.sha256(canonical(core).encode("ascii")).hexdigest()
    with OUTPUT.open("x", encoding="ascii") as handle:
        handle.write(canonical(core) + "\n")
    print(json.dumps(core, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
