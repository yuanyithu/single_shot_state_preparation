"""Independent algebra and raw-only audit for validations 047, 049, and 050."""

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

from data.expander_code.exp102.exp102_pipeline.diagnostics import bulk_ess, split_rhat
from data.expander_code.exp102.exp102_pipeline.io import (
    atomic_json,
    canonical_json,
    sha256_file,
    sha256_json,
)
from data.expander_code.exp102.exp102_pipeline.q0_center_preserving import (
    build_dressed_logical_catalog,
)
from data.expander_code.exp102.exp102_pipeline.q0_global import (
    state_label,
    uniform_hard_coset_state,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_collapsed import (
    build_classical_coset_mass,
    split_hgp_state,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_screen import _disorder
from data.expander_code.exp102.exp102_pipeline.q0_houdayer_pair import (
    deterministic_low_energy_logical_starts,
)
from data.expander_code.exp102.exp102_pipeline.q0_logical_stratified import (
    load_logical_stratified_frozen_artifact,
)
from data.expander_code.exp102.exp102_pipeline.registry import (
    load_frozen_code,
    load_registry,
)
from data.expander_code.exp102.exp102_pipeline.worker import build_model


AUDIT_VERSION = "exp102.q0_global_move.independent_audit.v0"
ROOT = Path(__file__).resolve().parent
EXP102_ROOT = ROOT.parents[1]
REGISTRY_PATH = EXP102_ROOT / "registry/registry.json"
V047 = EXP102_ROOT / "validation/047_q0_center_preserving_structure_20260724"
V049 = EXP102_ROOT / "validation/049_q0_random_full_column_local_v1_20260724"
V050 = EXP102_ROOT / "validation/050_q0_full_column_map_bridge_structure_20260724"


class AuditError(RuntimeError):
    pass


def _require(condition, message):
    if not condition:
        raise AuditError(message)


def _load_json(path):
    serialized = Path(path).read_text(encoding="ascii")
    value = json.loads(serialized)
    _require(serialized == canonical_json(value) + "\n", f"noncanonical JSON: {path}")
    return value


def _verify_self_hash(value, field):
    claimed = value[field]
    core = {key: item for key, item in value.items() if key != field}
    _require(sha256_json(core) == claimed, f"self-hash mismatch: {field}")
    return claimed


def _rank_uint64(values):
    pivots = {}
    for raw in np.asarray(values, dtype=np.uint64):
        residue = int(raw)
        while residue:
            pivot = residue.bit_length() - 1
            if pivot not in pivots:
                pivots[pivot] = residue
                break
            residue ^= pivots[pivot]
    return len(pivots)


def _labels_direct(frame, states):
    bits = (
        np.asarray(states, dtype=np.uint8).astype(np.int64)
        @ frame.W_basis.astype(np.int64).T
    ) % 2
    result = np.zeros(bits.shape[0], dtype=np.uint64)
    for bit in range(frame.k):
        result |= bits[:, bit].astype(np.uint64) << np.uint64(bit)
    return result


def _unpack(packed, width):
    return np.unpackbits(
        np.asarray(packed, dtype=np.uint8), axis=-1, count=int(width),
        bitorder="little",
    ).astype(np.uint8, copy=False)


def _mask(bits):
    value = 0
    for bit, entry in enumerate(np.asarray(bits, dtype=np.uint8)):
        value |= int(entry) << bit
    return value


def _b_columns(state, H):
    _unused, block = split_hgp_state(state, H)
    return np.asarray([_mask(block[:, column]) for column in range(H.shape[0])], dtype=np.uint32)


def _a_syndromes(H, syndrome, b_columns):
    rows, columns = H.shape
    y = np.asarray(syndrome, dtype=np.uint8).reshape(rows, columns)
    result = np.asarray([_mask(y[:, column]) for column in range(columns)], dtype=np.uint32)
    for column in range(columns):
        for row in np.flatnonzero(H[:, column]):
            result[column] ^= np.uint32(b_columns[int(row)])
    return result


def _context(cell):
    registry = load_registry(REGISTRY_PATH)
    _unused, code, H = load_frozen_code(REGISTRY_PATH, cell["code_id"])
    model, frame = build_model(H)
    uniform_seed, planted, syndrome = _disorder(registry, code, model, cell)
    _require(H.shape == (24, 32) and model.k == 64 and int(syndrome.sum()) == 160,
             "hard-cell identity changed")
    return registry, H, model, frame, int(uniform_seed), planted, syndrome


def _audit_047():
    config_path = EXP102_ROOT / "config/q0_center_preserving.structure.v0.json"
    config = _load_json(config_path)
    report = _load_json(V047 / "structure_report.json")
    report_sha = _verify_self_hash(report, "report_sha256")
    registry, _H, model, frame, uniform_seed, planted, syndrome = _context(config["cell"])
    _require(report["config_sha256"] == sha256_file(config_path)
             and report["registry_sha256"] == registry["registry_sha256"]
             and report["uniform_disorder_seed"] == uniform_seed,
             "047 frozen identity mismatch")
    artifact_path = EXP102_ROOT / config["artifact"]["relpath"]
    _require(sha256_file(artifact_path) == config["artifact"]["file_sha256"],
             "047 artifact hash mismatch")
    artifact = load_logical_stratified_frozen_artifact(artifact_path, model, frame)
    catalog = build_dressed_logical_catalog(
        model, frame, artifact, max_moves=config["candidate_rule"]["max_moves"],
    )
    moves = catalog.unpack_moves()
    anchors = catalog.unpack_anchors()
    signatures = np.asarray(catalog.signatures, dtype=np.uint64)
    residuals = (
        model.H_check.astype(np.int64) @ moves.T.astype(np.int64) % 2
    ).astype(np.uint8)
    direct_signatures = _labels_direct(frame, moves)
    _require(not residuals.any(), "047 catalog move left ker(H_Z)")
    _require(np.array_equal(direct_signatures, signatures),
             "047 signature is not Wd")
    _require(len({row.tobytes() for row in catalog.moves_packed}) == catalog.size,
             "047 catalog contains duplicate moves")
    _require(_rank_uint64(signatures) == model.k,
             "047 catalog lost signature rank")
    _require(catalog.catalog_sha256 == report["catalog"]["catalog_sha256"],
             "047 catalog hash mismatch")
    _require(np.array_equal(anchors, catalog.base_anchor[None, :] ^ moves),
             "047 anchor/move identity mismatch")

    odds_log = math.log(0.04 / 0.96)
    sweeps = int(config["gates"]["t3_catalog_sweeps"])
    threshold = float(config["gates"]["expected_accept_threshold"])

    def accessible_rank(state):
        delta = (state[None, :] ^ moves).sum(axis=1) - int(state.sum())
        acceptance = np.exp(np.minimum(0.0, delta.astype(np.float64) * odds_log))
        return _rank_uint64(signatures[sweeps * acceptance >= threshold]), acceptance

    base_rank, base_acceptance = accessible_rank(catalog.base_anchor)
    p_rank, _p_acceptance = accessible_rank(planted)
    _require(base_rank == report["profiles"]["BASE"][0]["accessible_signature_rank"] == 4,
             "047 base accessible rank mismatch")
    _require(p_rank == report["profiles"]["P"][0]["accessible_signature_rank"] == 1,
             "047 P accessible rank mismatch")
    order = sorted(range(catalog.size), key=lambda index: (
        1.0 / float(base_acceptance[index]), int(signatures[index]),
    ))
    pivots = {}
    selected = []
    for index in order:
        residue = int(signatures[index])
        while residue:
            pivot = residue.bit_length() - 1
            if pivot not in pivots:
                pivots[pivot] = residue
                selected.append(index)
                break
            residue ^= pivots[pivot]
        if len(pivots) == model.k:
            break
    bottleneck = sweeps / sum(1.0 / float(base_acceptance[index]) for index in selected)
    _require(math.isclose(
        bottleneck,
        report["optimistic_full_rank_bottleneck"]["best_equalized_expected_accepts_per_direction"],
        rel_tol=2e-15, abs_tol=0.0,
    ), "047 bottleneck mismatch")
    logical = deterministic_low_energy_logical_starts(
        model, frame, planted, count=8, orders=(1, 2, 3),
    )
    index_by_signature = {int(value): index for index, value in enumerate(signatures)}
    for record, claimed in zip(logical, report["exact_l_routes"]):
        index = index_by_signature[int(record["signature"])]
        endpoint = np.asarray(record["state"], dtype=np.uint8) ^ moves[index]
        _require(index == claimed["catalog_index"]
                 and int(endpoint.sum()) == claimed["endpoint_weight"]
                 and int(state_label(frame, endpoint)) == claimed["endpoint_label"]
                 and int(endpoint.sum()) <= int(np.asarray(record["state"]).sum()),
                 "047 exact L route mismatch")
    _require(report["checks"]["base_accessible_rank"] is False
             and report["checks"]["p_accessible_rank"] is False
             and report["status"] == "LOCAL_CENTER_PRESERVING_STRUCTURE_NOT_VIABLE",
             "047 failed conclusion changed")
    return {
        "catalog_sha256": catalog.catalog_sha256,
        "recomputed_base_accessible_rank": base_rank,
        "recomputed_p_accessible_rank": p_rank,
        "report_sha256": report_sha,
        "status": report["status"],
    }


def _frequencies(labels):
    result = []
    for row in labels:
        values, counts = np.unique(row, return_counts=True)
        result.append({int(value): int(count) / row.size for value, count in zip(values, counts)})
    return result


def _overlap(left, right):
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(label, 0.0) for label, value in left.items())


def _within(freq):
    values = [
        _overlap(freq[left], freq[right])
        for left in range(len(freq)) for right in range(left + 1, len(freq))
    ]
    return float(np.mean(values))


def _compare_float(actual, expected, name, tolerance=2e-12):
    _require(math.isclose(float(actual), float(expected), rel_tol=tolerance, abs_tol=tolerance),
             f"float mismatch: {name}")


def _audit_049():
    config_path = EXP102_ROOT / "config/q0_random_full_column.local.v1.json"
    config = _load_json(config_path)
    report = _load_json(V049 / "transport_report.json")
    manifest = _load_json(V049 / "task_manifest.json")
    report_sha = _verify_self_hash(report, "report_sha256")
    manifest_sha = _verify_self_hash(manifest, "manifest_sha256")
    _require(report["manifest_sha256"] == manifest_sha
             and report["config_sha256"] == sha256_file(config_path),
             "049 report/manifest identity mismatch")
    registry, H, model, frame, _uniform_seed, planted, syndrome = _context(config["cell"])
    logical = deterministic_low_energy_logical_starts(
        model, frame, planted, count=4, orders=(1, 2, 3),
    )
    mass = build_classical_coset_mass(H, 0.04, engine="numba")
    log_mass = np.log(mass)
    raw_paths = sorted((V049 / "raw").glob("*.npz"))
    raw_set_sha = hashlib.sha256("".join(
        f"{path.name}:{sha256_file(path)}\n" for path in raw_paths
    ).encode("ascii")).hexdigest()
    _require(len(raw_paths) == 12 and raw_set_sha == report["raw_set_sha256"],
             "049 raw-set identity mismatch")
    task_by_name = {
        f"{task['family']}_{task['index']:02d}.npz": task
        for task in manifest["tasks"]
    }
    loaded = {family: [] for family in ("P", "U", "L")}
    required_fields = None
    for path in raw_paths:
        task = task_by_name[path.name]
        with np.load(path, allow_pickle=False) as archive:
            raw = {name: archive[name].copy() for name in archive.files}
        if required_fields is None:
            required_fields = set(raw)
        _require(set(raw) == required_fields, "049 raw schemas differ")
        _require(str(raw["task_json"].item()) == canonical_json(task)
                 and str(raw["task_fingerprint"].item()) == task["task_fingerprint"]
                 and str(raw["config_sha256"].item()) == report["config_sha256"],
                 "049 raw task identity mismatch")
        seed_identity = hashlib.sha256(
            np.asarray([
                task["burn_update_seed"], task["measurement_update_seed"],
                task["observation_seed"],
            ], dtype=">u8").tobytes()
            + np.asarray(0.04, dtype=">f8").tobytes()
            + np.asarray([64, 256], dtype=">u8").tobytes()
        ).hexdigest()
        _require(str(raw["seed_identity_sha256"].item()) == seed_identity,
                 "049 seed identity mismatch")
        initial = _unpack(raw["initial_state_packed"], model.num_qubits)
        if task["family"] == "P":
            expected_initial = planted
        elif task["family"] == "U":
            expected_initial = uniform_hard_coset_state(
                model, syndrome, task["initialization_seed"],
            )
        else:
            expected_initial = logical[task["index"]]["state"]
        _require(np.array_equal(initial, expected_initial), "049 initial state mismatch")
        _require(np.array_equal(_b_columns(initial, H), raw["initial_b_columns"]),
                 "049 initial B mismatch")
        b_state = raw["initial_b_columns"].copy()
        for stage, updates, records in (("burn", 64, False), ("measurement", 256, True)):
            selected = raw[f"{stage}__selected_columns"]
            old = raw[f"{stage}__old_columns"]
            new = raw[f"{stage}__new_columns"]
            _require(selected.shape == (updates,), "049 transcript length mismatch")
            change_count = 0
            changed_bits = 0
            label_changes = 0
            previous_label = int(state_label(frame, initial)) if stage == "measurement" else None
            if records:
                states = _unpack(raw["measurement__states_packed"], model.num_qubits)
                labels = _labels_direct(frame, states)
            for clock in range(updates):
                column = int(selected[clock])
                _require(0 <= column < H.shape[0]
                         and int(old[clock]) == int(b_state[column]),
                         "049 old-column transcript mismatch")
                delta = int(old[clock]) ^ int(new[clock])
                if delta:
                    change_count += 1
                    changed_bits += delta.bit_count()
                b_state[column] = new[clock]
                if records:
                    _require(np.array_equal(b_state, raw["measurement__b_columns"][clock]),
                             "049 measurement B transcript mismatch")
                    state = states[clock]
                    residual = (
                        model.H_check.astype(np.int64) @ state.astype(np.int64) % 2
                    ).astype(np.uint8)
                    _require(np.array_equal(residual, syndrome)
                             and np.array_equal(_b_columns(state, H), b_state),
                             "049 observed state is not the stored hard-coset B state")
                    _require(int(state.sum()) == int(raw["measurement__weights"][clock])
                             and int(labels[clock]) == int(raw["measurement__labels"][clock]),
                             "049 state weight/label mismatch")
                    b_weight = sum(int(value).bit_count() for value in b_state)
                    _require(b_weight == int(raw["measurement__b_weights"][clock]),
                             "049 B weight mismatch")
                    likelihood = float(log_mass[_a_syndromes(H, syndrome, b_state)].sum())
                    _compare_float(likelihood, raw["measurement__b_likelihood"][clock],
                                   "049 B likelihood", tolerance=3e-14)
                    expected_block = min(7, 8 * clock // updates)
                    _require(expected_block == int(raw["measurement__blocks"][clock]),
                             "049 block clock mismatch")
                    if int(labels[clock]) != previous_label:
                        label_changes += 1
                    previous_label = int(labels[clock])
            counters = raw[f"{stage}__counters"]
            _require(int(counters[0]) == updates
                     and int(counters[1]) == change_count
                     and int(counters[2]) == changed_bits
                     and int(counters[3]) == (updates if records else 0)
                     and int(counters[4]) == (label_changes if records else 0),
                     "049 counters mismatch")
            if stage == "burn":
                _require(np.array_equal(b_state, raw["burn__final_b_columns"]),
                         "049 burn endpoint mismatch")
        _require(np.array_equal(b_state, raw["final_b_columns"]),
                 "049 final B mismatch")
        final_state = _unpack(raw["final_state_packed"], model.num_qubits)
        _require(np.array_equal(final_state, states[-1]), "049 final state mismatch")
        loaded[task["family"]].append(raw)

    summaries = {}
    for family, rows in loaded.items():
        labels = np.stack([row["measurement__labels"] for row in rows])
        weights = np.stack([row["measurement__weights"] for row in rows]).astype(np.float64)
        b_weights = np.stack([row["measurement__b_weights"] for row in rows]).astype(np.float64)
        likelihood = np.stack([row["measurement__b_likelihood"] for row in rows])
        frequencies = _frequencies(labels)
        summaries[family] = {
            "b_weight_mean_normalized": float(b_weights.mean() / (24 * 24)),
            "frequencies": frequencies,
            "observables": {
                name: {"bulk_ess": float(bulk_ess(values)),
                       "split_rhat": float(split_rhat(values))}
                for name, values in (("weight", weights), ("b_weight", b_weights),
                                     ("b_likelihood", likelihood))
            },
            "q_top_diagnostic": _within(frequencies),
            "weight_mean_normalized": float(weights.mean() / model.num_qubits),
        }
        for key, value in report["summaries"][family].items():
            if key == "observables":
                for observable, metrics in value.items():
                    for metric, expected in metrics.items():
                        _compare_float(
                            summaries[family]["observables"][observable][metric], expected,
                            f"049 {family} {observable} {metric}",
                        )
            else:
                _compare_float(summaries[family][key], value, f"049 {family} {key}")
    comparisons = {}
    for left, right in (("P", "U"), ("P", "L"), ("U", "L")):
        overlap = float(np.mean([
            _overlap(a, b)
            for a in summaries[left]["frequencies"]
            for b in summaries[right]["frequencies"]
        ]))
        comparisons[f"{left}_{right}"] = {
            "abs_b_weight_delta": abs(
                summaries[left]["b_weight_mean_normalized"]
                - summaries[right]["b_weight_mean_normalized"]
            ),
            "abs_q_top_delta": abs(
                summaries[left]["q_top_diagnostic"]
                - summaries[right]["q_top_diagnostic"]
            ),
            "abs_weight_delta": abs(
                summaries[left]["weight_mean_normalized"]
                - summaries[right]["weight_mean_normalized"]
            ),
            "d2_norm": float(
                summaries[left]["q_top_diagnostic"]
                + summaries[right]["q_top_diagnostic"] - 2.0 * overlap
            ),
        }
        for key, expected in report["comparisons"][f"{left}_{right}"].items():
            _compare_float(comparisons[f"{left}_{right}"][key], expected,
                           f"049 {left}_{right} {key}")
    _require(report["checks"] == {
        "burn_column_changes": False,
        "family_agreement": False,
        "measurement_column_changes": False,
        "measurement_logical_label_changes": False,
        "rhat_ess": False,
        "runtime": True,
    } and report["status"] == "LOCAL_RANDOM_FULL_COLUMN_TRANSPORT_UNRESOLVED",
             "049 failed conclusion changed")
    return {
        "manifest_sha256": manifest_sha,
        "raw_count": len(raw_paths),
        "raw_set_sha256": raw_set_sha,
        "report_sha256": report_sha,
        "status": report["status"],
    }


def _popcounts(values):
    raw = np.asarray(values, dtype=np.uint32)
    byte_view = raw.view(np.uint8).reshape(raw.size, 4)
    table = np.asarray([value.bit_count() for value in range(256)], dtype=np.uint8)
    return table[byte_view].sum(axis=1, dtype=np.int16)


def _conditional_probability_direct(H, syndrome, b_columns, column, target, log_mass):
    candidates = np.arange(1 << H.shape[0], dtype=np.uint32)
    old = np.uint32(b_columns[column])
    a_syndromes = _a_syndromes(H, syndrome, b_columns)
    log_weights = _popcounts(candidates).astype(np.float64) * math.log(0.04 / 0.96)
    for factor in np.flatnonzero(H[column]):
        indices = candidates ^ old ^ np.uint32(a_syndromes[int(factor)])
        log_weights += log_mass[indices]
    maximum = float(log_weights.max())
    weights = np.exp(log_weights - maximum)
    return float(weights[int(target)] / weights.sum(dtype=np.float64))


def _audit_050():
    config_path = EXP102_ROOT / "config/q0_full_column_map_bridge.structure.v0.json"
    config = _load_json(config_path)
    report = _load_json(V050 / "map_bridge_report.json")
    report_sha = _verify_self_hash(report, "report_sha256")
    _registry, H, model, frame, _uniform_seed, _planted, syndrome = _context(config["cell"])
    artifact_path = EXP102_ROOT / config["artifact"]["relpath"]
    _require(sha256_file(artifact_path) == config["artifact"]["file_sha256"],
             "050 artifact hash mismatch")
    with np.load(artifact_path, allow_pickle=False) as artifact:
        anchors = np.asarray(artifact["anchors"], dtype=np.uint8)
    residuals = (
        model.H_check.astype(np.int64) @ anchors.T.astype(np.int64) % 2
    ).T.astype(np.uint8)
    _require(np.array_equal(residuals, np.repeat(syndrome[None, :], 2, axis=0)),
             "050 MAP anchor left hard coset")
    labels = _labels_direct(frame, anchors)
    blocks = [split_hgp_state(anchor, H)[1] for anchor in anchors]
    difference = blocks[0] ^ blocks[1]
    differing_columns = np.flatnonzero(difference.any(axis=0))
    _require(int(difference.sum()) == 6
             and np.array_equal(differing_columns, [11, 17])
             and int(labels[0]) == int(labels[1]),
             "050 frozen bridge geometry changed")
    mass = build_classical_coset_mass(H, 0.04, engine="numba")
    log_mass = np.log(mass)
    for claimed in report["records"]:
        source = int(claimed["source_anchor"])
        target = int(claimed["target_anchor"])
        first = int(claimed["first_column"])
        second = int(claimed["second_column"])
        current = blocks[source].copy()
        b_columns = np.asarray([_mask(current[:, column]) for column in range(24)], dtype=np.uint32)
        target_first = _mask(blocks[target][:, first])
        first_probability = _conditional_probability_direct(
            H, syndrome, b_columns, first, target_first, log_mass,
        )
        current[:, first] = blocks[target][:, first]
        b_columns = np.asarray([_mask(current[:, column]) for column in range(24)], dtype=np.uint32)
        target_second = _mask(blocks[target][:, second])
        second_probability = _conditional_probability_direct(
            H, syndrome, b_columns, second, target_second, log_mass,
        )
        _compare_float(first_probability, claimed["first_probability_given_column"],
                       "050 first bridge probability", tolerance=4e-13)
        _compare_float(second_probability, claimed["second_probability_given_column"],
                       "050 second bridge probability", tolerance=4e-13)
        _compare_float(
            10240 * first_probability / 24.0,
            claimed["expected_first_departures_t1"],
            "050 expected T1 departure", tolerance=4e-13,
        )
    _require(report["status"] == "LOCAL_FULL_COLUMN_MAP_BRIDGE_STRUCTURE_VIABLE"
             and all(report["checks"].values()),
             "050 structural conclusion changed")
    return {
        "anchor_labels_equal": True,
        "difference_columns": differing_columns.tolist(),
        "difference_weight": int(difference.sum()),
        "report_sha256": report_sha,
        "status": report["status"],
    }


def main():
    output = ROOT / "audit_report.json"
    _require(not output.exists(), "audit report already exists")
    checks = {
        "validation_047": _audit_047(),
        "validation_049": _audit_049(),
        "validation_050": _audit_050(),
    }
    core = {
        "audit_version": AUDIT_VERSION,
        "checks": checks,
        "conclusion": (
            "047 and 049 failures are preserved; 050 is only a structural T1 justification"
        ),
        "provenance_limit": (
            "049 source_identity did not enumerate every transitive module; this audit "
            "therefore certifies raw algebra and the failed conclusion, not executable provenance"
        ),
        "status": "INDEPENDENT_AUDIT_PASS_FAILED_RESULTS_PRESERVED",
    }
    report = {**core, "audit_sha256": sha256_json(core)}
    atomic_json(output, report)
    print(canonical_json(report))


if __name__ == "__main__":
    main()
