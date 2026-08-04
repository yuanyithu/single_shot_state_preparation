"""Independent allow_pickle=False audit of the frozen BP-IMH v1 raw set."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.expander_code.exp102.exp102_pipeline.q0_bp_systematic import (
    build_bp_systematic_proposal,
)
from data.expander_code.exp102.exp102_pipeline.q0_global import uniform_hard_coset_state
from data.expander_code.exp102.exp102_pipeline.q0_hgp_screen import _disorder
from data.expander_code.exp102.exp102_pipeline.q0_houdayer_pair import (
    deterministic_low_energy_logical_starts,
)
from data.expander_code.exp102.exp102_pipeline.registry import load_frozen_code, load_registry
from data.expander_code.exp102.exp102_pipeline.worker import build_model


ROOT = Path(__file__).resolve().parent
EXP102_ROOT = ROOT.parents[1]
CONFIG_PATH = EXP102_ROOT / "config/q0_bp_imh.local.v1.json"
REGISTRY_PATH = EXP102_ROOT / "registry/registry.json"
REPORT_PATH = ROOT / "bp_imh_report.json"
RECEIPT_PATH = ROOT / "run_receipt.json"
MANIFEST_PATH = ROOT / "task_manifest.json"
OUTPUT_PATH = ROOT / "independent_audit_report.json"
AUDIT_VERSION = "exp102.q0_bp_imh.local.independent_audit.v1"


class AuditError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value):
    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path, self_hash=None):
    value = json.loads(Path(path).read_text(encoding="ascii"))
    if self_hash is not None:
        core = {key: item for key, item in value.items() if key != self_hash}
        require(value.get(self_hash) == sha256_json(core), f"self-hash mismatch: {path}")
    return value


def scalar(value, kind):
    array = np.asarray(value)
    require(array.shape == (), "raw scalar changed shape")
    return kind(array.item())


def order(name, size):
    if name == "forward":
        return np.arange(size, dtype=np.int32)
    return np.arange(size - 1, -1, -1, dtype=np.int32)


def label_values(W_basis, states):
    states = np.asarray(states, dtype=np.uint8)
    labels = np.zeros(states.shape[0], dtype=np.uint64)
    for bit, row in enumerate(np.asarray(W_basis, dtype=np.uint8)):
        support = np.flatnonzero(row)
        parity = states[:, support].sum(axis=1, dtype=np.int64) & 1
        labels |= parity.astype(np.uint64) << np.uint64(bit)
    return labels


def assert_hard_coset(H_check, states, syndrome, context):
    states = np.asarray(states, dtype=np.uint8)
    for row, check in enumerate(np.asarray(H_check, dtype=np.uint8)):
        support = np.flatnonzero(check)
        parity = states[:, support].sum(axis=1, dtype=np.int64) & 1
        require(np.all(parity == int(syndrome[row])), f"hard-coset failure: {context}")


def source_log_probabilities(proposals, state):
    return np.asarray([
        float(proposal.log_probability_state(state)) for proposal in proposals
    ], dtype=np.float64)


def combined_log_q(proposals, state):
    values = source_log_probabilities(proposals, state) + math.log(0.5)
    maximum = float(values.max())
    return maximum + math.log(float(np.exp(values - maximum).sum(dtype=np.float64)))


def frequencies(labels):
    result = []
    for row in np.asarray(labels, dtype=np.uint64):
        unique, counts = np.unique(row, return_counts=True)
        result.append({int(key): int(count) / row.size for key, count in zip(unique, counts)})
    return result


def overlap(left, right):
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(key, 0.0) for key, value in left.items())


def within(values):
    pairs = [
        overlap(values[left], values[right])
        for left in range(len(values)) for right in range(left + 1, len(values))
    ]
    return float(np.mean(pairs))


def cross(left, right):
    return float(np.mean([overlap(a, b) for a in left for b in right]))


def q_top(labels):
    uniform = math.ldexp(1.0, -64)
    return (within(frequencies(labels)) - uniform) / (1.0 - uniform)


def d2_norm(left_labels, right_labels):
    uniform = math.ldexp(1.0, -64)
    left, right = frequencies(left_labels), frequencies(right_labels)
    return (within(left) + within(right) - 2.0 * cross(left, right)) / (1.0 - uniform)


def atomic_json(path, value):
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="ascii") as handle:
            handle.write(canonical_json(value) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def build_context(config):
    registry = load_registry(REGISTRY_PATH)
    require(registry["registry_sha256"] == config["registry_sha256"], "registry changed")
    _unused, code, H = load_frozen_code(REGISTRY_PATH, config["cell"]["code_id"])
    model, frame = build_model(H)
    _uniform_seed, planted, syndrome = _disorder(registry, code, model, config["cell"])
    proposals = tuple(build_bp_systematic_proposal(
        model, syndrome, config["cell"]["p"],
        column_order=order(name, model.num_qubits),
        bp_iterations=config["bp"]["iterations"],
        bp_damping=config["bp"]["damping"],
        bp_llr_cap=config["bp"]["llr_cap"],
        min_probability=config["bp"]["min_probability"],
        component_weights=config["proposal_component_weights"],
    ) for name in ("forward", "reverse"))
    logical_starts = deterministic_low_energy_logical_starts(
        model, frame, planted, count=config["initialization"]["l_catalog_count"],
        orders=tuple(config["initialization"]["l_candidate_orders"]),
    )
    return registry, model, frame, planted, syndrome, proposals, logical_starts


def expected_initial(task, model, planted, syndrome, logical_starts):
    family = task["init_family"]
    if family == "P":
        return planted.copy()
    if family == "U":
        return uniform_hard_coset_state(model, syndrome, task["initialization_seed"])
    if family == "L":
        return logical_starts[task["trajectory_index"]]["state"].copy()
    raise AuditError("unknown initialization family")


def audit_raw(path, task, config, model, frame, syndrome, proposals, initial):
    with np.load(path, allow_pickle=False) as raw:
        require(scalar(raw["contract_version"], str) == config["contract_version"],
                "raw contract mismatch")
        require(scalar(raw["task_json"], str) == canonical_json(task), "raw task mismatch")
        require(scalar(raw["task_fingerprint"], str) == task["task_fingerprint"],
                "raw task fingerprint mismatch")
        require(scalar(raw["config_sha256"], str) == sha256_file(CONFIG_PATH),
                "raw config mismatch")
        require(scalar(raw["source_identity_sha256"], str)
                == task["source_identity_sha256"], "raw source mismatch")
        require(np.array_equal(raw["syndrome_packed"], np.packbits(syndrome, bitorder="little")),
                "raw syndrome mismatch")
        require(scalar(raw["sampler__seed"], int) == task["sampler_seed"], "raw seed mismatch")

        state = np.unpackbits(
            raw["sampler__initial_state_packed"], count=model.num_qubits, bitorder="little",
        ).astype(np.uint8, copy=False)
        require(np.array_equal(state, initial), "raw initial state mismatch")
        assert_hard_coset(model.H_check, state[None, :], syndrome, "initial")
        weight = int(state.sum())
        label = int(label_values(frame.W_basis, state[None, :])[0])
        current_log_q = combined_log_q(proposals, state)
        require(scalar(raw["sampler__initial_weight"], int) == weight
                and scalar(raw["sampler__initial_label"], int) == label
                and scalar(raw["sampler__initial_log_q"], float) == current_log_q,
                "raw initial derived values mismatch")

        summary = {}
        transcript_steps = 0
        for stage, steps in (("burn", 256), ("measurement", 2048)):
            prefix = f"sampler__{stage}_"
            proposed = np.unpackbits(
                raw[prefix + "proposal_states_packed"], axis=1,
                count=model.num_qubits, bitorder="little",
            ).astype(np.uint8, copy=False)
            stored = np.unpackbits(
                raw[prefix + "states_packed"], axis=1,
                count=model.num_qubits, bitorder="little",
            ).astype(np.uint8, copy=False)
            require(proposed.shape == stored.shape == (steps, model.num_qubits),
                    "raw transcript shape mismatch")
            assert_hard_coset(model.H_check, proposed, syndrome, stage)
            proposed_weights = proposed.sum(axis=1, dtype=np.int64)
            proposed_labels = label_values(frame.W_basis, proposed)
            require(np.array_equal(raw[prefix + "proposal_weights"], proposed_weights)
                    and np.array_equal(raw[prefix + "proposal_labels"], proposed_labels),
                    "raw proposal derived values mismatch")

            for index, proposal_state in enumerate(proposed):
                source = int(raw[prefix + "proposal_source_indices"][index])
                component = int(raw[prefix + "proposal_component_indices"][index])
                require(0 <= source < 2 and 0 <= component < 3,
                        "raw proposal provenance mismatch")
                source_values = source_log_probabilities(proposals, proposal_state)
                proposal_log_q = combined_log_q(proposals, proposal_state)
                require(float(raw[prefix + "proposal_source_log_q"][index])
                        == float(source_values[source])
                        and float(raw[prefix + "proposal_log_q"][index]) == proposal_log_q,
                        "raw proposal density mismatch")
                ratio = (
                    (int(proposed_weights[index]) - weight)
                    * math.log(config["cell"]["p"] / (1.0 - config["cell"]["p"]))
                    + current_log_q - proposal_log_q
                )
                clipped = min(0.0, ratio)
                probability = 1.0 if clipped == 0.0 else math.exp(clipped)
                decision = float(raw[prefix + "acceptance_uniforms"][index]) < probability
                changed = bool(decision and not np.array_equal(state, proposal_state))
                if decision:
                    state = proposal_state.copy()
                    weight = int(proposed_weights[index])
                    label = int(proposed_labels[index])
                    current_log_q = proposal_log_q
                require(float(raw[prefix + "log_acceptance"][index]) == clipped
                        and bool(raw[prefix + "accepted"][index]) == decision
                        and bool(raw[prefix + "state_changed"][index]) == changed
                        and np.array_equal(stored[index], state)
                        and int(raw[prefix + "weights"][index]) == weight
                        and int(raw[prefix + "labels"][index]) == label
                        and float(raw[prefix + "current_log_q"][index]) == current_log_q,
                        f"raw MH transcript mismatch: {stage}:{index}")
                transcript_steps += 1
            endpoint = "burn_end" if stage == "burn" else "final"
            require(np.array_equal(
                raw[f"sampler__{endpoint}_state_packed"], np.packbits(state, bitorder="little"),
            ) and scalar(raw[f"sampler__{endpoint}_weight"], int) == weight
                and scalar(raw[f"sampler__{endpoint}_label"], int) == label
                and scalar(raw[f"sampler__{endpoint}_log_q"], float) == current_log_q,
                "raw endpoint mismatch")
            summary[f"{stage}_state_changes"] = int(raw[prefix + "state_changed"].sum())
            if stage == "measurement":
                summary["labels"] = raw[prefix + "labels"].astype(np.uint64).copy()
                summary["weights"] = raw[prefix + "weights"].astype(np.float64).copy()
                b_start = 32 ** 2
                summary["b_weights"] = stored[:, b_start:b_start + 24 ** 2].sum(
                    axis=1, dtype=np.int64,
                ).astype(np.float64)
        summary["transcript_steps"] = transcript_steps
        return summary


def run():
    require(not OUTPUT_PATH.exists(), "refusing to overwrite independent audit")
    config_text = CONFIG_PATH.read_text(encoding="ascii")
    config = json.loads(config_text)
    require(config_text == canonical_json(config) + "\n", "config is not canonical")
    manifest = load_json(MANIFEST_PATH, "manifest_sha256")
    receipt = load_json(RECEIPT_PATH, "receipt_sha256")
    report = load_json(REPORT_PATH, "report_sha256")
    require(manifest["config_sha256"] == sha256_file(CONFIG_PATH), "manifest config mismatch")
    require(receipt["manifest_sha256"] == manifest["manifest_sha256"],
            "receipt manifest mismatch")
    require(report["manifest_sha256"] == manifest["manifest_sha256"]
            and report["receipt_sha256"] == receipt["receipt_sha256"],
            "report upstream identity mismatch")
    require(len(manifest["tasks"]) == len(receipt["raw_records"]) == 24,
            "raw task count mismatch")
    require(receipt["replay_count"] == 24, "runner replay count mismatch")
    require(receipt["raw_set_sha256"] == sha256_json(receipt["raw_records"]),
            "raw-set hash mismatch")

    registry, model, frame, planted, syndrome, proposals, logical_starts = build_context(config)
    source_shas = [proposal.proposal_sha256 for proposal in proposals]
    require(source_shas == manifest["tasks"][0]["source_proposal_sha256"],
            "proposal source identity mismatch")
    combined_core = {
        "source_proposal_sha256": source_shas,
        "version": "exp102.q0_bp_imh.v0",
        "weights": ["0.5", "0.5"],
    }
    combined_sha = sha256_json(combined_core)
    require(combined_sha == manifest["tasks"][0]["combined_proposal_sha256"],
            "combined proposal identity mismatch")

    task_by_fingerprint = {task["task_fingerprint"]: task for task in manifest["tasks"]}
    require(len(task_by_fingerprint) == 24, "duplicate task fingerprint")
    require(len({task["sampler_seed"] for task in manifest["tasks"]}) == 24
            and len({task["initialization_seed"] for task in manifest["tasks"]}) == 24,
            "duplicate raw seed identity")
    results = []
    for record in receipt["raw_records"]:
        task = task_by_fingerprint[record["task_fingerprint"]]
        path = EXP102_ROOT / record["path"]
        require(path.is_file() and sha256_file(path) == record["raw_sha256"],
                "raw file hash mismatch")
        initial = expected_initial(task, model, planted, syndrome, logical_starts)
        result = audit_raw(path, task, config, model, frame, syndrome, proposals, initial)
        results.append((task, result))

    family = {}
    for name in ("P", "U", "L"):
        items = [item for task, item in results if task["init_family"] == name]
        require(len(items) == 8, "family raw count mismatch")
        labels = np.stack([item["labels"] for item in items])
        weights = np.stack([item["weights"] for item in items])
        b_weights = np.stack([item["b_weights"] for item in items])
        family[name] = {
            "labels": labels,
            "q_top": q_top(labels),
            "normalized_weight_mean": float(weights.mean() / model.num_qubits),
            "normalized_b_weight_mean": float(b_weights.mean() / (24 ** 2)),
            "measurement_state_changes": [item["measurement_state_changes"] for item in items],
            "burn_state_changes": [item["burn_state_changes"] for item in items],
        }
    comparisons = {}
    for left, right in (("P", "U"), ("P", "L"), ("U", "L")):
        comparisons[f"{left}_vs_{right}"] = {
            "absolute_q_top_difference": abs(family[left]["q_top"] - family[right]["q_top"]),
            "d2_norm": d2_norm(family[left]["labels"], family[right]["labels"]),
            "absolute_weight_difference": abs(
                family[left]["normalized_weight_mean"]
                - family[right]["normalized_weight_mean"]
            ),
            "absolute_b_weight_difference": abs(
                family[left]["normalized_b_weight_mean"]
                - family[right]["normalized_b_weight_mean"]
            ),
        }
    minimum_changes = min(
        change for value in family.values() for change in value["measurement_state_changes"]
    )
    unresolved = bool(
        minimum_changes < config["gates"]["min_measurement_state_changes"]
        or any(value["absolute_q_top_difference"] > config["gates"]["max_abs_q_top_delta"]
               for value in comparisons.values())
        or any(max(0.0, value["d2_norm"]) > config["gates"]["character_d2_max"]
               for value in comparisons.values())
    )
    require(unresolved, "independent minimal fail-closed gates unexpectedly pass")
    require(report["terminal_status"] == "LOCAL_BP_IMH_TRANSPORT_UNRESOLVED",
            "runner terminal status differs from independent audit")

    public_family = {
        name: {key: value for key, value in values.items() if key != "labels"}
        for name, values in family.items()
    }
    core = {
        "acceptance_transcript_count": sum(
            item["transcript_steps"] for _task, item in results
        ),
        "audit_version": AUDIT_VERSION,
        "combined_proposal_sha256": combined_sha,
        "comparisons": comparisons,
        "config_sha256": sha256_file(CONFIG_PATH),
        "family_summaries": public_family,
        "manifest_sha256": manifest["manifest_sha256"],
        "raw_count": len(results),
        "raw_set_sha256": receipt["raw_set_sha256"],
        "receipt_sha256": receipt["receipt_sha256"],
        "registry_sha256": registry["registry_sha256"],
        "report_sha256": report["report_sha256"],
        "runner_terminal_status": report["terminal_status"],
        "status": "INDEPENDENT_RAW_AUDIT_PASS_UNRESOLVED_CONFIRMED",
    }
    output = {**core, "audit_sha256": sha256_json(core)}
    atomic_json(OUTPUT_PATH, output)
    print(output["status"])
    print(output["audit_sha256"])


if __name__ == "__main__":
    run()
