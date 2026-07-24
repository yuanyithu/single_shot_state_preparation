"""Independent raw-only audit of the frozen validation-057 T1 pair."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from data.expander_code.exp102.exp102_pipeline.io import canonical_json, sha256_json
from data.expander_code.exp102.exp102_pipeline.q0_hgp_collapsed import (
    build_classical_coset_mass,
)
from data.expander_code.exp102.exp102_pipeline.worker import build_model


ROOT = Path(__file__).resolve().parent
CONTROL = (
    ROOT.parent
    / "056_q0_random_full_column_direct_block_t1_m8_v2_20260724"
    / "control/control.npz"
)
RAW_ROOT = ROOT / "t1_pair_raw"
PRIMARY = ROOT / "m8_t1_pair_report.json"
SUCCESS = ROOT / "T1_PAIR_SUCCESS.json"
OUTPUT = ROOT / "independent_t1_pair_audit.json"
PROBE_VERSION = "exp102.q0_hgp_physical_pt.m8_t1_pair.v0"
RAW_VERSION = "exp102.q0_hgp_collapsed_physical_pt.raw.v0"
SAMPLER_VERSION = "exp102.q0_hgp_collapsed_physical_pt.v0"
SOURCE_COMMIT = "a90d3f01641f4ce1432f739d7a76cf6f9128885a"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1 << 20)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def sha256_float64_be(values):
    return hashlib.sha256(
        np.asarray(values, dtype=">f8").tobytes()
    ).hexdigest()


def scalar(data, name):
    value = data[name]
    require(value.shape == (), f"raw scalar shape changed: {name}")
    return value.item()


def state_label(frame, state):
    bits = frame.label_of(np.asarray(state, dtype=np.uint8))
    value = 0
    for bit, entry in enumerate(bits):
        value |= int(entry) << bit
    return value


def state_labels(frame, states):
    signatures = np.zeros(frame.num_qubits, dtype=np.uint64)
    for qubit in range(frame.num_qubits):
        value = np.uint64(0)
        for bit in np.flatnonzero(frame.W_basis[:, qubit]):
            value |= np.uint64(1) << np.uint64(bit)
        signatures[qubit] = value
    labels = np.empty(states.shape[0], dtype=np.uint64)
    for index, state in enumerate(states):
        active = signatures[np.flatnonzero(state)]
        labels[index] = (
            np.bitwise_xor.reduce(active) if active.size else np.uint64(0)
        )
    return labels


def syndromes(model, states):
    states = np.asarray(states, dtype=np.uint8)
    result = np.empty((states.shape[0], model.num_checks), dtype=np.uint8)
    for check in range(model.num_checks):
        support = np.flatnonzero(model.H_check[check])
        result[:, check] = states[:, support].sum(axis=1) & 1
    return result


def packed_b_trace(columns, rows):
    columns = np.asarray(columns, dtype=np.uint32)
    bits = np.empty((columns.shape[0], rows, rows), dtype=np.uint8)
    for row in range(rows):
        bits[:, row, :] = (
            columns >> np.uint32(row) & np.uint32(1)
        ).astype(np.uint8)
    return np.packbits(bits.reshape(columns.shape[0], -1), axis=1, bitorder="little")


def packed_character_means(states_packed, masks_packed):
    table = np.asarray([int(value).bit_count() for value in range(256)], dtype=np.uint8)
    means = np.empty(masks_packed.shape[0], dtype=np.float64)
    for index, mask in enumerate(masks_packed):
        parity = table[np.bitwise_and(states_packed, mask)].sum(axis=1) & 1
        means[index] = np.mean(1.0 - 2.0 * parity.astype(np.float64))
    return means


def logical_character_means(labels, masks):
    labels = np.asarray(labels, dtype=np.uint64)
    masks = np.asarray(masks, dtype=np.uint64)
    means = np.empty(masks.size, dtype=np.float64)
    for start in range(0, masks.size, 128):
        stop = min(masks.size, start + 128)
        parity = np.bitwise_count(
            np.bitwise_and(labels[:, None], masks[None, start:stop])
        ) & np.uint8(1)
        means[start:stop] = np.mean(
            1.0 - 2.0 * parity.astype(np.float64), axis=0,
        )
    return means


def cold_log_likelihood(H, syndrome, b_columns, cold_log_mass):
    rows, columns = H.shape
    Y = syndrome.reshape(rows, columns)
    y_masks = np.zeros(columns, dtype=np.uint32)
    for column in range(columns):
        for row in range(rows):
            y_masks[column] |= np.uint32(int(Y[row, column]) << row)
    result = np.zeros(b_columns.shape[0], dtype=np.float64)
    for factor in range(columns):
        values = np.full(b_columns.shape[0], y_masks[factor], dtype=np.uint32)
        for b_column in np.flatnonzero(H[:, factor]):
            values ^= b_columns[:, int(b_column)]
        result += cold_log_mass[values]
    return result


def expected_config():
    replicas = 32
    beta = np.asarray([
        index ** 2 / (replicas - 1) ** 2 for index in range(replicas)
    ], dtype=np.float64)
    beta[0], beta[-1] = 0.0, 1.0
    coupling = math.log(0.96 / 0.04)
    p_values = 1.0 / (1.0 + np.exp(beta * coupling))
    p_values[0], p_values[-1] = 0.5, 0.04
    config = {
        "beta_exponent": 2,
        "beta_values": beta.tolist(),
        "block_size": 8,
        "burn_rounds": 2048,
        "measurement_rounds": 8192,
        "method_id": "CPPT32",
        "num_replicas": replicas,
        "p_cold": 0.04,
        "p_hot": 0.5,
        "p_values": p_values.tolist(),
        "tempered_terms": "B_prior_and_A_coset_mass",
    }
    gates = {
        "max_b_character_d2_diagnostic": 0.08,
        "max_b_likelihood_delta_per_factor": 0.05,
        "max_b_normalized_weight_delta": 0.02,
        "max_logical_character_d2_diagnostic": 0.08,
        "max_normalized_weight_delta": 0.02,
        "max_q_top_plugin_delta_diagnostic": 0.08,
        "min_adjacent_swap_accepts": 20,
        "min_adjacent_swap_rate": 0.05,
        "min_cold_origin_fraction": 0.5,
        "min_round_trips": 4,
    }
    return config, p_values, gates


def audit_raw(path, family, H, syndrome, fixed, metadata, model, frame,
              logical_masks, b_masks_packed, config, p_values, config_sha,
              cold_log_mass):
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name].copy() for name in archive.files}
    required = {
        "artifact_json", "config_sha256", "elapsed_seconds", "probe_version",
        "source_commit", "sampler__beta_values", "sampler__burn_basis_seen",
        "sampler__burn_cold_b_weights", "sampler__burn_cold_weights",
        "sampler__burn_label", "sampler__burn_labels",
        "sampler__burn_state_packed", "sampler__cold_b_columns",
        "sampler__cold_visits_by_origin", "sampler__engine",
        "sampler__final_label", "sampler__final_origins_by_rung",
        "sampler__final_state_packed", "sampler__hot_visits_by_origin",
        "sampler__initial_label", "sampler__initial_state_packed",
        "sampler__local_attempts_by_rung", "sampler__local_changes_by_rung",
        "sampler__log_mass_tables_sha256", "sampler__measurement_block",
        "sampler__measurement_cold_b_weights", "sampler__measurement_labels",
        "sampler__measurement_residual_weights",
        "sampler__measurement_states_packed", "sampler__measurement_weights",
        "sampler__method_id", "sampler__p_values",
        "sampler__p_values_sha256", "sampler__raw_version",
        "sampler__round_trip_definition", "sampler__round_trips_by_origin",
        "sampler__sampler_config_json", "sampler__sampler_config_sha256",
        "sampler__seed_identity_json", "sampler__swap_accepts",
        "sampler__swap_attempts", "sampler__version",
    }
    require(set(data) == required, f"{family} raw schema changed")
    require(scalar(data, "source_commit") == SOURCE_COMMIT, "raw source changed")
    require(scalar(data, "config_sha256") == config_sha, "raw config changed")
    require(scalar(data, "probe_version") == PROBE_VERSION, "raw probe changed")
    require(scalar(data, "sampler__engine") == "numba", "raw engine changed")
    require(scalar(data, "sampler__method_id") == "CPPT32", "method changed")
    require(scalar(data, "sampler__raw_version") == RAW_VERSION, "raw version changed")
    require(scalar(data, "sampler__version") == SAMPLER_VERSION, "version changed")
    require(
        scalar(data, "sampler__sampler_config_json") == canonical_json(config)
        and scalar(data, "sampler__sampler_config_sha256") == sha256_json(config),
        "sampler config identity changed",
    )
    require(np.array_equal(data["sampler__p_values"], p_values), "p ladder changed")
    require(
        scalar(data, "sampler__p_values_sha256") == sha256_float64_be(p_values),
        "p ladder SHA changed",
    )
    seed = json.loads(scalar(data, "sampler__seed_identity_json"))
    require(
        seed == {
            "cell_fingerprint": metadata["cell_fingerprint"],
            "config_sha256": config_sha,
            "init_family": family,
            "method_id": "CPPT32",
            "registry_sha256": metadata["registry_sha256"],
            "resource_tier": "T1",
            "source_commit": SOURCE_COMMIT,
            "trajectory_index": 0,
            "trajectory_namespace": "q0_hgp_physical_pt_m8_t1_pair_v0",
        },
        "seed identity changed",
    )
    artifact = json.loads(scalar(data, "artifact_json"))
    require(
        artifact["log_mass_tables_sha256"]
        == scalar(data, "sampler__log_mass_tables_sha256")
        and artifact["shape"] == [32, 1 << H.shape[0]],
        "artifact binding changed",
    )
    states = np.unpackbits(
        data["sampler__measurement_states_packed"], axis=1,
        count=model.num_qubits, bitorder="little",
    ).astype(np.uint8, copy=False)
    require(states.shape == (8192, model.num_qubits), "state clock changed")
    residual = syndromes(model, states) ^ syndrome[None, :]
    require(not residual.any(), f"{family} measurement left hard coset")
    require(
        np.array_equal(data["sampler__measurement_residual_weights"], residual.sum(axis=1))
        and np.array_equal(data["sampler__measurement_weights"], states.sum(axis=1)),
        f"{family} cached measurement weights changed",
    )
    labels = state_labels(frame, states)
    require(
        np.array_equal(labels, data["sampler__measurement_labels"]),
        f"{family} cached labels changed",
    )
    b_bits = states[:, H.shape[1] ** 2:].reshape(-1, H.shape[0], H.shape[0])
    b_columns = data["sampler__cold_b_columns"]
    for column in range(H.shape[0]):
        values = np.zeros(states.shape[0], dtype=np.uint32)
        for row in range(H.shape[0]):
            values |= b_bits[:, row, column].astype(np.uint32) << np.uint32(row)
        require(np.array_equal(values, b_columns[:, column]), "B trace/state mismatch")
    b_weights = np.asarray([
        sum(int(value).bit_count() for value in row) for row in b_columns
    ], dtype=np.int32)
    require(
        np.array_equal(b_weights, data["sampler__measurement_cold_b_weights"]),
        f"{family} cached B weights changed",
    )
    initial = np.unpackbits(
        data["sampler__initial_state_packed"], count=model.num_qubits,
        bitorder="little",
    ).astype(np.uint8, copy=False)
    require(
        not (syndromes(model, initial[None, :])[0] ^ syndrome).any(),
        f"{family} initial state left hard coset",
    )
    require(
        int(scalar(data, "sampler__initial_label")) == state_label(frame, initial),
        f"{family} initial label changed",
    )
    if family == "P":
        require(np.array_equal(initial, fixed[0]), "P initializer changed")
    else:
        require(not np.array_equal(initial, fixed[0]), "U collapsed to P initializer")
    require(
        np.array_equal(data["sampler__final_state_packed"], data["sampler__measurement_states_packed"][-1])
        and int(scalar(data, "sampler__final_label")) == int(labels[-1]),
        f"{family} final state changed",
    )
    burn_state = np.unpackbits(
        data["sampler__burn_state_packed"], count=model.num_qubits,
        bitorder="little",
    ).astype(np.uint8, copy=False)
    require(
        not (syndromes(model, burn_state[None, :])[0] ^ syndrome).any()
        and int(scalar(data, "sampler__burn_label")) == state_label(frame, burn_state)
        and int(data["sampler__burn_cold_weights"][-1]) == int(burn_state.sum()),
        f"{family} burn endpoint changed",
    )
    expected_attempts = 24 * 3 * (2048 + 8192)
    require(
        np.all(data["sampler__local_attempts_by_rung"] == expected_attempts)
        and np.all(data["sampler__local_changes_by_rung"] >= 0),
        f"{family} local counters changed",
    )
    require(
        np.all(data["sampler__swap_attempts"] == 5120)
        and int(data["sampler__hot_visits_by_origin"].sum()) == 10240
        and int(data["sampler__cold_visits_by_origin"].sum()) == 10240
        and np.array_equal(
            np.sort(data["sampler__final_origins_by_rung"]), np.arange(32)
        ),
        f"{family} PT counters changed",
    )
    accepts = data["sampler__swap_accepts"].astype(np.float64)
    attempts = data["sampler__swap_attempts"].astype(np.float64)
    transport = {
        "cold_origin_fraction": float(
            np.count_nonzero(data["sampler__cold_visits_by_origin"]) / 32
        ),
        "min_edge_swap_accepts": int(accepts.min()),
        "min_edge_swap_rate": float((accepts / attempts).min()),
        "round_trips": int(data["sampler__round_trips_by_origin"].sum()),
    }
    b_packed = packed_b_trace(b_columns, H.shape[0])
    logical_means = logical_character_means(labels, logical_masks)
    b_means = packed_character_means(b_packed, b_masks_packed)
    likelihood = cold_log_likelihood(H, syndrome, b_columns, cold_log_mass)
    return {
        "b_character_means": b_means,
        "b_likelihood_mean_per_factor": float(likelihood.mean() / H.shape[1]),
        "b_weight_mean_normalized": float(b_weights.mean() / H.shape[0] ** 2),
        "elapsed_seconds": float(scalar(data, "elapsed_seconds")),
        "logical_character_means": logical_means,
        "normalized_weight_mean": float(states.sum(axis=1).mean() / model.num_qubits),
        "q_top_plugin_diagnostic": float(np.mean(logical_means ** 2)),
        "raw_file_sha256": sha256_file(path),
        "transport": transport,
    }


def main():
    require(not OUTPUT.exists(), "independent T1 audit already exists")
    primary = json.loads(PRIMARY.read_text(encoding="utf-8"))
    claimed = primary.pop("report_sha256")
    require(
        hashlib.sha256(json.dumps(
            primary, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")).hexdigest() == claimed,
        "primary report self-hash changed",
    )
    marker = json.loads(SUCCESS.read_text(encoding="utf-8"))
    require(
        marker == {
            "report_sha256": claimed,
            "source_commit": SOURCE_COMMIT,
            "status": "LOCAL_T1_PAIR_UNRESOLVED",
        },
        "success marker identity changed",
    )
    with np.load(CONTROL, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        H = archive["H"].copy()
        syndrome = np.unpackbits(
            archive["syndrome_packed"], count=H.shape[0] * H.shape[1],
            bitorder="little",
        ).astype(np.uint8, copy=False)
        fixed = np.unpackbits(
            archive["fixed_states_packed"], axis=1,
            count=H.shape[1] ** 2 + H.shape[0] ** 2, bitorder="little",
        ).astype(np.uint8, copy=False)
        logical_masks = archive["logical_character_masks"].copy()
        b_masks_packed = archive["b_character_masks_packed"].copy()
    model, frame = build_model(H)
    require(
        model.fingerprint() == metadata["model_fingerprint"]
        and frame.fingerprint() == metadata["frame_fingerprint"],
        "audit model/frame identity changed",
    )
    config, p_values, gates = expected_config()
    seed_config = {
        "config": config,
        "control_content_sha256": metadata["control_content_sha256"],
        "gates": gates,
        "probe_version": PROBE_VERSION,
    }
    config_sha = sha256_json(seed_config)
    require(
        config_sha == primary["config_sha256"]
        and config == primary["config"] and gates == primary["gates"],
        "primary frozen config changed",
    )
    cold_mass = build_classical_coset_mass(H, 0.04, engine="numba")
    cold_log_mass = np.log(cold_mass)
    records = {
        family: audit_raw(
            RAW_ROOT / f"{family}.npz", family, H, syndrome, fixed, metadata,
            model, frame, logical_masks, b_masks_packed, config, p_values,
            config_sha, cold_log_mass,
        )
        for family in ("P", "U")
    }
    P, U = records["P"], records["U"]
    comparison = {
        "b_character_d2_diagnostic": float(np.mean(
            (P["b_character_means"] - U["b_character_means"]) ** 2
        )),
        "b_likelihood_delta_per_factor": abs(
            P["b_likelihood_mean_per_factor"] - U["b_likelihood_mean_per_factor"]
        ),
        "b_normalized_weight_delta": abs(
            P["b_weight_mean_normalized"] - U["b_weight_mean_normalized"]
        ),
        "logical_character_d2_diagnostic": float(np.mean(
            (P["logical_character_means"] - U["logical_character_means"]) ** 2
        )),
        "normalized_weight_delta": abs(
            P["normalized_weight_mean"] - U["normalized_weight_mean"]
        ),
        "q_top_plugin_delta_diagnostic": abs(
            P["q_top_plugin_diagnostic"] - U["q_top_plugin_diagnostic"]
        ),
    }
    public_records = {
        family: {
            name: value for name, value in record.items()
            if not name.endswith("_means")
        }
        for family, record in records.items()
    }
    require(comparison == primary["comparison"], "primary pair comparison changed")
    require(public_records == primary["records"], "primary public records changed")
    transport_pass = all(
        record["transport"]["min_edge_swap_accepts"]
        >= gates["min_adjacent_swap_accepts"]
        and record["transport"]["min_edge_swap_rate"]
        >= gates["min_adjacent_swap_rate"]
        and record["transport"]["round_trips"] >= gates["min_round_trips"]
        and record["transport"]["cold_origin_fraction"]
        >= gates["min_cold_origin_fraction"]
        for record in records.values()
    )
    distribution_pass = (
        comparison["b_character_d2_diagnostic"]
        <= gates["max_b_character_d2_diagnostic"]
        and comparison["b_likelihood_delta_per_factor"]
        <= gates["max_b_likelihood_delta_per_factor"]
        and comparison["b_normalized_weight_delta"]
        <= gates["max_b_normalized_weight_delta"]
        and comparison["logical_character_d2_diagnostic"]
        <= gates["max_logical_character_d2_diagnostic"]
        and comparison["normalized_weight_delta"]
        <= gates["max_normalized_weight_delta"]
        and comparison["q_top_plugin_delta_diagnostic"]
        <= gates["max_q_top_plugin_delta_diagnostic"]
    )
    require(
        not transport_pass and not distribution_pass
        and primary["transport_gate"] is False
        and primary["distribution_necessary_gate"] is False
        and primary["status"] == "LOCAL_T1_PAIR_UNRESOLVED",
        "primary terminal decision changed",
    )
    payload = {
        "audit_version": "exp102.q0_hgp_physical_pt.m8_t1_pair.audit.v0",
        "primary_report_sha256": claimed,
        "raw_file_sha256": {
            family: records[family]["raw_file_sha256"] for family in records
        },
        "source_commit": SOURCE_COMMIT,
        "status": "INDEPENDENT_RAW_ONLY_AUDIT_PASS_LOCAL_T1_PAIR_UNRESOLVED",
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["audit_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    OUTPUT.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
