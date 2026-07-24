"""Run the frozen one-P/one-U CPPT32 T1 necessary-condition probe."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from data.expander_code.exp102.exp102_pipeline.io import sha256_json
from data.expander_code.exp102.exp102_pipeline.q0_global import (
    uniform_hard_coset_state,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_physical_pt import (
    CollapsedPhysicalPtConfig,
    PhysicalPtSeedIdentity,
    build_physical_pt_mass_artifact,
    run_collapsed_physical_pt_trajectory,
)
from data.expander_code.exp102.exp102_pipeline.worker import build_model


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[4]
CONTROL = (
    ROOT.parent
    / "056_q0_random_full_column_direct_block_t1_m8_v2_20260724"
    / "control/control.npz"
)
RAW_ROOT = ROOT / "t1_pair_raw"
OUTPUT = ROOT / "m8_t1_pair_report.json"
CONFIG = CollapsedPhysicalPtConfig("CPPT32", 0.04, 2048, 8192)
PROBE_VERSION = "exp102.q0_hgp_physical_pt.m8_t1_pair.v0"


def _source_commit():
    subprocess.run(["git", "diff", "--quiet"], cwd=PROJECT_ROOT, check=True)
    subprocess.run(
        ["git", "diff", "--cached", "--quiet"], cwd=PROJECT_ROOT, check=True,
    )
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True,
    ).strip()


def _sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1 << 20)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def _packed_b_trace(columns, rows):
    columns = np.asarray(columns, dtype=np.uint32)
    bits = np.empty((columns.shape[0], rows, rows), dtype=np.uint8)
    for row in range(rows):
        bits[:, row, :] = (
            columns >> np.uint32(row) & np.uint32(1)
        ).astype(np.uint8)
    return np.packbits(bits.reshape(columns.shape[0], -1), axis=1, bitorder="little")


def _packed_character_means(states_packed, masks_packed):
    table = np.asarray([int(value).bit_count() for value in range(256)], dtype=np.uint8)
    means = np.empty(masks_packed.shape[0], dtype=np.float64)
    for index, mask in enumerate(masks_packed):
        parity = table[np.bitwise_and(states_packed, mask)].sum(axis=1) & 1
        means[index] = np.mean(1.0 - 2.0 * parity.astype(np.float64))
    return means


def _logical_character_means(labels, masks):
    labels = np.asarray(labels, dtype=np.uint64)
    masks = np.asarray(masks, dtype=np.uint64)
    means = np.empty(masks.size, dtype=np.float64)
    for start in range(0, masks.size, 128):
        stop = min(masks.size, start + 128)
        values = np.bitwise_and(labels[:, None], masks[None, start:stop])
        parity = np.bitwise_count(values) & np.uint8(1)
        means[start:stop] = np.mean(
            1.0 - 2.0 * parity.astype(np.float64), axis=0,
        )
    return means


def _cold_log_likelihood(H, syndrome, b_columns, cold_log_mass):
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


def _transport_summary(result):
    attempts = result["swap_attempts"].astype(np.float64)
    accepts = result["swap_accepts"].astype(np.float64)
    rates = accepts / np.maximum(attempts, 1.0)
    return {
        "cold_origin_fraction": float(
            np.count_nonzero(result["cold_visits_by_origin"])
            / result["cold_visits_by_origin"].size
        ),
        "min_edge_swap_accepts": int(accepts.min()),
        "min_edge_swap_rate": float(rates.min()),
        "round_trips": int(result["round_trips_by_origin"].sum()),
    }


def main():
    if OUTPUT.exists() or RAW_ROOT.exists():
        raise RuntimeError("T1 pair output already exists")
    source_commit = _source_commit()
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
    if (
        model.fingerprint() != metadata["model_fingerprint"]
        or frame.fingerprint() != metadata["frame_fingerprint"]
    ):
        raise RuntimeError("T1 pair model/frame identity changed")
    seed_config = {
        "config": CONFIG.as_dict(),
        "control_content_sha256": metadata["control_content_sha256"],
        "gates": {
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
        },
        "probe_version": PROBE_VERSION,
    }
    config_sha = sha256_json(seed_config)
    artifact = build_physical_pt_mass_artifact(H, CONFIG.p_values, "numba")
    RAW_ROOT.mkdir(parents=False)
    records = {}
    for family in ("P", "U"):
        identity = PhysicalPtSeedIdentity(
            source_commit=source_commit,
            config_sha256=config_sha,
            registry_sha256=metadata["registry_sha256"],
            cell_fingerprint=metadata["cell_fingerprint"],
            method_id=CONFIG.method_id,
            resource_tier="T1",
            init_family=family,
            trajectory_index=0,
            trajectory_namespace="q0_hgp_physical_pt_m8_t1_pair_v0",
        )
        initial = (
            fixed[0].copy() if family == "P"
            else uniform_hard_coset_state(
                model, syndrome, identity.seed("initialize"),
            )
        )
        started = time.perf_counter()
        result = run_collapsed_physical_pt_trajectory(
            model, frame, H, syndrome, CONFIG, identity, initial,
            engine="numba", mass_artifact=artifact,
        )
        elapsed = time.perf_counter() - started
        raw_path = RAW_ROOT / f"{family}.npz"
        np.savez(
            raw_path,
            **{f"sampler__{name}": value for name, value in result.items()},
            artifact_json=np.array(json.dumps(
                artifact.as_dict(), sort_keys=True, separators=(",", ":"),
            )),
            config_sha256=np.array(config_sha),
            elapsed_seconds=np.array(elapsed, dtype=np.float64),
            probe_version=np.array(PROBE_VERSION),
            source_commit=np.array(source_commit),
        )
        b_packed = _packed_b_trace(result["cold_b_columns"], H.shape[0])
        logical_means = _logical_character_means(
            result["measurement_labels"], logical_masks,
        )
        b_means = _packed_character_means(b_packed, b_masks_packed)
        likelihood = _cold_log_likelihood(
            H, syndrome, result["cold_b_columns"],
            artifact.log_mass_tables[-1],
        )
        records[family] = {
            "b_character_means": b_means,
            "b_likelihood_mean_per_factor": float(likelihood.mean() / H.shape[1]),
            "b_weight_mean_normalized": float(
                result["measurement_cold_b_weights"].mean() / H.shape[0] ** 2
            ),
            "elapsed_seconds": float(elapsed),
            "logical_character_means": logical_means,
            "normalized_weight_mean": float(
                result["measurement_weights"].mean() / model.num_qubits
            ),
            "q_top_plugin_diagnostic": float(np.mean(logical_means ** 2)),
            "raw_file_sha256": _sha256_file(raw_path),
            "transport": _transport_summary(result),
        }
    gates = seed_config["gates"]
    P, U = records["P"], records["U"]
    comparison = {
        "b_character_d2_diagnostic": float(np.mean(
            (P["b_character_means"] - U["b_character_means"]) ** 2
        )),
        "b_likelihood_delta_per_factor": abs(
            P["b_likelihood_mean_per_factor"]
            - U["b_likelihood_mean_per_factor"]
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
    public_records = {
        family: {
            name: value for name, value in record.items()
            if not name.endswith("_means")
        }
        for family, record in records.items()
    }
    payload = {
        "artifact": artifact.as_dict(),
        "comparison": comparison,
        "config": CONFIG.as_dict(),
        "config_sha256": config_sha,
        "distribution_necessary_gate": bool(distribution_pass),
        "gates": gates,
        "probe_version": PROBE_VERSION,
        "records": public_records,
        "scope": {
            "formal_authorization": False,
            "maximum_status": "LOCAL_T1_PAIR_NECESSARY_GATES_PASS",
            "production_authorization": False,
            "replicated_convergence_claim": False,
        },
        "source_commit": source_commit,
        "status": (
            "LOCAL_T1_PAIR_NECESSARY_GATES_PASS"
            if transport_pass and distribution_pass
            else "LOCAL_T1_PAIR_UNRESOLVED"
        ),
        "transport_gate": bool(transport_pass),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["report_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    OUTPUT.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
