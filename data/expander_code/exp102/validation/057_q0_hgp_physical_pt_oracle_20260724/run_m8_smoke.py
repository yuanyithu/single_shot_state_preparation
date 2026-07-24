"""Run a frozen two-family CPPT32 m8 runtime/transport smoke."""

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
OUTPUT = ROOT / "m8_smoke_report.json"
CONFIG = CollapsedPhysicalPtConfig("CPPT32", 0.04, 8, 32)
PROBE_VERSION = "exp102.q0_hgp_physical_pt.m8_smoke.v0"


def _source_commit():
    subprocess.run(
        ["git", "diff", "--quiet"], cwd=PROJECT_ROOT, check=True,
    )
    subprocess.run(
        ["git", "diff", "--cached", "--quiet"], cwd=PROJECT_ROOT, check=True,
    )
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True,
    ).strip()


def _array_digest(result):
    digest = hashlib.sha256()
    for name in sorted(result):
        value = np.asarray(result[name])
        if value.dtype.kind not in "biuf":
            continue
        digest.update(name.encode("ascii") + b"\0")
        digest.update(value.dtype.str.encode("ascii") + b"\0")
        digest.update(np.asarray(value.shape, dtype=">u8").tobytes())
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def main():
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
    model, frame = build_model(H)
    if (
        model.fingerprint() != metadata["model_fingerprint"]
        or frame.fingerprint() != metadata["frame_fingerprint"]
    ):
        raise RuntimeError("m8 smoke model/frame identity changed")
    seed_config = {
        "config": CONFIG.as_dict(),
        "control_content_sha256": metadata["control_content_sha256"],
        "probe_version": PROBE_VERSION,
    }
    config_sha = sha256_json(seed_config)

    artifact_started = time.perf_counter()
    artifact = build_physical_pt_mass_artifact(
        H, CONFIG.p_values, "numba",
    )
    artifact_seconds = time.perf_counter() - artifact_started
    trajectories = {}
    for family in ("P", "U"):
        identity = PhysicalPtSeedIdentity(
            source_commit=source_commit,
            config_sha256=config_sha,
            registry_sha256=metadata["registry_sha256"],
            cell_fingerprint=metadata["cell_fingerprint"],
            method_id=CONFIG.method_id,
            resource_tier="LOCAL_SMOKE_8_32",
            init_family=family,
            trajectory_index=0,
            trajectory_namespace="q0_hgp_physical_pt_m8_smoke_v0",
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
        attempts = result["swap_attempts"].astype(np.float64)
        rates = np.divide(
            result["swap_accepts"], attempts,
            out=np.zeros_like(attempts), where=attempts > 0,
        )
        trajectories[family] = {
            "burn_final_b_weight": int(result["burn_cold_b_weights"][-1]),
            "burn_final_weight": int(result["burn_cold_weights"][-1]),
            "cold_b_weight_mean": float(
                np.mean(result["measurement_cold_b_weights"])
            ),
            "cold_weight_mean": float(np.mean(result["measurement_weights"])),
            "elapsed_seconds": float(elapsed),
            "final_label": int(result["final_label"]),
            "initial_label": int(result["initial_label"]),
            "minimum_swap_rate": float(rates.min(initial=1.0)),
            "origins_with_round_trip": int(
                np.count_nonzero(result["round_trips_by_origin"])
            ),
            "raw_numeric_sha256": _array_digest(result),
            "round_trips_total": int(result["round_trips_by_origin"].sum()),
            "swap_rates": rates.tolist(),
        }
    worst = max(value["elapsed_seconds"] for value in trajectories.values())
    projected_t1_seconds = worst * (2048 + 8192) / (
        CONFIG.burn_rounds + CONFIG.measurement_rounds
    )
    payload = {
        "artifact": artifact.as_dict(),
        "artifact_build_seconds": float(artifact_seconds),
        "config": CONFIG.as_dict(),
        "config_sha256": config_sha,
        "control_content_sha256": metadata["control_content_sha256"],
        "probe_version": PROBE_VERSION,
        "projected_t1_seconds_per_trajectory": float(projected_t1_seconds),
        "source_commit": source_commit,
        "status": (
            "LOCAL_SMOKE_RUNTIME_ELIGIBLE"
            if projected_t1_seconds <= 7200.0
            else "LOCAL_SMOKE_RUNTIME_EXHAUSTED"
        ),
        "trajectories": trajectories,
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
