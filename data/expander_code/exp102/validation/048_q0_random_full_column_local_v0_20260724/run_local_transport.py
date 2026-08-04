"""Run the frozen random-scan full-B-column local transport screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from data.expander_code.exp102.exp102_pipeline.diagnostics import bulk_ess, split_rhat
from data.expander_code.exp102.exp102_pipeline.io import (
    atomic_json,
    atomic_npz,
    canonical_json,
    load_npz_no_pickle,
    sha256_file,
    sha256_json,
)
from data.expander_code.exp102.exp102_pipeline.q0_global import (
    state_label,
    uniform_hard_coset_state,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_collapsed import (
    build_classical_coset_mass,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_full_column_gibbs import (
    build_full_column_candidate_cache,
    build_full_column_workspace,
    full_column_gibbs_update,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_random_full_column import (
    RANDOM_FULL_COLUMN_RAW_VERSION,
    RANDOM_FULL_COLUMN_VERSION,
    RandomFullColumnConfig,
    replay_random_full_column_trajectory,
    run_random_full_column_trajectory,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_screen import _disorder
from data.expander_code.exp102.exp102_pipeline.q0_houdayer_pair import (
    deterministic_low_energy_logical_starts,
)
from data.expander_code.exp102.exp102_pipeline.registry import load_frozen_code, load_registry
from data.expander_code.exp102.exp102_pipeline.seeds import derive_seed
from data.expander_code.exp102.exp102_pipeline.worker import build_model


CONTRACT_VERSION = "exp102.q0_random_full_column.local.v0"
REPORT_VERSION = "exp102.q0_random_full_column.local.report.v0"
ROOT = Path(__file__).resolve().parent
EXP102_ROOT = ROOT.parents[1]
REGISTRY_PATH = EXP102_ROOT / "registry/registry.json"
FAMILIES = ("P", "U", "L")


class LocalTransportError(RuntimeError):
    pass


def _require(condition, message):
    if not condition:
        raise LocalTransportError(message)


def _load_config(path):
    path = Path(path).resolve()
    serialized = path.read_text(encoding="ascii")
    config = json.loads(serialized)
    _require(serialized == canonical_json(config) + "\n", "config is not canonical")
    _require(set(config) == {
        "cell", "config_version", "contract_version", "gates", "initialization",
        "registry_sha256", "resource", "scope", "seed_namespace", "version",
    }, "config schema changed")
    _require(config["version"] == config["contract_version"] == CONTRACT_VERSION
             and config["config_version"]
             == "exp102.q0_random_full_column.local.config.v0",
             "config version changed")
    _require(config["cell"] == {
        "code_id": "m08_c06", "disorder_index": 0,
        "disorder_source": "attempt022", "p": 0.04,
    }, "hard sentinel changed")
    _require(config["initialization"] == {
        "families": ["P", "U", "L"], "l_candidate_orders": [1, 2, 3],
        "trajectory_count_per_family": 4,
    }, "initialization changed")
    _require(config["resource"] == {
        "burn_updates": 64, "measurement_updates": 256,
        "runtime_probe_updates": 2,
        "t1_projection": {
            "burn_updates": 2048, "measurement_updates": 8192,
            "safety_factor": 2.0,
        },
        "trajectory_wall_cap_seconds": 7200.0,
    }, "resource schedule changed")
    _require(config["gates"] == {
        "max_abs_family_q_top_delta": 0.2,
        "max_abs_normalized_b_weight_delta": 0.05,
        "max_abs_normalized_weight_delta": 0.05,
        "max_family_d2_norm": 0.2,
        "max_rhat": 1.2,
        "min_bulk_ess": 50.0,
        "min_burn_column_changes": 4,
        "min_measurement_column_changes": 16,
        "min_measurement_logical_label_changes": 16,
    }, "transport gates changed")
    _require(config["scope"] == {
        "formal_authorization": False,
        "maximum_terminal_status": "LOCAL_RANDOM_FULL_COLUMN_TRANSPORT_VIABLE",
        "posterior_estimation": "short_transport_diagnostic_only",
        "production_authorization": False,
        "remote_authorization": False,
    }, "scope changed")
    _require(config["seed_namespace"]
             == "exp102.q0_random_full_column.local.v0.20260724",
             "seed namespace changed")
    return config, sha256_file(path)


def _source_identity(config_path):
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"), check=True, capture_output=True, text=True,
    ).stdout.strip()
    paths = {
        "config": Path(config_path).resolve(),
        "kernel": EXP102_ROOT / "exp102_pipeline/q0_hgp_random_full_column.py",
        "full_column": EXP102_ROOT / "exp102_pipeline/q0_hgp_full_column_gibbs.py",
        "review": EXP102_ROOT / "reviews/RANDOM_FULL_COLUMN_REVIEW.md",
        "runner": Path(__file__).resolve(),
    }
    core = {
        "files": {name: sha256_file(path) for name, path in paths.items()},
        "source_commit": commit,
    }
    return {**core, "source_identity_sha256": sha256_json(core)}


def _context(config):
    registry = load_registry(REGISTRY_PATH)
    _require(registry["registry_sha256"] == config["registry_sha256"],
             "registry identity changed")
    _unused, code, H = load_frozen_code(REGISTRY_PATH, config["cell"]["code_id"])
    model, frame = build_model(H)
    uniform_seed, planted, syndrome = _disorder(registry, code, model, config["cell"])
    _require(H.shape == (24, 32) and model.num_qubits == 1600
             and model.num_checks == 768 and model.k == 64
             and int(syndrome.sum()) == 160, "hard sentinel dimensions changed")
    logical = deterministic_low_energy_logical_starts(
        model, frame, planted,
        count=config["initialization"]["trajectory_count_per_family"],
        orders=tuple(config["initialization"]["l_candidate_orders"]),
    )
    _require(len({int(state_label(frame, row["state"])) for row in logical}) == len(logical),
             "L starts lost distinct labels")
    return {
        "registry": registry, "H": H, "model": model, "frame": frame,
        "uniform_seed": int(uniform_seed), "planted": planted,
        "syndrome": syndrome, "logical": logical,
    }


def _tasks(context, config, config_sha, source_identity):
    tasks = []
    for family in FAMILIES:
        for index in range(config["initialization"]["trajectory_count_per_family"]):
            core = {
                "config_sha256": config_sha,
                "family": family,
                "index": index,
                "initialization_seed": derive_seed(
                    config["seed_namespace"], config_sha,
                    context["registry"]["registry_sha256"], family, index, "initialize",
                ),
                "observation_seed": derive_seed(
                    config["seed_namespace"], config_sha,
                    context["registry"]["registry_sha256"], family, index, "observe",
                ),
                "burn_update_seed": derive_seed(
                    config["seed_namespace"], config_sha,
                    context["registry"]["registry_sha256"], family, index, "burn",
                ),
                "measurement_update_seed": derive_seed(
                    config["seed_namespace"], config_sha,
                    context["registry"]["registry_sha256"], family, index, "measure",
                ),
                "raw_version": RANDOM_FULL_COLUMN_RAW_VERSION,
                "source_identity_sha256": source_identity["source_identity_sha256"],
            }
            tasks.append({**core, "task_fingerprint": sha256_json(core)})
    _require(len(tasks) == 12 and len({row["task_fingerprint"] for row in tasks}) == 12,
             "task manifest is incomplete")
    core = {
        "config_sha256": config_sha, "contract_version": CONTRACT_VERSION,
        "source_identity": source_identity, "tasks": tasks,
    }
    return {**core, "manifest_sha256": sha256_json(core)}


def _initial_state(context, task):
    if task["family"] == "P":
        return context["planted"].copy()
    if task["family"] == "U":
        return uniform_hard_coset_state(
            context["model"], context["syndrome"], task["initialization_seed"],
        )
    if task["family"] == "L":
        return context["logical"][task["index"]]["state"].copy()
    raise LocalTransportError("unknown family")


def _runtime_probe(context, config, mass, cache, workspace):
    from data.expander_code.exp102.exp102_pipeline.q0_hgp_collapsed import (
        _initial_collapsed_masks,
    )
    load_exp101 = __import__(
        "data.expander_code.exp102.exp102_pipeline.exp101_bridge",
        fromlist=["load_exp101"],
    ).load_exp101
    load_exp101()
    from exp101_certified_src.prng import PortablePrng

    b_columns, a_syndromes, _ = _initial_collapsed_masks(
        context["planted"], context["syndrome"], context["H"],
    )
    log_mass = np.log(mass)
    rng = PortablePrng(derive_seed(
        config["seed_namespace"], "runtime", "outcome_blind",
    ))
    syndrome_matrix = context["syndrome"].reshape(context["H"].shape)
    start = time.perf_counter()
    for update in range(config["resource"]["runtime_probe_updates"]):
        full_column_gibbs_update(
            b_columns, a_syndromes, context["H"], syndrome_matrix,
            update % context["H"].shape[0], log_mass, cache, workspace, rng,
        )
    elapsed = time.perf_counter() - start
    per_update = elapsed / config["resource"]["runtime_probe_updates"]
    t1 = config["resource"]["t1_projection"]
    projection = (
        per_update * (t1["burn_updates"] + t1["measurement_updates"])
        * t1["safety_factor"]
    )
    return {
        "probe_updates": config["resource"]["runtime_probe_updates"],
        "seconds_per_update": per_update,
        "t1_factor_two_projected_seconds": projection,
        "pass": projection <= config["resource"]["trajectory_wall_cap_seconds"],
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


def _family_summary(rows, model):
    labels = np.stack([row["measurement__labels"] for row in rows])
    weights = np.stack([row["measurement__weights"] for row in rows]).astype(np.float64)
    b_weights = np.stack([row["measurement__b_weights"] for row in rows]).astype(np.float64)
    likelihood = np.stack([row["measurement__b_likelihood"] for row in rows])
    freq = _frequencies(labels)
    purity = _within(freq)
    observables = {}
    for name, values in (("weight", weights), ("b_weight", b_weights),
                         ("b_likelihood", likelihood)):
        observables[name] = {
            "bulk_ess": float(bulk_ess(values)),
            "split_rhat": float(split_rhat(values)),
        }
    return {
        "b_weight_mean_normalized": float(b_weights.mean() / (24 * 24)),
        "frequencies": freq,
        "observables": observables,
        "q_top_diagnostic": purity,
        "weight_mean_normalized": float(weights.mean() / model.num_qubits),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config, config_sha = _load_config(args.config)
    source_identity = _source_identity(args.config)
    context = _context(config)
    manifest = _tasks(context, config, config_sha, source_identity)
    _require(not (ROOT / "task_manifest.json").exists(), "task manifest already exists")
    atomic_json(ROOT / "task_manifest.json", manifest)
    mass = build_classical_coset_mass(context["H"], 0.04, engine="reference")
    cache = build_full_column_candidate_cache(24, 0.04)
    workspace = build_full_column_workspace(cache)
    runtime = _runtime_probe(context, config, mass, cache, workspace)
    if not runtime["pass"]:
        core = {
            "config_sha256": config_sha, "contract_version": CONTRACT_VERSION,
            "manifest_sha256": manifest["manifest_sha256"],
            "report_version": REPORT_VERSION, "runtime": runtime,
            "scope": config["scope"], "source_identity": source_identity,
            "status": "LOCAL_RANDOM_FULL_COLUMN_RUNTIME_EXHAUSTED",
        }
        report = {**core, "report_sha256": sha256_json(core)}
        atomic_json(ROOT / "transport_report.json", report)
        print(canonical_json({"status": report["status"], "runtime": runtime}))
        return

    sampler_config = RandomFullColumnConfig(
        p=0.04, burn_updates=config["resource"]["burn_updates"],
        measurement_updates=config["resource"]["measurement_updates"],
    )
    raw_dir = ROOT / "raw"
    raw_dir.mkdir(exist_ok=False)
    loaded = {family: [] for family in FAMILIES}
    for task in manifest["tasks"]:
        initial = _initial_state(context, task)
        raw = run_random_full_column_trajectory(
            context["model"], context["frame"], context["H"], context["syndrome"],
            sampler_config, initial, task["burn_update_seed"],
            task["measurement_update_seed"], task["observation_seed"],
            mass=mass, cache=cache, workspace=workspace,
        )
        replay_random_full_column_trajectory(
            context["model"], context["frame"], context["H"], context["syndrome"],
            sampler_config, initial, task["burn_update_seed"],
            task["measurement_update_seed"], task["observation_seed"], raw,
            mass=mass, cache=cache, workspace=workspace,
        )
        payload = {
            "config_sha256": np.array(config_sha),
            "contract_version": np.array(CONTRACT_VERSION),
            "model_fingerprint": np.array(context["model"].fingerprint()),
            "raw_version": np.array(RANDOM_FULL_COLUMN_RAW_VERSION),
            "syndrome_packed": np.packbits(context["syndrome"], bitorder="little"),
            "task_fingerprint": np.array(task["task_fingerprint"]),
            "task_json": np.array(canonical_json(task)),
            "version": np.array(RANDOM_FULL_COLUMN_VERSION),
            **raw,
        }
        path = raw_dir / f"{task['family']}_{task['index']:02d}.npz"
        atomic_npz(path, **payload)
        loaded[task["family"]].append(raw)
        print(canonical_json({
            "burn_changes": int(raw["burn__counters"][1]),
            "family": task["family"], "index": task["index"],
            "label_changes": int(raw["measurement__counters"][4]),
            "measurement_changes": int(raw["measurement__counters"][1]),
        }), flush=True)

    summaries = {
        family: _family_summary(loaded[family], context["model"])
        for family in FAMILIES
    }
    comparisons = {}
    for left, right in (("P", "U"), ("P", "L"), ("U", "L")):
        overlap = float(np.mean([
            _overlap(a, b)
            for a in summaries[left]["frequencies"]
            for b in summaries[right]["frequencies"]
        ]))
        d2 = (
            summaries[left]["q_top_diagnostic"]
            + summaries[right]["q_top_diagnostic"] - 2.0 * overlap
        )
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
            "d2_norm": float(d2),
        }
    gates = config["gates"]
    checks = {
        "burn_column_changes": all(
            int(row["burn__counters"][1]) >= gates["min_burn_column_changes"]
            for rows in loaded.values() for row in rows
        ),
        "family_agreement": all(
            row["abs_q_top_delta"] <= gates["max_abs_family_q_top_delta"]
            and row["abs_b_weight_delta"] <= gates["max_abs_normalized_b_weight_delta"]
            and row["abs_weight_delta"] <= gates["max_abs_normalized_weight_delta"]
            and max(0.0, row["d2_norm"]) <= gates["max_family_d2_norm"]
            for row in comparisons.values()
        ),
        "measurement_column_changes": all(
            int(row["measurement__counters"][1])
            >= gates["min_measurement_column_changes"]
            for rows in loaded.values() for row in rows
        ),
        "measurement_logical_label_changes": all(
            int(row["measurement__counters"][4])
            >= gates["min_measurement_logical_label_changes"]
            for rows in loaded.values() for row in rows
        ),
        "rhat_ess": all(
            observable["split_rhat"] <= gates["max_rhat"]
            and observable["bulk_ess"] >= gates["min_bulk_ess"]
            for summary in summaries.values()
            for observable in summary["observables"].values()
        ),
        "runtime": runtime["pass"],
    }
    status = (
        "LOCAL_RANDOM_FULL_COLUMN_TRANSPORT_VIABLE"
        if all(checks.values())
        else "LOCAL_RANDOM_FULL_COLUMN_TRANSPORT_UNRESOLVED"
    )
    raw_files = sorted(raw_dir.glob("*.npz"))
    raw_set_sha = hashlib.sha256("".join(
        f"{path.name}:{sha256_file(path)}\n" for path in raw_files
    ).encode("ascii")).hexdigest()
    core = {
        "checks": checks, "comparisons": comparisons,
        "config_sha256": config_sha, "contract_version": CONTRACT_VERSION,
        "manifest_sha256": manifest["manifest_sha256"],
        "raw_count": len(raw_files), "raw_set_sha256": raw_set_sha,
        "report_version": REPORT_VERSION, "runtime": runtime,
        "scope": config["scope"], "source_identity": source_identity,
        "status": status,
        "summaries": {
            family: {key: value for key, value in summary.items() if key != "frequencies"}
            for family, summary in summaries.items()
        },
    }
    report = {**core, "report_sha256": sha256_json(core)}
    atomic_json(ROOT / "transport_report.json", report)
    print(canonical_json({
        "checks": checks, "report_sha256": report["report_sha256"],
        "status": status,
    }))


if __name__ == "__main__":
    main()
