"""Run and analyze the frozen local hybrid row-column B-transport pilot."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import hashlib
from importlib import import_module
import json
import math
from pathlib import Path
import re
import sys
import time

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.expander_code.exp102.exp102_pipeline.q0_global import (  # noqa: E402
    uniform_hard_coset_state,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_collapsed import (  # noqa: E402
    build_classical_coset_mass,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_full_column_gibbs import (  # noqa: E402
    build_full_column_direct_block_cache,
    build_full_column_direct_block_workspace,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_full_row_gibbs import (  # noqa: E402
    build_full_row_elimination_plan,
)
from data.expander_code.exp102.exp102_pipeline.q0_hgp_hybrid_gibbs import (  # noqa: E402
    HYBRID_COUNTERS,
    HYBRID_GIBBS_ENGINE,
    HYBRID_GIBBS_METHOD_ID,
    HYBRID_GIBBS_VERSION,
    HybridGibbsConfig,
    replay_hybrid_gibbs_trajectory,
    run_hybrid_gibbs_trajectory,
)
from data.expander_code.exp102.exp102_pipeline.seeds import derive_seed  # noqa: E402


ROOT = Path(__file__).resolve().parent
EXP102_ROOT = ROOT.parents[1]
CONFIG_PATH = ROOT / "pilot_config.json"
CONTROL_WORKFLOW = (
    "data.expander_code.exp102.validation."
    "056_q0_random_full_column_direct_block_t1_m8_v2_20260724.workflow"
)
VERSION = "exp102.q0_hybrid_row_column.local_pilot.v0"
RAW_VERSION = "exp102.q0_hybrid_row_column.local_pilot.raw.v0"
SCHEDULE_VERSION = "exp102.q0_hybrid_row_column.local_pilot.schedule.v0"
REPORT_VERSION = "exp102.q0_hybrid_row_column.local_pilot.report.v0"
SHA_RE = re.compile(r"[0-9a-f]{64}")
COMMIT_RE = re.compile(r"[0-9a-f]{40}")


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_json(value):
    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_json_exclusive(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="ascii") as handle:
        handle.write(canonical_json(value) + "\n")


def load_config():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    require(config["version"] == VERSION, "pilot version changed")
    require(config["cell"] == {
        "code_id": "m08_c06", "disorder_index": 0,
        "disorder_source": "attempt022", "p": 0.04,
    }, "pilot cell changed")
    require(config["initialization"] == {
        "families": ["P", "U", "M0", "S0"],
        "trajectories_per_family": 4,
    }, "pilot initialization panel changed")
    require(config["clocks"] == {
        "burn": 256, "measurement": 1024,
    }, "pilot clocks changed")
    require(config["workers"] == 4, "pilot worker count changed")
    require(config["implementation"]["method_id"] == HYBRID_GIBBS_METHOD_ID
            and config["implementation"]["engine"] == HYBRID_GIBBS_ENGINE,
            "pilot implementation identity changed")
    for path_field, sha_field in (
        ("hybrid_module", "hybrid_module_sha256"),
        ("row_module", "row_module_sha256"),
        ("column_module", "column_module_sha256"),
        ("runner", "runner_sha256"),
    ):
        path = EXP102_ROOT / config["implementation"][path_field]
        require(sha256_file(path) == config["implementation"][sha_field],
                f"pilot source binding changed: {path_field}")
    require(
        SHA_RE.fullmatch(config["predecessor_control_content_sha256"]) is not None,
        "pilot predecessor control SHA changed",
    )
    return config, sha256_file(CONFIG_PATH)


def load_context(config, config_sha):
    workflow = import_module(CONTROL_WORKFLOW)
    predecessor_config, predecessor_sha = workflow._load_config()
    context = workflow._load_control(
        workflow.SOURCE_CONTROL_DIR, predecessor_config, predecessor_sha,
    )
    require(
        context["metadata"]["control_content_sha256"]
        == config["predecessor_control_content_sha256"],
        "pilot predecessor control changed",
    )
    require(
        sha256_file(context["control_path"])
        == config["predecessor_control_file_sha256"],
        "pilot predecessor control file changed",
    )
    context["pilot_config"] = config
    context["pilot_config_sha"] = config_sha
    return context


def build_tasks(config, config_sha, control_sha, source_commit):
    tasks = []
    namespace = config["seed_namespace"]
    for family in config["initialization"]["families"]:
        for index in range(config["initialization"]["trajectories_per_family"]):
            prefix = (
                source_commit, config_sha, control_sha, family, index,
            )
            task = {
                "burn_seed": derive_seed(namespace, *prefix, "burn"),
                "family": family,
                "index": index,
                "initialization_seed": derive_seed(
                    namespace, *prefix, "initialization",
                ),
                "measurement_seed": derive_seed(
                    namespace, *prefix, "measurement",
                ),
                "observation_seed": derive_seed(
                    namespace, *prefix, "observation",
                ),
            }
            task["task_fingerprint"] = sha256_json(task)
            tasks.append(task)
    return tasks


def initial_state(context, task):
    family = task["family"]
    if family == "P":
        return context["fixed_states"][0].copy()
    if family == "M0":
        return context["fixed_states"][1].copy()
    if family == "S0":
        return context["fixed_states"][3].copy()
    require(family == "U", "unknown pilot family")
    return uniform_hard_coset_state(
        context["model"], context["syndrome"], task["initialization_seed"],
    )


def run_task(context, task, source_commit, mass, column_cache, row_plan):
    config = context["pilot_config"]
    sampler = HybridGibbsConfig(
        p=config["cell"]["p"],
        burn_clocks=config["clocks"]["burn"],
        measurement_clocks=config["clocks"]["measurement"],
    )
    initial = initial_state(context, task)
    sampling_started = time.perf_counter()
    raw = run_hybrid_gibbs_trajectory(
        context["model"], context["frame"], context["H"],
        context["syndrome"], sampler, initial, task["burn_seed"],
        task["measurement_seed"], task["observation_seed"], mass=mass,
        column_cache=column_cache,
        column_workspace=build_full_column_direct_block_workspace(column_cache),
        row_plan=row_plan,
    )
    sampling_seconds = time.perf_counter() - sampling_started
    replay_started = time.perf_counter()
    replay_ok = replay_hybrid_gibbs_trajectory(
        context["model"], context["frame"], context["H"],
        context["syndrome"], sampler, initial, task["burn_seed"],
        task["measurement_seed"], task["observation_seed"], raw, mass=mass,
        column_cache=column_cache,
        column_workspace=build_full_column_direct_block_workspace(column_cache),
        row_plan=row_plan,
    )
    replay_seconds = time.perf_counter() - replay_started
    require(replay_ok is True, "pilot runner replay failed")
    raw.update({
        "config_sha256": np.array(context["pilot_config_sha"]),
        "control_content_sha256": np.array(
            context["metadata"]["control_content_sha256"]
        ),
        "raw_version": np.array(RAW_VERSION),
        "replay_ok": np.array(True),
        "replay_seconds": np.array(replay_seconds, dtype=np.float64),
        "sampling_seconds": np.array(sampling_seconds, dtype=np.float64),
        "source_commit": np.array(source_commit),
        "task_json": np.array(canonical_json(task)),
        "version": np.array(HYBRID_GIBBS_VERSION),
    })
    return raw


def save_raw_exclusive(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(handle, **raw)


def b_bits(b_columns, rows):
    values = np.asarray(b_columns, dtype=np.uint32)
    return np.asarray([
        [
            [(int(state[column]) >> row) & 1 for column in range(rows)]
            for row in range(rows)
        ]
        for state in values
    ], dtype=np.float64)


def load_raw(path, receipt, context, task, source_commit, log_mass):
    require(sha256_file(path) == receipt["raw_sha256"], "pilot raw SHA changed")
    with np.load(path, allow_pickle=False) as archive:
        raw = {name: archive[name].copy() for name in archive.files}
    require(str(raw["source_commit"].item()) == source_commit
            and str(raw["config_sha256"].item()) == context["pilot_config_sha"]
            and str(raw["control_content_sha256"].item())
            == context["metadata"]["control_content_sha256"]
            and str(raw["raw_version"].item()) == RAW_VERSION
            and str(raw["version"].item()) == HYBRID_GIBBS_VERSION
            and str(raw["task_json"].item()) == canonical_json(task),
            "pilot raw identity changed")
    require(bool(raw["replay_ok"].item()), "pilot replay flag is false")
    states = np.unpackbits(
        raw["measurement__states_packed"], axis=1,
        count=context["model"].num_qubits, bitorder="little",
    ).astype(np.uint8)
    residuals = (
        context["model"].H_check.astype(np.int64)
        @ states.T.astype(np.int64) % 2
    ).T.astype(np.uint8)
    require(np.array_equal(
        residuals,
        np.repeat(context["syndrome"][None, :], states.shape[0], axis=0),
    ), "pilot raw state left hard coset")
    require(np.all(np.isfinite(raw["measurement__b_likelihood"])),
            "pilot likelihood is non-finite")
    expected_burn_likelihood = float(
        log_mass[raw["burn__final_a_syndromes"]].sum()
    )
    return raw, expected_burn_likelihood


def family_summary(records, rows, columns):
    first_weights = []
    last_weights = []
    first_likelihood = []
    last_likelihood = []
    last_bits = []
    trajectory_rows = []
    for task, raw, burn_likelihood in records:
        blocks = raw["measurement__blocks"]
        first = blocks < 4
        last = blocks >= 4
        first_weights.append(raw["measurement__b_weights"][first] / (rows * rows))
        last_weights.append(raw["measurement__b_weights"][last] / (rows * rows))
        first_likelihood.append(raw["measurement__b_likelihood"][first] / columns)
        last_likelihood.append(raw["measurement__b_likelihood"][last] / columns)
        last_bits.append(b_bits(raw["measurement__b_columns"][last], rows))
        burn_b_weight = sum(
            int(value).bit_count() for value in raw["burn__final_b_columns"]
        ) / (rows * rows)
        counters = raw["measurement__counters"]
        trajectory_rows.append({
            "burn_b_likelihood_per_factor": burn_likelihood / columns,
            "burn_b_weight_normalized": burn_b_weight,
            "column_changes": int(counters[2]),
            "index": int(task["index"]),
            "replay_inclusive_seconds": float(
                raw["sampling_seconds"].item() + raw["replay_seconds"].item()
            ),
            "row_changes": int(counters[5]),
        })
    first_weights = np.concatenate(first_weights)
    last_weights = np.concatenate(last_weights)
    first_likelihood = np.concatenate(first_likelihood)
    last_likelihood = np.concatenate(last_likelihood)
    last_bits = np.concatenate(last_bits, axis=0)
    return {
        "bit_means": last_bits.mean(axis=0),
        "first_b_likelihood_per_factor": float(first_likelihood.mean()),
        "first_b_weight_normalized": float(first_weights.mean()),
        "last_b_likelihood_per_factor": float(last_likelihood.mean()),
        "last_b_weight_normalized": float(last_weights.mean()),
        "trajectories": trajectory_rows,
    }


def analyze(run_root, context, tasks, source_commit, mass, receipts):
    rows, columns = context["H"].shape
    log_mass = np.log(mass)
    by_family = {family: [] for family in context["pilot_config"]["initialization"]["families"]}
    receipt_by_task = {row["task_fingerprint"]: row for row in receipts}
    for task in tasks:
        receipt = receipt_by_task[task["task_fingerprint"]]
        raw, burn_likelihood = load_raw(
            run_root / receipt["raw_relpath"], receipt, context, task,
            source_commit, log_mass,
        )
        by_family[task["family"]].append((task, raw, burn_likelihood))
    summaries = {
        family: family_summary(records, rows, columns)
        for family, records in by_family.items()
    }
    gates_config = context["pilot_config"]["gates"]
    comparisons = []
    comparison_pass = True
    names = list(summaries)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            left_summary = summaries[left]
            right_summary = summaries[right]
            weight_delta = abs(
                left_summary["last_b_weight_normalized"]
                - right_summary["last_b_weight_normalized"]
            )
            likelihood_delta = abs(
                left_summary["last_b_likelihood_per_factor"]
                - right_summary["last_b_likelihood_per_factor"]
            )
            bit_msd = float(np.mean(
                (left_summary["bit_means"] - right_summary["bit_means"]) ** 2
            ))
            valid = bool(
                weight_delta <= gates_config["max_b_weight_delta"]
                and likelihood_delta <= gates_config["max_b_likelihood_delta"]
                and bit_msd <= gates_config["max_b_bit_mean_squared_delta"]
            )
            comparison_pass &= valid
            comparisons.append({
                "b_bit_mean_squared_delta": bit_msd,
                "b_likelihood_delta": likelihood_delta,
                "b_weight_delta": weight_delta,
                "left": left,
                "right": right,
                "valid": valid,
            })
    drift = {}
    drift_pass = True
    for family, summary in summaries.items():
        weight_delta = abs(
            summary["first_b_weight_normalized"]
            - summary["last_b_weight_normalized"]
        )
        likelihood_delta = abs(
            summary["first_b_likelihood_per_factor"]
            - summary["last_b_likelihood_per_factor"]
        )
        valid = bool(
            weight_delta <= gates_config["max_b_weight_delta"]
            and likelihood_delta <= gates_config["max_b_likelihood_delta"]
        )
        drift_pass &= valid
        drift[family] = {
            "b_likelihood_delta": likelihood_delta,
            "b_weight_delta": weight_delta,
            "valid": valid,
        }
    u_collapse_count = sum(
        row["burn_b_weight_normalized"] <= gates_config["max_u_burn_b_weight"]
        and row["burn_b_likelihood_per_factor"]
        >= gates_config["min_u_burn_b_likelihood"]
        for row in summaries["U"]["trajectories"]
    )
    low_energy_motion = all(
        row["column_changes"] + row["row_changes"] > 0
        for family in ("P", "M0", "S0")
        for row in summaries[family]["trajectories"]
    )
    replay_inclusive_max = max(
        row["replay_inclusive_seconds"]
        for summary in summaries.values()
        for row in summary["trajectories"]
    )
    gates = {
        "between_family_b_distribution": comparison_pass,
        "low_energy_real_motion": low_energy_motion,
        "replay_and_identity": True,
        "runtime": replay_inclusive_max <= gates_config["max_replay_inclusive_seconds"],
        "u_burn_collapse": u_collapse_count >= gates_config["min_u_collapse_count"],
        "within_family_drift": drift_pass,
    }
    gates["all"] = all(gates.values())
    serializable_summaries = {
        family: {
            key: value for key, value in summary.items() if key != "bit_means"
        }
        for family, summary in summaries.items()
    }
    core = {
        "authority": context["pilot_config"]["authority"],
        "comparisons": comparisons,
        "config_sha256": context["pilot_config_sha"],
        "control_content_sha256": context["metadata"]["control_content_sha256"],
        "drift": drift,
        "family_summaries": serializable_summaries,
        "gates": gates,
        "raw_count": len(receipts),
        "raw_set_sha256": sha256_json([
            [row["task_fingerprint"], row["raw_sha256"]]
            for row in sorted(receipts, key=lambda value: value["task_fingerprint"])
        ]),
        "replay_inclusive_max_seconds": replay_inclusive_max,
        "report_version": REPORT_VERSION,
        "source_commit": source_commit,
        "status": (
            "LOCAL_HYBRID_B_NECESSARY_GATES_PASS"
            if gates["all"] else "LOCAL_HYBRID_B_NECESSARY_GATES_FAIL"
        ),
        "u_collapse_count": int(u_collapse_count),
    }
    core["report_sha256"] = sha256_json(core)
    write_json_exclusive(run_root / "pilot_report.json", core)
    return core


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--run-root", type=Path, default=ROOT / "local_run_v0")
    args = parser.parse_args()
    require(COMMIT_RE.fullmatch(args.source_commit) is not None,
            "source commit must be a full SHA")
    config, config_sha = load_config()
    context = load_context(config, config_sha)
    control_sha = context["metadata"]["control_content_sha256"]
    tasks = build_tasks(config, config_sha, control_sha, args.source_commit)
    manifest_core = {
        "config_sha256": config_sha,
        "control_content_sha256": control_sha,
        "schedule_version": SCHEDULE_VERSION,
        "source_commit": args.source_commit,
        "tasks": tasks,
    }
    manifest = {**manifest_core, "schedule_sha256": sha256_json(manifest_core)}
    write_json_exclusive(args.run_root / "schedule.json", manifest)
    mass = np.ascontiguousarray(
        build_classical_coset_mass(context["H"], config["cell"]["p"], engine="numba"),
        dtype=np.float64,
    )
    column_cache = build_full_column_direct_block_cache(
        context["H"].shape[0], config["cell"]["p"], mass,
    )
    row_plan = build_full_row_elimination_plan(context["H"])
    receipts = []
    with ThreadPoolExecutor(max_workers=config["workers"]) as executor:
        futures = {
            executor.submit(
                run_task, context, task, args.source_commit, mass,
                column_cache, row_plan,
            ): task
            for task in tasks
        }
        for future in as_completed(futures):
            task = futures[future]
            raw = future.result()
            relpath = Path("raw") / (
                f"{task['family']}_{task['index']:02d}_{task['task_fingerprint'][:12]}.npz"
            )
            path = args.run_root / relpath
            save_raw_exclusive(path, raw)
            receipts.append({
                "raw_relpath": str(relpath),
                "raw_sha256": sha256_file(path),
                "task_fingerprint": task["task_fingerprint"],
            })
            print(
                f"completed {len(receipts)}/{len(tasks)} "
                f"{task['family']}{task['index']}", flush=True,
            )
    receipts.sort(key=lambda value: value["task_fingerprint"])
    raw_manifest_core = {
        "receipts": receipts,
        "schedule_sha256": manifest["schedule_sha256"],
    }
    raw_manifest = {
        **raw_manifest_core,
        "raw_manifest_sha256": sha256_json(raw_manifest_core),
    }
    write_json_exclusive(args.run_root / "raw_manifest.json", raw_manifest)
    report = analyze(
        args.run_root, context, tasks, args.source_commit, mass, receipts,
    )
    print(json.dumps(report, sort_keys=True, indent=2))
    if not report["gates"]["all"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
