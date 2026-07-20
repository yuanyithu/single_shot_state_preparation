import argparse
import csv
from importlib import import_module
import json
import shutil
from pathlib import Path

import numpy as np

from .config import load_config
from .io import atomic_json, atomic_npz, sha256_file, sha256_json, verify_source_identity
from .registry import load_frozen_code, load_registry
from .pilot import recompute_frozen
from .tasks import task_records
from .worker import build_model, validate_production_raw


build_production_manifest = import_module(
    "data.expander_code.exp102.validation.002_numba_smoke_20260719.build_production_deployment"
).build_manifest


def _sem(values):
    return float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else np.nan


def _paired_z(differences):
    differences = np.asarray(differences, dtype=float)
    sem = _sem(differences)
    mean = float(np.mean(differences))
    if sem == 0:
        return 0.0 if mean == 0 else np.copysign(np.inf, mean)
    return mean / sem


def _verify_production_raw_manifests(raw_dir, registry, config, frozen):
    raw_dir = Path(raw_dir).resolve()
    manifest_dir = raw_dir / "_manifests"
    manifests = sorted(manifest_dir.glob("*.json")) if manifest_dir.is_dir() else []
    expected_nodes = {"nd-1", "nd-2", "nd-3"}
    if {path.stem for path in manifests} != expected_nodes:
        raise ValueError("production raw manifests must cover nd-1, nd-2, and nd-3")
    frozen_hash = sha256_json(frozen)
    listed = {}
    manifest_hashes = {}
    for path in manifests:
        manifest_hashes[path.stem] = sha256_file(path)
        manifest = json.loads(path.read_text(encoding="ascii"))
        if set(manifest) != {
                "raw_manifest_version", "node", "registry_sha256", "config_sha256",
                "frozen_config_sha256", "source_commit", "files"}:
            raise ValueError(f"production raw manifest schema mismatch: {path}")
        expected = {
            "raw_manifest_version": "exp102.production.raw.v1",
            "node": path.stem,
            "registry_sha256": registry["registry_sha256"],
            "config_sha256": config["config_sha256"],
            "frozen_config_sha256": frozen_hash,
            "source_commit": frozen["source_commit"],
        }
        if any(manifest.get(field) != value for field, value in expected.items()):
            raise ValueError(f"production raw manifest identity mismatch: {path}")
        files = manifest["files"]
        if not isinstance(files, list) or not files:
            raise ValueError(f"production raw manifest has no files: {path}")
        for item in files:
            if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
                raise ValueError(f"production raw manifest file entry is invalid: {path}")
            raw_path = (raw_dir / item["path"]).resolve()
            if raw_dir not in raw_path.parents or raw_path.suffix != ".npz" or not raw_path.is_file():
                raise ValueError(f"production raw manifest path is invalid: {item['path']}")
            if raw_path in listed:
                raise ValueError(f"production raw file is listed more than once: {raw_path}")
            if sha256_file(raw_path) != item["sha256"]:
                raise ValueError(f"production stage raw hash mismatch: {raw_path}")
            listed[raw_path] = item["sha256"]
    actual = {path.resolve() for path in raw_dir.rglob("*.npz")}
    if actual != set(listed) or len(listed) != 6144:
        raise ValueError("production raw files differ from the three completed stage manifests")
    return manifest_hashes


def _verify_production_control(raw_dir, registry_path, config_path, frozen_path,
                               registry, config, frozen):
    raw_dir = Path(raw_dir).resolve()
    run_root = raw_dir.parent.parent
    if raw_dir != run_root / "raw" / "production":
        raise ValueError("production raw directory must be RUN_ROOT/raw/production")
    frozen_path = Path(frozen_path).resolve()
    if frozen_path != run_root / "frozen.json":
        raise ValueError("production freezer must be RUN_ROOT/frozen.json")
    report_path = run_root / "pilot_report.json"
    task_plan_path = run_root / "task_plan.json"
    deployment_path = run_root / "production_deployment.json"
    for path in (report_path, task_plan_path, deployment_path):
        if not path.is_file():
            raise ValueError(f"production control file is missing: {path.name}")
    if frozen != recompute_frozen(report_path, registry_path, config_path):
        raise ValueError("production freezer differs from pilot raw/report recomputation")
    task_plan = json.loads(task_plan_path.read_text(encoding="ascii"))
    if (task_plan.get("status") != "PRODUCTION" or task_plan.get("num_tasks") != 6144
            or task_plan.get("registry_sha256") != registry["registry_sha256"]
            or task_plan.get("config_sha256") != config["config_sha256"]
            or task_plan.get("tasks") != task_records(registry, config, frozen)):
        raise ValueError("production task plan identity or coverage mismatch")
    deployment = json.loads(deployment_path.read_text(encoding="ascii"))
    expected_deployment = build_production_manifest(
        registry_path, config_path, frozen_path, report_path,
    )
    if deployment != expected_deployment:
        raise ValueError("production deployment differs from held-out recomputation")
    status_paths = {
        node: run_root / "status" / f"production_{node}.json"
        for node in ("nd-1", "nd-2", "nd-3")
    }
    if any(not path.is_file() for path in status_paths.values()):
        raise ValueError("three-node production status coverage is incomplete")
    file_hashes = {
        "task_plan_sha256": sha256_file(task_plan_path),
        "deployment_manifest_sha256": sha256_file(deployment_path),
        "pilot_report_sha256_file": sha256_file(report_path),
    }
    statuses = {}
    for node, path in status_paths.items():
        status = json.loads(path.read_text(encoding="ascii"))
        statuses[node] = status
        if (status.get("status") != "SUCCESS" or status.get("node") != node
                or status.get("source_commit") != frozen["source_commit"]
                or status.get("registry_sha256") != registry["registry_sha256"]
                or status.get("config_sha256") != config["config_sha256"]
                or status.get("frozen_config_sha256") != sha256_json(frozen)
                or any(status.get(field) != value for field, value in file_hashes.items())
                or status.get("computed", 0) + status.get("reused", 0) != status.get("expected")):
            raise ValueError(f"production status identity/count mismatch: {node}")
    if sum(status["expected"] for status in statuses.values()) != 6144:
        raise ValueError("production statuses do not cover exactly 6144 tasks")
    return statuses


def aggregate(raw_dir, registry_path, config_path, frozen_path, output_dir):
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    registry = load_registry(registry_path)
    config = load_config(config_path)
    frozen = json.loads(Path(frozen_path).read_text(encoding="ascii"))
    if (frozen.get("status") != "FROZEN_HELD_OUT_PASS" or frozen.get("engine") != "numba"
            or frozen.get("registry_sha256") != registry["registry_sha256"]
            or frozen.get("config_sha256") != config["config_sha256"]):
        raise ValueError("invalid frozen production identity")
    source_identity = verify_source_identity(Path.cwd(), frozen.get("source_commit", ""))
    statuses = _verify_production_control(
        raw_dir, registry_path, config_path, frozen_path, registry, config, frozen,
    )
    raw_manifest_hashes = _verify_production_raw_manifests(
        raw_dir, registry, config, frozen,
    )
    if any(statuses[node].get("raw_manifest_sha256") != raw_manifest_hashes[node]
           for node in raw_manifest_hashes):
        raise ValueError("production status/raw manifest hash mismatch")
    frozen_hash = sha256_json(frozen)
    p_values = np.asarray(config["p_values"])
    codes = registry["codes"]
    shape = (len(codes), len(p_values))
    mu_code = np.full(shape, np.nan); sem_code = np.full(shape, np.nan)
    code_status = np.full(shape, "INCOMPLETE", dtype="U24")
    present = np.zeros(shape, dtype=np.int16); valid_count = np.zeros(shape, dtype=np.int16)
    paired_z = np.full(shape, np.nan)
    raw_qtop = np.full((len(codes), len(p_values), config["num_disorders"]), np.nan)
    raw_collision = raw_qtop.copy(); raw_planted = raw_qtop.copy()
    for code_index, code in enumerate(codes):
        _, loaded_code, H = load_frozen_code(registry_path, code["code_id"])
        if loaded_code != code:
            raise ValueError(f"registry changed while loading {code['code_id']}")
        model, frame = build_model(H)
        for disorder in range(config["num_disorders"]):
            path = raw_dir / code["code_id"] / f"d{disorder:03d}.npz"
            if not path.exists():
                continue
            record = validate_production_raw(path, registry, code, config, frozen, model, frame)
            if record["disorder_index"] != disorder:
                raise ValueError(f"raw disorder identity mismatch in {path}")
            present[code_index] += 1
            raw_qtop[code_index, :, disorder] = record["qtop"]
            raw_collision[code_index, :, disorder] = record["collision_mass"]
            raw_planted[code_index, :, disorder] = record["planted_hit"]
            valid_count[code_index] += record["valid"].astype(np.int16)
        for p_index in range(len(p_values)):
            if present[code_index, p_index] < config["num_disorders"]:
                code_status[code_index, p_index] = "INCOMPLETE"
            elif valid_count[code_index, p_index] < config["num_disorders"]:
                code_status[code_index, p_index] = "SAMPLING_INSUFFICIENT"
            else:
                z = _paired_z(raw_collision[code_index, p_index] - raw_planted[code_index, p_index])
                paired_z[code_index, p_index] = z
                if abs(z) > config["production_gate"]["paired_audit_max_abs_z"]:
                    code_status[code_index, p_index] = "PLANTED_AUDIT_FAILED"
                else:
                    code_status[code_index, p_index] = "REPORTABLE"
                    values = raw_qtop[code_index, p_index]
                    mu_code[code_index, p_index] = np.mean(values)
                    sem_code[code_index, p_index] = _sem(values)
    m_values = np.arange(3, 9, dtype=np.int8)
    m_shape = (len(m_values), len(p_values))
    mu_m = np.full(m_shape, np.nan); sem_between = np.full(m_shape, np.nan)
    mean_within_sem = np.full(m_shape, np.nan); propagated_within_sem = np.full(m_shape, np.nan)
    m_status = np.full(m_shape, "INCOMPLETE", dtype="U24")
    covariance = np.full((len(m_values), len(p_values), len(p_values)), np.nan)
    for m_index, m in enumerate(m_values):
        indices = [i for i, code in enumerate(codes) if code["m"] == m]
        for p_index in range(len(p_values)):
            statuses = code_status[indices, p_index]
            if np.all(statuses == "REPORTABLE"):
                values = mu_code[indices, p_index]
                mu_m[m_index, p_index] = np.mean(values)
                sem_between[m_index, p_index] = _sem(values)
                mean_within_sem[m_index, p_index] = np.mean(sem_code[indices, p_index])
                propagated_within_sem[m_index, p_index] = np.sqrt(np.sum(sem_code[indices, p_index] ** 2)) / 8
                m_status[m_index, p_index] = "REPORTABLE"
            elif np.any(statuses == "INCOMPLETE"):
                m_status[m_index, p_index] = "INCOMPLETE"
            else:
                m_status[m_index, p_index] = "SAMPLING_INSUFFICIENT"
        if np.all(code_status[indices] == "REPORTABLE"):
            covariance[m_index] = np.cov(mu_code[indices], rowvar=False, ddof=1)
    result_path = output_dir / "exp102_results.npz"
    atomic_npz(result_path, p_values=p_values, m_values=m_values,
        code_ids=np.asarray([code["code_id"] for code in codes], dtype="U8"),
        code_m=np.asarray([code["m"] for code in codes], dtype=np.int8),
        mu_code=mu_code, sem_within_code=sem_code, code_status=code_status,
        present_disorders=present, valid_disorders=valid_count, planted_paired_z=paired_z,
        mu_m=mu_m, errorbar_between_code_sem=sem_between, m_status=m_status,
        average_within_code_sem=mean_within_sem, propagated_within_code_sem=propagated_within_sem,
        covariance_across_p=covariance, registry_sha256=np.array(registry["registry_sha256"]),
        config_sha256=np.array(config["config_sha256"]), frozen_config_sha256=np.array(frozen_hash),
        engine=np.array("numba"), source_commit=np.array(frozen["source_commit"]),
        physics_contract_version=np.array(config["physics_contract_version"]),
        pt_contract_version=np.array(config["pt_contract_version"]), scan_contract_version=np.array(config["scan_contract_version"]),
        main_errorbar_definition=np.array("std(code_means,ddof=1)/sqrt(8)"))
    _write_tables(output_dir, codes, p_values, mu_code, sem_code, code_status, mu_m, sem_between, m_status)
    shutil.copy2(registry_path, output_dir / "registry.json")
    shutil.copy2(Path(registry_path).parent / "code_registry.csv", output_dir / "code_registry.csv")
    shutil.copy2(config_path, output_dir / "production.v1.json")
    shutil.copy2(frozen_path, output_dir / "frozen.json")
    manifest = {"aggregation_version": "exp102.aggregation.v1", "result_file": result_path.name,
                "result_sha256": sha256_file(result_path), "registry_sha256": registry["registry_sha256"],
                "config_sha256": config["config_sha256"], "frozen_config_sha256": frozen_hash,
                "engine": "numba", "source_commit": frozen["source_commit"],
                "planned_tasks": 6144, "present_tasks": int(np.min(present, axis=1).sum()),
                "all_points_reportable": bool(np.all(m_status == "REPORTABLE")),
                "main_errorbar_definition": "std(code_means,ddof=1)/sqrt(8)",
                "registry_file_sha256": sha256_file(output_dir / "registry.json"),
                "config_file_sha256": sha256_file(output_dir / "production.v1.json"),
                "frozen_file_sha256": sha256_file(output_dir / "frozen.json"),
                "pilot_report_sha256": frozen["pilot_report_sha256"],
                "pilot_raw_evidence_sha256": frozen["raw_evidence_sha256"],
                "held_out_attempt_by_m": frozen["held_out_attempt_by_m"],
                "production_raw_manifest_sha256": raw_manifest_hashes,
                "source_identity": source_identity}
    atomic_json(output_dir / "aggregation_manifest.json", manifest)
    return manifest


def _write_tables(output_dir, codes, p_values, mu_code, sem_code, code_status, mu_m, sem_between, m_status):
    with open(output_dir / "qtop_per_code.csv", "w", newline="", encoding="ascii") as handle:
        writer = csv.writer(handle); writer.writerow(["code_id", "m", "p", "qtop", "within_code_sem", "status"])
        for i, code in enumerate(codes):
            for j, p in enumerate(p_values): writer.writerow([code["code_id"], code["m"], p, mu_code[i,j], sem_code[i,j], code_status[i,j]])
    with open(output_dir / "qtop_by_m.csv", "w", newline="", encoding="ascii") as handle:
        writer = csv.writer(handle); writer.writerow(["m", "p", "qtop", "between_code_sem", "status"])
        for i, m in enumerate(range(3, 9)):
            for j, p in enumerate(p_values): writer.writerow([m, p, mu_m[i,j], sem_between[i,j], m_status[i,j]])


def main(argv=None):
    parser = argparse.ArgumentParser()
    for name in ("raw_dir", "registry", "config", "frozen", "output_dir"): parser.add_argument(name)
    args = parser.parse_args(argv)
    print(json.dumps(aggregate(args.raw_dir, args.registry, args.config, args.frozen, args.output_dir), indent=2))


if __name__ == "__main__": main()
