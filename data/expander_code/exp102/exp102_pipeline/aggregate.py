import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np

from .config import load_config
from .io import atomic_json, atomic_npz, sha256_file, sha256_json
from .registry import load_registry


def _sem(values):
    return float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else np.nan


def _paired_z(differences):
    differences = np.asarray(differences, dtype=float)
    sem = _sem(differences)
    mean = float(np.mean(differences))
    if sem == 0:
        return 0.0 if mean == 0 else np.copysign(np.inf, mean)
    return mean / sem


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
        for disorder in range(config["num_disorders"]):
            path = raw_dir / code["code_id"] / f"d{disorder:03d}.npz"
            if not path.exists():
                continue
            with np.load(path, allow_pickle=False) as data:
                expected = {"registry_sha256": registry["registry_sha256"], "config_sha256": config["config_sha256"],
                            "frozen_config_sha256": frozen_hash, "code_id": code["code_id"], "disorder_index": disorder}
                for field, value in expected.items():
                    if str(data[field].item()) != str(value):
                        raise ValueError(f"raw fingerprint mismatch in {path}: {field}")
                scalar_expected = {
                    "engine": "numba", "source_commit": frozen["source_commit"],
                    "section_fingerprint": code["section_fingerprint"],
                    "logical_frame_fingerprint": code["logical_frame_fingerprint"],
                }
                for field, value in scalar_expected.items():
                    if field not in data or str(data[field].item()) != str(value):
                        raise ValueError(f"raw production identity mismatch in {path}: {field}")
                required_shapes = {"p_values": (7,), "qtop": (7,), "collision_mass": (7,),
                                   "planted_hit": (7,), "valid": (7,), "labels": (7, 4, frozen["by_m"][str(code["m"])]["measurement_rounds"])}
                for field, expected_shape in required_shapes.items():
                    if field not in data or data[field].shape != expected_shape:
                        raise ValueError(f"raw shape mismatch in {path}: {field}")
                if data["labels"].dtype != np.uint64 or data["valid"].dtype != np.bool_:
                    raise ValueError(f"raw dtype mismatch in {path}")
                if not np.array_equal(data["p_values"], p_values):
                    raise ValueError(f"raw p grid mismatch in {path}")
                present[code_index] += 1
                raw_qtop[code_index, :, disorder] = data["qtop"]
                raw_collision[code_index, :, disorder] = data["collision_mass"]
                raw_planted[code_index, :, disorder] = data["planted_hit"]
                valid_count[code_index] += data["valid"].astype(np.int16)
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
    manifest = {"aggregation_version": "exp102.aggregation.v1", "result_file": result_path.name,
                "result_sha256": sha256_file(result_path), "registry_sha256": registry["registry_sha256"],
                "config_sha256": config["config_sha256"], "frozen_config_sha256": frozen_hash,
                "engine": "numba", "source_commit": frozen["source_commit"],
                "planned_tasks": 6144, "present_tasks": int(np.min(present, axis=1).sum()),
                "all_points_reportable": bool(np.all(m_status == "REPORTABLE")),
                "main_errorbar_definition": "std(code_means,ddof=1)/sqrt(8)"}
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
