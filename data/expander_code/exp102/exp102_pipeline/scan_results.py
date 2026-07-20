import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np

from . import PHYSICS_VERSION, PT_VERSION, SCAN_VERSION
from .config import load_config, validate_pilot_candidate
from .io import sha256_file, sha256_json
from .registry import load_registry


REGISTRY_SHA256 = "883730e0ba548f6b358187d8f123fdd4d8aeb116f4bacda363c35c16d01ae40b"
CONFIG_SHA256 = "96b5957fb3f1f0fb520b5f635eb3424f2aa93c90c471e5f1d20013f1b76a7330"


def _verify_recorded_git_support(source_commit, support_files):
    """Bind copied support files to the recorded commit without requiring HEAD."""
    repository_root = Path(__file__).resolve().parents[4]
    subprocess.run(
        ("git", "-C", str(repository_root), "cat-file", "-e", f"{source_commit}^{{commit}}"),
        check=True, capture_output=True,
    )
    for repository_path, local_path in support_files:
        content = subprocess.run(
            ("git", "-C", str(repository_root), "show", f"{source_commit}:{repository_path}"),
            check=True, capture_output=True,
        ).stdout
        if hashlib.sha256(content).hexdigest() != sha256_file(local_path):
            raise ValueError(f"support file does not match recorded source commit: {repository_path}")


def load_exp102_publication_q_top(results_path, point_mask=None):
    results_path = Path(results_path)
    manifest_path = results_path.parent / "aggregation_manifest.json"
    if not manifest_path.exists():
        raise ValueError("missing exp102 aggregation manifest")
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    if (manifest.get("aggregation_version") != "exp102.aggregation.v1"
            or manifest.get("result_file") != results_path.name
            or manifest.get("main_errorbar_definition") != "std(code_means,ddof=1)/sqrt(8)"):
        raise ValueError("wrong exp102 aggregation contract")
    if manifest.get("result_sha256") != sha256_file(results_path):
        raise ValueError("exp102 result file SHA256 does not match manifest")
    if manifest.get("planned_tasks") != 6144 or manifest.get("present_tasks") != 6144:
        raise ValueError("exp102 manifest does not certify 6144 present production tasks")
    source_commit = str(manifest.get("source_commit", ""))
    if manifest.get("engine") != "numba" or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("exp102 manifest lacks production engine/source identity")
    if (manifest.get("registry_sha256") != REGISTRY_SHA256
            or manifest.get("config_sha256") != CONFIG_SHA256):
        raise ValueError("exp102 manifest registry/config identity mismatch")
    registry_path = results_path.parent / "registry.json"
    config_path = results_path.parent / "production.v1.json"
    frozen_path = results_path.parent / "frozen.json"
    support = (
        (registry_path, "registry_file_sha256"),
        (config_path, "config_file_sha256"),
        (frozen_path, "frozen_file_sha256"),
    )
    for path, field in support:
        if not path.is_file() or manifest.get(field) != sha256_file(path):
            raise ValueError(f"exp102 support file identity mismatch: {path.name}")
    _verify_recorded_git_support(source_commit, (
        ("data/expander_code/exp102/registry/registry.json", registry_path),
        ("data/expander_code/exp102/config/production.v1.json", config_path),
    ))
    registry = load_registry(registry_path, verify_files=False)
    config = load_config(config_path)
    frozen = json.loads(frozen_path.read_text(encoding="ascii"))
    if registry["registry_sha256"] != REGISTRY_SHA256 or config["config_sha256"] != CONFIG_SHA256:
        raise ValueError("exp102 support registry/config is not frozen production data")
    frozen_hash = sha256_json(frozen)
    if (frozen_hash != manifest.get("frozen_config_sha256")
            or frozen.get("status") != "FROZEN_HELD_OUT_PASS"
            or frozen.get("engine") != "numba"
            or frozen.get("source_commit") != source_commit
            or frozen.get("registry_sha256") != REGISTRY_SHA256
            or frozen.get("config_sha256") != CONFIG_SHA256):
        raise ValueError("exp102 freezer identity mismatch")
    for field in ("pilot_report_sha256", "raw_evidence_sha256"):
        if re.fullmatch(r"[0-9a-f]{64}", str(frozen.get(field, ""))) is None:
            raise ValueError(f"exp102 freezer lacks {field}")
    if (manifest.get("pilot_report_sha256") != frozen["pilot_report_sha256"]
            or manifest.get("pilot_raw_evidence_sha256") != frozen["raw_evidence_sha256"]
            or manifest.get("held_out_attempt_by_m") != frozen.get("held_out_attempt_by_m")):
        raise ValueError("exp102 pilot/freezer provenance mismatch")
    if set(frozen.get("by_m", {})) != {str(m) for m in range(3, 9)}:
        raise ValueError("exp102 freezer does not cover m=3..8")
    for candidate in frozen["by_m"].values():
        validate_pilot_candidate(candidate, config)
    raw_manifests = manifest.get("production_raw_manifest_sha256")
    if (not isinstance(raw_manifests, dict)
            or set(raw_manifests) != {"nd-1", "nd-2", "nd-3"}
            or any(re.fullmatch(r"[0-9a-f]{64}", str(value)) is None
                   for value in raw_manifests.values())):
        raise ValueError("exp102 production raw manifest provenance is incomplete")
    with np.load(results_path, allow_pickle=False) as data:
        expected = {"physics_contract_version": PHYSICS_VERSION, "pt_contract_version": PT_VERSION,
                    "scan_contract_version": SCAN_VERSION}
        for field, value in expected.items():
            if str(data[field].item()) != value:
                raise ValueError(f"publication loader rejects {field}={data[field].item()!r}")
        for field in ("registry_sha256", "config_sha256", "frozen_config_sha256"):
            if str(data[field].item()) != manifest[field]:
                raise ValueError(f"manifest/result mismatch: {field}")
        if str(data["engine"].item()) != "numba" or str(data["source_commit"].item()) != manifest["source_commit"]:
            raise ValueError("manifest/result engine or source mismatch")
        if (data["p_values"].shape != (7,) or data["p_values"].dtype != np.float64
                or not np.array_equal(data["p_values"], config["p_values"])):
            raise ValueError("publication p grid mismatch")
        if (data["m_values"].shape != (6,)
                or not np.array_equal(data["m_values"], config["m_values"])):
            raise ValueError("publication m grid mismatch")
        if (data["mu_m"].shape != (6, 7)
                or data["errorbar_between_code_sem"].shape != (6, 7)
                or data["m_status"].shape != (6, 7)
                or data["mu_code"].shape != (48, 7)
                or data["code_status"].shape != (48, 7)
                or data["present_disorders"].shape != (48, 7)
                or data["valid_disorders"].shape != (48, 7)):
            raise ValueError("publication result shape mismatch")
        if (data["mu_m"].dtype != np.float64
                or data["errorbar_between_code_sem"].dtype != np.float64
                or data["mu_code"].dtype != np.float64
                or data["m_status"].dtype.kind != "U"
                or data["code_status"].dtype.kind != "U"
                or data["present_disorders"].dtype.kind not in "iu"
                or data["valid_disorders"].dtype.kind not in "iu"):
            raise ValueError("publication result dtype mismatch")
        if str(data["main_errorbar_definition"].item()) != "std(code_means,ddof=1)/sqrt(8)":
            raise ValueError("main error bar definition was tampered")
        statuses = data["m_status"]
        if point_mask is None:
            mask = np.ones(statuses.shape, dtype=bool)
        else:
            mask = np.asarray(point_mask)
            if mask.dtype != np.bool_:
                raise ValueError("point mask must be a boolean array")
        if mask.shape != statuses.shape:
            raise ValueError("point mask shape mismatch")
        if not np.any(mask):
            raise ValueError("point mask must select at least one parameter point")
        failed = np.argwhere(mask & (statuses != "REPORTABLE"))
        if failed.size:
            i, j = failed[0]
            raise ValueError(f"selected m={int(data['m_values'][i])}, p={float(data['p_values'][j])} is {statuses[i,j]}")
        code_ids = [row["code_id"] for row in registry["codes"]]
        if not np.array_equal(data["code_ids"], code_ids):
            raise ValueError("publication code registry order mismatch")
        for m_index, m in enumerate(config["m_values"]):
            code_indices = [index for index, row in enumerate(registry["codes"]) if row["m"] == m]
            for p_index in range(len(config["p_values"])):
                if not mask[m_index, p_index]:
                    continue
                if (not np.all(data["code_status"][code_indices, p_index] == "REPORTABLE")
                        or not np.all(data["present_disorders"][code_indices, p_index] == 128)
                        or not np.all(data["valid_disorders"][code_indices, p_index] == 128)):
                    raise ValueError("selected publication point lacks complete reportable code evidence")
                values = data["mu_code"][code_indices, p_index]
                expected_mean = np.mean(values)
                expected_error = np.std(values, ddof=1) / np.sqrt(8)
                if (data["mu_m"][m_index, p_index] != expected_mean
                        or data["errorbar_between_code_sem"][m_index, p_index] != expected_error
                        or not np.isfinite(expected_mean) or not np.isfinite(expected_error)):
                    raise ValueError("selected publication statistic disagrees with per-code values")
        q_top = np.where(mask, data["mu_m"], np.nan)
        errorbar = np.where(mask, data["errorbar_between_code_sem"], np.nan)
        return {"m_values": data["m_values"].copy(), "p_values": data["p_values"].copy(),
                "q_top": q_top, "errorbar": errorbar,
                "point_mask": mask.copy()}
