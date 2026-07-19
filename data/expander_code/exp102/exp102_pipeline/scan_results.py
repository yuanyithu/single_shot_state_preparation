import json
from pathlib import Path

import numpy as np

from . import PHYSICS_VERSION, PT_VERSION, SCAN_VERSION
from .io import sha256_file


def load_exp102_publication_q_top(results_path, point_mask=None):
    results_path = Path(results_path)
    manifest_path = results_path.parent / "aggregation_manifest.json"
    if not manifest_path.exists():
        raise ValueError("missing exp102 aggregation manifest")
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    if manifest.get("result_sha256") != sha256_file(results_path):
        raise ValueError("exp102 result file SHA256 does not match manifest")
    if manifest.get("planned_tasks") != 6144 or manifest.get("present_tasks") != 6144:
        raise ValueError("exp102 manifest does not certify 6144 present production tasks")
    if manifest.get("engine") != "numba" or not manifest.get("source_commit"):
        raise ValueError("exp102 manifest lacks production engine/source identity")
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
        if data["p_values"].shape != (7,) or not np.array_equal(data["p_values"], [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]):
            raise ValueError("publication p grid mismatch")
        if data["mu_m"].shape != (6, 7) or data["errorbar_between_code_sem"].shape != (6, 7):
            raise ValueError("publication result shape mismatch")
        if str(data["main_errorbar_definition"].item()) != "std(code_means,ddof=1)/sqrt(8)":
            raise ValueError("main error bar definition was tampered")
        statuses = data["m_status"]
        mask = np.ones(statuses.shape, dtype=bool) if point_mask is None else np.asarray(point_mask, dtype=bool)
        if mask.shape != statuses.shape:
            raise ValueError("point mask shape mismatch")
        failed = np.argwhere(mask & (statuses != "REPORTABLE"))
        if failed.size:
            i, j = failed[0]
            raise ValueError(f"selected m={int(data['m_values'][i])}, p={float(data['p_values'][j])} is {statuses[i,j]}")
        return {"m_values": data["m_values"].copy(), "p_values": data["p_values"].copy(),
                "q_top": data["mu_m"].copy(), "errorbar": data["errorbar_between_code_sem"].copy(),
                "point_mask": mask.copy()}
