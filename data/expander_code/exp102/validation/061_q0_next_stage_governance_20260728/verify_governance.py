#!/usr/bin/env python3
"""Independently verify the generated Stage-0 governance package."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEPLOYMENT_PREFIX = "data/expander_code/exp102/deployment_worktrees"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    manifest = json.loads((HERE / "stage0_manifest.json").read_text(encoding="ascii"))
    for name, identity in manifest["outputs"].items():
        path = HERE / name
        assert path.stat().st_size == identity["size_bytes"], name
        assert sha256_file(path) == identity["sha256"], name

    inventory = json.loads((HERE / "dirty_root_inventory.json").read_text(encoding="ascii"))
    records = inventory["records"]
    assert len(records) == 1472
    paths = [row["path"] for row in records]
    assert len(paths) == len(set(paths))
    assert not any(path == DEPLOYMENT_PREFIX or path.startswith(f"{DEPLOYMENT_PREFIX}/") for path in paths)
    assert sum(row["status_code"] == " M" for row in records) == 5
    assert sum(row["status_code"] == "??" for row in records) == 1467

    with (HERE / "dirty_root_inventory.csv").open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert [row["path"] for row in csv_rows] == paths

    evidence = json.loads((HERE / "evidence_authority_matrix.json").read_text(encoding="ascii"))["rows"]
    assert len(evidence) == 63
    by_unit = {row["evidence_unit"]: row for row in evidence}
    assert set(f"{number:03d}" for number in range(1, 61)).issubset(by_unit)
    assert all(row["cell_certification"] == "NO" for row in evidence)
    assert all(row["formal_authority"] == "NO" for row in evidence)
    assert by_unit["013:HP64"]["method_internal_pass"] == "YES_5_OF_5"
    assert by_unit["013:HP64"]["cross_method_pass"] == "NO_0_OF_4_VS_MAM"
    assert by_unit["013:MAM-IMH8"]["method_internal_pass"] == "PARTIAL_1_OF_2"
    assert by_unit["060"]["canonical_git_state"] == "UNTRACKED_DRAFT"
    assert by_unit["060"]["raw_existence"] == "NO_SAMPLER_RAW"
    assert by_unit["060"]["replay_audit"] == "NOT_RUN"

    print(
        json.dumps(
            {
                "status": "STAGE0_GOVERNANCE_INDEPENDENT_PACKAGE_CHECK_PASS",
                "inventory_records": len(records),
                "evidence_rows": len(evidence),
                "manifest_outputs": len(manifest["outputs"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
