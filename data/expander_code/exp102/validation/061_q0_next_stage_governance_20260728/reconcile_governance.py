#!/usr/bin/env python3
"""Build a read-only Exp102 Stage-0 reconciliation and authority inventory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import stat
import subprocess
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_CANONICAL_HEAD = "bacf25a53870d04dc538d1caaee3336c8fe1be7a"
BASELINE_REF = "origin/main"
DEPLOYMENT_REL = "data/expander_code/exp102/deployment_worktrees"
VALIDATION_REL = Path("data/expander_code/exp102/validation")
OUTPUT_DIR = Path(__file__).resolve().parent


CURATED_EVIDENCE: dict[str, dict[str, str]] = {
    "003": {
        "terminal_status": "BENCHMARK_COMPLETE",
        "method_internal_pass": "N/A",
        "cross_method_pass": "N/A",
        "raw_existence": "BENCHMARK_ARTIFACTS",
        "replay_audit": "UNKNOWN",
        "scientific_role": "IMPLEMENTATION_AND_RUNTIME_BENCHMARK",
    },
    "005": {
        "terminal_status": "PT_V2_EXHAUSTED_ZERO_ROUND_TRIPS",
        "method_internal_pass": "NO",
        "cross_method_pass": "N/A",
        "raw_existence": "YES_ATTESTED",
        "replay_audit": "UNKNOWN",
        "scientific_role": "HISTORICAL_PT_DISCOVERY_FAILURE",
    },
    "006": {
        "terminal_status": "PA_DISCOVERY_EXHAUSTED",
        "method_internal_pass": "NO",
        "cross_method_pass": "N/A",
        "raw_existence": "YES_ATTESTED",
        "replay_audit": "UNKNOWN",
        "scientific_role": "HISTORICAL_PA_DISCOVERY_FAILURE",
    },
    "007": {
        "terminal_status": "REMOTE_PREFLIGHT_RUNTIME_EXHAUSTED",
        "method_internal_pass": "N/A",
        "cross_method_pass": "N/A",
        "raw_existence": "NO_SAMPLER_RAW",
        "replay_audit": "N/A",
        "scientific_role": "GLOBAL_DISCOVERY_IMPLEMENTATION_AND_PREFLIGHT",
    },
    "008": {
        "terminal_status": "FIRST_FAILURE_AUDIT_REPAIR_COMPLETE",
        "method_internal_pass": "N/A",
        "cross_method_pass": "N/A",
        "raw_existence": "NO_SAMPLER_RAW",
        "replay_audit": "PARTIAL",
        "scientific_role": "PORTABILITY_FAILURE_AUDIT",
    },
    "009": {
        "terminal_status": "RUNTIME_GATE_REPAIR_IMPLEMENTED",
        "method_internal_pass": "N/A",
        "cross_method_pass": "N/A",
        "raw_existence": "NO_SAMPLER_RAW",
        "replay_audit": "PARTIAL",
        "scientific_role": "RUNTIME_GATE_SEPARATION_AUDIT",
    },
    "010": {
        "terminal_status": "RUNTIME_EXHAUSTED_BEFORE_SCREEN",
        "method_internal_pass": "N/A",
        "cross_method_pass": "N/A",
        "raw_existence": "NO_SAMPLER_RAW",
        "replay_audit": "PASS",
        "scientific_role": "IMMUTABLE_PREFLIGHT_RUNTIME_FAILURE",
    },
    "011": {
        "terminal_status": "UNRESOLVED_NO_HARD_COSET_PASS",
        "method_internal_pass": "NO_0_OF_5_EACH",
        "cross_method_pass": "N/A_NO_SELECTED_PAIR",
        "raw_existence": "YES_15_BIAS_1280_MEASUREMENT_ATTESTED",
        "replay_audit": "PASS",
        "scientific_role": "GLOBAL_SAMPLER_DIAGNOSTIC_SCREEN",
    },
    "012": {
        "terminal_status": "LOCAL_FEASIBILITY_ONLY",
        "method_internal_pass": "PARTIAL",
        "cross_method_pass": "N/A",
        "raw_existence": "YES_ATTESTED",
        "replay_audit": "UNKNOWN",
        "scientific_role": "PROVISIONAL_COLLAPSED_HGP_DESIGN_EVIDENCE",
    },
    "013": {
        "terminal_status": "UNRESOLVED_MAP_MIXTURE_FAIL",
        "method_internal_pass": "MIXED_SEE_METHOD_ROWS",
        "cross_method_pass": "NO_0_OF_4_HP64_MAM_FAMILY_CELL_COMPARISONS",
        "raw_existence": "YES_384_SAMPLER_2_IS_ATTESTED",
        "replay_audit": "PASS_386_OF_386",
        "scientific_role": "PRE_PILOT_HGP_DIAGNOSTIC_SCREEN",
    },
    "014": {
        "terminal_status": "CONFLICT_CROSS_ENV_ARTIFACT_IDENTITY",
        "method_internal_pass": "N/A_NOT_RUN",
        "cross_method_pass": "N/A",
        "raw_existence": "NO_MEASUREMENT_RAW",
        "replay_audit": "FAIL_ARTIFACT_PORTABILITY",
        "scientific_role": "LOGICAL_SIGNATURE_ARTIFACT_FAILURE",
    },
    "015": {
        "terminal_status": "STATUS_NOT_CANONICALLY_SUMMARIZED",
        "method_internal_pass": "UNKNOWN",
        "cross_method_pass": "N/A",
        "raw_existence": "UNKNOWN",
        "replay_audit": "UNKNOWN",
        "scientific_role": "LOGICAL_SIGNATURE_SUCCESSOR_CONTRACT",
    },
    "047": {
        "terminal_status": "LOCAL_CENTER_PRESERVING_STRUCTURE_NOT_VIABLE",
        "method_internal_pass": "N/A_STRUCTURE_ONLY",
        "cross_method_pass": "N/A",
        "raw_existence": "NO_MARKOV_RAW",
        "replay_audit": "PASS_VIA_051",
        "scientific_role": "LOCAL_STRUCTURE_FEASIBILITY",
    },
    "051": {
        "terminal_status": "INDEPENDENT_AUDIT_PASS_FAILED_RESULTS_PRESERVED",
        "method_internal_pass": "N/A_AUDIT",
        "cross_method_pass": "N/A",
        "raw_existence": "N/A_AUDIT_ONLY",
        "replay_audit": "PASS",
        "scientific_role": "INDEPENDENT_AUDIT_OF_047_049_050",
    },
    "052": {
        "terminal_status": "RUNTIME_EXHAUSTED_BEFORE_MEASUREMENT",
        "method_internal_pass": "N/A_NOT_RUN",
        "cross_method_pass": "N/A",
        "raw_existence": "NO_MEASUREMENT_RAW",
        "replay_audit": "PASS_PREFLIGHT_AUDIT",
        "scientific_role": "RANDOM_FULL_COLUMN_RUNTIME_PREFLIGHT",
    },
    "053": {
        "terminal_status": "CONFLICT_AND_RUNTIME_EXHAUSTED_NO_T1_RAW",
        "method_internal_pass": "N/A_NOT_RUN",
        "cross_method_pass": "N/A",
        "raw_existence": "NO_MEASUREMENT_RAW",
        "replay_audit": "PASS_CONFLICT_CONFIRMED",
        "scientific_role": "STREAMING_IMPLEMENTATION_PREFLIGHT",
    },
    "054": {
        "terminal_status": "PREFLIGHT_PASS_ONLY",
        "method_internal_pass": "N/A_PREFLIGHT",
        "cross_method_pass": "N/A",
        "raw_existence": "NO_MEASUREMENT_RAW",
        "replay_audit": "PASS",
        "scientific_role": "DIRECT_BLOCK_PREFLIGHT",
    },
    "055": {
        "terminal_status": "RUNTIME_EXHAUSTED_ZERO_MEASUREMENT_RAW",
        "method_internal_pass": "N/A_NOT_RUN",
        "cross_method_pass": "N/A",
        "raw_existence": "NO_MEASUREMENT_RAW",
        "replay_audit": "PASS_PREFLIGHT_AUDIT",
        "scientific_role": "DIRECT_BLOCK_RUNTIME_PREFLIGHT",
    },
    "056": {
        "terminal_status": "UNRESOLVED_DIRECT_BLOCK_T1_M8",
        "method_internal_pass": "NO",
        "cross_method_pass": "N/A",
        "raw_existence": "YES_ATTESTED",
        "replay_audit": "PASS_RAW_ONLY",
        "scientific_role": "DIRECT_BLOCK_LOCAL_TRANSPORT_DIAGNOSTIC",
    },
    "057": {
        "terminal_status": "LOCAL_T1_PAIR_UNRESOLVED_DO_NOT_DEPLOY",
        "method_internal_pass": "NO",
        "cross_method_pass": "N/A_SAME_FAMILY",
        "raw_existence": "YES_2_RAW_ATTESTED",
        "replay_audit": "PASS_RAW_ONLY",
        "scientific_role": "COLLAPSED_PHYSICAL_PT_SAME_FAMILY_ORACLE",
    },
    "058": {
        "terminal_status": "LOCAL_CONDITIONAL_FEASIBLE_STANDALONE_TRANSPORT_NOT_VIABLE",
        "method_internal_pass": "NO_TRANSPORT",
        "cross_method_pass": "N/A_SAME_FAILURE_FAMILY",
        "raw_existence": "NO_SAMPLER_TRAJECTORY_RAW",
        "replay_audit": "PASS_TARGET_ONLY",
        "scientific_role": "EXACT_FULL_ROW_CONDITIONAL_FEASIBILITY",
    },
    "059": {
        "terminal_status": "LOCAL_HYBRID_B_NECESSARY_GATES_FAIL",
        "method_internal_pass": "NO",
        "cross_method_pass": "N/A_SAME_FAILURE_FAMILY",
        "raw_existence": "YES_16_TRAJECTORIES_ATTESTED",
        "replay_audit": "PASS_RAW_ONLY",
        "scientific_role": "LOCAL_HYBRID_ROW_COLUMN_TRANSPORT_DIAGNOSTIC",
    },
    "060": {
        "terminal_status": "PRE_RUN_NO_WIDTH_REPORT",
        "method_internal_pass": "N/A_STRUCTURE_NOT_RUN",
        "cross_method_pass": "N/A_SAME_FAILURE_FAMILY",
        "raw_existence": "NO_SAMPLER_RAW",
        "replay_audit": "NOT_RUN",
        "scientific_role": "UNTRACKED_LOCAL_STRUCTURE_DRAFT_ONLY",
    },
}


METHOD_ROWS = [
    {
        "evidence_unit": "013:HP64",
        "validation_id": "013",
        "terminal_status": "METHOD_INTERNAL_PASS_5_OF_5_DIAGNOSTIC_CELLS",
        "method_internal_pass": "YES_5_OF_5",
        "cross_method_pass": "NO_0_OF_4_VS_MAM",
        "cell_certification": "NO",
        "formal_authority": "NO",
        "raw_existence": "YES_160_TRAJECTORIES_ATTESTED",
        "replay_audit": "PASS_WITHIN_386_OF_386",
        "scientific_role": "PROMISING_PRIMARY_CANDIDATE_DIAGNOSTIC_ONLY",
        "authority_note": "No fresh T/2T, full cell, orthogonal confirmation, tuning, or held-out.",
    },
    {
        "evidence_unit": "013:HP32",
        "validation_id": "013",
        "terminal_status": "METHOD_INTERNAL_PARTIAL_3_OF_5",
        "method_internal_pass": "PARTIAL_3_OF_5",
        "cross_method_pass": "N/A_HP32_HP64_SAME_MECHANISM",
        "cell_certification": "NO",
        "formal_authority": "NO",
        "raw_existence": "YES_160_TRAJECTORIES_ATTESTED",
        "replay_audit": "PASS_WITHIN_386_OF_386",
        "scientific_role": "SAME_MECHANISM_DIAGNOSTIC_NOT_CONFIRMER",
        "authority_note": "Borderline m3 B-character failure and clear m5 B slow mode.",
    },
    {
        "evidence_unit": "013:MAM-IMH8",
        "validation_id": "013",
        "terminal_status": "METHOD_INTERNAL_PARTIAL_1_OF_2_HARD_CELLS",
        "method_internal_pass": "PARTIAL_1_OF_2",
        "cross_method_pass": "NO_0_OF_4_VS_HP64",
        "cell_certification": "NO",
        "formal_authority": "NO",
        "raw_existence": "YES_64_TRAJECTORIES_ATTESTED",
        "replay_audit": "PASS_WITHIN_386_OF_386",
        "scientific_role": "ORTHOGONAL_DIAGNOSTIC_M8_TRANSPORT_FAILURE_M6_DISCREPANCY",
        "authority_note": "m6 differs from HP64 by about 30 paired SE; m8 also fails internal gates.",
    },
]


def run_git(worktree: Path, *args: str, check: bool = True) -> bytes:
    env = os.environ.copy()
    env["GIT_OPTIONAL_LOCKS"] = "0"
    proc = subprocess.run(
        ["git", "-C", str(worktree), *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed in {worktree}: "
            f"{proc.stderr.decode('utf-8', 'replace')}"
        )
    return proc.stdout


def decode_path(value: bytes) -> str:
    return value.decode("utf-8", "surrogateescape")


def parse_porcelain_v1_z(raw: bytes) -> list[dict[str, str]]:
    fields = raw.split(b"\0")
    records: list[dict[str, str]] = []
    index = 0
    while index < len(fields):
        field = fields[index]
        index += 1
        if not field:
            continue
        if len(field) < 4 or field[2:3] != b" ":
            raise ValueError(f"Unexpected porcelain-v1 record: {field!r}")
        status_code = field[:2].decode("ascii")
        path = decode_path(field[3:])
        record = {"status_code": status_code, "path": path}
        if "R" in status_code or "C" in status_code:
            if index >= len(fields):
                raise ValueError("Rename/copy record lacks its second path")
            record["source_path"] = decode_path(fields[index])
            index += 1
        records.append(record)
    return records


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_path(path: Path) -> dict[str, Any]:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return {"state": "ABSENT", "object_type": None, "size_bytes": None, "sha256": None}
    mode = info.st_mode
    if stat.S_ISREG(mode):
        return {
            "state": "PRESENT",
            "object_type": "regular_file",
            "size_bytes": info.st_size,
            "sha256": sha256_file(path),
        }
    if stat.S_ISLNK(mode):
        target = os.readlink(path).encode("utf-8", "surrogateescape")
        return {
            "state": "PRESENT",
            "object_type": "symlink",
            "size_bytes": len(target),
            "sha256": sha256_bytes(target),
            "link_target": os.readlink(path),
        }
    if stat.S_ISDIR(mode):
        return {"state": "PRESENT", "object_type": "directory", "size_bytes": None, "sha256": None}
    return {"state": "PRESENT", "object_type": "special", "size_bytes": info.st_size, "sha256": None}


def baseline_tree(canonical: Path, ref: str) -> dict[str, dict[str, str]]:
    raw = run_git(canonical, "ls-tree", "-r", "-z", "--full-tree", ref)
    result: dict[str, dict[str, str]] = {}
    for field in raw.split(b"\0"):
        if not field:
            continue
        metadata, path_bytes = field.split(b"\t", 1)
        mode, object_type, oid = metadata.decode("ascii").split()
        result[decode_path(path_bytes)] = {"mode": mode, "git_type": object_type, "git_oid": oid}
    return result


def baseline_fingerprint(
    canonical: Path, tree: dict[str, dict[str, str]], path: str
) -> dict[str, Any]:
    entry = tree.get(path)
    if entry is None:
        return {"state": "ABSENT", "object_type": None, "size_bytes": None, "sha256": None}
    if entry["git_type"] != "blob":
        return {
            "state": "PRESENT",
            "object_type": entry["git_type"],
            "size_bytes": None,
            "sha256": None,
            **entry,
        }
    content = run_git(canonical, "cat-file", "blob", entry["git_oid"])
    object_type = "symlink" if entry["mode"] == "120000" else "regular_file"
    return {
        "state": "PRESENT",
        "object_type": object_type,
        "size_bytes": len(content),
        "sha256": sha256_bytes(content),
        **entry,
    }


def fingerprints_equal(left: dict[str, Any], right: dict[str, Any]) -> bool | None:
    if left["state"] != "PRESENT" or right["state"] != "PRESENT":
        return None
    if left["sha256"] is None or right["sha256"] is None:
        return None
    return left["object_type"] == right["object_type"] and left["sha256"] == right["sha256"]


def parse_worktree_list(raw: bytes) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    for line in raw.decode("utf-8", "surrogateescape").splitlines():
        if not line:
            if current:
                records.append(current)
                current = {}
            continue
        key, _, value = line.partition(" ")
        current[key] = value if value else True
    if current:
        records.append(current)
    return records


def worktree_statuses(canonical: Path) -> list[dict[str, Any]]:
    rows = parse_worktree_list(run_git(canonical, "worktree", "list", "--porcelain"))
    for row in rows:
        path = Path(str(row["worktree"])).resolve()
        status = run_git(
            path,
            "status",
            "--porcelain=v1",
            "--untracked-files=normal",
            check=False,
        ).decode("utf-8", "surrogateescape").splitlines()
        row["status_mode"] = "porcelain-v1; untracked-files=normal (directories not expanded)"
        row["dirty"] = bool(status)
        row["status_entry_count"] = len(status)
        row["status_entries"] = status
    return rows


def find_validation_dir(root: Path, validation_id: str) -> Path | None:
    parent = root / VALIDATION_REL
    if not parent.is_dir():
        return None
    matches = sorted(parent.glob(f"{validation_id}_*"))
    if not matches:
        return None
    if len(matches) > 1:
        raise RuntimeError(f"Multiple validation directories for {validation_id}: {matches}")
    return matches[0]


def git_state_for_directory(canonical: Path, directory: Path | None) -> str:
    if directory is None:
        return "MISSING"
    relative = directory.relative_to(canonical).as_posix()
    tracked = run_git(canonical, "ls-files", "--", relative).splitlines()
    return "TRACKED" if tracked else "UNTRACKED_DRAFT"


def build_evidence_matrix(canonical: Path, dirty_root: Path, direct_block: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number in range(1, 61):
        validation_id = f"{number:03d}"
        canonical_dir = find_validation_dir(canonical, validation_id)
        dirty_dir = find_validation_dir(dirty_root, validation_id)
        direct_dir = find_validation_dir(direct_block, validation_id)
        override = CURATED_EVIDENCE.get(validation_id, {})
        row: dict[str, Any] = {
            "evidence_unit": validation_id,
            "validation_id": validation_id,
            "canonical_path": (
                canonical_dir.relative_to(canonical).as_posix() if canonical_dir else None
            ),
            "canonical_git_state": git_state_for_directory(canonical, canonical_dir),
            "dirty_root_path": (
                dirty_dir.relative_to(dirty_root).as_posix() if dirty_dir else None
            ),
            "direct_block_path": (
                direct_dir.relative_to(direct_block).as_posix() if direct_dir else None
            ),
            "terminal_status": override.get("terminal_status", "UNKNOWN"),
            "method_internal_pass": override.get("method_internal_pass", "UNKNOWN"),
            "cross_method_pass": override.get("cross_method_pass", "UNKNOWN"),
            "cell_certification": "NO",
            "formal_authority": "NO",
            "raw_existence": override.get("raw_existence", "UNKNOWN"),
            "replay_audit": override.get("replay_audit", "UNKNOWN"),
            "scientific_role": override.get("scientific_role", "UNKNOWN"),
            "authority_note": (
                "No Exp102 validation currently certifies a (m,p) point or creates formal authority."
            ),
        }
        rows.append(row)
        if validation_id == "013":
            for method_row in METHOD_ROWS:
                method_copy = dict(method_row)
                method_copy.update(
                    {
                        "canonical_path": row["canonical_path"],
                        "canonical_git_state": row["canonical_git_state"],
                        "dirty_root_path": row["dirty_root_path"],
                        "direct_block_path": row["direct_block_path"],
                    }
                )
                rows.append(method_copy)
    return rows


def write_json(path: Path, value: Any) -> None:
    text = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    atomic_write(path, text.encode("ascii"))


def atomic_write(path: Path, content: bytes) -> None:
    if path.parent.resolve() != OUTPUT_DIR:
        raise RuntimeError(f"Refusing to write outside {OUTPUT_DIR}: {path}")
    with tempfile.NamedTemporaryFile(dir=OUTPUT_DIR, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    path.chmod(0o644)


def write_inventory_csv(path: Path, records: list[dict[str, Any]]) -> None:
    columns = [
        "status_code",
        "path",
        "source_path",
        "current_object_type",
        "current_size_bytes",
        "current_sha256",
        "origin_main_state",
        "origin_main_sha256",
        "same_as_origin_main",
        "direct_block_state",
        "direct_block_sha256",
        "same_as_direct_block",
    ]
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=OUTPUT_DIR, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in records:
            writer.writerow({key: row.get(key) for key in columns})
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    path.chmod(0o644)


def markdown_escape(value: Any) -> str:
    if value is None:
        return "N/A"
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def reconciliation_markdown(
    metadata: dict[str, Any], records: list[dict[str, Any]], worktrees: list[dict[str, Any]]
) -> str:
    status_counts = Counter(row["status_code"] for row in records)
    origin_overlap = [row for row in records if row["origin_main_state"] == "PRESENT"]
    direct_overlap = [row for row in records if row["direct_block_state"] == "PRESENT"]
    origin_different = [row for row in origin_overlap if row["same_as_origin_main"] is False]
    direct_different = [row for row in direct_overlap if row["same_as_direct_block"] is False]
    tracked_rows = [row for row in records if row["status_code"] != "??"]

    lines = [
        "# Exp102 Stage-0 reconciliation report",
        "",
        f"Generated: `{metadata['generated_at_utc']}` under conda environment `{metadata['conda_environment']}`.",
        "",
        "This report is read-only evidence. It does not reconcile, copy, delete, merge, commit, or launch anything.",
        "The complete path-level inventory is in `dirty_root_inventory.json` and `.csv`.",
        "",
        "## Source identities",
        "",
        f"- Dirty root HEAD: `{metadata['dirty_root_head']}`; branch `{metadata['dirty_root_branch']}`.",
        f"- Canonical worktree HEAD: `{metadata['canonical_head']}`; expected `{EXPECTED_CANONICAL_HEAD}`.",
        f"- Baseline `{BASELINE_REF}`: `{metadata['baseline_head']}`.",
        f"- Dirty root versus baseline: ahead `{metadata['dirty_root_ahead']}`, behind `{metadata['dirty_root_behind']}`.",
        f"- Direct-block draft HEAD: `{metadata['direct_block_head']}`.",
        f"- Deployment subtree excluded from dirty-root enumeration: `{DEPLOYMENT_REL}`.",
        "",
        "## Inventory summary",
        "",
        f"- Dirty modified/untracked file records outside deployment worktrees: **{len(records)}**.",
        f"- Status counts: `{dict(sorted(status_counts.items()))}`.",
        f"- Paths also present in origin/main: **{len(origin_overlap)}**; byte-different: **{len(origin_different)}**.",
        f"- Paths also present in direct-block draft: **{len(direct_overlap)}**; byte-different: **{len(direct_different)}**.",
        "- SHA values are SHA-256 of regular-file bytes; symlinks hash their link-target bytes.",
        "",
        "## Tracked dirty paths",
        "",
        "| XY | Path | Current SHA-256 | Same as origin/main | Same as direct-block |",
        "|---|---|---|---:|---:|",
    ]
    for row in tracked_rows:
        lines.append(
            "| "
            + " | ".join(
                markdown_escape(row.get(key))
                for key in (
                    "status_code",
                    "path",
                    "current_sha256",
                    "same_as_origin_main",
                    "same_as_direct_block",
                )
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Overlap differences requiring an ownership decision",
            "",
            "These rows exist in at least one comparison source but are not byte-identical to it. No choice is made here.",
            "",
            "| Path | XY | Same as origin/main | Same as direct-block |",
            "|---|---|---:|---:|",
        ]
    )
    overlap_differences = sorted(
        {
            row["path"]: row
            for row in origin_different + direct_different
        }.values(),
        key=lambda item: item["path"],
    )
    for row in overlap_differences:
        lines.append(
            "| "
            + " | ".join(
                markdown_escape(row.get(key))
                for key in ("path", "status_code", "same_as_origin_main", "same_as_direct_block")
            )
            + " |"
        )
    if not overlap_differences:
        lines.append("| _None_ | N/A | N/A | N/A |")

    lines.extend(
        [
            "",
            "## Registered worktrees",
            "",
            "Each worktree was queried with `GIT_OPTIONAL_LOCKS=0` and untracked directories collapsed; no deployment tree was recursively inventoried.",
            "",
            "| Worktree | HEAD | Branch/state | Dirty | Status entries |",
            "|---|---|---|---:|---:|",
        ]
    )
    for row in worktrees:
        branch = row.get("branch", "detached" if row.get("detached") else "UNKNOWN")
        lines.append(
            f"| {markdown_escape(row['worktree'])} | `{row.get('HEAD', 'UNKNOWN')}` | "
            f"{markdown_escape(branch)} | {row['dirty']} | {row['status_entry_count']} |"
        )

    lines.extend(
        [
            "",
            "## Governance decision",
            "",
            "- The dirty root is not a safe development or merge target.",
            "- No root path was changed; byte comparisons are evidence for a later human-owned reconciliation.",
            "- Canonical follow-up work must remain on the bacf25a-based successor branch and explicitly select any dirty-root draft it needs.",
            "- Validation 060 is present as an untracked draft, remains `PRE-RUN`, and has no immutable or sampler authority.",
            "- No remote, sampler, formal, held-out, or production stage is authorized by this report.",
            "",
        ]
    )
    return "\n".join(lines)


def evidence_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Exp102 validation evidence and authority matrix",
        "",
        "`UNKNOWN` is intentional when the canonical source does not prove a field. `N/A` means the field is not applicable.",
        "A method-internal pass is never promoted to cell certification or formal authority.",
        "",
        "| Unit | Canonical state | Terminal status | Internal pass | Cross-method pass | Cell certified | Formal | Raw | Replay/audit | Scientific role |",
        "|---|---|---|---|---|---:|---:|---|---|---|",
    ]
    keys = (
        "evidence_unit",
        "canonical_git_state",
        "terminal_status",
        "method_internal_pass",
        "cross_method_pass",
        "cell_certification",
        "formal_authority",
        "raw_existence",
        "replay_audit",
        "scientific_role",
    )
    for row in rows:
        lines.append("| " + " | ".join(markdown_escape(row.get(key)) for key in keys) + " |")
    lines.extend(
        [
            "",
            "## Key interpretations",
            "",
            "- `013:HP64` passed its own diagnostic gates on 5/5 single-disorder cells, but HP64/MAM passed 0/4 family-cell comparisons. No cell is certified.",
            "- `013:MAM-IMH8` passed its own gates on m6 yet differs from HP64 by about 30 paired SE; m8 additionally fails its own mixing gates.",
            "- HP32 and HP64 are settings of the same collapsed mechanism and cannot independently confirm each other.",
            "- Validation 060 is an untracked pre-run structure draft with no width report and no sampler raw. Even a survivor would be a same-family primary candidate, not an orthogonal confirmer.",
            "- Missing canonical validation directories are recorded as `UNKNOWN`; dirty-root presence does not create canonical or formal authority.",
            "- Project-wide status remains pre-pilot: no `READY_FOR_FORMAL`, no `FROZEN_HELD_OUT_PASS`, and no production authorization.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dirty-root", type=Path, required=True)
    parser.add_argument("--direct-block", type=Path, required=True)
    parser.add_argument("--baseline-ref", default=BASELINE_REF)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    canonical = Path(
        run_git(OUTPUT_DIR, "rev-parse", "--show-toplevel").decode("utf-8").strip()
    ).resolve()
    dirty_root = args.dirty_root.resolve()
    direct_block = args.direct_block.resolve()
    for label, path in (("dirty root", dirty_root), ("direct block", direct_block)):
        if not path.is_dir():
            raise RuntimeError(f"{label} is not a directory: {path}")
        run_git(path, "rev-parse", "--show-toplevel")

    canonical_head = run_git(canonical, "rev-parse", "HEAD").decode("ascii").strip()
    baseline_head = run_git(canonical, "rev-parse", args.baseline_ref).decode("ascii").strip()
    if canonical_head != EXPECTED_CANONICAL_HEAD or baseline_head != EXPECTED_CANONICAL_HEAD:
        raise RuntimeError(
            f"Canonical identity mismatch: HEAD={canonical_head}, {args.baseline_ref}={baseline_head}"
        )
    if OUTPUT_DIR.relative_to(canonical) != VALIDATION_REL / OUTPUT_DIR.name:
        raise RuntimeError(f"Unexpected output directory: {OUTPUT_DIR}")

    pathspec = [
        "--",
        ".",
        f":(exclude){DEPLOYMENT_REL}",
        f":(exclude){DEPLOYMENT_REL}/**",
    ]
    status_raw = run_git(
        dirty_root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        *pathspec,
    )
    status_records = parse_porcelain_v1_z(status_raw)
    baseline = baseline_tree(canonical, args.baseline_ref)

    inventory: list[dict[str, Any]] = []
    for status_record in status_records:
        relative = status_record["path"]
        if relative == DEPLOYMENT_REL or relative.startswith(f"{DEPLOYMENT_REL}/"):
            raise RuntimeError(f"Deployment path escaped exclusion: {relative}")
        current = fingerprint_path(dirty_root / relative)
        origin = baseline_fingerprint(canonical, baseline, relative)
        direct = fingerprint_path(direct_block / relative)
        inventory.append(
            {
                **status_record,
                "current_state": current["state"],
                "current_object_type": current["object_type"],
                "current_size_bytes": current["size_bytes"],
                "current_sha256": current["sha256"],
                "origin_main_state": origin["state"],
                "origin_main_object_type": origin["object_type"],
                "origin_main_size_bytes": origin["size_bytes"],
                "origin_main_sha256": origin["sha256"],
                "same_as_origin_main": fingerprints_equal(current, origin),
                "direct_block_state": direct["state"],
                "direct_block_object_type": direct["object_type"],
                "direct_block_size_bytes": direct["size_bytes"],
                "direct_block_sha256": direct["sha256"],
                "same_as_direct_block": fingerprints_equal(current, direct),
            }
        )
    inventory.sort(key=lambda row: row["path"])

    worktrees = worktree_statuses(canonical)
    evidence_rows = build_evidence_matrix(canonical, dirty_root, direct_block)
    ahead_behind = run_git(
        dirty_root, "rev-list", "--left-right", "--count", f"HEAD...{args.baseline_ref}"
    ).decode("ascii").split()
    metadata = {
        "version": "exp102.stage0_governance.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV", "UNKNOWN"),
        "dirty_root": str(dirty_root),
        "dirty_root_head": run_git(dirty_root, "rev-parse", "HEAD").decode("ascii").strip(),
        "dirty_root_branch": run_git(dirty_root, "branch", "--show-current").decode("utf-8").strip(),
        "dirty_root_ahead": int(ahead_behind[0]),
        "dirty_root_behind": int(ahead_behind[1]),
        "canonical_worktree": str(canonical),
        "canonical_head": canonical_head,
        "baseline_ref": args.baseline_ref,
        "baseline_head": baseline_head,
        "direct_block": str(direct_block),
        "direct_block_head": run_git(direct_block, "rev-parse", "HEAD").decode("ascii").strip(),
        "deployment_subtree_excluded": DEPLOYMENT_REL,
        "dirty_record_count": len(inventory),
        "evidence_row_count": len(evidence_rows),
    }

    write_json(OUTPUT_DIR / "dirty_root_inventory.json", {"metadata": metadata, "records": inventory})
    write_inventory_csv(OUTPUT_DIR / "dirty_root_inventory.csv", inventory)
    write_json(OUTPUT_DIR / "worktree_status.json", {"metadata": metadata, "worktrees": worktrees})
    write_json(OUTPUT_DIR / "evidence_authority_matrix.json", {"metadata": metadata, "rows": evidence_rows})
    atomic_write(
        OUTPUT_DIR / "RECONCILIATION_REPORT.md",
        reconciliation_markdown(metadata, inventory, worktrees).encode("utf-8"),
    )
    atomic_write(
        OUTPUT_DIR / "EVIDENCE_AUTHORITY_MATRIX.md",
        evidence_markdown(evidence_rows).encode("utf-8"),
    )

    output_names = [
        "dirty_root_inventory.json",
        "dirty_root_inventory.csv",
        "worktree_status.json",
        "evidence_authority_matrix.json",
        "RECONCILIATION_REPORT.md",
        "EVIDENCE_AUTHORITY_MATRIX.md",
    ]
    manifest = {
        "metadata": metadata,
        "outputs": {
            name: {"size_bytes": (OUTPUT_DIR / name).stat().st_size, "sha256": sha256_file(OUTPUT_DIR / name)}
            for name in output_names
        },
    }
    write_json(OUTPUT_DIR / "stage0_manifest.json", manifest)
    print(json.dumps({"status": "STAGE0_GOVERNANCE_REPORT_COMPLETE", **metadata}, sort_keys=True))


if __name__ == "__main__":
    main()
