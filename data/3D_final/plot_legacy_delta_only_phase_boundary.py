#!/usr/bin/env python3
"""Validate and render the summary-derived legacy_delta_only phase boundary.

This script intentionally performs no fit, interpolation, bootstrap, or raw-data
analysis. It renders the frozen finite-size crossing estimates in the adjacent
CSV after checking them against the adjacent metadata JSON and source hashes.
"""

from __future__ import annotations

import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


SCRIPT = Path(__file__).resolve()
OUTPUT_DIR = SCRIPT.parent
REPO_ROOT = SCRIPT.parents[2]
CSV_PATH = OUTPUT_DIR / "legacy_delta_only_phase_boundary.csv"
JSON_PATH = OUTPUT_DIR / "legacy_delta_only_phase_boundary.json"
PNG_PATH = OUTPUT_DIR / "legacy_delta_only_phase_boundary.png"
PDF_PATH = OUTPUT_DIR / "legacy_delta_only_phase_boundary.pdf"

FIELDS = (
    "model",
    "observable",
    "role",
    "p",
    "q_crossing",
    "ci95_low",
    "ci95_high",
    "L_pair",
    "disorders",
    "nboot",
    "source",
)
EXPECTED = {
    (Decimal("0.05"), "w0"): (Decimal("0.0320"), Decimal("0.0304"), Decimal("0.0350")),
    (Decimal("0.05"), "q_top=q_W"): (Decimal("0.0343"), Decimal("0.0309"), Decimal("0.0409")),
    (Decimal("0.11"), "w0"): (Decimal("0.0338"), Decimal("0.0314"), Decimal("0.0358")),
    (Decimal("0.11"), "q_top=q_W"): (Decimal("0.0400"), Decimal("0.0327"), Decimal("0.0420")),
    (Decimal("0.17"), "w0"): (Decimal("0.0351"), Decimal("0.0284"), Decimal("0.0367")),
    (Decimal("0.17"), "q_top=q_W"): (Decimal("0.0411"), Decimal("0.0371"), Decimal("0.0436")),
    (Decimal("0.21"), "w0"): (Decimal("0.0344"), Decimal("0.0284"), Decimal("0.0370")),
    (Decimal("0.21"), "q_top=q_W"): (Decimal("0.0405"), Decimal("0.0365"), Decimal("0.0429")),
    (Decimal("0.22"), "w0"): (Decimal("0.0349"), Decimal("0.0327"), Decimal("0.0363")),
    (Decimal("0.22"), "q_top=q_W"): (Decimal("0.0404"), Decimal("0.0371"), Decimal("0.0431")),
}
ANCHOR_P = Decimal("0.22684326965111049")


def fail(message: str) -> None:
    raise ValueError(message)


def as_decimal(value: object, field: str) -> Decimal:
    try:
        return Decimal(str(value))
    except Exception as exc:
        raise ValueError(f"invalid decimal in {field}: {value!r}") from exc


def canonical_record(record: dict[str, object]) -> tuple[object, ...]:
    if set(record) != set(FIELDS):
        fail(f"record fields differ from schema: {sorted(record)}")
    return (
        str(record["model"]),
        str(record["observable"]),
        str(record["role"]),
        as_decimal(record["p"], "p"),
        as_decimal(record["q_crossing"], "q_crossing"),
        as_decimal(record["ci95_low"], "ci95_low"),
        as_decimal(record["ci95_high"], "ci95_high"),
        str(record["L_pair"]),
        int(record["disorders"]),
        int(record["nboot"]),
        str(record["source"]),
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_and_validate() -> tuple[list[dict[str, str]], dict[str, object]]:
    with CSV_PATH.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != FIELDS:
            fail(f"CSV header must be exactly {FIELDS}")
        rows = list(reader)

    if len(rows) != len(EXPECTED):
        fail(f"expected {len(EXPECTED)} rows, found {len(rows)}")

    keys: set[tuple[str, str, str, Decimal, str]] = set()
    observed: dict[tuple[Decimal, str], tuple[Decimal, Decimal, Decimal]] = {}
    for row in rows:
        p = as_decimal(row["p"], "p")
        point = as_decimal(row["q_crossing"], "q_crossing")
        low = as_decimal(row["ci95_low"], "ci95_low")
        high = as_decimal(row["ci95_high"], "ci95_high")
        unique_key = (row["model"], row["observable"], row["role"], p, row["L_pair"])
        if unique_key in keys:
            fail(f"duplicate CSV key: {unique_key}")
        keys.add(unique_key)
        if not low <= point <= high:
            fail(f"CI does not contain point estimate for {unique_key}")
        if row["model"] != "legacy_delta_only":
            fail(f"unexpected model: {row['model']}")
        expected_role = "threshold" if row["observable"] == "w0" else "companion_only"
        if row["observable"] not in {"w0", "q_top=q_W"} or row["role"] != expected_role:
            fail(f"invalid observable/role pair: {row['observable']}/{row['role']}")
        if row["L_pair"] != "L3-L7" or int(row["disorders"]) != 384 or int(row["nboot"]) != 10000:
            fail(f"invalid finite-size provenance for {unique_key}")
        source_path = REPO_ROOT / row["source"]
        if not source_path.is_file():
            fail(f"missing row source: {row['source']}")
        observed[(p, row["observable"])] = (point, low, high)

    if observed != EXPECTED:
        fail(f"CSV values differ from the frozen source values: {observed}")

    metadata = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1:
        fail("unsupported metadata schema_version")
    permissions = metadata.get("model_permissions", {})
    if permissions.get("model") != "legacy_delta_only":
        fail("metadata model permission is not legacy_delta_only")
    if permissions.get("threshold_observable") != "w0":
        fail("metadata must designate only w0 as the threshold observable")
    if permissions.get("companion_role") != "companion_only":
        fail("metadata must designate q_top=q_W as companion only")

    csv_records = sorted((canonical_record(row) for row in rows), key=repr)
    json_records = sorted((canonical_record(row) for row in metadata.get("records", [])), key=repr)
    if csv_records != json_records:
        fail("CSV and JSON records are not numerically and semantically identical")

    listed_sources: set[str] = set()
    for item in metadata.get("source_files", []):
        path_text = str(item.get("path", ""))
        expected_hash = str(item.get("sha256", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
            fail(f"invalid SHA256 for {path_text}")
        path = REPO_ROOT / path_text
        if not path.is_file() or sha256(path) != expected_hash:
            fail(f"source hash mismatch: {path_text}")
        listed_sources.add(path_text)
    row_sources = {row["source"] for row in rows}
    anchor_source = str(metadata.get("internal_q0_anchor", {}).get("source", ""))
    if not row_sources | {anchor_source} <= listed_sources:
        fail("not every row and anchor source has a verified SHA256")

    anchor = metadata.get("internal_q0_anchor", {})
    if as_decimal(anchor.get("p"), "anchor p") != ANCHOR_P or as_decimal(anchor.get("q"), "anchor q") != 0:
        fail("internal q=0 anchor differs from the frozen exp10 value")
    if anchor.get("role") != "finite_size_internal_anchor" or anchor.get("exact_or_asymptotic") is not False:
        fail("internal anchor permissions are invalid")

    unsampled = metadata.get("not_sampled", {})
    interval = [as_decimal(value, "not-sampled interval") for value in unsampled.get("p_open_interval", [])]
    if interval != [Decimal("0.22"), ANCHOR_P]:
        fail("not-sampled interval must be the open interval from p=0.22 to the q=0 anchor")
    if unsampled.get("rendering") != "shaded_only" or unsampled.get("connector_or_schematic_knee") is not False:
        fail("not-sampled interval must be shaded without a connector or schematic knee")

    provenance = metadata.get("analysis_provenance", {})
    forbidden_true = ("fit_performed", "interpolation_performed", "local_rebootstrap_performed")
    if not provenance.get("summary_derived") or any(provenance.get(key) is not False for key in forbidden_true):
        fail("analysis provenance permits an unsupported fit, interpolation, or local bootstrap")

    raw = metadata.get("raw_status", {}).get("exp41_005_p022_per_disorder_npz", {})
    frozen = raw.get("frozen_history", {})
    frozen_path = REPO_ROOT / str(frozen.get("path", ""))
    frozen_available = any(frozen_path.rglob("*.npz")) if frozen_path.is_dir() else False
    if frozen_available != frozen.get("available_in_worktree"):
        fail("exp41/005 frozen raw availability has changed; update metadata before rendering")
    if not frozen_available and frozen.get("status") != "missing_from_frozen_history":
        fail("missing frozen exp41/005 raw must be recorded explicitly")

    recovered = raw.get("recovered_activity_copy", {})
    if recovered.get("status") != "restored_and_validated_for_pilot":
        fail("exp41/005 activity-copy status is not validated")
    if recovered.get("preserved_in_git") is not False:
        fail("recovered raw must not be represented as a tracked figure artifact")
    recovered_hashes: dict[str, str] = {}
    for item in recovered.get("files", []):
        path_text = str(item.get("path", ""))
        expected_hash = str(item.get("sha256", ""))
        if path_text in recovered_hashes or not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
            fail("invalid or duplicate recovered exp41/005 raw record")
        recovered_hashes[path_text] = expected_hash
    if len(recovered_hashes) != 2:
        fail("exactly two recovered exp41/005 raw shards are required")
    validation_path = REPO_ROOT / str(recovered.get("validation", ""))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation_hashes = {
        str(source.get("sha256", ""))
        for source in validation.get("historical_grid", {}).get("sources", [])
    }
    if validation.get("status") != "PASS" or validation_hashes != set(recovered_hashes.values()):
        fail("recovered exp41/005 raw does not match the PASS validation record")

    return rows, metadata


def render(rows: list[dict[str, str]], metadata: dict[str, object]) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.labelsize": 10.5,
            "axes.titlesize": 10.2,
            "legend.fontsize": 8.8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    colors = {"w0": "#146c94", "q_top=q_W": "#c05a16"}
    markers = {"w0": "o", "q_top=q_W": "D"}
    labels = {
        "w0": "w0 L3-L7 crossing (threshold observable)",
        "q_top=q_W": "q_top = q_W L3-L7 (companion only)",
    }

    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    fig.subplots_adjust(left=0.105, right=0.975, top=0.82, bottom=0.285)

    anchor_p = float(metadata["internal_q0_anchor"]["p"])
    band = ax.axvspan(
        0.22,
        anchor_p,
        facecolor="#e4e7eb",
        edgecolor="#a9afb7",
        hatch="////",
        linewidth=0.6,
        zorder=0,
    )

    handles = []
    for observable in ("w0", "q_top=q_W"):
        series = sorted(
            (row for row in rows if row["observable"] == observable),
            key=lambda row: float(row["p"]),
        )
        x = [float(row["p"]) for row in series]
        y = [float(row["q_crossing"]) for row in series]
        low = [value - float(row["ci95_low"]) for value, row in zip(y, series)]
        high = [float(row["ci95_high"]) - value for value, row in zip(y, series)]
        filled = colors[observable] if observable == "w0" else "white"
        handle = ax.errorbar(
            x,
            y,
            yerr=[low, high],
            linestyle="none",
            marker=markers[observable],
            markersize=7.2,
            markerfacecolor=filled,
            markeredgecolor=colors[observable],
            markeredgewidth=1.45,
            color=colors[observable],
            elinewidth=1.35,
            capsize=3.2,
            capthick=1.2,
            label=labels[observable],
            zorder=4 if observable == "w0" else 3,
        )
        handles.append(handle)

    anchor_handle = ax.plot(
        [anchor_p],
        [0.0],
        marker="*",
        markersize=12,
        markerfacecolor="#252a30",
        markeredgecolor="#252a30",
        linestyle="none",
        label="q = 0 internal anchor (finite size)",
        zorder=5,
    )[0]
    handles.extend([anchor_handle, Patch(facecolor="#e4e7eb", edgecolor="#a9afb7", hatch="////", label="not sampled")])

    ax.annotate(
        "q = 0 finite-size\ninternal anchor",
        xy=(anchor_p, 0.0),
        xytext=(0.193, 0.0074),
        arrowprops={"arrowstyle": "-", "color": "#59616a", "linewidth": 0.9},
        color="#30363d",
        ha="left",
        va="center",
        fontsize=8.7,
    )
    ax.text(
        (0.22 + anchor_p) / 2,
        0.016,
        "not sampled",
        rotation=90,
        ha="center",
        va="center",
        fontsize=7.8,
        color="#59616a",
    )

    ax.set_xlim(0.04, 0.2325)
    ax.set_ylim(-0.0032, 0.0475)
    ax.set_xticks([0.05, 0.10, 0.15, 0.20, 0.22])
    ax.set_yticks([0.00, 0.01, 0.02, 0.03, 0.04])
    ax.set_xlabel("Data-error rate p")
    ax.set_ylabel("Finite-size crossing in q")
    ax.grid(axis="both", color="#d7dce1", linewidth=0.65, alpha=0.8, zorder=-1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3.5, width=0.8)

    fig.suptitle(
        "legacy_delta_only: finite-size phase-boundary summary",
        x=0.105,
        y=0.955,
        ha="left",
        fontsize=14,
        fontweight="bold",
        color="#20252b",
    )
    ax.set_title(
        "Summary-derived crossings only; no asymptotic fit or interpolation",
        loc="left",
        pad=11,
        color="#59616a",
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.53, 0.105),
        ncol=2,
        frameon=False,
        handlelength=2.2,
        columnspacing=2.2,
    )
    fig.text(
        0.105,
        0.033,
        "Historical 95% CIs inherited verbatim; not re-bootstrapped locally and not labeled as paired-disorder CIs.\n"
        "All colored points: L3-L7 finite-size crossings, 384 disorders, nboot = 10,000 (exp41/003-006).",
        ha="left",
        va="bottom",
        fontsize=7.7,
        color="#59616a",
        linespacing=1.35,
    )

    pdf_metadata = {
        "Title": "legacy_delta_only finite-size phase-boundary summary",
        "Author": "Project D",
        "Subject": "Summary-derived finite-size crossings from exp41/003-006",
        "Keywords": "legacy_delta_only, w0, q_top, finite-size crossing",
    }
    fig.savefig(PNG_PATH, dpi=400, facecolor="white")
    fig.savefig(PDF_PATH, format="pdf", facecolor="white", metadata=pdf_metadata)
    plt.close(fig)


def main() -> None:
    rows, metadata = load_and_validate()
    render(rows, metadata)
    print(f"validated {len(rows)} frozen records and {len(metadata['source_files'])} source hashes")
    print(f"wrote {PNG_PATH}")
    print(f"wrote {PDF_PATH}")


if __name__ == "__main__":
    main()
