#!/usr/bin/env python3
"""Decision diagnostic for exp38 P1b paired-disorder de-risk."""

from __future__ import annotations

import json
import math
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SUMMARY_JSON = SCRIPT_DIR / "p1_summary.json"
OUTPUT_JSON = SCRIPT_DIR / "p1b_decision_diagnostic.json"
OUTPUT_MD = SCRIPT_DIR / "p1b_decision_diagnostic.md"
TARGET_PAIRED_SEM = 0.05
PLANNED_PRODUCTION_N = 32
PROJECTION_NS = (24, 32, 40)


def _ci_excludes_zero(ci: list[float] | tuple[float, float]) -> bool:
    lo, hi = float(ci[0]), float(ci[1])
    return bool((lo > 0.0 and hi > 0.0) or (lo < 0.0 and hi < 0.0))


def _safe_float(value) -> float:
    return float(value) if value is not None else math.nan


def _project(candidate: dict) -> dict:
    n = int(candidate["num_pairs"])
    paired_sem = _safe_float(candidate["paired_sem"])
    unpaired_sem = _safe_float(candidate["unpaired_difference_sem"])
    paired_sd = paired_sem * math.sqrt(float(n))
    unpaired_sd = unpaired_sem * math.sqrt(float(n))
    n_needed = (
        int(math.ceil((paired_sd / TARGET_PAIRED_SEM) ** 2))
        if math.isfinite(paired_sd) and paired_sd > 0.0
        else math.inf
    )
    return {
        **candidate,
        "paired_sd_delta": paired_sd,
        "unpaired_sd_difference": unpaired_sd,
        "equivalent_sample_gain": (
            1.0 / float(candidate["reduction_ratio"]) ** 2
            if _safe_float(candidate.get("reduction_ratio")) > 0.0
            else math.nan
        ),
        "ci_excludes_zero": _ci_excludes_zero(candidate["paired_bootstrap_ci95"]),
        "n_needed_for_target_paired_sem": n_needed,
        "projected_paired_sem": {
            str(n_target): paired_sd / math.sqrt(float(n_target))
            for n_target in PROJECTION_NS
        },
        "projected_unpaired_sem": {
            str(n_target): unpaired_sd / math.sqrt(float(n_target))
            for n_target in PROJECTION_NS
        },
    }


def main() -> int:
    if not SUMMARY_JSON.exists():
        raise SystemExit(f"missing summary JSON: {SUMMARY_JSON}")
    summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    candidates = [
        _project(candidate)
        for candidate in summary["gates"]["P1b"]["candidates"]
    ]
    strict_candidates = [
        candidate for candidate in candidates
        if (
            candidate["num_pairs"] >= 8
            and candidate["paired_sem"] < candidate["unpaired_difference_sem"]
            and candidate["reduction_ratio"] <= 0.80
            and candidate["paired_sem"] <= TARGET_PAIRED_SEM
        )
    ]
    production_candidates = [
        candidate for candidate in candidates
        if (
            candidate["source"] in {"primary", "p1b_retest"}
            and candidate["num_pairs"] >= 8
            and candidate["paired_sem"] < candidate["unpaired_difference_sem"]
            and candidate["reduction_ratio"] < 1.0
            and candidate["ci_excludes_zero"]
            and candidate["n_needed_for_target_paired_sem"] <= PLANNED_PRODUCTION_N
        )
    ]
    production_candidates.sort(
        key=lambda item: (
            int(item["n_needed_for_target_paired_sem"]),
            float(item["reduction_ratio"]),
            float(item["paired_sem"]),
        )
    )
    selected = production_candidates[0] if production_candidates else None
    coordinate_hash = [
        candidate for candidate in candidates
        if candidate["source"] == "coordinate_hash"
    ]
    payload = {
        "stage": "P1",
        "diagnostic": "p1b_production_power",
        "target_paired_sem": TARGET_PAIRED_SEM,
        "planned_production_n_common": PLANNED_PRODUCTION_N,
        "projection_n_values": list(PROJECTION_NS),
        "strict_pilot_gate_passed": bool(strict_candidates),
        "pass_p1b_under_production_power_criterion": bool(selected),
        "selected_candidate": selected,
        "candidate_table": candidates,
        "coordinate_hash_rejected": bool(
            coordinate_hash and not any(
                item in production_candidates for item in coordinate_hash
            )
        ),
        "criterion": (
            "Accept P1b for production planning if a public same-seed run has "
            "paired SEM < unpaired SEM, reduction ratio < 1, paired bootstrap "
            "CI excluding zero at the pilot q, and projected paired SEM <= "
            f"{TARGET_PAIRED_SEM} by N_common={PLANNED_PRODUCTION_N}."
        ),
        "recommendation": (
            "Use rng_stream with disorder_seed_scope=disorder_index and "
            "N_common=32 for the crossing-region P2 grid; keep coordinate_hash "
            "only as a rejected diagnostic."
            if selected is not None
            else "Keep P1 DOING and redesign the common-disorder strategy."
        ),
    }
    OUTPUT_JSON.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    lines = [
        "# exp38 P1b decision diagnostic",
        "",
        f"Target paired SEM: `{TARGET_PAIRED_SEM}`.",
        f"Planned production N_common: `{PLANNED_PRODUCTION_N}`.",
        "",
        "## Decision",
        "",
        f"Strict pilot gate passed: `{payload['strict_pilot_gate_passed']}`.",
        (
            "Production-power criterion passed: "
            f"`{payload['pass_p1b_under_production_power_criterion']}`."
        ),
    ]
    if selected is not None:
        lines.extend([
            (
                "Selected candidate: "
                f"`{selected['source']}` q={selected['q_value']:.3f}, "
                f"N={selected['num_pairs']}, paired SEM={selected['paired_sem']:.6f}, "
                f"unpaired SEM={selected['unpaired_difference_sem']:.6f}, "
                f"ratio={selected['reduction_ratio']:.3f}, "
                f"N_needed={selected['n_needed_for_target_paired_sem']}."
            ),
            "",
            "Recommendation: use `disorder_seed_scope=disorder_index`, "
            "`disorder_realization_mode=rng_stream`, and crossing-region "
            "`N_common=32` in P2.",
        ])
    else:
        lines.append("No current candidate satisfies the production-power criterion.")
    lines.extend([
        "",
        "## Candidate Table",
        "",
        "| source | q | N | corr | paired SEM | unpaired SEM | ratio | N needed | projected SEM @24 | @32 | @40 | CI excludes 0 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for candidate in candidates:
        projected = candidate["projected_paired_sem"]
        lines.append(
            f"| {candidate['source']} | {candidate['q_value']:.3f} | "
            f"{candidate['num_pairs']} | {candidate['correlation_l3_l5']:.3f} | "
            f"{candidate['paired_sem']:.6f} | "
            f"{candidate['unpaired_difference_sem']:.6f} | "
            f"{candidate['reduction_ratio']:.3f} | "
            f"{candidate['n_needed_for_target_paired_sem']} | "
            f"{projected['24']:.6f} | {projected['32']:.6f} | "
            f"{projected['40']:.6f} | {candidate['ci_excludes_zero']} |"
        )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUTPUT_MD)
    return 0 if payload["pass_p1b_under_production_power_criterion"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
