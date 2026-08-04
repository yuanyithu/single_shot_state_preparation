"""Raw-only analyzer for the frozen local BP-systematic IID-MIS diagnostic."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from data.expander_code.exp102.exp102_pipeline.io import atomic_json, canonical_json, sha256_file, sha256_json
from data.expander_code.exp102.exp102_pipeline.q0_crossfit_importance import (
    crossfit_collision_ratio,
    crossfit_distribution_distance,
)
from data.expander_code.exp102.exp102_pipeline.q0_iid_importance import weight_diagnostics
from data.expander_code.exp102.exp102_pipeline.q0_iid_provenance import (
    IID_PROVENANCE_VERSION,
    ProvenancedIidMixtureDraws,
    validate_provenanced_stratified_iid_mixture,
)

from bp_iid_common import (
    LOCAL_CONTRACT_VERSION,
    LOCAL_RAW_FIELDS,
    BpIidConflictError,
    build_context,
    build_proposals,
    load_config,
    raw_content_sha256,
    raw_metadata,
    seed_schedule,
)


def _scalar(data, name):
    value = data[name]
    if value.shape != ():
        raise BpIidConflictError(f"BP-systematic IID raw scalar {name} has a shape")
    return value.item()


def _load_raw(path, config, config_sha256, registry, model, uniform_seed, syndrome,
              proposal_rows, seeds):
    try:
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != LOCAL_RAW_FIELDS:
                raise BpIidConflictError("BP-systematic IID raw field set changed")
            metadata_json = str(_scalar(data, "metadata_json"))
            metadata = json.loads(metadata_json)
            if canonical_json(metadata) != metadata_json:
                raise BpIidConflictError("BP-systematic IID raw metadata is noncanonical")
            expected = raw_metadata(
                config, config_sha256, registry, model, uniform_seed, syndrome, proposal_rows, seeds,
            )
            expected["iid_provenance_version"] = IID_PROVENANCE_VERSION
            if metadata != expected:
                raise BpIidConflictError("BP-systematic IID raw identity changed")
            arrays = {name: data[name].copy() for name in LOCAL_RAW_FIELDS - {
                "metadata_json", "content_sha256",
            }}
            if any(value.dtype.hasobject for value in arrays.values()):
                raise BpIidConflictError("BP-systematic IID raw has an object array")
            if str(_scalar(data, "content_sha256")) != raw_content_sha256(metadata_json, arrays):
                raise BpIidConflictError("BP-systematic IID raw content SHA changed")
    except BpIidConflictError:
        raise
    except Exception as exc:
        raise BpIidConflictError(f"BP-systematic IID raw cannot be loaded: {exc}") from exc
    schedule = config["sample_schedule"]
    return ProvenancedIidMixtureDraws(
        **arrays, block_count=schedule["block_count"],
        draws_per_proposal_per_block=schedule["draws_per_proposal_per_block"],
        proposal_count=len(config["proposals"]), coordinate_dimension=832,
    )


def _as_jsonable(values):
    return {name: value.tolist() if isinstance(value, np.ndarray) else value
            for name, value in values.items()}


def _source_results(draws, config, model, proposal_rows):
    result = {}
    blocks = config["sample_schedule"]["block_count"]
    for source, (spec, _proposal, _identity) in enumerate(proposal_rows):
        select = draws.source_indices == source
        collision = crossfit_collision_ratio(
            draws.labels[select], draws.source_log_importance[select],
            block_count=blocks, logical_dimension=model.k,
        )
        result[spec["id"]] = {
            "collision": collision.as_dict(),
            "weights": _as_jsonable(weight_diagnostics(
                draws.source_log_importance[select], block_count=blocks,
            )),
            "role": "primary_bp_systematic" if spec["kind"] == "bp_systematic"
                    else config["gates"]["stress_role"],
        }
    return result


def _component_diagnostics(draws, proposal_rows):
    result = {}
    for source, (spec, proposal, _identity) in enumerate(proposal_rows):
        selected = draws.source_indices == source
        logs = draws.source_log_importance[selected]
        relative = np.exp(logs - float(logs.max()))
        normalized = relative / float(relative.sum(dtype=np.float64))
        components = draws.component_indices[selected]
        counts = []
        for component in range(int(proposal.num_components)):
            component_mask = components == component
            weights = normalized[component_mask]
            counts.append({
                "component_index": component,
                "draw_count": int(component_mask.sum()),
                "normalized_importance_mass": float(weights.sum(dtype=np.float64)),
                "maximum_normalized_importance_weight": (
                    float(weights.max()) if weights.size else 0.0
                ),
            })
        anchors = draws.anchor_indices[selected]
        result[spec["id"]] = {
            "components": counts,
            "anchor_indices": sorted(int(value) for value in np.unique(anchors) if value >= 0),
        }
    return result


def _primary_gates(config, source_results, mixture_collision, mixture_weights, d2,
                   component_diagnostics):
    gates = config["gates"]
    first, second = gates["primary_proposal_ids"]
    checks = {}
    for identifier in (first, second):
        values = source_results[identifier]["weights"]
        checks[f"{identifier}_weight_stability"] = (
            values["minimum_block_effective_sample_size"] >= gates["min_primary_block_effective_sample_size"]
            and values["maximum_block_normalized_weight"] <= gates["max_primary_block_normalized_weight"]
        )
        checks[f"{identifier}_component_coverage"] = all(
            item["draw_count"] >= gates["min_component_draws_per_block_total"]
            for item in component_diagnostics[identifier]["components"]
        )
    left = source_results[first]["collision"]
    right = source_results[second]["collision"]
    delta = abs(float(left["q_top"]) - float(right["q_top"]))
    delta_se = math.hypot(float(left["q_top_jackknife_se"]),
                          float(right["q_top_jackknife_se"]))
    checks["primary_q_top_agreement"] = (
        delta <= gates["max_abs_primary_q_top_delta"]
        and delta <= gates["agreement_sigma_multiple"] * delta_se + gates["agreement_slack"]
    )
    checks["primary_distribution_agreement"] = (
        max(0.0, float(d2["d2_norm"]))
        + 3.0 * float(d2["d2_norm_jackknife_se"]) <= gates["max_primary_d2_upper"]
    )
    checks["mixture_weight_stability"] = (
        mixture_weights["minimum_block_effective_sample_size"] >= gates["min_mixture_block_effective_sample_size"]
        and mixture_weights["maximum_block_normalized_weight"] <= gates["max_mixture_block_normalized_weight"]
    )
    checks["mixture_q_top_precision"] = (
        float(mixture_collision["q_top_jackknife_se"])
        <= gates["max_mixture_q_top_jackknife_se"]
    )
    return checks, {"q_top_delta": delta, "q_top_delta_jackknife_se": delta_se}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to replace BP-systematic IID report: {args.output}")
    config, config_sha256 = load_config(args.config)
    _root, registry, _code, _H, model, frame, uniform_seed, syndrome = build_context(config)
    proposal_rows = build_proposals(config, model, syndrome)
    proposals = tuple(row[1] for row in proposal_rows)
    seeds = seed_schedule(config, config_sha256, registry, proposal_rows)
    draws = _load_raw(
        args.raw, config, config_sha256, registry, model, uniform_seed, syndrome,
        proposal_rows, seeds,
    )
    validate_provenanced_stratified_iid_mixture(
        draws, model, frame, syndrome, config["cell"]["p"], proposals,
        config["proposal_mixture_weights"],
    )
    blocks = config["sample_schedule"]["block_count"]
    mixture_collision = crossfit_collision_ratio(
        draws.labels, draws.mixture_log_importance, block_count=blocks,
        logical_dimension=model.k,
    ).as_dict()
    mixture_weights = _as_jsonable(weight_diagnostics(
        draws.mixture_log_importance, block_count=blocks,
    ))
    source_results = _source_results(draws, config, model, proposal_rows)
    source_index = {spec["id"]: index for index, (spec, _proposal, _identity)
                    in enumerate(proposal_rows)}
    first, second = config["gates"]["primary_proposal_ids"]
    d2 = crossfit_distribution_distance(
        draws.labels[draws.source_indices == source_index[first]],
        draws.source_log_importance[draws.source_indices == source_index[first]],
        draws.labels[draws.source_indices == source_index[second]],
        draws.source_log_importance[draws.source_indices == source_index[second]],
        block_count=blocks, logical_dimension=model.k,
    ).as_dict()
    component_diagnostics = _component_diagnostics(draws, proposal_rows)
    checks, comparison = _primary_gates(
        config, source_results, mixture_collision, mixture_weights, d2,
        component_diagnostics,
    )
    terminal = (
        "LOCAL_BP_SYSTEMATIC_IID_FEASIBILITY_PASS"
        if all(checks.values()) else "LOCAL_BP_SYSTEMATIC_IID_FEASIBILITY_UNRESOLVED"
    )
    report = {
        "report_version": "exp102.q0_bp_systematic_iid.local.report.v0",
        "contract_version": LOCAL_CONTRACT_VERSION,
        "authority": "local_empirical_feasibility_only_not_a_posterior_result",
        "terminal_status": terminal,
        "does_not_establish": [
            "A certified posterior, purity, q_top, or physical parameter-point result.",
            "Coverage of every unobserved target mode or a rigorous tail bound.",
            "Any remote, formal, held-out, or production authorization.",
        ],
        "config_sha256": config_sha256,
        "registry_sha256": registry["registry_sha256"],
        "raw_path": str(args.raw),
        "raw_sha256": sha256_file(args.raw),
        "raw_replay": "PASS",
        "mixture_collision_diagnostic": mixture_collision,
        "mixture_weight_diagnostics": mixture_weights,
        "source_diagnostics": source_results,
        "primary_distribution_distance": d2,
        "primary_comparison": comparison,
        "component_provenance_diagnostics": component_diagnostics,
        "gates": checks,
        "stress_proposal": {
            "id": config["gates"]["stress_proposal_id"],
            "role": config["gates"]["stress_role"],
            "diagnostic": source_results[config["gates"]["stress_proposal_id"]],
        },
    }
    report["report_sha256"] = sha256_json(report)
    atomic_json(args.output, report)


if __name__ == "__main__":
    main()
