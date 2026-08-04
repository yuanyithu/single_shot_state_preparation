"""Independent raw-only analysis for the frozen local m8 iid-MIS diagnostic."""

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

from data.expander_code.exp102.exp102_pipeline.io import (
    atomic_json,
    canonical_json,
    sha256_file,
    sha256_json,
)
from data.expander_code.exp102.exp102_pipeline.q0_crossfit_importance import (
    crossfit_collision_ratio,
    crossfit_distribution_distance,
)
from data.expander_code.exp102.exp102_pipeline.q0_iid_importance import (
    IID_IMPORTANCE_VERSION,
    IidMixtureDraws,
    validate_stratified_iid_mixture,
    weight_diagnostics,
)

from local_iid_is_common import (
    LOCAL_CONTRACT_VERSION,
    LOCAL_RAW_FIELDS,
    LocalIidIsConflictError,
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
        raise LocalIidIsConflictError(f"iid-MIS raw scalar {name} has a shape")
    return value.item()


def _load_raw(path, config, config_sha256, registry, model, uniform_seed, syndrome,
              proposal_rows, seeds):
    try:
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != LOCAL_RAW_FIELDS:
                raise LocalIidIsConflictError("iid-MIS raw field set changed")
            metadata_json = str(_scalar(data, "metadata_json"))
            metadata = json.loads(metadata_json)
            if canonical_json(metadata) != metadata_json:
                raise LocalIidIsConflictError("iid-MIS raw metadata is noncanonical")
            expected_metadata = raw_metadata(
                config, config_sha256, registry, model, uniform_seed, syndrome,
                proposal_rows, seeds,
            )
            expected_metadata["iid_importance_version"] = IID_IMPORTANCE_VERSION
            if metadata != expected_metadata:
                raise LocalIidIsConflictError("iid-MIS raw identity changed")
            arrays = {
                name: data[name].copy()
                for name in LOCAL_RAW_FIELDS - {"metadata_json", "content_sha256"}
            }
            if any(value.dtype.hasobject for value in arrays.values()):
                raise LocalIidIsConflictError("iid-MIS raw has an object array")
            if str(_scalar(data, "content_sha256")) != raw_content_sha256(metadata_json, arrays):
                raise LocalIidIsConflictError("iid-MIS raw content SHA changed")
    except LocalIidIsConflictError:
        raise
    except Exception as exc:
        raise LocalIidIsConflictError(f"iid-MIS raw cannot be loaded: {exc}") from exc
    schedule = config["sample_schedule"]
    return IidMixtureDraws(
        **arrays,
        block_count=schedule["block_count"],
        draws_per_proposal_per_block=schedule["draws_per_proposal_per_block"],
        proposal_count=len(config["proposals"]),
    )


def _source_results(draws, config, model):
    results = {}
    blocks = config["sample_schedule"]["block_count"]
    for source, spec in enumerate(config["proposals"]):
        select = draws.source_indices == source
        result = crossfit_collision_ratio(
            draws.labels[select], draws.source_log_importance[select],
            block_count=blocks, logical_dimension=model.k,
        )
        diagnostics = weight_diagnostics(
            draws.source_log_importance[select], block_count=blocks,
        )
        results[spec["id"]] = {
            "collision": result.as_dict(),
            "weights": {name: value.tolist() if isinstance(value, np.ndarray) else value
                        for name, value in diagnostics.items()},
            "role": spec.get("role", "rebuilt_primary_independent_proposal"),
        }
    return results


def _primary_gate(config, source_results, mixture_result, mixture_diagnostics, d2):
    gates = config["gates"]
    primary_a, primary_b = gates["primary_proposal_ids"]
    checks = {}
    for identifier in (primary_a, primary_b):
        values = source_results[identifier]
        checks[f"{identifier}_weight_stability"] = (
            values["weights"]["minimum_block_effective_sample_size"]
            >= gates["min_primary_block_effective_sample_size"]
            and values["weights"]["maximum_block_normalized_weight"]
            <= gates["max_primary_block_normalized_weight"]
        )
    left = source_results[primary_a]["collision"]
    right = source_results[primary_b]["collision"]
    delta = abs(float(left["q_top"]) - float(right["q_top"]))
    delta_se = math.hypot(float(left["q_top_jackknife_se"]),
                          float(right["q_top_jackknife_se"]))
    checks["primary_q_top_agreement"] = (
        delta <= gates["max_abs_primary_q_top_delta"]
        and delta <= gates["agreement_sigma_multiple"] * delta_se + gates["agreement_slack"]
    )
    checks["primary_distribution_agreement"] = (
        max(0.0, float(d2["d2_norm"]))
        + 3.0 * float(d2["d2_norm_jackknife_se"])
        <= gates["max_primary_d2_upper"]
    )
    checks["mixture_weight_stability"] = (
        mixture_diagnostics["minimum_block_effective_sample_size"]
        >= gates["min_mixture_block_effective_sample_size"]
        and mixture_diagnostics["maximum_block_normalized_weight"]
        <= gates["max_mixture_block_normalized_weight"]
    )
    checks["mixture_q_top_precision"] = (
        float(mixture_result["q_top_jackknife_se"])
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
        raise FileExistsError(f"refusing to replace iid-MIS report: {args.output}")
    config, config_sha256 = load_config(args.config)
    _root, registry, _code, _H, model, frame, uniform_seed, syndrome = build_context(config)
    proposal_rows = build_proposals(config, _root, model, frame, syndrome)
    proposals = tuple(row[1] for row in proposal_rows)
    seeds = seed_schedule(config, config_sha256, registry, proposal_rows)
    draws = _load_raw(
        args.raw, config, config_sha256, registry, model, uniform_seed, syndrome,
        proposal_rows, seeds,
    )
    validate_stratified_iid_mixture(
        draws, model, frame, syndrome, config["cell"]["p"], proposals,
        config["proposal_mixture_weights"],
    )
    blocks = config["sample_schedule"]["block_count"]
    mixture_collision = crossfit_collision_ratio(
        draws.labels, draws.mixture_log_importance,
        block_count=blocks, logical_dimension=model.k,
    )
    mixture_diagnostics = weight_diagnostics(draws.mixture_log_importance, block_count=blocks)
    source_results = _source_results(draws, config, model)
    first, second = config["gates"]["primary_proposal_ids"]
    source_index = {value["id"]: index for index, value in enumerate(config["proposals"])}
    d2 = crossfit_distribution_distance(
        draws.labels[draws.source_indices == source_index[first]],
        draws.source_log_importance[draws.source_indices == source_index[first]],
        draws.labels[draws.source_indices == source_index[second]],
        draws.source_log_importance[draws.source_indices == source_index[second]],
        block_count=blocks, logical_dimension=model.k,
    ).as_dict()
    mixture_weight_dict = {
        name: value.tolist() if isinstance(value, np.ndarray) else value
        for name, value in mixture_diagnostics.items()
    }
    checks, primary_comparison = _primary_gate(
        config, source_results, mixture_collision.as_dict(), mixture_weight_dict, d2,
    )
    terminal = (
        "LOCAL_IID_IS_EMPIRICAL_FEASIBILITY_PASS"
        if all(checks.values()) else "LOCAL_IID_IS_EMPIRICAL_FEASIBILITY_UNRESOLVED"
    )
    report = {
        "report_version": "exp102.q0_iid_is.local.report.v0",
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
        "mixture_collision_diagnostic": mixture_collision.as_dict(),
        "mixture_weight_diagnostics": mixture_weight_dict,
        "source_diagnostics": source_results,
        "primary_distribution_distance": d2,
        "primary_comparison": primary_comparison,
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
