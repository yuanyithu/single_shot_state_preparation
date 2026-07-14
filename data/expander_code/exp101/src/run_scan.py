"""Versioned exp101 scan pipeline for the aligned reduced posterior.

Every disorder task is identified by a fingerprint containing the physics and
scan contracts, canonical ensemble, code/frame fingerprints, resolved engine,
and complete sampler/estimator configuration.  Chunks are reusable only when
that fingerprint matches.  The merged primary result is
``q_top_estimate_per_disorder``; invalid tasks remain inspectable but never
enter disorder averages.
"""

import argparse
import hashlib
import json
import math
import platform
import subprocess
import time
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np

from .families import find_family_seed
from .gates import (
    GateThresholds,
    evaluate_convergence_gate,
    evaluate_pt_convergence_gate,
    run_multi_start,
)
from .graphs import (
    complete_bipartite_graph,
    cycle_parity_check_matrix,
    random_biregular_graph_from_m,
    repetition_parity_check_matrix,
)
from .hgp import classical_parity_check_matrix, hgp_from_H
from .logicals import logical_pauli_operators
from .model import (
    ACCEPTED_ENSEMBLES,
    PHYSICS_CONTRACT_VERSION,
    STATE_PREP_PROTOCOL,
    SYNDROME_SEMANTICS,
    assemble_sector_model,
    disorder_from_uniforms,
    normalize_ensemble,
    wire_ensemble,
)
from .observables import (
    aggregate_independent_chain_observables,
    build_observable_frame,
    build_observable_set,
    sampled_nonzero_character_mean,
    sector_weights_from_characters,
)
from .pt import PtConfig, run_parallel_tempering
from .reference_mcmc import ReferenceMcmcConfig
from .sector_ti import (
    FULL_SECTOR_TI_MAX_K,
    SectorTiConfig,
    run_sector_ti,
)

PROTOCOL_VERSION = "exp101.scan.v3"
SCAN_CONTRACT_VERSION = PROTOCOL_VERSION

AGGREGATION_REPORTABLE = "REPORTABLE"
AGGREGATION_SAMPLING_INSUFFICIENT = "SAMPLING_INSUFFICIENT"
AGGREGATION_INCOMPLETE = "INCOMPLETE"
AGGREGATION_FORMAL_ONLY = "FORMAL_ONLY"

AGGREGATION_POLICY = {
    "point_eligibility": "all_planned_disorders_valid",
    "fraction_denominator": "planned_disorders",
    "maximum_invalid_disorders": 0,
    "maximum_missing_disorders": 0,
    "conditional_statistics_purpose": "diagnostics_only",
    "conditional_statistics_are_publication_eligible": False,
    "crossing_input_policy": "whole_point_nan_unless_reportable",
}

ENGINE_FULL_TI = "full_sector_ti"
ENGINE_PT = "parallel_tempering_observable_sampling"
ENGINE_Q0 = "validated_q0_sampling"
ENGINE_DIRECT = "direct_observable_sampling"
REQUESTED_ENGINES = ("auto", "ti", "pt", "direct")

DEFAULT_TI = asdict(SectorTiConfig())
DEFAULT_DIRECT = {
    **asdict(ReferenceMcmcConfig()),
    "num_burn_in_sweeps": 500,
    "num_measurements": 4000,
    "record_observable_trajectory": True,
    "num_starts": 4,
    "gate_thresholds": asdict(GateThresholds()),
}
DEFAULT_Q0 = {**DEFAULT_DIRECT, "num_starts": 8}
DEFAULT_PT = {
    **asdict(PtConfig()),
    "record_observable_trajectory": True,
    "num_instances": 4,
    "gate_thresholds": asdict(GateThresholds()),
}

_EXP101_ROOT = Path(__file__).resolve().parent.parent
_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]


def state_prep_protocol_for_sector(sector):
    """Return the aligned preparation label for a CSS error sector."""
    if sector == "x_error":
        return STATE_PREP_PROTOCOL
    if sector == "z_error":
        return "zero_Xcheck_Z"
    raise ValueError("sector must be x_error|z_error")


# ---------- JSON/fingerprint helpers ----------

def _jsonable(value):
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _canonical_json(value):
    return json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"),
        ensure_ascii=True,
    )


def _fingerprint(value):
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def implementation_fingerprint():
    """Hash the executable exp101 Python sources and physics contract.

    Git commit metadata alone is insufficient when a task is run from a dirty
    tree, and it can incorrectly attribute a reused chunk to a later commit.
    The content hash is therefore part of every task identity.
    """
    paths = sorted((_EXP101_ROOT / "src").glob("*.py"))
    paths.append(_EXP101_ROOT / "PHYSICS_CONTRACT.md")
    digest = hashlib.sha256(b"exp101.implementation.v1\0")
    for path in paths:
        relative = path.relative_to(_EXP101_ROOT).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "little"))
        digest.update(relative)
        payload = path.read_bytes()
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def _atomic_write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(data), handle, indent=1, ensure_ascii=False)
    tmp.replace(path)


def _git_provenance():
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=_REPOSITORY_ROOT, timeout=10, check=False,
        ).stdout.strip() or "unknown"
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            capture_output=True, text=True, cwd=_REPOSITORY_ROOT, timeout=30,
            check=False,
        )
        dirty = bool(status.stdout) if status.returncode == 0 else None
        return {"git_commit_sha": sha, "git_worktree_dirty": dirty}
    except Exception:
        return {"git_commit_sha": "unknown", "git_worktree_dirty": None}


def _versions():
    versions = {"python": platform.python_version(), "numpy": np.__version__}
    try:
        import numba

        versions["numba"] = numba.__version__
    except ImportError:
        versions["numba"] = None
    return versions


# ---------- code family and routing ----------

def build_code(family, size, family_rule="full_rank", family_seed=None):
    """Return ``(H_Z, H_X, logicals, metadata)`` for a registered family."""
    if family == "expander34":
        if size == 1:
            graph = random_biregular_graph_from_m(1, 3, 4, 12345)
            seed_used = 12345
        elif family_seed is not None:
            graph = random_biregular_graph_from_m(size, 3, 4, family_seed)
            seed_used = family_seed
        else:
            seed_used, _, graph, _, _, _ = find_family_seed(size, family_rule)
        classical = classical_parity_check_matrix(graph)
        meta = {
            "family": family, "size": int(size), "seed": int(seed_used),
            "rule": family_rule,
        }
    elif family == "toric":
        classical = cycle_parity_check_matrix(size)
        meta = {"family": family, "size": int(size), "rule": family_rule}
    elif family == "surface":
        classical = repetition_parity_check_matrix(size)
        meta = {"family": family, "size": int(size), "rule": family_rule}
    elif family == "k43":
        classical = classical_parity_check_matrix(complete_bipartite_graph(4, 3))
        meta = {"family": family, "size": 1, "rule": family_rule}
    else:
        raise ValueError(f"unknown family {family}")
    H_Z, H_X = hgp_from_H(classical)
    logicals = logical_pauli_operators(H_X, H_Z)
    meta["classical_sha"] = hashlib.sha256(
        np.ascontiguousarray(classical).tobytes()
    ).hexdigest()
    return H_Z, H_X, logicals, meta


def task_seed(family_fp, sector, ensemble, p, q, disorder_index, stream):
    """Portable seed; deprecated ensemble aliases intentionally collide."""
    canonical_ensemble = normalize_ensemble(ensemble, warn_alias=False)
    payload = (
        f"{PROTOCOL_VERSION}|{family_fp}|{sector}|{canonical_ensemble}"
        f"|p={float(p):.17g}|q={float(q):.17g}"
        f"|dis={int(disorder_index)}|{stream}"
    )
    digest = hashlib.sha256(payload.encode("ascii")).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def resolve_engine(requested_engine, k, q):
    """Resolve the public engine request before any task is launched."""
    requested_engine = str(requested_engine)
    if requested_engine not in REQUESTED_ENGINES:
        raise ValueError(f"engine must be one of {REQUESTED_ENGINES}")
    k, q = int(k), float(q)
    if requested_engine == "auto":
        if k <= FULL_SECTOR_TI_MAX_K:
            return ENGINE_FULL_TI
        return ENGINE_PT if q > 0.0 else ENGINE_Q0
    if requested_engine == "ti":
        if k > FULL_SECTOR_TI_MAX_K:
            raise ValueError(
                "engine=ti is forbidden for k>10; use engine=pt or auto"
            )
        return ENGINE_FULL_TI
    if requested_engine == "pt":
        if q <= 0.0:
            raise ValueError("parallel tempering is only supported for q>0")
        return ENGINE_PT
    return ENGINE_Q0 if q == 0.0 else ENGINE_DIRECT


def _resolved_engine_config(requested, resolved, supplied):
    supplied = dict(supplied or {})
    nested = any(key in supplied for key in ("ti", "direct", "q0", "pt"))
    if nested:
        key = {
            ENGINE_FULL_TI: "ti", ENGINE_PT: "pt",
            ENGINE_Q0: "q0", ENGINE_DIRECT: "direct",
        }[resolved]
        supplied = dict(supplied.get(key, {}))
    defaults = {
        ENGINE_FULL_TI: DEFAULT_TI,
        ENGINE_PT: DEFAULT_PT,
        ENGINE_Q0: DEFAULT_Q0,
        ENGINE_DIRECT: DEFAULT_DIRECT,
    }[resolved]
    merged = {**defaults, **supplied}
    if "gate_thresholds" in defaults:
        merged["gate_thresholds"] = {
            **defaults["gate_thresholds"],
            **dict(supplied.get("gate_thresholds", {})),
        }
    # Gates consume trajectories.  Normalize this forced execution behavior
    # before fingerprinting so persisted configuration equals what ran.
    if resolved in (ENGINE_PT, ENGINE_Q0, ENGINE_DIRECT):
        merged["record_observable_trajectory"] = True
    return merged


def _model_cache_key(family, size, sector, family_rule, family_seed):
    return (
        str(family), int(size), str(sector), str(family_rule),
        None if family_seed is None else int(family_seed),
    )


def _get_model(cache, family, size, sector, family_rule, family_seed):
    key = _model_cache_key(family, size, sector, family_rule, family_seed)
    if key not in cache:
        H_Z, H_X, logicals, meta = build_code(
            family, size, family_rule, family_seed
        )
        model = assemble_sector_model(H_X, H_Z, logicals, sector=sector)
        frame = build_observable_frame(model)
        meta = dict(meta)
        meta.update({
            "family_rule": family_rule,
            "requested_family_seed": family_seed,
            "code_fingerprint": model.fingerprint(),
            "logical_sector_section_fingerprint": (
                model.logical_sector_section.fingerprint()
            ),
            "observable_frame_fingerprint": frame.fingerprint(),
        })
        cache[key] = (model, frame, meta)
    return cache[key]


def _family_scope_fingerprint(meta, sector, family_rule, family_seed):
    return _fingerprint({
        "family": meta,
        "sector": sector,
        "family_rule": family_rule,
        "requested_family_seed": family_seed,
    })


def _task_identity(
    meta, sector, ensemble, p, q, disorder_index, requested_engine,
    resolved_engine, engine_config, u_rand_count, u_set_seed,
    source_implementation_fingerprint,
):
    result = {
        "physics_contract_version": PHYSICS_CONTRACT_VERSION,
        "scan_contract_version": PROTOCOL_VERSION,
        "implementation_fingerprint": source_implementation_fingerprint,
        "state_prep_protocol": state_prep_protocol_for_sector(sector),
        "syndrome_semantics": SYNDROME_SEMANTICS,
        "canonical_ensemble": ensemble,
        "sector": sector,
        "family": meta,
        "p": float(p),
        "q": float(q),
        "disorder_index": int(disorder_index),
        "requested_engine": requested_engine,
        "resolved_engine": resolved_engine,
        "engine_config": engine_config,
        "estimator_config": {
            "num_random_nonbasis_characters": int(u_rand_count),
            "u_set_seed": u_set_seed,
            "minimum_independent_chains": 4,
            "q0_default_independent_chains": 8,
            "squared_moment_estimator": "independent_chain_u_statistic",
            "character_sampling_error": "finite_population_correction",
        },
    }
    return result


def _observable_set_fingerprint(observable_set):
    return hashlib.sha256(
        b"exp101.observable_set.v2\0"
        + np.ascontiguousarray(observable_set.u_bitmasks).tobytes()
        + np.ascontiguousarray(observable_set.W_rows).tobytes()
    ).hexdigest()


def _planted_class_bitmask(wiring):
    return sum(
        1 << bit for bit, value in enumerate(wiring.planted_logical_class)
        if value
    )


def _gate_thresholds_from_config(config):
    raw = config.pop("gate_thresholds", None)
    return GateThresholds(**dict(raw or {}))


def _dataclass_kwargs(cls, config, extra_keys=()):
    allowed = {field.name for field in fields(cls)}
    unknown = set(config) - allowed - set(extra_keys)
    if unknown:
        raise ValueError(
            f"unknown {cls.__name__} configuration keys: {sorted(unknown)}"
        )
    return {key: value for key, value in config.items() if key in allowed}


def _sampled_estimator_result(
    observable_set, relative_chain_means, absolute_chain_means, ensemble,
    planted_class, minimum_chains,
):
    relative_chain_means = np.asarray(relative_chain_means, dtype=np.float64)
    absolute_chain_means = np.asarray(absolute_chain_means, dtype=np.float64)
    estimates = aggregate_independent_chain_observables(
        observable_set, relative_chain_means, character_frame="relative"
    )
    raw_q_top, raw_fpc_se = sampled_nonzero_character_mean(
        observable_set, estimates["m2_u_pooled_square_raw"]
    )
    pooled_relative = estimates["m_u_pooled"]
    pooled_absolute = np.mean(absolute_chain_means, axis=0)
    weights_absolute = None
    weights_relative = None
    if observable_set.tier == "full":
        weights_absolute = sector_weights_from_characters(
            observable_set, pooled_absolute
        )
        weights_relative = sector_weights_from_characters(
            observable_set, pooled_relative
        )

    purity = float(estimates["posterior_purity"])
    minimum_purity = 1.0 / (1 << observable_set.k)
    physical = minimum_purity <= purity <= 1.0
    failures = []
    if relative_chain_means.shape[0] < int(minimum_chains):
        failures.append(f"independent_chain_count<{int(minimum_chains)}")
    if not physical:
        failures.append("debiased_posterior_purity_out_of_range")

    estimated_bounds_valid = physical and not failures
    if ensemble == "true_posterior":
        planted_mass = estimates["posterior_mass_on_planted_class"]
        # Character inversion of finite MCMC means is not an exact sector
        # posterior and therefore cannot be labelled paper MLD success.
        map_success = None
        lower = purity if estimated_bounds_valid else None
        upper = float(np.sqrt(purity)) if estimated_bounds_valid else None
        posterior_purity = purity
    else:
        planted_mass = None
        map_success = None
        lower = None
        upper = None
        posterior_purity = None

    result = {
        "character_means_relative": pooled_relative,
        "character_means_absolute": pooled_absolute,
        "chain_character_means_relative": relative_chain_means,
        "chain_character_means_absolute": absolute_chain_means,
        "m2_u_pooled_square_raw": estimates["m2_u_pooled_square_raw"],
        "m2_u_debiased": estimates["m2_u_debiased"],
        "m2_u_debiased_jackknife_se": (
            estimates["m2_u_debiased_jackknife_se"]
        ),
        "q_top_estimate": float(estimates["q_top"]),
        "q_top_absolute": float(estimates["q_top"]),
        "q_top_relative": float(estimates["q_top"]),
        "q_top_estimator_name": "independent_chain_u_statistic",
        "q_top_raw_pooled_square": float(raw_q_top),
        "q_top_chain_jackknife_se": float(
            estimates["q_top_chain_jackknife_se"]
        ),
        "q_top_character_sampling_se": float(
            estimates["q_top_character_sampling_se"]
        ),
        "q_top_raw_character_sampling_se": float(raw_fpc_se),
        "formal_sector_purity": purity,
        "posterior_purity": posterior_purity,
        "posterior_mass_on_planted_class": planted_mass,
        "posterior_mass_character_sampling_se": (
            estimates["posterior_mass_character_sampling_se"]
            if ensemble == "true_posterior" else None
        ),
        "map_success_probability": map_success,
        "map_success_algebraic_lower_bound": None,
        "map_success_algebraic_upper_bound": None,
        "map_success_estimated_lower_bound": lower,
        "map_success_estimated_upper_bound": upper,
        "map_success_bound_kind": (
            "sampled_u_statistic_plugin_no_coverage"
            if estimated_bounds_valid and ensemble == "true_posterior"
            else "unavailable"
        ),
        "map_success_bound_has_confidence_coverage": False,
        "largest_sector_mass": (
            float(np.max(weights_absolute))
            if weights_absolute is not None else None
        ),
        "weights_absolute": weights_absolute,
        "weights_relative": weights_relative,
        "weights_estimator_name": (
            "character_inversion_of_sample_means"
            if weights_absolute is not None else None
        ),
        "weights_are_exact_sector_posterior": False,
        "estimator_failure_reasons": failures,
        "debiased_purity_in_physical_range": physical,
        "num_independent_chains": int(relative_chain_means.shape[0]),
        "planted_logical_class_bitmask": int(planted_class),
    }
    if ensemble == "true_posterior":
        return result

    result.update({
        "formal_sector_characters_relative": pooled_relative,
        "formal_sector_characters_absolute": pooled_absolute,
        "formal_chain_sector_characters_relative": relative_chain_means,
        "formal_chain_sector_characters_absolute": absolute_chain_means,
        "formal_sector_character_m2_pooled_square_raw": (
            estimates["m2_u_pooled_square_raw"]
        ),
        "formal_sector_character_m2_debiased": estimates["m2_u_debiased"],
        "formal_sector_character_m2_debiased_jackknife_se": (
            estimates["m2_u_debiased_jackknife_se"]
        ),
        "formal_q_top": float(estimates["q_top"]),
        "formal_q_top_absolute": float(estimates["q_top"]),
        "formal_q_top_relative": float(estimates["q_top"]),
        "formal_q_top_estimator_name": "independent_chain_u_statistic",
        "formal_q_top_raw_pooled_square": float(raw_q_top),
        "formal_q_top_chain_jackknife_se": float(
            estimates["q_top_chain_jackknife_se"]
        ),
        "formal_q_top_character_sampling_se": float(
            estimates["q_top_character_sampling_se"]
        ),
        "formal_q_top_raw_character_sampling_se": float(raw_fpc_se),
        "formal_sector_weights_absolute": weights_absolute,
        "formal_sector_weights_relative": weights_relative,
        "formal_weights_estimator_name": (
            "character_inversion_of_sample_means"
            if weights_absolute is not None else None
        ),
        "formal_weights_cover_all_sectors": observable_set.tier == "full",
        "formal_weights_are_exact_sector_posterior": False,
    })
    for name in (
        "character_means_relative", "character_means_absolute",
        "chain_character_means_relative", "chain_character_means_absolute",
        "m2_u_pooled_square_raw", "m2_u_debiased",
        "m2_u_debiased_jackknife_se", "q_top_estimate", "q_top_absolute",
        "q_top_relative", "q_top_estimator_name",
        "q_top_raw_pooled_square", "q_top_chain_jackknife_se",
        "q_top_character_sampling_se", "q_top_raw_character_sampling_se",
        "posterior_purity", "posterior_mass_on_planted_class",
        "posterior_mass_character_sampling_se", "map_success_probability",
        "map_success_algebraic_lower_bound",
        "map_success_algebraic_upper_bound",
        "map_success_estimated_lower_bound",
        "map_success_estimated_upper_bound",
        "weights_absolute", "weights_relative", "weights_estimator_name",
    ):
        result[name] = None
    result["map_success_bound_kind"] = "unavailable"
    result["map_success_bound_has_confidence_coverage"] = False
    result["weights_cover_all_sectors"] = False
    return result


# ---------- task execution ----------

def run_single_task(
    models_cache, family, size, sector, ensemble, p, q, disorder_index,
    engine, engine_config, family_rule, family_seed, u_rand_count,
    task_fingerprint=None, execution_provenance=None,
):
    """Run one deterministic disorder task and return a JSON-safe mapping."""
    ensemble = normalize_ensemble(ensemble, warn_alias=False)
    model, frame, meta = _get_model(
        models_cache, family, size, sector, family_rule, family_seed
    )
    requested_engine = str(engine)
    resolved_engine = resolve_engine(requested_engine, model.k, q)
    engine_config = _resolved_engine_config(
        requested_engine, resolved_engine, engine_config
    )
    family_fp = _family_scope_fingerprint(
        meta, sector, family_rule, family_seed
    )
    disorder_seed = task_seed(
        family_fp, sector, ensemble, p, q, disorder_index, "disorder"
    )
    u_set_seed = (
        task_seed(family_fp, sector, ensemble, p, q, disorder_index, "u_set")
        if model.k > FULL_SECTOR_TI_MAX_K else None
    )
    source_implementation_fingerprint = implementation_fingerprint()
    identity = _task_identity(
        meta, sector, ensemble, p, q, disorder_index, requested_engine,
        resolved_engine, engine_config, u_rand_count, u_set_seed,
        source_implementation_fingerprint,
    )
    computed_fingerprint = _fingerprint(identity)
    if task_fingerprint is not None and task_fingerprint != computed_fingerprint:
        raise ValueError("task fingerprint changed between dispatch and worker")
    task_fingerprint = computed_fingerprint
    engine_seed = task_seed(
        family_fp, sector, ensemble, p, q, disorder_index,
        f"engine:{resolved_engine}:{task_fingerprint}",
    )

    rng_disorder = np.random.default_rng(disorder_seed)
    disorder = disorder_from_uniforms(
        model, p, q,
        data_uniforms=rng_disorder.random(model.num_qubits),
        syndrome_uniforms=rng_disorder.random(model.num_checks),
    )
    wiring = wire_ensemble(model, disorder, ensemble, frame)
    planted_class = _planted_class_bitmask(wiring)
    git_provenance = dict(execution_provenance or _git_provenance())
    started = time.perf_counter()
    result = {
        "physics_contract_version": PHYSICS_CONTRACT_VERSION,
        "scan_contract_version": PROTOCOL_VERSION,
        "implementation_fingerprint": source_implementation_fingerprint,
        **git_provenance,
        "state_prep_protocol": state_prep_protocol_for_sector(sector),
        "syndrome_semantics": SYNDROME_SEMANTICS,
        "task_fingerprint": task_fingerprint,
        "family": meta,
        "sector": sector,
        "canonical_ensemble": ensemble,
        "ensemble": ensemble,
        "p": float(p),
        "q": float(q),
        "disorder_index": int(disorder_index),
        "k": int(model.k),
        "n": int(model.num_qubits),
        "epsilon_data_weight": int(disorder.epsilon_data_weight),
        "measurement_error_weight": int(disorder.measurement_error_weight),
        "disorder_seed": int(disorder_seed),
        "engine_seed": int(engine_seed),
        "requested_engine": requested_engine,
        "resolved_engine": resolved_engine,
        "resolved_engine_config": engine_config,
        "code_fingerprint": meta["code_fingerprint"],
        "section_fingerprint": meta["logical_sector_section_fingerprint"],
        "observable_frame_fingerprint": meta["observable_frame_fingerprint"],
        "planted_logical_class_bitmask": int(planted_class),
        "failure_reasons": [],
    }

    if resolved_engine == ENGINE_FULL_TI:
        config = SectorTiConfig(**_dataclass_kwargs(SectorTiConfig, engine_config))
        ti = run_sector_ti(model, frame, wiring, config, seed=engine_seed)
        if ensemble == "true_posterior":
            characters_absolute = np.asarray(ti["characters_absolute"])
            characters_relative = np.asarray(ti["characters_relative"])
        else:
            characters_absolute = np.asarray(
                ti["formal_sector_characters_absolute"]
            )
            characters_relative = np.asarray(
                ti["formal_sector_characters_relative"]
            )
        result.update({
            **ti,
            "character_count": int(characters_absolute.size),
            "u_bitmasks": np.arange(
                1, 1 << model.k, dtype=np.int64
            ),
            "u_rand_seed": None,
            "observable_set_fingerprint": _fingerprint({
                "frame": frame.fingerprint(), "tier": "full",
                "k": model.k,
            }),
        })
        if ensemble == "true_posterior":
            result.update({
                "character_means_absolute": characters_absolute,
                "character_means_relative": characters_relative,
                "m2_u_pooled_square_raw": characters_relative**2,
                "m2_u_debiased": characters_relative**2,
                "m2_u_debiased_jackknife_se": np.full(
                    characters_relative.shape, np.nan
                ),
                "q_top_estimate": float(ti["q_top"]),
                "q_top_absolute": float(ti["q_top"]),
                "q_top_relative": float(ti["q_top"]),
                "q_top_estimator_name": ti["q_top_estimator_name"],
                "q_top_chain_jackknife_se": None,
                "q_top_character_sampling_se": 0.0,
                "q_top_raw_pooled_square": float(ti["q_top"]),
                "weights_estimator_name": "full_sector_ti",
                "weights_cover_all_sectors": True,
                "weights_are_exact_sector_posterior": bool(ti.get(
                    "weights_are_exact_sector_posterior", False
                )),
            })
        else:
            result.update({
                "formal_sector_characters_absolute": characters_absolute,
                "formal_sector_characters_relative": characters_relative,
                "formal_sector_character_m2_pooled_square_raw": (
                    characters_relative**2
                ),
                "formal_sector_character_m2_debiased": (
                    characters_relative**2
                ),
                "formal_sector_character_m2_debiased_jackknife_se": np.full(
                    characters_relative.shape, np.nan
                ),
                "formal_q_top_raw_pooled_square": ti["formal_q_top"],
                "formal_q_top_chain_jackknife_se": None,
                "formal_q_top_character_sampling_se": 0.0,
                "formal_q_top_raw_character_sampling_se": 0.0,
                "formal_weights_estimator_name": "full_sector_ti",
                "formal_weights_cover_all_sectors": True,
                "formal_weights_are_exact_sector_posterior": bool(ti.get(
                    "formal_weights_are_exact_sector_posterior", False
                )),
                "character_means_absolute": None,
                "character_means_relative": None,
                "m2_u_pooled_square_raw": None,
                "m2_u_debiased": None,
                "m2_u_debiased_jackknife_se": None,
                "q_top_estimate": None,
                "q_top_absolute": None,
                "q_top_relative": None,
                "q_top_estimator_name": None,
                "q_top_chain_jackknife_se": None,
                "q_top_character_sampling_se": None,
                "q_top_raw_pooled_square": None,
                "weights_estimator_name": None,
                "weights_cover_all_sectors": False,
                "weights_are_exact_sector_posterior": False,
            })
        if not ti["valid_for_aggregation"]:
            result["failure_reasons"].extend(
                ti["flags"].split(";")
            )
    else:
        observable_set = build_observable_set(
            frame,
            u_rand_seed=u_set_seed,
            num_random_u=u_rand_count,
        )
        result.update({
            "character_count": int(observable_set.num_u),
            "u_bitmasks": observable_set.u_bitmasks,
            "u_rand_seed": observable_set.u_rand_seed,
            "observable_set_fingerprint": _observable_set_fingerprint(
                observable_set
            ),
            "observable_tier": observable_set.tier,
            "weights_cover_all_sectors": observable_set.tier == "full",
        })
        config_dict = dict(engine_config)
        thresholds = _gate_thresholds_from_config(config_dict)

        if resolved_engine == ENGINE_PT:
            num_instances = int(config_dict.pop("num_instances", 4))
            pt_config = PtConfig(**_dataclass_kwargs(PtConfig, config_dict))
            pt_results = []
            instance_seeds = []
            for instance in range(num_instances):
                instance_seed = task_seed(
                    family_fp, sector, ensemble, p, q, disorder_index,
                    f"pt_instance:{instance}:{task_fingerprint}",
                )
                instance_seeds.append(instance_seed)
                pt_results.append(run_parallel_tempering(
                    model, frame, observable_set, wiring, pt_config,
                    seed=instance_seed,
                ))
            gate = evaluate_pt_convergence_gate(
                pt_results, observable_set, thresholds=thresholds,
                min_instances=4,
            )
            relative_means = np.stack([
                item.m_u_cold_relative for item in pt_results
            ])
            absolute_means = np.stack([
                item.m_u_cold_absolute for item in pt_results
            ])
            result.update(_sampled_estimator_result(
                observable_set, relative_means, absolute_means, ensemble,
                planted_class, minimum_chains=4,
            ))
            result.update({
                "pt_instance_seeds": instance_seeds,
                "pt_ladder_p": pt_results[0].ladder_p,
                "pt_ladder_q": pt_results[0].ladder_q,
                "pt_swap_attempts_per_instance": np.stack([
                    item.swap_attempts for item in pt_results
                ]),
                "pt_swap_accepts_per_instance": np.stack([
                    item.swap_accepts for item in pt_results
                ]),
                "pt_swap_rates_per_instance": np.stack([
                    item.swap_rates() for item in pt_results
                ]),
                "pt_round_trips_per_instance": np.asarray([
                    item.round_trips for item in pt_results
                ], dtype=np.int64),
                "pt_burn_in_round_trips_per_instance": np.asarray([
                    item.burn_in_round_trips
                    if item.burn_in_round_trips is not None else -1
                    for item in pt_results
                ], dtype=np.int64),
                "pt_measurement_round_trips_per_instance": np.asarray([
                    item.measurement_round_trips
                    if item.measurement_round_trips is not None
                    else item.round_trips
                    for item in pt_results
                ], dtype=np.int64),
                "pt_cold_logical_acceptance_per_instance": np.stack([
                    item.cold_logical_acceptance_per_u()
                    for item in pt_results
                ]),
            })
        else:
            num_starts = int(config_dict.pop(
                "num_starts", 8 if resolved_engine == ENGINE_Q0 else 4
            ))
            config_dict.pop("record_observable_trajectory", None)
            mcmc_config = ReferenceMcmcConfig(
                record_observable_trajectory=True,
                **_dataclass_kwargs(ReferenceMcmcConfig, config_dict),
            )
            starts = run_multi_start(
                model, frame, observable_set, wiring, mcmc_config,
                base_seed=engine_seed, num_starts=num_starts,
            )
            gate = evaluate_convergence_gate(
                starts, thresholds=thresholds
            )
            relative_means = np.stack([
                item["m_u_relative"] for item in starts
            ])
            absolute_means = np.stack([
                item["m_u_absolute"] for item in starts
            ])
            minimum_chains = 8 if resolved_engine == ENGINE_Q0 else 4
            result.update(_sampled_estimator_result(
                observable_set, relative_means, absolute_means, ensemble,
                planted_class, minimum_chains=minimum_chains,
            ))
            result.update({
                "start_sector_bitmasks": [
                    item["sector_bitmask"] for item in starts
                ],
                "logical_acceptance_per_chain": np.stack([
                    item["acceptance"]["logical_per_u"] for item in starts
                ]),
            })

        result["gate_passed"] = bool(gate.passed)
        result["gate_failed_checks"] = list(gate.failed_checks)
        result["gate_metrics"] = gate.metrics
        result["gate_thresholds"] = asdict(gate.thresholds)
        result["gate_notes"] = gate.notes
        if not gate.passed:
            result["failure_reasons"].extend(gate.failed_checks)
        result["failure_reasons"].extend(
            result.pop("estimator_failure_reasons")
        )

    result["failure_reasons"] = list(dict.fromkeys(result["failure_reasons"]))
    result["numerically_valid"] = not result["failure_reasons"]
    result["formal_only"] = bool(
        ensemble == "legacy_delta_only" and result["numerically_valid"]
    )
    result["valid_for_aggregation"] = bool(
        ensemble == "true_posterior" and result["numerically_valid"]
    )
    if not result["valid_for_aggregation"]:
        result["map_success_algebraic_lower_bound"] = None
        result["map_success_algebraic_upper_bound"] = None
        result["map_success_estimated_lower_bound"] = None
        result["map_success_estimated_upper_bound"] = None
        result["map_success_bound_kind"] = "unavailable"
        result["map_success_bound_has_confidence_coverage"] = False
    if result["valid_for_aggregation"]:
        result["task_status"] = "VALID"
        result["flags"] = "PASS"
    elif result["formal_only"]:
        result["task_status"] = "FORMAL_ONLY"
        result["flags"] = "FORMAL_ONLY:legacy_delta_only"
    else:
        result["task_status"] = "INVALID"
        result["flags"] = "INVALID:" + ";".join(result["failure_reasons"])
    result["wall_time_seconds"] = time.perf_counter() - started
    return _jsonable(result)


_WORKER_CACHE = {}


def _chunk_matches(path, task_fingerprint, source_implementation_fingerprint):
    try:
        with Path(path).open(encoding="utf-8") as handle:
            payload = json.load(handle)
        return (
            payload.get("protocol") == PROTOCOL_VERSION
            and payload.get("task_fingerprint") == task_fingerprint
            and payload.get("result", {}).get("task_fingerprint")
            == task_fingerprint
            and payload.get("implementation_fingerprint")
            == source_implementation_fingerprint
            and payload.get("result", {}).get("implementation_fingerprint")
            == source_implementation_fingerprint
            and payload.get("result", {}).get("scan_contract_version")
            == PROTOCOL_VERSION
            and payload.get("result", {}).get("physics_contract_version")
            == PHYSICS_CONTRACT_VERSION
        )
    except (OSError, json.JSONDecodeError, TypeError, AttributeError):
        return False


def _chunk_worker(spec):
    chunk_path = Path(spec["chunk_path"])
    if (
        not spec.get("force_recompute", False)
        and _chunk_matches(
            chunk_path, spec["task_fingerprint"],
            spec["implementation_fingerprint"],
        )
    ):
        return spec["tag"], "reused", 0.0
    try:
        result = run_single_task(
            _WORKER_CACHE,
            spec["family"], spec["size"], spec["sector"], spec["ensemble"],
            spec["p"], spec["q"], spec["disorder_index"], spec["engine"],
            spec["engine_config"], spec["family_rule"], spec["family_seed"],
            spec["u_rand_count"], task_fingerprint=spec["task_fingerprint"],
            execution_provenance=spec["execution_provenance"],
        )
        _atomic_write_json(chunk_path, {
            "protocol": PROTOCOL_VERSION,
            "task_fingerprint": spec["task_fingerprint"],
            "implementation_fingerprint": spec["implementation_fingerprint"],
            "result": result,
        })
        return spec["tag"], "computed", result["wall_time_seconds"]
    except Exception as error:  # one bad cell must not erase other evidence
        return spec["tag"], "failed", repr(error)


def _build_specs(
    output_dir, family, size_list, p_value, q_values, num_disorders, sector,
    ensemble, engine, engine_config, family_rule, family_seed, u_rand_count,
):
    if int(num_disorders) <= 0:
        raise ValueError("num_disorders must be positive")
    cache = {}
    specs = []
    source_implementation_fingerprint = implementation_fingerprint()
    execution_provenance = _git_provenance()
    for size in size_list:
        model, frame, meta = _get_model(
            cache, family, size, sector, family_rule, family_seed
        )
        family_fp = _family_scope_fingerprint(
            meta, sector, family_rule, family_seed
        )
        for q in map(float, q_values):
            resolved = resolve_engine(engine, model.k, q)
            config = _resolved_engine_config(engine, resolved, engine_config)
            for disorder_index in range(int(num_disorders)):
                u_set_seed = (
                    task_seed(
                        family_fp, sector, ensemble, p_value, q,
                        disorder_index, "u_set",
                    )
                    if model.k > FULL_SECTOR_TI_MAX_K else None
                )
                identity = _task_identity(
                    meta, sector, ensemble, p_value, q, disorder_index,
                    engine, resolved, config, u_rand_count, u_set_seed,
                    source_implementation_fingerprint,
                )
                fingerprint = _fingerprint(identity)
                q_bits = int(np.asarray(q, dtype=np.float64).view(np.uint64))
                tag = (
                    f"m{int(size)}_qbits{q_bits:016x}_d{disorder_index}_"
                    f"{fingerprint[:16]}"
                )
                specs.append({
                    "tag": tag,
                    "chunk_path": str(
                        Path(output_dir) / "chunks" / f"task_{tag}.json"
                    ),
                    "task_fingerprint": fingerprint,
                    "implementation_fingerprint": (
                        source_implementation_fingerprint
                    ),
                    "execution_provenance": execution_provenance,
                    "family": family,
                    "size": int(size),
                    "sector": sector,
                    "ensemble": ensemble,
                    "p": float(p_value),
                    "q": q,
                    "disorder_index": disorder_index,
                    "engine": engine,
                    "resolved_engine": resolved,
                    "engine_config": config,
                    "family_rule": family_rule,
                    "family_seed": family_seed,
                    "u_rand_count": int(u_rand_count),
                })
    return specs


def scan(
    output_dir, family, size_list, p_value, q_values, num_disorders,
    sector="x_error", ensemble="true_posterior", engine="auto",
    engine_config=None, family_rule="full_rank", family_seed=None,
    u_rand_count=64, num_workers=None, force_recompute=False,
):
    """Run/reuse atomic chunks and merge them into ``scan_results.npz``."""
    output_dir = Path(output_dir)
    canonical_ensemble = normalize_ensemble(ensemble)
    specs = _build_specs(
        output_dir, family, size_list, p_value, q_values, num_disorders,
        sector, canonical_ensemble, engine, engine_config, family_rule,
        family_seed, u_rand_count,
    )
    (output_dir / "chunks").mkdir(parents=True, exist_ok=True)
    pending = []
    for spec in specs:
        matches = _chunk_matches(
            spec["chunk_path"], spec["task_fingerprint"],
            spec["implementation_fingerprint"],
        )
        if force_recompute or not matches:
            pending.append({**spec, "force_recompute": bool(force_recompute)})
    reused = len(specs) - len(pending)
    computed = 0
    failed = []
    if num_workers and int(num_workers) > 1 and pending:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(
            max_workers=int(num_workers), mp_context=mp.get_context("spawn")
        ) as pool:
            for tag, status, info in pool.map(_chunk_worker, pending):
                if status == "computed":
                    computed += 1
                elif status == "failed":
                    failed.append((tag, info))
    else:
        for spec in pending:
            tag, status, info = _chunk_worker(spec)
            if status == "computed":
                computed += 1
            elif status == "failed":
                failed.append((tag, info))

    npz_path = merge(
        output_dir, family, size_list, p_value, q_values, num_disorders,
        sector, canonical_ensemble, engine, engine_config, family_rule,
        family_seed=family_seed, u_rand_count=u_rand_count,
        expected_specs=specs,
    )
    return npz_path, {
        "reused": reused, "computed": computed, "failed": failed,
        "total": len(specs), "num_workers": int(num_workers or 1),
    }


# ---------- merge ----------

def _safe_float(value, default=np.nan):
    return default if value is None else float(value)


def merge(
    output_dir, family, size_list, p_value, q_values, num_disorders, sector,
    ensemble, engine, engine_config, family_rule, family_seed=None,
    u_rand_count=64, expected_specs=None,
):
    """Merge current-v3 chunks with parameter-point fail-closed outputs."""
    output_dir = Path(output_dir)
    ensemble = normalize_ensemble(ensemble, warn_alias=False)
    if expected_specs is None:
        expected_specs = _build_specs(
            output_dir, family, size_list, p_value, q_values, num_disorders,
            sector, ensemble, engine, engine_config, family_rule, family_seed,
            u_rand_count,
        )
    spec_by_key = {
        (spec["size"], float(spec["q"]), spec["disorder_index"]): spec
        for spec in expected_specs
    }
    num_m, num_q, num_d = len(size_list), len(q_values), int(num_disorders)

    def load(size, q, disorder_index):
        spec = spec_by_key[(int(size), float(q), int(disorder_index))]
        if not _chunk_matches(
            spec["chunk_path"], spec["task_fingerprint"],
            spec["implementation_fingerprint"],
        ):
            return None
        try:
            with Path(spec["chunk_path"]).open(encoding="utf-8") as handle:
                return json.load(handle)["result"]
        except (OSError, json.JSONDecodeError, KeyError):
            return None

    all_results = [
        [[load(size, float(q), d) for d in range(num_d)] for q in q_values]
        for size in size_list
    ]
    present = [
        item for per_m in all_results for per_q in per_m
        for item in per_q if item is not None
    ]
    shape = (num_m, num_q, num_d)
    max_characters = max(
        (int(item.get("character_count", 0)) for item in present), default=0
    )
    max_chains = max(
        (len(
            item.get("chain_character_means_relative")
            or item.get("formal_chain_sector_characters_relative")
            or []
        ) for item in present),
        default=0,
    )
    max_weights = max(
        (len(
            item.get("weights_absolute")
            or item.get("formal_sector_weights_absolute")
            or []
        ) for item in present),
        default=0,
    )
    max_pt_instances = max(
        (len(item.get("pt_round_trips_per_instance") or [])
         for item in present),
        default=0,
    )
    max_pt_temperatures = max(
        (len(item.get("pt_ladder_p") or []) for item in present), default=0
    )
    max_pt_edges = max(max_pt_temperatures - 1, 0)
    max_pt_logicals = max(
        (int(item.get("k", 0)) for item in present), default=0
    )

    def scalar(getter, default=np.nan, dtype=np.float64):
        array = np.full(shape, default, dtype=dtype)
        for i in range(num_m):
            for j in range(num_q):
                for d in range(num_d):
                    item = all_results[i][j][d]
                    if item is None:
                        continue
                    value = getter(item)
                    if value is not None:
                        array[i, j, d] = value
        return array

    def padded(key, width, dtype=np.float64, fill=np.nan):
        array = np.full((*shape, max(width, 1)), fill, dtype=dtype)
        for i in range(num_m):
            for j in range(num_q):
                for d in range(num_d):
                    item = all_results[i][j][d]
                    values = None if item is None else item.get(key)
                    if values is None:
                        continue
                    values = np.asarray(values, dtype=dtype)
                    array[i, j, d, :values.size] = values
        return array

    character_fields = {}
    for key in (
        "u_bitmasks", "character_means_absolute", "character_means_relative",
        "m2_u_pooled_square_raw", "m2_u_debiased",
        "m2_u_debiased_jackknife_se",
    ):
        dtype = np.int64 if key == "u_bitmasks" else np.float64
        fill = -1 if key == "u_bitmasks" else np.nan
        character_fields[f"{key}_per_disorder"] = padded(
            key, max_characters, dtype=dtype, fill=fill
        )
    formal_character_fields = {}
    for key in (
        "formal_sector_characters_absolute",
        "formal_sector_characters_relative",
        "formal_sector_character_m2_pooled_square_raw",
        "formal_sector_character_m2_debiased",
        "formal_sector_character_m2_debiased_jackknife_se",
    ):
        formal_character_fields[f"{key}_per_disorder"] = padded(
            key, max_characters
        )

    chain_relative = np.full(
        (*shape, max(max_chains, 1), max(max_characters, 1)), np.nan
    )
    chain_absolute = np.full_like(chain_relative, np.nan)
    formal_chain_relative = np.full_like(chain_relative, np.nan)
    formal_chain_absolute = np.full_like(chain_relative, np.nan)
    for i in range(num_m):
        for j in range(num_q):
            for d in range(num_d):
                item = all_results[i][j][d]
                if item is None:
                    continue
                for key, destination in (
                    ("chain_character_means_relative", chain_relative),
                    ("chain_character_means_absolute", chain_absolute),
                    (
                        "formal_chain_sector_characters_relative",
                        formal_chain_relative,
                    ),
                    (
                        "formal_chain_sector_characters_absolute",
                        formal_chain_absolute,
                    ),
                ):
                    values = item.get(key)
                    if values is None:
                        continue
                    values = np.asarray(values, dtype=np.float64)
                    destination[i, j, d, :values.shape[0], :values.shape[1]] = values

    weights_absolute = padded("weights_absolute", max_weights)
    weights_relative = padded("weights_relative", max_weights)
    formal_weights_absolute = padded(
        "formal_sector_weights_absolute", max_weights
    )
    formal_weights_relative = padded(
        "formal_sector_weights_relative", max_weights
    )
    delta_f = padded("delta_f", max_weights)
    delta_f_infinite_mask = padded(
        "delta_f_infinite_mask", max_weights, dtype=bool, fill=False
    )
    delta_f_stderr = padded("delta_f_stderr", max_weights)
    character_count = scalar(
        lambda item: item.get("character_count", 0), default=0, dtype=np.int64
    )
    independent_chain_count = scalar(
        lambda item: item.get("num_independent_chains", 0),
        default=0, dtype=np.int64,
    )
    character_mask = np.arange(max(max_characters, 1))[None, None, None, :] \
        < character_count[..., None]
    independent_chain_mask = (
        np.arange(max(max_chains, 1))[None, None, None, :]
        < independent_chain_count[..., None]
    )

    q_top = scalar(lambda item: item.get("q_top_estimate"))
    formal_q_top = scalar(lambda item: item.get("formal_q_top"))
    q_top_stderr = scalar(lambda item: (
        item.get("q_top_stderr")
        if item.get("q_top_stderr") is not None
        else item.get("q_top_chain_jackknife_se")
    ))
    valid = scalar(
        lambda item: bool(item.get("valid_for_aggregation", False)),
        default=False, dtype=bool,
    )
    numerical_valid = scalar(
        lambda item: bool(item.get(
            "numerically_valid", item.get("valid_for_aggregation", False)
        )),
        default=False, dtype=bool,
    )
    formal_only = scalar(
        lambda item: bool(item.get("formal_only", False)),
        default=False, dtype=bool,
    )
    present_mask = np.asarray([
        [[all_results[i][j][d] is not None for d in range(num_d)]
         for j in range(num_q)] for i in range(num_m)
    ], dtype=bool)
    valid_count = valid.sum(axis=2).astype(np.int64)
    numerical_valid_count = numerical_valid.sum(axis=2).astype(np.int64)
    formal_only_count = formal_only.sum(axis=2).astype(np.int64)
    invalid_count = (
        present_mask & ~valid & ~formal_only
    ).sum(axis=2).astype(np.int64)
    missing_count = (~present_mask).sum(axis=2).astype(np.int64)
    present_count = present_mask.sum(axis=2).astype(np.int64)
    planned_count = np.full(
        (num_m, num_q), num_d, dtype=np.int64
    )
    paper_aggregation_fraction = valid_count / planned_count
    numerical_pass_fraction = numerical_valid_count / planned_count
    conditional_mean_q_top = np.full((num_m, num_q), np.nan)
    conditional_sem_q_top = np.full_like(conditional_mean_q_top, np.nan)
    for i in range(num_m):
        for j in range(num_q):
            values = q_top[i, j][valid[i, j]]
            if values.size:
                conditional_mean_q_top[i, j] = values.mean()
            if values.size > 1:
                conditional_sem_q_top[i, j] = (
                    values.std(ddof=1) / np.sqrt(values.size)
                )

    aggregation_status = np.full(
        (num_m, num_q), AGGREGATION_SAMPLING_INSUFFICIENT,
        dtype="U24",
    )
    aggregation_failure_reasons_object = np.full(
        (num_m, num_q), "", dtype=object
    )
    for i in range(num_m):
        for j in range(num_q):
            reasons = []
            if ensemble == "legacy_delta_only":
                aggregation_status[i, j] = AGGREGATION_FORMAL_ONLY
                reasons.append("legacy_delta_only_not_publication_eligible")
            elif missing_count[i, j] > 0:
                aggregation_status[i, j] = AGGREGATION_INCOMPLETE
            elif invalid_count[i, j] > 0 or valid_count[i, j] != num_d:
                aggregation_status[
                    i, j
                ] = AGGREGATION_SAMPLING_INSUFFICIENT
            else:
                aggregation_status[i, j] = AGGREGATION_REPORTABLE
            if missing_count[i, j] > 0:
                reasons.append("missing_disorders_present")
            if invalid_count[i, j] > 0:
                reasons.append("invalid_disorders_present")
            if (
                ensemble == "true_posterior"
                and missing_count[i, j] == 0
                and invalid_count[i, j] == 0
                and valid_count[i, j] != num_d
            ):
                reasons.append("not_all_planned_disorders_valid")
            aggregation_failure_reasons_object[i, j] = ";".join(reasons)

    reportable = aggregation_status == AGGREGATION_REPORTABLE
    mean_q_top = np.where(reportable, conditional_mean_q_top, np.nan)
    sem_q_top = np.where(reportable, conditional_sem_q_top, np.nan)
    q_crossing = np.where(reportable[..., None], q_top, np.nan)
    aggregation_reason_width = max(
        1,
        max(
            len(str(value))
            for value in aggregation_failure_reasons_object.flat
        ),
    )
    aggregation_failure_reasons = (
        aggregation_failure_reasons_object.astype(
            f"U{aggregation_reason_width}"
        )
    )

    string_shape = shape
    flags = np.full(string_shape, "MISSING", dtype="U512")
    failure_reasons_object = np.full(string_shape, "", dtype=object)
    estimator_names = np.full(string_shape, "", dtype="U64")
    formal_estimator_names = np.full(string_shape, "", dtype="U64")
    weights_estimator_names = np.full(string_shape, "", dtype="U64")
    formal_weights_estimator_names = np.full(string_shape, "", dtype="U64")
    map_success_bound_kinds = np.full(
        string_shape, "unavailable", dtype="U48"
    )
    resolved_engines = np.full(string_shape, "", dtype="U64")
    ti_endpoint_modes = np.full(string_shape, "", dtype="U64")
    task_fingerprints = np.full(string_shape, "", dtype="U64")
    implementation_fingerprints = np.full(string_shape, "", dtype="U64")
    gate_json_object = np.full(string_shape, "", dtype=object)
    ti_json_object = np.full(string_shape, "", dtype=object)
    observable_set_fingerprints = np.full(string_shape, "", dtype="U64")
    code_fingerprints = np.full(string_shape, "", dtype="U64")
    section_fingerprints = np.full(string_shape, "", dtype="U64")
    observable_frame_fingerprints = np.full(string_shape, "", dtype="U64")
    for i in range(num_m):
        for j in range(num_q):
            for d in range(num_d):
                item = all_results[i][j][d]
                if item is None:
                    continue
                flags[i, j, d] = item.get("flags", "")
                failure_reasons_object[i, j, d] = ";".join(
                    item.get("failure_reasons", [])
                )
                estimator_names[i, j, d] = item.get(
                    "q_top_estimator_name", ""
                ) or ""
                formal_estimator_names[i, j, d] = item.get(
                    "formal_q_top_estimator_name", ""
                ) or ""
                weights_estimator_names[i, j, d] = (
                    item.get("weights_estimator_name") or ""
                )
                formal_weights_estimator_names[i, j, d] = (
                    item.get("formal_weights_estimator_name") or ""
                )
                map_success_bound_kinds[i, j, d] = item.get(
                    "map_success_bound_kind", "unavailable"
                )
                resolved_engines[i, j, d] = item.get("resolved_engine", "")
                ti_endpoint_modes[i, j, d] = item.get("endpoint_mode") or ""
                task_fingerprints[i, j, d] = item.get("task_fingerprint", "")
                implementation_fingerprints[i, j, d] = item.get(
                    "implementation_fingerprint", ""
                )
                observable_set_fingerprints[i, j, d] = item.get(
                    "observable_set_fingerprint", ""
                )
                code_fingerprints[i, j, d] = item.get("code_fingerprint", "")
                section_fingerprints[i, j, d] = item.get(
                    "section_fingerprint", ""
                )
                observable_frame_fingerprints[i, j, d] = item.get(
                    "observable_frame_fingerprint", ""
                )
                gate_json_object[i, j, d] = _canonical_json({
                    "passed": item.get("gate_passed"),
                    "failed_checks": item.get("gate_failed_checks"),
                    "metrics": item.get("gate_metrics"),
                    "thresholds": item.get("gate_thresholds"),
                    "notes": item.get("gate_notes"),
                })
                if item.get("resolved_engine") == ENGINE_FULL_TI:
                    ti_json_object[i, j, d] = _canonical_json({
                        "kp_grid": item.get("kp_grid"),
                        "endpoint_mode": item.get("endpoint_mode"),
                        "grid_tv": item.get("grid_tv"),
                        "grid_q_top_abs_diff": item.get(
                            "grid_q_top_abs_diff"
                        ),
                        "proposal_summary": item.get("proposal_summary"),
                        "flags": item.get("flags"),
                    })
    failure_width = max(
        1, max(len(str(value)) for value in failure_reasons_object.flat)
    )
    gate_width = max(1, max(len(str(value)) for value in gate_json_object.flat))
    ti_width = max(1, max(len(str(value)) for value in ti_json_object.flat))
    failure_reasons = failure_reasons_object.astype(f"U{failure_width}")
    gate_json = gate_json_object.astype(f"U{gate_width}")
    ti_json = ti_json_object.astype(f"U{ti_width}")

    pt_ladder_p = np.full((*shape, max(max_pt_temperatures, 1)), np.nan)
    pt_ladder_q = np.full_like(pt_ladder_p, np.nan)
    pt_round_trips = np.full(
        (*shape, max(max_pt_instances, 1)), -1, dtype=np.int64
    )
    pt_burn_in_round_trips = np.full_like(pt_round_trips, -1)
    pt_measurement_round_trips = np.full_like(pt_round_trips, -1)
    pt_swap_rates = np.full(
        (*shape, max(max_pt_instances, 1), max(max_pt_edges, 1)), np.nan
    )
    pt_swap_attempts = np.full(
        (*shape, max(max_pt_instances, 1), max(max_pt_edges, 1)),
        -1, dtype=np.int64,
    )
    pt_swap_accepts = np.full_like(pt_swap_attempts, -1)
    pt_cold_logical_acceptance = np.full(
        (*shape, max(max_pt_instances, 1), max(max_pt_logicals, 1)), np.nan
    )
    for i in range(num_m):
        for j in range(num_q):
            for d in range(num_d):
                item = all_results[i][j][d]
                if item is None or item.get("pt_ladder_p") is None:
                    continue
                lp = np.asarray(item["pt_ladder_p"])
                lq = np.asarray(item["pt_ladder_q"])
                rt = np.asarray(item["pt_round_trips_per_instance"])
                burn_rt = np.asarray(item.get(
                    "pt_burn_in_round_trips_per_instance",
                    np.full(rt.shape, -1),
                ))
                measurement_rt = np.asarray(item.get(
                    "pt_measurement_round_trips_per_instance", rt
                ))
                sr = np.asarray(item["pt_swap_rates_per_instance"])
                sa = np.asarray(item["pt_swap_attempts_per_instance"])
                sc = np.asarray(item["pt_swap_accepts_per_instance"])
                la = np.asarray(item["pt_cold_logical_acceptance_per_instance"])
                pt_ladder_p[i, j, d, :lp.size] = lp
                pt_ladder_q[i, j, d, :lq.size] = lq
                pt_round_trips[i, j, d, :rt.size] = rt
                pt_burn_in_round_trips[
                    i, j, d, :burn_rt.size
                ] = burn_rt
                pt_measurement_round_trips[
                    i, j, d, :measurement_rt.size
                ] = measurement_rt
                pt_swap_rates[i, j, d, :sr.shape[0], :sr.shape[1]] = sr
                pt_swap_attempts[i, j, d, :sa.shape[0], :sa.shape[1]] = sa
                pt_swap_accepts[i, j, d, :sc.shape[0], :sc.shape[1]] = sc
                pt_cold_logical_acceptance[
                    i, j, d, :la.shape[0], :la.shape[1]
                ] = la

    per_size_meta = {}
    per_size_k = {}
    per_size_code_fp = {}
    per_size_section_fp = {}
    per_size_observable_fp = {}
    for i, size in enumerate(size_list):
        first = next(
            (item for row in all_results[i] for item in row if item is not None),
            None,
        )
        per_size_meta[str(size)] = None if first is None else first["family"]
        per_size_k[str(size)] = None if first is None else first["k"]
        per_size_code_fp[str(size)] = None if first is None else first["code_fingerprint"]
        per_size_section_fp[str(size)] = None if first is None else first["section_fingerprint"]
        per_size_observable_fp[str(size)] = None if first is None else first["observable_frame_fingerprint"]

    resolved_engine_configs = {}
    for item in present:
        resolved = item["resolved_engine"]
        config = item.get("resolved_engine_config", {})
        encoded = _canonical_json(config)
        existing = resolved_engine_configs.setdefault(resolved, [])
        if not any(_canonical_json(value) == encoded for value in existing):
            existing.append(config)

    task_git_shas = sorted(set(
        str(item.get("git_commit_sha", "unknown")) for item in present
    ))
    task_dirty_values = sorted(set(
        item.get("git_worktree_dirty") for item in present
    ), key=lambda value: str(value))
    implementation_fingerprint_values = sorted(set(
        str(item.get("implementation_fingerprint", "")) for item in present
    ))
    merge_git = _git_provenance()
    task_git_worktree_dirty = (
        True if True in task_dirty_values
        else False if task_dirty_values == [False]
        else None
    )
    manifest = {
        "protocol": PROTOCOL_VERSION,
        "scan_contract_version": PROTOCOL_VERSION,
        "physics_contract_version": PHYSICS_CONTRACT_VERSION,
        "state_prep_protocol": state_prep_protocol_for_sector(sector),
        "syndrome_semantics": SYNDROME_SEMANTICS,
        "family": family,
        "family_rule": family_rule,
        "family_seed": family_seed,
        "sector": sector,
        "canonical_ensemble": ensemble,
        "ensemble": ensemble,
        "requested_engine": engine,
        "resolved_engines": sorted(set(
            item["resolved_engine"] for item in present
        )),
        "engine_config": engine_config,
        "resolved_engine_configs": resolved_engine_configs,
        "estimator_config": {
            "u_rand_count": int(u_rand_count),
            "minimum_independent_chains": 4,
            "q0_default_independent_chains": 8,
            "sampled_square_estimator": "independent_chain_u_statistic",
            "random_character_error": "finite_population_correction",
        },
        "aggregation_policy": dict(AGGREGATION_POLICY),
        "code_size_list": list(map(int, size_list)),
        "p_value": float(p_value),
        "q_values": list(map(float, q_values)),
        "num_disorder_samples": num_d,
        "implementation_fingerprint": (
            implementation_fingerprint_values[0]
            if len(implementation_fingerprint_values) == 1 else "mixed"
        ),
        "git_commit_sha": (
            task_git_shas[0] if len(task_git_shas) == 1 else "mixed"
        ),
        "git_worktree_dirty": task_git_worktree_dirty,
        "task_git_commit_shas": task_git_shas,
        "task_git_worktree_dirty_values": task_dirty_values,
        "merge_git_commit_sha": merge_git["git_commit_sha"],
        "merge_git_worktree_dirty": merge_git["git_worktree_dirty"],
        "versions": _versions(),
        "hostname": platform.node(),
        "per_size_meta": per_size_meta,
        "per_size_k": per_size_k,
        "per_size_code_fingerprint": per_size_code_fp,
        "per_size_section_fingerprint": per_size_section_fp,
        "per_size_observable_frame_fingerprint": per_size_observable_fp,
        "valid_disorder_count": valid_count.tolist(),
        "planned_disorder_count": planned_count.tolist(),
        "present_disorder_count": present_count.tolist(),
        "numerically_valid_disorder_count": numerical_valid_count.tolist(),
        "formal_only_disorder_count": formal_only_count.tolist(),
        "invalid_disorder_count": invalid_count.tolist(),
        "missing_disorder_count": missing_count.tolist(),
        "aggregation_status_per_point": aggregation_status.tolist(),
        "aggregation_failure_reasons_per_point": (
            aggregation_failure_reasons.tolist()
        ),
        "reportable_for_crossing_fss": reportable.tolist(),
        "missing_chunks": int(missing_count.sum()),
        "fraction_semantics": {
            "paper_aggregation_fraction": (
                "valid_for_aggregation / planned disorder count"
            ),
            "numerical_pass_fraction": (
                "numerically_valid / planned disorder count"
            ),
        },
        "publication_loader": "src.scan_results.load_publication_q_top",
        "character_layout": (
            "last axis padded to actual max character_count; use "
            "character_count_per_disorder/character_mask_per_disorder"
        ),
        "created_unix": time.time(),
    }

    npz_path = output_dir / "scan_results.npz"
    np.savez_compressed(
        npz_path,
        manifest_json=json.dumps(manifest, ensure_ascii=False),
        scan_contract_version=np.asarray(PROTOCOL_VERSION),
        physics_contract_version=np.asarray(PHYSICS_CONTRACT_VERSION),
        state_prep_protocol=np.asarray(state_prep_protocol_for_sector(sector)),
        syndrome_semantics=np.asarray(SYNDROME_SEMANTICS),
        canonical_ensemble=np.asarray(ensemble),
        requested_engine=np.asarray(engine),
        code_size_list=np.asarray(size_list, dtype=np.int64),
        lattice_size_list=np.asarray(size_list, dtype=np.int64),
        q_values=np.asarray(q_values, dtype=np.float64),
        p_value=np.float64(p_value),
        q_top_estimate_per_disorder=q_top,
        q_top_crossing_input_per_disorder=q_crossing,
        formal_q_top_per_disorder=formal_q_top,
        formal_q_top_absolute_per_disorder=scalar(
            lambda item: item.get("formal_q_top_absolute")
        ),
        formal_q_top_relative_per_disorder=scalar(
            lambda item: item.get("formal_q_top_relative")
        ),
        formal_q_top_estimator_name_per_disorder=formal_estimator_names,
        formal_q_top_stderr_per_disorder=scalar(
            lambda item: (
                item.get("formal_q_top_stderr")
                if item.get("formal_q_top_stderr") is not None
                else item.get("formal_q_top_chain_jackknife_se")
            )
        ),
        formal_q_top_raw_pooled_square_per_disorder=scalar(
            lambda item: item.get("formal_q_top_raw_pooled_square")
        ),
        formal_q_top_chain_jackknife_se_per_disorder=scalar(
            lambda item: item.get("formal_q_top_chain_jackknife_se")
        ),
        formal_q_top_character_sampling_se_per_disorder=scalar(
            lambda item: item.get("formal_q_top_character_sampling_se")
        ),
        formal_q_top_raw_character_sampling_se_per_disorder=scalar(
            lambda item: item.get(
                "formal_q_top_raw_character_sampling_se"
            )
        ),
        q_top_absolute_per_disorder=scalar(
            lambda item: item.get("q_top_absolute")
        ),
        q_top_relative_per_disorder=scalar(
            lambda item: item.get("q_top_relative")
        ),
        q_top_estimator_name_per_disorder=estimator_names,
        q_top_stderr_per_disorder=q_top_stderr,
        q_top_raw_pooled_square_per_disorder=scalar(
            lambda item: item.get("q_top_raw_pooled_square")
        ),
        q_top_chain_jackknife_se_per_disorder=scalar(
            lambda item: item.get("q_top_chain_jackknife_se")
        ),
        q_top_character_sampling_se_per_disorder=scalar(
            lambda item: item.get("q_top_character_sampling_se")
        ),
        q_top_raw_character_sampling_se_per_disorder=scalar(
            lambda item: item.get("q_top_raw_character_sampling_se")
        ),
        q_top_per_disorder=q_top,
        mean_q_top_estimate=mean_q_top,
        disorder_sem_q_top_estimate=sem_q_top,
        mean_q_top=mean_q_top,
        disorder_sem_q_top=sem_q_top,
        conditional_mean_q_top_estimate_valid_only=(
            conditional_mean_q_top
        ),
        conditional_disorder_sem_q_top_estimate_valid_only=(
            conditional_sem_q_top
        ),
        aggregation_status_per_point=aggregation_status,
        aggregation_failure_reasons_per_point=(
            aggregation_failure_reasons
        ),
        reportable_for_crossing_fss=reportable,
        valid_for_aggregation=valid,
        numerically_valid=numerical_valid,
        formal_only=formal_only,
        valid_disorder_count=valid_count,
        planned_disorder_count=planned_count,
        present_disorder_count=present_count,
        numerically_valid_disorder_count=numerical_valid_count,
        formal_only_disorder_count=formal_only_count,
        invalid_disorder_count=invalid_count,
        missing_disorder_count=missing_count,
        character_count_per_disorder=character_count,
        character_mask_per_disorder=character_mask,
        independent_chain_count_per_disorder=independent_chain_count,
        independent_chain_mask_per_disorder=independent_chain_mask,
        u_rand_seed_per_disorder=scalar(
            lambda item: item.get("u_rand_seed"), default=-1, dtype=np.int64
        ),
        m_u_absolute_per_disorder=character_fields[
            "character_means_absolute_per_disorder"
        ],
        m_u_relative_per_disorder=character_fields[
            "character_means_relative_per_disorder"
        ],
        chain_character_means_relative_per_disorder=chain_relative,
        chain_character_means_absolute_per_disorder=chain_absolute,
        formal_chain_sector_characters_relative_per_disorder=(
            formal_chain_relative
        ),
        formal_chain_sector_characters_absolute_per_disorder=(
            formal_chain_absolute
        ),
        weights_absolute_per_disorder=weights_absolute,
        weights_relative_per_disorder=weights_relative,
        formal_sector_weights_absolute_per_disorder=formal_weights_absolute,
        formal_sector_weights_relative_per_disorder=formal_weights_relative,
        weights_estimator_name_per_disorder=weights_estimator_names,
        formal_weights_estimator_name_per_disorder=(
            formal_weights_estimator_names
        ),
        weights_are_exact_sector_posterior_per_disorder=scalar(
            lambda item: bool(
                item.get("weights_are_exact_sector_posterior", False)
            ),
            default=False, dtype=bool,
        ),
        weights_cover_all_sectors_per_disorder=scalar(
            lambda item: bool(item.get("weights_cover_all_sectors", False)),
            default=False, dtype=bool,
        ),
        formal_weights_are_exact_sector_posterior_per_disorder=scalar(
            lambda item: bool(item.get(
                "formal_weights_are_exact_sector_posterior", False
            )),
            default=False, dtype=bool,
        ),
        formal_weights_cover_all_sectors_per_disorder=scalar(
            lambda item: bool(item.get(
                "formal_weights_cover_all_sectors", False
            )),
            default=False, dtype=bool,
        ),
        weights_per_disorder=weights_relative,
        delta_f_per_disorder=delta_f,
        delta_f_infinite_mask_per_disorder=delta_f_infinite_mask,
        delta_f_stderr_per_disorder=delta_f_stderr,
        ti_grid_tv_per_disorder=scalar(
            lambda item: item.get("grid_tv")
        ),
        ti_grid_q_top_abs_diff_per_disorder=scalar(
            lambda item: item.get("grid_q_top_abs_diff")
        ),
        ti_endpoint_mode_per_disorder=ti_endpoint_modes,
        ti_diagnostics_json_per_disorder=ti_json,
        posterior_purity_per_disorder=scalar(
            lambda item: item.get("posterior_purity")
        ),
        formal_sector_purity_per_disorder=scalar(
            lambda item: item.get("formal_sector_purity")
        ),
        posterior_mass_on_planted_class_per_disorder=scalar(
            lambda item: item.get("posterior_mass_on_planted_class")
        ),
        posterior_mass_character_sampling_se_per_disorder=scalar(
            lambda item: item.get("posterior_mass_character_sampling_se")
        ),
        map_success_probability_per_disorder=scalar(
            lambda item: item.get("map_success_probability")
        ),
        map_success_algebraic_lower_bound_per_disorder=scalar(
            lambda item: item.get("map_success_algebraic_lower_bound")
        ),
        map_success_algebraic_upper_bound_per_disorder=scalar(
            lambda item: item.get("map_success_algebraic_upper_bound")
        ),
        map_success_estimated_lower_bound_per_disorder=scalar(
            lambda item: item.get("map_success_estimated_lower_bound")
        ),
        map_success_estimated_upper_bound_per_disorder=scalar(
            lambda item: item.get("map_success_estimated_upper_bound")
        ),
        map_success_bound_kind_per_disorder=map_success_bound_kinds,
        map_success_bound_has_confidence_coverage_per_disorder=scalar(
            lambda item: bool(item.get(
                "map_success_bound_has_confidence_coverage", False
            )),
            default=False,
            dtype=bool,
        ),
        largest_sector_mass_per_disorder=scalar(
            lambda item: item.get("largest_sector_mass")
        ),
        planted_logical_class_bitmask_per_disorder=scalar(
            lambda item: item.get("planted_logical_class_bitmask"),
            default=-1, dtype=np.int64,
        ),
        resolved_engine_per_disorder=resolved_engines,
        task_fingerprint_per_disorder=task_fingerprints,
        implementation_fingerprint_per_disorder=implementation_fingerprints,
        implementation_fingerprint=np.asarray(
            manifest["implementation_fingerprint"]
        ),
        git_commit_sha=np.asarray(manifest["git_commit_sha"]),
        git_worktree_dirty=np.asarray(
            bool(task_git_worktree_dirty)
            if task_git_worktree_dirty is not None else False,
            dtype=bool,
        ),
        git_worktree_dirty_known=np.asarray(
            task_git_worktree_dirty is not None, dtype=bool
        ),
        code_fingerprint_per_disorder=code_fingerprints,
        section_fingerprint_per_disorder=section_fingerprints,
        observable_frame_fingerprint_per_disorder=observable_frame_fingerprints,
        observable_set_fingerprint_per_disorder=observable_set_fingerprints,
        flags_per_disorder=flags,
        failure_reasons_per_disorder=failure_reasons,
        gate_diagnostics_json_per_disorder=gate_json,
        pt_ladder_p_per_disorder=pt_ladder_p,
        pt_ladder_q_per_disorder=pt_ladder_q,
        pt_round_trips_per_disorder=pt_round_trips,
        pt_burn_in_round_trips_per_disorder=pt_burn_in_round_trips,
        pt_measurement_round_trips_per_disorder=(
            pt_measurement_round_trips
        ),
        pt_swap_rates_per_disorder=pt_swap_rates,
        pt_swap_attempts_per_disorder=pt_swap_attempts,
        pt_swap_accepts_per_disorder=pt_swap_accepts,
        pt_cold_logical_acceptance_per_disorder=pt_cold_logical_acceptance,
        wall_time_seconds_per_disorder=scalar(
            lambda item: item.get("wall_time_seconds")
        ),
        disorder_seed_per_disorder=scalar(
            lambda item: item.get("disorder_seed"), default=-1, dtype=np.int64
        ),
        sample_seed_per_disorder=scalar(
            lambda item: item.get("engine_seed"), default=-1, dtype=np.int64
        ),
        paper_aggregation_fraction=paper_aggregation_fraction,
        numerical_pass_fraction=numerical_pass_fraction,
        **character_fields,
        **formal_character_fields,
    )
    _atomic_write_json(output_dir / "manifest.json", manifest)
    return npz_path


def build_arg_parser():
    parser = argparse.ArgumentParser(description="exp101 aligned scan entry")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--family", default="expander34",
        choices=["expander34", "toric", "surface", "k43"],
    )
    parser.add_argument("--size-list", type=int, nargs="+", required=True)
    parser.add_argument("--p-value", type=float, required=True)
    parser.add_argument("--q-values", type=float, nargs="+", required=True)
    parser.add_argument("--num-disorders", type=int, required=True)
    parser.add_argument(
        "--sector", default="x_error", choices=["x_error", "z_error"]
    )
    parser.add_argument(
        "--ensemble", default="true_posterior", choices=ACCEPTED_ENSEMBLES
    )
    parser.add_argument("--engine", default="auto", choices=REQUESTED_ENGINES)
    parser.add_argument(
        "--family-rule", default="full_rank",
        choices=["full_rank", "full_rank_d3"],
    )
    parser.add_argument("--family-seed", type=int, default=None)
    parser.add_argument("--num-random-u", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--force-recompute", action="store_true")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    npz_path, report = scan(
        args.output_dir, args.family, args.size_list, args.p_value,
        args.q_values, args.num_disorders, sector=args.sector,
        ensemble=args.ensemble, engine=args.engine,
        family_rule=args.family_rule, family_seed=args.family_seed,
        u_rand_count=args.num_random_u, num_workers=args.num_workers,
        force_recompute=args.force_recompute,
    )
    print(f"npz: {npz_path}\nreport: {report}")


if __name__ == "__main__":
    main()
