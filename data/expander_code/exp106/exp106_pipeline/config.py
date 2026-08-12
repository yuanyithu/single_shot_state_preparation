import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

from . import (
    CONFIG_SCHEMA,
    EXPERIMENT_ID,
    PILOT_CONFIG_SCHEMA,
    PILOT_REMOTE_CONFIG_SCHEMA,
)
from .io import sha256_json


# The readout error rate is the whole point of exp106 and is fixed for every
# task in the experiment. It is a token rather than a float so that it enters
# seed derivation and identity hashes as an exact decimal string.
#
# 0.01 is an interior point of the bracket exp105 left open. exp104 certified a
# crossing at p = 0.05512 with q = 0; exp105 certified no crossing at any p with
# q = 0.05; so the readout threshold of this decoder on this family lies strictly
# inside (0, 0.05). exp106 measures one point in that interval.
Q_TOKEN = "0.01"

M_VALUES = [3, 4, 5, 6, 7, 8]
# Census only. m = 2 is measured for the ensemble composition census -- the
# distance spectrum f_d(m) that explained exp103's anomaly -- and never enters a
# panel, an aggregation or the terminal decision. exp106 has no Track B: exp105
# established that full-sector TI cannot certify a q_top anchor at q > 0, and
# permanent discipline 13 forbids adding budget to the same instrument.
CENSUS_ONLY_M_VALUES = [2]

# ---------------------------------------------------------------------------
# Locating pilot. Frozen before the pilot runs. Pilot raw is never merged into
# production and never enters a published statistic; its only function is to
# evaluate the two freezing rules in EXPERIMENT_CONTRACT.md section 6.
# ---------------------------------------------------------------------------
PILOT_M_VALUES = [3, 8]
# The grid is dense across p in [0.02, 0.06] and covers both its edges, fixed
# before anything ran. This is not exp105's grid and deliberately so. exp104
# measured Delta38 negative exactly on p in [0.02, 0.055] at q = 0, with a full
# depth of only 0.053 and a crossing at 0.05512; readout noise can only move that
# window down and make it shallower. exp105's pilot grid was log-spaced for a
# low-p regime and put seven of its fourteen points below 0.016, which is the
# wrong place to look for a residual dip at q = 0.01. Half the spacing here is
# 0.005 so that the grid rule's [p_lo - 2h, p_hi + 2h] window stays narrow.
PILOT_P_TOKENS = [
    "0.005", "0.0075", "0.01", "0.015", "0.02", "0.025", "0.03",
    "0.035", "0.04", "0.045", "0.05", "0.055", "0.06", "0.07",
]
PILOT_CODES_PER_M = 200
PILOT_TRIALS_PER_CODE_P = 4
PILOT_CODES_PER_TASK = {3: 50, 8: 5}
# The pilot *registry* carries every m, the pilot *panel* only the primary pair.
# The extra rows exist so that `remote_cli cost-benchmark` can time all six sizes
# on nd-3 before the production registry exists -- which is what breaks exp105's
# circularity, where the only path to nd-3 costs was a preflight that already
# required the frozen plan. Those rows are never scanned, never aggregated and
# never published; two codes per size is enough to bound a cost.
PILOT_REGISTRY_CODES_PER_M = {3: 200, 4: 2, 5: 2, 6: 2, 7: 2, 8: 200}
# Trials per timed batch in the cost benchmark. Fixed at the top of the allowed
# range so that c_m is a mean over the same batch size production will use.
COST_BENCHMARK_TRIALS = 6

# ---------------------------------------------------------------------------
# Production plan. NOT frozen: Validation 003 evaluates the section 6 rules on
# measured pilot statistics and measured nd-3 costs, and filling these four
# constants in is the freeze. Until then every production entry point raises
# ProductionPlanNotFrozen, so a half-frozen plan is impossible by construction.
#
# The costs must come from nd-3 and not from the machine that happens to run the
# pilot. exp105 evaluated the same rule on macmini numbers and its nd-3 resource
# gate blocked at 5,368 reserved core-hours against a cap of 800, because a trial
# at m = 8 costs 4.88 s on nd-3 against 0.61 s on the macmini. exp106 therefore
# measures c_m and kappa_m with `remote_cli cost-benchmark` on nd-3 before the
# allocation rule is evaluated at all.
# ---------------------------------------------------------------------------
PRODUCTION_PLAN_FROZEN = False
P_TOKENS = None
CODES_PER_M = None
TRIALS_PER_CODE_P = None
CODES_PER_TASK = None

# Inputs to the frozen allocation rule, fixed here so that Validation 003
# evaluates a rule rather than choosing an outcome.
#
# 800 projected core-hours, against exp105's 290. Two independent reasons: the
# whole exp106 grid sits in the expensive half of the p range, where belief
# propagation exhausts max_iter and OSD does all the work (per-trial cost at
# m = 8 on nd-3 runs from 0.0043 s at p = 0.001 to 4.90 s at p = 0.07, a factor
# of 1,135), so the same budget buys about half the trials; and the contrast to
# be resolved is far smaller, since the q = 0 dip this experiment asks about is
# only 0.053 deep and any residual at q = 0.01 is shallower still.
GENERATION_BUDGET_CORE_HOURS = 800.0
DIAGNOSTIC_BUDGET_SHARE = 0.06
DIAGNOSTIC_M_VALUES = [4, 5, 6, 7]
PRIMARY_M_VALUES = [3, 8]
TRIALS_PER_CODE_P_RANGE = (3, 6)
# A task is one contiguous block of codes at one m across the whole p grid. The
# block size is a rule, not a hand-picked number: it is the largest block whose
# projected duration stays under this target, so that a failed task loses a
# bounded amount of work and the longest-first schedule has a short tail.
TARGET_TASK_SECONDS = 300.0
PRODUCTION_GRID_POINTS = 10
PRODUCTION_GRID_CLIP = ("0.0005", "0.10")
PRODUCTION_GRID_DECIMALS = 4
# Used only if the pilot finds no sign change anywhere. exp105's fallback was
# log-spaced down to 0.001, on the argument that if the crossing is not where the
# pilot looked then its order of magnitude is what is uncertain. That argument
# does not transfer: here the crossing's location at q = 0 is *known* to three
# significant figures, and the only question is whether q = 0.01 has erased it.
# So the fallback covers the window exp104 measured negative, and a no-crossing
# terminal then means "certified positive across the whole window where q = 0 was
# negative" rather than "positive somewhere else".
FALLBACK_P_TOKENS = [
    "0.005", "0.01", "0.015", "0.02", "0.025", "0.03", "0.035",
    "0.045", "0.055", "0.07",
]

# A fresh master seed. exp106 draws its own ensemble so that no exp104 or exp105
# code enters an exp106 panel by construction; both registries are still used,
# read only, by the two equality gates.
MASTER_SEED_HEX = "c48b2f7e05a1d6934be7c02f8d5169ab3e7f04c2915d8b6a70e3fc41d92b58a7"

ENSEMBLE = {
    "graph_family": "random_biregular_bipartite_simple",
    "d_A": 3,
    "d_B": 4,
    "construction": "configuration_model_reject_multi_edge",
    "max_attempts": 10000,
    "acceptance_rule": "full_row_rank_and_unique_H",
    "post_selection": "none",
}

# Byte for byte the frozen exp103.decoder_mc.v2 / exp104 identity, except that
# max_iter follows the block length actually being decoded. At q > 0 that block
# is the augmented [H_Z | I], so the iteration budget is n + n_c. The q = 0
# comparison path of the equality gate decodes H_Z alone with max_iter = n, so
# exp104 comparability is unaffected.
DECODER = {
    "bp_method": "product_sum",
    "max_iter": "n_plus_nc",
    "schedule": "serial",
    "serial_schedule_order": "natural",
    "osd_method": "osd_0",
    "osd_order": 0,
    "omp_thread_count": 1,
    "augmented_matrix": "H_Z_hstack_identity",
    "error_channel": "p_on_data_q_on_checks",
}
# The same installed backend exp104 qualified: identical Python, numpy, scipy
# and ldpc versions and the identical compiled extension.
DECODER_BINARY_SHA256 = "944a96a657a89fbd04c127edb2eba1033f56de0161ddcd2ba7e57dee76777ccc"
DECODER_BINARY_SUFFIX = ".cpython-312-darwin.so"

NAMESPACES = {
    "ensemble": "exp106.noisy_syndrome_mc.ensemble.v1",
    "benchmark": "exp106.noisy_syndrome_mc.benchmark.v1",
    "measurement": "exp106.noisy_syndrome_mc.measurement.v1",
    "replay": "exp106.noisy_syndrome_mc.replay.v1",
    "bootstrap": "exp106.noisy_syndrome_mc.bootstrap.v1",
    "pilot": "exp106.noisy_syndrome_mc.pilot.v1",
}
REGISTRY_PATH = "data/expander_code/exp106/config/ensemble_registry.v1.npz"
PILOT_REGISTRY_PATH = "data/expander_code/exp106/config/ensemble_registry.pilot.v1.npz"
# The pilot draws its codes from a separate seed namespace as well as a separate
# file, so that no code used to choose the frozen grid is later measured on it.
REGISTRY_PATH_BY_PHASE = {
    "pilot": PILOT_REGISTRY_PATH,
    "pilot_remote": PILOT_REGISTRY_PATH,
    "production": REGISTRY_PATH,
    "production_remote": REGISTRY_PATH,
}
REGISTRY_NAMESPACE_BY_PHASE = {
    "pilot": "pilot",
    "pilot_remote": "pilot",
    "production": "ensemble",
    "production_remote": "ensemble",
}

CROSSING = {
    "primary_contrast": "m08_minus_m03",
    "direction": "negative_at_lower_p_positive_at_higher_p",
    "bracket_requires_adjacent_grid_points": False,
    "simultaneous_band_scope": "primary_contrast_grid_only",
    "location_estimator": "linear_interpolation_first_sign_change_in_bracket",
    "adjacent_contrasts": "diagnostic_pointwise_only",
    # At q = 0.01 either answer is a real physical possibility: the crossing may
    # survive from q = 0, shallower and shifted, or the readout channel may
    # already have erased it. Neither is an experimental failure, and the
    # contract names both legitimate terminals.
    "no_crossing_is_a_legitimate_terminal": True,
}
REPLAY = {
    "policy": "committed_random_subsample",
    "fraction": 0.10,
    "always_include_block_index": 0,
}
BOOTSTRAP = {
    "replicates": 20000,
    "confidence": 0.95,
    "method": "cluster_bootstrap_over_codes_max_abs_on_primary_contrast",
    "resample_within_m": True,
}
PREFLIGHT = {
    "code_indices": [0, 1],
    "m_values": [3, 4, 5, 6, 7, 8],
    # The production grid is frozen only at Validation 003, so the benchmarked
    # points are named by position rather than by value. Cost rises with p only
    # through belief propagation, which already exhausts max_iter everywhere in
    # this window, so the ends and the middle bound the grid.
    "p_token_selection": "first_middle_last",
    "num_workers": 8,
    "reserve_multiplier": 2.0,
    "stage_core_hour_cap": 100.0,
    "stage_wall_hour_cap": 12.0,
    "peak_rss_gib_cap": 12.0,
    "analysis_core_hours": 1.0,
    "fixed_overhead_core_hours": 1.0,
}

OBJECTIVE = "bposd_ensemble_block_logical_failure_crossing_q001"

TOP_LEVEL_FIELDS = {
    "schema_version", "experiment_id", "objective", "registry_path",
    "registry_sha256", "source_commit", "source_tree_sha256",
    "master_seed_hex", "m_values", "codes_per_m", "p_tokens", "q_token",
    "trials_per_code_p", "codes_per_task", "ensemble", "decoder",
    "environment", "decoder_binary", "namespaces", "bootstrap", "replay",
    "preflight", "crossing", "phase",
}
REMOTE_CONFIG_SCHEMA = "exp106.config.remote.v1"
REMOTE_CONFIG_PATH = "data/expander_code/exp106/config/noisy_mc.remote.v1.json"
PILOT_REMOTE_CONFIG_PATH = (
    "data/expander_code/exp106/config/noisy_mc.pilot.remote.v1.json"
)
REMOTE_EXECUTION_PROFILE = "exp106.remote_execution.v1"
# exp106 reuses the environment exp103 built and exp104 qualified rather than
# creating its own. Rebuilding would recompile the decoder and could change its
# binary hash, and a byte-identical decoder is what keeps exp104's q = 0 cells
# comparable evidence.
REMOTE_CONDA_PREFIX = "/home/DATA1/users/yuany/.single_shot/cache/exp103_remote_v1_env"
REMOTE_DECODER_BINARY_SHA256 = "3a5a7dc2c1ed015eb137ef5823d7e2d13c2d851fe895788adc3bded4e4d0c079"
REMOTE_DECODER_BINARY_SUFFIX = ".cpython-312-x86_64-linux-gnu.so"
REMOTE_TOP_LEVEL_FIELDS = TOP_LEVEL_FIELDS | {
    "execution_profile", "ldpc_source", "support_packages",
}
REMOTE_EXECUTION_FIELDS = {
    "profile_id", "entry_host", "compute_host", "conda_environment",
    "num_workers", "omp_thread_count", "run_root", "log_root",
    "reserve_multiplier", "stage_core_hour_cap", "stage_wall_hour_cap",
    "peak_rss_gib_cap",
}
# One definition, read both by the config author and by the validator. exp105
# kept two copies of these numbers and the same class of duplication -- the
# benchmark grid, defined once in preflight and once in remote_cli -- produced a
# KeyError in the middle of a resource gate.
#
# 72 workers, against exp105's 64. nd-3 has 96 logical CPUs on 48 physical cores,
# and another user has held ten cores at 100 percent continuously; 72 takes the
# available parallelism without contending for their hyperthreads.
#
# The caps follow permanent discipline 11 from the frozen 800 core-hour
# generation budget: reserved = 2 x (800 + 80 replay + 1 analysis + 1 overhead),
# and wall = (800 + 80) / 72 + 2.
REMOTE_EXECUTION_DEFAULTS = {
    "profile_id": REMOTE_EXECUTION_PROFILE,
    "entry_host": "yuany",
    "compute_host": "nd-3",
    "num_workers": 72,
    "omp_thread_count": 1,
    "run_root": "~/.single_shot/runs",
    "log_root": "~/.single_shot/logs",
    "reserve_multiplier": 2.0,
    "stage_core_hour_cap": 1800.0,
    "stage_wall_hour_cap": 20.0,
    "peak_rss_gib_cap": 128.0,
}
REMOTE_LDPC_SOURCE = {
    "repository": "https://github.com/quantumgizmos/ldpc.git",
    "commit": "d3429964cd4ffe1abfc041c6ec8b8425cb174f40",
    "archive_url": (
        "https://github.com/quantumgizmos/ldpc/archive/"
        "d3429964cd4ffe1abfc041c6ec8b8425cb174f40.tar.gz"
    ),
    "archive_path": (
        "/home/DATA1/users/yuany/.single_shot/cache/ldpc-d3429964.tar.gz"
    ),
    "archive_sha256": "76af0f01446ee7cbed33a47d6b597c10d8d12b2f10d508911b3d0763844d467e",
    "rng_hpp_sha256": "300281477f91ed5ab3ba300a5e379797349db4048d8e37a21082fc5d5e15ba7c",
}
REMOTE_SUPPORT_PACKAGES = {
    "numba": "0.66.0",
    "llvmlite": "0.48.0",
    "pytest": "9.1.1",
    "matplotlib": "3.11.1",
    "stim": "1.15.0",
    "sinter": "1.15.0",
    "pymatching": "2.2.2",
    "networkx": "3.6.1",
    "tqdm": "4.70.0",
}

SCHEMA_BY_PHASE = {
    "pilot": PILOT_CONFIG_SCHEMA,
    "pilot_remote": PILOT_REMOTE_CONFIG_SCHEMA,
    "production": CONFIG_SCHEMA,
    "production_remote": REMOTE_CONFIG_SCHEMA,
}
# The pilot is the one phase allowed to exist before the production plan is
# frozen, because it is what freezes it. It has a remote form because the
# allocation rule needs per-trial costs measured on nd-3, and measuring them
# under anything less than the full identity binding is how exp105 came to
# evaluate its rule on the wrong machine's numbers.
REMOTE_SCHEMAS = {REMOTE_CONFIG_SCHEMA, PILOT_REMOTE_CONFIG_SCHEMA}
LOCAL_SCHEMAS = {PILOT_CONFIG_SCHEMA, CONFIG_SCHEMA}
# The pilot phases never publish, so they are exempt from the production
# plan freeze; every production entry point is not.
PILOT_PHASES = {"pilot", "pilot_remote"}


class ProductionPlanNotFrozen(RuntimeError):
    """Raised when a production entry point runs before Validation 003."""


def require_production_plan_frozen():
    if not PRODUCTION_PLAN_FROZEN or None in (
        P_TOKENS, CODES_PER_M, TRIALS_PER_CODE_P, CODES_PER_TASK,
    ):
        raise ProductionPlanNotFrozen(
            "the exp106 production p grid and per-m allocation are not frozen; "
            "Validation 003 must evaluate the contract section 6 rules first"
        )


def plan_for_phase(phase):
    """Return (m_values, p_tokens, codes_per_m, trials, codes_per_task)."""
    if phase in PILOT_PHASES:
        return (
            list(PILOT_M_VALUES), list(PILOT_P_TOKENS),
            {m: PILOT_CODES_PER_M for m in PILOT_M_VALUES},
            {m: PILOT_TRIALS_PER_CODE_P for m in PILOT_M_VALUES},
            dict(PILOT_CODES_PER_TASK),
        )
    if phase in ("production", "production_remote"):
        require_production_plan_frozen()
        return (
            list(M_VALUES), list(P_TOKENS), dict(CODES_PER_M),
            dict(TRIALS_PER_CODE_P), dict(CODES_PER_TASK),
        )
    raise ValueError(f"unknown exp106 phase {phase!r}")


def tasks_per_m(codes_per_m, codes_per_task):
    tasks = {}
    for m, count in codes_per_m.items():
        size = codes_per_task[m]
        if count % size:
            raise ValueError(f"codes_per_m[{m}] is not a multiple of its block size")
        tasks[m] = count // size
    return tasks


def normalize_p_token(value, tokens):
    """Resolve a p to its exact decimal token inside the given frozen grid."""
    tokens = list(tokens)
    if isinstance(value, str):
        token = value
    else:
        try:
            numeric = Decimal(str(value))
        except (InvalidOperation, ValueError):
            raise ValueError(f"p is outside the frozen grid: {value!r}") from None
        matches = [item for item in tokens if Decimal(item) == numeric]
        if len(matches) != 1:
            raise ValueError(f"p is outside the frozen grid: {value!r}")
        token = matches[0]
    if token not in tokens:
        raise ValueError(f"p is outside the frozen grid: {value!r}")
    return token


def normalize_q_token(value):
    if isinstance(value, str):
        token = value
    else:
        try:
            numeric = Decimal(str(value))
        except (InvalidOperation, ValueError):
            raise ValueError(f"q is outside the frozen contract: {value!r}") from None
        token = Q_TOKEN if numeric == Decimal(Q_TOKEN) else None
    if token != Q_TOKEN:
        raise ValueError(f"q is outside the frozen contract: {value!r}")
    return token


# Code ids are structural: format and a known m. Panel membership is enforced
# where it is actually knowable -- by the registry index and the task blocking
# -- because exp106 panels differ per m and per phase.
_CODE_ID_PATTERN = re.compile(r"m(\d{2})_c(\d{6})")
_ALL_M_VALUES = sorted(set(M_VALUES) | set(CENSUS_ONLY_M_VALUES))
MAX_CODE_INDEX = 1_000_000


def code_id(m, index):
    if int(m) not in _ALL_M_VALUES:
        raise ValueError(f"m is outside the frozen family: {m!r}")
    if isinstance(index, bool) or not isinstance(index, int):
        raise ValueError("code index must be an integer")
    if not 0 <= index < MAX_CODE_INDEX:
        raise ValueError(f"code index is out of range: {index!r}")
    return f"m{int(m):02d}_c{int(index):06d}"


def parse_code_id(value):
    match = _CODE_ID_PATTERN.fullmatch(str(value))
    if match is None:
        raise ValueError(f"malformed exp106 code id: {value!r}")
    m = int(match.group(1))
    index = int(match.group(2))
    if m not in _ALL_M_VALUES or not 0 <= index < MAX_CODE_INDEX:
        raise ValueError(f"code id is outside the frozen family: {value!r}")
    return m, index


def block_code_indices(config, m, block_index):
    """Return the frozen contiguous code index range owned by one task."""
    config = ensure_config(config)
    m = int(m)
    if m not in config["m_values"]:
        raise ValueError(f"m is outside the frozen panel: {m!r}")
    counts = {int(key): int(value) for key, value in config["codes_per_m"].items()}
    sizes = {int(key): int(value) for key, value in config["codes_per_task"].items()}
    total = tasks_per_m(counts, sizes)[m]
    if isinstance(block_index, bool) or not isinstance(block_index, int):
        raise ValueError("block index must be an integer")
    if not 0 <= block_index < total:
        raise ValueError(f"block index is outside the frozen plan: {block_index!r}")
    size = sizes[m]
    start = block_index * size
    return list(range(start, start + size))


def _validate_remote_execution(config):
    profile = config["execution_profile"]
    if not isinstance(profile, dict) or set(profile) != REMOTE_EXECUTION_FIELDS:
        raise ValueError("unexpected exp106 remote execution profile fields")
    for field, expected in REMOTE_EXECUTION_DEFAULTS.items():
        if profile[field] != expected:
            raise ValueError(f"remote execution profile mismatch for {field}")
    if (
        not isinstance(profile["conda_environment"], str)
        or profile["conda_environment"] != REMOTE_CONDA_PREFIX
        or profile["conda_environment"] != config["environment"]["conda_environment"]
    ):
        raise ValueError("remote execution conda environment is not frozen consistently")
    if profile["omp_thread_count"] != config["decoder"]["omp_thread_count"]:
        raise ValueError("remote execution OpenMP setting differs from the decoder")


def _validate(config):
    schema = config.get("schema_version")
    phase = config.get("phase")
    if phase not in SCHEMA_BY_PHASE or SCHEMA_BY_PHASE[phase] != schema:
        raise ValueError("exp106 config phase and schema disagree")
    remote = schema in REMOTE_SCHEMAS
    expected_fields = REMOTE_TOP_LEVEL_FIELDS if remote else TOP_LEVEL_FIELDS
    if set(config) != expected_fields:
        raise ValueError("unexpected exp106 config fields")
    if config["experiment_id"] != EXPERIMENT_ID:
        raise ValueError("exp106 config identity mismatch")
    if config["objective"] != OBJECTIVE:
        raise ValueError("unexpected exp106 objective")
    if config["registry_path"] != REGISTRY_PATH_BY_PHASE[phase]:
        raise ValueError("exp106 must use the canonical ensemble registry for its phase")
    if config["q_token"] != Q_TOKEN:
        raise ValueError("exp106 is a fixed-q experiment; q must be the frozen token")

    m_values, p_tokens, codes_per_m, trials, codes_per_task = plan_for_phase(phase)
    if config["m_values"] != m_values:
        raise ValueError("panel sizes are not the frozen plan for this phase")
    if config["p_tokens"] != p_tokens:
        raise ValueError("unexpected p grid for this phase")
    if config["codes_per_m"] != {str(m): codes_per_m[m] for m in m_values}:
        raise ValueError("codes per m is not the frozen allocation")
    if config["trials_per_code_p"] != {str(m): trials[m] for m in m_values}:
        raise ValueError("trials per code-p is not the frozen allocation")
    if config["codes_per_task"] != {str(m): codes_per_task[m] for m in m_values}:
        raise ValueError("task blocking is not frozen")
    tasks_per_m(codes_per_m, codes_per_task)

    if config["ensemble"] != ENSEMBLE:
        raise ValueError("ensemble definition differs from the frozen rule")
    if config["decoder"] != DECODER:
        raise ValueError("decoder parameters differ from the frozen BP+OSD-0 identity")
    if remote:
        _validate_remote_execution(config)
        if config["ldpc_source"] != REMOTE_LDPC_SOURCE:
            raise ValueError("remote ldpc source provenance differs from the frozen release")
        if config["support_packages"] != REMOTE_SUPPORT_PACKAGES:
            raise ValueError("remote support package identity differs from the frozen environment")
    if config["namespaces"] != NAMESPACES or len(set(NAMESPACES.values())) != len(NAMESPACES):
        raise ValueError("seed namespaces must be frozen and disjoint")
    if config["master_seed_hex"] != MASTER_SEED_HEX:
        raise ValueError("exp106 master seed differs from the fresh frozen seed")
    for field in ("registry_sha256", "source_tree_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", config[field]):
            raise ValueError(f"{field} must be a SHA256")
    if not re.fullmatch(r"[0-9a-f]{40}", config["source_commit"]):
        raise ValueError("source_commit must be a full lowercase Git SHA")
    env = config["environment"]
    if remote:
        expected_remote_environment = {
            "device_name": "nd-3",
            "hostname": "nd-3",
            "conda_environment": config["execution_profile"]["conda_environment"],
            "conda_prefix_matches_python": True,
            "python": "3.12.12", "numpy": "2.4.1", "scipy": "1.17.0", "ldpc": "2.4.1",
        }
        if env != expected_remote_environment:
            raise ValueError("remote environment identity differs from the frozen contract")
    elif env != {
        "device_name": "macmini",
        "hostname": "ymini.local",
        "conda_environment": "12",
        "conda_prefix_matches_python": True,
        "python": "3.12.12", "numpy": "2.4.1", "scipy": "1.17.0", "ldpc": "2.4.1",
    }:
        raise ValueError("canonical environment identity differs from the frozen contract")
    if type(env["conda_prefix_matches_python"]) is not bool:
        raise ValueError("conda prefix attestation must be a boolean")
    binary = config["decoder_binary"]
    if set(binary) != {"module", "filename_suffix", "sha256"}:
        raise ValueError("unexpected decoder binary identity fields")
    if binary["module"] != "ldpc.bposd_decoder._bposd_decoder":
        raise ValueError("decoder extension module mismatch")
    if remote:
        if (
            binary["sha256"] != REMOTE_DECODER_BINARY_SHA256
            or binary["filename_suffix"] != REMOTE_DECODER_BINARY_SUFFIX
        ):
            raise ValueError("remote decoder extension binary is not fully frozen")
    elif binary["sha256"] != DECODER_BINARY_SHA256 or binary["filename_suffix"] != DECODER_BINARY_SUFFIX:
        raise ValueError("decoder extension binary differs from the frozen backend")
    if config["bootstrap"] != BOOTSTRAP:
        raise ValueError("bootstrap family is not frozen")
    if config["replay"] != REPLAY:
        raise ValueError("replay policy is not frozen")
    if config["preflight"] != PREFLIGHT:
        raise ValueError("local resource preflight is not frozen")
    if config["crossing"] != CROSSING:
        raise ValueError("crossing decision family is not frozen")
    if remote and (
        config["source_commit"] == "0" * 40
        or config["source_tree_sha256"] == "0" * 64
    ):
        raise ValueError("remote source identity contains an unresolved placeholder")


_CANONICAL_FILENAME = {
    PILOT_CONFIG_SCHEMA: "noisy_mc.pilot.v1.json",
    PILOT_REMOTE_CONFIG_SCHEMA: "noisy_mc.pilot.remote.v1.json",
    CONFIG_SCHEMA: "noisy_mc.v1.json",
    REMOTE_CONFIG_SCHEMA: "noisy_mc.remote.v1.json",
}


def load_config(path):
    path = Path(path)
    raw = json.loads(path.read_text(encoding="ascii"))
    filename = _CANONICAL_FILENAME.get(raw.get("schema_version"))
    if filename is None:
        raise ValueError("unknown exp106 config schema")
    canonical_path = Path(__file__).resolve().parents[1] / "config" / filename
    if path.resolve() != canonical_path.resolve():
        raise ValueError("formal exp106 config must be the canonical config artifact")
    config = raw
    _validate(config)
    resolved = dict(config)
    resolved["config_sha256"] = sha256_json(config)
    resolved["config_path"] = str(path.resolve())
    return resolved


def ensure_config(config):
    if isinstance(config, (str, Path)):
        return load_config(config)
    value = dict(config)
    claimed = value.pop("config_sha256", None)
    value.pop("config_path", None)
    _validate(value)
    actual = sha256_json(value)
    if claimed is not None and claimed != actual:
        raise ValueError("config SHA256 mismatch")
    value["config_sha256"] = actual
    return value
