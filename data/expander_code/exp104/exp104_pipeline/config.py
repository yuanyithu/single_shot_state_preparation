import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

from . import CONFIG_SCHEMA, EXPERIMENT_ID
from .io import sha256_json


# The measurement grid stops at 0.10. exp103 spent roughly 60 percent of its
# compute on p >= 0.10, where every code fails essentially always, the
# between-code standard deviation is below 0.001, and belief propagation always
# exhausts max_iter = n. Truncating there costs no resolvable information.
P_TOKENS = [f"0.{value:02d}" for value in range(2, 11)]
M_VALUES = list(range(3, 9))
CODES_PER_M = 2000
TRIALS_PER_CODE_P = 4
# One task loads its code models once and sweeps the whole p grid, so the block
# sizes are chosen to divide CODES_PER_M and to keep every task near seven
# minutes on nd-3 given the measured per-code cost.
CODES_PER_TASK = {3: 250, 4: 100, 5: 40, 6: 20, 7: 10, 8: 5}
TASKS_PER_M = {m: CODES_PER_M // CODES_PER_TASK[m] for m in M_VALUES}

MASTER_SEED_HEX = "a39f90001b56ed7ac70334ad61cda086bac08b1c360cb9a5274487eff0de6b6b"
# Ensemble definition. Codes are generated from seeds and accepted on algebraic
# structure alone; nothing is ever selected on decoder performance. The
# acceptance rate is size dependent (0.738 at m=3 rising to 0.993 at m=8) and
# that is part of the definition, not a correction applied afterwards.
ENSEMBLE = {
    "graph_family": "random_biregular_bipartite_simple",
    "d_A": 3,
    "d_B": 4,
    "construction": "configuration_model_reject_multi_edge",
    "max_attempts": 10000,
    "acceptance_rule": "full_row_rank_and_unique_H",
    "post_selection": "none",
}
# Byte-identical to the frozen exp103.decoder_mc.v2 identity so that the 624
# exp103 cells stay comparable evidence. max_iter = n dominates the cost but
# changing it would change the physics being measured.
DECODER = {
    "bp_method": "product_sum",
    "max_iter": "n",
    "schedule": "serial",
    "serial_schedule_order": "natural",
    "osd_method": "osd_0",
    "osd_order": 0,
    "omp_thread_count": 1,
}
DECODER_BINARY_SHA256 = "944a96a657a89fbd04c127edb2eba1033f56de0161ddcd2ba7e57dee76777ccc"
DECODER_BINARY_SUFFIX = ".cpython-312-darwin.so"
NAMESPACES = {
    "ensemble": "exp104.ensemble_mc.ensemble.v1",
    "benchmark": "exp104.ensemble_mc.benchmark.v1",
    "measurement": "exp104.ensemble_mc.measurement.v1",
    "replay": "exp104.ensemble_mc.replay.v1",
    "bootstrap": "exp104.ensemble_mc.bootstrap.v1",
}
REGISTRY_PATH = "data/expander_code/exp104/config/ensemble_registry.v1.json"
CROSSING = {
    "primary_contrast": "m08_minus_m03",
    "direction": "negative_at_lower_p_positive_at_higher_p",
    # exp103's rule demanded a certified sign change at adjacent grid points,
    # which forces certification of a contrast that vanishes at the crossing.
    # exp104 certifies a negative point and a later positive point and reports
    # the bracket between them.
    "bracket_requires_adjacent_grid_points": False,
    "simultaneous_band_scope": "primary_contrast_grid_only",
    "location_estimator": "linear_interpolation_first_sign_change_in_bracket",
    "adjacent_contrasts": "diagnostic_pointwise_only",
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
}
# Every m is benchmarked directly rather than extrapolated from an anchor, so
# the projected cost is a bound the gate can actually discriminate with.
PREFLIGHT = {
    "code_indices": [0, 1],
    "m_values": [3, 4, 5, 6, 7, 8],
    "p_tokens": ["0.02", "0.06", "0.10"],
    "num_workers": 8,
    "reserve_multiplier": 2.0,
    "stage_core_hour_cap": 100.0,
    "stage_wall_hour_cap": 12.0,
    "peak_rss_gib_cap": 12.0,
    "analysis_core_hours": 1.0,
    "fixed_overhead_core_hours": 1.0,
}

TOP_LEVEL_FIELDS = {
    "schema_version", "experiment_id", "objective", "registry_path",
    "registry_sha256", "source_commit", "source_tree_sha256",
    "master_seed_hex", "m_values", "codes_per_m", "p_tokens",
    "trials_per_code_p", "codes_per_task", "ensemble", "decoder",
    "environment", "decoder_binary", "namespaces", "bootstrap", "replay",
    "preflight", "crossing",
}
REMOTE_CONFIG_SCHEMA = "exp104.config.remote.v1"
REMOTE_CONFIG_PATH = "data/expander_code/exp104/config/ensemble_mc.remote.v1.json"
REMOTE_EXECUTION_PROFILE = "exp104.remote_execution.v1"
REMOTE_CONDA_PREFIX = "/home/DATA1/users/yuany/.single_shot/cache/exp104_remote_v1_env"
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
# Identical upstream release to exp103, so the compiled decoder is the same
# object on the same node.
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


def normalize_p_token(value):
    if isinstance(value, str):
        token = value
    else:
        try:
            numeric = Decimal(str(value))
        except (InvalidOperation, ValueError):
            raise ValueError(f"p is outside the frozen grid: {value!r}") from None
        matches = [token for token in P_TOKENS if Decimal(token) == numeric]
        if len(matches) != 1:
            raise ValueError(f"p is outside the frozen grid: {value!r}")
        token = matches[0]
    if token not in P_TOKENS:
        raise ValueError(f"p is outside the frozen grid: {value!r}")
    return token


def code_id(m, index):
    if m not in M_VALUES:
        raise ValueError(f"m is outside the frozen panel: {m!r}")
    if isinstance(index, bool) or not isinstance(index, int):
        raise ValueError("code index must be an integer")
    if not 0 <= index < CODES_PER_M:
        raise ValueError(f"code index is outside the frozen panel: {index!r}")
    return f"m{int(m):02d}_c{int(index):05d}"


def parse_code_id(value):
    match = re.fullmatch(r"m(\d{2})_c(\d{5})", str(value))
    if match is None:
        raise ValueError(f"malformed exp104 code id: {value!r}")
    m = int(match.group(1))
    index = int(match.group(2))
    if m not in M_VALUES or not 0 <= index < CODES_PER_M:
        raise ValueError(f"code id is outside the frozen panel: {value!r}")
    return m, index


def block_code_indices(m, block_index):
    """Return the frozen contiguous code index range owned by one task."""
    if m not in M_VALUES:
        raise ValueError(f"m is outside the frozen panel: {m!r}")
    if isinstance(block_index, bool) or not isinstance(block_index, int):
        raise ValueError("block index must be an integer")
    if not 0 <= block_index < TASKS_PER_M[m]:
        raise ValueError(f"block index is outside the frozen plan: {block_index!r}")
    size = CODES_PER_TASK[m]
    start = block_index * size
    return list(range(start, start + size))


def _validate_remote_execution(config):
    profile = config["execution_profile"]
    if not isinstance(profile, dict) or set(profile) != REMOTE_EXECUTION_FIELDS:
        raise ValueError("unexpected exp104 remote execution profile fields")
    expected_fixed = {
        "profile_id": REMOTE_EXECUTION_PROFILE,
        "entry_host": "yuany",
        "compute_host": "nd-3",
        "num_workers": 64,
        "omp_thread_count": 1,
        "run_root": "~/.single_shot/runs",
        "log_root": "~/.single_shot/logs",
        "reserve_multiplier": 2.0,
        "stage_core_hour_cap": 900.0,
        "stage_wall_hour_cap": 16.0,
        "peak_rss_gib_cap": 128.0,
    }
    for field, expected in expected_fixed.items():
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
    remote = schema == REMOTE_CONFIG_SCHEMA
    expected_fields = REMOTE_TOP_LEVEL_FIELDS if remote else TOP_LEVEL_FIELDS
    if set(config) != expected_fields:
        raise ValueError("unexpected exp104 config fields")
    if schema not in {CONFIG_SCHEMA, REMOTE_CONFIG_SCHEMA} or config["experiment_id"] != EXPERIMENT_ID:
        raise ValueError("exp104 config identity mismatch")
    if config["objective"] != "bposd_ensemble_block_logical_failure_crossing_q0":
        raise ValueError("unexpected exp104 objective")
    if config["registry_path"] != REGISTRY_PATH:
        raise ValueError("exp104 must use the canonical ensemble registry")
    if config["m_values"] != M_VALUES or config["codes_per_m"] != CODES_PER_M:
        raise ValueError("primary panel must retain the full frozen ensemble")
    if config["p_tokens"] != P_TOKENS:
        raise ValueError("unexpected p grid")
    if config["trials_per_code_p"] != TRIALS_PER_CODE_P:
        raise ValueError("trials per code-p is not the frozen allocation")
    if config["codes_per_task"] != {str(m): CODES_PER_TASK[m] for m in M_VALUES}:
        raise ValueError("task blocking is not frozen")
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
        raise ValueError("exp104 master seed differs from the fresh frozen seed")
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


def load_config(path):
    path = Path(path)
    raw = json.loads(path.read_text(encoding="ascii"))
    filename = (
        "ensemble_mc.remote.v1.json"
        if raw.get("schema_version") == REMOTE_CONFIG_SCHEMA
        else "ensemble_mc.v1.json"
    )
    canonical_path = Path(__file__).resolve().parents[1] / "config" / filename
    if path.resolve() != canonical_path.resolve():
        raise ValueError("formal exp104 config must be the canonical config artifact")
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
