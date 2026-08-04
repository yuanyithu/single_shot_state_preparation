import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

from . import CONFIG_SCHEMA, EXPERIMENT_ID
from .io import sha256_json


P_TOKENS = [f"0.{value:02d}" for value in range(2, 15)]
M_VALUES = list(range(3, 9))
CODE_IDS = [f"m{m:02d}_c{code:02d}" for m in M_VALUES for code in range(8)]
REGISTRY_PATH = "data/expander_code/exp102/registry/registry.json"
REGISTRY_SHA256 = "883730e0ba548f6b358187d8f123fdd4d8aeb116f4bacda363c35c16d01ae40b"
MASTER_SEED_HEX = "08e1c6bd6607989e9ffc4506d73a687d9776672c9191d24414f54503153c9cb7"
BPLSD_BINARY_SHA256 = "180dfe755fcd5cb3dbc37e3291a8e05a3e488fcb6bfceea283f8c6392450a592"
BPLSD_BINARY_SUFFIX = ".cpython-312-darwin.so"
DECODER = {
    "bp_method": "product_sum",
    "max_iter": "n",
    "schedule": "serial",
    "serial_schedule_order": "natural",
    "lsd_method": "LSD_CS",
    "lsd_order": 0,
    "bits_per_step": 1,
    "always_run_lsd": False,
    "omp_thread_count": 1,
}
NAMESPACES = {
    "benchmark": "exp103.decoder_mc.benchmark.v1",
    "measurement": "exp103.decoder_mc.measurement.v1",
    "replay": "exp103.decoder_mc.replay.v1",
    "bootstrap": "exp103.decoder_mc.bootstrap.v1",
}
TOP_LEVEL_FIELDS = {
    "schema_version", "experiment_id", "objective", "registry_path",
    "registry_sha256", "source_commit", "source_tree_sha256",
    "master_seed_hex", "m_values", "codes_per_m", "p_tokens",
    "shards_per_code_p", "trials_per_shard", "decoder", "environment",
    "bplsd_binary", "namespaces", "bootstrap", "preflight",
    "stage_m_values", "preregistered_point_masks", "crossing",
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


def _validate(config):
    if set(config) != TOP_LEVEL_FIELDS:
        raise ValueError("unexpected exp103 config fields")
    if config["schema_version"] != CONFIG_SCHEMA or config["experiment_id"] != EXPERIMENT_ID:
        raise ValueError("exp103 config identity mismatch")
    if config["objective"] != "bplsd_block_logical_failure_crossing_q0":
        raise ValueError("unexpected exp103 objective")
    if config["registry_path"] != REGISTRY_PATH or config["registry_sha256"] != REGISTRY_SHA256:
        raise ValueError("exp103 must use the canonical frozen 48-code registry")
    if config["m_values"] != M_VALUES or config["codes_per_m"] != 8:
        raise ValueError("primary panel must retain all 48 frozen codes")
    if config["p_tokens"] != P_TOKENS:
        raise ValueError("unexpected p grid")
    if config["shards_per_code_p"] != 4 or config["trials_per_shard"] != 2500:
        raise ValueError("each code-p must contain four 2500-trial shards")
    if config["decoder"] != DECODER:
        raise ValueError("decoder parameters differ from the frozen BpLSD identity")
    if config["namespaces"] != NAMESPACES or len(set(NAMESPACES.values())) != 4:
        raise ValueError("seed namespaces must be frozen and disjoint")
    if config["stage_m_values"] != {"stage1": [3, 4, 5], "stage2": [6, 7, 8]}:
        raise ValueError("stage size groups are not frozen")
    if config["preregistered_point_masks"] != {"full": [True] * len(P_TOKENS)}:
        raise ValueError("only the full p-grid publication mask is preregistered")
    if config["master_seed_hex"] != MASTER_SEED_HEX:
        raise ValueError("exp103 master seed differs from the fresh frozen seed")
    for field in ("registry_sha256", "source_tree_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", config[field]):
            raise ValueError(f"{field} must be a SHA256")
    if not re.fullmatch(r"[0-9a-f]{40}", config["source_commit"]):
        raise ValueError("source_commit must be a full lowercase Git SHA")
    env = config["environment"]
    if env != {
        "device_name": "macmini",
        "hostname": "ymini.local",
        "conda_environment": "12",
        "conda_prefix_matches_python": True,
        "python": "3.12.12", "numpy": "2.4.1", "scipy": "1.17.0", "ldpc": "2.4.1",
    }:
        raise ValueError("canonical environment identity differs from the frozen contract")
    if type(env["conda_prefix_matches_python"]) is not bool:
        raise ValueError("conda prefix attestation must be a boolean")
    binary = config["bplsd_binary"]
    if set(binary) != {"module", "filename_suffix", "sha256"}:
        raise ValueError("unexpected BpLSD binary identity fields")
    if binary["module"] != "ldpc.bplsd_decoder._bplsd_decoder":
        raise ValueError("BpLSD extension module mismatch")
    if binary["sha256"] != BPLSD_BINARY_SHA256 or binary["filename_suffix"] != BPLSD_BINARY_SUFFIX:
        raise ValueError("BpLSD extension binary differs from the frozen backend")
    bootstrap = config["bootstrap"]
    if bootstrap != {
        "replicates": 20000,
        "confidence": 0.95,
        "method": "grouped_code_curve_plus_parametric_binomial_max_abs",
    }:
        raise ValueError("bootstrap family is not frozen")
    preflight = config["preflight"]
    if preflight != {
        "code_ids": ["m03_c00", "m05_c00", "m08_c00"],
        "p_tokens": ["0.02", "0.08", "0.14"],
        "trials_per_task": 20,
        "num_workers": 8,
        "reserve_multiplier": 2.0,
        "stage_core_hour_cap": 100.0,
        "stage_wall_hour_cap": 12.0,
        "peak_rss_gib_cap": 12.0,
        "analysis_core_hours": 0.5,
        "fixed_overhead_core_hours": 0.5,
    }:
        raise ValueError("local resource preflight is not frozen")
    crossing = config["crossing"]
    if crossing != {
        "primary_contrast": "m08_minus_m03",
        "direction": "negative_at_lower_p_positive_at_higher_p",
        "bracket_requires_adjacent_grid_points": True,
        "triple_groups": [[3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8]],
        "linear_interpolation": "plot_only",
    }:
        raise ValueError("crossing decision family is not frozen")


def load_config(path):
    path = Path(path)
    canonical_path = Path(__file__).resolve().parents[1] / "config" / "decoder_mc.v1.json"
    if path.resolve() != canonical_path.resolve():
        raise ValueError("formal exp103 config must be the canonical config artifact")
    config = json.loads(path.read_text(encoding="ascii"))
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
