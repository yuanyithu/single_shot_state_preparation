import json
from pathlib import Path

from . import PHYSICS_VERSION, PT_VERSION, SCAN_VERSION
from .io import sha256_json


REQUIRED_VERSIONS = {
    "physics_contract_version": PHYSICS_VERSION,
    "pt_contract_version": PT_VERSION,
    "scan_contract_version": SCAN_VERSION,
}

EXPECTED_LADDER_CANDIDATES = [
    *({"p_hot": p_hot, "num_temperatures": temperatures}
      for p_hot in (0.45, 0.475)
      for temperatures in (8, 12, 16, 24, 32, 48, 64)),
    *({"p_hot": 0.49, "num_temperatures": temperatures}
      for temperatures in (8, 12, 16, 24, 32, 48, 64, 96, 128)),
]
EXPECTED_GAMMA_CANDIDATES = [0.75, 1.0, 1.5]
EXPECTED_ROUND_CANDIDATES = [
    [500, 2000], [1000, 4000], [2000, 8000], [4000, 16000], [8000, 32000],
]
EXPECTED_PRODUCTION_GATE = {
    "min_swap_rate": 0.05,
    "min_swap_accepts": 20,
    "min_round_trips": 4,
    "min_sector_changing_round_trips": 2,
    "min_hot_logical_rate": 0.01,
    "min_hot_logical_accepts_per_basis": 20,
    "max_rhat": 1.05,
    "min_ess": 200,
    "max_instance_mean_spread": 0.10,
    "paired_audit_max_abs_z": 5.0,
}
EXPECTED_TOP_LEVEL_FIELDS = {
    *REQUIRED_VERSIONS,
    "ensemble", "sector", "engine", "q", "m_values", "codes_per_m",
    "num_disorders", "num_instances", "p_values", "pilot", "production_gate",
}
PILOT_CANDIDATE_FIELDS = {
    "p_hot", "num_temperatures", "gamma", "burn_rounds", "measurement_rounds",
    "sweeps_per_round", "logical_move_repeat",
}


def load_config(path):
    with open(path, encoding="utf-8") as handle:
        config = json.load(handle)
    if set(config) != EXPECTED_TOP_LEVEL_FIELDS:
        raise ValueError("unexpected production config fields")
    for key, expected in REQUIRED_VERSIONS.items():
        if config.get(key) != expected:
            raise ValueError(f"{key} must be {expected!r}")
    if config.get("q") != 0:
        raise ValueError("exp102 permits q=0 only")
    if config.get("ensemble") != "true_posterior" or config.get("sector") != "x_error":
        raise ValueError("production identity requires true_posterior/x_error")
    if config.get("engine") != "numba":
        raise ValueError("exp102 production requires engine=numba")
    if config.get("num_instances") != 4 or config.get("num_disorders") != 128:
        raise ValueError("production requires four PT instances and 128 disorders")
    if config.get("m_values") != [3, 4, 5, 6, 7, 8] or config.get("codes_per_m") != 8:
        raise ValueError("production requires m=3..8 and eight codes per m")
    if config.get("p_values") != [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]:
        raise ValueError("unexpected production p grid")
    pilot = config.get("pilot")
    if not isinstance(pilot, dict) or set(pilot) != {
            "ladder_candidates", "gamma_candidates", "round_candidates"}:
        raise ValueError("unexpected pilot schedule fields")
    if pilot["ladder_candidates"] != EXPECTED_LADDER_CANDIDATES:
        raise ValueError("unexpected ordered pilot ladder candidates")
    if pilot["gamma_candidates"] != EXPECTED_GAMMA_CANDIDATES:
        raise ValueError("unexpected pilot gamma candidates")
    if pilot["round_candidates"] != EXPECTED_ROUND_CANDIDATES:
        raise ValueError("unexpected pilot round candidates")
    if config.get("production_gate") != EXPECTED_PRODUCTION_GATE:
        raise ValueError("unexpected production gate")
    resolved = dict(config)
    resolved["config_sha256"] = sha256_json(config)
    resolved["config_path"] = str(Path(path).resolve())
    return resolved


def validate_pilot_candidate(candidate, config):
    if set(candidate) != PILOT_CANDIDATE_FIELDS:
        raise ValueError("pilot candidate fields are incomplete")
    normalized = {
        "p_hot": float(candidate["p_hot"]),
        "num_temperatures": int(candidate["num_temperatures"]),
        "gamma": float(candidate["gamma"]),
        "burn_rounds": int(candidate["burn_rounds"]),
        "measurement_rounds": int(candidate["measurement_rounds"]),
        "sweeps_per_round": int(candidate["sweeps_per_round"]),
        "logical_move_repeat": int(candidate["logical_move_repeat"]),
    }
    pilot = config["pilot"]
    ladder_identity = {
        "p_hot": normalized["p_hot"],
        "num_temperatures": normalized["num_temperatures"],
    }
    if ladder_identity not in pilot["ladder_candidates"]:
        raise ValueError("pilot ladder pair is outside the frozen schedule")
    if normalized["gamma"] not in pilot["gamma_candidates"]:
        raise ValueError("pilot candidate gamma is outside the frozen schedule")
    if [normalized["burn_rounds"], normalized["measurement_rounds"]] not in pilot["round_candidates"]:
        raise ValueError("pilot candidate round budget is outside the frozen schedule")
    if normalized["sweeps_per_round"] != 1 or normalized["logical_move_repeat"] != 1:
        raise ValueError("pilot candidate move counts differ from the frozen schedule")
    return normalized
