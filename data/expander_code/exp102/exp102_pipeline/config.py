import json
from pathlib import Path

from . import PHYSICS_VERSION, PT_VERSION, SCAN_VERSION
from .io import sha256_json


REQUIRED_VERSIONS = {
    "physics_contract_version": PHYSICS_VERSION,
    "pt_contract_version": PT_VERSION,
    "scan_contract_version": SCAN_VERSION,
}


def load_config(path):
    with open(path, encoding="utf-8") as handle:
        config = json.load(handle)
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
    if config.get("p_values") != [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]:
        raise ValueError("unexpected production p grid")
    resolved = dict(config)
    resolved["config_sha256"] = sha256_json(config)
    resolved["config_path"] = str(Path(path).resolve())
    return resolved
