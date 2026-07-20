import json
import os
from pathlib import Path

import numba
import numpy as np

from data.expander_code.exp102.exp102_pipeline.config import load_config
from data.expander_code.exp102.exp102_pipeline.io import sha256_json, verify_source_identity
from data.expander_code.exp102.exp102_pipeline.registry import load_frozen_code, load_registry


source_identity = verify_source_identity(Path.cwd(), os.environ.get("EXP102_SOURCE_COMMIT", ""))
registry = load_registry("data/expander_code/exp102/registry/registry.json")
config = load_config("data/expander_code/exp102/config/production.v1.json")
with open("data/expander_code/exp102/config/pilot_schedule.v1.json", encoding="ascii") as handle:
    schedule = json.load(handle)
expected_registry = "883730e0ba548f6b358187d8f123fdd4d8aeb116f4bacda363c35c16d01ae40b"
expected_config = "96b5957fb3f1f0fb520b5f635eb3424f2aa93c90c471e5f1d20013f1b76a7330"
expected_extended_ladders = [
    {"p_hot": 0.49, "num_temperatures": 96},
    {"p_hot": 0.49, "num_temperatures": 128},
]
checks = (
    (registry["registry_sha256"] == expected_registry, "registry SHA256 mismatch"),
    (len(registry["codes"]) == 48, "registry must contain exactly 48 codes"),
    (config["config_sha256"] == expected_config, "config SHA256 mismatch"),
    (config["engine"] == "numba", "production engine must be numba"),
    (len(config["pilot"]["ladder_candidates"]) == 23, "pilot ladder count mismatch"),
    (config["pilot"]["ladder_candidates"][-2:] == expected_extended_ladders,
     "R96/R128 ladder suffix mismatch"),
    (schedule["registry_sha256"] == registry["registry_sha256"],
     "schedule registry identity mismatch"),
    (schedule["config_sha256"] == config["config_sha256"],
     "schedule config identity mismatch"),
    (schedule["selection_policy"]
     == "ordered_ladder_pairs_then_min_core_time_gamma_then_raise_round_budget",
     "schedule selection policy mismatch"),
    (len(schedule["candidates"]) == 345, "schedule candidate count mismatch"),
    (len(schedule["tuning_tasks"]) == 576, "schedule tuning task count mismatch"),
    (len(schedule["held_out_tasks"]) == 2688, "schedule held-out task count mismatch"),
)
for passed, message in checks:
    if not passed:
        raise RuntimeError(message)
for code in registry["codes"]:
    load_frozen_code("data/expander_code/exp102/registry/registry.json", code["code_id"])
print(json.dumps({
    "numpy": np.__version__, "numba": numba.__version__,
    "registry": registry["registry_sha256"], "config": config["config_sha256"],
    "schedule": sha256_json(schedule), "seed_rebuilt_codes": len(registry["codes"]),
    "source": source_identity,
}, sort_keys=True))
