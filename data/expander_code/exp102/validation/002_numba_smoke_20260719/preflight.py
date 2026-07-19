import json

import numba
import numpy as np

from data.expander_code.exp102.exp102_pipeline.config import load_config
from data.expander_code.exp102.exp102_pipeline.registry import load_registry


registry = load_registry("data/expander_code/exp102/registry/registry.json")
config = load_config("data/expander_code/exp102/config/production.v1.json")
assert registry["registry_sha256"] == "883730e0ba548f6b358187d8f123fdd4d8aeb116f4bacda363c35c16d01ae40b"
assert len(registry["codes"]) == 48
assert config["config_sha256"] == "758e2804476c5cb0422ef5813952a3779722c1a3ed47a7298f3948f9daee241f"
assert config["engine"] == "numba"
print(json.dumps({"numpy": np.__version__, "numba": numba.__version__, "registry": registry["registry_sha256"], "config": config["config_sha256"]}, sort_keys=True))
