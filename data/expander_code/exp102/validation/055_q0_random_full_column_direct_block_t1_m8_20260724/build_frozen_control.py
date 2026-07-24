"""Build the fresh direct-block T1 control without changing frozen v0 code."""

from __future__ import annotations

from importlib import import_module
import json
from pathlib import Path


CONTRACT_VERSION = "exp102.q0_random_full_column_direct_block.t1_m8.v1"
CONTROL_VERSION = "exp102.q0_random_full_column_direct_block.t1_m8.control.v1"


def _load_config(path):
    legacy = import_module(
        "data.expander_code.exp102.validation."
        "052_q0_random_full_column_t1_m8_20260724.build_frozen_control"
    )
    path = Path(path).resolve()
    serialized = path.read_text(encoding="ascii")
    config = json.loads(serialized)
    legacy._require(
        serialized == legacy.canonical_json(config) + "\n",
        "config is not canonical JSON",
    )
    legacy._require(
        config["version"] == config["contract_version"] == CONTRACT_VERSION,
        "direct-block T1 contract version changed",
    )
    return config, legacy.sha256_file(path)


def main():
    legacy = import_module(
        "data.expander_code.exp102.validation."
        "052_q0_random_full_column_t1_m8_20260724.build_frozen_control"
    )
    legacy.CONTROL_VERSION = CONTROL_VERSION
    legacy._load_config = _load_config
    legacy.main()


if __name__ == "__main__":
    main()
