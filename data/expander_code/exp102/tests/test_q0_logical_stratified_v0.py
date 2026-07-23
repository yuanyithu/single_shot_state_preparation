import json

import pytest

from data.expander_code.exp102.exp102_pipeline.io import canonical_json
from data.expander_code.exp102.exp102_pipeline.q0_logical_stratified_v0 import (
    LogicalStratifiedV0ConflictError,
    V0_CONTRACT_VERSION,
    _portable_raw_sha,
    load_v0_config,
)
from data.expander_code.exp102.exp102_pipeline.registry import load_registry


def _config_path():
    return "data/expander_code/exp102/config/q0_logical_stratified.v0.json"


def test_v0_config_is_narrow_diagnostic_contract():
    registry = load_registry("data/expander_code/exp102/registry/registry.json")
    config = load_v0_config(_config_path(), registry=registry)
    assert config["contract_version"] == V0_CONTRACT_VERSION
    assert config["cell"]["code_id"] == "m08_c06"
    assert config["cell"]["p"] == pytest.approx(0.04)
    assert config["scope"]["formal_authorization"] is False
    assert config["scope"]["production_authorization"] is False
    assert [item["alpha_temperature"] for item in config["candidates"]] == [0.5, 1.0]


def test_v0_config_rejects_scope_or_gate_edits(tmp_path):
    registry = load_registry("data/expander_code/exp102/registry/registry.json")
    original = json.loads(open(_config_path(), encoding="ascii").read())
    original["scope"]["formal_authorization"] = True
    path = tmp_path / "bad.json"
    path.write_text(canonical_json(original) + "\n", encoding="ascii")
    with pytest.raises(LogicalStratifiedV0ConflictError, match="schedule/gates/scope"):
        load_v0_config(path, registry=registry)


def test_portable_digest_excludes_only_timing_fields():
    first = {
        "labels": [1, 2], "core_seconds": 1.0, "wall_seconds": 2.0,
    }
    second = {
        "labels": [1, 2], "core_seconds": 9.0, "wall_seconds": 20.0,
    }
    changed = {
        "labels": [1, 3], "core_seconds": 1.0, "wall_seconds": 2.0,
    }
    assert _portable_raw_sha(first) == _portable_raw_sha(second)
    assert _portable_raw_sha(first) != _portable_raw_sha(changed)
