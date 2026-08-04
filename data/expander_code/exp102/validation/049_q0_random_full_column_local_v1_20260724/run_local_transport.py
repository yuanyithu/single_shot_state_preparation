"""Fresh V1 wrapper for the validation-048 core with an exact Numba mass table."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.expander_code.exp102.exp102_pipeline.io import canonical_json, sha256_file, sha256_json


ROOT = Path(__file__).resolve().parent
EXP102_ROOT = ROOT.parents[1]
CORE_PATH = EXP102_ROOT / "validation/048_q0_random_full_column_local_v0_20260724/run_local_transport.py"
CONTRACT_VERSION = "exp102.q0_random_full_column.local.v1"
CONFIG_VERSION = "exp102.q0_random_full_column.local.config.v1"
SEED_NAMESPACE = "exp102.q0_random_full_column.local.v1.20260724"


spec = importlib.util.spec_from_file_location("exp102_random_full_column_v0_core", CORE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load frozen validation-048 core")
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


def _require(condition, message):
    if not condition:
        raise core.LocalTransportError(message)


def _load_config(path):
    path = Path(path).resolve()
    serialized = path.read_text(encoding="ascii")
    config = json.loads(serialized)
    _require(serialized == canonical_json(config) + "\n", "V1 config is not canonical")
    _require(set(config) == {
        "cell", "config_version", "contract_version", "gates", "initialization",
        "mass_engine", "registry_sha256", "resource", "scope", "seed_namespace",
        "version",
    }, "V1 config schema changed")
    _require(config["version"] == config["contract_version"] == CONTRACT_VERSION
             and config["config_version"] == CONFIG_VERSION,
             "V1 config version changed")
    _require(config["mass_engine"]
             == "numba_exact_replayed_against_reference_on_small_hgp",
             "V1 mass engine changed")
    reference = json.loads(
        (EXP102_ROOT / "config/q0_random_full_column.local.v0.json").read_text(
            encoding="ascii",
        )
    )
    for field in ("cell", "gates", "initialization", "registry_sha256", "resource", "scope"):
        _require(config[field] == reference[field], f"V1 scientific field changed: {field}")
    _require(config["seed_namespace"] == SEED_NAMESPACE, "V1 seed namespace changed")
    return config, sha256_file(path)


def _source_identity(config_path):
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"), check=True, capture_output=True, text=True,
    ).stdout.strip()
    paths = {
        "config": Path(config_path).resolve(),
        "core_runner": CORE_PATH,
        "kernel": EXP102_ROOT / "exp102_pipeline/q0_hgp_random_full_column.py",
        "full_column": EXP102_ROOT / "exp102_pipeline/q0_hgp_full_column_gibbs.py",
        "review": EXP102_ROOT / "reviews/RANDOM_FULL_COLUMN_REVIEW.md",
        "v1_runner": Path(__file__).resolve(),
    }
    value = {
        "files": {name: sha256_file(path) for name, path in paths.items()},
        "source_commit": commit,
    }
    return {**value, "source_identity_sha256": sha256_json(value)}


original_mass_builder = core.build_classical_coset_mass


def _numba_mass_builder(H, p, *, engine):
    _require(engine == "reference", "core requested an unexpected mass engine")
    return original_mass_builder(H, p, engine="numba")


core.CONTRACT_VERSION = CONTRACT_VERSION
core.REPORT_VERSION = "exp102.q0_random_full_column.local.report.v1"
core.RANDOM_FULL_COLUMN_RAW_VERSION = "exp102.q0_hgp_random_full_column.raw.v1"
core.ROOT = ROOT
core._load_config = _load_config
core._source_identity = _source_identity
core.build_classical_coset_mass = _numba_mass_builder


if __name__ == "__main__":
    core.main()
