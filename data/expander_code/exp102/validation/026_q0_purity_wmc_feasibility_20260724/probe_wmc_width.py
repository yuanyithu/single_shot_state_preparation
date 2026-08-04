"""Bounded feasibility probe for a direct q_top weighted-model count.

For a q=0 hard syndrome y and b=p/(1-p), the desired sector purity can be
written without a Markov chain as C/Z**2, where Z is a one-copy weighted count
and C is a two-copy weighted count constrained to equal logical labels.  This
script does not approximate either count.  It only asks whether the existing
strict exact-factor-elimination engine reaches a manageable induced width on
the m8 hard sentinel before any numerical counting is attempted.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

if __package__ in (None, ""):
    PROJECT_ROOT = Path(__file__).resolve().parents[5]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from data.expander_code.exp102.exp102_pipeline.io import atomic_json, sha256_file, sha256_json
from data.expander_code.exp102.exp102_pipeline.registry import load_frozen_code, load_registry
from data.expander_code.exp102.exp102_pipeline.seeds import derive_seed
from data.expander_code.exp102.exp102_pipeline.worker import build_model


PROBE_VERSION = "exp102.q0_purity_wmc_feasibility.v0"
ROOT = Path(__file__).resolve().parent
EXP102_ROOT = ROOT.parents[1]
REGISTRY_PATH = EXP102_ROOT / "registry" / "registry.json"
WMC_SOURCE = (
    EXP102_ROOT / "validation" / "007_q0_global_discovery_20260721" / "wmc_feasibility.py"
)
CAPS = (20, 24, 32, 40, 48, 64)


class ProbeError(RuntimeError):
    pass


def _require(condition, message):
    if not condition:
        raise ProbeError(message)


def _load_wmc_module():
    spec = importlib.util.spec_from_file_location("exp102_wmc_probe_engine", WMC_SOURCE)
    _require(spec is not None and spec.loader is not None, "cannot load strict WMC engine")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _context():
    registry = load_registry(REGISTRY_PATH)
    _, code, H = load_frozen_code(REGISTRY_PATH, "m08_c06")
    model, frame = build_model(H)
    uniform_seed = derive_seed(
        "pilot_ladder_m8_attempt22", registry["registry_sha256"],
        code["code_id"], 0, "uniforms",
    )
    epsilon = (
        np.random.Generator(np.random.PCG64(uniform_seed)).random(model.num_qubits)
        < 0.04
    ).astype(np.uint8)
    syndrome = (
        model.H_check.astype(np.int64) @ epsilon.astype(np.int64) % 2
    ).astype(np.uint8)
    _require(syndrome.any(), "m8 hard syndrome unexpectedly vanished")
    return registry, code, model, frame, epsilon, syndrome


def _source_binding():
    source_commit = subprocess.run(
        ("git", "rev-parse", "HEAD"), check=True, capture_output=True, text=True,
    ).stdout.strip()
    files = {
        "probe": sha256_file(Path(__file__)),
        "wmc_engine": sha256_file(WMC_SOURCE),
        "registry": sha256_file(REGISTRY_PATH),
    }
    core = {"source_commit": source_commit, "files": files}
    return {**core, "source_binding_sha256": sha256_json(core)}


def _width_records(wmc, factors, variable_count, caps):
    records = []
    for cap in caps:
        started = time.monotonic()
        try:
            _, width = wmc.min_degree_order(
                factors, variable_count, int(cap), time.monotonic() + 45.0,
            )
            records.append({
                "max_exact_width": int(cap), "status": "COMPLETE",
                "induced_width": int(width), "wall_seconds": time.monotonic() - started,
            })
        except wmc.WidthLimit as exc:
            width, remaining, _ = exc.args[0]
            records.append({
                "max_exact_width": int(cap), "status": "INCONCLUSIVE_WIDTH",
                "first_exceeded_width": int(width),
                "remaining_variables": None if remaining is None else int(remaining),
                "wall_seconds": time.monotonic() - started,
            })
        except wmc.SolverTimeout:
            records.append({
                "max_exact_width": int(cap), "status": "INCONCLUSIVE_TIMEOUT",
                "wall_seconds": time.monotonic() - started,
            })
    return records


def run_probe():
    wmc = _load_wmc_module()
    registry, code, model, frame, epsilon, syndrome = _context()
    object.__setattr__(model, "_wmc_frame_W", np.ascontiguousarray(frame.W_basis, dtype=np.uint8))
    try:
        cases = []
        for name, replicas, logical_collision in (
            ("Z_single", 1, False),
            ("C_same_logical_label_double", 2, True),
        ):
            factors, variable_count = wmc.posterior_factors(
                model, syndrome, 0.04, replicas=replicas,
                logical_collision=logical_collision,
            )
            cases.append({
                "name": name,
                "replicas": replicas,
                "logical_collision_constraint": logical_collision,
                "encoded_variable_count": int(variable_count),
                "factor_count": int(len(factors)),
                "maximum_initial_factor_arity": int(max(len(factor.scope) for factor in factors)),
                "caps": _width_records(wmc, factors, variable_count, CAPS),
            })
    finally:
        object.__delattr__(model, "_wmc_frame_W")
    return {
        "cell": {
            "code_id": code["code_id"], "p": 0.04, "disorder_index": 0,
            "disorder_source": "attempt022",
        },
        "registry_sha256": registry["registry_sha256"],
        "physical_dimensions": {
            "single_qubits": int(model.num_qubits),
            "hard_coset_dimension": int(model.num_qubits - model.H_check.shape[0]),
            "logical_dimension": int(model.k),
            "planted_weight_diagnostic_only": int(epsilon.sum()),
            "syndrome_weight": int(syndrome.sum()),
        },
        "identity": {
            "Z": "sum_{H e=y} b**|e|",
            "C": "sum_{H e1=H e2=y,W(e1 xor e2)=0} b**(|e1|+|e2|)",
            "purity": "C/Z**2",
            "q_top": "(2**k * C/Z**2 - 1)/(2**k - 1)",
        },
        "cases": cases,
        "interpretation": {
            "result": "NO_NUMERICAL_COUNT_ATTEMPTED",
            "meaning": (
                "The listed failures only rule out this exact ternary-XOR "
                "factor-elimination encoding at the tested width caps."
            ),
            "does_not_rule_out": [
                "A different exact contraction or certified branch-and-bound.",
                "A rigorous collapsed-B tail-bound construction.",
                "The q=0 posterior or any physical parameter point.",
            ],
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "wmc_width_probe.json")
    args = parser.parse_args(argv)
    core = {
        "probe_version": PROBE_VERSION,
        "status": "EXPLORATORY_WMC_WIDTH_ONLY_NO_QTOP_OR_READINESS_CLAIM",
        "scope": {
            "posterior_estimation": False,
            "formal_authorization": False,
            "remote_authorization": False,
            "production_authorization": False,
        },
        "source_binding": _source_binding(),
        "probe": run_probe(),
    }
    report = {**core, "report_sha256": sha256_json(core)}
    atomic_json(args.output, report)
    print(report["report_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
