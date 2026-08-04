"""Parallel, bit-exact replay validation for the frozen UARE V0 raw screen.

The bound V1 runner has a post-replay summary defect.  This validator leaves
that runner untouched, imports it from its immutable path, and uses its raw
validator plus replay routine before writing a distinct V2 evidence file.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import sys
import traceback
from pathlib import Path

import numpy as np

from data.expander_code.exp102.exp102_pipeline.io import atomic_json, sha256_file, sha256_json


REPLAY_VERSION = "exp102.q0_hgp_uniform_anchor_pt.replay_validate.v2"
ROOT = Path(__file__).resolve().parent
FROZEN_RUNNER = ROOT / "run_local_viability.py"
_WORKER = None


class ReplayValidationConflict(RuntimeError):
    pass


def _require(condition, message):
    if not condition:
        raise ReplayValidationConflict(message)


def _load_frozen_runner():
    spec = importlib.util.spec_from_file_location("exp102_uare_frozen_runner_v1", FROZEN_RUNNER)
    _require(spec is not None and spec.loader is not None, "cannot load frozen UARE runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_run_complete(path, manifest):
    try:
        complete = json.loads(path.read_text(encoding="ascii"))
    except Exception as exc:
        raise ReplayValidationConflict(f"cannot read RUN_COMPLETE.json: {exc}") from exc
    core = {key: value for key, value in complete.items() if key != "run_sha256"}
    _require(complete.get("run_sha256") == sha256_json(core), "RUN_COMPLETE hash mismatch")
    _require(complete.get("manifest_sha256") == manifest["manifest_sha256"], "RUN_COMPLETE manifest mismatch")
    _require(complete.get("raw_count") == len(manifest["tasks"]), "RUN_COMPLETE raw count mismatch")
    names = [item.get("filename") for item in complete.get("raw", [])]
    _require(len(names) == len(manifest["tasks"]) and len(set(names)) == len(names),
             "RUN_COMPLETE entries are incomplete")
    return complete


def _load_state(manifest_path):
    runner = _load_frozen_runner()
    manifest_path = Path(manifest_path)
    output_root = manifest_path.parent
    manifest = runner._load_manifest(manifest_path)
    _require((output_root / "RUN_COMPLETE.json").is_file(), "frozen run is incomplete")
    _load_run_complete(output_root / "RUN_COMPLETE.json", manifest)
    with np.load(output_root / "CONTROL.npz", allow_pickle=False) as archive:
        frozen_l_move = archive["l_move"].copy()
    context = runner._context(frozen_l_move=frozen_l_move, frozen_l_metadata=manifest["l_start"])
    runner._validate_manifest_context(manifest, context, output_root / "CONTROL.npz")
    control = runner._load_control(output_root / "CONTROL.npz", context, manifest)
    return runner, manifest, context, control, output_root


def _init_worker(manifest_path):
    global _WORKER
    _WORKER = _load_state(manifest_path)


def _replay_task(task):
    _require(_WORKER is not None, "replay worker was not initialized")
    runner, manifest, context, control, output_root = _WORKER
    path = runner._task_output_path(output_root, task)
    _require(path.is_file(), f"missing raw task: {path.name}")
    raw = runner._validate_one_raw(path, context, manifest, control, task)
    runner._replay_one(context, manifest, control, task, raw)
    return path.name, sha256_file(path)


def replay(manifest_path, workers):
    manifest_path = Path(manifest_path)
    _require(workers > 0, "workers must be positive")
    runner, manifest, _, _, output_root = _load_state(manifest_path)
    del runner
    report_path = output_root / "REPLAY_V2.json"
    failed_path = output_root / "REPLAY_V2_FAILED.json"
    _require(not report_path.exists() and not failed_path.exists(), "V2 replay artifact already exists")
    try:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers, initializer=_init_worker, initargs=(str(manifest_path),)
        ) as executor:
            results = list(executor.map(_replay_task, manifest["tasks"]))
        results.sort(key=lambda item: item[0])
        raw_sha256 = {name: digest for name, digest in results}
        _require(len(raw_sha256) == len(manifest["tasks"]), "replay task result count mismatch")
        core = {
            "replay_version": REPLAY_VERSION,
            "manifest_sha256": manifest["manifest_sha256"],
            "source_binding_sha256": manifest["source_binding"]["source_binding_sha256"],
            "frozen_runner_sha256": sha256_file(FROZEN_RUNNER),
            "validator_source_sha256": sha256_file(Path(__file__)),
            "raw_sha256": raw_sha256,
            "task_count": len(results),
            "workers": int(workers),
            "all_bit_identical": True,
        }
        report = {**core, "replay_sha256": sha256_json(core)}
        atomic_json(report_path, report)
        return report
    except Exception as exc:
        core = {
            "replay_version": REPLAY_VERSION,
            "manifest_path": str(manifest_path),
            "validator_source_sha256": sha256_file(Path(__file__)),
            "failure": repr(exc),
            "traceback": traceback.format_exc(),
        }
        atomic_json(failed_path, {**core, "failure_sha256": sha256_json(core)})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "local_hard_viability" / "MANIFEST.json")
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args(argv)
    result = replay(args.manifest, args.workers)
    print(result["replay_sha256"])


if __name__ == "__main__":
    main()
