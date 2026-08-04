"""Parallel bit-exact replay for the immutable UASRE local raw set."""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
from pathlib import Path
import sys
import traceback

import numpy as np

if __package__ in (None, ""):
    _PROJECT_ROOT = Path(__file__).resolve().parents[5]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

from data.expander_code.exp102.exp102_pipeline.io import atomic_json, sha256_file, sha256_json


REPLAY_VERSION = "exp102.q0_hgp_aux_stabilizer_pt.replay_validate.v1"
ROOT = Path(__file__).resolve().parent
FROZEN_RUNNER = ROOT / "run_local_viability.py"
WORKER_STATE = None


class ReplayConflict(RuntimeError):
    pass


def _require(condition, message):
    if not condition:
        raise ReplayConflict(message)


def _load_frozen_runner():
    spec = importlib.util.spec_from_file_location("exp102_uasre_frozen_runner_v1", FROZEN_RUNNER)
    _require(spec is not None and spec.loader is not None, "cannot load frozen UASRE runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_state(manifest_path):
    runner = _load_frozen_runner()
    manifest_path = Path(manifest_path)
    root = manifest_path.parent
    manifest = runner._load_manifest(manifest_path)
    _require((root / "RUN_COMPLETE.json").is_file(), "run is incomplete")
    runner._validate_run_complete(root, manifest)
    with np.load(root / "CONTROL.npz", allow_pickle=False) as archive:
        frozen_l_move = archive["l_move"].copy()
    context = runner._context(frozen_l_move=frozen_l_move, frozen_l_metadata=manifest["l_start"])
    runner._validate_manifest_context(manifest, context, root / "CONTROL.npz")
    control = runner._load_control(root / "CONTROL.npz", context, manifest)
    return runner, manifest, context, control, root


def _init_worker(manifest_path):
    global WORKER_STATE
    WORKER_STATE = _load_state(manifest_path)


def _replay_task(task):
    _require(WORKER_STATE is not None, "replay worker is uninitialized")
    runner, manifest, context, control, root = WORKER_STATE
    path = runner._task_output_path(root, task)
    raw = runner._validate_one_raw(path, context, manifest, control, task)
    runner._replay_one(context, manifest, control, task, raw)
    return path.name, sha256_file(path)


def replay(manifest_path, workers):
    manifest_path = Path(manifest_path)
    _require(int(workers) > 0, "workers must be positive")
    runner, manifest, _, _, root = _load_state(manifest_path)
    del runner
    report_path = root / "REPLAY.json"
    failed_path = root / "REPLAY_FAILED.json"
    _require(not report_path.exists() and not failed_path.exists(), "replay already has a terminal marker")
    try:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=int(workers), initializer=_init_worker, initargs=(str(manifest_path),)
        ) as executor:
            results = list(executor.map(_replay_task, manifest["tasks"]))
        results.sort(key=lambda item: item[0])
        raw_sha256 = {name: digest for name, digest in results}
        _require(len(raw_sha256) == len(manifest["tasks"]), "replay result count changed")
        core = {
            "replay_version": REPLAY_VERSION, "manifest_sha256": manifest["manifest_sha256"],
            "source_binding_sha256": manifest["source_binding"]["source_binding_sha256"],
            "frozen_runner_sha256": sha256_file(FROZEN_RUNNER),
            "validator_source_sha256": sha256_file(Path(__file__)),
            "raw_sha256": raw_sha256, "task_count": len(results), "workers": int(workers),
            "all_bit_identical": True,
        }
        report = {**core, "replay_sha256": sha256_json(core)}
        atomic_json(report_path, report)
        return report
    except Exception as exc:
        core = {
            "replay_version": REPLAY_VERSION, "manifest_path": str(manifest_path),
            "validator_source_sha256": sha256_file(Path(__file__)),
            "failure": repr(exc), "traceback": traceback.format_exc(),
        }
        atomic_json(failed_path, {**core, "failure_sha256": sha256_json(core)})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "local_hard_viability" / "MANIFEST.json")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args(argv)
    print(replay(args.manifest, args.workers)["replay_sha256"])


if __name__ == "__main__":
    main()
