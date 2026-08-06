"""Read-only diagnostic: is the frozen BpLSD decode reproducible on identical input?

Loads failed measurement shards, re-decodes them twice, and reports which
raw fields disagree. Writes nothing; retains no new measurement.
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from data.expander_code.exp103.exp103_pipeline.config import load_config
from data.expander_code.exp103.exp103_pipeline.model import load_model, parity_product
from data.expander_code.exp103.exp103_pipeline.raw import load_raw, raw_filename
from data.expander_code.exp103.exp103_pipeline.seeds import derive_seed
from data.expander_code.exp103.exp103_pipeline.worker import make_decoder, score_residual_pairing

CONFIG = "data/expander_code/exp103/config/decoder_mc.remote.v2.json"
STAGE1 = Path.home() / ".single_shot/runs/exp103_remote_v2_001/raw/stage1"


def decode_pass(model, token, config, seed, trials):
    decoder = make_decoder(model, float(token), config)
    rng = np.random.Generator(np.random.PCG64(seed))
    out = {
        "failure": np.zeros(trials, np.bool_),
        "match": np.zeros(trials, np.bool_),
        "labels": np.zeros((trials, model.k), np.uint8),
        "converge": np.zeros(trials, np.bool_),
        "iter": np.zeros(trials, np.int32),
        "correction": np.zeros((trials, model.n), np.uint8),
        "syndrome_ok": np.zeros(trials, np.bool_),
    }
    for trial in range(trials):
        error = (rng.random(model.n) < float(token)).astype(np.uint8)
        syndrome = parity_product(model.H_Z, error)
        correction = decoder.decode(syndrome)
        failed, matched, labels = score_residual_pairing(model, error, correction)
        out["failure"][trial] = failed
        out["match"][trial] = matched
        out["labels"][trial] = labels
        out["converge"][trial] = bool(decoder.converge)
        out["iter"][trial] = int(decoder.iter)
        out["correction"][trial] = correction
        out["syndrome_ok"][trial] = not (parity_product(model.H_Z, correction) ^ syndrome).any()
    return out


def analyse(key):
    code_id, token, shard = key
    config = load_config(CONFIG)
    model = load_model(config, code_id)
    raw = load_raw(STAGE1 / raw_filename(code_id, token, shard))
    seed = derive_seed(config, "measurement", code_id, token, shard)
    trials = int(raw["completed_trials"])
    run_a = decode_pass(model, token, config, seed, trials)
    run_b = decode_pass(model, token, config, seed, trials)

    def diff(a, b):
        return np.flatnonzero(a != b) if a.ndim == 1 else np.flatnonzero((a != b).any(axis=1))

    report = {"code_id": code_id, "p_token": token, "shard": shard, "trials": trials}
    for name, raw_field in (
        ("failure", "failure_flags"), ("match", "syndrome_match"),
        ("labels", "logical_labels"), ("converge", "bp_converged"),
        ("iter", "bp_iterations"),
    ):
        report[f"raw_vs_run_a__{name}"] = int(diff(raw[raw_field], run_a[name]).size)
        report[f"run_a_vs_run_b__{name}"] = int(diff(run_a[name], run_b[name]).size)
    report["run_a_vs_run_b__correction"] = int(diff(run_a["correction"], run_b["correction"]).size)
    report["run_a_all_corrections_match_syndrome"] = bool(run_a["syndrome_ok"].all())
    report["run_b_all_corrections_match_syndrome"] = bool(run_b["syndrome_ok"].all())

    changed = diff(run_a["correction"], run_b["correction"])
    report["differing_trials_sample"] = [int(x) for x in changed[:8]]
    report["differing_trials_bp_converged_a"] = [bool(run_a["converge"][i]) for i in changed[:8]]
    report["differing_trials_logical_class_changed"] = int(
        sum(1 for i in changed if not np.array_equal(run_a["labels"][i], run_b["labels"][i]))
    )
    report["bp_nonconvergence_rate"] = float(1.0 - run_a["converge"].mean())
    report["bp_nonconverged_among_differing"] = float(
        np.mean([not run_a["converge"][i] for i in changed]) if changed.size else float("nan")
    )
    return report


if __name__ == "__main__":
    keys = [tuple(k) for k in json.loads(Path(sys.argv[1]).read_text())]
    with ProcessPoolExecutor(max_workers=len(keys)) as pool:
        for result in pool.map(analyse, keys):
            print(json.dumps(result, sort_keys=True), flush=True)
