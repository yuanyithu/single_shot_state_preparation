"""Read-only: on every shard whose bit-exact replay failed, does the PHYSICAL
outcome reproduce? Reports disagreement counts only; no rate, no aggregate."""
import hashlib
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


def analyse(key):
    code_id, token, shard = key
    config = load_config(CONFIG)
    model = load_model(config, code_id)
    raw = load_raw(STAGE1 / raw_filename(code_id, token, shard))
    seed = derive_seed(config, "measurement", code_id, token, shard)
    trials = int(raw["completed_trials"])
    decoder = make_decoder(model, float(token), config)
    rng = np.random.Generator(np.random.PCG64(seed))
    error_digest = hashlib.sha256()
    counts = dict.fromkeys(
        ("failure", "syndrome_match", "labels", "bp_converged", "bp_iterations"), 0
    )
    illegal = 0
    for trial in range(trials):
        error = (rng.random(model.n) < float(token)).astype(np.uint8)
        error_digest.update(error.tobytes())
        syndrome = parity_product(model.H_Z, error)
        correction = decoder.decode(syndrome)
        if (parity_product(model.H_Z, correction) ^ syndrome).any():
            illegal += 1
        failed, matched, labels = score_residual_pairing(model, error, correction)
        counts["failure"] += int(failed != bool(raw["failure_flags"][trial]))
        counts["syndrome_match"] += int(matched != bool(raw["syndrome_match"][trial]))
        counts["labels"] += int(not np.array_equal(labels, raw["logical_labels"][trial]))
        counts["bp_converged"] += int(bool(decoder.converge) != bool(raw["bp_converged"][trial]))
        counts["bp_iterations"] += int(int(decoder.iter) != int(raw["bp_iterations"][trial]))
    return {
        "code_id": code_id, "p_token": token, "shard": shard, "trials": trials,
        "error_stream_exact": error_digest.hexdigest() == raw["error_stream_sha256"],
        "corrections_all_legal": illegal == 0,
        **{f"disagree_{name}": value for name, value in counts.items()},
    }


if __name__ == "__main__":
    keys = [tuple(k) for k in json.loads(Path(sys.argv[1]).read_text())]
    with ProcessPoolExecutor(max_workers=int(sys.argv[2])) as pool:
        results = list(pool.map(analyse, keys))
    for item in results:
        print(json.dumps(item, sort_keys=True), flush=True)
    totals = {
        key: sum(item[key] for item in results)
        for key in results[0] if key.startswith("disagree_")
    }
    print("TOTALS " + json.dumps({
        "shards": len(results),
        "trials": sum(item["trials"] for item in results),
        "error_stream_exact_all": all(item["error_stream_exact"] for item in results),
        "corrections_all_legal": all(item["corrections_all_legal"] for item in results),
        **totals,
    }, sort_keys=True), flush=True)
