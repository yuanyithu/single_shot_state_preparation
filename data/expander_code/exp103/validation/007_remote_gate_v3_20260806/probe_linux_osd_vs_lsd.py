"""Confirm on the frozen Linux build: is BP+OSD deterministic, and what is its binary SHA?"""
import hashlib
import importlib
import sys
from pathlib import Path

import numpy as np
from ldpc import BpOsdDecoder

from data.expander_code.exp103.exp103_pipeline.config import load_config
from data.expander_code.exp103.exp103_pipeline.model import load_model, parity_product
from data.expander_code.exp103.exp103_pipeline.worker import make_decoder

CONFIG = "data/expander_code/exp103/config/decoder_mc.remote.v2.json"
module = importlib.import_module("ldpc.bposd_decoder._bposd_decoder")
path = Path(module.__file__).resolve()
print("osd module file:", path.name)
print("osd binary sha256:", hashlib.sha256(path.read_bytes()).hexdigest())

config = load_config(CONFIG)
for code_id, p_token, trials in (("m04_c01", "0.11", 1200), ("m05_c01", "0.12", 800)):
    model = load_model(config, code_id)
    rng = np.random.default_rng(20260806)
    syndromes = [
        parity_product(model.H_Z, (rng.random(model.n) < float(p_token)).astype(np.uint8))
        for _ in range(trials)
    ]

    def sweep(build):
        corrections = np.zeros((trials, model.n), np.uint8)
        converged = np.zeros(trials, np.bool_)
        decoder = build()
        for index, syndrome in enumerate(syndromes):
            corrections[index] = decoder.decode(syndrome)
            converged[index] = bool(decoder.converge)
        return corrections, converged

    def osd():
        return BpOsdDecoder(
            model.H_Z_sparse, error_rate=float(p_token), bp_method="product_sum",
            max_iter=model.n, schedule="serial",
            serial_schedule_order=list(range(model.n)),
            osd_method="osd_0", osd_order=0,
        )

    osd_a, conv = sweep(osd)
    osd_b, _ = sweep(osd)
    lsd_a, _ = sweep(lambda: make_decoder(model, float(p_token), config))
    lsd_b, _ = sweep(lambda: make_decoder(model, float(p_token), config))
    print(
        f"{code_id} p={p_token} n={model.n} trials={trials} "
        f"bp_noconv={1.0 - conv.mean():.4f} | "
        f"OSD differs {int((osd_a != osd_b).any(axis=1).sum())} | "
        f"LSD differs {int((lsd_a != lsd_b).any(axis=1).sum())}"
    )
