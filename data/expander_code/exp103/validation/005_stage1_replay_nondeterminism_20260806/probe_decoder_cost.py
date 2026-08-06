"""Rough per-trial cost of BP+LSD versus BP+OSD, to size a decoder switch."""
import sys, time
import numpy as np
from ldpc import BpOsdDecoder

sys.path.insert(0, "/Users/jarvis/Desktop/project D")
from data.expander_code.exp103.exp103_pipeline.config import load_config
from data.expander_code.exp103.exp103_pipeline.model import load_model, parity_product
from data.expander_code.exp103.exp103_pipeline.worker import make_decoder

config = load_config("data/expander_code/exp103/config/decoder_mc.v1.json")
print(f"{'code':>9} {'p':>6} {'n':>5} {'LSD s/trial':>12} {'OSD0 s/trial':>13} {'ratio':>7}")
for code_id, p_token, trials in (("m03_c00", "0.14", 300), ("m05_c01", "0.12", 200), ("m08_c00", "0.08", 40)):
    model = load_model(config, code_id)
    rng = np.random.default_rng(7)
    syndromes = [parity_product(model.H_Z, (rng.random(model.n) < float(p_token)).astype(np.uint8))
                 for _ in range(trials)]
    lsd = make_decoder(model, float(p_token), config)
    start = time.perf_counter()
    for syndrome in syndromes: lsd.decode(syndrome)
    lsd_time = (time.perf_counter() - start) / trials
    osd = BpOsdDecoder(model.H_Z_sparse, error_rate=float(p_token), bp_method="product_sum",
                       max_iter=model.n, schedule="serial",
                       serial_schedule_order=list(range(model.n)), osd_method="osd_0", osd_order=0)
    start = time.perf_counter()
    for syndrome in syndromes: osd.decode(syndrome)
    osd_time = (time.perf_counter() - start) / trials
    print(f"{code_id:>9} {p_token:>6} {model.n:>5} {lsd_time:>12.5f} {osd_time:>13.5f} {osd_time/lsd_time:>7.2f}")
