"""Is BP+OSD deterministic where BP+LSD is not? Probe only; no measurement."""
import sys
import numpy as np
from ldpc import BpOsdDecoder

sys.path.insert(0, "/Users/jarvis/Desktop/project D")
from data.expander_code.exp103.exp103_pipeline.config import load_config
from data.expander_code.exp103.exp103_pipeline.model import load_model, parity_product

CODE_ID, P, TRIALS = sys.argv[1], sys.argv[2], int(sys.argv[3])
config = load_config("data/expander_code/exp103/config/decoder_mc.v1.json")
model = load_model(config, CODE_ID)
rng = np.random.default_rng(20260806)
errors = (rng.random((TRIALS, model.n)) < float(P)).astype(np.uint8)
syndromes = [parity_product(model.H_Z, e) for e in errors]


def build(osd_method, osd_order):
    return BpOsdDecoder(
        model.H_Z_sparse, error_rate=float(P), bp_method="product_sum",
        max_iter=model.n, schedule="serial",
        serial_schedule_order=list(range(model.n)),
        osd_method=osd_method, osd_order=osd_order,
    )


for osd_method, osd_order in (("osd_0", 0), ("osd_cs", 4)):
    corrections, converged = [], []
    for _pass in range(2):
        decoder = build(osd_method, osd_order)
        corr = np.zeros((TRIALS, model.n), np.uint8)
        conv = np.zeros(TRIALS, np.bool_)
        for index, syndrome in enumerate(syndromes):
            corr[index] = decoder.decode(syndrome)
            conv[index] = bool(decoder.converge)
        corrections.append(corr)
        converged.append(conv)
    differing = int((corrections[0] != corrections[1]).any(axis=1).sum())
    labels_a = np.array([parity_product(model.logical_Z, np.bitwise_xor(e, c))
                         for e, c in zip(errors, corrections[0])])
    labels_b = np.array([parity_product(model.logical_Z, np.bitwise_xor(e, c))
                         for e, c in zip(errors, corrections[1])])
    class_diff = int((labels_a != labels_b).any(axis=1).sum())
    print(f"{osd_method} order={osd_order}: correction differs {differing}/{TRIALS} | "
          f"logical class differs {class_diff}/{TRIALS} | "
          f"converge differs {int((converged[0] != converged[1]).sum())} | "
          f"bp noconv {1.0 - converged[0].mean():.4f}")
