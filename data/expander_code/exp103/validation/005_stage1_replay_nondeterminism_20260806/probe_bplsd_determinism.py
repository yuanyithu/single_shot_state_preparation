"""Across p, does decoder nondeterminism change the physical failure flag?"""
import sys
import numpy as np

sys.path.insert(0, "/Users/jarvis/Desktop/project D")
from data.expander_code.exp103.exp103_pipeline.config import load_config
from data.expander_code.exp103.exp103_pipeline.model import load_model, parity_product
from data.expander_code.exp103.exp103_pipeline.worker import make_decoder, score_residual_pairing

CODE_ID, TRIALS = sys.argv[1], int(sys.argv[2])
config = load_config("data/expander_code/exp103/config/decoder_mc.v1.json")
model = load_model(config, CODE_ID)
print(f"code={CODE_ID} n={model.n} k={model.k} trials={TRIALS} per p")
print(f"{'p':>6} {'P_fail(A)':>10} {'P_fail(B)':>10} {'corr diff':>10} {'class diff':>11} {'FAIL-FLAG diff':>15} {'bp noconv':>10}")
for p_token in ("0.04", "0.06", "0.08", "0.10"):
    rng = np.random.default_rng(20260806)
    errors = (rng.random((TRIALS, model.n)) < float(p_token)).astype(np.uint8)

    def run():
        decoder = make_decoder(model, float(p_token), config)
        fail = np.zeros(TRIALS, np.bool_)
        lab = np.zeros((TRIALS, model.k), np.uint8)
        corr = np.zeros((TRIALS, model.n), np.uint8)
        conv = np.zeros(TRIALS, np.bool_)
        for index, error in enumerate(errors):
            correction = decoder.decode(parity_product(model.H_Z, error))
            failed, _m, label = score_residual_pairing(model, error, correction)
            fail[index], lab[index], corr[index] = failed, label, correction
            conv[index] = bool(decoder.converge)
        return fail, lab, corr, conv

    fail_a, lab_a, corr_a, conv_a = run()
    fail_b, lab_b, corr_b, _ = run()
    print(f"{p_token:>6} {fail_a.mean():>10.4f} {fail_b.mean():>10.4f} "
          f"{int((corr_a != corr_b).any(axis=1).sum()):>10} "
          f"{int((lab_a != lab_b).any(axis=1).sum()):>11} "
          f"{int((fail_a != fail_b).sum()):>15} {1.0 - conv_a.mean():>10.4f}")
