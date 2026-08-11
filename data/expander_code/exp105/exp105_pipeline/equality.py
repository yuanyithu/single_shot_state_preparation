"""Cross-package equality against the frozen exp104 code path.

exp105 adds a `q > 0` branch to a machine exp104 already certified. The risk
that creates is not in the new branch, which is tested directly, but in the
shared parts: ensemble reconstruction, model assembly, seed derivation and
scoring. This module pins those down by running exp104's own package and
exp105's side by side on the same codes and the same seeds.

Two claims are checked, and they are the two that matter:

1. **Same code.** exp105 rebuilt from an exp104 registry row gives byte-identical
   H_Z, H_X and logical frame.
2. **Same verdict at q = 0.** With the readout channel switched off the two
   scoring criteria are the same function. exp104 asks whether the residual has
   zero syndrome and trivial pairing against logical_Z; exp105 asks whether
   phi_r of the residual vanishes. At q = 0 the decoder's correction reproduces
   the syndrome exactly, so the residual lies in ker(H_Z), the section term in
   phi_r drops out, and the two reduce to the same test. That is an identity, so
   a disagreement is a bug rather than a statistical fluctuation.

exp104's production raw lives on nd-3 and is not tracked in Git, so this is a
package-to-package comparison rather than a replay of stored files -- the same
form exp104's own Validation 002 used against exp103.
"""

import numpy as np
from ldpc import BpOsdDecoder

from .model import load_model, logical_label, parity_product


def exp104_row_to_exp105(row):
    """Re-key one exp104 registry row into the exp105 code-id format."""
    row = dict(row)
    row["code_id"] = "m{:02d}_c{:06d}".format(int(row["m"]), int(row["code_index"]))
    return row


def _unaugmented_decoder(model, p):
    """The exp104 decoder identity: H_Z alone, scalar rate, max_iter = n."""
    return BpOsdDecoder(
        model.H_Z_sparse,
        error_rate=float(p),
        bp_method="product_sum",
        max_iter=model.n,
        schedule="serial",
        serial_schedule_order=list(range(model.n)),
        osd_method="osd_0",
        osd_order=0,
        omp_thread_count=1,
    )


def compare_q0_trials(exp104_model, exp105_model, p, seed, trials):
    """Run the q = 0 path through both packages on one shared stream.

    Returns a dict of counters and the first disagreement, if any. Nothing is
    selected on outcome: every trial is compared and every mismatch is reported.
    """
    if exp104_model.n != exp105_model.n or exp104_model.k != exp105_model.k:
        raise ValueError("models disagree on code parameters")
    for name in ("H_Z", "H_X", "logical_Z"):
        if not np.array_equal(getattr(exp104_model, name), getattr(exp105_model, name)):
            raise ValueError(f"models disagree on {name}")

    decoder_a = _unaugmented_decoder(exp104_model, p)
    decoder_b = _unaugmented_decoder(exp105_model, p)
    rng_a = np.random.Generator(np.random.PCG64(seed))
    rng_b = np.random.Generator(np.random.PCG64(seed))

    compared = 0
    failures = 0
    mismatches = []
    for trial in range(trials):
        error_a = (rng_a.random(exp104_model.n) < float(p)).astype(np.uint8)
        error_b = (rng_b.random(exp105_model.n) < float(p)).astype(np.uint8)
        if not np.array_equal(error_a, error_b):
            mismatches.append({"trial": trial, "field": "error_stream"})
            break

        syndrome = parity_product(exp104_model.H_Z, error_a)
        correction_a = decoder_a.decode(syndrome)
        correction_b = decoder_b.decode(syndrome)
        if not np.array_equal(correction_a, correction_b):
            mismatches.append({"trial": trial, "field": "correction"})
            break

        residual = np.bitwise_xor(error_a, correction_a)
        syndrome_match = not parity_product(exp104_model.H_Z, residual).any()
        pairing_labels = parity_product(exp104_model.logical_Z, residual)
        failed_104 = (not syndrome_match) or bool(pairing_labels.any())

        phi_labels = logical_label(exp105_model, residual)
        failed_105 = bool(phi_labels.any())

        if failed_104 != failed_105:
            mismatches.append({
                "trial": trial, "field": "verdict",
                "exp104": bool(failed_104), "exp105": bool(failed_105),
                "syndrome_match": bool(syndrome_match),
            })
            break
        if syndrome_match and not np.array_equal(pairing_labels, phi_labels):
            mismatches.append({"trial": trial, "field": "labels"})
            break
        if int(decoder_a.iter) != int(decoder_b.iter):
            mismatches.append({"trial": trial, "field": "bp_iterations"})
            break

        compared += 1
        failures += int(failed_105)

    return {
        "compared": compared,
        "requested": int(trials),
        "failures": failures,
        "mismatches": mismatches,
    }


def build_paired_models(exp104_row):
    """Build the same code through both packages from one exp104 registry row."""
    from data.expander_code.exp104.exp104_pipeline.model import (
        load_model as load_exp104_model,
    )

    return (
        load_exp104_model(dict(exp104_row)),
        load_model(exp104_row_to_exp105(exp104_row)),
    )
