"""Cross-package equality against the frozen exp104 and exp105 code paths.

exp106 is a port, and the whole risk of a port is that something changed which
nobody meant to change. Two gates pin that down, and between them they cover
both branches of the machine.

**The exp104 gate, at q = 0.** exp104 certified the unaugmented path. Two claims
are checked:

1. *Same code.* exp106 rebuilt from an exp104 registry row gives byte-identical
   H_Z, H_X and logical frame.
2. *Same verdict.* With the readout channel switched off the two scoring criteria
   are the same function. exp104 asks whether the residual has zero syndrome and
   trivial pairing against logical_Z; exp106 asks whether phi_r of the residual
   vanishes. At q = 0 the correction reproduces the syndrome exactly, so the
   residual lies in ker(H_Z), the section term in phi_r drops out, and the two
   reduce to the same test. That is an identity, so a disagreement is a bug
   rather than a statistical fluctuation.

**The exp105 gate, at q = 0.05.** This is the stronger one and it is why exp106
carries a second gate at all. The exp104 comparison cannot reach the augmented
matrix, the mixed error channel, the readout draw or the q > 0 failure criterion,
because at q = 0 none of them is exercised. exp105 ran all of it for 1,057,020
trials, so exp106 is required to reproduce it **bit for bit** on exp105's own
frozen registry, seeds and q: identical corrections, identical logical labels,
identical readout-match flags, identical verdicts and identical iteration counts.
Here the two packages are meant to be the same function outright, not merely to
agree in some limit.

Neither experiment's production raw is tracked in Git, so both are
package-to-package comparisons rather than replays of stored files -- the same
form exp104's own Validation 002 used against exp103.
"""

import numpy as np
from ldpc import BpOsdDecoder

from .model import load_model, logical_label, parity_product
from .worker import score_logical_class


def exp104_row_to_exp106(row):
    """Re-key one exp104 registry row into the exp106 code-id format."""
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


def compare_q0_trials(exp104_model, exp106_model, p, seed, trials):
    """Run the q = 0 path through both packages on one shared stream.

    Returns a dict of counters and the first disagreement, if any. Nothing is
    selected on outcome: every trial is compared and every mismatch is reported.
    """
    if exp104_model.n != exp106_model.n or exp104_model.k != exp106_model.k:
        raise ValueError("models disagree on code parameters")
    for name in ("H_Z", "H_X", "logical_Z"):
        if not np.array_equal(getattr(exp104_model, name), getattr(exp106_model, name)):
            raise ValueError(f"models disagree on {name}")

    decoder_a = _unaugmented_decoder(exp104_model, p)
    decoder_b = _unaugmented_decoder(exp106_model, p)
    rng_a = np.random.Generator(np.random.PCG64(seed))
    rng_b = np.random.Generator(np.random.PCG64(seed))

    compared = 0
    failures = 0
    mismatches = []
    for trial in range(trials):
        error_a = (rng_a.random(exp104_model.n) < float(p)).astype(np.uint8)
        error_b = (rng_b.random(exp106_model.n) < float(p)).astype(np.uint8)
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

        phi_labels = logical_label(exp106_model, residual)
        failed_105 = bool(phi_labels.any())

        if failed_104 != failed_105:
            mismatches.append({
                "trial": trial, "field": "verdict",
                "exp104": bool(failed_104), "exp106": bool(failed_105),
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
        load_model(exp104_row_to_exp106(exp104_row)),
    )


# ---------------------------------------------------------------------------
# The exp105 gate: the augmented q > 0 path, bit for bit
# ---------------------------------------------------------------------------

def _augmented_decoder(model, p, q):
    """The frozen q > 0 decoder identity: [H_Z | I], mixed channel, n + n_c."""
    width = model.n + model.n_checks
    return BpOsdDecoder(
        model.H_augmented_sparse,
        error_channel=[float(p)] * model.n + [float(q)] * model.n_checks,
        bp_method="product_sum",
        max_iter=width,
        schedule="serial",
        serial_schedule_order=list(range(width)),
        osd_method="osd_0",
        osd_order=0,
        omp_thread_count=1,
    )


def build_paired_exp105_models(exp105_row):
    """Build the same code through exp105's package and exp106's.

    exp105 and exp106 share the registry column order and the code-id format, so
    the row needs no re-keying -- unlike exp104's, which predates both.
    """
    from data.expander_code.exp105.exp105_pipeline.model import (
        load_model as load_exp105_model,
    )

    return load_exp105_model(dict(exp105_row)), load_model(dict(exp105_row))


def compare_augmented_trials(exp105_model, exp106_model, p, q, seed, trials):
    """Run the augmented q > 0 path through both packages on one shared stream.

    Mirrors the frozen trial loop exactly: one continuous stream draws the data
    error and then the readout error, the effective syndrome is
    `H_Z eps xor mu`, and the augmented decoder is asked for a correction over
    `n + n_c` bits. Every trial is compared on every recorded field; nothing is
    selected on outcome.
    """
    from data.expander_code.exp105.exp105_pipeline.worker import (
        score_logical_class as exp105_score,
    )

    if (exp105_model.n, exp105_model.k, exp105_model.n_checks) != (
        exp106_model.n, exp106_model.k, exp106_model.n_checks
    ):
        raise ValueError("models disagree on code parameters")
    for name in ("H_Z", "H_X", "logical_Z", "label_basis"):
        if not np.array_equal(getattr(exp105_model, name), getattr(exp106_model, name)):
            raise ValueError(f"models disagree on {name}")

    decoder_a = _augmented_decoder(exp105_model, p, q)
    decoder_b = _augmented_decoder(exp106_model, p, q)
    rng_a = np.random.Generator(np.random.PCG64(seed))
    rng_b = np.random.Generator(np.random.PCG64(seed))

    compared = 0
    failures = 0
    mismatches = []
    for trial in range(trials):
        error_a = (rng_a.random(exp105_model.n) < float(p)).astype(np.uint8)
        readout_a = (rng_a.random(exp105_model.n_checks) < float(q)).astype(np.uint8)
        error_b = (rng_b.random(exp106_model.n) < float(p)).astype(np.uint8)
        readout_b = (rng_b.random(exp106_model.n_checks) < float(q)).astype(np.uint8)
        if not (np.array_equal(error_a, error_b) and np.array_equal(readout_a, readout_b)):
            mismatches.append({"trial": trial, "field": "input_stream"})
            break

        effective = np.bitwise_xor(
            parity_product(exp105_model.H_Z, error_a), readout_a,
        )
        correction_a = decoder_a.decode(effective)
        correction_b = decoder_b.decode(effective)
        if not np.array_equal(correction_a, correction_b):
            mismatches.append({"trial": trial, "field": "correction"})
            break
        if int(decoder_a.iter) != int(decoder_b.iter):
            mismatches.append({"trial": trial, "field": "bp_iterations"})
            break

        failed_a, match_a, labels_a = exp105_score(
            exp105_model, error_a, readout_a, correction_a,
        )
        failed_b, match_b, labels_b = score_logical_class(
            exp106_model, error_b, readout_b, correction_b,
        )
        if failed_a != failed_b or match_a != match_b:
            mismatches.append({
                "trial": trial, "field": "verdict",
                "exp105": bool(failed_a), "exp106": bool(failed_b),
                "readout_match_exp105": bool(match_a),
                "readout_match_exp106": bool(match_b),
            })
            break
        if not np.array_equal(labels_a, labels_b):
            mismatches.append({"trial": trial, "field": "labels"})
            break

        compared += 1
        failures += int(failed_b)

    return {
        "compared": compared,
        "requested": int(trials),
        "failures": failures,
        "mismatches": mismatches,
    }
