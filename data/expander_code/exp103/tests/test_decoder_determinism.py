"""The gate whose absence let a randomized decoder reach a formal stage.

Validation 005 found that `ldpc.BpLsdDecoder` can return a different legal
correction, in a different logical class, for an identical syndrome, at a rate
of about 1e-5 per trial on Linux and 1.1e-2 on macOS. These tests pin the
decoder identity exactly and re-decode an identical syndrome sequence with two
freshly constructed decoders.

This is early warning, not the authoritative guarantee: a few hundred trials
catch a percent-level rate outright but cannot resolve 1e-5. The bit-exact full
replay over every measurement shard remains the authoritative check, and it is
what caught the v1 defect.
"""

import numpy as np
import pytest

from data.expander_code.exp103.exp103_pipeline import replay, worker
from data.expander_code.exp103.exp103_pipeline.model import load_model, parity_product


DETERMINISM_CODE = "m04_c01"
DETERMINISM_P = "0.11"
DETERMINISM_TRIALS = 300


def _sweep(build, syndromes, n):
    decoder = build()
    corrections = np.zeros((len(syndromes), n), dtype=np.uint8)
    converged = np.zeros(len(syndromes), dtype=np.bool_)
    iterations = np.zeros(len(syndromes), dtype=np.int32)
    for index, syndrome in enumerate(syndromes):
        corrections[index] = decoder.decode(syndrome)
        converged[index] = bool(decoder.converge)
        iterations[index] = int(decoder.iter)
    return corrections, converged, iterations


@pytest.mark.parametrize(
    ("factory_module", "factory_name"),
    [(worker, "make_decoder"), (replay, "_decoder")],
)
def test_frozen_decoder_is_deterministic_where_bp_does_not_converge(
    frozen_config, factory_module, factory_name,
):
    model = load_model(frozen_config, DETERMINISM_CODE)
    rng = np.random.default_rng(20260806)
    syndromes = [
        parity_product(
            model.H_Z,
            (rng.random(model.n) < float(DETERMINISM_P)).astype(np.uint8),
        )
        for _ in range(DETERMINISM_TRIALS)
    ]

    def build():
        return getattr(factory_module, factory_name)(
            model, float(DETERMINISM_P), frozen_config,
        )

    first, converged, iterations = _sweep(build, syndromes, model.n)
    second, converged_again, iterations_again = _sweep(build, syndromes, model.n)

    # A decoder that always converges would never reach the post-processing
    # stage, making the comparison below vacuous.
    assert float(1.0 - converged.mean()) > 0.5
    assert np.array_equal(first, second)
    assert np.array_equal(converged, converged_again)
    assert np.array_equal(iterations, iterations_again)


def test_frozen_decoder_identity_excludes_the_randomized_lsd_stage(frozen_config):
    spec = frozen_config["decoder"]
    assert spec["osd_method"] == "osd_0" and spec["osd_order"] == 0
    assert not any(key.startswith("lsd_") for key in spec)
    assert "always_run_lsd" not in spec and "bits_per_step" not in spec
    assert frozen_config["decoder_binary"]["module"] == "ldpc.bposd_decoder._bposd_decoder"
    assert frozen_config["objective"] == "bposd_block_logical_failure_crossing_q0"
    assert worker.BpOsdDecoder is replay.BpOsdDecoder
