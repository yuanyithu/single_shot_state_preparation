"""Resident regression gate for permanent discipline 15.

exp103 was blocked at `BLOCKED_REPLAY_NONDETERMINISM` because `BpLsdDecoder`'s
localised-statistics stage is randomized, so a bit-exact replay gate was
unsatisfiable. exp104 replays only a committed ten percent of tasks, which makes
this test load-bearing rather than decorative: determinism must be measured in
the target regime, in this interpreter, on this build, every time the suite runs.

The regime that matters is the one where belief propagation fails to converge
and the post-processor decides the answer. That is where the superseded decoder
was randomized, so the test asserts the regime is actually reached before it
asserts determinism inside it.
"""

import numpy as np
import pytest

from data.expander_code.exp104.exp104_pipeline import replay, worker
from data.expander_code.exp104.exp104_pipeline.model import load_model, parity_product


DETERMINISM_M = 4
DETERMINISM_CODE_INDEX = 0
DETERMINISM_P = "0.10"
DETERMINISM_TRIALS = 150


def _decode_stream(decoder, model, p, seed, trials):
    rng = np.random.Generator(np.random.PCG64(seed))
    corrections = np.zeros((trials, model.n), dtype=np.uint8)
    converged = np.zeros(trials, dtype=np.bool_)
    iterations = np.zeros(trials, dtype=np.int32)
    for trial in range(trials):
        error = (rng.random(model.n) < float(p)).astype(np.uint8)
        syndrome = parity_product(model.H_Z, error)
        corrections[trial] = decoder.decode(syndrome)
        converged[trial] = bool(decoder.converge)
        iterations[trial] = int(decoder.iter)
    return corrections, converged, iterations


@pytest.fixture(scope="module")
def determinism_model(registry_rows):
    from data.expander_code.exp104.exp104_pipeline.config import code_id

    return load_model(registry_rows[code_id(DETERMINISM_M, DETERMINISM_CODE_INDEX)])


@pytest.mark.parametrize(("module", "factory_name"), [
    (worker, "make_decoder"),
    (replay, "_decoder"),
])
def test_frozen_decoder_is_deterministic_where_bp_does_not_converge(
    module, factory_name, determinism_model, frozen_config,
):
    factory = getattr(module, factory_name)

    def build():
        if factory_name == "make_decoder":
            return factory(determinism_model, float(DETERMINISM_P), frozen_config)
        return factory(determinism_model, float(DETERMINISM_P))

    first, converged, iterations = _decode_stream(
        build(), determinism_model, DETERMINISM_P, 7, DETERMINISM_TRIALS,
    )
    second, converged_again, iterations_again = _decode_stream(
        build(), determinism_model, DETERMINISM_P, 7, DETERMINISM_TRIALS,
    )

    non_convergence = float(1.0 - converged.mean())
    assert non_convergence > 0.5, (
        "determinism must be measured where the post-processor decides the "
        f"answer; only {non_convergence:.3f} of trials failed to converge"
    )
    assert np.array_equal(first, second)
    assert np.array_equal(converged, converged_again)
    assert np.array_equal(iterations, iterations_again)


def test_worker_and_replay_decoders_agree_trial_by_trial(
    determinism_model, frozen_config,
):
    """The two independently constructed decoders must be the same function."""
    from_worker, _, _ = _decode_stream(
        worker.make_decoder(determinism_model, float(DETERMINISM_P), frozen_config),
        determinism_model, DETERMINISM_P, 11, 60,
    )
    from_replay, _, _ = _decode_stream(
        replay._decoder(determinism_model, float(DETERMINISM_P)),
        determinism_model, DETERMINISM_P, 11, 60,
    )
    assert np.array_equal(from_worker, from_replay)


def test_frozen_decoder_identity_excludes_the_randomized_lsd_stage(frozen_config):
    spec = frozen_config["decoder"]
    assert spec["osd_method"] == "osd_0"
    assert spec["osd_order"] == 0
    assert not any(key.startswith("lsd_") for key in spec)
    assert "bp_lsd" not in spec.get("bp_method", "")
