"""Resident determinism regression for the augmented decoder.

Permanent discipline 15: a bit-for-bit replay gate may only be built on measured
determinism, never on assumed determinism. exp106 replays a preregistered ten
percent of tasks, so this file is the evidence that licenses the other ninety.

It is deliberately awkward in one respect. The interesting case is the one where
belief propagation does *not* converge and ordered statistics decoding takes
over, because that is the path where an implementation is most likely to depend
on uninitialised state or iteration order. So the test first asserts that the
production operating point actually reaches that path, and only then asserts
determinism inside it. A determinism test that quietly ran on converged
instances would prove nothing about the run it is supposed to protect.
"""

import numpy as np
import pytest

from data.expander_code.exp106.exp106_pipeline.model import (
    clear_model_cache,
    load_model,
    parity_product,
)
from data.expander_code.exp106.exp106_pipeline.worker import make_decoder
from data.expander_code.exp106.exp106_pipeline.config import code_id


TRIALS = 24


def _stream(model, p, q, seed, trials=TRIALS):
    rng = np.random.Generator(np.random.PCG64(seed))
    payload = []
    for _ in range(trials):
        error = (rng.random(model.n) < p).astype(np.uint8)
        readout = (rng.random(model.n_checks) < q).astype(np.uint8)
        payload.append((error, readout))
    return payload


def _decode_all(model, config, p, payload):
    decoder = make_decoder(model, p, config)
    corrections = []
    iterations = []
    converged = []
    for error, readout in payload:
        effective = np.bitwise_xor(parity_product(model.H_Z, error), readout)
        corrections.append(np.array(decoder.decode(effective), copy=True))
        iterations.append(int(decoder.iter))
        converged.append(bool(decoder.converge))
    return corrections, iterations, converged


@pytest.fixture(scope="module")
def m3_model(request):
    rows = request.getfixturevalue("registry_rows")
    clear_model_cache()
    model = load_model(rows[code_id(3, 0)])
    yield model
    clear_model_cache()


def test_augmented_decoder_reaches_the_non_convergent_path(m3_model, frozen_config):
    """The production operating point must actually exercise OSD fallback."""
    p = float(frozen_config["p_tokens"][-1])
    q = float(frozen_config["q_token"])
    payload = _stream(m3_model, p, q, seed=90210)
    _, iterations, converged = _decode_all(m3_model, frozen_config, p, payload)
    width = m3_model.n + m3_model.n_checks
    assert any(not flag for flag in converged), (
        "no trial exhausted belief propagation, so the determinism assertion "
        "below would not cover the ordered-statistics path"
    )
    assert max(iterations) == width, (
        f"expected max_iter={width} to be reached; saw {max(iterations)}"
    )


def test_augmented_decoder_is_deterministic_on_repeat(m3_model, frozen_config):
    p = float(frozen_config["p_tokens"][-1])
    q = float(frozen_config["q_token"])
    payload = _stream(m3_model, p, q, seed=90210)

    first = _decode_all(m3_model, frozen_config, p, payload)
    second = _decode_all(m3_model, frozen_config, p, payload)

    assert any(not flag for flag in first[2]), "determinism measured only on the BP path"
    for index, (left, right) in enumerate(zip(first[0], second[0])):
        assert np.array_equal(left, right), f"correction differs on repeat at trial {index}"
    assert first[1] == second[1]
    assert first[2] == second[2]


def test_augmented_decoder_is_deterministic_across_fresh_instances(
    m3_model, frozen_config,
):
    """A second, independently constructed decoder object must agree exactly."""
    p = float(frozen_config["p_tokens"][len(frozen_config["p_tokens"]) // 2])
    q = float(frozen_config["q_token"])
    payload = _stream(m3_model, p, q, seed=1357)

    first = _decode_all(m3_model, frozen_config, p, payload)
    second = _decode_all(m3_model, frozen_config, p, payload)

    for index, (left, right) in enumerate(zip(first[0], second[0])):
        assert np.array_equal(left, right), f"fresh decoder differs at trial {index}"
    assert first[1] == second[1]


def test_decode_does_not_depend_on_call_order(m3_model, frozen_config):
    """Decoding is a function of its input, not of what was decoded before it."""
    p = float(frozen_config["p_tokens"][-1])
    q = float(frozen_config["q_token"])
    payload = _stream(m3_model, p, q, seed=24680)

    forward, forward_iterations, _ = _decode_all(m3_model, frozen_config, p, payload)
    reverse, reverse_iterations, _ = _decode_all(
        m3_model, frozen_config, p, list(reversed(payload)),
    )
    for index in range(len(payload)):
        assert np.array_equal(forward[index], reverse[len(payload) - 1 - index]), (
            f"correction depends on call order at trial {index}"
        )
        assert forward_iterations[index] == reverse_iterations[len(payload) - 1 - index]
