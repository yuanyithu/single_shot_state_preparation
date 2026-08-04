"""Exact independence-MH using a mixture of BP-systematic hard-coset proposals."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np

from .exp101_bridge import load_exp101
from .io import canonical_json
from .q0_global import state_label
from .q0_iid_importance import mixture_log_probability_state


BP_IMH_VERSION = "exp102.q0_bp_imh.v0"
BP_IMH_RAW_VERSION = "exp102.q0_bp_imh.local.raw.v0"


class BpImhError(ValueError):
    """Raised when the combined proposal or IMH transcript is invalid."""


def _require(condition, message):
    if not condition:
        raise BpImhError(message)


def _bits(value, *, ndim, name):
    array = np.asarray(value)
    _require(array.ndim == int(ndim), f"{name} has the wrong dimension")
    _require(np.issubdtype(array.dtype, np.bool_) or np.issubdtype(array.dtype, np.integer),
             f"{name} is not binary data")
    _require(not np.any((array != 0) & (array != 1)), f"{name} is not binary")
    return np.ascontiguousarray(array, dtype=np.uint8)


def _probabilities(values, count):
    result = np.asarray(values, dtype=np.float64)
    _require(result.shape == (int(count),) and np.all(np.isfinite(result))
             and np.all(result > 0.0), "combined proposal weights are invalid")
    _require(abs(float(result.sum()) - 1.0) <= 1e-14,
             "combined proposal weights are not normalized")
    result = result / float(result.sum())
    result[-1] = 1.0 - float(result[:-1].sum())
    _require(np.all(result > 0.0), "combined proposal normalization lost support")
    return np.ascontiguousarray(result)


def _hard_residual(H_check, state, syndrome):
    return (
        H_check.astype(np.int64) @ state.astype(np.int64) % 2
    ).astype(np.uint8) ^ syndrome


@dataclass(frozen=True)
class CombinedBpProposal:
    """A fixed outer mixture of exact full-support BP-systematic proposals."""

    proposals: tuple
    weights: np.ndarray
    proposal_sha256: str

    def __post_init__(self):
        proposals = tuple(self.proposals)
        _require(len(proposals) >= 2, "combined BP proposal needs at least two sources")
        _require(all(hasattr(proposal, "sample")
                     and hasattr(proposal, "log_probability_state")
                     for proposal in proposals),
                 "combined BP source lacks sampling or density")
        weights = _probabilities(self.weights, len(proposals))
        object.__setattr__(self, "proposals", proposals)
        weights.setflags(write=False)
        object.__setattr__(self, "weights", weights)

    @property
    def num_sources(self):
        return len(self.proposals)

    @property
    def num_qubits(self):
        return int(self.proposals[0].coordinates.num_qubits)

    def log_probability_state(self, state):
        return mixture_log_probability_state(self.proposals, self.weights, state)

    def sample(self, rng):
        threshold = float(rng.random())
        _require(math.isfinite(threshold) and 0.0 <= threshold < 1.0,
                 "combined proposal source uniform is invalid")
        cumulative = 0.0
        source = self.num_sources - 1
        for index, probability in enumerate(self.weights):
            cumulative += float(probability)
            if threshold < cumulative:
                source = index
                break
        draw = self.proposals[source].sample(rng)
        _require(isinstance(draw, dict)
                 and {"state", "coordinate", "log_q", "component_index"} <= set(draw),
                 "combined BP source returned an incomplete draw")
        state = _bits(draw["state"], ndim=1, name="combined proposal state")
        _require(state.shape == (self.num_qubits,), "combined proposal state length changed")
        source_log_q = float(draw["log_q"])
        _require(math.isfinite(source_log_q), "combined proposal source density is nonfinite")
        values = np.empty(self.num_sources, dtype=np.float64)
        for index, (weight, proposal) in enumerate(zip(
                self.weights, self.proposals, strict=True)):
            log_q = source_log_q if index == source else float(proposal.log_probability_state(state))
            _require(math.isfinite(log_q), "combined proposal cross-density is nonfinite")
            values[index] = math.log(float(weight)) + log_q
        maximum = float(values.max())
        log_q = maximum + math.log(float(np.exp(values - maximum).sum(dtype=np.float64)))
        _require(math.isfinite(log_q), "combined proposal mixture density is nonfinite")
        return {
            "state": state,
            "source_index": int(source),
            "component_index": int(draw["component_index"]),
            "source_log_q": source_log_q,
            "log_q": log_q,
        }


def combine_bp_proposals(proposals, weights):
    proposals = tuple(proposals)
    normalized = _probabilities(weights, len(proposals))
    identities = []
    for proposal in proposals:
        identity = getattr(proposal, "proposal_sha256", None)
        _require(isinstance(identity, str) and len(identity) == 64,
                 "combined BP source identity is invalid")
        identities.append(identity)
    core = {
        "version": BP_IMH_VERSION,
        "source_proposal_sha256": identities,
        "weights": [format(float(value), ".17g") for value in normalized],
    }
    digest = hashlib.sha256(canonical_json(core).encode("ascii")).hexdigest()
    return CombinedBpProposal(proposals, normalized, digest)


def log_acceptance_ratio(current_weight, current_log_q, proposal_weight,
                         proposal_log_q, p):
    """Return the unclipped exact log MH ratio for the q=0 target."""
    p = float(p)
    _require(math.isfinite(p) and 0.0 < p < 0.5, "BP-IMH p must lie in (0,.5)")
    current_weight = int(current_weight)
    proposal_weight = int(proposal_weight)
    current_log_q = float(current_log_q)
    proposal_log_q = float(proposal_log_q)
    _require(current_weight >= 0 and proposal_weight >= 0
             and math.isfinite(current_log_q) and math.isfinite(proposal_log_q),
             "BP-IMH acceptance inputs are invalid")
    return (
        (proposal_weight - current_weight) * math.log(p / (1.0 - p))
        + current_log_q - proposal_log_q
    )


def acceptance_decision(log_ratio, uniform):
    log_ratio = float(log_ratio)
    uniform = float(uniform)
    _require(math.isfinite(log_ratio) and math.isfinite(uniform)
             and 0.0 <= uniform < 1.0,
             "BP-IMH acceptance decision inputs are invalid")
    log_acceptance = min(0.0, log_ratio)
    probability = 1.0 if log_acceptance == 0.0 else math.exp(log_acceptance)
    return bool(uniform < probability), log_acceptance


def _run_stage(model, frame, syndrome, p, proposal, rng, initial_state,
               initial_weight, initial_label, initial_log_q, steps):
    steps = int(steps)
    _require(steps > 0, "BP-IMH stage must have a positive fixed clock")
    packed_width = (model.num_qubits + 7) // 8
    proposal_states = np.empty((steps, packed_width), dtype=np.uint8)
    proposal_labels = np.empty(steps, dtype=np.uint64)
    proposal_weights = np.empty(steps, dtype=np.int32)
    proposal_source_indices = np.empty(steps, dtype=np.int8)
    proposal_component_indices = np.empty(steps, dtype=np.int8)
    proposal_source_log_q = np.empty(steps, dtype=np.float64)
    proposal_log_q = np.empty(steps, dtype=np.float64)
    acceptance_uniforms = np.empty(steps, dtype=np.float64)
    log_acceptance = np.empty(steps, dtype=np.float64)
    accepted = np.empty(steps, dtype=np.uint8)
    state_changed = np.empty(steps, dtype=np.uint8)
    states = np.empty((steps, packed_width), dtype=np.uint8)
    labels = np.empty(steps, dtype=np.uint64)
    weights = np.empty(steps, dtype=np.int32)
    current_log_q_values = np.empty(steps, dtype=np.float64)

    state = np.ascontiguousarray(initial_state, dtype=np.uint8).copy()
    weight = int(initial_weight)
    label = int(initial_label)
    current_log_q = float(initial_log_q)
    for step in range(steps):
        draw = proposal.sample(rng)
        proposed = np.asarray(draw["state"], dtype=np.uint8)
        proposed_weight = int(proposed.sum())
        proposed_label = int(state_label(frame, proposed))
        ratio = log_acceptance_ratio(
            weight, current_log_q, proposed_weight, draw["log_q"], p,
        )
        uniform = float(rng.random())
        decision, clipped = acceptance_decision(ratio, uniform)
        changed = bool(decision and not np.array_equal(state, proposed))
        if decision:
            state = proposed.copy()
            weight = proposed_weight
            label = proposed_label
            current_log_q = float(draw["log_q"])

        proposal_states[step] = np.packbits(proposed, bitorder="little")
        proposal_labels[step] = np.uint64(proposed_label)
        proposal_weights[step] = proposed_weight
        proposal_source_indices[step] = int(draw["source_index"])
        proposal_component_indices[step] = int(draw["component_index"])
        proposal_source_log_q[step] = float(draw["source_log_q"])
        proposal_log_q[step] = float(draw["log_q"])
        acceptance_uniforms[step] = uniform
        log_acceptance[step] = clipped
        accepted[step] = np.uint8(decision)
        state_changed[step] = np.uint8(changed)
        states[step] = np.packbits(state, bitorder="little")
        labels[step] = np.uint64(label)
        weights[step] = weight
        current_log_q_values[step] = current_log_q
    return state, weight, label, current_log_q, {
        "proposal_states_packed": proposal_states,
        "proposal_labels": proposal_labels,
        "proposal_weights": proposal_weights,
        "proposal_source_indices": proposal_source_indices,
        "proposal_component_indices": proposal_component_indices,
        "proposal_source_log_q": proposal_source_log_q,
        "proposal_log_q": proposal_log_q,
        "acceptance_uniforms": acceptance_uniforms,
        "log_acceptance": log_acceptance,
        "accepted": accepted,
        "state_changed": state_changed,
        "states_packed": states,
        "labels": labels,
        "weights": weights,
        "current_log_q": current_log_q_values,
    }


def run_bp_imh_trajectory(model, frame, syndrome, p, proposal, initial_state,
                          seed, *, burn_steps, measurement_steps):
    """Run one fixed-clock BP-IMH trajectory and retain the full transcript."""
    if not isinstance(proposal, CombinedBpProposal):
        raise TypeError("BP-IMH proposal has the wrong type")
    syndrome = _bits(syndrome, ndim=1, name="BP-IMH syndrome")
    initial = _bits(initial_state, ndim=1, name="BP-IMH initial state")
    _require(syndrome.shape == (model.num_checks,)
             and initial.shape == (model.num_qubits,),
             "BP-IMH model dimensions changed")
    _require(not _hard_residual(model.H_check, initial, syndrome).any(),
             "BP-IMH initial state is outside the hard coset")
    initial_weight = int(initial.sum())
    initial_label = int(state_label(frame, initial))
    initial_log_q = float(proposal.log_probability_state(initial))
    _require(math.isfinite(initial_log_q), "BP-IMH initial proposal density is nonfinite")

    load_exp101()
    from exp101_certified_src.prng import PortablePrng

    rng = PortablePrng(int(seed))
    burn_state, burn_weight, burn_label, burn_log_q, burn = _run_stage(
        model, frame, syndrome, p, proposal, rng, initial,
        initial_weight, initial_label, initial_log_q, burn_steps,
    )
    final_state, final_weight, final_label, final_log_q, measurement = _run_stage(
        model, frame, syndrome, p, proposal, rng, burn_state,
        burn_weight, burn_label, burn_log_q, measurement_steps,
    )
    return {
        "seed": np.asarray(int(seed), dtype=np.uint64),
        "initial_state_packed": np.packbits(initial, bitorder="little"),
        "initial_weight": np.asarray(initial_weight, dtype=np.int32),
        "initial_label": np.asarray(initial_label, dtype=np.uint64),
        "initial_log_q": np.asarray(initial_log_q, dtype=np.float64),
        "burn_end_state_packed": np.packbits(burn_state, bitorder="little"),
        "burn_end_weight": np.asarray(burn_weight, dtype=np.int32),
        "burn_end_label": np.asarray(burn_label, dtype=np.uint64),
        "burn_end_log_q": np.asarray(burn_log_q, dtype=np.float64),
        "final_state_packed": np.packbits(final_state, bitorder="little"),
        "final_weight": np.asarray(final_weight, dtype=np.int32),
        "final_label": np.asarray(final_label, dtype=np.uint64),
        "final_log_q": np.asarray(final_log_q, dtype=np.float64),
        **{f"burn_{key}": value for key, value in burn.items()},
        **{f"measurement_{key}": value for key, value in measurement.items()},
    }


def replay_bp_imh_trajectory(model, frame, syndrome, p, proposal, initial_state,
                             seed, raw, *, burn_steps, measurement_steps):
    """Rerun every proposal and acceptance decision from the original seed."""
    replay = run_bp_imh_trajectory(
        model, frame, syndrome, p, proposal, initial_state, seed,
        burn_steps=burn_steps, measurement_steps=measurement_steps,
    )
    _require(set(raw) == set(replay), "BP-IMH replay field set changed")
    for key, expected in replay.items():
        actual = np.asarray(raw[key])
        expected = np.asarray(expected)
        _require(actual.dtype == expected.dtype and actual.shape == expected.shape
                 and np.array_equal(actual, expected),
                 f"BP-IMH replay mismatch: {key}")
    return True


def validate_bp_imh_transcript(model, frame, syndrome, p, proposal, raw,
                               *, burn_steps, measurement_steps):
    """Independently rebuild state-derived values and MH decisions from raw."""
    syndrome = _bits(syndrome, ndim=1, name="BP-IMH syndrome")
    width = (model.num_qubits + 7) // 8
    initial = np.unpackbits(
        np.asarray(raw["initial_state_packed"], dtype=np.uint8), bitorder="little",
        count=model.num_qubits,
    ).astype(np.uint8, copy=False)
    _require(initial.shape == (model.num_qubits,)
             and not _hard_residual(model.H_check, initial, syndrome).any(),
             "BP-IMH raw initial state is invalid")
    state = initial.copy()
    weight = int(state.sum())
    label = int(state_label(frame, state))
    current_log_q = float(proposal.log_probability_state(state))
    _require(int(raw["initial_weight"]) == weight
             and int(raw["initial_label"]) == label
             and float(raw["initial_log_q"]) == current_log_q,
             "BP-IMH raw initial derived values changed")

    for stage, steps in (("burn", int(burn_steps)), ("measurement", int(measurement_steps))):
        proposed_packed = np.asarray(raw[f"{stage}_proposal_states_packed"], dtype=np.uint8)
        states_packed = np.asarray(raw[f"{stage}_states_packed"], dtype=np.uint8)
        _require(proposed_packed.shape == states_packed.shape == (steps, width),
                 f"BP-IMH {stage} packed states changed shape")
        proposed_states = np.unpackbits(
            proposed_packed, axis=1, count=model.num_qubits, bitorder="little",
        ).astype(np.uint8, copy=False)
        stored_states = np.unpackbits(
            states_packed, axis=1, count=model.num_qubits, bitorder="little",
        ).astype(np.uint8, copy=False)
        for index, proposed in enumerate(proposed_states):
            _require(not _hard_residual(model.H_check, proposed, syndrome).any(),
                     f"BP-IMH {stage} proposal escaped the hard coset")
            proposed_weight = int(proposed.sum())
            proposed_label = int(state_label(frame, proposed))
            proposed_log_q = float(proposal.log_probability_state(proposed))
            source = int(raw[f"{stage}_proposal_source_indices"][index])
            component = int(raw[f"{stage}_proposal_component_indices"][index])
            _require(0 <= source < proposal.num_sources
                     and 0 <= component < proposal.proposals[source].num_components,
                     f"BP-IMH {stage} proposal provenance is invalid")
            source_log_q = float(proposal.proposals[source].log_probability_state(proposed))
            ratio = log_acceptance_ratio(
                weight, current_log_q, proposed_weight, proposed_log_q, p,
            )
            decision, clipped = acceptance_decision(
                ratio, float(raw[f"{stage}_acceptance_uniforms"][index]),
            )
            changed = bool(decision and not np.array_equal(state, proposed))
            if decision:
                state = proposed.copy()
                weight = proposed_weight
                label = proposed_label
                current_log_q = proposed_log_q
            _require(int(raw[f"{stage}_proposal_labels"][index]) == proposed_label
                     and int(raw[f"{stage}_proposal_weights"][index]) == proposed_weight
                     and float(raw[f"{stage}_proposal_source_log_q"][index]) == source_log_q
                     and float(raw[f"{stage}_proposal_log_q"][index]) == proposed_log_q
                     and float(raw[f"{stage}_log_acceptance"][index]) == clipped
                     and bool(raw[f"{stage}_accepted"][index]) == decision
                     and bool(raw[f"{stage}_state_changed"][index]) == changed
                     and np.array_equal(stored_states[index], state)
                     and int(raw[f"{stage}_labels"][index]) == label
                     and int(raw[f"{stage}_weights"][index]) == weight
                     and float(raw[f"{stage}_current_log_q"][index]) == current_log_q,
                     f"BP-IMH {stage} transcript changed at step {index}")
        prefix = "burn_end" if stage == "burn" else "final"
        _require(np.array_equal(
            np.asarray(raw[f"{prefix}_state_packed"], dtype=np.uint8),
            np.packbits(state, bitorder="little"),
        ) and int(raw[f"{prefix}_weight"]) == weight
            and int(raw[f"{prefix}_label"]) == label
            and float(raw[f"{prefix}_log_q"]) == current_log_q,
            f"BP-IMH {stage} endpoint changed")
    return True
