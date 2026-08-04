from dataclasses import replace

import numpy as np
import pytest

from data.expander_code.exp102.exp102_pipeline.q0_logical_stratified import (
    DecodedCandidateTranscript,
    LogicalStratifiedConfig,
    LogicalStratifiedConflictError,
    LogicalStratifiedSeedIdentity,
    STRATIFIED_METHOD_ID,
    _bits_to_uint64,
    _gf2_row_products,
    _signature_rank_masks,
    _uint64_to_bits,
    build_logical_stratified_frozen_artifact,
    build_hgp_signature_codebook,
    build_logical_stratified_proposal,
    build_stratified_anchor_catalog,
    catalog_character_probability_mass,
    generate_bplsd_stratified_catalog,
    load_logical_stratified_frozen_artifact,
    replay_logical_stratified_trajectory,
    run_logical_stratified_trajectory,
    validate_decoded_candidate_transcript,
    validate_hgp_signature_codebook,
    validate_logical_stratified_frozen_artifact,
    validate_logical_stratified_proposal,
    validate_stratified_anchor_catalog,
    write_logical_stratified_frozen_artifact,
)
from data.expander_code.exp102.exp102_pipeline.q0_map_mixture import (
    enumerate_affine_states,
    independence_transition_matrix,
)
from data.expander_code.exp102.exp102_pipeline.worker import build_model


def _model(classical):
    return build_model(np.asarray(classical, dtype=np.uint8))


def _syndrome(model, nonzero=True):
    witness = np.zeros(model.num_qubits, dtype=np.uint8)
    if nonzero:
        witness[[0, model.num_qubits - 1]] = 1
    return (
        model.H_check.astype(np.int64) @ witness.astype(np.int64) % 2
    ).astype(np.uint8)


def _coset_states(model, syndrome):
    generators = np.vstack((model.stabilizer_rows, model.logical_move_basis))
    base = model.logical_sector_section.apply(syndrome, strict=True)
    states = np.repeat(base[None, :], 1 << generators.shape[0], axis=0)
    for coefficient in range(states.shape[0]):
        for bit, row in enumerate(generators):
            if (coefficient >> bit) & 1:
                states[coefficient] ^= row
    return states


def _label(frame, state):
    bits = frame.label_of(state)
    return sum(int(value) << bit for bit, value in enumerate(bits))


def _exact_sector_catalog_inputs(model, frame, syndrome):
    states = _coset_states(model, syndrome)
    by_label = {}
    for state in states:
        label = _label(frame, state)
        candidate = (int(state.sum()), np.packbits(state, bitorder="little").tobytes(), state)
        previous = by_label.get(label)
        if previous is None or candidate[:2] < previous[:2]:
            by_label[label] = candidate
    base_label, base_record = min(
        by_label.items(), key=lambda item: (item[1][0], item[0], item[1][1]),
    )
    candidate_labels = [label for label in sorted(by_label) if label != base_label]
    anchors = np.asarray([by_label[label][2] for label in candidate_labels], dtype=np.uint8)
    deltas = np.asarray([label ^ base_label for label in candidate_labels], dtype=np.uint64)
    move_weights = np.asarray(
        [int(np.count_nonzero(anchors[index] ^ base_record[2]))
         for index in range(anchors.shape[0])],
        dtype=np.int32,
    )
    return base_record[2], anchors, deltas, move_weights, states


def _catalog(model, frame, syndrome, max_anchors):
    base, anchors, deltas, move_weights, states = _exact_sector_catalog_inputs(
        model, frame, syndrome,
    )
    catalog = build_stratified_anchor_catalog(
        model, frame, syndrome, base, anchors, deltas, move_weights,
        max_anchors=max_anchors, decoder_identity="tiny_exact",
        codebook_sha256="a" * 64,
    )
    return catalog, states


def test_sparse_gf2_candidate_replay_matches_dense_integer_product():
    matrix = np.asarray([
        [0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 0, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 1, 0, 0, 1, 1, 0],
    ], dtype=np.uint8)
    states = np.asarray([
        [0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 0, 1],
        [1, 1, 0, 0, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1],
    ], dtype=np.uint8)
    expected = (
        matrix.astype(np.int64) @ states.T.astype(np.int64) % 2
    ).astype(np.uint8)
    assert np.array_equal(_gf2_row_products(matrix, states), expected)


@pytest.mark.parametrize("classical", [
    [[1, 1, 1]],
    [[1, 1, 0], [0, 1, 1]],
])
def test_hgp_signature_codebook_is_deterministic_unique_and_rank_complete(classical):
    H = np.asarray(classical, dtype=np.uint8)
    model, frame = _model(H)
    first = build_hgp_signature_codebook(model, frame, H)
    second = build_hgp_signature_codebook(model, frame, H)
    assert first.codebook_sha256 == second.codebook_sha256
    assert np.array_equal(first.signatures, second.signatures)
    assert np.array_equal(first.logical_move_weights, second.logical_move_weights)
    assert np.unique(first.signatures).size == first.size
    assert not np.any(first.signatures == np.uint64(0))
    assert _signature_rank_masks(first.signatures, model.k) == model.k


def test_catalog_is_hard_coset_exact_label_exact_and_affine_rank_complete():
    model, frame = _model([[1, 1, 1]])
    syndrome = _syndrome(model)
    catalog, _ = _catalog(model, frame, syndrome, max_anchors=model.k + 1)
    validate_stratified_anchor_catalog(model, frame, syndrome, catalog)
    recovered = (
        model.H_check.astype(np.int64) @ catalog.anchors.T.astype(np.int64) % 2
    ).T.astype(np.uint8)
    assert np.array_equal(recovered, np.repeat(syndrome[None, :], catalog.size, axis=0))
    assert np.array_equal(
        catalog.labels,
        np.asarray([_label(frame, row) for row in catalog.anchors], dtype=np.uint64),
    )
    assert _signature_rank_masks(catalog.label_deltas[1:], model.k) == model.k
    assert catalog.size == model.k + 1

    corrupted = catalog.anchors.copy()
    corrupted[1, 0] ^= np.uint8(1)
    with pytest.raises(LogicalStratifiedConflictError, match="hard coset"):
        validate_stratified_anchor_catalog(
            model, frame, syndrome, replace(catalog, anchors=corrupted),
        )


def test_bplsd_catalog_generation_is_worker_count_independent_and_exact():
    H = np.asarray([[1, 1, 1]], dtype=np.uint8)
    model, frame = _model(H)
    syndrome = _syndrome(model)
    codebook = build_hgp_signature_codebook(model, frame, H)
    serial = generate_bplsd_stratified_catalog(
        model, frame, H, syndrome, 0.10, codebook,
        max_anchors=8, decoder_max_iter=16, chunk_size=3, num_workers=1,
    )
    parallel = generate_bplsd_stratified_catalog(
        model, frame, H, syndrome, 0.10, codebook,
        max_anchors=8, decoder_max_iter=16, chunk_size=3, num_workers=2,
    )
    assert serial.catalog_sha256 == parallel.catalog_sha256
    assert serial.candidate_transcript_sha256 == parallel.candidate_transcript_sha256
    assert np.array_equal(serial.anchors, parallel.anchors)
    assert np.array_equal(serial.labels, parallel.labels)
    validate_stratified_anchor_catalog(model, frame, syndrome, serial)


@pytest.mark.parametrize("classical", [
    [[1, 1, 1]],
    [[1, 1, 0], [0, 1, 1]],
])
@pytest.mark.parametrize("nonzero", [False, True])
@pytest.mark.parametrize("p", [0.04, 0.10, 0.25])
def test_stratified_proposal_normalization_detailed_balance_and_stationarity(
        classical, nonzero, p):
    model, frame = _model(classical)
    syndrome = _syndrome(model, nonzero=nonzero)
    catalog, states = _catalog(model, frame, syndrome, max_anchors=model.k + 1)
    proposal = build_logical_stratified_proposal(
        model, frame, catalog, p=p,
        uniform_label_probability=0.03, catalog_uniform_floor=0.07,
    )
    coordinates, enumerated = enumerate_affine_states(proposal.coordinates)
    assert {
        np.packbits(row, bitorder="little").tobytes() for row in states
    } == {
        np.packbits(row, bitorder="little").tobytes() for row in enumerated
    }
    q = np.exp(np.asarray([
        proposal.log_probability_coordinates(row) for row in coordinates
    ]))
    assert np.allclose(
        np.asarray([proposal.log_probability_state(row) for row in enumerated]),
        np.asarray([proposal.log_probability_coordinates(row) for row in coordinates]),
        atol=1e-14, rtol=0.0,
    )
    assert np.all(q > 0.0)
    assert float(q.sum()) == pytest.approx(1.0, abs=3e-14)
    target = (p / (1.0 - p)) ** enumerated.sum(axis=1)
    target /= target.sum()
    transition = independence_transition_matrix(target, q)
    assert np.max(np.abs(transition.sum(axis=1) - 1.0)) <= 3e-15
    flow = target[:, None] * transition
    assert np.max(np.abs(flow - flow.T)) <= 1e-13
    assert np.max(np.abs(target @ transition - target)) <= 1e-13


def test_uniform_label_defense_covers_non_catalog_labels_and_affine_offset():
    model, frame = _model([[1, 1, 1]])
    syndrome = _syndrome(model)
    catalog, _ = _catalog(model, frame, syndrome, max_anchors=model.k + 1)
    proposal = build_logical_stratified_proposal(model, frame, catalog, p=0.04)
    coordinates, states = enumerate_affine_states(proposal.coordinates)
    for coordinate, state in zip(coordinates, states):
        assert proposal.label_from_coordinates(coordinate) == _label(frame, state)
        recovered = proposal.logical_coordinates_for_label(_label(frame, state))
        assert np.array_equal(
            recovered,
            coordinate[proposal.stabilizer_dimension:],
        )
        assert math_is_finite(proposal.log_probability_coordinates(coordinate))


def math_is_finite(value):
    return bool(np.isfinite(float(value)))


def test_runner_records_actual_cross_label_transport_separately_from_acceptance():
    H = np.asarray([[1, 1, 1]], dtype=np.uint8)
    model, frame = _model(H)
    syndrome = _syndrome(model)
    codebook = build_hgp_signature_codebook(model, frame, H)
    catalog = generate_bplsd_stratified_catalog(
        model, frame, H, syndrome, 0.25, codebook,
        max_anchors=1 << model.k, decoder_max_iter=16, chunk_size=3,
    )
    proposal = build_logical_stratified_proposal(
        model, frame, catalog, p=0.25,
        uniform_label_probability=0.1, catalog_uniform_floor=0.1,
    )
    config = LogicalStratifiedConfig(p=0.25, burn_steps=8, measurement_steps=16)
    seed = LogicalStratifiedSeedIdentity(
        source_commit="1" * 40,
        config_sha256="2" * 64,
        registry_sha256="3" * 64,
        cell_fingerprint="4" * 64,
        init_family="U",
        trajectory_index=0,
        resource_tier="test",
        trajectory_namespace="q0_logical_stratified_test",
    )
    artifact = build_logical_stratified_frozen_artifact(
        model, frame, H, syndrome, codebook, catalog, proposal,
        identity={
            "source_commit": seed.source_commit,
            "config_sha256": seed.config_sha256,
            "registry_sha256": seed.registry_sha256,
            "cell_fingerprint": seed.cell_fingerprint,
            "purpose": "tiny_test",
        },
    )
    raw = run_logical_stratified_trajectory(
        model, frame, syndrome, config, seed, catalog.anchors[0],
        artifact=artifact,
    )
    assert raw["method_id"] == STRATIFIED_METHOD_ID
    assert not raw["measurement_residual_weights"].any()
    expected = (
        raw["measurement_accepted"].astype(bool)
        & raw["measurement_state_changed"].astype(bool)
        & (raw["measurement_proposal_labels"]
           != np.concatenate((
               np.asarray([raw["burn_label"]], dtype=np.uint64),
               raw["measurement_labels"][:-1],
           )))
    )
    assert np.array_equal(raw["measurement_label_changed"].astype(bool), expected)
    assert int(raw["measurement_cross_label_changes"]) == int(expected.sum())
    assert replay_logical_stratified_trajectory(
        model, frame, syndrome, config, seed, catalog.anchors[0], raw,
        artifact=artifact,
    )
    tampered = dict(raw)
    tampered["burn_accepted"] = raw["burn_accepted"].copy()
    tampered["burn_accepted"][0] ^= np.uint8(1)
    with pytest.raises(LogicalStratifiedConflictError, match="replay mismatch"):
        replay_logical_stratified_trajectory(
            model, frame, syndrome, config, seed, catalog.anchors[0], tampered,
            artifact=artifact,
        )


def test_complete_decoder_transcript_and_frozen_artifact_reject_tampering(tmp_path):
    H = np.asarray([[1, 1, 1]], dtype=np.uint8)
    model, frame = _model(H)
    syndrome = _syndrome(model)
    codebook = build_hgp_signature_codebook(model, frame, H)
    catalog = generate_bplsd_stratified_catalog(
        model, frame, H, syndrome, 0.10, codebook,
        max_anchors=8, decoder_max_iter=16, chunk_size=3,
    )
    transcript = catalog.candidate_transcript
    assert isinstance(transcript, DecodedCandidateTranscript)
    validate_decoded_candidate_transcript(
        model, frame, H, syndrome, codebook, catalog.anchors[0], transcript,
    )
    proposal = build_logical_stratified_proposal(
        model, frame, catalog, p=0.10, alpha_temperature=0.5,
    )
    validate_logical_stratified_proposal(model, frame, syndrome, catalog, proposal)
    artifact = build_logical_stratified_frozen_artifact(
        model, frame, H, syndrome, codebook, catalog, proposal,
        identity={"purpose": "tamper_test"},
    )
    validate_logical_stratified_frozen_artifact(model, frame, artifact)
    artifact_path = tmp_path / "artifact.npz"
    written = write_logical_stratified_frozen_artifact(
        artifact_path, model, frame, artifact,
    )
    assert len(written["artifact_content_sha256"]) == 64
    assert len(written["artifact_file_sha256"]) == 64
    loaded = load_logical_stratified_frozen_artifact(
        artifact_path, model, frame,
    )
    assert loaded.descriptor == artifact.descriptor
    with pytest.raises(FileExistsError):
        write_logical_stratified_frozen_artifact(artifact_path, model, frame, artifact)

    valid_index = int(np.flatnonzero(transcript.valid)[0])
    corrupted_packed = transcript.decoded_packed.copy()
    corrupted_packed[valid_index, 0] ^= np.uint8(1)
    tampered_transcript = replace(transcript, decoded_packed=corrupted_packed)
    with pytest.raises(LogicalStratifiedConflictError):
        validate_decoded_candidate_transcript(
            model, frame, H, syndrome, codebook, catalog.anchors[0],
            tampered_transcript,
        )

    corrupted_moves = codebook.moves_packed.copy()
    corrupted_moves[0, 0] ^= np.uint8(1)
    tampered_codebook = replace(codebook, moves_packed=corrupted_moves)
    with pytest.raises(LogicalStratifiedConflictError):
        validate_hgp_signature_codebook(model, frame, H, tampered_codebook)


def test_pre_registered_alpha_temperatures_and_character_balance_are_explicit():
    model, frame = _model([[1, 1, 1]])
    syndrome = _syndrome(model)
    catalog, _ = _catalog(model, frame, syndrome, max_anchors=1 << model.k)
    cold = build_logical_stratified_proposal(
        model, frame, catalog, p=0.04, alpha_temperature=1.0,
    )
    warm = build_logical_stratified_proposal(
        model, frame, catalog, p=0.04, alpha_temperature=0.5,
    )
    assert cold.proposal_sha256 != warm.proposal_sha256
    masks = np.asarray([1 << bit for bit in range(model.k)], dtype=np.uint64)
    for proposal in (cold, warm):
        plus, minus = catalog_character_probability_mass(proposal, masks)
        assert np.all(plus > 0.0)
        assert np.all(minus > 0.0)
        assert np.allclose(plus + minus, 1.0)
    with pytest.raises(ValueError, match="pre-registered"):
        build_logical_stratified_proposal(
            model, frame, catalog, p=0.04, alpha_temperature=0.75,
        )


def test_strict_input_types_and_uint64_bit_63_are_not_silently_reinterpreted():
    with pytest.raises(ValueError, match="positive integer"):
        LogicalStratifiedConfig(p=0.04, burn_steps=True, measurement_steps=8)
    with pytest.raises(ValueError, match="positive integer"):
        LogicalStratifiedConfig(p=0.04, burn_steps=4.0, measurement_steps=8)
    with pytest.raises(ValueError, match="P, U, or L"):
        LogicalStratifiedSeedIdentity(
            source_commit="1" * 40, config_sha256="2" * 64,
            registry_sha256="3" * 64, cell_fingerprint="4" * 64,
            init_family="zero", trajectory_index=0, resource_tier="test",
            trajectory_namespace="strict_test",
        )
    highest = np.uint64(1) << np.uint64(63)
    bits = _uint64_to_bits(highest, 64)
    assert bits[63] == 1
    assert _bits_to_uint64(bits) == highest
