"""Outcome-free geometry probe for legal m8 T1 initialization candidates."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.expander_code.exp102.exp102_pipeline.io import atomic_json, sha256_file, sha256_json
from data.expander_code.exp102.exp102_pipeline.q0_center_preserving import build_dressed_logical_catalog
from data.expander_code.exp102.exp102_pipeline.q0_global import state_label
from data.expander_code.exp102.exp102_pipeline.q0_hgp_collapsed import split_hgp_state
from data.expander_code.exp102.exp102_pipeline.q0_hgp_screen import _disorder
from data.expander_code.exp102.exp102_pipeline.q0_houdayer_pair import deterministic_low_energy_logical_starts
from data.expander_code.exp102.exp102_pipeline.q0_logical_stratified import load_logical_stratified_frozen_artifact
from data.expander_code.exp102.exp102_pipeline.registry import load_frozen_code, load_registry
from data.expander_code.exp102.exp102_pipeline.worker import build_model


ROOT = Path(__file__).resolve().parent
EXP102_ROOT = ROOT.parents[1]
REGISTRY_PATH = EXP102_ROOT / "registry/registry.json"
CELL = {"code_id": "m08_c06", "disorder_index": 0, "disorder_source": "attempt022", "p": 0.04}
MAP_RELPATH = Path(
    "validation/013_q0_hgp_global_screen_20260722/remote_run/"
    "exp102_q0_hgp_screen_v2_20260722_4d134ee/hgp_global/artifacts/"
    "map_artifacts/b4ccfd16bed39aac912b8aa129485dbc2f7ac696724e12f38b23a5f83da521fa.npz"
)
LSI_RELPATH = Path(
    "validation/015_q0_logical_stratified_v0b_20260723/remote_run/"
    "exp102_q0_lsi_v0d_20260723_9f0c473/pulled_run/artifacts/artifacts/lsi_imh_tau_10.npz"
)


def _profile(name, state, H, frame):
    _A, B = split_hgp_state(state, H)
    return {
        "b_packed_hex": np.packbits(B.reshape(-1), bitorder="little").tobytes().hex(),
        "b_weight": int(B.sum()),
        "label": int(state_label(frame, state)),
        "name": name,
        "state_packed_hex": np.packbits(state, bitorder="little").tobytes().hex(),
        "weight": int(state.sum()),
    }


def main():
    registry = load_registry(REGISTRY_PATH)
    _unused, code, H = load_frozen_code(REGISTRY_PATH, CELL["code_id"])
    model, frame = build_model(H)
    _seed, planted, syndrome = _disorder(registry, code, model, CELL)
    with np.load(EXP102_ROOT / MAP_RELPATH, allow_pickle=False) as archive:
        map_anchors = np.asarray(archive["anchors"], dtype=np.uint8)
    artifact = load_logical_stratified_frozen_artifact(EXP102_ROOT / LSI_RELPATH, model, frame)
    catalog = build_dressed_logical_catalog(model, frame, artifact, max_moves=127)
    catalog_anchors = catalog.unpack_anchors()
    logical = deterministic_low_energy_logical_starts(
        model, frame, planted, count=8, orders=(1, 2, 3),
    )
    candidates = [("P", planted), ("MAP_B0", map_anchors[0]), ("MAP_B1", map_anchors[1])]
    candidates.extend((f"L{index}", row["state"]) for index, row in enumerate(logical))
    candidates.extend((f"CAT{index:03d}", state) for index, state in enumerate(catalog_anchors))
    profiles = [_profile(name, state, H, frame) for name, state in candidates]
    states = {name: np.asarray(state, dtype=np.uint8) for name, state in candidates}
    base_names = ["P", "MAP_B0", "MAP_B1"] + [f"L{index}" for index in range(8)]
    pairwise = []
    for left_index, left in enumerate(base_names):
        for right in base_names[left_index + 1:]:
            _a_left, b_left = split_hgp_state(states[left], H)
            _a_right, b_right = split_hgp_state(states[right], H)
            pairwise.append({
                "b_hamming": int((b_left ^ b_right).sum()),
                "full_hamming": int((states[left] ^ states[right]).sum()),
                "label_hamming": int((state_label(frame, states[left]) ^ state_label(frame, states[right])).bit_count()),
                "left": left,
                "right": right,
            })

    # Select eight catalog anchors without looking at sampler results: maximize
    # the minimum B distance to P/MAP and already selected anchors, then prefer
    # lower physical weight and canonical catalog index.
    selected = []
    reference_states = [planted, map_anchors[0], map_anchors[1]]
    reference_b = [split_hgp_state(state, H)[1] for state in reference_states]
    unused = list(range(catalog.size))
    while len(selected) < 8:
        scored = []
        for index in unused:
            state = catalog_anchors[index]
            _a, block = split_hgp_state(state, H)
            distances = [int((block ^ other).sum()) for other in reference_b]
            scored.append((min(distances), -int(state.sum()), -index, index))
        chosen = max(scored)[-1]
        selected.append(chosen)
        unused.remove(chosen)
        reference_b.append(split_hgp_state(catalog_anchors[chosen], H)[1])
    selected_profiles = [_profile(f"CAT{index:03d}", catalog_anchors[index], H, frame) for index in selected]
    core = {
        "catalog_sha256": catalog.catalog_sha256,
        "cell": CELL,
        "lsi_file_sha256": sha256_file(EXP102_ROOT / LSI_RELPATH),
        "map_file_sha256": sha256_file(EXP102_ROOT / MAP_RELPATH),
        "pairwise_base": pairwise,
        "profiles": profiles,
        "selected_catalog_indices": selected,
        "selected_catalog_profiles": selected_profiles,
        "selection_rule": "greedy_maximin_B_distance_then_low_weight_then_catalog_index",
        "syndrome_weight": int(syndrome.sum()),
    }
    report = {**core, "report_sha256": sha256_json(core)}
    output = ROOT / "initialization_geometry_report.json"
    if output.exists():
        raise RuntimeError("initialization geometry report already exists")
    atomic_json(output, report)
    print(json.dumps({
        "base": [row for row in profiles if row["name"] in base_names],
        "selected_catalog_indices": selected,
        "selected_catalog_profiles": selected_profiles,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
