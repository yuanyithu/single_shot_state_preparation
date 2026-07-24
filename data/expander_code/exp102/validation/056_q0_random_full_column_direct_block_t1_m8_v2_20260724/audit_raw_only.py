"""Independent allow-pickle-false audit of validation-056 raw and estimates.

This audit does not import or call the trajectory sampler, replay runner, or
primary analyzer.  It reconstructs the hard-coset algebra, collapsed-B trace,
labels, weights, likelihoods, character q_top/D2, and MAP bridge directly from
the immutable raw files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from data.expander_code.exp102.exp102_pipeline.io import (
    atomic_json,
    canonical_json,
    sha256_file,
    sha256_json,
)
from data.expander_code.exp102.exp102_pipeline.worker import build_model


AUDIT_VERSION = "exp102.q0_random_full_column_direct_block.t1_m8.audit.v2"
CONTRACT_VERSION = "exp102.q0_random_full_column_direct_block.t1_m8.v2"
RAW_VERSION = "exp102.q0_random_full_column_direct_block.t1_m8.raw.v2"
NODE_REPORT_VERSION = (
    "exp102.q0_random_full_column_direct_block.t1_m8.node_report.v2"
)
REPORT_VERSION = "exp102.q0_random_full_column_direct_block.t1_m8.report.v2"
DIRECT_VERSION = "exp102.q0_hgp_random_full_column_direct_block.v1"
DIRECT_ENGINE = "numba_direct_positive_fixed_block_12"
METHOD_ID = "RFCG-C24-DPB12-S1"
SOURCE_COMMIT = "6933e319b27840976f34e27c0d11313b6973cbe3"
ARCHIVE_SHA256 = "b62d0e22b7e37f8ca90186cc1d6d5bd9fe6e8d9b2568d9de569fd275ebb13eb5"
SOURCE_MANIFEST_SHA256 = (
    "135eb089bf1ca60a1009965847fbefef6c9bc238ed3db52258f311845c817e48"
)
CONFIG_SHA256 = "70285cf7ae8ecb7d062af7d72980e504edb42313d3e6708ab1e26a3bfbdf899d"
CONTROL_CONTENT_SHA256 = (
    "49665fb9b42d977edfa3ee23218effd7c11563f49715b09a4307aa63edf79c48"
)
SCHEDULE_SHA256 = "ca057fbc2c76de2715dc7318f2f2c5d15567aeef403583df6dc958c28eec58d3"
PREFLIGHT_SHA256 = "4a868f91e2ed90db21733c49acd07855f42d8f7d4585a94a794641b088d2b0f2"
FAMILIES = ("P", "U", "M0", "M1", "S")
P_VALUE = 0.04
MASK64 = (1 << 64) - 1
ROOT = Path(__file__).resolve().parent
EXP102_ROOT = ROOT.parents[1]
CONFIG_PATH = (
    EXP102_ROOT / "config/q0_random_full_column_direct_block.t1_m8.v2.json"
)


class AuditError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def load_canonical(path):
    path = Path(path)
    serialized = path.read_text(encoding="ascii")
    value = json.loads(serialized)
    require(serialized == canonical_json(value) + "\n", f"noncanonical JSON: {path}")
    return value


def verify_self_hash(value, field):
    require(field in value, f"missing self-hash: {field}")
    core = {key: item for key, item in value.items() if key != field}
    require(sha256_json(core) == value[field], f"self-hash mismatch: {field}")


def content_sha(control_version, metadata, arrays):
    metadata = dict(metadata)
    metadata.pop("control_content_sha256", None)
    digest = hashlib.sha256(control_version.encode("ascii") + b"\0")
    digest.update(canonical_json(metadata).encode("ascii") + b"\0")
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("ascii") + b"\0")
        digest.update(value.dtype.str.encode("ascii") + b"\0")
        digest.update(np.asarray(value.shape, dtype=">u8").tobytes())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def gf2_rank(rows):
    matrix = np.asarray(rows, dtype=np.uint8).copy()
    rank = 0
    for column in range(matrix.shape[1]):
        pivots = np.flatnonzero(matrix[rank:, column])
        if not pivots.size:
            continue
        pivot = rank + int(pivots[0])
        matrix[[rank, pivot]] = matrix[[pivot, rank]]
        active = np.flatnonzero(matrix[:, column])
        active = active[active != rank]
        matrix[active] ^= matrix[rank]
        rank += 1
        if rank == matrix.shape[0]:
            break
    return rank


def splitmix64(seed, count):
    state = int(seed) & MASK64
    result = []
    for _ in range(int(count)):
        state = (state + 0x9E3779B97F4A7C15) & MASK64
        value = state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
        result.append(value ^ (value >> 31))
    return result


def portable_bits(seed, count):
    s0, s1 = splitmix64(seed, 2)
    if s0 == 0 and s1 == 0:
        s1 = 1
    result = np.empty(int(count), dtype=np.uint8)
    for index in range(int(count)):
        x, y = s0, s1
        s0 = y
        x = (x ^ (x << 23)) & MASK64
        x ^= x >> 17
        x ^= y ^ (y >> 26)
        s1 = x
        result[index] = (x + y) & 1
    return result


def uniform_hard_coset_state(model, syndrome, seed):
    stabilizers = np.asarray(model.stabilizer_rows, dtype=np.uint8)
    logicals = np.asarray(model.logical_move_basis, dtype=np.uint8)
    check_rank = gf2_rank(model.H_check)
    affine_dimension = model.num_qubits - check_rank
    require(gf2_rank(stabilizers) == stabilizers.shape[0],
            "stabilizer basis is dependent")
    basis = np.vstack((stabilizers, logicals))
    require(
        basis.shape[0] == affine_dimension
        and gf2_rank(basis) == affine_dimension,
        "stabilizer/logical rows are not a full hard-coset basis",
    )
    state = model.logical_sector_section.apply(
        syndrome, strict=True,
    ).astype(np.uint8)
    coefficients = portable_bits(seed, basis.shape[0])
    for row in np.flatnonzero(coefficients):
        state ^= basis[row]
    return np.ascontiguousarray(state)


def qubit_signatures(frame):
    require(int(frame.k) <= 64, "logical labels require k <= 64")
    signatures = np.zeros(frame.num_qubits, dtype=np.uint64)
    for bit in range(int(frame.k)):
        signatures[np.asarray(frame.W_basis[bit], dtype=bool)] ^= (
            np.uint64(1) << np.uint64(bit)
        )
    return signatures


def classical_coset_mass(H, p):
    H = np.asarray(H, dtype=np.uint8)
    rows = H.shape[0]
    require(rows <= 24, "independent mass table exceeded its width cap")
    powers = np.left_shift(np.uint64(1), np.arange(rows, dtype=np.uint64))
    column_masks = np.einsum(
        "rn,r->n", H.astype(np.uint64), powers, optimize=False,
    )
    indices = np.arange(1 << rows, dtype=np.uint64)
    mass = np.zeros(1 << rows, dtype=np.float64)
    mass[0] = 1.0
    for mask in column_masks:
        previous = mass
        mass = (1.0 - float(p)) * previous + float(p) * previous[
            indices ^ mask
        ]
    require(
        np.all(np.isfinite(mass)) and np.all(mass > 0.0)
        and abs(float(mass.sum()) - 1.0) <= 5e-13,
        "independent classical coset mass is invalid",
    )
    return np.ascontiguousarray(mass)


def unpack(packed, width):
    return np.unpackbits(
        np.asarray(packed, dtype=np.uint8), axis=-1, count=int(width),
        bitorder="little",
    ).astype(np.uint8, copy=False)


def columns_to_b_packed(columns, r):
    columns = np.asarray(columns, dtype=np.uint32)
    bits = (
        columns[:, None, :] >> np.arange(r, dtype=np.uint32)[None, :, None]
    ) & np.uint32(1)
    return np.packbits(
        bits.astype(np.uint8).reshape(columns.shape[0], r * r),
        axis=1, bitorder="little",
    )


def state_b_columns(state, H):
    r, n = H.shape
    block = np.asarray(state, dtype=np.uint8)[n * n:].reshape(r, r)
    powers = np.left_shift(np.uint32(1), np.arange(r, dtype=np.uint32))
    return np.einsum(
        "ij,i->j", block.astype(np.uint32), powers, optimize=False,
    ).astype(np.uint32)


def labels_from_states(states, signatures, chunk=256):
    result = np.empty(states.shape[0], dtype=np.uint64)
    signatures = np.asarray(signatures, dtype=np.uint64)
    for start in range(0, states.shape[0], chunk):
        stop = min(start + chunk, states.shape[0])
        result[start:stop] = np.bitwise_xor.reduce(
            np.where(
                states[start:stop].astype(bool),
                signatures[None, :],
                np.uint64(0),
            ),
            axis=1,
        )
    return result


def verify_hard_coset(states, H, syndrome, chunk=256):
    r, n = H.shape
    target = np.asarray(syndrome, dtype=np.uint8).reshape(r, n)
    H64 = H.astype(np.int64)
    for start in range(0, states.shape[0], chunk):
        stop = min(start + chunk, states.shape[0])
        batch = states[start:stop]
        A = batch[:, :n * n].reshape(-1, n, n).astype(np.int64)
        B = batch[:, n * n:].reshape(-1, r, r).astype(np.int64)
        observed = (
            np.einsum("ij,tjk->tik", H64, A, optimize=False)
            + np.einsum("tij,jk->tik", B, H64, optimize=False)
        ) & 1
        require(
            np.array_equal(
                observed.astype(np.uint8),
                np.repeat(target[None, :, :], stop - start, axis=0),
            ),
            "a measurement state left the hard coset",
        )


def replay_b(initial, selected, old, new):
    state = np.asarray(initial, dtype=np.uint32).copy()
    trace = np.empty((len(selected), state.size), dtype=np.uint32)
    changes = 0
    changed_bits = 0
    for clock, column_value in enumerate(selected):
        column = int(column_value)
        require(
            0 <= column < state.size and int(state[column]) == int(old[clock]),
            "B transcript old column mismatch",
        )
        delta = int(old[clock]) ^ int(new[clock])
        changes += int(delta != 0)
        changed_bits += delta.bit_count()
        state[column] = new[clock]
        trace[clock] = state
    return trace, changes, changed_bits


def b_log_likelihood(b_packed, H, syndrome, log_mass):
    r, n = H.shape
    bits = unpack(b_packed, r * r)
    B = bits.reshape(-1, r, r).astype(np.int64, copy=False)
    bh = np.einsum(
        "tij,jk->tik", B, H.astype(np.int64), optimize=False,
    ) & 1
    Y = np.asarray(syndrome, dtype=np.uint8).reshape(r, n)
    syndromes = bh ^ Y[None, :, :]
    powers = np.left_shift(np.int64(1), np.arange(r, dtype=np.int64))
    indices = np.einsum(
        "trn,r->tn", syndromes.astype(np.int64, copy=False), powers,
        optimize=False,
    )
    values = np.asarray(log_mass, dtype=np.float64)
    return np.asarray(
        [float(values[row].sum()) for row in indices], dtype=np.float64,
    )


def character_values(labels, masks):
    labels = np.asarray(labels, dtype=np.uint64)
    masks = np.asarray(masks, dtype=np.uint64)
    output = np.empty((labels.size, masks.size), dtype=np.int8)
    for start in range(0, masks.size, 128):
        stop = min(start + 128, masks.size)
        parity = np.bitwise_count(
            labels[:, None] & masks[None, start:stop]
        ) & np.uint8(1)
        output[:, start:stop] = 1 - 2 * parity.astype(np.int8)
    return output


def character_means(chains, masks):
    return np.stack([
        character_values(labels, masks).mean(axis=0) for labels in chains
    ])


def u_squares(means):
    means = np.asarray(means, dtype=np.float64)
    count = means.shape[0]
    return (
        np.square(means.sum(axis=0)) - np.square(means).sum(axis=0)
    ) / (count * (count - 1))


def population_mean(values, basis_positions, k):
    values = np.asarray(values, dtype=np.float64)
    basis = np.zeros(values.size, dtype=bool)
    basis[np.asarray(basis_positions, dtype=np.int64)] = True
    sampled = values[~basis]
    total = (1 << int(k)) - 1
    remaining = total - int(k)
    estimate = (float(values[basis].sum()) + remaining * float(sampled.mean())) / total
    fraction = sampled.size / remaining
    finite_se = (
        remaining / total
        * math.sqrt((1.0 - fraction) * float(sampled.var(ddof=1)) / sampled.size)
    )
    return float(estimate), float(finite_se)


def qtop_estimate(means, basis_positions, k):
    estimate, character_se = population_mean(
        u_squares(means), basis_positions, k,
    )
    delete = np.asarray([
        population_mean(
            u_squares(np.delete(means, omitted, axis=0)), basis_positions, k,
        )[0]
        for omitted in range(means.shape[0])
    ])
    trajectory_se = math.sqrt(
        (delete.size - 1) / delete.size
        * float(np.square(delete - delete.mean()).sum())
    )
    return {
        "q_top": estimate,
        "q_top_character_se": character_se,
        "q_top_total_se": math.hypot(trajectory_se, character_se),
        "q_top_trajectory_se": trajectory_se,
    }


def character_d2(means_a, means_b, basis_positions, k):
    def estimate(left, right):
        per_character = (
            u_squares(left) + u_squares(right)
            - 2.0 * left.mean(axis=0) * right.mean(axis=0)
        )
        return population_mean(per_character, basis_positions, k)

    value, character_se = estimate(means_a, means_b)
    variance = 0.0
    for side in (0, 1):
        source = means_a if side == 0 else means_b
        delete = []
        for omitted in range(source.shape[0]):
            left = np.delete(means_a, omitted, axis=0) if side == 0 else means_a
            right = np.delete(means_b, omitted, axis=0) if side == 1 else means_b
            delete.append(estimate(left, right)[0])
        delete = np.asarray(delete)
        variance += (
            (delete.size - 1) / delete.size
            * float(np.square(delete - delete.mean()).sum())
        )
    return value, math.sqrt(variance + character_se**2)


def b_character_values(states, masks):
    states = np.asarray(states, dtype=np.uint8)
    masks = np.asarray(masks, dtype=np.uint8)
    parity = np.zeros((states.shape[0], masks.shape[0]), dtype=np.uint8)
    byte_parity = (
        np.bitwise_count(np.arange(256, dtype=np.uint8)) & np.uint8(1)
    ).astype(np.uint8)
    for byte in range(states.shape[1]):
        parity ^= byte_parity[states[:, byte, None] & masks[None, :, byte]]
    return 1.0 - 2.0 * parity.astype(np.float64)


def split_rhat(chains):
    chains = np.asarray(chains, dtype=np.float64)
    half = chains.shape[1] // 2
    if half < 2:
        return math.inf
    split = np.concatenate((chains[:, :half], chains[:, -half:]), axis=0)
    within = float(np.mean(np.var(split, axis=1, ddof=1)))
    if within == 0.0:
        return 1.0 if np.all(split == split[0, 0]) else math.inf
    between = half * float(np.var(np.mean(split, axis=1), ddof=1))
    return math.sqrt(((half - 1) / half * within + between / half) / within)


def bulk_ess(chains):
    chains = np.asarray(chains, dtype=np.float64)
    total = chains.size
    centered = chains - chains.mean(axis=1, keepdims=True)
    variance = float(np.mean(centered * centered))
    if variance == 0.0:
        return float(total)
    rho_sum = 0.0
    previous_pair = math.inf
    for lag in range(1, min(chains.shape[1] - 1, 1000), 2):
        pair = 0.0
        for offset in (0, 1):
            pair += float(np.mean(
                centered[:, :-(lag + offset)]
                * centered[:, lag + offset:]
            )) / variance
        pair = min(pair, previous_pair)
        if pair <= 0.0:
            break
        rho_sum += pair
        previous_pair = pair
    return float(min(total, total / max(1.0 + 2.0 * rho_sum, 1.0)))


def observable_diagnostic(chains):
    chains = np.asarray(chains, dtype=np.float64)
    require(
        chains.ndim == 2 and chains.shape[0] == 8,
        "diagnostic chain panel changed",
    )
    if np.unique(chains).size == 1:
        return {
            "bulk_ess": float(chains.size),
            "degenerate": True,
            "split_rhat": 1.0,
        }
    return {
        "bulk_ess": bulk_ess(chains),
        "degenerate": False,
        "split_rhat": split_rhat(chains),
    }


def b_d2(means_a, means_b):
    def estimate(left, right):
        return float(np.mean(
            u_squares(left) + u_squares(right)
            - 2.0 * left.mean(axis=0) * right.mean(axis=0)
        ))

    value = estimate(means_a, means_b)
    variance = 0.0
    for side in (0, 1):
        source = means_a if side == 0 else means_b
        delete = []
        for omitted in range(source.shape[0]):
            left = np.delete(means_a, omitted, axis=0) if side == 0 else means_a
            right = np.delete(means_b, omitted, axis=0) if side == 1 else means_b
            delete.append(estimate(left, right))
        delete = np.asarray(delete)
        variance += (
            (delete.size - 1) / delete.size
            * float(np.square(delete - delete.mean()).sum())
        )
    return value, math.sqrt(variance)


def mean_and_se(chains, normalizer):
    trajectory = np.asarray(chains, dtype=np.float64).mean(axis=1) / normalizer
    return (
        float(trajectory.mean()),
        float(trajectory.std(ddof=1) / math.sqrt(trajectory.size)),
    )


def collision(chains):
    frequencies = []
    for labels in chains:
        values, counts = np.unique(labels, return_counts=True)
        frequencies.append(dict(zip(
            (int(value) for value in values),
            counts.astype(np.float64) / labels.size,
        )))
    overlaps = []
    for left in range(len(frequencies)):
        for right in range(left + 1, len(frequencies)):
            a, b = frequencies[left], frequencies[right]
            if len(a) > len(b):
                a, b = b, a
            overlaps.append(sum(value * b.get(label, 0.0) for label, value in a.items()))
    return float(np.mean(overlaps))


def assert_close(actual, expected, label, tolerance=2e-12):
    actual = float(actual)
    expected = float(expected)
    if math.isinf(actual) and actual == expected:
        return
    require(
        math.isfinite(actual)
        and math.isfinite(expected)
        and abs(actual - expected) <= tolerance,
        f"estimate mismatch: {label}: {actual} != {expected}",
    )


def load_control(run_root, config):
    manifest = load_canonical(run_root / "control/control_manifest.json")
    verify_self_hash(manifest, "manifest_sha256")
    control_path = run_root / "control/control.npz"
    require(
        manifest["config_sha256"] == CONFIG_SHA256
        and manifest["control_content_sha256"] == CONTROL_CONTENT_SHA256
        and manifest["control_file_sha256"] == sha256_file(control_path),
        "control manifest identity mismatch",
    )
    with np.load(control_path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        arrays = {
            name: archive[name].copy()
            for name in archive.files if name != "metadata_json"
        }
    require(
        metadata["config_sha256"] == CONFIG_SHA256
        and metadata["control_content_sha256"] == CONTROL_CONTENT_SHA256
        and metadata["control_version"]
        == "exp102.q0_random_full_column_direct_block.t1_m8.control.v2"
        and content_sha(metadata["control_version"], metadata, arrays)
        == CONTROL_CONTENT_SHA256,
        "control content identity mismatch",
    )
    H = np.asarray(arrays["H"], dtype=np.uint8)
    model, frame = build_model(H)
    syndrome = unpack(arrays["syndrome_packed"], model.num_checks)
    fixed = unpack(arrays["fixed_states_packed"], model.num_qubits)
    verify_hard_coset(fixed, H, syndrome)
    require(int(syndrome.sum()) == 160, "hard-cell syndrome changed")
    require(config["cell"] == metadata["cell"], "control cell changed")
    return metadata, arrays, H, model, frame, syndrome, fixed


def verify_schedule(run_root, config):
    schedule = load_canonical(run_root / "control/schedule.json")
    verify_self_hash(schedule, "schedule_sha256")
    require(
        schedule["schedule_sha256"] == SCHEDULE_SHA256
        and schedule["contract_version"] == CONTRACT_VERSION
        and schedule["config_sha256"] == CONFIG_SHA256
        and schedule["control_content_sha256"] == CONTROL_CONTENT_SHA256
        and schedule["source_identity"] == {
            "archive_sha256": ARCHIVE_SHA256,
            "source_commit": SOURCE_COMMIT,
            "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        },
        "schedule identity mismatch",
    )
    tasks = schedule["tasks"]
    require(len(tasks) == 40, "schedule task count changed")
    for task in tasks:
        core = {
            key: value for key, value in task.items()
            if key not in {"owner", "task_fingerprint"}
        }
        require(
            sha256_json(core) == task["task_fingerprint"]
            and task["method_id"] == METHOD_ID
            and task["raw_version"] == RAW_VERSION,
            "task fingerprint or method changed",
        )
    for field in (
        "task_fingerprint", "initialization_seed", "burn_update_seed",
        "measurement_update_seed", "observation_seed",
    ):
        require(len({task[field] for task in tasks}) == 40, f"duplicate {field}")
    ownership = {
        node: sum(task["owner"] == node for task in tasks)
        for node in config["resource"]["allowed_nodes"]
    }
    require(ownership == {"nd-1": 14, "nd-2": 13, "nd-3": 13},
            "ownership changed")
    return schedule


def verify_preflights(run_root):
    preflight = load_canonical(run_root / "preflight/aggregate.json")
    verify_self_hash(preflight, "preflight_sha256")
    require(
        preflight["preflight_sha256"] == PREFLIGHT_SHA256
        and preflight["status"] == "PASS"
        and preflight["exact_consensus"] is True
        and preflight["schedule_sha256"] == SCHEDULE_SHA256,
        "v2 preflight identity changed",
    )
    portable = load_canonical(
        run_root / "portable_preflight/preflight/aggregate.json"
    )
    verify_self_hash(portable, "aggregate_sha256")
    require(
        portable["status"] == "PASS"
        and portable["exact_consensus"] is True
        and portable["source_commit"] == SOURCE_COMMIT,
        "portable preflight did not pass",
    )
    return preflight, portable


def expected_initial(task, fixed, model, syndrome):
    family = task["family"]
    if family == "P":
        return fixed[0]
    if family == "M0":
        return fixed[1]
    if family == "M1":
        return fixed[2]
    if family == "S":
        return fixed[3 + int(task["index"])]
    require(family == "U", "unknown initialization family")
    return uniform_hard_coset_state(
        model, syndrome, task["initialization_seed"],
    )


def audit_raw(path, task, source, config, arrays, H, model, frame, syndrome,
              fixed, log_mass):
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name].copy() for name in archive.files}
    required = {
        "archive_sha256", "config_sha256", "contract_version",
        "control_content_sha256", "model_fingerprint", "raw_version",
        "replay_seconds", "sampling_seconds", "schedule_sha256", "source_commit",
        "source_manifest_sha256", "syndrome_packed", "task_fingerprint",
        "task_json", "version", "burn__counters", "burn__final_b_columns",
        "burn__selected_columns", "burn__old_columns", "burn__new_columns",
        "conditional_engine", "final_b_columns", "final_state_packed",
        "initial_b_columns", "initial_state_packed", "measurement__counters",
        "measurement__selected_columns", "measurement__old_columns",
        "measurement__new_columns", "measurement__b_columns",
        "measurement__b_likelihood", "measurement__b_weights",
        "measurement__blocks", "measurement__labels",
        "measurement__states_packed", "measurement__weights",
        "seed_identity_sha256",
    }
    require(set(data) == required, "raw schema changed")
    expected_scalars = {
        "archive_sha256": source["archive_sha256"],
        "conditional_engine": DIRECT_ENGINE,
        "config_sha256": CONFIG_SHA256,
        "contract_version": CONTRACT_VERSION,
        "control_content_sha256": CONTROL_CONTENT_SHA256,
        "model_fingerprint": model.fingerprint(),
        "raw_version": RAW_VERSION,
        "schedule_sha256": SCHEDULE_SHA256,
        "source_commit": source["source_commit"],
        "source_manifest_sha256": source["source_manifest_sha256"],
        "task_fingerprint": task["task_fingerprint"],
        "task_json": canonical_json(task),
        "version": DIRECT_VERSION,
    }
    for name, expected in expected_scalars.items():
        require(str(data[name].item()) == expected, f"raw identity mismatch: {name}")
    require(
        np.array_equal(data["syndrome_packed"], arrays["syndrome_packed"])
        and math.isfinite(float(data["sampling_seconds"]))
        and float(data["sampling_seconds"]) > 0.0
        and math.isfinite(float(data["replay_seconds"]))
        and float(data["replay_seconds"]) > 0.0,
        "raw syndrome or timing invalid",
    )
    seed_identity = hashlib.sha256(
        DIRECT_VERSION.encode("ascii") + b"\0"
        + np.asarray([
            task["burn_update_seed"], task["measurement_update_seed"],
            task["observation_seed"],
        ], dtype=">u8").tobytes()
        + np.asarray(0.04, dtype=">f8").tobytes()
        + np.asarray([
            config["resource"]["burn_updates"],
            config["resource"]["measurement_updates"],
        ], dtype=">u8").tobytes()
    ).hexdigest()
    require(str(data["seed_identity_sha256"].item()) == seed_identity,
            "seed identity mismatch")

    initial = unpack(data["initial_state_packed"], model.num_qubits)
    require(np.array_equal(initial, expected_initial(
        task, fixed, model, syndrome,
    )), "initial state mismatch")
    verify_hard_coset(initial[None, :], H, syndrome)
    require(np.array_equal(state_b_columns(initial, H), data["initial_b_columns"]),
            "initial B mismatch")
    burn, burn_changes, burn_bits = replay_b(
        data["initial_b_columns"], data["burn__selected_columns"],
        data["burn__old_columns"], data["burn__new_columns"],
    )
    measurement, measurement_changes, measurement_bits = replay_b(
        data["burn__final_b_columns"], data["measurement__selected_columns"],
        data["measurement__old_columns"], data["measurement__new_columns"],
    )
    require(
        np.array_equal(burn[-1], data["burn__final_b_columns"])
        and np.array_equal(measurement, data["measurement__b_columns"])
        and np.array_equal(measurement[-1], data["final_b_columns"]),
        "B endpoints changed",
    )
    require(
        np.array_equal(
            data["burn__counters"][:4],
            [burn.shape[0], burn_changes, burn_bits, 0],
        )
        and int(data["burn__counters"][4]) == 0,
        "burn counters changed",
    )
    states = unpack(data["measurement__states_packed"], model.num_qubits)
    verify_hard_coset(states, H, syndrome)
    r, n = H.shape
    b_packed = columns_to_b_packed(measurement, r)
    state_b = np.packbits(
        states[:, n * n:].reshape(-1, r * r), axis=1, bitorder="little",
    )
    require(np.array_equal(b_packed, state_b), "state/B trace mismatch")
    signatures = qubit_signatures(frame)
    labels = labels_from_states(states, signatures)
    weights = states.sum(axis=1).astype(np.float64)
    require(
        np.array_equal(labels, data["measurement__labels"])
        and np.array_equal(weights, data["measurement__weights"]),
        "label or weight mismatch",
    )
    initial_label = int(labels_from_states(initial[None, :], signatures)[0])
    label_changes = int(labels[0] != initial_label)
    label_changes += int(np.count_nonzero(labels[1:] != labels[:-1]))
    require(
        np.array_equal(
            data["measurement__counters"][:4],
            [measurement.shape[0], measurement_changes, measurement_bits,
             measurement.shape[0]],
        )
        and int(data["measurement__counters"][4]) == label_changes,
        "measurement counters changed",
    )
    b_weights = np.bitwise_count(b_packed).sum(axis=1).astype(np.float64)
    likelihood = b_log_likelihood(b_packed, H, syndrome, log_mass)
    require(
        np.array_equal(b_weights, data["measurement__b_weights"])
        and np.array_equal(likelihood, data["measurement__b_likelihood"]),
        "B weight or likelihood mismatch",
    )
    blocks = np.minimum(
        7, 8 * np.arange(labels.size) // labels.size,
    ).astype(np.int8)
    require(np.array_equal(blocks, data["measurement__blocks"]),
            "measurement blocks changed")
    require(np.array_equal(
        unpack(data["final_state_packed"], model.num_qubits), states[-1],
    ), "final state changed")
    return {
        "b_likelihood": likelihood,
        "b_packed": b_packed,
        "b_weights": b_weights,
        "burn_b_packed": columns_to_b_packed(burn, r),
        "burn_changes": burn_changes,
        "family": task["family"],
        "index": int(task["index"]),
        "initial_b_packed": columns_to_b_packed(
            data["initial_b_columns"][None, :], r,
        )[0],
        "label_changes": label_changes,
        "labels": labels,
        "measurement_changes": measurement_changes,
        "weights": weights,
    }


def family_core(records, arrays, model, H, config):
    records = sorted(records, key=lambda row: row["index"])
    require([row["index"] for row in records] == list(range(8)),
            "family trajectory panel changed")
    gates = config["gates"]
    masks = arrays["logical_character_masks"]
    basis = arrays["logical_basis_positions"]
    logical_means = character_means([row["labels"] for row in records], masks)
    qtop = qtop_estimate(logical_means, basis, model.k)
    weights = np.stack([row["weights"] for row in records])
    b_weights = np.stack([row["b_weights"] for row in records])
    likelihood = np.stack([row["b_likelihood"] for row in records])
    b_chains = np.stack([
        b_character_values(row["b_packed"], arrays["b_character_masks_packed"])
        for row in records
    ]).astype(np.int8)
    b_means = b_chains.mean(axis=1)
    weight_mean, weight_se = mean_and_se(weights, model.num_qubits)
    b_weight_mean, b_weight_se = mean_and_se(b_weights, H.shape[0] ** 2)
    likelihood_mean, likelihood_se = mean_and_se(likelihood, H.shape[1])
    diagnostic_masks = np.concatenate((
        masks[basis], np.delete(masks, basis)[:64],
    ))
    logical_diagnostics = []
    for mask in diagnostic_masks:
        chains = np.stack([
            character_values(row["labels"], mask[None])[:, 0]
            for row in records
        ])
        logical_diagnostics.append(observable_diagnostic(chains))
    b_diagnostics = [
        observable_diagnostic(b_chains[:, :, index])
        for index in range(b_chains.shape[2])
    ]
    scalar_diagnostics = [
        observable_diagnostic(likelihood),
        observable_diagnostic(b_weights),
        observable_diagnostic(weights),
    ]
    all_diagnostics = [
        *logical_diagnostics, *b_diagnostics, *scalar_diagnostics,
    ]
    nondegenerate_ess = [
        row["bulk_ess"] for row in all_diagnostics if not row["degenerate"]
    ]
    max_rhat = max(row["split_rhat"] for row in all_diagnostics)
    min_ess = min(nondegenerate_ess) if nondegenerate_ess else math.inf
    dense_start = H.shape[0] ** 2 + 2 * H.shape[0]
    require(
        b_chains.shape[2] - dense_start
        == config["statistics"]["b_dense_character_count"],
        "B-character panel layout changed",
    )
    dense_nondegenerate = sum(
        not row["degenerate"] for row in b_diagnostics[dense_start:]
    )
    failures = []
    if not math.isfinite(qtop["q_top_total_se"]) or (
        qtop["q_top_total_se"] > gates["max_q_top_se"]
    ):
        failures.append("q_top_se")
    if max_rhat > gates["max_rhat"]:
        failures.append("rhat")
    if min_ess < gates["min_bulk_ess"]:
        failures.append("bulk_ess")
    if dense_nondegenerate < gates["min_dense_b_characters_nondegenerate"]:
        failures.append("b_dense_characters_uninformative")
    if any(
        row["burn_changes"] < gates["min_burn_column_changes_per_trajectory"]
        for row in records
    ):
        failures.append("burn_column_changes")
    if any(
        row["measurement_changes"]
        < gates["min_measurement_column_changes_per_trajectory"]
        for row in records
    ):
        failures.append("measurement_column_changes")
    if any(
        row["label_changes"]
        < gates["min_measurement_label_changes_per_trajectory"]
        for row in records
    ):
        failures.append("measurement_label_changes")
    return {
        **qtop,
        "b_chains": b_chains,
        "b_dense_nondegenerate": dense_nondegenerate,
        "b_likelihood_mean_per_factor": likelihood_mean,
        "b_likelihood_mean_per_factor_se": likelihood_se,
        "b_means": b_means,
        "b_weight_chains": b_weights,
        "b_weight_mean_normalized": b_weight_mean,
        "b_weight_mean_normalized_se": b_weight_se,
        "collision_q_top_diagnostic": collision([row["labels"] for row in records]),
        "failures": failures,
        "likelihood_chains": likelihood,
        "logical_means": logical_means,
        "max_rhat": float(max_rhat),
        "min_nondegenerate_bulk_ess": float(min_ess),
        "normalized_weight_mean": weight_mean,
        "normalized_weight_mean_se": weight_se,
        "records": records,
        "valid": not failures,
        "weight_chains": weights,
    }


def verify_family_report(name, actual, claimed):
    for field in (
        "b_likelihood_mean_per_factor", "b_likelihood_mean_per_factor_se",
        "b_weight_mean_normalized", "b_weight_mean_normalized_se",
        "collision_q_top_diagnostic", "normalized_weight_mean",
        "normalized_weight_mean_se", "q_top", "q_top_character_se",
        "q_top_total_se", "q_top_trajectory_se",
        "max_rhat", "min_nondegenerate_bulk_ess",
    ):
        assert_close(actual[field], claimed[field], f"{name}.{field}")
    for field in ("b_dense_nondegenerate", "failures", "valid"):
        require(actual[field] == claimed[field], f"{name}.{field} changed")
    transitions = [{
        "burn_column_changes": row["burn_changes"],
        "index": row["index"],
        "measurement_column_changes": row["measurement_changes"],
        "measurement_label_changes": row["label_changes"],
    } for row in actual["records"]]
    require(transitions == claimed["transition_counts"],
            f"{name} transition counts changed")


def pooled_b_rhat(left, right):
    maximum = max(
        split_rhat(np.concatenate((
            left["b_weight_chains"], right["b_weight_chains"],
        ), axis=0)),
        split_rhat(np.concatenate((
            left["likelihood_chains"], right["likelihood_chains"],
        ), axis=0)),
    )
    for index in range(left["b_chains"].shape[2]):
        values = np.concatenate((
            left["b_chains"][:, :, index],
            right["b_chains"][:, :, index],
        ), axis=0)
        if np.unique(values).size > 1:
            maximum = max(maximum, split_rhat(values))
    return float(maximum)


def verify_pair_report(left_name, right_name, left, right, claimed, arrays,
                       model, H, config):
    gates = config["gates"]
    logical_value, logical_se = character_d2(
        left["logical_means"], right["logical_means"],
        arrays["logical_basis_positions"], model.k,
    )
    b_value, b_se = b_d2(left["b_means"], right["b_means"])
    pooled = pooled_b_rhat(left, right)
    b_mean_delta = np.abs(
        left["b_means"].mean(axis=0) - right["b_means"].mean(axis=0)
    )
    b_mean_se = np.sqrt(
        left["b_means"].var(axis=0, ddof=1) / left["b_means"].shape[0]
        + right["b_means"].var(axis=0, ddof=1) / right["b_means"].shape[0]
    )
    b_character_pass_mask = (
        (b_mean_delta <= gates["max_abs_b_character_mean_delta"])
        & (b_mean_delta <= gates["delta_sigma_multiplier"] * b_mean_se
           + gates["b_character_delta_sigma_slack"])
    )
    values = {
        "b_character_d2": b_value,
        "b_character_d2_se": b_se,
        "b_likelihood_delta_per_factor": abs(
            left["b_likelihood_mean_per_factor"]
            - right["b_likelihood_mean_per_factor"]
        ),
        "b_likelihood_delta_per_factor_se": math.hypot(
            left["b_likelihood_mean_per_factor_se"],
            right["b_likelihood_mean_per_factor_se"],
        ),
        "b_weight_delta_normalized": abs(
            left["b_weight_mean_normalized"]
            - right["b_weight_mean_normalized"]
        ),
        "b_weight_delta_normalized_se": math.hypot(
            left["b_weight_mean_normalized_se"],
            right["b_weight_mean_normalized_se"],
        ),
        "logical_d2": logical_value,
        "logical_d2_total_se": logical_se,
        "pooled_b_rhat": pooled,
        "q_top_delta": abs(left["q_top"] - right["q_top"]),
        "q_top_delta_se": math.hypot(
            left["q_top_total_se"], right["q_top_total_se"],
        ),
        "weight_delta_normalized": abs(
            left["normalized_weight_mean"] - right["normalized_weight_mean"]
        ),
        "weight_delta_normalized_se": math.hypot(
            left["normalized_weight_mean_se"],
            right["normalized_weight_mean_se"],
        ),
    }
    checks = {
        "b_character_d2": max(0.0, b_value) + 3.0 * b_se
        <= gates["max_b_character_d2_upper"],
        "b_character_means": bool(np.all(b_character_pass_mask)),
        "b_likelihood": (
            values["b_likelihood_delta_per_factor"]
            <= gates["max_b_log_likelihood_delta_per_factor"]
            and values["b_likelihood_delta_per_factor"]
            <= 3.0 * values["b_likelihood_delta_per_factor_se"]
            + 1.0 / H.shape[1]
        ),
        "b_weight": (
            values["b_weight_delta_normalized"]
            <= gates["max_b_normalized_weight_delta"]
            and values["b_weight_delta_normalized"]
            <= 3.0 * values["b_weight_delta_normalized_se"]
            + 1.0 / (H.shape[0] ** 2)
        ),
        "logical_d2": max(0.0, logical_value) + 3.0 * logical_se
        <= gates["max_d2_upper"],
        "pooled_b_rhat": pooled <= gates["max_rhat"],
        "q_top": (
            values["q_top_delta"] <= gates["max_abs_delta_q_top"]
            and values["q_top_delta"]
            <= gates["delta_sigma_multiplier"] * values["q_top_delta_se"]
            + gates["delta_sigma_slack"]
        ),
        "weight": (
            values["weight_delta_normalized"]
            <= gates["max_normalized_weight_delta"]
            and values["weight_delta_normalized"]
            <= 3.0 * values["weight_delta_normalized_se"]
            + gates["normalized_weight_sigma_slack_qubits"] / model.num_qubits
        ),
    }
    require(
        claimed["left"] == left_name and claimed["right"] == right_name,
        "pair ordering changed",
    )
    for field, value in values.items():
        assert_close(value, claimed[field], f"{left_name}/{right_name}.{field}")
    require(
        claimed["b_character_failed_count"]
        == int(np.count_nonzero(~b_character_pass_mask))
        and claimed["checks"] == checks
        and claimed["valid"] == all(checks.values()),
        f"{left_name}/{right_name} gate result changed",
    )


def map_bridge(records_by_family, arrays, gates):
    fixed_b = arrays["fixed_b_blocks"]
    result = {}
    for family, source, target in (
        ("M0", fixed_b[1], fixed_b[2]),
        ("M1", fixed_b[2], fixed_b[1]),
    ):
        counts = []
        for row in records_by_family[family]:
            combined = np.concatenate((row["burn_b_packed"], row["b_packed"]), axis=0)
            bits = unpack(combined, source.size).reshape(-1, *source.shape)
            source_distance = np.count_nonzero(
                bits ^ source[None, :, :], axis=(1, 2),
            )
            target_distance = np.count_nonzero(
                bits ^ target[None, :, :], axis=(1, 2),
            )
            counts.append(int(np.count_nonzero(target_distance < source_distance)))
        result[family] = {
            "aggregate_opposite_basin_visits": int(sum(counts)),
            "opposite_basin_visits_per_trajectory": counts,
            "trajectories_visiting_opposite_basin": int(sum(value > 0 for value in counts)),
            "valid": (
                sum(counts) >= gates["min_aggregate_opposite_map_basin_visits"]
                and sum(value > 0 for value in counts)
                >= gates["min_trajectories_visiting_opposite_map_basin"]
            ),
        }
    return result


def constant_b_failures(records, masks):
    measurement = np.concatenate([row["b_packed"] for row in records], axis=0)
    values = b_character_values(measurement, masks).astype(np.int8)
    failures = []
    for character in range(masks.shape[0]):
        unique = np.unique(values[:, character])
        if unique.size != 1:
            continue
        common = int(unique[0])
        mask = masks[character:character + 1]
        for row in records:
            initial = int(b_character_values(
                row["initial_b_packed"][None, :], mask,
            )[0, 0])
            if initial == common:
                continue
            burn = b_character_values(row["burn_b_packed"], mask)[:, 0]
            if not np.any(burn == common):
                failures.append({
                    "character": character,
                    "family": row["family"],
                    "index": row["index"],
                })
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_root = Path(args.run_root).resolve()
    output = Path(args.output).resolve()
    require(not output.exists(), "audit output already exists")

    require(sha256_file(CONFIG_PATH) == CONFIG_SHA256, "local v2 config changed")
    config = load_canonical(CONFIG_PATH)
    metadata, arrays, H, model, frame, syndrome, fixed = load_control(
        run_root, config,
    )
    schedule = verify_schedule(run_root, config)
    preflight, portable = verify_preflights(run_root)
    report = load_canonical(args.report)
    verify_self_hash(report, "report_sha256")
    require(
        report["report_version"] == REPORT_VERSION
        and report["contract_version"] == CONTRACT_VERSION
        and report["schedule_sha256"] == SCHEDULE_SHA256
        and report["preflight_sha256"] == PREFLIGHT_SHA256
        and report["config_sha256"] == CONFIG_SHA256
        and report["control_content_sha256"] == CONTROL_CONTENT_SHA256,
        "primary report identity changed",
    )
    require(
        report["source_identity"] == schedule["source_identity"]
        and report["scope"] == config["scope"],
        "primary report source or scope changed",
    )

    task_by_fingerprint = {
        task["task_fingerprint"]: task for task in schedule["tasks"]
    }
    raw_paths = {}
    node_reports = {}
    for node in config["resource"]["allowed_nodes"]:
        node_report = load_canonical(
            run_root / f"measurement/{node}/node_report.json"
        )
        verify_self_hash(node_report, "node_report_sha256")
        require(
            node_report["node_report_version"] == NODE_REPORT_VERSION
            and node_report["status"] == "COMPLETE"
            and node_report["node"] == node
            and node_report["schedule_sha256"] == SCHEDULE_SHA256
            and node_report["preflight_sha256"] == PREFLIGHT_SHA256
            and node_report["source_identity"] == schedule["source_identity"]
            and node_report["raw_count"]
            == len(schedule["ownership"][node])
            and len(node_report["raw_records"])
            == len(schedule["ownership"][node]),
            "node report changed",
        )
        node_reports[node] = node_report["node_report_sha256"]
        for record in node_report["raw_records"]:
            path = run_root / f"measurement/{node}/raw/{record['file']}"
            require(sha256_file(path) == record["raw_sha256"], "raw hash mismatch")
            fingerprint = record["task_fingerprint"]
            require(
                fingerprint in task_by_fingerprint
                and task_by_fingerprint[fingerprint]["owner"] == node
                and record["family"]
                == task_by_fingerprint[fingerprint]["family"]
                and record["index"]
                == task_by_fingerprint[fingerprint]["index"]
                and math.isfinite(float(record["sampling_seconds"]))
                and float(record["sampling_seconds"]) > 0.0
                and math.isfinite(float(record["replay_seconds"]))
                and float(record["replay_seconds"]) > 0.0,
                "node raw record identity changed",
            )
            require(fingerprint not in raw_paths, "duplicate raw fingerprint")
            raw_paths[fingerprint] = path
    require(set(raw_paths) == set(task_by_fingerprint), "raw task set incomplete")

    mass = classical_coset_mass(H, P_VALUE)
    log_mass = np.log(mass)
    source = schedule["source_identity"]
    records = [
        audit_raw(
            raw_paths[task["task_fingerprint"]], task, source, config, arrays,
            H, model, frame, syndrome, fixed, log_mass,
        )
        for task in schedule["tasks"]
    ]
    records_by_family = {
        family: sorted(
            [row for row in records if row["family"] == family],
            key=lambda row: row["index"],
        )
        for family in FAMILIES
    }
    family_values = {
        family: family_core(
            records_by_family[family], arrays, model, H, config,
        )
        for family in FAMILIES
    }
    require(set(report["families"]) == set(FAMILIES),
            "primary family panel changed")
    for family in FAMILIES:
        verify_family_report(
            family, family_values[family], report["families"][family],
        )
    comparison_by_pair = {
        (row["left"], row["right"]): row for row in report["comparisons"]
    }
    expected_pairs = {
        (left, right)
        for left_index, left in enumerate(FAMILIES)
        for right in FAMILIES[left_index + 1:]
    }
    require(
        len(report["comparisons"]) == len(expected_pairs)
        and set(comparison_by_pair) == expected_pairs,
        "primary pair panel changed",
    )
    for left_index, left in enumerate(FAMILIES):
        for right in FAMILIES[left_index + 1:]:
            verify_pair_report(
                left, right, family_values[left], family_values[right],
                comparison_by_pair[(left, right)], arrays, model, H, config,
            )
    bridge = map_bridge(records_by_family, arrays, config["gates"])
    require(bridge == report["map_bridge"], "MAP bridge gate changed")
    freeze_failures = constant_b_failures(
        records, arrays["b_character_masks_packed"],
    )
    require(
        freeze_failures == report["constant_b_freeze_failures"],
        "constant-B freeze result changed",
    )
    expected_checks = {
        "all_families": all(
            family_values[family]["valid"] for family in FAMILIES
        ),
        "all_pairwise_comparisons": all(
            comparison_by_pair[pair]["valid"] for pair in expected_pairs
        ),
        "constant_b_freeze": not freeze_failures,
        "map_bridge": all(row["valid"] for row in bridge.values()),
        "raw_identity_and_algebra": True,
    }
    expected_status = (
        "DIAGNOSTIC_DIRECT_BLOCK_T1_M8_VIABLE"
        if all(expected_checks.values())
        else "UNRESOLVED_DIRECT_BLOCK_T1_M8"
    )
    require(
        report["checks"] == expected_checks
        and report["status"] == expected_status,
        "primary terminal gate or status changed",
    )
    ordered_paths = sorted(raw_paths.values(), key=lambda path: path.as_posix())
    raw_set_sha = hashlib.sha256("".join(
        f"{path.relative_to(run_root).as_posix()}:{sha256_file(path)}\n"
        for path in ordered_paths
    ).encode("ascii")).hexdigest()
    require(
        report["raw_count"] == 40
        and report["raw_set_sha256"] == raw_set_sha
        and report["node_report_sha256"] == node_reports,
        "primary report raw evidence changed",
    )

    core = {
        "archive_sha256": ARCHIVE_SHA256,
        "audit_version": AUDIT_VERSION,
        "checks": {
            "control_and_schedule": True,
            "family_and_pair_gates": True,
            "hard_coset_and_transcripts": True,
            "identity_and_seed_hashes": True,
            "map_bridge_and_freeze": True,
            "primary_estimates_recomputed": True,
            "primary_report_self_hash": True,
            "three_node_preflights": True,
        },
        "config_sha256": CONFIG_SHA256,
        "control_content_sha256": metadata["control_content_sha256"],
        "portable_preflight_sha256": portable["aggregate_sha256"],
        "preflight_sha256": preflight["preflight_sha256"],
        "primary_report_sha256": report["report_sha256"],
        "primary_status": report["status"],
        "raw_count": len(records),
        "raw_set_sha256": raw_set_sha,
        "schedule_sha256": schedule["schedule_sha256"],
        "source_commit": SOURCE_COMMIT,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "status": "INDEPENDENT_RAW_ONLY_AUDIT_PASS",
    }
    audit = {**core, "audit_sha256": sha256_json(core)}
    atomic_json(output, audit)
    print(canonical_json(audit))


if __name__ == "__main__":
    main()
